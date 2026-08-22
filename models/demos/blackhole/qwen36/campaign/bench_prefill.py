# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill ms/token and TTFT decomposition on the chunked traced path (single user).

One model is built at the largest requested ISL; the 2048-token chunk-outer
prefill trace is captured once (compile cost reported separately), then each ISL
is prefilled repeatedly (prefill_traced_chunked zeroes GDN state per call, and KV
pages are rewritten causally each run, so cache reuse across ISLs/reps is safe).
TTFT decomposes into: capture_s (one-time compile), exec (device prefill), and
readback (logits D2H + first-token argmax). Emits one BENCH_JSON line per ISL.

Run (P150x8):
    MESH_DEVICE=P150x8 HF_MODEL=/path/to/qwen38-27b-weights \\
        pytest models/demos/blackhole/qwen36/campaign/bench_prefill.py -v -s

Knobs (env):
    QWEN38_PREFILL_ISLS      comma list, each >= 2048 (default "2048,10240,65536")
    QWEN38_PREFILL_REPEATS   measured repeats per ISL; 0 = auto (3 for ISL <= 16384,
                             else 1). With >1 repeats the first run is warmup.
    QWEN38_BENCH_REAL_PROMPT 1 = corpus prompts instead of tiled local text
"""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.campaign.bench_common import bench_prompt, emit_bench_json, stats_ms
from models.demos.blackhole.qwen36.demo.text_demo import (
    BLOCK_SIZE,
    DEVICE_PARAMS,
    _MESH_SHAPE,
    _blocks_for,
    _should_use_chunked_trace,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

CHUNK = 2048
_ISLS = sorted(int(x) for x in os.environ.get("QWEN38_PREFILL_ISLS", "2048,10240,65536").split(","))
_REPEATS = int(os.environ.get("QWEN38_PREFILL_REPEATS", "0"))


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_bench_prefill(mesh_device):
    from transformers import AutoTokenizer

    # Sub-chunk ISLs (e.g. 1024) are padded up to one 2048 bucket with actual_len
    # masking below — the padded-chunk cost IS the current serving path's TTFT for
    # short prompts, so it is a legitimate ladder point (grouped short-prefill is a
    # separate optimization, benched by its own hook when it lands).
    assert all(isl >= 1 for isl in _ISLS), f"ISLs must be positive, got {_ISLS}"
    device = mesh_device
    device.enable_program_cache()

    max_isl = max(_ISLS)
    num_blocks = _blocks_for(max_isl, 64)
    # Multiple of 32 blocks for chunked SDPA page-table alignment (as _run_tp_generation does).
    num_blocks = ((num_blocks + 31) // 32) * 32
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len)
    load_s = time.time() - t0
    logger.info(f"model load {load_s:.1f}s ({len(model.layers)} layers, {model.num_devices} devices)")
    assert _should_use_chunked_trace(model), "chunk-seq GDN prefill must be enabled"

    mesh = model.mesh_device
    vocab = model.args.vocab_size
    kv_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    signpost("compile_prefill")
    t0 = time.time()
    # All benched ISLs are >= CHUNK, so masked-bucket warmup is dead weight here.
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=CHUNK, warmup_masked_buckets=False)
    capture_s = time.time() - t0
    logger.info(f"prefill chunk-trace captured in {capture_s:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    ids_full = bench_prompt(max_isl, tokenizer)

    # Descending: the proven >=CHUNK points emit their BENCH_JSON before any
    # experimental sub-chunk point can fail the test.
    for isl in sorted(_ISLS, reverse=True):
        reps = _REPEATS if _REPEATS > 0 else (3 if isl <= 16384 else 1)
        ids = ids_full[:, :isl]
        bucket = -(-isl // CHUNK) * CHUNK
        pad = bucket - isl
        # Pad the bucket with the last real token (token 0 corrupts GDN state).
        padded = torch.cat([ids, ids[:, -1:].expand(1, pad)], dim=1) if pad else ids

        exec_samples, read_samples = [], []
        warmup_exec_s = None
        runs = reps + 1 if reps > 1 else reps
        signpost("inference_prefill")
        for rep in range(runs):
            # Repeats are safe without host-side state work: prefill_traced_chunked
            # zeroes GDN state itself at sequence start, and every KV page it reads
            # was written causally earlier in the same run.
            t0 = time.time()
            logits = model.prefill_traced_chunked(padded, page_table, actual_len=isl)
            ttnn.synchronize_device(mesh)
            exec_s = time.time() - t0

            t1 = time.time()
            if model.num_devices > 1:
                row = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                row = row.reshape(-1, vocab)[0].float()
            else:
                row = ttnn.to_torch(logits).float().reshape(-1)[:vocab]
            first_token = int(row.argmax())
            read_s = time.time() - t1
            assert not torch.isnan(row).any(), f"NaN in prefill logits at ISL {isl}"
            assert 0 <= first_token < vocab

            if rep == 0 and runs > reps:
                warmup_exec_s = exec_s
            else:
                exec_samples.append(exec_s)
                read_samples.append(read_s)
            logger.info(
                f"ISL {isl} rep {rep}: exec {exec_s:.3f}s ({exec_s / isl * 1000:.3f} ms/tok), "
                f"read {read_s * 1000:.1f}ms"
            )

        st = stats_ms(exec_samples)
        median_exec_s = st["median_ms"] / 1000.0
        median_read_s = sorted(read_samples)[len(read_samples) // 2]
        config = {
            "batch": 1,
            "isl": isl,
            "chunk": CHUNK,
            "repeats": len(exec_samples),
            "n_layers": len(model.layers),
            "num_devices": model.num_devices,
        }
        metrics = {
            "exec": st,
            "ms_per_token": round(median_exec_s / isl * 1000.0, 4),
            "prefill_tok_s": round(isl / median_exec_s, 1),
            "readback_ms": round(median_read_s * 1000.0, 2),
            "ttft_s": round(median_exec_s + median_read_s, 3),
            "capture_s": round(capture_s, 2),
            "warmup_exec_s": round(warmup_exec_s, 3) if warmup_exec_s is not None else None,
            "model_load_s": round(load_s, 1),
        }
        emit_bench_json("prefill", config, metrics)
        logger.info(
            f"[bench_prefill] ISL={isl}: {metrics['ms_per_token']} ms/token "
            f"({metrics['prefill_tok_s']} tok/s), TTFT {metrics['ttft_s']}s "
            f"(+{capture_s:.1f}s one-time capture)"
        )
