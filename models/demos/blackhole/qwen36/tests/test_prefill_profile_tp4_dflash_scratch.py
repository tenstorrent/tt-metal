# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH TTFT harness for the TP=4 DFlash2 prompt prefill (lane L, opt round 5): the eager, tap-capturing
prefill_for_spec the DFlash serving class runs for every prompt (batch8-dflash2: max_batch_size 8, slot 0), with the
drafter context fill (DFlash2ServingDecoder.ingest_prompt) per chunk, vs the same prefill without taps and vs the plain
traced chunk path, in one process on the (1,4) mesh.

    MESH_DEVICE=P150x4 TT_MESH_GRAPH_DESC_PATH=.../p300_x2_mesh_graph_descriptor.textproto HF_MODEL=.../weights_qwen38 \
    DFLASH_WEIGHTS=.../dflash_weights PROFILE_ISLS=2048,8192,32768 PROFILE_MODES=taps,notaps,traced \
    pytest models/demos/blackhole/qwen36/tests/test_prefill_profile_tp4_dflash_scratch.py

Env: PROFILE_ISLS (prompt lengths; the model is sized for the largest), PROFILE_MODES (comma list of taps / notaps /
traced; traced captures the plain chunk trace, so it runs LAST), PROFILE_REPEATS (default 2), PROFILE_REAL_PROMPT=1
(the 4k sample text, repeated), PROFILE_DUMP_LOGITS=<tag> (logits_<tag>_<mode>_<isl>.pt in PROFILE_OUT_DIR),
PROFILE_B (max_batch_size, default 8), PROFILE_CHUNK_TIMING=1 (per-chunk host stamps: prefill part vs on_chunk part).
"""

import hashlib
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "l1_small_size": 24576,
        "trace_region_size": 1024 * 1024 * 1024,
    }
]
BLOCK_SIZE = 64
CHUNK = 2048
_OUT_DIR = os.environ.get("PROFILE_OUT_DIR", "/home/ttuser/experiments/qwen36_27b/profiles/opt_round5/laneL")


def _prompt_ids(model, isl):
    torch.manual_seed(0)
    token_ids = torch.randint(1000, 100000, (1, isl), dtype=torch.int64)
    if os.environ.get("PROFILE_REAL_PROMPT") == "1":
        import json

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
        with open("models/demos/blackhole/qwen36/demo/sample_prompts/input_data_long_4k.json") as f:
            text = json.load(f)[0]["prompt"]
        ids = tok(text, return_tensors="pt")["input_ids"]
        while ids.shape[1] < isl:
            ids = torch.cat([ids, ids], dim=1)
        token_ids = ids[:, :isl].to(torch.int64)
    return token_ids


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_profile_tp4_dflash(mesh_device):
    from models.demos.blackhole.qwen36.tt.dflash2_serving import DFlash2DualBucketDecoder

    device = mesh_device
    device.enable_program_cache()
    isls = [int(v) for v in os.environ.get("PROFILE_ISLS", "2048,8192,32768").split(",")]
    modes = os.environ.get("PROFILE_MODES", "taps,notaps,traced").split(",")
    assert "traced" not in modes or modes[-1] == "traced", "traced parks a plain trace: run it last"
    repeats = int(os.environ.get("PROFILE_REPEATS", "2"))
    B = int(os.environ.get("PROFILE_B", "8"))
    top = ((max(isls) + CHUNK - 1) // CHUNK) * CHUNK
    nb = ((max(64, (top + 256 + BLOCK_SIZE - 1) // BLOCK_SIZE) + 31) // 32) * 32
    t0 = time.time()
    # PROFILE_LAYERS="0,1,2,3": a layer subset for tracy per-op profiles (notaps / traced modes only: the taps live at
    # layers 5..61).
    _layers = os.environ.get("PROFILE_LAYERS")
    layer_indices = [int(v) for v in _layers.split(",")] if _layers else None
    assert layer_indices is None or "taps" not in modes, "taps need the full model"
    model = Qwen36Model.from_pretrained(
        device, max_batch_size=B, max_seq_len=nb * BLOCK_SIZE, layer_indices=layer_indices
    )
    logger.info(f"[PROFILE] model load {time.time() - t0:.1f}s layers={len(model.layers)} B={B} blocks={nb}")
    kv_shape = [nb + 1, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat8_b, batch_size=B)
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    model.set_gdn_fused_decode(True)  # the serving class does this before building the decoder
    dec = None
    if "taps" in modes:
        dec = DFlash2DualBucketDecoder(model, num_blocks=nb)  # batch8-dflash2: buckets 8x4 + 4x8
        dec.alloc()
    chunk_log = os.environ.get("PROFILE_CHUNK_TIMING") == "1"

    results = {}
    for mode in modes:
        if mode == "traced":
            # The plain B>1 path: the chunk trace bakes the persistent B=1 prefill scratch (bound for capture and for
            # every replay, as prefill_paged_slots does).
            _prev = model._bind_gdn_prefill_scratch()
            t0 = time.time()
            model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK)
            logger.info(f"[PROFILE] plain chunk trace capture {time.time() - t0:.1f}s")
        for isl in isls:
            ids = _prompt_ids(model, isl)
            stamps = []

            def on_chunk(hidden, chunk_start, valid_len):
                stamps.append(("c", time.perf_counter()))
                taps = model.take_dflash_eager_taps()
                if mode == "taps":
                    dec.ingest_prompt(0, taps, chunk_start + valid_len, chunk_start=chunk_start)
                elif taps is not None:
                    for t in taps:
                        ttnn.deallocate(t)
                if chunk_log:
                    ttnn.synchronize_device(device)
                stamps.append(("i", time.perf_counter()))

            def run():
                if mode == "traced":
                    return model.prefill_traced_chunked(ids, page_table, actual_len=isl)
                if dec is not None:
                    dec.ctx_len[0] = 0
                model._dflash_tap = mode == "taps"
                try:
                    return model.prefill_for_spec(ids.to(torch.int32), page_table, isl, on_chunk, slot=0)
                finally:
                    model._dflash_tap = False

            lg = run()  # warm (compiles anything lazily built)
            ttnn.synchronize_device(device)
            times = []
            for _ in range(repeats):
                stamps.clear()
                t0 = time.perf_counter()
                stamps.append(("0", t0))
                lg = run()
                ttnn.synchronize_device(device)
                times.append(time.perf_counter() - t0)
            if chunk_log and stamps:
                prev, parts = stamps[0][1], []
                for tag, t in stamps[1:]:
                    parts.append(f"{tag}{(t - prev) * 1e3:.1f}")
                    prev = t
                print(f"PROFILE_CHUNKS mode={mode} isl={isl} " + " ".join(parts))
            if hasattr(lg, "shape") and not isinstance(lg, torch.Tensor):
                lt = ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).reshape(-1)[: model.vocab_size].float()
            else:
                lt = lg.reshape(-1)[: model.vocab_size].float()
            dump = os.environ.get("PROFILE_DUMP_LOGITS")
            if dump:
                torch.save(lt, f"{_OUT_DIR}/logits_{dump}_{mode}_{isl}.pt")
            # Durable numerics record (review F1): hash per (mode, isl), and in-process equality across modes below.
            print(f"PROFILE_LOGITS_SHA256 mode={mode} isl={isl} {hashlib.sha256(lt.numpy().tobytes()).hexdigest()}")
            results[(mode, isl)] = lt.clone()
            t2 = torch.topk(lt, 2)
            print(
                f"PROFILE_RESULT mode={mode} isl={isl} ttft_s={min(times):.4f} mean_s={sum(times) / len(times):.4f} "
                f"argmax={int(t2.indices[0])} top2gap={float(t2.values[0] - t2.values[1]):.4f}"
            )
    # Cross-mode logits equality, logged (review F1): e.g. taps (eager prompt prefill) vs traced (plain chunk path).
    for isl in isls:
        present = [m for m in modes if (m, isl) in results]
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                x, y = results[(a, isl)], results[(b, isl)]
                print(
                    f"PROFILE_EQUAL isl={isl} {a}_vs_{b} torch_equal={torch.equal(x, y)} "
                    f"max_abs={float((x - y).abs().max()):.6g}"
                )
