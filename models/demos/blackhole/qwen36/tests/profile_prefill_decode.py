# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy / wall-clock profiling harness for the SERVED Qwen3.8-27B path, in-process on a mesh.

Builds the model the way qwen36_vllm.Qwen36ForCausalLM does (create_tt_model via Qwen36Model.from_pretrained,
HF_MODEL=Qwen/Qwen3.8-27B resolved from the offline HF snapshot, batched [BMAX,...] GDN state + paged KV), captures
the prefill chunk trace against the B=1 GDN scratch (warmup_model_prefill), then:

  phase 1  prefill_paged_slots for ISL 128 and 4096 (PROFILE_PREFILL_REPEATS each), wall-clocked, tracy signposts
           prefill_<isl>_start / prefill_<isl>_stop around each ISL block;
  phase 2  traced decode at widths PROFILE_DECODE_WIDTHS (default "1,32" = the b1 bucket and the full row width)
           for PROFILE_DECODE_STEPS (50) steps, inputs refreshed per step via copy_host_to_device as served decode
           does; PROFILE_DEVICE_SAMPLING=1 adds the per-bucket on-device greedy sampling trace (served
           sample_on_device_mode=decode_only); PROFILE_DECODE_EAGER=1 runs the steps eagerly instead (per-op tracy
           rows; test_decode_profile_scratch.py notes tracy's ops report may choke on trace ops).

Wrapper: scripts/profile_tp8.sh [tag]  (MESH_DEVICE=P150x8 -> TP=8; P150x4 + TT_VISIBLE_DEVICES=2,3,4,5 -> TP=4)
Direct:  MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/profile_prefill_decode.py -x -s
Env: PROFILE_ISLS ("128,4096"), PROFILE_PREFILL_REPEATS (3), PROFILE_DECODE_WIDTHS ("1,32"), PROFILE_DECODE_STEPS (50),
     PROFILE_BMAX (32), PROFILE_DEVICE_SAMPLING (0), PROFILE_DECODE_EAGER (0), TT_METAL_DEVICE_PROFILER (tracy).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device

try:
    from tracy import signpost
except ImportError:  # plain pytest run without the tracy package on the path

    def signpost(*_a, **_k):
        pass


def _env_list(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x]


BMAX = int(os.environ.get("PROFILE_BMAX", "32"))
ISLS = _env_list("PROFILE_ISLS", "128,4096")
PREFILL_REPEATS = int(os.environ.get("PROFILE_PREFILL_REPEATS", "3"))
WIDTHS = _env_list("PROFILE_DECODE_WIDTHS", f"1,{BMAX}")
STEPS = int(os.environ.get("PROFILE_DECODE_STEPS", "50"))
DEVICE_SAMPLING = os.environ.get("PROFILE_DEVICE_SAMPLING", "0") == "1"
DECODE_EAGER = os.environ.get("PROFILE_DECODE_EAGER", "0") == "1"
CHUNK = 2048  # qwen36_vllm._PREFILL_WARMUP_CHUNK
PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"

# Blocks per user: longest prompt + 256 slack + decode steps, rounded to a multiple of 8 (page-table stick % 32 == 0).
_need = max(ISLS) + 256 + STEPS
BPU = ((-(-_need // BLOCK_SIZE) + 7) // 8) * 8


def _read_profiler(device):
    if PROFILING and hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.ReadDeviceProfiler(device)  # drain device profiler buffers between phases (text_demo does this)


def _stats(ms):
    return f"min {min(ms):.1f} / med {sorted(ms)[len(ms) // 2]:.1f} / max {max(ms):.1f} ms"


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_profile_prefill_decode(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    assert all(1 <= w <= BMAX for w in WIDTHS), f"PROFILE_DECODE_WIDTHS {WIDTHS} must be within [1, {BMAX}]"

    # --- model: same construction as Qwen36ForCausalLM.initialize_vllm_model (HF_MODEL from env) ---
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(
        f"[prof] model load {time.perf_counter() - t0:.1f}s  mesh={_MESH_SHAPE} layers={len(model.layers)} "
        f"BMAX={BMAX} BPU={BPU} ISLs={ISLS} widths={WIDTHS} steps={STEPS} device_sampling={DEVICE_SAMPLING}"
    )
    if DEVICE_SAMPLING and model.sampling is None:
        pytest.skip("on-device sampling unsupported on this mesh/vocab")

    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    results = {}
    try:
        # --- prefill warmup: mirror Qwen36ForCausalLM.warmup_model_prefill (batched -> B=1 scratch bound) ---
        t0 = time.perf_counter()
        pt_full = torch.arange(BMAX * BPU, dtype=torch.int32).reshape(1, -1)  # BMAX*BPU is a multiple of 32
        prev = model._bind_gdn_prefill_scratch()
        try:
            model.capture_prefill_trace_chunked(device, pt_full, chunk_size=CHUNK, capture_chunk_trace=True)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        model.warmup_gdn_slot_write()
        for layer in model.layers:
            if not layer.is_full_attention and hasattr(layer.attention, "warmup_hist_device_pack"):
                layer.attention.warmup_hist_device_pack()
        ttnn.synchronize_device(device)
        logger.info(
            f"[prof] prefill warmup (chunk trace + masked buckets + slot-write + hist pack) {time.perf_counter() - t0:.1f}s"
        )
        _read_profiler(device)

        # --- phase 1: served prefill (prefill_paged_slots) per ISL ---
        torch.manual_seed(0)
        for isl in ISLS:
            ids = torch.randint(1000, 100000, (1, isl), dtype=torch.int32)
            # untimed warm request: anything lazily allocated/compiled happens here, not in the profiled region
            model.prefill_paged_slots([ids], page_tables[0:1], [0], valid_lens=[isl])
            ttnn.synchronize_device(device)
            times = []
            signpost(f"prefill_{isl}_start")
            for rep in range(PREFILL_REPEATS):
                slot = (rep + 1) % BMAX
                t0 = time.perf_counter()
                lg = model.prefill_paged_slots([ids], page_tables[slot : slot + 1], [slot], valid_lens=[isl])
                ttnn.synchronize_device(device)
                times.append(1e3 * (time.perf_counter() - t0))
            signpost(f"prefill_{isl}_stop")
            top = int(lg[0].reshape(-1)[: model.vocab_size].float().argmax())
            results[f"prefill_{isl}"] = min(times)
            logger.info(f"[prof] PREFILL isl={isl} x{PREFILL_REPEATS}: {_stats(times)}  (last top token {top})")
            _read_profiler(device)

        # --- phase 2: decode at each bucket width (test_decode_width_scaling_traced + served per-step input refresh) ---
        pos0 = min(ISLS)
        for width in WIDTHS:
            model._reset_gdn_state_for_new_sequence()  # in place: keeps the prefill trace's baked addresses valid
            tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
            positions = torch.full((width,), pos0, dtype=torch.int32)
            pt = page_tables[:width]
            on_dev = DEVICE_SAMPLING

            # compile run (eager) so the capture records only pre-compiled programs
            dev0 = model.prepare_inputs_decode(tokens, positions, pt)
            lg0 = model.ttnn_decode_forward(
                dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3], on_device_logits=on_dev
            )
            if on_dev:
                model.sampling.set_trace_bucket(width)
                model.sampling.sample(lg0, enable_trace=False)
            ttnn.synchronize_device(device)

            host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
            dev = copy_host_to_device(host, mesh_device=device)
            tid = None
            if not DECODE_EAGER:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                lg = model.ttnn_decode_forward(
                    dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=on_dev
                )
                ttnn.end_trace_capture(device, tid, cq_id=0)
                ttnn.synchronize_device(device)
                if on_dev:  # per-bucket sampling trace bound to THIS bucket's logits tensor
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(device)
                    model.sampling.sample(lg, enable_trace=True, skip_precompile=True)
                    ttnn.synchronize_device(device)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)  # warm replay
                ttnn.synchronize_device(device)

            step_ms, sampled = [], None
            signpost(f"decode_w{width}_start")
            for s in range(STEPS):
                positions = torch.full((width,), pos0 + 1 + s, dtype=torch.int32)
                host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
                t0 = time.perf_counter()
                copy_host_to_device(host_tensors=host[:3], device_tensors=dev[:3])
                if DECODE_EAGER:
                    out = model.ttnn_decode_forward(
                        dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=on_dev
                    )
                    if on_dev:
                        sampled = model.sampling.sample(out, enable_trace=False)
                else:
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    if on_dev:
                        sampled = model.sampling.sample(lg, enable_trace=True)
                ttnn.synchronize_device(device)
                step_ms.append(1e3 * (time.perf_counter() - t0))
            signpost(f"decode_w{width}_stop")
            if sampled is not None:
                tok = sampled[0] if isinstance(sampled, tuple) else sampled
                ids = model.process_output_decode(tok, width, is_tokens=True)
                logger.info(f"[prof] width={width} sampled ids (first 4): {ids[:4].tolist()}")
            results[f"decode_w{width}"] = sorted(step_ms)[len(step_ms) // 2]
            logger.info(
                f"[prof] DECODE width={width} {'eager' if DECODE_EAGER else 'traced'} x{STEPS}: {_stats(step_ms)} "
                f"(incl. per-step input copy){'  + device sampling' if on_dev else ''}"
            )
            if on_dev:
                model.sampling.reset_trace()
            if tid is not None:
                ttnn.release_trace(device, tid)
            _read_profiler(device)
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()
    print("PROFILE_RESULT " + " ".join(f"{k}={v:.2f}ms" for k, v in results.items()))
