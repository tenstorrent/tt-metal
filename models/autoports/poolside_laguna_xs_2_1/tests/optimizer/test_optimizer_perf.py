# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-execution workload for tt_hw_planner optimizer runs on Laguna-S-2.1 (p150x4).

Measures the serving decode path: one captured decode trace per token (embed -> layers -> final norm ->
vocab-sharded LM head -> Sampling1D greedy on device -> token fed back on device, positions advanced on device),
replayed non-blocking with one sync at the end (token-out, no per-token readback).

Environment: TT_PERF_LAYERS caps the depth (unset or 0 = all 48 layers); TT_PERF_OSL_TOKENS sets the output
length; TT_METAL_DEVICE_PROFILER=1 or TT_PERF_TRACE=0 runs eager; OPTIMIZER_DECODE_ONLY=1 opens the profiled
window at decode; PERF_GATE_ROLE=verdict refuses anything but every layer, trace on and >= 128 decode tokens.
"""

from __future__ import annotations

import os
import statistics
import time

# Serving knobs of the qualified p150x4 Laguna-S profile (serve_vllm.sh), set before any Laguna import.
for _k, _v in {
    "TT_LAGUNA_MODEL": "poolside/Laguna-S-2.1",
    "LAGUNA_PROFILE": "p150x4",
    "TT_VISIBLE_DEVICES": "0,1,2,3",
    "LAGUNA_FABRIC_CONFIG": "FABRIC_1D_RING",
    "TT_LAGUNA_CCL_TOPOLOGY": "ring",
    "TT_LAGUNA_CCL_NUM_LINKS": "2",
    "TT_LAGUNA_DECODE_SDPA_PC": "1",
    "TT_LAGUNA_DECODE_K": "64",
    "TT_LAGUNA_DECODE_EXP": "0",
    "TT_LAGUNA_DECODE_MAXCORES": "16",
    "TT_LAGUNA_PREFILL_FAST": "1",
    "TT_LAGUNA_PIPE_CHUNK": "2048",
    "TT_LAGUNA_WEIGHT_CACHE": "/home/ttuser/benchmark-results/laguna-opt-weight-cache",
}.items():
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile  # noqa: E402
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator  # noqa: E402

# The checkpoint this gate runs; a literal so the optimizer's roofline can resolve the model.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "poolside/Laguna-S-2.1"
FULL_DEPTH = 48
TRACE_REGION = max(int(os.environ.get("TT_PERF_TRACE_REGION") or 0), 200_000_000)
INPUT_IDS = [1000 + ((index * 7919) % 99000) for index in range(128)]
INPUT_TOKENS = len(INPUT_IDS)
MAX_SEQ_LEN = 2048
PREFILL_SAMPLES = 3

STAGE_PREFILL = "prefill"
STAGE_DECODE = "decode"
STAGE_PATH = "trace+1cq"
DECODE_ONLY = os.environ.get("OPTIMIZER_DECODE_ONLY") == "1"
PROFILER_BUFFER_AT_IMPORT = os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") or "default"


PERF_HISTORY_PATH = os.environ.get("OPTIMIZER_PERF_HISTORY") or "/home/ttuser/benchmark-results/ashwary-laguna/perf_history.jsonl"


def _append_perf_history(record: dict) -> None:
    """One JSON line per gate run, so decode speed can be followed over the optimizer's lifetime."""
    import json
    import subprocess

    try:
        record["tree"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        os.makedirs(os.path.dirname(PERF_HISTORY_PATH), exist_ok=True)
        with open(PERF_HISTORY_PATH, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception as error:  # noqa: BLE001
        print(f"PERF_HISTORY not written ({error})", flush=True)


def signpost(name: str, enabled: bool) -> None:
    """Tracy signpost, only under the device profiler. Call sites pass literal names (the optimizer's harness
    scanner reads `signpost("...")` calls from the source)."""
    if not enabled:
        return
    try:
        from tracy import signpost as tracy_signpost

        tracy_signpost(name)
        print(f"PERF_GATE_SIGNPOST {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [perf-gate] signpost {name!r} not emitted ({type(exc).__name__}: {exc})", flush=True)


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
def test_optimizer_direct_perf():
    """128-token prefill, then device-resident greedy decode on QB2 (p150x4)."""
    depth = max(0, int(os.environ.get("TT_PERF_LAYERS", "0") or 0))
    output_tokens = max(2, int(os.environ.get("TT_PERF_OSL_TOKENS", "256")))
    decode_tokens = output_tokens - 1
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    enable_trace = not profiling and os.environ.get("TT_PERF_TRACE", "1") != "0"
    role = os.environ.get("PERF_GATE_ROLE", "manual")
    prefill_samples = 1 if profiling else PREFILL_SAMPLES
    print(
        f"GATE_CONFIG role={role} model={HF_MODEL_ID} profiling={int(profiling)} trace={int(enable_trace)} "
        f"layers={depth or FULL_DEPTH} isl={INPUT_TOKENS} osl={output_tokens} decode_tokens={decode_tokens} "
        f"prefill_samples={prefill_samples} profiler_buffer={PROFILER_BUFFER_AT_IMPORT} decode_only={int(DECODE_ONLY)} "
        f"trace_region={TRACE_REGION}",
        flush=True,
    )
    if role == "verdict":
        assert depth == 0, f"verdict must measure every layer, got TT_PERF_LAYERS={depth}"
        assert enable_trace, "verdict must run trace+1cq, but TT_PERF_TRACE=0 disabled it"
        assert decode_tokens >= 128, f"verdict needs sustained decode; got {decode_tokens} token(s)"

    mesh = gen = None
    try:
        mesh = open_mesh(ttnn, resolve_profile("p150x4", trace_region_size=TRACE_REGION))
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=MAX_SEQ_LEN, num_layers=depth or None)
        assert len(gen.model.layers) == (depth or FULL_DEPTH)
        assert mesh.get_num_devices() == 4
        m = gen.model
        gen._ensure_cache(1, MAX_SEQ_LEN)
        kv, pt = gen._kv_cache, gen._page_table
        P = INPUT_TOKENS

        def prefill():
            x = m.embed_prefill(gen._tokens_to_device(torch.tensor(INPUT_IDS)))
            h = m.prefill_layers(x, kv, pt, user_id=0, start_pos=0)
            last = ttnn.slice(h, [0, P - 1, 0], [1, P, gen.hidden])
            tb = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
            gen._greedy_sample(m.lm_head_shards_decode(ttnn.reshape(last, (1, 1, 1, gen.hidden))), 1, tb)
            return tb

        # Warm-up: compile prefill, and (trace mode) capture the decode trace at position P.
        first = gen._read_token(prefill(), 1)[0]
        ttnn.synchronize_device(mesh)
        if enable_trace:
            st = gen._decode_trace_state(1, pt, P, first)
            tok, cur, ridx, tid = st["tok"], st["cur"], st["ridx"], st["tid"]

            def stage():
                ttnn.copy_host_to_device_tensor(gen._host_rank4_tok(first), tok)
                ttnn.copy_host_to_device_tensor(gen._host_pos(P), cur)
                ttnn.copy_host_to_device_tensor(gen._host_ridx(P), ridx)

            _ = stage()
            for _ in range(4):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
        else:
            gen.decode_forward(torch.tensor([[first]]), torch.tensor([P]), page_table=pt, kv_cache=kv)
            ttnn.synchronize_device(mesh)

        signpost("start", profiling and not DECODE_ONLY)
        prefill_ms = []
        for _ in range(prefill_samples):
            signpost("stage:prefill", profiling and not DECODE_ONLY)
            t = time.perf_counter()
            # Assigned, not a bare call: the optimizer brackets the last bare no-argument call on the
            # profiled path with start/stop signposts (agent.stage_marks), which would move the profiled
            # window from decode to prefill. This gate's own signposts already mark the decode stage.
            _ = prefill()
            ttnn.synchronize_device(mesh)
            prefill_ms.append((time.perf_counter() - t) * 1000.0)
            signpost("stage:prefill:end", profiling and not DECODE_ONLY)
        ttft_ms = statistics.median(prefill_ms)

        if enable_trace:
            _ = stage()
            ttnn.synchronize_device(mesh)
        signpost("start", profiling and DECODE_ONLY)
        signpost("stage:decode", profiling)
        decode_start = time.perf_counter()
        if enable_trace:
            for _ in range(decode_tokens):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        else:
            token = first
            for i in range(decode_tokens):
                token = int(gen.decode_forward(torch.tensor([[token]]), torch.tensor([P + i]), page_table=pt, kv_cache=kv)[0])
        ttnn.synchronize_device(mesh)
        end = time.perf_counter()
        signpost("stage:decode:end", profiling)
        signpost("stop", profiling)

        decode_seconds = end - decode_start
        decode_tokens_per_second = decode_tokens / decode_seconds
        per_token_ms = 1000.0 / decode_tokens_per_second
        wall_ms = ttft_ms + decode_seconds * 1000.0
        print(
            f"PERF wall_ms={wall_ms:.3f} ttft_ms={ttft_ms:.3f} "
            f"ttft_samples_ms={','.join(f'{ms:.3f}' for ms in prefill_ms)} "
            f"decode_tokens_per_second={decode_tokens_per_second:.3f}",
            flush=True,
        )
        _append_perf_history(
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "role": role,
                "profiling": int(profiling),
                "trace": int(enable_trace),
                "layers": depth or FULL_DEPTH,
                "decode_tokens": decode_tokens,
                "decode_ms_per_token": round(per_token_ms, 4),
                "decode_tokens_per_second": round(decode_tokens_per_second, 3),
                "ttft_ms": round(ttft_ms, 3),
            }
        )
        if not profiling and enable_trace:
            if not DECODE_ONLY:
                for ms in prefill_ms:
                    print(f"TRACE_STAGE_MS[{STAGE_PREFILL}]={ms:.4f} path={STAGE_PATH}", flush=True)
                print(f"TRACE_STAGE_ITEMS[{STAGE_PREFILL}]={INPUT_TOKENS}", flush=True)
            print(f"TRACE_STAGE_MS[{STAGE_DECODE}]={per_token_ms:.4f} path={STAGE_PATH}", flush=True)
            print(f"TRACE_PER_TOKEN_MS={per_token_ms:.4f}", flush=True)
            print("TRACE_HEADLINE_UNIT=token", flush=True)
            if DECODE_ONLY:
                print(f"TRACE_PIPELINE_MS={per_token_ms:.4f} TRACE_STAGES=1", flush=True)
            else:
                print(f"TRACE_PIPELINE_MS={ttft_ms + per_token_ms:.4f} TRACE_STAGES=2", flush=True)
            print(f"TRACE_PREFILL_MS={ttft_ms:.6f}", flush=True)
            print(f"TRACE_PREFILL_PATH={STAGE_PATH}", flush=True)
            print(f"PERF_ISL_TOKENS={INPUT_TOKENS}", flush=True)
            print("DP=1 TP=4 shard_active=True", flush=True)
            print(f"TRACE_REPLAY_PATH={STAGE_PATH} batch=1", flush=True)
        elif not profiling:
            print(f"FORWARD_WALL_MS={wall_ms:.6f}", flush=True)
    finally:
        if gen is not None:
            try:
                gen.teardown()
            except Exception:  # noqa: BLE001
                pass
        gen = None
        if mesh is not None:
            close_mesh(ttnn, mesh)
