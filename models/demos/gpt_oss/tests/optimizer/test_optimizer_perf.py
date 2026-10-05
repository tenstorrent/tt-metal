# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode workload for optimizer runs on gpt-oss-20b (QuietBox 2, 1x4 mesh, batch 1, all 24 layers).

Measures decode the way demo/text_demo.py measures it for prefill_128: the demo's first prompt is prefilled,
then Generator.decode_forward runs with the decode trace and on-device greedy sampling, one call per token,
each call reading the sampled token back to the host. The first decode call compiles and captures the trace
and is not timed; decode ms/token is the mean wall time of the remaining calls (tokens/s/user = 1000 / that).

Environment: TT_PERF_LAYERS caps the depth (unset or 0 = all 24 layers); TT_PERF_OSL_TOKENS sets the number of decode calls (default 129: 1 capture + 128 timed);
TT_METAL_DEVICE_PROFILER=1 or TT_PERF_TRACE=0 runs eager decode, and under the profiler the decode window is
capped at TT_PERF_PROFILE_DECODE_TOKENS (default 4) eager steps; OPTIMIZER_DECODE_ONLY=1 opens the profiled
window at decode; PERF_GATE_ROLE=verdict refuses anything but every layer, trace on and >= 128 timed decode tokens.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_MODEL", "openai/gpt-oss-20b")
# Weight upload from the ttnn cache through the pinned host-memory path is ~10x slower on this QB2 (#57763);
# the copy path is used instead. Host load time only; device timing is unaffected.
os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
os.environ.setdefault(
    "TT_CACHE_PATH", str(Path(__file__).resolve().parents[5] / "generated" / "gpt_oss_20b_tt_cache")
)

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.gpt_oss.tests.optimizer.gate_support import (  # noqa: E402
    MESH_SHAPE,
    PROMPTS,
    append_history,
    build_generator,
    device_params,
    greedy_sampling_params,
)

# The checkpoint this gate runs; a literal so the optimizer's roofline can resolve the model.
HF_MODEL_ID = "openai/gpt-oss-20b"
FULL_DEPTH = 24
PREFILL_SAMPLES = 3

STAGE_PREFILL = "prefill"
STAGE_DECODE = "decode"
STAGE_PATH = "trace+1cq"
DECODE_ONLY = os.environ.get("OPTIMIZER_DECODE_ONLY") == "1"
PROFILER_BUFFER_AT_IMPORT = os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") or "default"


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


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", [(MESH_SHAPE, device_params())], ids=["1x4"], indirect=True)
def test_optimizer_direct_perf(mesh_device, device_params):
    """The demo's prefill_128 prompt, then traced batch-1 decode with on-device greedy sampling on QB2 (1x4)."""
    depth = max(0, int(os.environ.get("TT_PERF_LAYERS", "0") or 0))
    output_tokens = max(2, int(os.environ.get("TT_PERF_OSL_TOKENS", "129")))
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if profiling:
        output_tokens = min(output_tokens, 1 + max(1, int(os.environ.get("TT_PERF_PROFILE_DECODE_TOKENS", "4"))))
    enable_trace = not profiling and os.environ.get("TT_PERF_TRACE", "1") != "0"
    timed_tokens = output_tokens - 1
    role = os.environ.get("PERF_GATE_ROLE", "manual")
    prefill_samples = 1 if profiling else PREFILL_SAMPLES
    print(
        f"GATE_CONFIG role={role} model={HF_MODEL_ID} profiling={int(profiling)} trace={int(enable_trace)} "
        f"layers={depth or FULL_DEPTH} decode_calls={output_tokens} timed_decode_tokens={timed_tokens} "
        f"prefill_samples={prefill_samples} profiler_buffer={PROFILER_BUFFER_AT_IMPORT} decode_only={int(DECODE_ONLY)} "
        f"trace_region={device_params.get('trace_region_size', 'yaml')}",
        flush=True,
    )
    if role == "verdict":
        assert depth == 0, f"verdict must measure every layer, got TT_PERF_LAYERS={depth}"
        assert enable_trace, "verdict must run trace+1cq, but TT_PERF_TRACE=0 disabled it"
        assert timed_tokens >= 128, f"verdict needs sustained decode; got {timed_tokens} timed token(s)"

    generator, model_args, model, page_table, kv_cache, tokenizer = build_generator(mesh_device, num_layers=depth or None)
    assert model[0].args.n_layers == (depth or FULL_DEPTH) and mesh_device.get_num_devices() == 4

    prompt = json.loads(PROMPTS.read_text())[0]["prompt"]
    prompt_ids = model_args[0].encode_prompt(prompt, instruct=False)
    P = len(prompt_ids)

    def prefill():
        return generator.prefill_forward_text(
            torch.tensor([prompt_ids]), page_table=page_table, kv_cache=kv_cache, prompt_lens=[P], enable_trace=False
        )

    # Warm-up prefill (compile), as the demo's compile_prefill pass.
    _ = prefill()
    signpost("start", profiling and not DECODE_ONLY)
    prefill_ms = []
    first = None
    for _ in range(prefill_samples):
        signpost("stage:prefill", profiling and not DECODE_ONLY)
        t = time.perf_counter()
        logits = prefill()
        first = int(torch.argmax(logits.reshape(-1)))
        prefill_ms.append((time.perf_counter() - t) * 1000.0)
        signpost("stage:prefill:end", profiling and not DECODE_ONLY)
    ttft_ms = statistics.median(prefill_ms)

    sampling = greedy_sampling_params()
    out_tok = torch.tensor([first])
    current_pos = torch.tensor([P])
    step_ms = []
    tokens = [first]

    def decode_step(iteration):
        return generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling,
            reload_inputs=iteration == 0 or not enable_trace,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=iteration == 0,
        )

    # Decode call 0 compiles (and in trace mode captures) the decode program; it is outside the timed window.
    out_tok, _ = decode_step(0)
    out_tok = out_tok.reshape(-1)[:1]
    tokens.append(int(out_tok[0]))
    current_pos += 1

    signpost("start", profiling and DECODE_ONLY)
    signpost("stage:decode", profiling)
    decode_start = time.perf_counter()
    for iteration in range(1, output_tokens):
        t = time.perf_counter()
        out_tok, _ = decode_step(iteration)
        out_tok = out_tok.reshape(-1)[:1]
        tokens.append(int(out_tok[0]))
        step_ms.append((time.perf_counter() - t) * 1000.0)
        current_pos += 1
    ttnn.synchronize_device(mesh_device)
    decode_seconds = time.perf_counter() - decode_start
    signpost("stage:decode:end", profiling)
    signpost("stop", profiling)

    per_token_ms = statistics.fmean(step_ms)
    tokens_per_second_per_user = 1000.0 / per_token_ms
    wall_ms = ttft_ms + decode_seconds * 1000.0
    print(f"GATE_TEXT {tokenizer.decode(tokens)[:300]!r}", flush=True)
    print(
        f"PERF wall_ms={wall_ms:.3f} ttft_ms={ttft_ms:.3f} "
        f"ttft_samples_ms={','.join(f'{ms:.3f}' for ms in prefill_ms)} decode_ms_per_token={per_token_ms:.4f} "
        f"decode_median_ms={statistics.median(step_ms):.4f} decode_tokens_per_second_per_user={tokens_per_second_per_user:.3f}",
        flush=True,
    )
    append_history(
        "OPTIMIZER_PERF_HISTORY",
        {
            "role": role,
            "profiling": int(profiling),
            "trace": int(enable_trace),
            "layers": depth or FULL_DEPTH,
            "decode_tokens": timed_tokens,
            "decode_ms_per_token": round(per_token_ms, 4),
            "decode_median_ms": round(statistics.median(step_ms), 4),
            "decode_tokens_per_second_per_user": round(tokens_per_second_per_user, 3),
            "ttft_ms": round(ttft_ms, 3),
        },
    )
    if not profiling and enable_trace:
        if not DECODE_ONLY:
            for ms in prefill_ms:
                print(f"TRACE_STAGE_MS[{STAGE_PREFILL}]={ms:.4f} path={STAGE_PATH}", flush=True)
            print(f"TRACE_STAGE_ITEMS[{STAGE_PREFILL}]={P}", flush=True)
        print(f"TRACE_STAGE_MS[{STAGE_DECODE}]={per_token_ms:.4f} path={STAGE_PATH}", flush=True)
        print(f"TRACE_PER_TOKEN_MS={per_token_ms:.4f}", flush=True)
        print("TRACE_HEADLINE_UNIT=token", flush=True)
        if DECODE_ONLY:
            print(f"TRACE_PIPELINE_MS={per_token_ms:.4f} TRACE_STAGES=1", flush=True)
        else:
            print(f"TRACE_PIPELINE_MS={ttft_ms + per_token_ms:.4f} TRACE_STAGES=2", flush=True)
        print(f"TRACE_PREFILL_MS={ttft_ms:.6f}", flush=True)
        print(f"TRACE_PREFILL_PATH={STAGE_PATH}", flush=True)
        print(f"PERF_ISL_TOKENS={P}", flush=True)
        print("DP=1 TP=4 shard_active=True", flush=True)
        print(f"TRACE_REPLAY_PATH={STAGE_PATH} batch=1", flush=True)
    elif not profiling:
        print(f"FORWARD_WALL_MS={wall_ms:.6f}", flush=True)
