# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-execution workload for optimizer runs on Gemma 4 (any variant in HF_MODEL).

The Gemma 4 counterpart of models/tt_transformers/tests/test_optimizer_perf.py: the same stages,
signposts and printed lines, through gemma4's own Generator (Gemma4Generator.from_pretrained ->
prefill_forward_text -> decode_forward, as demo/text_demo_v2.py drives it, without its warmup of
every prefill length),
with on-device greedy sampling; each decode step's token comes back to the host, as in the demo.

  TT_PERF_OSL_TOKENS   output length (default 256: 255 timed decode steps)
  TT_PERF_TRACE=0      eager instead of trace; the device profiler also disables trace
  TT_PERF_LAYERS       cap the depth (profiler coverage); unset means every layer
  OPTIMIZER_DECODE_ONLY=1 / OPTIMIZER_PREFILL_ONLY=1  one stage only, as in the tt-transformers test

Weights come from the converted-weight store through a fresh per-run TT_CACHE_PATH.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.tt_transformers.tests.optimizer_weight_cache import RunCache

HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "google/gemma-4-26B-A4B-it"
MESH_SHAPE = (1, 4)
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_TRACE_REGION_SIZE") or 256_000_000)
MAX_SEQ_LEN = 1024
PAGE_BLOCK_SIZE = 32
INPUT_TOKENS = 128
PREFILL_SAMPLES = 3

STAGE_PREFILL = "prefill"
STAGE_DECODE = "decode"
STAGE_PATH = "trace+1cq"
DECODE_ONLY = os.environ.get("OPTIMIZER_DECODE_ONLY") == "1"
PREFILL_ONLY = os.environ.get("OPTIMIZER_PREFILL_ONLY") == "1"

CACHE_ROOT = Path(__file__).resolve().parents[4] / "generated" / "optimizer_cache"
PROFILER_BUFFER_AT_IMPORT = os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") or "default"

# The prompt: the gate's public-domain text, first INPUT_TOKENS tokens.
from models.demos.gemma4.tests.test_optimizer_gemma4_pcc import encode_text  # noqa: E402


def signpost(name: str, enabled: bool) -> None:
    """Tracy signpost when `enabled`. Call sites pass literal names (an optimizer's harness scanner
    reads `signpost("...")` calls from the source).

    Where the stage markers go depends on the mode. Under the device profiler (trace off) they
    bracket the measured steps, which is the window the profile is cut to. With trace on they
    bracket the steps that capture each trace, because those are the only steps whose ttnn calls
    the host issues: tt-opt's call-site discovery runs this test unprofiled, tags each call with
    the stage its signposts put it in, and a trace replay issues no calls to tag. Gated on the
    profiler alone, every record carried no stage and prefill and decode calls of one shape could
    not be told apart (2026-09-28). Outside a capture tracy.signpost is a tracy_message plus a
    log line, so the markers cost nothing and none sits inside a timed span."""
    if not enabled:
        return
    try:
        from tracy import signpost as tracy_signpost

        tracy_signpost(name)
        print(f"PERF_GATE_SIGNPOST {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [perf-gate] signpost {name!r} not emitted ({type(exc).__name__}: {exc})", flush=True)


def _full_depth() -> int:
    import json

    try:
        config = json.loads((Path(os.environ["HF_MODEL"]) / "config.json").read_text())
        return int(config.get("text_config", config)["num_hidden_layers"])
    except (OSError, ValueError, KeyError):
        return 0


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
def test_optimizer_gemma4_perf(monkeypatch):
    """Measure 128-token prefill and device-resident greedy decode on QB2."""
    from models.demos.gemma4.demo.sampling_utils import build_device_sampling_params, model_can_sample_on_device
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ["HF_MODEL"]
    mesh_device = generator = tt_kv_cache = None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = RunCache(CACHE_ROOT, "gemma4-perf-")
    try:
        monkeypatch.setenv("TT_CACHE_PATH", cache.path)
        depth = max(0, int(os.environ.get("TT_PERF_LAYERS", "0") or 0))
        output_tokens = max(2, int(os.environ.get("TT_PERF_OSL_TOKENS", "256")))
        decode_tokens = output_tokens - 1
        profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        enable_trace = not profiling and os.environ.get("TT_PERF_TRACE", "1") != "0"
        role = os.environ.get("PERF_GATE_ROLE", "manual")
        prefill_samples = 1 if profiling else PREFILL_SAMPLES
        print(
            f"GATE_CONFIG role={role} model={HF_MODEL_ID} profiling={int(profiling)} trace={int(enable_trace)} "
            f"layers={depth or _full_depth()} isl={INPUT_TOKENS} osl={output_tokens} decode_tokens={decode_tokens} "
            f"prefill_samples={prefill_samples} profiler_buffer={PROFILER_BUFFER_AT_IMPORT} "
            f"decode_only={int(DECODE_ONLY)} prefill_only={int(PREFILL_ONLY)}",
            flush=True,
        )
        assert not (DECODE_ONLY and PREFILL_ONLY), "OPTIMIZER_DECODE_ONLY and OPTIMIZER_PREFILL_ONLY are both set"
        if role == "verdict":
            assert depth == 0, f"verdict must measure every layer, got TT_PERF_LAYERS={depth}"
            assert enable_trace, "verdict must run trace+1cq, but TT_PERF_TRACE=0 disabled it"
            assert PREFILL_ONLY or decode_tokens >= 128, (
                f"verdict needs sustained decode; got {decode_tokens} token(s) from TT_PERF_OSL_TOKENS={output_tokens}"
            )
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
                trace_region_size=TRACE_REGION_SIZE,
                l1_small_size=24576,
                num_command_queues=1,
            )
            paged_attention_config = PagedAttentionConfig(
                block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)
            )
            lc = resolve_gemma4_demo_long_context(MAX_SEQ_LEN, mesh_device, model_path, paged_attention=True)
            generator, tt_kv_cache, _tokenizer = cache.build(
                lambda: Gemma4Generator.from_pretrained(
                    mesh_device=mesh_device,
                    model_path=model_path,
                    max_batch_size=1,
                    max_seq_len=MAX_SEQ_LEN,
                    num_layers=depth or None,
                    paged_attention_config=paged_attention_config,
                    bounded_sliding_kv_cache=lc["bounded_sliding"],
                ),
                # gemma4's own checkpoint loader, so the store's deferred weights engage here too.
                loaders=[(Gemma4ModelArgs, "load_state_dict")],
            )
            gc.collect()
            cache.loaded()

            model = generator.model[0]
            can_sample = model_can_sample_on_device(model)
            assert can_sample, "on-device sampling is unavailable; the production decode path samples on device"
            sampling_params = build_device_sampling_params({"temperature": 0, "top_p": 1.0}, can_sample=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            input_ids = torch.tensor([encode_text(tokenizer)[:INPUT_TOKENS]], dtype=torch.long)
            assert input_ids.shape[1] == INPUT_TOKENS
            page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
                1, paged_attention_config.max_num_blocks
            )

            # No warmup_model_prefill: it captures a prefill trace for every supported length (128,
            # 512 and 1024 here), and this test only ever runs 128 tokens. The two unused captures
            # were ~52 s of a ~111 s timed run (2026-09-28). The warm-up prefill() below compiles
            # and captures the 128-token trace on its first traced call (the generator captures a
            # length it has no trace for), before any decode trace exists, which is the order the
            # warmup kept.

            def prefill():
                out = generator.prefill_forward_text(
                    input_ids,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=[INPUT_TOKENS],
                    warmup_prefill=False,
                    enable_trace=enable_trace,
                    sampling_params=sampling_params,
                )
                first = out[0] if isinstance(out, tuple) else out
                return first.long().reshape(1, 1)

            def decode_step(out_tok, current_pos):
                # The demo's loop (demo/text_demo_v2.py): each step's sampled token comes back to the
                # host and is fed to the next.
                decode_out, _ = generator.decode_forward(
                    out_tok,
                    current_pos,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    enable_trace=enable_trace,
                    sampling_params=sampling_params,
                )
                return decode_out.long().view(1, 1)

            # Warm-up: one prefill, which compiles and captures the prefill trace before timing (and
            # compiles the eager path under the profiler). The decode trace is
            # captured by the first decode step after the timed prefills (the demo's order); capturing
            # it before them and replaying it after three prefill replays never completed on QB2
            # (2026-09-28), while the demo's prefill-then-decode order ran.
            # With trace on, this is the step whose prefill calls the host issues (see signpost).
            signpost("stage:prefill", enable_trace and not DECODE_ONLY)
            prefill()
            ttnn.synchronize_device(mesh_device)
            signpost("stage:prefill:end", enable_trace and not DECODE_ONLY)

            signpost("start", profiling and not DECODE_ONLY)
            prefill_ms = []
            first_token = None
            for _ in range(prefill_samples):
                signpost("stage:prefill", profiling and not DECODE_ONLY)
                started = time.perf_counter()
                first_token = prefill()
                ttnn.synchronize_device(mesh_device)
                prefill_ms.append((time.perf_counter() - started) * 1000.0)
                signpost("stage:prefill:end", profiling and not DECODE_ONLY)
            ttft_ms = statistics.median(prefill_ms)
            signpost("stop", profiling and PREFILL_ONLY)

            decode_seconds = 0.0
            if not PREFILL_ONLY:
                # Step 0 compiles and captures the decode trace; it is not timed (the demo excludes it
                # the same way). The timed steps follow it.
                # With trace on, step 0 is the step whose decode calls the host issues.
                current_pos = torch.tensor([INPUT_TOKENS], dtype=torch.int64)
                signpost("stage:decode", enable_trace)
                out_tok = decode_step(first_token, current_pos)
                current_pos += 1
                ttnn.synchronize_device(mesh_device)
                signpost("stage:decode:end", enable_trace)
                signpost("start", profiling and DECODE_ONLY)
                signpost("stage:decode", profiling)
                decode_start = time.perf_counter()
                for _ in range(decode_tokens):
                    out_tok = decode_step(out_tok, current_pos)
                    current_pos += 1
                ttnn.synchronize_device(mesh_device)
                decode_seconds = time.perf_counter() - decode_start
                signpost("stage:decode:end", profiling)
                signpost("stop", profiling)

            wall_ms = ttft_ms + decode_seconds * 1000.0
            samples = ",".join(f"{ms:.3f}" for ms in prefill_ms)
            if PREFILL_ONLY:
                print(f"PERF wall_ms={wall_ms:.3f} ttft_ms={ttft_ms:.3f} ttft_samples_ms={samples}", flush=True)
            else:
                decode_tokens_per_second = decode_tokens / decode_seconds
                per_token_ms = 1000.0 / decode_tokens_per_second
                print(
                    f"PERF wall_ms={wall_ms:.3f} ttft_ms={ttft_ms:.3f} ttft_samples_ms={samples} "
                    f"decode_tokens_per_second={decode_tokens_per_second:.3f}",
                    flush=True,
                )
            if not profiling and enable_trace:
                if not DECODE_ONLY:
                    for ms in prefill_ms:
                        print(f"TRACE_STAGE_MS[{STAGE_PREFILL}]={ms:.4f} path={STAGE_PATH}", flush=True)
                    print(f"TRACE_STAGE_ITEMS[{STAGE_PREFILL}]={INPUT_TOKENS}", flush=True)
                if PREFILL_ONLY:
                    print(f"TRACE_PER_TOKEN_MS={ttft_ms:.4f}", flush=True)
                    print("TRACE_HEADLINE_UNIT=inference", flush=True)
                    print(
                        f"TRACE_PIPELINE_MS={ttft_ms:.4f} TRACE_STAGES=1 (no recurring stage: headline=pipeline sum)",
                        flush=True,
                    )
                else:
                    print(f"TRACE_STAGE_MS[{STAGE_DECODE}]={per_token_ms:.4f} path={STAGE_PATH}", flush=True)
                    print(f"TRACE_PER_TOKEN_MS={per_token_ms:.4f}", flush=True)
                    print("TRACE_HEADLINE_UNIT=token", flush=True)
                    stages = 1 if DECODE_ONLY else 2
                    pipeline = per_token_ms if DECODE_ONLY else ttft_ms + per_token_ms
                    print(f"TRACE_PIPELINE_MS={pipeline:.4f} TRACE_STAGES={stages}", flush=True)
                print(f"TRACE_PREFILL_MS={ttft_ms:.6f}", flush=True)
                print(f"TRACE_PREFILL_PATH={STAGE_PATH}", flush=True)
                print(f"PERF_ISL_TOKENS={INPUT_TOKENS}", flush=True)
                print(f"DP=1 TP={MESH_SHAPE[1]} shard_active=True", flush=True)
                print(f"TRACE_REPLAY_PATH={STAGE_PATH} batch=1", flush=True)
            elif not profiling:
                print(f"FORWARD_WALL_MS={wall_ms:.6f}", flush=True)
        finally:
            # Drop every reference to device tensors before the mesh closes: the model, the
            # sampling parameters and the KV cache, not only the generator.
            generator = model = sampling_params = None
            tt_kv_cache = None
            gc.collect()
            if mesh_device is not None:
                for submesh in list(mesh_device.get_submeshes()):
                    if submesh is not mesh_device:
                        ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    finally:
        cache.cleanup()
