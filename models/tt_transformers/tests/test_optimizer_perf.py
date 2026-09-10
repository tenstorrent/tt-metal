# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-execution workload for production Llama optimizer runs."""

from __future__ import annotations

import gc
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.sampling import SamplingParams
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


INPUT_IDS = [128000] + [1000 + ((index * 7919) % 120000) for index in range(127)]
INPUT_TOKENS = len(INPUT_IDS)
MAX_SEQ_LEN = 2048

# Weight cache for the gate. create_tt_model takes a warm-cache shortcut (load_tensor_flatbuffer on
# the .tensorbin files) whenever TT_CACHE_PATH already holds a complete cache, and that shortcut has
# hung on the 1 GB embedding file under the profiler; a fresh directory forces the HF conversion
# path, which does not. Kept under the repo's gitignored generated/ so it can never be swept into a
# model commit and needs no directory outside the checkout.
CACHE_ROOT = Path(__file__).resolve().parents[3] / "generated" / "optimizer_cache"

# tt-metal reads every TT_METAL_* setting exactly once, when the shared library loads during
# `import ttnn` above (RunTimeOptions::InitializeFromEnvVars). A value set later in this process is
# never seen by the device runtime. Capture what the library actually saw so GATE_CONFIG can show
# whether a profiler buffer size reached it.
PROFILER_BUFFER_AT_IMPORT = os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") or "default"


def _prefill(generator, input_ids, page_table, kv_cache, sampling_params, enable_trace):
    result = generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[INPUT_TOKENS],
        sampling_params=sampling_params,
        enable_trace=enable_trace,
        warmup_prefill=False,
    )
    first_token = result[0] if isinstance(result, tuple) else result
    return first_token.reshape(1, 1)


def _decode(generator, first_token, input_ids, page_table, kv_cache, sampling_params, decode_tokens, enable_trace):
    current_pos = torch.tensor([INPUT_TOKENS], dtype=torch.int64)
    for decode_index in range(decode_tokens):
        generator.decode_forward(
            first_token,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=False,
            sampling_params=sampling_params,
            reset_batch=(decode_index == 0),
            prompt_tokens=input_ids if decode_index == 0 else None,
            output_tokens=first_token if decode_index == 0 else None,
        )
        current_pos += 1


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(1800)
def test_optimizer_direct_perf(monkeypatch):
    """Measure 128-token prefill and device-resident greedy decode on QB2.

    TT_PERF_LAYERS caps the depth for the profiler's coverage ladder; unset means all 32 layers.
    TT_PERF_OSL_TOKENS sets the output length; TT_PERF_TRACE=0 or the device profiler disables trace.
    """
    mesh_device = generator = model = model_args = state_dict = tt_kv_cache = None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="perf-", dir=str(CACHE_ROOT)) as cache_dir:
        monkeypatch.setenv("TT_CACHE_PATH", cache_dir)
        depth = max(0, int(os.environ.get("TT_PERF_LAYERS", "0") or 0))
        output_tokens = max(2, int(os.environ.get("TT_PERF_OSL_TOKENS", "256")))
        decode_tokens = output_tokens - 1
        profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        enable_trace = not profiling and os.environ.get("TT_PERF_TRACE", "1") != "0"
        # Every one of these comes from the environment, and a persistent shell carries stale values
        # across runs. Silent defaults turned two leaks into plausible-looking numbers: OSL=2 timed a
        # single token as if it were sustained decode, and TT_PERF_TRACE=0 reported eager dispatch as
        # the traced production path. State the resolved configuration, then refuse the ones that
        # cannot be a production verdict rather than measuring them anyway.
        role = os.environ.get("PERF_GATE_ROLE", "manual")
        print(
            f"GATE_CONFIG role={role} profiling={int(profiling)} trace={int(enable_trace)} "
            f"layers={depth or 32} isl={INPUT_TOKENS} osl={output_tokens} decode_tokens={decode_tokens} "
            f"profiler_buffer={PROFILER_BUFFER_AT_IMPORT}",
            flush=True,
        )
        if role == "verdict":
            # Only the full-pipeline verdict is held to this. The coverage probe runs this same node
            # at OSL=1 to read op signatures, and tracy runs it capped -- both are legitimate.
            assert depth == 0, f"verdict must measure all 32 layers, got TT_PERF_LAYERS={depth}"
            assert enable_trace, "verdict must run trace+1cq, but TT_PERF_TRACE=0 disabled it"
            assert decode_tokens >= 128, (
                f"verdict needs sustained decode; got {decode_tokens} token(s) "
                f"from TT_PERF_OSL_TOKENS={output_tokens}"
            )
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 4),
                trace_region_size=52_000_000,
                num_command_queues=1,
            )
            paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
            optimizations = lambda args: DecodersPrecision.performance(args.n_layers, args.model_name)
            model_args, model, tt_kv_cache, state_dict = create_tt_model(
                mesh_device,
                instruct=True,
                max_batch_size=1,
                optimizations=optimizations,
                max_seq_len=MAX_SEQ_LEN,
                paged_attention_config=paged_attention_config,
                dtype=ttnn.bfloat8_b,
                num_layers=depth or None,
                use_prefetcher=False,
                use_hf_rope=False,
            )
            expected_layers = depth or 32
            assert model_args.n_layers == expected_layers
            assert len(model.layers) == expected_layers
            assert mesh_device.get_num_devices() == 4
            assert model._supports_on_device_sampling
            assert model.sampling is not None
            state_dict = None
            gc.collect()

            generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
            input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)
            page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
                1, paged_attention_config.max_num_blocks
            )
            kv_cache = [tt_kv_cache]
            sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=0)

            warmup_token = _prefill(generator, input_ids, page_table, kv_cache, sampling_params, enable_trace)
            generator.decode_forward(
                warmup_token,
                torch.tensor([INPUT_TOKENS], dtype=torch.int64),
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                read_from_device=False,
                sampling_params=sampling_params,
                reset_batch=True,
                prompt_tokens=input_ids,
                output_tokens=warmup_token,
            )
            ttnn.synchronize_device(mesh_device)

            start = time.perf_counter()
            first_token = _prefill(generator, input_ids, page_table, kv_cache, sampling_params, enable_trace)
            ttnn.synchronize_device(mesh_device)
            ttft_end = time.perf_counter()
            _decode(
                generator,
                first_token,
                input_ids,
                page_table,
                kv_cache,
                sampling_params,
                decode_tokens,
                enable_trace,
            )
            ttnn.synchronize_device(mesh_device)
            end = time.perf_counter()

            decode_seconds = end - ttft_end
            ttft_ms = (ttft_end - start) * 1000.0
            decode_tokens_per_second = decode_tokens / decode_seconds
            print(
                f"PERF wall_ms={(end - start) * 1000.0:.3f} "
                f"ttft_ms={ttft_ms:.3f} "
                f"decode_tokens_per_second={decode_tokens_per_second:.3f}",
                flush=True,
            )
            if not profiling and enable_trace:
                print(f"TRACE_PER_TOKEN_MS={1000.0 / decode_tokens_per_second:.6f}", flush=True)
                print(f"TRACE_PREFILL_MS={ttft_ms:.6f}", flush=True)
                print("TRACE_HEADLINE_UNIT=token", flush=True)
                print(f"PERF_ISL_TOKENS={INPUT_TOKENS}", flush=True)
                print("DP=1 TP=4 shard_active=True", flush=True)
                print("TRACE_REPLAY_PATH=trace+1cq batch=1", flush=True)
                print("TRACE_PREFILL_PATH=trace+1cq", flush=True)
            elif not profiling:
                print(f"FORWARD_WALL_MS={(end - start) * 1000.0:.6f}", flush=True)
        finally:
            generator = None
            model = None
            model_args = None
            state_dict = None
            tt_kv_cache = None
            gc.collect()
            if mesh_device is not None:
                for submesh in list(mesh_device.get_submeshes()):
                    if submesh is not mesh_device:
                        ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
