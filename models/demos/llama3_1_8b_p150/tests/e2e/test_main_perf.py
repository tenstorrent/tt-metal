# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PERFORMANCE profile for the 'main' pipeline of the self-contained
Llama-3.1-8B-Instruct demo.

Derived from the demo's PCC/correctness flow (``simple_text_demo.test_demo_text``),
but stripped to ONLY the on-device TTNN forward: no reference/torch model, no
token-accuracy, no PCC / verify_accuracy / verify_perf comparisons. The build +
prefill + decode run IN-PROCESS (never via a subprocess / pytest node-id) so
tracy can instrument every device op.

The heavy axis for an LLM is TOKENS. We cap both depth (num_layers via
TT_PERF_LAYERS) and generated tokens (TT_PERF_MAX_NEW_TOKENS) to a small
representative pass; a full-length decode is a correctness stress size, not what
a perf profile needs. The device profiler is drained model-agnostically by
wrapping every ttnn ``FastOperation`` so the 12000-marker buffer never overflows.
"""
from __future__ import annotations

import os

# Pin identity BEFORE importing the demo (the demo module also pins it on import).
os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"

import time

import pytest
import torch

import ttnn
from models.demos.llama3_1_8b_p150.demo.simple_text_demo import get_default_mesh_device_param, prepare_generator_args
from models.demos.llama3_1_8b_p150.tt.common import preprocess_inputs_prefill, sample_host
from models.demos.llama3_1_8b_p150.tt.generator import Generator, SamplingParams
from models.demos.llama3_1_8b_p150.tt.model_config import DecodersPrecision

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# --- Bounded build config (small, representative; NOT the demo's production shapes) ---
INSTRUCT = True
MAX_SEQ_LEN = int(os.environ.get("TT_PERF_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = 1
DATA_PARALLEL = 1
PAGED_ATTENTION = True
PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
NUM_LAYERS = int(os.environ.get("TT_PERF_LAYERS", "2"))
USE_PREFETCHER = False
USE_HF_ROPE = False


def _optimizations(model_args):
    return DecodersPrecision.performance(model_args.n_layers, model_args.model_name)


_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# Match the demo's own device_params (fabric + 2 CQs), then reserve the trace budget when profiling.
_DEV_PARAMS = {"fabric_config": True, "num_command_queues": 2}
if _PERF_TRACE:
    # Reserve the trace + 2-CQ budget at device-open, ONCE, for baseline and every candidate: the
    # second queue and the trace region exist before any candidate runs, so trace+2CQ is the fixed
    # measurement mode (never a per-candidate downgrade for lack of a queue). A device/config that
    # genuinely can't open 2 CQs still degrades gracefully in measure_adapter; override with TT_PERF_NUM_CQ.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), get_default_mesh_device_param())
    ],
    indirect=True,
)
def test_main_perf(mesh_device, reset_seeds):
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
    counter = [0]
    _orig = []

    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try:
                    ttnn.ReadDeviceProfiler(mesh_device)
                except Exception:
                    pass
            return r

        return inner

    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":
                _orig.append((_mod, _n, _op))
                setattr(_mod, _n, _draining(_op))

    out = None
    _fw_ms = 0.0
    try:
        # 1) build the pipeline EXACTLY as the demo does (prepare_generator_args + Generator), bounded.
        global_batch_size = BATCH_SIZE * DATA_PARALLEL
        (
            model_args,
            model,
            page_table,
            tt_kv_cache,
            tokenizer,
            processor,
            local_data_parallel,
            local_submesh_indices,
        ) = prepare_generator_args(
            num_devices=num_devices,
            data_parallel=DATA_PARALLEL,
            mesh_device=mesh_device,
            instruct=INSTRUCT,
            global_batch_size=global_batch_size,
            optimizations=_optimizations,
            max_seq_len=MAX_SEQ_LEN,
            page_params=PAGE_PARAMS,
            paged_attention=PAGED_ATTENTION,
            num_layers=NUM_LAYERS,
            use_prefetcher=USE_PREFETCHER,
            use_hf_rope=USE_HF_ROPE,
        )
        global_batch_size = BATCH_SIZE * local_data_parallel

        generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

        # Small, representative prompt (NOT the demo's full-length input file).
        input_prompts = ["Hello, tell me a short story."] * global_batch_size

        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts, tokenizer, model_args, INSTRUCT, PERF_MAX_NEW_TOKENS, max_prefill_len=MAX_SEQ_LEN
        )
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

        # argmax (greedy) sampling, matching the batch-1 demo config.
        sampling_params = {"temperature": 0, "top_p": 0.08, "top_k": 32}
        device_sampling_params = (
            SamplingParams(
                temperature=sampling_params["temperature"],
                top_k=sampling_params["top_k"],
                top_p=sampling_params["top_p"],
                seed=None,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                enable_log_probs=False,
            )
            if model[0]._supports_on_device_sampling
            else None
        )
        prefill_sampling_params = device_sampling_params

        _fw0 = time.monotonic()

        # --- Prefill (single bounded forward, no trace so every op is visible to tracy) ---
        prefill_out = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            sampling_params=prefill_sampling_params,
            warmup_prefill=False,
            enable_trace=False,
        )
        if prefill_sampling_params is not None and isinstance(prefill_out, tuple):
            prefilled_token, _prefill_log_probs = prefill_out
        else:
            logits = prefill_out
            prefilled_token = torch.argmax(logits, dim=-1)

        # --- Decode loop, capped to PERF_MAX_NEW_TOKENS ---
        current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])
        out_tok = prefilled_token
        for iteration in range(PERF_MAX_NEW_TOKENS):
            logits, log_probs = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                reset_batch=(iteration == 0),
                sampling_params=device_sampling_params,
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
            )
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)
            else:
                _, out_tok = sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )
            current_pos += 1

        out = out_tok
        _fw_ms = (time.monotonic() - _fw0) * 1000.0

        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)

    print("FORWARD_WALL_MS=%.4f" % _fw_ms)
    assert out is not None  # perf only — NO PCC

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.llama3_1_8b_p150.tt.pipeline import build_pipeline

                return build_pipeline(
                    dev,
                    instruct=INSTRUCT,
                    max_seq_len=MAX_SEQ_LEN,
                    batch_size=BATCH_SIZE,
                    data_parallel=DATA_PARALLEL,
                    paged_attention=PAGED_ATTENTION,
                    page_params=PAGE_PARAMS,
                    num_layers=NUM_LAYERS,
                    use_prefetcher=USE_PREFETCHER,
                    use_hf_rope=USE_HF_ROPE,
                    optimizations=_optimizations,
                )

            _prompt_ids = [128000, 9906, 11, 3371]
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
            # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
            # contract for pipelines that expose only decode_step.
            _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, mesh_device, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
