# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 ISL sweep — batch-1 / batch-8 / batch-32 and long-context perf rows.

Each pytest id matches one README Per-ISL table row. Run the full sweep via
``models/demos/gemma4/scripts/run_text_demo_v2_isl_sweep.sh`` or one bucket:

    MESH_DEVICE=T3K HF_MODEL=google/gemma-4-31B-it pytest \\
        models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "long-context-128k" -s --timeout 1800

``CI=true`` skips every row except ``ci-1`` unless ``GEMMA4_ISL_SWEEP=1``.
"""

import os

import pytest

from models.demos.gemma4.demo.text_demo_v2 import (
    _device_params,
    _mesh_device_param,
    _run_spec_decode,
    _run_spec_decode_batched,
    load_inputs,
    run_demo_text,
)


# Parameters mirror the Gemma3 demo layout (subset): a latency config, a long-context
# config, and a CI config. Gemma4 runs batch=1, so throughput/DP rows are omitted.
@pytest.mark.parametrize(
    "input_prompts, instruct, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, "
    "sampling_params, stop_at_eos, ci_only, enable_trace",
    [
        (  # batch-1 (latency) — single user, short prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1024,
            1,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # batch-8 (throughput) — 8 concurrent users, short prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1024,
            8,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # batch-32 (max throughput) — 32 concurrent users (decode batch ceiling).
            # max_seq_len=1024 (short prompts; matches batch-8). Prefill is micro-
            # batched at ≤4 users (GEMMA4_MAX_BATCHED_PREFILL_USERS): true B≥8
            # wedges on P150x8 after the first all_gather. See generator.py.
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1024,
            32,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-4k — single user, long prompt
            "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json",
            True,
            4096,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        # NOTE on long-context-32k/64k/128k/256k (see GEMMA4_LONG_CONTEXT_POLICY):
        #   Coherence target: 4k–128k on QB2 + LoudBox (12B/31B) and P150 (≤12B).
        #   Defaults (MESH_DEVICE + HF_MODEL) — no extra env needed:
        #     QB2: 31B bounded @ 64k, chunk=2048 @ ≥128k; 12B/26B unbound→128k
        #     P150x8: 31B/26B unbound→64k, bounded+chunk=2048 @ ≥128k
        #             (unbounded 128k → "lapped…"); 12B/E2B/E4B unbound→256k
        #     P150: E2B/E4B unbound→256k; 12B bounded+chunked @ ≥64k →256k
        #   Override: GEMMA4_BOUNDED_SLIDING, GEMMA4_GEN_PREFILL_CHUNK,
        #   GEMMA4_DEMO_SINGLE_CHUNK (avoid for quality).
        (  # long-context-32k
            "models/tt_transformers/demo/sample_prompts/input_data_long_32k.json",
            True,
            32 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 512},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-64k
            "models/tt_transformers/demo/sample_prompts/input_data_long_64k.json",
            True,
            64 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-128k
            "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json",
            True,
            128 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-256k — 31B policy auto multi-chunk (DRAM)
            "models/tt_transformers/demo/sample_prompts/input_data_long_256k.json",
            True,
            256 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # ci-1 — single user, fixed iteration count for perf tracking
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            8192,
            1,
            512,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            False,
            True,
            True,
        ),
    ],
    ids=[
        "batch-1",
        "batch-8",
        "batch-32",
        "long-context-4k",
        "long-context-32k",
        "long-context-64k",
        "long-context-128k",
        "long-context-256k",
        "ci-1",
    ],
)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        # MESH_DEVICE → (rows, cols); unset → (1, N) over all visible devices.
        _mesh_device_param()
    ],
    indirect=True,
)
def test_demo_text(
    input_prompts,
    instruct,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    stop_at_eos,
    mesh_device,
    is_ci_env,
    ci_only,
    enable_trace,
    reset_seeds,
    request,
):
    """Gemma4 text generation through the Generator interface, modeled on the Gemma3 demo."""
    _isl_sweep = os.environ.get("GEMMA4_ISL_SWEEP", "").lower() in ("1", "true", "yes")
    if is_ci_env and not ci_only and not _isl_sweep:
        pytest.skip("CI only runs the CI-only configs")

    max_generated_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", max_generated_tokens))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max_seq_len))
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None
    batch_size = int(os.environ.get("GEMMA4_BATCH", batch_size))
    _decode_trace = os.environ.get("GEMMA4_DECODE_TRACE")
    if _decode_trace is not None:
        enable_trace = _decode_trace.lower() in ("1", "true", "yes")

    if request.config.getoption("--speculative"):
        draft_len = request.config.getoption("--spec-draft-len")
        if draft_len is None:
            draft_len = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
        if batch_size != 1:
            prompts = load_inputs(input_prompts, batch_size, instruct)
            _run_spec_decode_batched(
                prompts=prompts,
                instruct=instruct,
                max_seq_len=max_seq_len,
                max_generated_tokens=max_generated_tokens,
                page_params=page_params,
                sampling_params=sampling_params,
                mesh_device=mesh_device,
                enable_trace=enable_trace,
                draft_len=draft_len,
                num_layers=num_layers,
                input_prompts=input_prompts,
            )
            return
        prompt = load_inputs(input_prompts, 1, instruct)[0]
        _run_spec_decode(
            prompt=prompt,
            instruct=instruct,
            max_seq_len=max_seq_len,
            max_generated_tokens=max_generated_tokens,
            page_params=page_params,
            sampling_params=sampling_params,
            mesh_device=mesh_device,
            enable_trace=enable_trace,
            draft_len=draft_len,
            num_layers=num_layers,
            input_prompts=input_prompts,
        )
        return

    run_demo_text(
        input_prompts=input_prompts,
        instruct=instruct,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        max_generated_tokens=max_generated_tokens,
        paged_attention=paged_attention,
        page_params=page_params,
        sampling_params=sampling_params,
        stop_at_eos=stop_at_eos,
        mesh_device=mesh_device,
        is_ci_env=is_ci_env,
        enable_trace=enable_trace,
        num_layers=num_layers,
    )
