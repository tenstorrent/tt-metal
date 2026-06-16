# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ISL (input sequence length / context window) sweep for the Mistral-24B end-to-end
vision-text pipeline.

This reuses the exact e2e pipeline from ``test_end2end.py`` (real image + text prefill
via the vision tower, then autoregressive decode) and sweeps the context window
``max_seq_len`` while holding the output sequence length fixed at 200 tokens.

The only difference from ``test_end2end.py`` is the *text* prompt: instead of the fixed
question, the text is sourced from the *Tale of Two Cities* corpus
(``models/tt_transformers/tests/tale-of-two-cities.txt.bz2``) — the same long-context
input the tt_transformers prefill ISL tests use. The corpus is encoded and sliced so the
text length scales with the swept context window (leaving headroom for the image tokens
and the fixed output length). The image is still included, so the full vision+text model
is exercised.

The paged KV-cache block pool is scaled per-point so the cache can hold the full context
(the default 1024 blocks only covers ISL <= 32k). Sweep points whose image+text prompt
does not fit the context window (e.g. ISL 128 < image tokens) are skipped.

After the whole sweep, a summary of decode throughput (tok/s/user and aggregate tok/s)
per context length is logged.
"""

import bz2
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.model_config import DecodersPrecision

from models.experimental.mistral_24b.tests.pipeline_tests.test_end2end import (
    fabric_1d_trace_device_params,
    load_separate_models_like_test_end2end,
    process_real_vision_inputs,
    run_generation_exactly_like_test_end2end,
    setup_vision_model_args,
    validate_e2e_outputs,
)

# Output sequence length held fixed across the whole sweep.
OUTPUT_SEQ_LEN = 200

# Same image as test_end2end.py; the vision tower still runs so the full model is exercised.
IMAGE_URL = "https://img.freepik.com/premium-photo/girl-hugging-dog-with-girl-hugging-her_737761-2565.jpg"

# Long-context corpus shared with the tt_transformers prefill ISL tests.
TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)
PROMPT_FILE = os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/tale-of-two-cities.txt.bz2")

# Token headroom reserved (within max_seq_len) for the image's vision tokens + chat template
# overhead, so the image + text prompt fits the context window with room for OUTPUT_SEQ_LEN.
IMAGE_TOKEN_MARGIN = 1280

# (id, batch_size, max_seq_len) sweep points: batch 1 over context windows 128..128k.
ISL_SWEEP = [
    ("b1_isl4k", 1, 4 * 1024),
    ("b1_isl8k", 1, 8 * 1024),
    ("b1_isl16k", 1, 16 * 1024),
    ("b1_isl32k", 1, 32 * 1024),
    ("b1_isl64k", 1, 64 * 1024),
    ("b1_isl128k", 1, 128 * 1024),
]

# Accumulates per-point throughput across the parametrized sweep; logged once at module end.
_SWEEP_RESULTS = []


@pytest.fixture(scope="module", autouse=True)
def _log_isl_sweep_summary():
    """After all sweep points run, log decode throughput per context length."""
    yield
    if not _SWEEP_RESULTS:
        logger.info("ISL sweep summary: no points completed.")
        return
    logger.info("")
    logger.info("===== ISL sweep decode throughput summary =====")
    logger.info(f"{'config':<12} {'max_seq_len':>12} {'prefill_tok':>12} {'decode_t/s/u':>14} {'decode_t/s':>12}")
    for r in _SWEEP_RESULTS:
        logger.info(
            f"{r['sweep_id']:<12} {r['max_seq_len']:>12} {r['prefill_tokens']:>12} "
            f"{r['decode_t_s_u']:>14.2f} {r['decode_t_s']:>12.2f}"
        )
    logger.info("===============================================")


def build_tale_of_two_cities_text(model_args, max_seq_len):
    """Return a Tale-of-Two-Cities text chunk sized to (roughly) fill ``max_seq_len``.

    Mirrors the tt_transformers prefill tests' "encode corpus, slice to length" approach,
    adapted for the vision pipeline: the text token budget is ``max_seq_len`` minus the
    output length and headroom for the image's vision tokens. The exact prefill length is
    set by the processor (image + chat template + text) and is logged by the generation
    helper.
    """
    with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
        corpus = f.read()

    text_token_budget = max_seq_len - OUTPUT_SEQ_LEN - IMAGE_TOKEN_MARGIN
    if text_token_budget <= 0:
        # Context too small to even hold the image; use a tiny excerpt. The point is skipped
        # below because the image+text prompt exceeds the context window.
        text_token_budget = 16

    corpus_tokens = model_args.tokenizer.encode(corpus)
    chunk = model_args.tokenizer.decode(corpus_tokens[:text_token_budget])
    logger.info(
        f"Tale-of-Two-Cities text budget ~{text_token_budget} tokens "
        f"(max_seq_len={max_seq_len}, OSL={OUTPUT_SEQ_LEN}, image_margin={IMAGE_TOKEN_MARGIN})"
    )
    return chunk


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "weights",
    ["instruct"],
    ids=["full"],
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["accuracy"],
)
@pytest.mark.parametrize(
    "sweep_id, batch_size, max_seq_len",
    ISL_SWEEP,
    ids=[case[0] for case in ISL_SWEEP],
)
@pytest.mark.parametrize(
    "device_params",
    fabric_1d_trace_device_params(num_command_queues=1),  # Arch-adaptive trace region: 30 MiB WH / 35 MiB BH.
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P150x4": (1, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_isl_sweep(
    weights,
    sweep_id,
    max_seq_len,
    batch_size,
    page_params,
    optimizations,
    mesh_device,
    reset_seeds,
    request,
    device_params,
):
    """Sweep context window (ISL) for the e2e vision-text pipeline; OSL fixed at 200.

    Text prompt is sourced from the Tale of Two Cities corpus (the image is still included).
    """
    logger.info(f"Starting ISL sweep point '{sweep_id}': batch_size={batch_size}, max_seq_len={max_seq_len}")

    # bfloat8_b for memory efficiency, matching the e2e test.
    dtype = ttnn.bfloat8_b
    paged_attention = True

    # Setup vision-enabled model configuration. max_seq_len is the swept context window.
    model_args, instruct = setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations)

    # Build the multimodal prompt: same image as the e2e test, but the text comes from the
    # Tale of Two Cities corpus (sliced to scale with the context window).
    tale_text = build_tale_of_two_cities_text(model_args, max_seq_len)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_URL},
                {"type": "text", "text": tale_text},
            ],
        }
    ]
    processed_inputs = process_real_vision_inputs(messages, model_args)

    # The image's vision tokens (~1024) plus the (clamped) text may exceed a very small
    # context window. Skip cleanly rather than letting prefill assert "seq len exceeds
    # max seq len" (e.g. ISL 128 cannot hold the image).
    actual_isl = processed_inputs["input_ids"].shape[1]
    if actual_isl > max_seq_len:
        pytest.skip(
            f"prompt is {actual_isl} tokens (image + text) but context window max_seq_len={max_seq_len}; "
            f"cannot run this ISL point."
        )

    # Replicate the single-prompt processor output across the batch dimension.
    if batch_size > 1:
        for key in ("input_ids", "pixel_values", "attention_mask", "image_sizes"):
            value = processed_inputs.get(key)
            if torch.is_tensor(value):
                processed_inputs[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))

    # Scale the paged KV-cache block pool to the swept ISL. The cache must hold the prefill
    # (~max_seq_len) plus the decoded tokens (OUTPUT_SEQ_LEN); the default 1024 blocks
    # (32768 positions) only covers ISL <= 32k, so 64k/128k would overflow it and wedge decode.
    # NOTE: must run before load_separate_models, which allocates the text model's KV cache.
    block_size = page_params["page_block_size"]
    needed_positions = max_seq_len + OUTPUT_SEQ_LEN
    num_blocks = max(page_params["page_max_num_blocks"], -(-needed_positions // block_size))
    num_blocks = -(-num_blocks // batch_size) * batch_size  # page_table reshape needs divisibility by batch
    page_params = {"page_block_size": block_size, "page_max_num_blocks": num_blocks}
    logger.info(
        f"Paged KV cache: {num_blocks} blocks x {block_size} = {num_blocks * block_size} positions "
        f"(scaled for max_seq_len={max_seq_len}, batch={batch_size})"
    )

    # Load separate vision and text models following the e2e pattern.
    logger.info("Loading separate vision and text models...")
    vision_model, text_model = load_separate_models_like_test_end2end(
        model_args, mesh_device, dtype, paged_attention, page_params
    )

    # Prepare paged attention config + page table (same construction as the e2e test).
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )

    # Run prefill + decode; OSL fixed at OUTPUT_SEQ_LEN. metrics_out captures the computed
    # throughput so it can be summarized after the whole sweep.
    logger.info(f"Running generation: OSL={OUTPUT_SEQ_LEN}")
    metrics = {}
    results = run_generation_exactly_like_test_end2end(
        vision_model,
        text_model,
        processed_inputs,
        model_args,
        page_table,
        paged_attention_config,
        max_gen_len=OUTPUT_SEQ_LEN,
        metrics_out=metrics,
    )

    # Record this point's decode throughput for the end-of-sweep summary.
    _SWEEP_RESULTS.append(
        {
            "sweep_id": sweep_id,
            "max_seq_len": max_seq_len,
            "prefill_tokens": metrics.get("prefill_tokens", actual_isl),
            "decode_t_s_u": metrics.get("decode_t/s/u", 0.0),
            "decode_t_s": metrics.get("decode_t/s", 0.0),
        }
    )

    # Length-only validation; this is a throughput sweep, not a correctness check.
    validation_passed = validate_e2e_outputs(results, expected_min_tokens=1)
    assert (
        validation_passed and len(results) > 0
    ), f"ISL sweep '{sweep_id}' failed - generated {len(results)} tokens, validation: {validation_passed}"
    logger.info(f"ISL sweep point '{sweep_id}' PASSED — generated {len(results)} tokens")
