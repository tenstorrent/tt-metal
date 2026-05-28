# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Gemma4 prefill trace policy for standalone and vLLM generators."""

import torch
from loguru import logger

from models.tt_transformers.tt.generator import (
    MAX_BATCHED_PREFILL_SEQ_LEN,
    SUPPORTED_PREFILL_BATCH_SIZES,
    max_prefill_chunk_size_cutoff,
)

# Kernel sequence lengths that may capture/replay prefill device traces (MoE only).
GEMMA4_TRACE_PREFILL_SEQ_LENS = [128, 512, 1024, 2048, 4096]


def model_uses_pli(model) -> bool:
    """True for E2B/E4B-style models with per-layer-input embeddings."""
    return bool(getattr(model, "hidden_size_per_layer_input", 0))


def patch_gemma4_trace_model_args(model_args, *, prefill_trace_enabled: bool = True) -> None:
    """Configure trace_prefill_supported_seq_lens and can_enable_trace on model_args."""
    if prefill_trace_enabled:
        model_args.trace_prefill_supported_seq_lens = list(GEMMA4_TRACE_PREFILL_SEQ_LENS)
        uses_pli = bool(getattr(model_args, "hidden_size_per_layer_input", 0))
        model_args.can_enable_trace = (
            lambda prefill_seq_len, num_cached_tokens=0: not uses_pli
            and num_cached_tokens == 0
            and prefill_seq_len in model_args.trace_prefill_supported_seq_lens
        )
    else:
        model_args.trace_prefill_supported_seq_lens = []
        model_args.can_enable_trace = lambda prefill_seq_len, num_cached_tokens=0: False


def maybe_disable_pli_prefill_trace(enable_trace: bool, model, batch_size: int = 1) -> bool:
    """Return False when PLI prefill must not use trace capture.

    PLI prefill uploads per-layer inputs via ttnn.from_torch inside forward, which
    triggers TT_FATAL during trace capture. Decode trace is unaffected.
    """
    if enable_trace and model_uses_pli(model):
        logger.info(
            "Disabling prefill trace on PLI model (batch_size={}): "
            "in-forward ttnn.from_torch PLI upload is incompatible with trace capture",
            batch_size,
        )
        return False
    return enable_trace


def warmup_gemma4_batched_prefill_traces(
    generator,
    kv_cache,
    *,
    enable_trace: bool,
    can_sample_on_device,
    non_greedy_decoding_on_device,
) -> None:
    """Capture prefill traces for MoE models across batch sizes and trace ISLs.

    Sweeps ``SUPPORTED_PREFILL_BATCH_SIZES`` × ``trace_prefill_supported_seq_lens``,
    skipping combinations that meet or exceed ``MAX_BATCHED_PREFILL_SEQ_LEN`` (128k).
    Caller must set ``generator.already_warmed_up_prefill`` before calling if needed,
    or this function sets it on entry.
    """
    if generator.already_warmed_up_prefill:
        return
    generator.already_warmed_up_prefill = True

    model_args = generator.model_args[0]
    sequence_lengths_to_warmup = model_args.get_warmup_prefill_supported_seq_lens()
    trace_isls = set(model_args.trace_prefill_supported_seq_lens)
    max_batch_size = model_args.max_batch_size
    warmup_batch_sizes = tuple(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b <= max_batch_size)

    logger.info(
        "Gemma4 batched prefill trace warmup: batch sizes {} x trace ISLs {}",
        warmup_batch_sizes,
        sorted(trace_isls),
    )

    skip_sequence_lengths = False
    sampling_parameters_sweeped = False

    for model_id in range(generator.data_parallel):
        for supported_length in sequence_lengths_to_warmup:
            if supported_length not in trace_isls:
                continue
            if model_id != 0 and (supported_length not in trace_isls or not enable_trace):
                continue

            for batch_size in warmup_batch_sizes:
                if batch_size * supported_length >= MAX_BATCHED_PREFILL_SEQ_LEN:
                    logger.info(
                        "Skipping batched prefill trace warmup for batch_size={}, seq_len={}: "
                        "exceeds {} token limit",
                        batch_size,
                        supported_length,
                        MAX_BATCHED_PREFILL_SEQ_LEN,
                    )
                    continue

                warmup_args = generator._mock_tokens(batch_size, supported_length, kv_cache, model_id)

                if warmup_args["page_table"] is None and max_prefill_chunk_size_cutoff(
                    supported_length, model_args.max_prefill_chunk_size
                ):
                    logger.warning(
                        "Skipping warmup for sequence lengths after: {} because they are greater than "
                        "the max prefill chunk size and paged attention is disabled",
                        supported_length,
                    )
                    skip_sequence_lengths = True
                    break

                if not sampling_parameters_sweeped:
                    sampling_params = generator._create_sampling_params(
                        can_sample_on_device=can_sample_on_device,
                        non_greedy_decoding_on_device=non_greedy_decoding_on_device,
                        batch_size=batch_size,
                    )
                else:
                    sampling_params = [None]

                for param in sampling_params:
                    logger.info(
                        "Warming up prefill trace for sequence length: {} batch size: {} " "with sampling params: {}",
                        supported_length,
                        batch_size,
                        param,
                    )
                    generator.prefill_forward_text(
                        **warmup_args,
                        kv_cache=kv_cache,
                        enable_trace=enable_trace,
                        model_id_warmup=model_id,
                        sampling_params=param,
                    )

                sampling_parameters_sweeped = True

            if skip_sequence_lengths:
                break

        if skip_sequence_lengths:
            break

    if getattr(model_args, "is_multimodal", False):
        vision_chunk_size = getattr(model_args, "vision_chunk_size", 896)
        vision_channels = getattr(model_args, "vision_in_channels", 3)
        model_id = 0
        warmup_pixel_values = [torch.zeros((1, vision_channels, vision_chunk_size, vision_chunk_size))]
        prefill_forward_args = generator._mock_tokens(1, 128, kv_cache, model_id)

        logger.info("Warming up vision encoder with image size {}x{}", vision_chunk_size, vision_chunk_size)
        generator.prefill_forward_text(
            **prefill_forward_args,
            kv_cache=kv_cache,
            enable_trace=False,
            model_id_warmup=model_id,
            sampling_params=None,
            pixel_values=warmup_pixel_values,
            image_sizes=[(vision_chunk_size, vision_chunk_size)],
        )
        logger.info("Vision encoder warmup completed")
