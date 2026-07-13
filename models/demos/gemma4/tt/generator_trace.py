# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Gemma4 prefill trace policy for standalone and vLLM generators."""

import os

import torch
from loguru import logger

from models.tt_transformers.tt.generator import (
    MAX_BATCHED_PREFILL_SEQ_LEN,
    SUPPORTED_PREFILL_BATCH_SIZES,
    max_prefill_chunk_size_cutoff,
)

# Kernel sequence lengths that may capture/replay prefill device traces (MoE only).
#
# Each bucket is captured as a *separate resident prefill trace* at warmup, and
# the high buckets are large for big models (e.g. the 4096 bucket on
# Gemma4-26B-A4B is ~0.5 GB on its own). ``GEMMA4_TRACE_PREFILL_SEQ_LENS`` lets a
# deployment trim the set to the prompt lengths it actually serves — e.g. a
# throughput benchmark with short prompts only needs the smallest bucket — which
# directly shrinks the required ``trace_region_size``. The override drives both
# warmup capture (``patch_gemma4_trace_model_args``) and runtime eligibility
# (``can_gemma4_enable_prefill_trace``) so they stay consistent.
_DEFAULT_TRACE_PREFILL_SEQ_LENS = [128, 512, 1024, 2048, 4096]


def _resolve_trace_prefill_seq_lens() -> list[int]:
    override = os.environ.get("GEMMA4_TRACE_PREFILL_SEQ_LENS")
    if override is None:
        return list(_DEFAULT_TRACE_PREFILL_SEQ_LENS)
    return [int(x) for x in override.split(",") if x.strip()]


GEMMA4_TRACE_PREFILL_SEQ_LENS = _resolve_trace_prefill_seq_lens()

# Prefill trace is disabled above 4k ISL (no perf gain, OOM risk) and at or above 32k
# batched virtual tokens (batch_size × padded prefill length). Prefills above the
# cap are instead kept safe by generator-level chunking (see
# resolve_gemma4_prefill_chunk_size): an 8192 prompt runs as 4096-token chunks
# rather than one full-length op, so it neither wedges the fetch queue (#49083)
# nor OOMs a whole-length trace.
GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN = 4096
GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS = 32 * 1024

# Default generator-level prefill chunk when GEMMA4_GEN_PREFILL_CHUNK is unset,
# applied on QB2 ONLY (P150x4 = 1x4 Blackhole / P300X2, the board this was
# validated on).
#
# Every other model bounds its prefill chunk: the shared tt_transformers/Qwen
# generator defaults max_prefill_chunk_size to a per-(model,device) value (4096
# for combos without a table entry, which is Gemma4's case), DeepSeek uses a
# tiered chunk table, and Qwen3.6 chunk-traces at 2048. Gemma4 alone used to
# default to a SINGLE full-length chunk (max_seq_len in the vLLM generator, a
# power-of-2 in the demo generator) — divergent from each other and from the
# reference, and exactly the unbounded full-sequence prefill op that wedges the
# fetch queue at ISL>=8192 (#49083) or OOMs a whole-length trace.
#
# 4096 is the largest chunk validated on QB2 to both trace safely
# (<= GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN) and clear the #49083 wedge
# (repro_prefill_hang.py, REPRO_ISLS=8192,8192 -> ALL_DONE, no wedge/OOM). It is
# NOT applied to other boards (Wormhole T3K/TG, single-chip Blackhole, etc.),
# where neither the wedge nor this number was validated — those keep the
# caller's prior default (see resolve_gemma4_prefill_chunk_size).
GEMMA4_DEFAULT_PREFILL_CHUNK = 4096

# Device name (models.tt_transformers.tt.model_config.determine_device_name) of
# the QB2 board the bounded default is validated for: 4-chip Blackhole mesh.
_QB2_DEVICE_NAME = "P150x4"


def _is_qb2(mesh_device) -> bool:
    """True only for the QB2 board (P150x4 / P300X2, 1x4 Blackhole)."""
    if mesh_device is None:
        return False
    try:
        from models.tt_transformers.tt.model_config import determine_device_name

        return determine_device_name(mesh_device) == _QB2_DEVICE_NAME
    except Exception:
        return False


def resolve_gemma4_prefill_chunk_size(max_seq_len: int, mesh_device=None, non_qb2_default=None) -> int:
    """Generator-level prefill chunk size for the vLLM serving generator.

    (The demo generator forces a single chunk instead — Gemma4's multi-chunk
    prefill is not validated for output correctness at long context, see
    models/demos/gemma4/tt/generator.py. vLLM must chunk anyway to dodge the
    #49083 serving-path wedge.)

    ``GEMMA4_GEN_PREFILL_CHUNK`` (a 2048-multiple) overrides on any board.
    Otherwise the bounded QB2 default (``GEMMA4_DEFAULT_PREFILL_CHUNK``, clamped
    to ``max_seq_len``) applies ONLY on QB2 (P150x4), where an 8192+ prompt then
    runs as 4096-token chunks instead of one full-length op. On every other board
    the QB2-tuned number is not applied — the caller's ``non_qb2_default`` (the
    prior vLLM default, ``max_seq_len``) is used so unvalidated boards keep their
    existing behavior.
    """
    override = int(os.environ.get("GEMMA4_GEN_PREFILL_CHUNK", "0"))
    if override > 0:
        return override
    if _is_qb2(mesh_device):
        return min(GEMMA4_DEFAULT_PREFILL_CHUNK, max_seq_len)
    return non_qb2_default if non_qb2_default is not None else max_seq_len


def model_uses_pli(model) -> bool:
    """True for E2B/E4B-style models with per-layer-input embeddings."""
    return bool(getattr(model, "hidden_size_per_layer_input", 0))


def can_gemma4_enable_prefill_trace(
    prefill_seq_len: int,
    *,
    batch_size: int = 1,
    num_cached_tokens: int = 0,
    uses_pli: bool = False,
) -> bool:
    """Return True when Gemma4 prefill device trace may be captured or replayed."""
    if uses_pli or num_cached_tokens != 0:
        return False
    if prefill_seq_len > GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN:
        return False
    if prefill_seq_len not in GEMMA4_TRACE_PREFILL_SEQ_LENS:
        return False
    if batch_size * prefill_seq_len >= GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS:
        return False
    return True


def apply_gemma4_prefill_trace_policy(
    enable_trace: bool,
    prefill_seq_len: int,
    batch_size: int,
    model,
) -> bool:
    """Apply Gemma4 prefill trace limits; log and return False when trace is disabled."""
    if not enable_trace:
        return False
    if can_gemma4_enable_prefill_trace(
        prefill_seq_len,
        batch_size=batch_size,
        uses_pli=model_uses_pli(model),
    ):
        return True
    if prefill_seq_len > GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN:
        logger.info(
            "Disabling prefill trace for seq_len={}: above {} ISL (no perf gain, OOM risk)",
            prefill_seq_len,
            GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
        )
    elif batch_size * prefill_seq_len >= GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS:
        logger.info(
            "Disabling prefill trace for batch_size={} seq_len={}: "
            "{}+ batched virtual tokens (no perf gain, OOM risk)",
            batch_size,
            prefill_seq_len,
            GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS,
        )
    else:
        logger.info(
            "Disabling prefill trace for batch_size={} seq_len={}: not eligible for capture",
            batch_size,
            prefill_seq_len,
        )
    return False


def resolve_gemma4_prefill_trace_enable(
    enable_trace: bool,
    model,
    model_args,
    *,
    batch_size: int,
    prefill_seq_lens: list[int],
    can_batch_prefill: bool,
) -> bool:
    """Resolve whether prefill trace stays enabled for this batch/prefill shape."""
    trace_batch_size = batch_size
    if can_batch_prefill:
        trace_batch_size = next(
            (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
            model_args.max_batch_size,
        )
    return apply_gemma4_prefill_trace_policy(
        enable_trace,
        prefill_seq_lens[0],
        trace_batch_size,
        model,
    )


def patch_gemma4_trace_model_args(model_args, *, prefill_trace_enabled: bool = True) -> None:
    """Configure trace_prefill_supported_seq_lens and can_enable_trace on model_args."""
    if prefill_trace_enabled:
        model_args.trace_prefill_supported_seq_lens = [
            length for length in GEMMA4_TRACE_PREFILL_SEQ_LENS if length <= GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN
        ]
        uses_pli = bool(getattr(model_args, "hidden_size_per_layer_input", 0))

        def _can_enable_trace(prefill_seq_len, num_cached_tokens=0, batch_size=1):
            return can_gemma4_enable_prefill_trace(
                prefill_seq_len,
                batch_size=batch_size,
                num_cached_tokens=num_cached_tokens,
                uses_pli=uses_pli,
            )

        model_args.can_enable_trace = _can_enable_trace
    else:
        model_args.trace_prefill_supported_seq_lens = []
        model_args.can_enable_trace = lambda prefill_seq_len, num_cached_tokens=0, batch_size=1: False


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


def skip_gemma4_full_prefill_warmup(generator) -> None:
    """Skip the full batch×ISL prefill warmup sweep on the next ``prefill_forward_text`` call."""
    generator.already_warmed_up_prefill = True


def warmup_gemma4_prefill_bucket(
    generator,
    kv_cache,
    *,
    enable_trace: bool,
    **prefill_kwargs,
) -> None:
    """Compile or capture prefill trace for one bucket only (no full warmup matrix).

    Used by parity/perf tests that exercise a single ``(batch_size, prefill_seq_len)``
    combination. Production/demo startup should still call
    :func:`warmup_gemma4_model_prefill` for the full sweep.
    """
    skip_gemma4_full_prefill_warmup(generator)
    generator.prefill_forward_text(
        **prefill_kwargs,
        kv_cache=kv_cache,
        enable_trace=maybe_disable_pli_prefill_trace(enable_trace, generator.model[0]),
        warmup_prefill=False,
    )


def warmup_gemma4_batched_prefill_traces(
    generator,
    kv_cache,
    *,
    enable_trace: bool,
    can_sample_on_device,
    greedy_only: bool = False,
    prefill_forward_fn=None,
) -> None:
    """Capture prefill traces for MoE models across batch sizes and trace ISLs.

    Sweeps ``SUPPORTED_PREFILL_BATCH_SIZES`` × ``trace_prefill_supported_seq_lens``,
    skipping combinations that meet or exceed ``MAX_BATCHED_PREFILL_SEQ_LEN`` (128k).
    Caller must set ``generator.already_warmed_up_prefill`` before calling if needed,
    or this function sets it on entry.

    ``prefill_forward_fn`` selects the entry point used for each capture. It
    defaults to ``generator.prefill_forward_text`` (demo / uniform-page-table
    path). The vLLM hybrid bridge passes ``generator.prefill_forward`` so the
    capture runs *through* the per-layer page-table routing and binds the
    traced paged ops to the persistent per-layer buffers — exactly how decode
    warmup binds via ``decode_forward``. Pre-capturing the prefill buckets at
    warmup (before any traced decode) keeps runtime prefills to trace *replay*,
    avoiding the #49083 cold-eager-capture fetch-queue wedge.
    """
    if generator.already_warmed_up_prefill:
        return
    generator.already_warmed_up_prefill = True

    prefill_forward = prefill_forward_fn if prefill_forward_fn is not None else generator.prefill_forward_text

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
                        greedy_only=greedy_only,
                        batch_size=batch_size,
                    )
                else:
                    sampling_params = [None]

                capture_trace = apply_gemma4_prefill_trace_policy(
                    enable_trace,
                    supported_length,
                    batch_size,
                    generator.model[model_id],
                )

                for param in sampling_params:
                    if capture_trace:
                        logger.info(
                            "Warming up prefill trace for sequence length: {} batch size: {} "
                            "with sampling params: {}",
                            supported_length,
                            batch_size,
                            param,
                        )
                    else:
                        logger.info(
                            "Warming up prefill (trace off) for sequence length: {} batch size: {} "
                            "with sampling params: {}",
                            supported_length,
                            batch_size,
                            param,
                        )
                    prefill_forward(
                        **warmup_args,
                        kv_cache=kv_cache,
                        enable_trace=capture_trace,
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
        prefill_forward(
            **prefill_forward_args,
            kv_cache=kv_cache,
            enable_trace=False,
            model_id_warmup=model_id,
            sampling_params=None,
            pixel_values=warmup_pixel_values,
            image_sizes=[(vision_chunk_size, vision_chunk_size)],
        )
        logger.info("Vision encoder warmup completed")


def warmup_gemma4_model_prefill(
    generator,
    kv_cache,
    *,
    enable_trace,
    can_sample_on_device,
    greedy_only: bool = False,
    prefill_forward_fn=None,
) -> None:
    """Shared prefill warmup for standalone and vLLM Gemma4 generators.

    ``prefill_forward_fn`` (vLLM hybrid bridge only) routes the trace capture
    through the per-layer page-table path; see
    :func:`warmup_gemma4_batched_prefill_traces`.
    """
    enable_trace = maybe_disable_pli_prefill_trace(enable_trace, generator.model[0])
    if enable_trace:
        warmup_gemma4_batched_prefill_traces(
            generator,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
            prefill_forward_fn=prefill_forward_fn,
        )
        return
    from models.tt_transformers.tt.generator import Generator

    Generator.warmup_model_prefill(
        generator,
        kv_cache=kv_cache,
        enable_trace=enable_trace,
        can_sample_on_device=can_sample_on_device,
        greedy_only=greedy_only,
    )
