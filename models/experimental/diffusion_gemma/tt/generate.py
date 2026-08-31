# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device generation helpers for DiffusionGemma (#47464).

This module owns the outer block-generation glue that is specific to
DiffusionGemma: prompt prefill, per-block denoise, committing each clean argmax
canvas to the frozen KV cache, and host-side decode helpers.
"""

from __future__ import annotations

import os
import time
from numbers import Integral
from typing import Callable, NamedTuple

from loguru import logger
import torch
import ttnn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt import degeneracy
from models.experimental.diffusion_gemma.tt.ccl import replicate_mapper as _replicate_mapper
from models.experimental.diffusion_gemma.tt.degeneracy import DegenerateBlockError
from models.experimental.diffusion_gemma.tt.commit_decode import commit_decode_forward
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    make_generation_logits_fn_builder_from_checkpoint_state,
)
from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block as tt_denoise_block
from models.experimental.diffusion_gemma.tt import sampling as TS


class GeneratedBlock(NamedTuple):
    committed: torch.Tensor
    next_pos: int
    trajectory: DenoiseTrajectory


class DeviceGeneration(NamedTuple):
    generated: torch.Tensor
    prompt_len: int
    next_pos: int
    trajectories: list[DenoiseTrajectory]


class DeviceTextGeneration(NamedTuple):
    prompt_tokens: torch.Tensor
    generation: DeviceGeneration
    sequences: torch.Tensor
    text: list[str]


class PromptPrefill(NamedTuple):
    prompt_len: int
    cache_len: int


def _deallocate_decode_inputs(device_inputs) -> None:
    for value in device_inputs:
        if value is not None and hasattr(value, "deallocate"):
            value.deallocate(True)


def _deallocate_if_possible(value) -> None:
    if value is None or not hasattr(value, "deallocate"):
        return
    try:
        value.deallocate(True)
    except Exception:
        # Best-effort cleanup while preserving the original device failure.
        pass


def _validate_nonnegative_integer_token_tensor(
    tokens: torch.Tensor,
    *,
    name: str,
    shape_name: str,
    allow_empty_length: bool = False,
) -> None:
    if tokens.dtype == torch.bool or torch.is_floating_point(tokens) or torch.is_complex(tokens):
        raise ValueError(f"{name} must contain integer token ids")
    if tokens.dim() != 2:
        raise ValueError(f"{name} must have shape {shape_name}")
    if tokens.shape[0] <= 0:
        raise ValueError(f"{name} batch size must be positive")
    if tokens.shape[1] <= 0 and not allow_empty_length:
        raise ValueError(f"{name} length must be positive")
    if torch.any(tokens < 0).item():
        raise ValueError(f"{name} must be non-negative")
    if torch.any(tokens > torch.iinfo(torch.int32).max).item():
        raise ValueError(f"{name} must fit int32 device token buffers")


def _validate_position_span(start_pos: int, length: int, *, name: str) -> None:
    if isinstance(start_pos, bool) or not isinstance(start_pos, Integral):
        raise ValueError(f"{name} must be an integer")
    if start_pos < 0:
        raise ValueError(f"{name} must be non-negative")
    if length <= 0:
        return
    if start_pos + length - 1 > torch.iinfo(torch.int32).max:
        raise ValueError(f"{name} span must fit int32 device positions")


def _validate_prefill_result(result, *, raw_prompt_len: int) -> PromptPrefill:
    if isinstance(result, PromptPrefill):
        prompt_len = result.prompt_len
        cache_len = result.cache_len
    else:
        prompt_len = result
        cache_len = result

    _validate_position_span(prompt_len, 1, name="prefill prompt_len")
    _validate_position_span(cache_len, 1, name="prefill cache_len")
    if prompt_len != raw_prompt_len:
        raise ValueError("prefill prompt_len must match prompt_tokens length")
    if cache_len < prompt_len:
        raise ValueError("prefill cache_len must cover prompt_len")
    return PromptPrefill(prompt_len=int(prompt_len), cache_len=int(cache_len))


def host_canvas_to_device(mesh_device, canvas_tokens: torch.Tensor):
    """Move host canvas token ids ``[batch, canvas_len]`` to W3 device layout."""
    _validate_nonnegative_integer_token_tensor(
        canvas_tokens,
        name="canvas_tokens",
        shape_name="[batch, canvas_len]",
    )
    batch, canvas_len = canvas_tokens.shape
    return ttnn.from_torch(
        canvas_tokens.view(batch, 1, canvas_len, 1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def host_gumbel_noise_to_device(mesh_device, gumbel_noise: torch.Tensor):
    """Move host Gumbel noise to denoise logits layout ``[batch, 1, canvas_len, vocab]``."""
    _validate_gumbel_noise(gumbel_noise)
    if gumbel_noise.dim() == 3:
        gumbel_noise = gumbel_noise.unsqueeze(1)
    return ttnn.from_torch(
        gumbel_noise.to(torch.float32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def host_tokens_to_device(mesh_device, tokens: torch.Tensor):
    """Move host token ids ``[batch, seq_len]`` to Gemma4 token layout."""
    _validate_nonnegative_integer_token_tensor(tokens, name="tokens", shape_name="[batch, seq_len]")
    return ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def _as_prompt_token_tensor(input_ids) -> torch.Tensor:
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    elif hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    elif hasattr(input_ids, "ids"):
        input_ids = input_ids.ids
    if isinstance(input_ids, torch.Tensor):
        tokens = input_ids
    else:
        tokens = torch.as_tensor(input_ids)
    if tokens.numel() > 0 and (
        tokens.dtype == torch.bool or torch.is_floating_point(tokens) or torch.is_complex(tokens)
    ):
        raise ValueError("prompt token ids must be integers")
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    _validate_nonnegative_integer_token_tensor(tokens, name="prompt token ids", shape_name="[batch, seq_len]")
    tokens = tokens.to(torch.long)
    return tokens


def _assert_chat_template_honors_thinking(tokenizer, messages, *, add_generation_prompt: bool) -> None:
    """Raise when the chat template silently drops ``enable_thinking``.

    ``apply_chat_template`` forwards unrecognised kwargs into the jinja context, so a template
    with no thinking branch renders the *same* string and the caller walks away holding a
    non-thinking prompt while believing it asked for thinking. An explicit
    ``enable_thinking=True`` must fail loudly rather than degrade (#48291).
    """
    render_kwargs = {"add_generation_prompt": add_generation_prompt, "tokenize": False}
    plain = tokenizer.apply_chat_template(messages, **render_kwargs)
    thinking = tokenizer.apply_chat_template(messages, enable_thinking=True, **render_kwargs)
    if plain == thinking:
        raise ValueError(
            "enable_thinking=True was requested but this tokenizer's chat template renders an "
            "identical prompt with and without it, i.e. the flag is being ignored. Use a "
            "DiffusionGemma tokenizer whose template emits the <|think|> turn, or drop the flag "
            "-- do not proceed, since the model would silently run the non-thinking contract."
        )


def tokenize_prompt(
    tokenizer,
    prompt,
    *,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> torch.Tensor:
    """Tokenize a prompt string or chat messages into host ids ``[batch, seq_len]``.

    ``enable_thinking`` selects DiffusionGemma's thinking contract, which the chat template
    renders as an extra ``<|think|>`` system turn. It defaults to ``None`` = pass nothing to
    the template; ``True``/``False`` are
    forwarded explicitly. An ignored ``True`` raises (see
    :func:`_assert_chat_template_honors_thinking`) instead of silently handing back the
    non-thinking prompt.
    """
    if isinstance(prompt, torch.Tensor):
        if enable_thinking is not None:
            raise ValueError(
                "enable_thinking cannot be applied to pre-tokenized prompt ids; pass the prompt "
                "as a string or chat messages so the chat template can render the thinking turn"
            )
        return _as_prompt_token_tensor(prompt)

    if hasattr(tokenizer, "apply_chat_template"):
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        template_kwargs = {}
        if enable_thinking is not None:
            if enable_thinking:
                _assert_chat_template_honors_thinking(tokenizer, messages, add_generation_prompt=add_generation_prompt)
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        return _as_prompt_token_tensor(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **template_kwargs,
            )
        )

    if not isinstance(prompt, str):
        raise ValueError("chat messages require a tokenizer with apply_chat_template")
    if enable_thinking is not None:
        raise ValueError(
            "enable_thinking requires a tokenizer with apply_chat_template; this tokenizer has "
            "no chat template, so the thinking turn cannot be rendered"
        )
    if callable(tokenizer):
        return _as_prompt_token_tensor(tokenizer(prompt, return_tensors="pt"))
    return _as_prompt_token_tensor(tokenizer.encode(prompt))


def _embed_tokens_dg(tt_model, tt_tokens):
    """``Gemma4Model.embed_tokens`` with DG's semaphore-passing all-gather.

    Byte-for-byte the shared op sequence (embedding -> scale -> unsqueeze -> TP
    all-gather); the only substitution is the gather, from the shared
    ``ccl_allgather``'s plain ``ttnn.all_gather`` to DG's ``ccl_allgather``, which
    passes pre-created semaphores and so selects ``all_gather_async``'s
    ``MINIMAL_DEFAULT`` factory instead of a factory that calls
    ``create_global_semaphore`` three times. A gather moves data without
    arithmetic, so this is a structural identity, not an accuracy trade.

    Why it matters: with the prefill lm_head skipped (``tt/prefill_logits.py``)
    this would be DG's last semaphore-creating collective on the non-traced path,
    semaphore setup blocks in ``finish_nolock``, and ``embed_host_tokens`` is
    called by BOTH prefill and per-block commit.
    """
    if tt_model.embedding_weight is None:
        raise RuntimeError("Embedding weights not loaded")
    embeds = ttnn.embedding(tt_tokens, tt_model.embedding_weight, dtype=ttnn.bfloat16)
    embeds = ttnn.mul(embeds, tt_model.embed_scale)
    if tt_model.mesh_config is not None and tt_model.mesh_config.tp > 1:
        from models.experimental.diffusion_gemma.tt.ccl import ccl_allgather

        embeds = ttnn.unsqueeze_to_4D(embeds)
        # all_gather_async silently falls back to composite_all_gather for
        # ROW_MAJOR inputs.  That composite is implemented with all_broadcast,
        # ignores our pre-created semaphores, and can deadlock once persistent
        # denoise traces exist.  Hidden size is tile-aligned, so tilizing first
        # keeps this on the semaphore-passing MINIMAL_DEFAULT implementation.
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
        embeds = ccl_allgather(embeds, tt_model.mesh_config, tt_model.ccl_manager)
    return embeds


def embed_host_tokens(tt_model, tokens: torch.Tensor):
    """Embed host token ids as ``[1, 1, seq_len, hidden]`` TILE states."""
    tt_tokens = host_tokens_to_device(tt_model.mesh_device, tokens)
    embeds = _embed_tokens_dg(tt_model, tt_tokens)
    tt_tokens.deallocate(True)
    embeds = ttnn.reshape(embeds, (1, 1, tokens.shape[1], tt_model.hidden_size))
    return ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)


def _pad_prompt_tokens_for_prefill(prompt_tokens: torch.Tensor, *, multiple: int = 32) -> torch.Tensor:
    pad = (-prompt_tokens.shape[1]) % multiple
    if pad == 0:
        return prompt_tokens
    padding = torch.zeros((prompt_tokens.shape[0], pad), dtype=prompt_tokens.dtype, device=prompt_tokens.device)
    return torch.cat([prompt_tokens, padding], dim=1)


def fixed_prefill_chunks_enabled() -> bool:
    """Whether every prompt uses the one fixed-size bounded prefill program."""
    return os.environ.get("DG_PREFILL_FIXED_CHUNKS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def prefill_prompt_tokens(
    tt_model,
    prompt_tokens: torch.Tensor,
    *,
    page_table=None,
    page_tables_per_layer=None,
    execution_len: int | None = None,
) -> PromptPrefill:
    """Write prompt token K/V into the frozen cache.

    The Gemma4 prefill path pads device tokens to tile multiples before writing
    K/V. Denoise must read that same aligned frozen-prefix span, while host text
    I/O keeps the natural prompt length.

    ``execution_len`` optionally pads only the *compute* shape farther (for
    coarse compile buckets). The returned ``cache_len`` remains the nearest
    tile-aligned logical prefix. Causal prefill means the right-padding tail
    cannot affect any real prompt token; denoise hides everything after
    ``cache_len`` as uncommitted, and bounded sliding layers use
    ``get_last_token`` to avoid writing the bucket tail into their circular
    cache. Thus compile-shape bucketing does not move the canvas start position.
    """
    _validate_prompt_tokens(prompt_tokens)
    if prompt_tokens.shape[0] != 1:
        raise NotImplementedError("prefill_prompt_tokens currently supports batch=1")
    prompt_len = prompt_tokens.shape[1]
    cache_tokens = _pad_prompt_tokens_for_prefill(prompt_tokens)
    cache_len = cache_tokens.shape[1]
    chunk_size = int(os.environ.get("DG_PREFILL_CHUNK_SIZE", "4096"))
    # A 32K monolithic prefill materializes a ~352 MiB all-reduce temporary
    # that cannot fit beside a resident 32K denoise trace, so the 32K boundary
    # itself must stay in the same bounded-memory regime as larger prefills.
    use_fixed_chunks = fixed_prefill_chunks_enabled()
    use_bounded_chunks = use_fixed_chunks or (execution_len is not None and int(execution_len) >= 32768)
    if execution_len is None:
        prefill_tokens = (
            _pad_prompt_tokens_for_prefill(prompt_tokens, multiple=chunk_size) if use_fixed_chunks else cache_tokens
        )
    else:
        execution_len = int(execution_len)
        if execution_len <= 0 or execution_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"prefill execution_len must be a positive {ttnn.TILE_SIZE}-token multiple")
        if execution_len < cache_len:
            raise ValueError(f"prefill execution_len {execution_len} cannot cover cache_len {cache_len}")
        padding_multiple = chunk_size if use_bounded_chunks else execution_len
        prefill_tokens = _pad_prompt_tokens_for_prefill(prompt_tokens, multiple=padding_multiple)
        if not use_bounded_chunks and int(prefill_tokens.shape[1]) != execution_len:
            raise RuntimeError(
                f"prefill execution padding produced length {prefill_tokens.shape[1]}, expected {execution_len}"
            )
    if use_bounded_chunks:
        if chunk_size <= 0 or chunk_size % 128 != 0:
            raise ValueError("DG_PREFILL_CHUNK_SIZE must be a positive 128-token multiple")
        if not getattr(tt_model, "_dg_model_owned_hybrid_kv", False):
            raise RuntimeError("prefill buckets above 32K require DG model-owned hybrid KV")
        host_page_tables = getattr(tt_model, "_dg_hybrid_host_page_tables_per_layer", None)
        device_page_tables = page_tables_per_layer or getattr(tt_model, "_dg_hybrid_page_tables_per_layer", None)
        if host_page_tables is None or device_page_tables is None:
            raise RuntimeError("bounded chunked prefill requires attached hybrid page tables")

        from models.experimental.diffusion_gemma.tt.chunked_prefill import chunked_prefill

        discarded = chunked_prefill(
            tt_model,
            input_ids_torch=prefill_tokens,
            embeds_torch=None,
            kv_cache=tt_model.tt_kv_cache,
            page_tables_torch_per_layer=host_page_tables,
            page_tables_per_layer=device_page_tables,
            embed_chunk_fn=embed_host_tokens,
            block_size=int(tt_model._dg_hybrid_block_size),
            chunk_size=chunk_size,
            return_last_logits=False,
            valid_prompt_len=prompt_len,
        )
        if discarded is not None:
            discarded.deallocate(True)
        return PromptPrefill(prompt_len=prompt_len, cache_len=cache_len)

    prompt_embeds = embed_host_tokens(tt_model, prefill_tokens)
    from models.experimental.diffusion_gemma.tt.prefill_logits import discard_prefill_logits
    from models.experimental.diffusion_gemma.tt.prefill_moe import use_tuned_prefill_moe

    # This forward is run for its K/V writes; whatever it returns is deallocated
    # unread two lines down. discard_prefill_logits stops the shared backbone
    # ending it in the lm_head, whose TP all-gather calls create_global_semaphore
    # and drains the command queue once per prefill -- see prefill_logits.py.
    # get_last_token is still passed: it is what the slice would use if the flag
    # were ever off, and the backbone reads it before this branch on other paths.
    with use_tuned_prefill_moe(tt_model), discard_prefill_logits(tt_model):
        discarded = tt_model(
            prompt_embeds,
            is_decode=False,
            input_ids_torch=prefill_tokens,
            get_last_token=((prompt_len - 1) // 32) * 32,
            page_table=page_table,
            page_tables_per_layer=(
                page_tables_per_layer
                if page_tables_per_layer is not None
                else getattr(tt_model, "_dg_hybrid_page_tables_per_layer", None)
            ),
        )
    if discarded is not None:
        discarded.deallocate(True)
    return PromptPrefill(prompt_len=prompt_len, cache_len=cache_len)


def _resolve_generation_logits_fn(
    tt_model,
    logits_fn,
    logits_fn_builder,
    *,
    prompt_tokens: torch.Tensor,
    prompt_len: int,
    page_table=None,
    page_tables_per_layer=None,
):
    if logits_fn is not None:
        if logits_fn_builder is not None:
            raise ValueError("pass either logits_fn or logits_fn_builder, not both")
        return logits_fn
    if logits_fn_builder is None:
        raise ValueError("logits_fn or logits_fn_builder is required")
    return logits_fn_builder(
        tt_model,
        prompt_tokens=prompt_tokens,
        prompt_len=prompt_len,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )


def _validate_logits_fn_args(logits_fn, logits_fn_builder) -> None:
    if logits_fn is not None and logits_fn_builder is not None:
        raise ValueError("pass either logits_fn or logits_fn_builder, not both")


def _validate_replay_canvases(canvases, *, value_kind: str = "host_canvases") -> None:
    if not canvases:
        return
    if canvases[0].dim() != 2:
        raise ValueError("host_canvases must contain tensors with shape [batch, canvas_len]")
    expected_shape = tuple(canvases[0].shape)
    for canvas in canvases:
        if canvas.dim() != 2 or tuple(canvas.shape) != expected_shape:
            raise ValueError("host_canvases must all have shape [batch, canvas_len]")
        _validate_nonnegative_integer_token_tensor(canvas, name=value_kind, shape_name="[batch, canvas_len]")


def _validate_replay_canvas_blocks(blocks, *, kind: str) -> None:
    expected_shape = None
    for block in blocks:
        _validate_replay_canvases(block, value_kind=kind)
        if not block:
            continue
        block_shape = tuple(block[0].shape)
        if expected_shape is None:
            expected_shape = block_shape
        elif block_shape != expected_shape:
            raise ValueError(f"{kind} must all have shape [batch, canvas_len]")


def _check_replay_block_index(block_idx: int, num_blocks: int, *, kind: str) -> None:
    if isinstance(block_idx, bool) or not isinstance(block_idx, Integral):
        raise IndexError(f"{kind} block index must be an integer")
    if block_idx < 0 or block_idx >= num_blocks:
        raise IndexError(f"{kind} block index {block_idx} out of range for {num_blocks} blocks")


def _check_replay_step_index(step: int, num_steps: int, *, block_idx: int, kind: str) -> None:
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise IndexError(f"{kind} step index must be an integer")
    if step < 0 or step >= num_steps:
        raise IndexError(f"{kind} step index {step} out of range for block {block_idx} with {num_steps} steps")


def make_host_canvas_init_fn(mesh_device, host_canvases):
    """Create a ``generate_blocks`` init hook from fixed host canvas tensors."""
    canvases = [canvas.clone() for canvas in host_canvases]
    _validate_replay_canvases(canvases)

    def init_canvas_fn(block_idx: int, start_pos: int):
        del start_pos
        _check_replay_block_index(block_idx, len(canvases), kind="host canvas replay")
        return host_canvas_to_device(mesh_device, canvases[block_idx].clone())

    return init_canvas_fn


def _validate_gumbel_noise(noise: torch.Tensor) -> None:
    if noise.dim() == 3:
        if any(dim <= 0 for dim in noise.shape):
            raise ValueError("gumbel_noise dimensions must be positive")
        return
    if noise.dim() == 4 and noise.shape[1] == 1:
        if noise.shape[0] <= 0 or noise.shape[2] <= 0 or noise.shape[3] <= 0:
            raise ValueError("gumbel_noise dimensions must be positive")
        return
    raise ValueError("gumbel_noise must have shape [batch, canvas_len, vocab] or [batch, 1, canvas_len, vocab]")


def _logical_gumbel_noise_shape(noise: torch.Tensor) -> tuple[int, int, int]:
    _validate_gumbel_noise(noise)
    if noise.dim() == 3:
        return tuple(noise.shape)
    return (noise.shape[0], noise.shape[2], noise.shape[3])


def _validate_gumbel_noise_blocks(blocks) -> None:
    expected_shape = None
    for block in blocks:
        for noise in block:
            shape = _logical_gumbel_noise_shape(noise)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "host_gumbel_noise must all have shape [batch, canvas_len, vocab] "
                    "or [batch, 1, canvas_len, vocab]"
                )


def make_host_gumbel_noise_fn(mesh_device, host_gumbel_noise):
    """Create ``generate_blocks`` Gumbel noise hooks from fixed host noise tensors.

    Injection hook: replays a recorded torch run's noise for HF<->TT determinism
    checks; it is not a serving mode.
    """
    blocks = [[noise.clone() for noise in block] for block in host_gumbel_noise]
    _validate_gumbel_noise_blocks(blocks)

    def gumbel_noise_for_block(block_idx: int):
        _check_replay_block_index(block_idx, len(blocks), kind="host gumbel replay")

        def gumbel_noise_for_step(step: int):
            _check_replay_step_index(step, len(blocks[block_idx]), block_idx=block_idx, kind="host gumbel replay")
            return host_gumbel_noise_to_device(mesh_device, blocks[block_idx][step].clone())

        return gumbel_noise_for_step

    return gumbel_noise_for_block


def make_host_noise_tokens_fn(mesh_device, host_noise_tokens):
    """Create ``generate_blocks`` renoise hooks from fixed host token tensors.

    Injection hook: replays a recorded torch run's noise for HF<->TT determinism
    checks; it is not a serving mode.
    """
    blocks = [[tokens.clone() for tokens in block] for block in host_noise_tokens]
    _validate_replay_canvas_blocks(blocks, kind="host_noise_tokens")

    def noise_tokens_for_block(block_idx: int):
        _check_replay_block_index(block_idx, len(blocks), kind="host noise-token replay")

        def noise_tokens_for_step(step: int):
            _check_replay_step_index(
                step,
                len(blocks[block_idx]),
                block_idx=block_idx,
                kind="host noise-token replay",
            )
            return host_canvas_to_device(mesh_device, blocks[block_idx][step].clone())

        return noise_tokens_for_step

    return noise_tokens_for_block


def _check_random_token_args(batch: int, canvas_len: int, vocab_size: int) -> None:
    if isinstance(batch, bool) or not isinstance(batch, Integral):
        raise ValueError("batch must be an integer")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if isinstance(canvas_len, bool) or not isinstance(canvas_len, Integral):
        raise ValueError("canvas_len must be an integer")
    if canvas_len <= 0:
        raise ValueError("canvas_len must be positive")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral):
        raise ValueError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")


def _validate_host_rand_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("host random token seed must be an integer")
    if seed < 0:
        raise ValueError("host random token seed must be non-negative")
    return int(seed)


def make_seeded_host_canvas_init_fn(
    mesh_device,
    *,
    batch: int,
    canvas_len: int,
    vocab_size: int,
    seed: int,
):
    """Create a seeded random-token canvas init hook for runnable TT generation."""
    _check_random_token_args(batch, canvas_len, vocab_size)
    seed = _validate_host_rand_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def init_canvas_fn(block_idx: int, start_pos: int):
        del block_idx, start_pos
        canvas = torch.randint(0, vocab_size, (batch, canvas_len), dtype=torch.long, generator=generator)
        return host_canvas_to_device(mesh_device, canvas)

    return init_canvas_fn


def make_seeded_host_noise_tokens_fn(
    mesh_device,
    *,
    batch: int,
    canvas_len: int,
    vocab_size: int,
    seed: int,
):
    """Create seeded random-token renoise hooks for ``generate_blocks``.

    DiffusionGemma uses random token ids, not a mask token, for rejected canvas
    positions. Until a uint32 device randint path exists, this helper keeps the
    token-noise source explicit and reproducible while preserving the device
    denoise loop's token layout.
    """
    _check_random_token_args(batch, canvas_len, vocab_size)
    seed = _validate_host_rand_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def noise_tokens_for_block(block_idx: int):
        del block_idx

        def noise_tokens_for_step(step: int):
            del step
            tokens = torch.randint(0, vocab_size, (batch, canvas_len), dtype=torch.long, generator=generator)
            return host_canvas_to_device(mesh_device, tokens)

        return noise_tokens_for_step

    return noise_tokens_for_block


def make_seeded_gumbel_noise_fn(
    mesh_device,
    *,
    batch: int,
    canvas_len: int,
    vocab_size: int,
    seed: int,
):
    """Create deterministic device Gumbel noise hooks for ``generate_blocks``.

    Draws plain vocab-innermost device noise; the SFPU RNG is lane-salted per
    element (tt-metal#52024), so the plain draw is the trusted path.
    """
    _check_random_token_args(batch, canvas_len, vocab_size)
    seed = TS._validate_ttnn_rand_seed(seed)

    def gumbel_noise_for_block(block_idx: int, attempt: int = 0):
        # `attempt` offsets the seed so a retry after a degenerate block draws DIFFERENT noise.
        # Without it a retry would reproduce the trajectory exactly and be a silent no-op.
        block_seed = seed + block_idx * 1_000_003 + attempt * 7_919_003

        def gumbel_noise_for_step(step: int):
            # Vocab innermost, deliberately: the lane-salted SFPU RNG (tt-metal#52024)
            # makes the plain draw safe.
            return TS.sample_gumbel_noise(
                (batch, 1, canvas_len, vocab_size),
                device=mesh_device,
                seed=block_seed + step,
            )

        return gumbel_noise_for_step

    gumbel_noise_for_block.supports_retry = True
    return gumbel_noise_for_block


def commit_canvas_tokens(
    tt_model,
    canvas_tokens: torch.Tensor,
    *,
    start_pos: int,
    page_table=None,
    page_tables_per_layer=None,
) -> None:
    """Append committed canvas token ids to the model KV cache.

    Args:
        tt_model: Gemma4-backed DiffusionGemma model with ``tt_kv_cache``.
        canvas_tokens: Host token ids ``[batch, canvas_len]``. This matches the
            current W3 controller output, which returns the clean argmax canvas
            on host for trajectory comparison.
        start_pos: Absolute position of ``canvas_tokens[:, 0]``. For block ``N``
            this is ``prompt_len + N * canvas_len``.
        page_table: Optional shared page table for the decode append path.
        page_tables_per_layer: Optional hybrid per-layer page tables.
    """
    _validate_nonnegative_integer_token_tensor(
        canvas_tokens,
        name="canvas_tokens",
        shape_name="[batch, canvas_len]",
    )
    if canvas_tokens.shape[0] != 1:
        raise NotImplementedError("commit_canvas_tokens currently supports batch=1")
    _validate_position_span(start_pos, canvas_tokens.shape[1], name="start_pos")

    for offset in range(canvas_tokens.shape[1]):
        token = canvas_tokens[:, offset]
        position = torch.tensor([start_pos + offset], dtype=torch.int32)
        device_inputs = tt_model.prepare_inputs_decode(token, position, page_table=page_table)
        hidden, _ = commit_decode_forward(
            tt_model,
            device_inputs[0],
            device_inputs[1],
            device_inputs[2],
            device_inputs[3],
            page_tables_per_layer=page_tables_per_layer,
            skip_lm_head=True,
        )
        hidden.deallocate(True)
        _deallocate_decode_inputs(device_inputs)


def _set_q_rope_offset(logits_fn, q_rope_offset: int) -> None:
    if hasattr(logits_fn, "q_rope_offset"):
        logits_fn.q_rope_offset = q_rope_offset


def _contains_stop_token(tokens: torch.Tensor, stop_token_ids) -> bool:
    if stop_token_ids is None:
        return False
    ids = _normalize_eos_token_ids(stop_token_ids, kind="stop_token_ids")
    return any(token_id in ids for token_id in tokens.reshape(-1).tolist())


def _validate_num_blocks(num_blocks: int) -> None:
    if isinstance(num_blocks, bool) or not isinstance(num_blocks, Integral):
        raise ValueError("num_blocks must be an integer")
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")


def _validate_canvas_length(config: DiffusionConfig) -> None:
    if isinstance(config.canvas_length, bool) or not isinstance(config.canvas_length, Integral):
        raise ValueError("config.canvas_length must be an integer")
    if config.canvas_length <= 0:
        raise ValueError("config.canvas_length must be positive")


def _validate_max_new_tokens(max_new_tokens: int | None) -> int | None:
    if max_new_tokens is None:
        return None
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
        raise ValueError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    return int(max_new_tokens)


def _validate_max_new_tokens_capacity(num_blocks: int, canvas_length: int, max_new_tokens: int | None) -> int | None:
    max_new_tokens = _validate_max_new_tokens(max_new_tokens)
    if max_new_tokens is None:
        return None
    if max_new_tokens > num_blocks * canvas_length:
        raise ValueError("num_blocks is too small for max_new_tokens")
    return max_new_tokens


def _positive_int_attr(value) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, Integral):
        return None
    value = int(value)
    return value if value > 0 else None


def _infer_context_limit(tt_model) -> int | None:
    """Best-effort current context window for this TT model instance.

    Prefer ``tt_model.max_seq_len`` because that is the allocated cache/RoPE span
    for the current run. Fall back to HF config fields when tests or wrappers only
    expose the model's declared context length.
    """
    candidates = []
    for attr in ("max_seq_len", "max_context_len", "max_position_embeddings", "max_context"):
        candidates.append(getattr(tt_model, attr, None))

    hf_config = getattr(tt_model, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    for cfg in (hf_config, text_config):
        if cfg is None:
            continue
        for attr in ("max_seq_len", "max_context_len", "max_position_embeddings", "max_context"):
            candidates.append(getattr(cfg, attr, None))

    for value in candidates:
        limit = _positive_int_attr(value)
        if limit is not None:
            return limit
    return None


def _validate_generation_context_capacity(tt_model, prompt_len: int, num_blocks: int, config: DiffusionConfig) -> None:
    """Validate the physical committed-token span against the context window."""
    generated_len = num_blocks * config.canvas_length
    _validate_position_span(prompt_len, generated_len, name="prompt_len")
    if generated_len == 0:
        return
    limit = _infer_context_limit(tt_model)
    if limit is None:
        return
    end_pos = prompt_len + generated_len
    if end_pos > limit:
        raise ValueError(
            "prompt cache plus committed canvas blocks exceeds model context window: "
            f"{prompt_len} + {generated_len} = {end_pos} > {limit}"
        )


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _validate_vocab_size(vocab_size: int) -> int:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral):
        raise ValueError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return int(vocab_size)


def _validate_prompt_tokens(prompt_tokens: torch.Tensor) -> None:
    _validate_nonnegative_integer_token_tensor(prompt_tokens, name="prompt_tokens", shape_name="[batch, seq_len]")


def _validate_committed_block_shape(committed: torch.Tensor, *, batch_size: int, canvas_length: int) -> None:
    if committed.dim() != 2:
        raise ValueError("block.committed must have shape [batch, canvas_len]")
    if committed.shape != (batch_size, canvas_length):
        raise ValueError(f"block.committed must have shape [{batch_size}, {canvas_length}]")
    _validate_nonnegative_integer_token_tensor(
        committed,
        name="block.committed",
        shape_name="[batch, canvas_len]",
    )


def _empty_device_generation(batch_size: int, prompt_len: int, *, device=None) -> DeviceGeneration:
    return DeviceGeneration(
        generated=torch.zeros((batch_size, 0), dtype=torch.long, device=device),
        prompt_len=prompt_len,
        next_pos=prompt_len,
        trajectories=[],
    )


def _resolve_default_commit_fn(page_table=None, page_tables_per_layer=None) -> Callable[..., None]:
    """Pick the commit path: batched single-prefill for contiguous or DG hybrid KV.

    The default batched commit (``tt.commit_batched``) supports the contiguous
    cache and DG's identity-mapped model-owned per-layer hybrid cache. A
    legacy/shared page table can carry arbitrary vLLM block ownership, so it
    still uses the sequential path. Imported lazily so that path keeps no import
    dependency on ``tt.commit_batched`` (which imports helpers from this module).
    """
    from models.experimental.diffusion_gemma.tt.commit_batched import select_commit_fn

    if page_table is not None:
        return commit_canvas_tokens
    return select_commit_fn()


def denoise_and_commit_block(
    tt_model,
    logits_fn,
    init_canvas,
    config: DiffusionConfig,
    *,
    start_pos: int,
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
    page_table=None,
    page_tables_per_layer=None,
    denoise_block_fn: Callable[..., DenoiseTrajectory] | None = None,
    commit_fn: Callable[..., None] | None = None,
    timings: dict[str, float] | None = None,
    stop_token_ids=None,
    retry_noise_fn=None,
    retry_init_canvas_fn=None,
) -> GeneratedBlock:
    """Denoise one canvas, commit the clean argmax, and advance position.

    ``start_pos`` is the absolute canvas start for this block. When ``logits_fn``
    is a ``DenoiseLogitsAdapter`` this helper updates its ``q_rope_offset`` so
    canvas RoPE positions advance with each committed block.

    ``commit_fn`` defaults to the batched single-prefill commit for contiguous
    caches, falling back to the sequential :func:`commit_canvas_tokens` when a
    shared page table is supplied (see :func:`_resolve_default_commit_fn`).
    """
    if commit_fn is None:
        commit_fn = _resolve_default_commit_fn(page_table, page_tables_per_layer)
    if denoise_block_fn is None:
        denoise_block_fn = tt_denoise_block
    _set_q_rope_offset(logits_fn, start_pos)
    benign_ids = None if stop_token_ids is None else _normalize_eos_token_ids(stop_token_ids, kind="stop_token_ids")
    policy = degeneracy.resolve_policy()

    # `retry` needs to be able to draw DIFFERENT noise; a retry that reproduces the same trajectory
    # is a silent no-op, so refuse it up front rather than pretend to retry.
    max_attempts = 1
    if policy == "retry":
        if retry_noise_fn is None or retry_init_canvas_fn is None:
            raise ValueError(
                "DG_DEGENERACY_POLICY=retry needs retry_noise_fn and retry_init_canvas_fn; the "
                "caller owns both the noise factory and the initial canvas, and the denoise path "
                "consumes the canvas it is given"
            )
        if not getattr(retry_noise_fn, "supports_retry", False):
            raise ValueError(
                "DG_DEGENERACY_POLICY=retry requires an attempt-aware noise factory (one marked "
                "supports_retry). The per-step noise functions are pure in (block, step), so "
                "retrying with this factory would redraw identical noise and change nothing. "
                "Deterministic sources (argmax) cannot support retry by construction."
            )
        max_attempts = 1 + degeneracy.resolve_retries()

    denoise_s = 0.0
    attempt = 0
    while True:
        canvas_for_attempt = init_canvas if attempt == 0 else retry_init_canvas_fn()
        noise_for_attempt = gumbel_noise_fn if attempt == 0 else retry_noise_fn(attempt)
        denoise_t0 = time.perf_counter()
        trajectory = denoise_block_fn(
            logits_fn,
            canvas_for_attempt,
            config,
            gumbel_noise_fn=noise_for_attempt,
            noise_tokens_fn=noise_tokens_fn,
        )
        denoise_s += time.perf_counter() - denoise_t0
        if trajectory.committed is None:
            raise RuntimeError("denoise trajectory did not produce committed canvas tokens")
        _validate_committed_block_shape(trajectory.committed, batch_size=1, canvas_length=config.canvas_length)

        if policy == "off":
            degeneracy_stats = {}
            break
        # Before the commit, never after: a degenerate canvas that reaches the KV cache conditions
        # every later block, which is what makes the degenerate state near-absorbing.
        degeneracy_stats, is_bad = degeneracy.evaluate(trajectory.committed, stop_token_ids=benign_ids)
        logger.info(
            "DG_DEGENERACY start_pos={} attempt={} distinct={}/{} top_id={} top_frac={:.4f} max_run={} "
            "stop_tail={} content_tokens={} content_top_id={} content_top_frac={} content_max_run={} "
            "verdict={}".format(
                start_pos,
                attempt,
                degeneracy_stats["distinct"],
                degeneracy_stats["tokens"],
                degeneracy_stats["top_id"],
                degeneracy_stats["top_frac"],
                degeneracy_stats["max_run"],
                # The content region is the decision basis; without it in the telemetry a later
                # triage cannot tell a rejected collapse from a rejected normal completion.
                degeneracy_stats.get("stop_tail", "na"),
                degeneracy_stats.get("content_tokens", "na"),
                degeneracy_stats.get("content_top_id", "na"),
                degeneracy_stats.get("content_top_frac", "na"),
                degeneracy_stats.get("content_max_run", "na"),
                "degenerate" if is_bad else "ok",
            )
        )
        if not is_bad:
            break
        message = degeneracy.describe(degeneracy_stats)
        if policy == "warn":
            logger.warning(message)
            break
        attempt += 1
        if attempt >= max_attempts:
            # `stop`, or `retry` with every attempt exhausted: never commit it.
            raise degeneracy.DegenerateBlockError(message, tokens=trajectory.committed, stats=degeneracy_stats)
        logger.warning(f"{message}; re-denoising with fresh noise (attempt {attempt}/{max_attempts - 1})")

    # Same probe as the prefill one (tt/serving.py): commit is the other hot NON-traced
    # dispatch; the probe splits its wall time into this thread's CPU vs device wait.
    commit_probe = os.environ.get("DG_PREFILL_CPU_PROBE", "0") == "1"
    commit_cpu0 = (time.process_time(), time.thread_time()) if commit_probe else None
    commit_t0 = time.perf_counter()
    commit_fn(
        tt_model,
        trajectory.committed,
        start_pos=start_pos,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    commit_s = time.perf_counter() - commit_t0
    if commit_cpu0 is not None:
        commit_proc = time.process_time() - commit_cpu0[0]
        commit_thread = time.thread_time() - commit_cpu0[1]
        logger.info(
            f"DG_COMMIT_CPU start_pos={start_pos} wall_s={commit_s:.4f} "
            f"process_cpu_s={commit_proc:.4f} thread_cpu_s={commit_thread:.4f} "
            f"thread_cpu_frac={commit_thread / max(commit_s, 1e-9):.3f}"
        )
    if timings is not None:
        timings.update(denoise_s=denoise_s, commit_s=commit_s)
    next_pos = start_pos + trajectory.committed.shape[1]
    advance_prefix = getattr(logits_fn, "advance_prefix_after_commit", None)
    if callable(advance_prefix):
        advance_prefix(next_pos)
    return GeneratedBlock(
        committed=trajectory.committed,
        next_pos=next_pos,
        trajectory=trajectory,
    )


def _release_generation_logits_state(logits_fn) -> None:
    """Best-effort teardown for direct generator-owned adapter state."""
    attr = "_upfront_traced_denoise_controller"
    controller = getattr(logits_fn, attr, None)
    if controller is not None:
        try:
            controller.release()
        except BaseException as cleanup_error:
            logger.error(f"failed to release generator controller {attr}: {cleanup_error}")
        finally:
            try:
                delattr(logits_fn, attr)
            except AttributeError:
                # A test double or proxy may expose the controller via its class;
                # shadow it so this request cannot reuse the released state.
                try:
                    setattr(logits_fn, attr, None)
                except BaseException as cleanup_error:
                    logger.error(f"failed to clear generator controller {attr}: {cleanup_error}")
    reset = getattr(logits_fn, "reset", None)
    if callable(reset):
        try:
            reset()
        except BaseException as cleanup_error:
            logger.error(f"failed to reset generator logits state: {cleanup_error}")


def generate_blocks(
    tt_model,
    logits_fn,
    *,
    prompt_len: int,
    num_blocks: int,
    config: DiffusionConfig,
    init_canvas_fn: Callable[[int, int], object] | None = None,
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
    page_table=None,
    page_tables_per_layer=None,
    stop_token_ids=None,
    batch_size: int = 1,
    block_fn: Callable[..., GeneratedBlock] = denoise_and_commit_block,
) -> DeviceGeneration:
    """Run the minimal device outer loop for ``num_blocks`` canvases.

    ``init_canvas_fn(block_idx, start_pos)`` supplies the initial device canvas
    for each block. Use ``make_host_canvas_init_fn`` when replaying fixed torch /
    HF canvases. This helper owns commit-append and absolute position advancement.
    """
    _validate_num_blocks(num_blocks)
    _validate_canvas_length(config)
    _validate_batch_size(batch_size)
    _validate_generation_context_capacity(tt_model, prompt_len, num_blocks, config)
    init_canvas_fn = _resolve_init_canvas_fn(num_blocks, init_canvas_fn)
    next_pos = prompt_len
    committed_blocks: list[torch.Tensor] = []
    trajectories: list[DenoiseTrajectory] = []

    try:
        for block_idx in range(num_blocks):
            logger.info(f"[generate_blocks] block {block_idx + 1}/{num_blocks} start_pos={next_pos}")
            init_canvas = init_canvas_fn(block_idx, next_pos)
            try:
                block = block_fn(
                    tt_model,
                    logits_fn,
                    init_canvas,
                    config,
                    start_pos=next_pos,
                    gumbel_noise_fn=gumbel_noise_fn(block_idx) if gumbel_noise_fn else None,
                    noise_tokens_fn=noise_tokens_fn(block_idx) if noise_tokens_fn else None,
                    page_table=page_table,
                    page_tables_per_layer=page_tables_per_layer,
                    stop_token_ids=stop_token_ids,
                    retry_noise_fn=gumbel_noise_fn,
                    retry_init_canvas_fn=(
                        (lambda idx=block_idx, pos=next_pos: init_canvas_fn(idx, pos))
                        if init_canvas_fn is not None
                        else None
                    ),
                )
            except DegenerateBlockError as degenerate:
                # Uncommitted by construction, so the generation simply ends one block early with
                # everything that was healthy. Emitting the canvas would be the degenerate output.
                _deallocate_if_possible(init_canvas)
                logger.warning(f"[generate_blocks] stopping at block {block_idx + 1}: {degenerate}")
                break
            except Exception:
                _deallocate_if_possible(init_canvas)
                raise
            _validate_committed_block_shape(block.committed, batch_size=batch_size, canvas_length=config.canvas_length)
            expected_next_pos = next_pos + block.committed.shape[1]
            if block.next_pos != expected_next_pos:
                raise ValueError("block.next_pos must equal start_pos + committed length")
            committed_blocks.append(block.committed)
            trajectories.append(block.trajectory)
            next_pos = block.next_pos
            if _contains_stop_token(block.committed, stop_token_ids):
                logger.info(f"[generate_blocks] stop token in block {block_idx + 1}; halting at next_pos={next_pos}")
                break

        generated = (
            torch.cat(committed_blocks, dim=1) if committed_blocks else torch.zeros((batch_size, 0), dtype=torch.long)
        )
        return DeviceGeneration(
            generated=generated, prompt_len=prompt_len, next_pos=next_pos, trajectories=trajectories
        )
    finally:
        _release_generation_logits_state(logits_fn)


def generate_from_prompt_tokens(
    tt_model,
    logits_fn,
    prompt_tokens: torch.Tensor,
    *,
    num_blocks: int,
    config: DiffusionConfig,
    init_canvas_fn: Callable[[int, int], object] | None = None,
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
    page_table=None,
    page_tables_per_layer=None,
    stop_token_ids=None,
    logits_fn_builder=None,
    prefill_fn: Callable[..., int] = prefill_prompt_tokens,
    blocks_fn: Callable[..., DeviceGeneration] = generate_blocks,
) -> DeviceGeneration:
    """Prefill prompt K/V, then generate and commit ``num_blocks`` canvases.

    This host-token entrypoint is intentionally below tokenizer/detokenizer level:
    callers still own chat templating, production canvas RNG, and EOS policy. It
    is the reusable bridge needed by device-vs-HF replay tests, where prompt ids
    and per-step noise are injected exactly.
    """
    _validate_num_blocks(num_blocks)
    _validate_canvas_length(config)
    _validate_logits_fn_args(logits_fn, logits_fn_builder)
    _validate_prompt_tokens(prompt_tokens)
    init_canvas_fn = _resolve_init_canvas_fn(num_blocks, init_canvas_fn)
    if num_blocks == 0:
        return _empty_device_generation(
            prompt_tokens.shape[0],
            prompt_tokens.shape[1],
            device=prompt_tokens.device,
        )
    prompt_len = prefill_fn(
        tt_model,
        prompt_tokens,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    prefill = _validate_prefill_result(prompt_len, raw_prompt_len=prompt_tokens.shape[1])
    _validate_generation_context_capacity(tt_model, prefill.cache_len, num_blocks, config)
    logits_fn = _resolve_generation_logits_fn(
        tt_model,
        logits_fn,
        logits_fn_builder,
        prompt_tokens=prompt_tokens,
        prompt_len=prefill.cache_len,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    return blocks_fn(
        tt_model,
        logits_fn,
        prompt_len=prefill.cache_len,
        num_blocks=num_blocks,
        config=config,
        init_canvas_fn=init_canvas_fn,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
        stop_token_ids=stop_token_ids,
        batch_size=prompt_tokens.shape[0],
    )


def generation_sequences(prompt_tokens: torch.Tensor, generation: DeviceGeneration) -> torch.Tensor:
    """Return HF-style ``[prompt, generated]`` token sequences for decode/e2e checks."""
    _validate_prompt_tokens(prompt_tokens)
    _validate_nonnegative_integer_token_tensor(
        generation.generated,
        name="generation.generated",
        shape_name="[batch, seq_len]",
        allow_empty_length=True,
    )
    _validate_position_span(generation.prompt_len, 1, name="generation.prompt_len")
    _validate_position_span(generation.next_pos, 1, name="generation.next_pos")
    if prompt_tokens.shape[0] != generation.generated.shape[0]:
        raise ValueError("prompt_tokens and generation.generated batch sizes must match")
    if prompt_tokens.shape[1] > generation.prompt_len:
        raise ValueError("prompt_tokens length must not exceed generation.prompt_len")
    expected_next_pos = generation.prompt_len + generation.generated.shape[1]
    if generation.next_pos != expected_next_pos:
        raise ValueError("generation.next_pos must equal generation.prompt_len + generated length")
    return torch.cat([prompt_tokens, generation.generated], dim=1)


def _normalize_eos_token_ids(eos_token_id, *, kind: str = "eos_token_id"):
    if eos_token_id is None:
        return None
    message = f"{kind} must be an int or iterable of ints"

    def validate_token_id(token_id) -> int:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise ValueError(message)
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError(f"{kind} must be non-negative")
        if token_id > torch.iinfo(torch.int32).max:
            raise ValueError(f"{kind} must fit int32 token ids")
        return token_id

    if isinstance(eos_token_id, Integral):
        return {validate_token_id(eos_token_id)}

    if isinstance(eos_token_id, (str, bytes)):
        raise ValueError(message)

    try:
        ids = list(eos_token_id)
    except TypeError as exc:
        raise ValueError(message) from exc

    return {validate_token_id(token_id) for token_id in ids}


def _trim_generated_token_ids(generated: torch.Tensor, *, max_new_tokens: int | None = None, eos_token_id=None):
    max_new_tokens = _validate_max_new_tokens(max_new_tokens)

    eos_ids = _normalize_eos_token_ids(eos_token_id)
    rows = []
    for row in generated.tolist():
        if max_new_tokens is not None:
            row = row[:max_new_tokens]
        if eos_ids:
            for idx, token_id in enumerate(row):
                if token_id in eos_ids:
                    row = row[: idx + 1]
                    break
        rows.append(row)
    return rows


def generation_token_ids(
    prompt_tokens: torch.Tensor,
    generation: DeviceGeneration,
    *,
    skip_prompt: bool = True,
    max_new_tokens: int | None = None,
    eos_token_id=None,
):
    """Return decoded token-id rows, applying output-only length/EOS stops."""
    generation_sequences(prompt_tokens, generation)  # validates shape and prompt length.
    generated_rows = _trim_generated_token_ids(
        generation.generated,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    if skip_prompt:
        return generated_rows
    return [prompt_row + generated_row for prompt_row, generated_row in zip(prompt_tokens.tolist(), generated_rows)]


def decode_generation(
    tokenizer,
    prompt_tokens: torch.Tensor,
    generation: DeviceGeneration,
    *,
    skip_prompt: bool = True,
    max_new_tokens: int | None = None,
    eos_token_id=None,
    **decode_kwargs,
):
    """Decode generated token ids with a HuggingFace-style tokenizer.

    ``skip_prompt=True`` mirrors the common text-generation UX: return only the
    generated continuation. Set it to ``False`` for HF-style full sequences.
    ``max_new_tokens`` and ``eos_token_id`` trim only the host-visible output;
    the device KV cache may already contain a full committed canvas block.
    """
    token_ids = generation_token_ids(
        prompt_tokens,
        generation,
        skip_prompt=skip_prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(token_ids, **decode_kwargs)
    return [tokenizer.decode(ids, **decode_kwargs) for ids in token_ids]


def _infer_generation_vocab_size(tokenizer, tt_model=None):
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return vocab_size
    try:
        return len(tokenizer)
    except TypeError:
        pass
    vocab_size = getattr(tt_model, "vocab_size", None)
    if vocab_size is not None:
        return vocab_size
    hf_config = getattr(tt_model, "hf_config", None)
    return getattr(hf_config, "vocab_size", None)


def _unused_canvas_init_fn(block_idx: int, start_pos: int):
    del block_idx, start_pos
    raise RuntimeError("init_canvas_fn should not be called when num_blocks is zero")


def _resolve_init_canvas_fn(num_blocks: int, init_canvas_fn):
    if init_canvas_fn is not None:
        return init_canvas_fn
    if num_blocks == 0:
        return _unused_canvas_init_fn
    raise ValueError("init_canvas_fn is required unless num_blocks is zero")


def generate_text(
    tt_model,
    logits_fn,
    tokenizer,
    prompt,
    *,
    num_blocks: int,
    config: DiffusionConfig,
    init_canvas_fn: Callable[[int, int], object] | None = None,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
    page_table=None,
    page_tables_per_layer=None,
    max_new_tokens: int | None = None,
    eos_token_id=None,
    stop_token_ids=None,
    skip_prompt: bool = True,
    decode_kwargs: dict | None = None,
    logits_fn_builder=None,
    prefill_fn: Callable[..., int] = prefill_prompt_tokens,
    blocks_fn: Callable[..., DeviceGeneration] = generate_blocks,
) -> DeviceTextGeneration:
    """Tokenize a prompt, run device generation, and decode host-visible text."""
    _validate_num_blocks(num_blocks)
    _validate_canvas_length(config)
    max_new_tokens = _validate_max_new_tokens_capacity(num_blocks, config.canvas_length, max_new_tokens)
    init_canvas_fn = _resolve_init_canvas_fn(num_blocks, init_canvas_fn)
    prompt_tokens = tokenize_prompt(
        tokenizer,
        prompt,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
    )
    generation = generate_from_prompt_tokens(
        tt_model,
        logits_fn,
        prompt_tokens,
        num_blocks=num_blocks,
        config=config,
        init_canvas_fn=init_canvas_fn,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
        stop_token_ids=stop_token_ids if stop_token_ids is not None else eos_token_id,
        logits_fn_builder=logits_fn_builder,
        prefill_fn=prefill_fn,
        blocks_fn=blocks_fn,
    )
    text = decode_generation(
        tokenizer,
        prompt_tokens,
        generation,
        skip_prompt=skip_prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        **(decode_kwargs or {}),
    )
    return DeviceTextGeneration(
        prompt_tokens=prompt_tokens,
        generation=generation,
        sequences=generation_sequences(prompt_tokens, generation),
        text=text,
    )


def generate_text_from_checkpoint_state(
    tt_model,
    tokenizer,
    prompt,
    *,
    dg_state_dict,
    num_blocks: int | None = None,
    config: DiffusionConfig | None = None,
    init_canvas_fn: Callable[[int, int], object] | None = None,
    vocab_size: int | None = None,
    seed: int | None = None,
    gumbel_seed: int | None = None,
    noise_seed: int | None = None,
    batch: int = 1,
    adapter_kwargs: dict | None = None,
    logits_fn_builder_factory=make_generation_logits_fn_builder_from_checkpoint_state,
    generate_text_fn=generate_text,
    **generate_kwargs,
) -> DeviceTextGeneration:
    """Run prompt-to-text generation using raw DiffusionGemma checkpoint state."""
    config = DiffusionConfig() if config is None else config
    _validate_canvas_length(config)
    max_new_tokens = generate_kwargs.get("max_new_tokens")
    if num_blocks is None:
        if max_new_tokens is None:
            raise ValueError("num_blocks is required unless max_new_tokens is provided")
        max_new_tokens = _validate_max_new_tokens(max_new_tokens)
        generate_kwargs["max_new_tokens"] = max_new_tokens
        num_blocks = (max_new_tokens + config.canvas_length - 1) // config.canvas_length
    _validate_num_blocks(num_blocks)
    max_new_tokens = _validate_max_new_tokens_capacity(num_blocks, config.canvas_length, max_new_tokens)
    if "max_new_tokens" in generate_kwargs:
        generate_kwargs["max_new_tokens"] = max_new_tokens
    if num_blocks > 0:
        _validate_batch_size(batch)
        if seed is not None:
            seed = _validate_host_rand_seed(seed)
    if vocab_size is None and num_blocks > 0:
        vocab_size = _infer_generation_vocab_size(tokenizer, tt_model)
    if vocab_size is not None:
        vocab_size = _validate_vocab_size(vocab_size)
    if init_canvas_fn is None and num_blocks == 0:
        init_canvas_fn = _unused_canvas_init_fn
    if init_canvas_fn is None:
        if vocab_size is None or seed is None:
            raise ValueError("init_canvas_fn is required unless vocab_size and seed are provided")
        init_canvas_fn = make_seeded_host_canvas_init_fn(
            tt_model.mesh_device,
            batch=batch,
            canvas_len=config.canvas_length,
            vocab_size=vocab_size,
            seed=seed,
        )
    init_canvas_fn = _resolve_init_canvas_fn(num_blocks, init_canvas_fn)
    if num_blocks > 0 and "noise_tokens_fn" not in generate_kwargs and (seed is not None or noise_seed is not None):
        if vocab_size is None:
            raise ValueError("noise_tokens_fn requires vocab_size or tokenizer/model vocab metadata")
        generate_kwargs["noise_tokens_fn"] = make_seeded_host_noise_tokens_fn(
            tt_model.mesh_device,
            batch=batch,
            canvas_len=config.canvas_length,
            vocab_size=vocab_size,
            seed=noise_seed if noise_seed is not None else seed + 1,
        )
    if num_blocks > 0 and "gumbel_noise_fn" not in generate_kwargs and (seed is not None or gumbel_seed is not None):
        if vocab_size is None:
            raise ValueError("gumbel_noise_fn requires vocab_size or tokenizer/model vocab metadata")
        gumbel_seed_value = gumbel_seed if gumbel_seed is not None else seed + 2
        gumbel_seed_value = TS._validate_ttnn_rand_seed(gumbel_seed_value)
        generate_kwargs["gumbel_noise_fn"] = make_seeded_gumbel_noise_fn(
            tt_model.mesh_device,
            batch=batch,
            canvas_len=config.canvas_length,
            vocab_size=vocab_size,
            seed=gumbel_seed_value,
        )
    if "eos_token_id" in generate_kwargs:
        _normalize_eos_token_ids(generate_kwargs["eos_token_id"])
    else:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            _normalize_eos_token_ids(eos_token_id)
            generate_kwargs["eos_token_id"] = eos_token_id
    if "decode_kwargs" not in generate_kwargs:
        generate_kwargs["decode_kwargs"] = {"skip_special_tokens": True}
    adapter_kwargs = dict(adapter_kwargs or {})
    adapter_kwargs.setdefault("max_denoise_steps", config.max_denoise_steps)
    adapter_kwargs.setdefault("temperature_start", config.temperature_start)
    adapter_kwargs.setdefault("temperature_end", config.temperature_end)
    if "config" not in adapter_kwargs:
        adapter_config = getattr(tt_model, "hf_config", None)
        if adapter_config is not None:
            adapter_kwargs["config"] = adapter_config
    logits_fn_builder = None
    if num_blocks > 0:
        logits_fn_builder = logits_fn_builder_factory(
            dg_state_dict,
            **adapter_kwargs,
        )
    return generate_text_fn(
        tt_model,
        None,
        tokenizer,
        prompt,
        num_blocks=num_blocks,
        config=config,
        init_canvas_fn=init_canvas_fn,
        logits_fn_builder=logits_fn_builder,
        **generate_kwargs,
    )
