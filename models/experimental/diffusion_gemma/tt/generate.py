# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device generation helpers for DiffusionGemma (#47464).

This module owns the outer block-generation glue that is specific to
DiffusionGemma. It starts with the commit step: once the denoise controller has
chosen the clean argmax canvas, append those tokens to the frozen KV cache using
Gemma4's decode path in ``COMMIT_APPEND`` phase.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
import ttnn

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
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


def _deallocate_decode_inputs(device_inputs) -> None:
    for value in device_inputs:
        if value is not None and hasattr(value, "deallocate"):
            value.deallocate(True)


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def host_canvas_to_device(mesh_device, canvas_tokens: torch.Tensor):
    """Move host canvas token ids ``[batch, canvas_len]`` to W3 device layout."""
    if canvas_tokens.dim() != 2:
        raise ValueError("canvas_tokens must have shape [batch, canvas_len]")
    batch, canvas_len = canvas_tokens.shape
    return ttnn.from_torch(
        canvas_tokens.view(batch, 1, canvas_len, 1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def host_tokens_to_device(mesh_device, tokens: torch.Tensor):
    """Move host token ids ``[batch, seq_len]`` to Gemma4 token layout."""
    if tokens.dim() != 2:
        raise ValueError("tokens must have shape [batch, seq_len]")
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
    if isinstance(input_ids, torch.Tensor):
        tokens = input_ids.to(torch.long)
    else:
        tokens = torch.tensor(input_ids, dtype=torch.long)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.dim() != 2:
        raise ValueError("prompt token ids must have shape [batch, seq_len]")
    return tokens


def tokenize_prompt(
    tokenizer,
    prompt,
    *,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    """Tokenize a prompt string or chat messages into host ids ``[batch, seq_len]``."""
    if isinstance(prompt, torch.Tensor):
        return _as_prompt_token_tensor(prompt)

    if hasattr(tokenizer, "apply_chat_template"):
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        return _as_prompt_token_tensor(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
            )
        )

    if not isinstance(prompt, str):
        raise ValueError("chat messages require a tokenizer with apply_chat_template")
    if callable(tokenizer):
        return _as_prompt_token_tensor(tokenizer(prompt, return_tensors="pt"))
    return _as_prompt_token_tensor(tokenizer.encode(prompt))


def embed_host_tokens(tt_model, tokens: torch.Tensor):
    """Embed host token ids as ``[1, 1, seq_len, hidden]`` TILE states."""
    tt_tokens = host_tokens_to_device(tt_model.mesh_device, tokens)
    embeds = tt_model.embed_tokens(tt_tokens)
    tt_tokens.deallocate(True)
    embeds = ttnn.reshape(embeds, (1, 1, tokens.shape[1], tt_model.hidden_size))
    return ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)


def prefill_prompt_tokens(tt_model, prompt_tokens: torch.Tensor, *, page_table=None, page_tables_per_layer=None) -> int:
    """Write prompt token K/V into the frozen cache and return prompt length."""
    if prompt_tokens.dim() != 2:
        raise ValueError("prompt_tokens must have shape [batch, seq_len]")
    if prompt_tokens.shape[0] != 1:
        raise NotImplementedError("prefill_prompt_tokens currently supports batch=1")
    prompt_embeds = embed_host_tokens(tt_model, prompt_tokens)
    logits = tt_model(
        prompt_embeds,
        is_decode=False,
        input_ids_torch=prompt_tokens,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    logits.deallocate(True)
    return prompt_tokens.shape[1]


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


def make_host_canvas_init_fn(mesh_device, host_canvases):
    """Create a ``generate_blocks`` init hook from fixed host canvas tensors."""
    canvases = list(host_canvases)

    def init_canvas_fn(block_idx: int, start_pos: int):
        del start_pos
        return host_canvas_to_device(mesh_device, canvases[block_idx])

    return init_canvas_fn


def _check_random_token_args(batch: int, canvas_len: int, vocab_size: int) -> None:
    if batch <= 0:
        raise ValueError("batch must be positive")
    if canvas_len <= 0:
        raise ValueError("canvas_len must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")


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
    """Create deterministic device Gumbel noise hooks for ``generate_blocks``."""
    _check_random_token_args(batch, canvas_len, vocab_size)

    def gumbel_noise_for_block(block_idx: int):
        def gumbel_noise_for_step(step: int):
            return TS.sample_gumbel_noise(
                (batch, 1, canvas_len, vocab_size),
                device=mesh_device,
                seed=seed + block_idx * 1_000_003 + step,
            )

        return gumbel_noise_for_step

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
    if canvas_tokens.dim() != 2:
        raise ValueError("canvas_tokens must have shape [batch, canvas_len]")
    if canvas_tokens.shape[0] != 1:
        raise NotImplementedError("commit_canvas_tokens currently supports batch=1")

    for offset in range(canvas_tokens.shape[1]):
        token = canvas_tokens[:, offset]
        position = torch.tensor([start_pos + offset], dtype=torch.int32)
        device_inputs = tt_model.prepare_inputs_decode(token, position, page_table=page_table)
        logits, _ = tt_model.ttnn_decode_forward(
            device_inputs[0],
            device_inputs[1],
            device_inputs[2],
            device_inputs[3],
            page_tables_per_layer=page_tables_per_layer,
            kv_phase=KVCachePhase.COMMIT_APPEND,
        )
        logits.deallocate(True)
        _deallocate_decode_inputs(device_inputs)


def _set_q_rope_offset(logits_fn, q_rope_offset: int) -> None:
    if hasattr(logits_fn, "q_rope_offset"):
        logits_fn.q_rope_offset = q_rope_offset


def _contains_stop_token(tokens: torch.Tensor, stop_token_ids) -> bool:
    if stop_token_ids is None:
        return False
    ids = _normalize_eos_token_ids(stop_token_ids)
    return any(token_id in ids for token_id in tokens.reshape(-1).tolist())


def _validate_num_blocks(num_blocks: int) -> None:
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")


def _validate_canvas_length(config: DiffusionConfig) -> None:
    if config.canvas_length <= 0:
        raise ValueError("config.canvas_length must be positive")


def _validate_max_new_tokens_capacity(num_blocks: int, canvas_length: int, max_new_tokens: int | None) -> None:
    if max_new_tokens is None:
        return
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens > num_blocks * canvas_length:
        raise ValueError("num_blocks is too small for max_new_tokens")


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
    denoise_block_fn: Callable[..., DenoiseTrajectory] = tt_denoise_block,
    commit_fn: Callable[..., None] = commit_canvas_tokens,
) -> GeneratedBlock:
    """Denoise one canvas, commit the clean argmax, and advance position.

    ``start_pos`` is the absolute canvas start for this block. When ``logits_fn``
    is a ``DenoiseLogitsAdapter`` this helper updates its ``q_rope_offset`` so
    canvas RoPE positions advance with each committed block.
    """
    _set_q_rope_offset(logits_fn, start_pos)
    trajectory = denoise_block_fn(
        logits_fn,
        init_canvas,
        config,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
    )
    if trajectory.committed is None:
        raise RuntimeError("denoise trajectory did not produce committed canvas tokens")
    commit_fn(
        tt_model,
        trajectory.committed,
        start_pos=start_pos,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    return GeneratedBlock(
        committed=trajectory.committed,
        next_pos=start_pos + trajectory.committed.shape[1],
        trajectory=trajectory,
    )


def generate_blocks(
    tt_model,
    logits_fn,
    *,
    prompt_len: int,
    num_blocks: int,
    config: DiffusionConfig,
    init_canvas_fn: Callable[[int, int], object],
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
    page_table=None,
    page_tables_per_layer=None,
    stop_token_ids=None,
    block_fn: Callable[..., GeneratedBlock] = denoise_and_commit_block,
) -> DeviceGeneration:
    """Run the minimal device outer loop for ``num_blocks`` canvases.

    ``init_canvas_fn(block_idx, start_pos)`` supplies the initial device canvas
    for each block. Use ``make_host_canvas_init_fn`` when replaying fixed torch /
    HF canvases; the full prompt/tokenizer path will later own default canvas
    creation. This helper owns commit-append and absolute position advancement.
    """
    _validate_num_blocks(num_blocks)
    _validate_canvas_length(config)
    next_pos = prompt_len
    committed_blocks: list[torch.Tensor] = []
    trajectories: list[DenoiseTrajectory] = []

    for block_idx in range(num_blocks):
        init_canvas = init_canvas_fn(block_idx, next_pos)
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
        )
        committed_blocks.append(block.committed)
        trajectories.append(block.trajectory)
        next_pos = block.next_pos
        if _contains_stop_token(block.committed, stop_token_ids):
            break

    generated = torch.cat(committed_blocks, dim=1) if committed_blocks else torch.zeros((1, 0), dtype=torch.long)
    return DeviceGeneration(generated=generated, prompt_len=prompt_len, next_pos=next_pos, trajectories=trajectories)


def generate_from_prompt_tokens(
    tt_model,
    logits_fn,
    prompt_tokens: torch.Tensor,
    *,
    num_blocks: int,
    config: DiffusionConfig,
    init_canvas_fn: Callable[[int, int], object],
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
    prompt_len = prefill_fn(
        tt_model,
        prompt_tokens,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    logits_fn = _resolve_generation_logits_fn(
        tt_model,
        logits_fn,
        logits_fn_builder,
        prompt_tokens=prompt_tokens,
        prompt_len=prompt_len,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    return blocks_fn(
        tt_model,
        logits_fn,
        prompt_len=prompt_len,
        num_blocks=num_blocks,
        config=config,
        init_canvas_fn=init_canvas_fn,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
        stop_token_ids=stop_token_ids,
    )


def generation_sequences(prompt_tokens: torch.Tensor, generation: DeviceGeneration) -> torch.Tensor:
    """Return HF-style ``[prompt, generated]`` token sequences for decode/e2e checks."""
    if prompt_tokens.dim() != 2:
        raise ValueError("prompt_tokens must have shape [batch, seq_len]")
    if generation.generated.dim() != 2:
        raise ValueError("generation.generated must have shape [batch, seq_len]")
    if prompt_tokens.shape[0] != generation.generated.shape[0]:
        raise ValueError("prompt_tokens and generation.generated batch sizes must match")
    if prompt_tokens.shape[1] != generation.prompt_len:
        raise ValueError("prompt_tokens length must match generation.prompt_len")
    return torch.cat([prompt_tokens, generation.generated], dim=1)


def _normalize_eos_token_ids(eos_token_id):
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def _trim_generated_token_ids(generated: torch.Tensor, *, max_new_tokens: int | None = None, eos_token_id=None):
    if max_new_tokens is not None and max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

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


def generate_text(
    tt_model,
    logits_fn,
    tokenizer,
    prompt,
    *,
    num_blocks: int,
    config: DiffusionConfig,
    init_canvas_fn: Callable[[int, int], object],
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
    _validate_max_new_tokens_capacity(num_blocks, config.canvas_length, max_new_tokens)
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
    if num_blocks is None:
        max_new_tokens = generate_kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            raise ValueError("num_blocks is required unless max_new_tokens is provided")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        num_blocks = (max_new_tokens + config.canvas_length - 1) // config.canvas_length
    _validate_num_blocks(num_blocks)
    _validate_max_new_tokens_capacity(num_blocks, config.canvas_length, generate_kwargs.get("max_new_tokens"))
    if vocab_size is None:
        vocab_size = _infer_generation_vocab_size(tokenizer, tt_model)
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
    if "noise_tokens_fn" not in generate_kwargs and (seed is not None or noise_seed is not None):
        if vocab_size is None:
            raise ValueError("noise_tokens_fn requires vocab_size or tokenizer/model vocab metadata")
        generate_kwargs["noise_tokens_fn"] = make_seeded_host_noise_tokens_fn(
            tt_model.mesh_device,
            batch=batch,
            canvas_len=config.canvas_length,
            vocab_size=vocab_size,
            seed=noise_seed if noise_seed is not None else seed + 1,
        )
    if "gumbel_noise_fn" not in generate_kwargs and (seed is not None or gumbel_seed is not None):
        if vocab_size is None:
            raise ValueError("gumbel_noise_fn requires vocab_size or tokenizer/model vocab metadata")
        generate_kwargs["gumbel_noise_fn"] = make_seeded_gumbel_noise_fn(
            tt_model.mesh_device,
            batch=batch,
            canvas_len=config.canvas_length,
            vocab_size=vocab_size,
            seed=gumbel_seed if gumbel_seed is not None else seed + 2,
        )
    if "eos_token_id" not in generate_kwargs:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
    if "decode_kwargs" not in generate_kwargs:
        generate_kwargs["decode_kwargs"] = {"skip_special_tokens": True}
    adapter_kwargs = dict(adapter_kwargs or {})
    if "config" not in adapter_kwargs:
        adapter_config = getattr(tt_model, "hf_config", None)
        if adapter_config is not None:
            adapter_kwargs["config"] = adapter_config
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
