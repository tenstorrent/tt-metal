# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common TT helpers for Dots OCR (prefill padding, vision fusion, paging, sampling).

Public API:
- `merge_vision_tokens`         — scatter vision embeds into text embeds via `image_token_id`.
- `preprocess_inputs_prefill`   — pad each user's input embeddings to a power-of-2 prefill length.
- `text_rope_from_hf`           — build host cos/sin from the HF reference text model (Qwen2-style RoPE).
- `PagedAttentionConfig`        — block/size config for paged KV.
- `get_padded_prefill_len`      — prefill-length rounding (<=128 to 128, else min(power-of-2, multiple-of-2048)).
- `num_blocks_in_seq` / `get_block_size` / `get_max_prefill_chunk_size` — paging helpers.
- `sample_host`                 — host-side sampler with optional `ttnn` up-cast.
- `nearest_multiple` / `nearest_pow_2` — tiling helpers.

These follow the ``models.tt_transformers.tt.generator.Generator`` prefill/decode contract.
"""

from __future__ import annotations

import math
import os

import torch
from loguru import logger

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.tt_transformers.tt.common import sample_top_p
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

# ---------------------------------------------------------------------------
# Fusion: vision embeddings into text embeddings
# ---------------------------------------------------------------------------


def merge_vision_tokens(
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    hf_config,
) -> torch.Tensor:
    """
    Scatter ``image_embeds`` into ``input_embeds`` at every position where
    ``input_ids == hf_config.image_token_id``.

    Matches the Dots/Qwen2-VL reference fusion: ``image_token_id`` is declared on the
    top-level HF config (``model.config.image_token_id``).

    Args:
        input_ids:     [B, S] token ids (from the processor).
        input_embeds:  [B, S, D] text embeddings (from ``model.get_input_embeddings()``).
        image_embeds:  [N_image_tokens, D] vision features from the vision tower / TT vision stack.
        hf_config:     HF model config carrying ``image_token_id``.

    Returns:
        [B, S, D] fused embeddings.
    """
    image_token_id = int(getattr(hf_config, "image_token_id"))
    n_image_tokens = int((input_ids == image_token_id).sum().item())
    n_image_features = int(image_embeds.shape[0])
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens={n_image_tokens}, features={n_image_features}"
        )

    mask = input_ids == image_token_id
    mask_expanded = mask.unsqueeze(-1).expand_as(input_embeds)
    image_mask = mask_expanded.to(input_embeds.device)
    return input_embeds.masked_scatter(image_mask, image_embeds)


# ---------------------------------------------------------------------------
# Prefill: pad each user's embeddings to a power-of-2 prefill length
# ---------------------------------------------------------------------------


def preprocess_inputs_prefill(
    input_embeds,
    model_args,
    attention_mask: torch.Tensor,
    pad_embedding: torch.Tensor,
    max_prefill_len: int | None = None,
):
    """
    Pad each user's fused ``input_embeds`` to a common prefill length.

    Each user's prefill length is the smallest ``max(128, next_pow2(len(embed)))`` that fits under
    ``max_prefill_len`` (defaults to ``model_args.max_seq_len``). Padding copies ``pad_embedding``
    to avoid polluting attention with zeros.

    Args:
        input_embeds:    list of per-user ``[S_i, D]`` embedding tensors.
        model_args:      ``DotsModelArgs`` / ``ModelArgs`` providing ``max_seq_len``.
        attention_mask:  [B, S] mask from the processor (1 = real token).
        pad_embedding:   [D] embedding used for padding (typically the ``pad_token_id`` embedding).
        max_prefill_len: optional upper bound; defaults to ``model_args.max_seq_len``.

    Returns:
        ``(input_prefill, decoding_pos, prefill_lens)`` — padded batch tensor and per-user metadata.
    """
    if max_prefill_len is None:
        max_prefill_len = model_args.max_seq_len

    logger.info("Encoded prompt lengths: " + ", ".join(str(len(prompt)) for prompt in input_embeds))

    max_prompt_len = max(len(x) for x in input_embeds)
    assert (
        max_prompt_len <= max_prefill_len
    ), f"Max prompt length {max_prompt_len} exceeds max prefill len {max_prefill_len}"

    logger.info(f"# of users: {len(input_embeds)}")
    input_prefill: list[torch.Tensor] = []
    decoding_pos: list[int] = []
    prefill_lens: list[int] = []

    for i, input_embed in enumerate(input_embeds):
        user_attention_mask = attention_mask[i]
        actual_prompt_len = int(user_attention_mask.sum().item())
        prefill_seq_len = min(max_prefill_len, max(2 ** math.ceil(math.log(len(input_embed), 2)), 128))

        input_prefill_i = torch.empty((prefill_seq_len, pad_embedding.shape[-1]), dtype=pad_embedding.dtype)
        input_prefill_i[:] = pad_embedding
        input_prefill_i[:actual_prompt_len, :] = input_embed[:actual_prompt_len, :]
        input_prefill.append(input_prefill_i)

        decoding_pos.append(actual_prompt_len)
        prefill_lens.append(prefill_seq_len)

    input_prefill_tensor = torch.stack(input_prefill)  # [B, prefill_seq_len, D]
    return input_prefill_tensor, decoding_pos, prefill_lens


# ---------------------------------------------------------------------------
# RoPE: build host cos/sin from the HF reference text model (Qwen2-style)
# ---------------------------------------------------------------------------


def text_rope_from_hf(
    inputs,
    input_embeds: torch.Tensor,
    reference_model,
    model_args,
    pad_token_id: int,
):
    """
    Precompute host cos/sin matrices for Dots' text decoder from the HF reference model.

    Dots uses a Qwen2-style rotary embedding on the **text** stack (no multimodal rope_deltas),
    so we reuse ``reference_model.model.language_model.rotary_emb`` when available, falling back
    to the model's top-level ``rotary_emb`` attribute.

    Returns:
        ``(cos, sin)`` — ``[1, 1, max_seq_len, head_dim]`` in meta-style interleaved format,
        compatible with ``DotsTransformer.prepare_inputs_prefill``.
    """
    max_seq_len = min(
        model_args.max_seq_len,
        max(2 ** math.ceil(math.log(inputs.input_ids.shape[-1], 2)), 128),
    )
    padded_inputs = torch.nn.functional.pad(
        inputs.input_ids, (0, max_seq_len - inputs.input_ids.shape[-1]), value=pad_token_id
    )
    position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0).expand(padded_inputs.shape[0], -1)

    lang = getattr(getattr(reference_model, "model", reference_model), "language_model", None)
    rotary_emb = getattr(lang, "rotary_emb", None) if lang is not None else None
    if rotary_emb is None:
        rotary_emb = getattr(reference_model.model, "rotary_emb", None)
    if rotary_emb is None:
        raise AttributeError("Reference HF model does not expose a `rotary_emb` module for text RoPE.")

    cos, sin = rotary_emb(input_embeds, position_ids)
    if cos.dim() == 3:  # [B, S, D] -> [B, 1, S, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    cos, sin = convert_rope_style_hf_to_meta(cos, sin)
    return cos, sin


# ---------------------------------------------------------------------------
# Paging helpers
# ---------------------------------------------------------------------------


class PagedAttentionConfig:
    """Thin config for paged KV cache (block size + number of blocks)."""

    def __init__(self, block_size: int = 32, max_num_blocks: int = 1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def nearest_pow_2(x: int) -> int:
    return 2 ** math.ceil(math.log2(x))


def get_padded_prefill_len(seq_len: int) -> int:
    """Pad ``seq_len`` to 128 (small) or min(next_pow2, next_mult_of_2048) (large)."""
    if seq_len <= 128:
        return 128
    pow_2_pad = nearest_pow_2(seq_len)
    mult_2048_pad = 2048 * math.ceil(seq_len / 2048)
    return min(pow_2_pad, mult_2048_pad)


def num_blocks_in_seq(seq_len: int, block_size: int) -> int:
    return math.ceil(seq_len / block_size)


def get_block_size(kv_cache) -> int:
    return kv_cache[0][0].shape[2]


def get_max_prefill_chunk_size(seq_len: int, max_prefill_seq_len: int) -> int:
    """Largest multiple of 2048 that divides ``seq_len`` and is ``<= max_prefill_seq_len``."""
    MIN_CHUNK_SIZE = 2048
    if not isinstance(seq_len, int) or not isinstance(max_prefill_seq_len, int):
        raise TypeError("Both seq_len and max_prefill_seq_len must be integers.")
    if seq_len <= 0 or max_prefill_seq_len <= 0:
        raise ValueError("Both seq_len and max_prefill_seq_len must be positive integers.")
    if seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of {MIN_CHUNK_SIZE}.")
    if max_prefill_seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"max_prefill_seq_len ({max_prefill_seq_len}) must be a multiple of {MIN_CHUNK_SIZE}.")

    max_possible_chunk = min(max_prefill_seq_len, seq_len)
    for chunk_size in range(max_possible_chunk, 0, -MIN_CHUNK_SIZE):
        if seq_len % chunk_size == 0:
            return chunk_size
    raise ValueError("No valid chunk size found")


# ---------------------------------------------------------------------------
# Sampling (host fallback + optional ttnn up-cast)
# ---------------------------------------------------------------------------


def sample_host(tt_input, mesh_device, temperature: float = 0.6, top_p: float = 0.08, on_host: bool = True):
    """Host-side sampler.

    - If ``mesh_device`` is truthy, assume ``tt_input`` is a ``ttnn.Tensor`` and pull it back.
    - Otherwise assume ``tt_input`` is already a torch tensor.

    Returns ``(maybe_ttnn_tensor, torch_tensor)``. In ``on_host=True`` mode the ttnn tensor is
    created without a device (host-only).
    """
    ttnn = get_ttnn()
    vocab_size = tt_input.shape[-1]
    if mesh_device and ttnn is not None:
        pt_input = ttnn.to_torch(
            tt_input,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=list(mesh_device.shape)),
        )[:, :1, :, :vocab_size]
    else:
        pt_input = tt_input[..., :vocab_size]

    if temperature > 0:
        probs = torch.softmax(pt_input / temperature, dim=-1)
        pt_out = sample_top_p(probs.squeeze(), top_p)
        if mesh_device:
            pt_out = pt_out.view(1, 1, 1, -1)
    else:
        if mesh_device:
            pt_out = torch.argmax(pt_input, dim=-1, keepdim=True).transpose(-1, -2)
        else:
            pt_out = torch.argmax(pt_input, dim=-1)

    if mesh_device is None or ttnn is None:
        if pt_out.dim() == 1:
            pt_out = pt_out.unsqueeze(0)
        return None, pt_out

    if on_host:
        return (
            ttnn.as_tensor(
                pt_out,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
            ),
            pt_out,
        )
    return (
        ttnn.from_torch(
            pt_out,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        pt_out,
    )


def nearest_multiple(x: int, multiple_of: int) -> int:
    return math.ceil(x / multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# Debug tensor save/load (useful during bring-up)
# ---------------------------------------------------------------------------


def check_tensor(ttnn_tensor, name: str, mesh_device) -> None:
    ttnn = get_ttnn()
    if ttnn is None or mesh_device is None:
        return
    our = torch.Tensor(ttnn.to_torch(ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)))
    if not os.path.exists(f"{name}.pt"):
        torch.save(our, f"{name}.pt")
        return
    ref = torch.load(f"{name}.pt")
    if not torch.allclose(our, ref):
        logger.error(f"Tensor {name} mismatch")
    else:
        logger.info(f"Tensor match: {name}")
