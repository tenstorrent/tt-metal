# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common TT helpers for Dots OCR (prefill padding, vision fusion, paging, sampling).

Public API:
- `merge_vision_tokens`         — scatter vision embeds into text embeds via `image_token_id` (host torch).
- `merge_vision_tokens_ttnn`   — same scatter on device; ``ttnn.scatter`` (see also ``fused_ttnn_embeddings_to_torch``).
- `text_embeds_from_ttnn_embedding` / `pad_embedding_ttnn` — host [B,S,D] / [D] from the TT token table.
- `text_embeds_from_ttnn_embedding_ttnn` / `pad_embedding_ttnn_tensor` — ttnn [B,S,D] / [1,D] on the mesh.
- `preprocess_inputs_prefill`   — pad each user's input embeddings to a power-of-2 prefill length (host).
- `preprocess_inputs_prefill_ttnn` — same padding with ``ttnn.concat`` / ``repeat`` / ``stack`` (stays on device).
- `ttnn_fused_batch_to_user_list` — split ttnn ``[B,S,D]`` into a list of ``[S,D]`` for ttnn prefill.
- `fused_ttnn_embeddings_to_torch` — D2H readback before :class:`DotsTransformer.prepare_inputs_prefill`.
- `text_rope_from_hf`           — build host cos/sin from the HF reference text model (Qwen2-style RoPE).
- `PagedAttentionConfig`        — block/size config for paged KV.
- `get_padded_prefill_len`      — prefill-length rounding (<=128 to 128, else min(power-of-2, multiple-of-2048)).
- `num_blocks_in_seq` / `get_block_size` / `get_max_prefill_chunk_size` — paging helpers.
- `sample_host`                 — host-side sampler with optional `ttnn` up-cast.
- `nearest_multiple` / `nearest_pow_2` — tiling helpers.

These follow the ``models.tt_transformers.tt.generator.Generator`` prefill/decode contract.

Device path (keeps text + vision fusion on the mesh before a single readback to torch):

- ``text_embeds_from_ttnn_embedding_ttnn`` + ``pad_embedding_ttnn_tensor`` — embeddings stay on device.
- ``merge_vision_tokens_ttnn`` + ``preprocess_inputs_prefill_ttnn`` — :func:`ttnn.scatter` fusion + pad/concat/``ttnn.stack``.
- ``fused_ttnn_embeddings_to_torch`` — :func:`ttnn.to_torch` for :class:`DotsTransformer.prepare_inputs_prefill` (sharded H2D upload via existing ``ttnn.from_torch``).
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
        input_embeds:  [B, S, D] text embeddings (e.g. :func:`text_embeds_from_ttnn_embedding` or HF embeds).
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


def _ttnn_embd_4d_to_torch_batched(
    embd4,
    mesh,
    *,
    b: int,
    s_len: int,
) -> torch.Tensor:
    """Convert ``unsqueeze_to_4D( embd( tokens ) )`` to torch [B, S, D]."""
    ttnn = get_ttnn()
    composer = None
    if mesh is not None and hasattr(ttnn, "ConcatMeshToTensor"):
        try:
            n = mesh.get_num_devices()
            if n is not None and n > 1:
                composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
        except Exception:
            pass
    out = ttnn.to_torch(embd4, mesh_composer=composer) if composer is not None else ttnn.to_torch(embd4)
    if composer is not None and out.dim() >= 1 and mesh is not None:
        try:
            n = mesh.get_num_devices()
            if n and n > 1 and int(out.shape[0]) % n == 0:
                out = out[: int(out.shape[0]) // n]
        except Exception:
            pass
    out = out.to(torch.bfloat16)
    d = int(out.shape[-1])
    ntok = b * s_len
    if int(out.numel()) != ntok * d:
        raise RuntimeError(f"token embed to_torch: expected {ntok * d} elements, got {out.numel()}")
    return out.reshape(b, s_len, d)


def text_embeds_from_ttnn_embedding(tt_model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    [B, S] token indices → [B, S, D] **bfloat16** (host) using the TT model's ``embd`` module
    (``ttnn.embedding`` over ``tok_embeddings.weight`` on the mesh).

    Replaces ``ref.model.get_input_embeddings()(input_ids)`` with the same table as
    :class:`models.tt_transformers.tt.model.Transformer` 2D ``prepare_inputs_prefill`` (id path).
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for text_embeds_from_ttnn_embedding")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [B, S]")
    mesh = tt_model.mesh_device
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    b, s_len = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    if b > 1:
        flat = input_ids.reshape(1, 1, 1, b * s_len)
    else:
        flat = input_ids.reshape(1, 1, 1, s_len)
    tt_tok = ttnn.from_torch(
        flat.to(torch.int32),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    embd = tt_model.embd(tt_tok, memory_config=mem)
    embd4 = ttnn.unsqueeze_to_4D(embd)
    return _ttnn_embd_4d_to_torch_batched(embd4, mesh, b=b, s_len=s_len)


def pad_embedding_ttnn(tt_model, pad_token_id: int) -> torch.Tensor:
    """
    [D] **bfloat16** padding row for ``preprocess_inputs_prefill``, from the TT token table at
    ``pad_token_id`` (replaces host ``get_input_embeddings()([pad_token_id]).squeeze(0)``).
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for pad_embedding_ttnn")
    mesh = tt_model.mesh_device
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    x = torch.tensor([[[[int(pad_token_id)]]]], dtype=torch.int32)
    tt_tok = ttnn.from_torch(
        x,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    embd = tt_model.embd(tt_tok, memory_config=mem)
    embd4 = ttnn.unsqueeze_to_4D(embd)
    row = _ttnn_embd_4d_to_torch_batched(embd4, mesh, b=1, s_len=1)
    return row.reshape(-1).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Device-side text embeddings, fusion, and ttnn prefill (D2H once in fused_ttnn_embeddings_to_torch)
# ---------------------------------------------------------------------------


def _mesh_composer_bsh_to_torch(ttnn, mesh: object) -> object | None:
    if mesh is None or not hasattr(ttnn, "ConcatMesh2dToTensor"):
        return None
    try:
        n = mesh.get_num_devices()
    except Exception:
        return None
    if n is None or n <= 1:
        return None
    try:
        return ttnn.ConcatMesh2dToTensor(mesh, dims=(1, -1), mesh_shape=list(mesh.shape))
    except Exception:
        return None


def fused_ttnn_embeddings_to_torch(tt: object, mesh: object) -> torch.Tensor:
    """ttnn [B, S, D] fused → bfloat16 torch for ``Generator.prefill_forward_text``."""
    ttnn = get_ttnn()
    if ttnn is None or not isinstance(tt, ttnn.Tensor):
        raise TypeError("fused_ttnn_embeddings_to_torch expects a ttnn.Tensor on device")
    comp = _mesh_composer_bsh_to_torch(ttnn, mesh)
    t = ttnn.to_torch(tt, mesh_composer=comp) if comp is not None else ttnn.to_torch(tt)
    t = t.to(torch.bfloat16)
    if t.dim() != 3:
        raise ValueError(f"expected [B, S, D] after to_torch, got {tuple(t.shape)}")
    return t


def text_embeds_from_ttnn_embedding_ttnn(tt_model, input_ids: torch.Tensor) -> object:
    """[B, S] token ids → ttnn [B, S, D] on the mesh (no D2H)."""
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for text_embeds_from_ttnn_embedding_ttnn")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [B, S]")
    mesh = tt_model.mesh_device
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    b, s_len = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    h = int(getattr(tt_model.args, "dim", None) or 1536)
    if b > 1:
        flat = input_ids.reshape(1, 1, 1, b * s_len)
    else:
        flat = input_ids.reshape(1, 1, 1, s_len)
    tt_tok = ttnn.from_torch(
        flat.to(torch.int32),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    embd = tt_model.embd(tt_tok, memory_config=mem)
    embd4 = ttnn.unsqueeze_to_4D(embd)
    try:
        return ttnn.reshape(embd4, (b, s_len, h))
    except Exception:
        if hasattr(ttnn, "to_layout") and mem is not None:
            try:
                embd4 = ttnn.to_layout(embd4, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
            except Exception:
                pass
        return ttnn.reshape(embd4, (b, s_len, h))


def pad_embedding_ttnn_tensor(tt_model, pad_token_id: int) -> object:
    """ttnn [1, D] bfloat16 pad row on device."""
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for pad_embedding_ttnn_tensor")
    mesh = tt_model.mesh_device
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    h = int(getattr(tt_model.args, "dim", None) or 1536)
    x = torch.tensor([[[[int(pad_token_id)]]]], dtype=torch.int32)
    tt_tok = ttnn.from_torch(
        x,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    embd = tt_model.embd(tt_tok, memory_config=mem)
    embd4 = ttnn.unsqueeze_to_4D(embd)
    try:
        return ttnn.reshape(embd4, (1, h))
    except Exception:
        if hasattr(ttnn, "to_layout") and mem is not None:
            try:
                embd4 = ttnn.to_layout(embd4, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
            except Exception:
                pass
        return ttnn.reshape(embd4, (1, h))


def merge_vision_tokens_ttnn(
    input_ids: torch.Tensor,
    input_embeds: object,
    image_embeds: object,
    hf_config: object,
    mesh_device: object,
) -> object:
    """
    Device-side scatter. ``image_embeds`` is torch [N, H] (uploaded) or ttnn [N, H].
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for merge_vision_tokens_ttnn")
    if not isinstance(input_embeds, ttnn.Tensor):
        raise TypeError("input_embeds must be ttnn.Tensor [B, S, H]")

    image_token_id = int(getattr(hf_config, "image_token_id"))
    n_image = int((input_ids == image_token_id).sum().item())
    n_f = int(getattr(image_embeds, "shape")[0])
    if n_image != n_f:
        raise ValueError(f"Image features and image tokens do not match: tokens={n_image}, features={n_f}")
    if n_image == 0:
        return input_embeds

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if not isinstance(image_embeds, ttnn.Tensor):
        fkw: dict = {"device": mesh_device, "dtype": ttnn.bfloat16, "layout": ttnn.ROW_MAJOR_LAYOUT}
        if mem is not None:
            fkw["memory_config"] = mem
        image_tt = ttnn.from_torch(image_embeds.to(torch.bfloat16), **fkw)
    else:
        image_tt = image_embeds

    B, S, H = int(input_embeds.shape[0]), int(input_embeds.shape[1]), int(input_embeds.shape[2])
    mask_idx = torch.where(input_ids.view(-1) == image_token_id)[0]
    mask_idx = mask_idx.contiguous().view(n_image, 1).expand(n_image, H)
    mask_idx_tt = ttnn.from_torch(
        mask_idx.to(torch.int32), device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    flat = ttnn.reshape(input_embeds, (-1, H))
    flat = ttnn.scatter(flat, 0, mask_idx_tt, image_tt)
    return ttnn.reshape(flat, (B, S, H))


def _ttnn_expand_pad_block(ttnn, pad_1d: object, n_rows: int, d: int) -> object:
    row2 = ttnn.reshape(pad_1d, (1, d))
    if hasattr(ttnn, "Shape"):
        try:
            return ttnn.repeat(row2, ttnn.Shape((n_rows, 1)))
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return ttnn.expand(row2, (n_rows, d))
    except (TypeError, ValueError, RuntimeError):
        return ttnn.repeat(row2, (n_rows, 1))


def preprocess_inputs_prefill_ttnn(
    input_embeds: list,
    model_args: object,
    attention_mask: torch.Tensor,
    pad_embedding: object,
    max_prefill_len: int | None = None,
) -> tuple[object, list[int], list[int]]:
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for preprocess_inputs_prefill_ttnn")
    if not isinstance(pad_embedding, ttnn.Tensor):
        raise TypeError("pad_embedding must be ttnn (use pad_embedding_ttnn_tensor)")

    if max_prefill_len is None:
        max_prefill_len = int(getattr(model_args, "max_seq_len"))

    slens = [int(t.shape[0]) for t in input_embeds]  # type: ignore[union-attr]
    logger.info("Encoded prompt lengths: " + ", ".join(str(s) for s in slens))
    max_prompt_len = max(slens)
    assert (
        max_prompt_len <= max_prefill_len
    ), f"Max prompt length {max_prompt_len} exceeds max prefill len {max_prefill_len}"

    logger.info(f"# of users: {len(input_embeds)}")
    out_rows: list = []
    decoding_pos: list[int] = []
    prefill_lens: list[int] = []
    d = int(pad_embedding.shape[1])

    for i, input_embed in enumerate(input_embeds):
        if not isinstance(input_embed, ttnn.Tensor):
            raise TypeError("input_embeds must be ttnn.Tensors on device")
        umask = attention_mask[i]
        actual = int(umask.sum().item())
        seq_tok = int(input_embed.shape[0])
        prefill_seq = int(min(max_prefill_len, max(2 ** math.ceil(math.log(max(seq_tok, 1), 2)), 128)))
        if actual > prefill_seq:
            raise ValueError(f"User {i}: actual {actual} cannot fit prefill {prefill_seq}")
        n_pad = prefill_seq - actual
        if n_pad > 0:
            pad_block = _ttnn_expand_pad_block(ttnn, pad_embedding, n_pad, d)
            head = input_embed[0:actual, :]
            prefill_i = ttnn.concat([head, pad_block], dim=0)
        else:
            prefill_i = input_embed[0:actual, :]
        if int(prefill_i.shape[0]) != prefill_seq:
            raise RuntimeError(
                f"ttnn prefill: expected length {prefill_seq}, got {int(prefill_i.shape[0])} (n_pad={n_pad})"
            )
        out_rows.append(prefill_i)
        decoding_pos.append(actual)
        prefill_lens.append(prefill_seq)

    if len({int(t.shape[0]) for t in out_rows}) != 1:
        raise NotImplementedError(
            "Batched ttnn prefill requires the same prefill length per user (use batch=1 or torch prefill path)"
        )
    if len(out_rows) == 1:
        stacked = ttnn.unsqueeze(out_rows[0], 0)
    else:
        stacked = ttnn.stack(out_rows, dim=0)
    return stacked, decoding_pos, prefill_lens


def ttnn_fused_batch_to_user_list(fused_bsh: object) -> list:
    """
    Split a fused **[B, S, D]** ttnn tensor into a **list** of **[S, D]** tensors (one per user),
    as required by :func:`preprocess_inputs_prefill_ttnn`.
    """
    ttnn = get_ttnn()
    if ttnn is None or not isinstance(fused_bsh, ttnn.Tensor):
        raise TypeError("ttnn_fused_batch_to_user_list expects ttnn.Tensor [B, S, D]")
    b = int(fused_bsh.shape[0])
    if b == 1:
        return [fused_bsh[0, :, :]]
    return [fused_bsh[i, :, :] for i in range(b)]


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


def argmax_token_id_ttnn(
    tt_logits: object,
    *,
    mesh_device: object,
    batch_size: int = 1,
    layout: str = "decode",
) -> torch.Tensor:
    """
    Greedy token selection on device (same row selection as :meth:`Transformer.process_output_decode` + host ``argmax``).

    **Decode** outputs from ``ttnn_decode_forward`` are shaped like the pre-``process_output_decode`` host tensor:
    the **batch** dimension is typically padded to 32 in dim 2 (or dim 1 for some 3D tilings) — *not* a 32-wide
    sequence block. :meth:`~models.tt_transformers.tt.model.Transformer.process_output_decode` does
    ``tt_out[:, :, :B, :vocab_size]``; we must take the first ``batch_size`` slots, **not** the last index (which
    would read an unrelated padded slot and corrupt generation).

    **Prefill** blocks (32-token *sequence* chunks) use dim 2 for sequence; raw 4D prefill logits are not used
    here (prefill uses host ``[B, 1, V]`` via :func:`argmax_token_id_host_via_ttnn` before this helper).

    Args:
        tt_logits: ttnn tensor with vocab in the last dimension (e.g. ``[1, 1, 32, V]`` for decode).
        mesh_device: mesh used for correct to_torch composition.
        batch_size: Active batch (decode slots ``0 .. batch_size-1`` along the padded dim).
        layout: ``"decode"`` (default) — batch padding; ``"prefill_block"`` — last sequence index in a 32-chunk (rare).

    Returns:
        torch int64 tensor shaped [B, 1] token ids (host).
    """
    ttnn = get_ttnn()
    if ttnn is None or not isinstance(tt_logits, ttnn.Tensor):
        raise TypeError("argmax_token_id_ttnn expects tt_logits as a ttnn.Tensor")
    if layout not in ("decode", "prefill_block"):
        raise ValueError(f"layout must be 'decode' or 'prefill_block', got {layout!r}")
    b_req = int(batch_size)
    if b_req < 1:
        raise ValueError(f"batch_size must be >= 1, got {b_req}")

    vocab = int(tt_logits.shape[-1])
    x = tt_logits
    try:
        if len(x.shape) == 4:
            d0, d1, s_blk, _v = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
            if s_blk > 1:
                if layout == "decode":
                    take = min(b_req, s_blk)
                    x = ttnn.slice(x, (0, 0, 0, 0), (d0, d1, take, vocab))
                else:
                    # Prefill: one 32-token *sequence* chunk — last real token in block
                    x = ttnn.slice(x, (0, 0, s_blk - 1, 0), (d0, d1, s_blk, vocab))
        elif len(x.shape) == 3:
            d0, s_blk, _v = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
            if s_blk > 1:
                if layout == "decode":
                    take = min(b_req, s_blk)
                    x = ttnn.slice(x, (0, 0, 0), (d0, take, vocab))
                else:
                    x = ttnn.slice(x, (0, s_blk - 1, 0), (d0, s_blk, vocab))
    except Exception:
        # If slice isn't supported for this layout, fall back to flattening (may be wrong; prefer fixing slice).
        x = tt_logits

    flat = ttnn.reshape(x, (-1, vocab))
    idx = ttnn.argmax(flat, dim=-1)
    comp = _mesh_composer_bsh_to_torch(ttnn, mesh_device)
    idx_pt = ttnn.to_torch(idx, mesh_composer=comp) if comp is not None else ttnn.to_torch(idx)
    if idx_pt.dim() == 0:
        idx_pt = idx_pt.view(1)
    idx_pt = idx_pt.to(torch.int64).view(-1, 1)
    try:
        if x is not tt_logits:
            ttnn.deallocate(x)
    except Exception:
        pass
    try:
        ttnn.deallocate(flat)
    except Exception:
        pass
    try:
        ttnn.deallocate(idx)
    except Exception:
        pass
    return idx_pt


def argmax_token_id_host_via_ttnn(logits: torch.Tensor, *, mesh_device: object) -> torch.Tensor:
    """
    Greedy token selection using TTNN even when logits are on host.

    This avoids ``torch.argmax`` at the cost of a small H2D + D2H of the last-vocab row.
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for argmax_token_id_host_via_ttnn")
    if logits.dim() == 3:
        row = logits[:, -1:, :]  # [B, 1, V]
    elif logits.dim() == 2:
        row = logits[:, :]
    else:
        raise ValueError(f"expected logits rank 2/3, got shape {tuple(logits.shape)}")
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    fkw: dict = {"device": mesh_device, "dtype": ttnn.bfloat16, "layout": ttnn.ROW_MAJOR_LAYOUT}
    if mem is not None:
        fkw["memory_config"] = mem
    tt = ttnn.from_torch(row.to(torch.bfloat16), **fkw)
    out = argmax_token_id_ttnn(tt, mesh_device=mesh_device)
    try:
        ttnn.deallocate(tt)
    except Exception:
        pass
    return out


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
