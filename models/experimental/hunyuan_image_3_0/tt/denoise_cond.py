# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device denoise conditioning helpers: wte embed, attention mask upload,
# I2I base_embeds upload, and per-step distill timestep/guidance scatter.

from __future__ import annotations

from typing import Callable

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.tt.attention.mask import build_attention_mask_tt


def replicate_to_mesh(mesh_device, tensor_host, *, dtype=ttnn.bfloat16):
    """Upload a host torch tensor replicated across a mesh."""
    return ttnn.from_torch(
        tensor_host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def make_wte_embed_fn(mesh_device, wte_weight: torch.Tensor, *, replicate_fn: Callable | None = None):
    """Return ``embed_ids(input_ids) -> ttnn [B, S, H]`` using device wte."""
    w = wte_weight.detach().float()
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if replicate_fn is not None else None
    embed_weight = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    def embed_ids(input_ids: torch.Tensor):
        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
        ids_tt = ttnn.from_torch(
            ids.long(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        emb = ttnn.embedding(ids_tt, embed_weight, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(ids_tt)
        if len(emb.shape) == 4 and emb.shape[0] == 1:
            b, s, h = emb.shape[1], emb.shape[2], emb.shape[3]
            emb = ttnn.reshape(emb, [b, s, h])
        return emb

    embed_ids.weight = embed_weight
    return embed_ids


def split_t2i_text_embeds_tt(emb_tt, gen_slice: slice):
    """Split device ``[B, S, H]`` embeddings into ``text_pre`` / ``text_post`` around gen span."""
    b, s, h = emb_tt.shape
    text_pre = ttnn.slice(emb_tt, [0, 0, 0], [b, gen_slice.start, h]) if gen_slice.start > 0 else None
    text_post = ttnn.slice(emb_tt, [0, gen_slice.stop, 0], [b, s, h]) if gen_slice.stop < s else None
    return text_pre, text_post


def replace_token_at_tt(base_embeds, pos: int, token_b1h):
    """Replace sequence position ``pos`` in ``base_embeds`` ``[B,S,H]`` with ``token_b1h`` ``[B,1,H]``."""
    b, s, h = base_embeds.shape
    pieces = []
    if pos > 0:
        pieces.append(ttnn.slice(base_embeds, [0, 0, 0], [b, pos, h]))
    pieces.append(token_b1h)
    if pos + 1 < s:
        pieces.append(ttnn.slice(base_embeds, [0, pos + 1, 0], [b, s, h]))
    out = ttnn.concat(pieces, dim=1)
    for p in pieces:
        if p is not token_b1h:
            ttnn.deallocate(p)
    ttnn.deallocate(base_embeds)
    return out


def _embed_scalar_tt(embedder, scalar: float, batch: int = 1):
    """Run a resident ``HunyuanTtTimestepEmbedder`` for one scalar -> ``[B,1,H]`` TILE."""
    t = torch.tensor([float(scalar)] * batch, dtype=torch.float32)
    out = embedder.forward(t)
    h = int(out.shape[-1])
    out = ttnn.reshape(out, [batch, 1, h])
    return out


def scatter_step_embeds_tt(
    base_embeds,
    *,
    t_scalar: float,
    gen_timestep_scatter_index: torch.Tensor | None = None,
    tt_timestep_emb=None,
    guidance_scalar: float | None = None,
    guidance_scatter_index: torch.Tensor | None = None,
    tt_guidance_emb=None,
    t_r_scalar: float | None = None,
    gen_timestep_r_scatter_index: torch.Tensor | None = None,
    tt_timestep_r_emb=None,
):
    """Device-side mirror of ``scatter_distill_step_embeds`` (mutates via slice+concat)."""
    hidden = base_embeds
    bsz = int(hidden.shape[0])

    if gen_timestep_scatter_index is not None and tt_timestep_emb is not None:
        pos = int(gen_timestep_scatter_index.reshape(-1)[0].item())
        tok = _embed_scalar_tt(tt_timestep_emb, t_scalar, bsz)
        hidden = replace_token_at_tt(hidden, pos, tok)
        ttnn.deallocate(tok)

    if guidance_scalar is not None and guidance_scatter_index is not None and tt_guidance_emb is not None:
        pos = int(guidance_scatter_index.reshape(-1)[0].item())
        tok = _embed_scalar_tt(tt_guidance_emb, guidance_scalar, bsz)
        hidden = replace_token_at_tt(hidden, pos, tok)
        ttnn.deallocate(tok)

    if t_r_scalar is not None and gen_timestep_r_scatter_index is not None and tt_timestep_r_emb is not None:
        pos = int(gen_timestep_r_scatter_index.reshape(-1)[0].item())
        tok = _embed_scalar_tt(tt_timestep_r_emb, t_r_scalar, bsz)
        hidden = replace_token_at_tt(hidden, pos, tok)
        ttnn.deallocate(tok)

    return hidden


def attention_mask_tt_for_cond(mesh_device, cond: dict, *, seq_len: int, attn_spans=None, bsz: int | None = None):
    """Build or upload an additive attention mask on device."""
    if attn_spans is not None:
        batch = bsz if bsz is not None else cond.get("batch", 1)
        if batch == 1:
            return build_attention_mask_tt(
                mesh_device,
                seq_len,
                image_slices=[attn_spans],
                bsz=1,
                dtype=ttnn.bfloat16,
            )
        return build_attention_mask_tt(
            mesh_device,
            seq_len,
            image_slices=[attn_spans] * batch,
            bsz=batch,
            dtype=ttnn.bfloat16,
        )

    mask = cond.get("attention_mask")
    if isinstance(mask, torch.Tensor):
        batch = cond.get("batch", 1)
        s = mask.shape[-1]
        host = mask.reshape(batch, 1, s, s).to(torch.bfloat16)
        return replicate_to_mesh(mesh_device, host)
    return mask


def upload_denoise_cond(
    mesh_device,
    cond: dict,
    *,
    replicate_fn: Callable | None = None,
    seq_len: int | None = None,
    attn_spans=None,
):
    """Upload host denoise cond fields once; enable persistent on-device base_embeds."""
    rep = replicate_fn or (lambda t: replicate_to_mesh(mesh_device, t))
    out = dict(cond)
    batch = cond.get("batch", 1)

    host_base = cond.get("base_embeds_host")
    if host_base is not None:
        out["base_embeds"] = rep(host_base.to(torch.bfloat16))
        out["base_embeds_persistent"] = True

    if seq_len is not None:
        out["attention_mask"] = attention_mask_tt_for_cond(
            mesh_device,
            cond,
            seq_len=seq_len,
            attn_spans=attn_spans,
            bsz=batch,
        )
    elif isinstance(cond.get("attention_mask"), torch.Tensor):
        s = cond["attention_mask"].shape[-1]
        out["attention_mask"] = rep(cond["attention_mask"].reshape(batch, 1, s, s).to(torch.bfloat16))

    if isinstance(cond.get("text_pre"), torch.Tensor):
        out["text_pre"] = rep(cond["text_pre"].to(torch.bfloat16))
    if isinstance(cond.get("text_post"), torch.Tensor):
        out["text_post"] = rep(cond["text_post"].to(torch.bfloat16))

    return out
