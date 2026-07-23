# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# I2I cond token instantiation — pure TTNN (mirrors ref/image_gen/input_instantiate).
# Patch/time embedders run on device; sequence inject uses device concat scatter.

from __future__ import annotations

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown
from models.experimental.hunyuan_image_3_0.tt.image_gen.sequence_scatter import mask_to_spans, scatter_token_spans
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.vae.cond_posterior import latent_bthwc_to_patch_input


def _mask_row(mask, row: int) -> list:
    if hasattr(mask, "ndim") and int(mask.ndim) > 1:
        row = mask[row]
    if hasattr(row, "tolist"):
        return row.tolist()
    return list(row)


def _index_row(index, row: int) -> list[int]:
    if hasattr(index, "ndim") and int(index.ndim) > 1:
        row = index[row]
    if hasattr(row, "tolist"):
        return [int(v) for v in row.tolist()]
    return [int(v) for v in row]


def _zero_timestep(batch: int) -> torch.Tensor:
    """Host ``t=0`` vector for ``HunyuanTtTimestepEmbedder`` (uploads as ``[1,1,N,1]``)."""
    return torch.zeros(batch, dtype=torch.float32)


def _timestep_for_embed(t):
    """Normalize timesteps for ``HunyuanTtTimestepEmbedder.forward`` (host or device).

    Device tensors stay on device — no D2H. Host torch vectors are reshaped to 1-D.
    """
    if isinstance(t, torch.Tensor):
        return t.reshape(-1).float()
    return t


def _tokens_from_latents_one(
    patch_embed: HunyuanTtUNetDown,
    time_embed: HunyuanTtTimestepEmbedder,
    latents_bthwc: ttnn.Tensor,
) -> ttnn.Tensor:
    """``[1,1,H,W,Z]`` -> ``[1,n,H]`` TILE. ResBlock adaGN only supports ``B=1``."""
    te_tt = time_embed.forward(_zero_timestep(1), keep_resident=False)
    x_flat, b, h, w = latent_bthwc_to_patch_input(latents_bthwc)
    assert b == 1, f"expected batch-1 latent slice, got B={b}"
    tok_flat, _, _ = patch_embed.forward_latent(x_flat, te_tt, 1, h, w)
    ttnn.deallocate(te_tt, force=False)
    ttnn.deallocate(x_flat, force=False)
    tok_flat = ttnn.to_layout(tok_flat, ttnn.TILE_LAYOUT)
    n_total = int(tok_flat.shape[2])
    hidden = int(tok_flat.shape[3])
    n_per = n_total // b
    return ttnn.reshape(tok_flat, [1, n_per, hidden])


def _tokens_from_latents(
    patch_embed: HunyuanTtUNetDown,
    time_embed: HunyuanTtTimestepEmbedder,
    latents_bthwc: ttnn.Tensor,
    mesh_device,
) -> ttnn.Tensor:
    """``[B,1,H,W,Z]`` -> ``[B,n,H]`` TILE."""
    del mesh_device
    bsz, t_len, h, w, c = (int(latents_bthwc.shape[i]) for i in range(5))
    if bsz == 1:
        return _tokens_from_latents_one(patch_embed, time_embed, latents_bthwc)

    rows: list[ttnn.Tensor] = []
    for i in range(bsz):
        lat_i = ttnn.slice(latents_bthwc, [i, 0, 0, 0, 0], [i + 1, t_len, h, w, c])
        rows.append(_tokens_from_latents_one(patch_embed, time_embed, lat_i))
        ttnn.deallocate(lat_i, force=False)
    out = ttnn.concat(rows, dim=0)
    for r in rows:
        ttnn.deallocate(r, force=False)
    return out


def _inject_row(hidden_tt: ttnn.Tensor, row: int, tokens: ttnn.Tensor, mask_row_vals: list) -> ttnn.Tensor:
    """Inject ``tokens`` ``[1,n,H]`` into batch row ``row`` at ``mask_row_vals`` spans."""
    spans = mask_to_spans(mask_row_vals)
    paired = []
    cursor = 0
    hidden_size = int(tokens.shape[2])
    for sl in spans:
        n = sl.stop - sl.start
        chunk = ttnn.slice(tokens, [0, cursor, 0], [1, cursor + n, hidden_size])
        paired.append((sl, chunk))
        cursor += n
    if row == 0 and int(hidden_tt.shape[0]) == 1:
        out = scatter_token_spans(hidden_tt, paired)
        if out is not hidden_tt:
            ttnn.deallocate(hidden_tt, force=False)
    else:
        row_hidden = ttnn.slice(hidden_tt, [row, 0, 0], [row + 1, hidden_tt.shape[1], hidden_size])
        row_out = scatter_token_spans(row_hidden, paired)
        ttnn.deallocate(row_hidden, force=False)
        pre = ttnn.slice(hidden_tt, [0, 0, 0], [row, hidden_tt.shape[1], hidden_size]) if row > 0 else None
        post = (
            ttnn.slice(hidden_tt, [row + 1, 0, 0], [hidden_tt.shape[0], hidden_tt.shape[1], hidden_size])
            if row + 1 < int(hidden_tt.shape[0])
            else None
        )
        pieces = [p for p in (pre, row_out, post) if p is not None]
        out = ttnn.concat(pieces, dim=0)
        if pre is not None:
            ttnn.deallocate(pre, force=False)
        if post is not None:
            ttnn.deallocate(post, force=False)
        ttnn.deallocate(row_out, force=False)
        ttnn.deallocate(hidden_tt, force=False)
    for _, c in paired:
        ttnn.deallocate(c, force=False)
    return out


def instantiate_vae_image_tokens_tt(
    hidden_tt: ttnn.Tensor,
    latents: ttnn.Tensor | list[ttnn.Tensor] | list[list[ttnn.Tensor]],
    image_mask,
    patch_embed: HunyuanTtUNetDown,
    time_embed: HunyuanTtTimestepEmbedder,
    mesh_device,
) -> ttnn.Tensor:
    """Patch-embed cond VAE latents on device and inject into ``hidden_tt`` on device."""
    if hidden_tt is None:
        raise ValueError("hidden_tt is required for I2I cond path")

    if isinstance(latents, ttnn.Tensor):
        tokens = _tokens_from_latents(patch_embed, time_embed, latents, mesh_device)
        bsz = int(hidden_tt.shape[0])
        out = hidden_tt
        for row in range(bsz):
            row_tok = ttnn.slice(tokens, [row, 0, 0], [row + 1, tokens.shape[1], tokens.shape[2]])
            out = _inject_row(out, row, row_tok, _mask_row(image_mask, row))
            ttnn.deallocate(row_tok, force=False)
        ttnn.deallocate(tokens, force=False)
        return out

    out = hidden_tt
    for row_i, lat_row in enumerate(latents):
        if isinstance(lat_row, ttnn.Tensor):
            toks = _tokens_from_latents(patch_embed, time_embed, lat_row, mesh_device)
            out = _inject_row(out, row_i, toks, _mask_row(image_mask, row_i))
            ttnn.deallocate(toks, force=False)
        else:
            chunks = [_tokens_from_latents(patch_embed, time_embed, lat_i, mesh_device) for lat_i in lat_row]
            toks = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=1)
            for c in chunks[1:]:
                ttnn.deallocate(c, force=False)
            out = _inject_row(out, row_i, toks, _mask_row(image_mask, row_i))
            ttnn.deallocate(toks, force=False)
    return out


def instantiate_continuous_tokens_tt(
    hidden_tt: ttnn.Tensor,
    timesteps_tt: ttnn.Tensor | torch.Tensor | list[ttnn.Tensor] | list[torch.Tensor] | None,
    timesteps_index,
    timestep_emb: HunyuanTtTimestepEmbedder,
    mesh_device,
) -> ttnn.Tensor:
    """Scatter on-device ``timestep_emb`` outputs at ``timesteps_index``."""
    del mesh_device
    if timesteps_tt is None or timesteps_index is None:
        return hidden_tt

    bsz = int(hidden_tt.shape[0])
    hidden_size = int(hidden_tt.shape[2])
    out = hidden_tt

    def _scatter_row(row: int, t_vec) -> None:
        nonlocal out
        # Logical [1,1,1,H] — not M=32 resident padding used by denoise ResBlocks.
        te = timestep_emb.forward(_timestep_for_embed(t_vec), keep_resident=False)
        emb = ttnn.reshape(te, [1, 1, hidden_size])
        ttnn.deallocate(te, force=False)
        positions = _index_row(timesteps_index, row)
        if not positions:
            ttnn.deallocate(emb, force=False)
            return
        row_hidden = ttnn.slice(out, [row, 0, 0], [row + 1, out.shape[1], hidden_size])
        paired = [(slice(pos, pos + 1), emb) for pos in positions]
        row_out = scatter_token_spans(row_hidden, paired)
        ttnn.deallocate(row_hidden, force=False)
        ttnn.deallocate(emb, force=False)
        pre = ttnn.slice(out, [0, 0, 0], [row, out.shape[1], hidden_size]) if row > 0 else None
        post = ttnn.slice(out, [row + 1, 0, 0], [bsz, out.shape[1], hidden_size]) if row + 1 < bsz else None
        pieces = [p for p in (pre, row_out, post) if p is not None]
        prev = out
        out = ttnn.concat(pieces, dim=0)
        if pre is not None:
            ttnn.deallocate(pre, force=False)
        if post is not None:
            ttnn.deallocate(post, force=False)
        ttnn.deallocate(row_out, force=False)
        if prev is not out:
            ttnn.deallocate(prev, force=False)

    if isinstance(timesteps_tt, list):
        for row, t_i in enumerate(timesteps_tt):
            _scatter_row(row, t_i)
        return out

    for row in range(bsz):
        if isinstance(timesteps_tt, torch.Tensor):
            t_row = timesteps_tt[row : row + 1] if int(timesteps_tt.shape[0]) == bsz else timesteps_tt
        else:
            # Prefer [1,1,N,1] / [N] device tensors; slice only when batched 1-D.
            if len(timesteps_tt.shape) == 1 and int(timesteps_tt.shape[0]) == bsz:
                t_row = ttnn.slice(timesteps_tt, [row], [row + 1])
            else:
                t_row = timesteps_tt
        _scatter_row(row, t_row)
        if not isinstance(timesteps_tt, torch.Tensor) and t_row is not timesteps_tt:
            ttnn.deallocate(t_row, force=False)
    return out


def scatter_distill_step_embeds_tt(
    base_embeds_tt: ttnn.Tensor,
    *,
    t_scalar: float,
    gen_timestep_scatter_index,
    timestep_emb: HunyuanTtTimestepEmbedder,
    mesh_device,
    guidance_scalar: float | None = None,
    guidance_scatter_index=None,
    guidance_emb: HunyuanTtTimestepEmbedder | None = None,
    t_r_scalar: float | None = None,
    gen_timestep_r_scatter_index=None,
    timestep_r_emb: HunyuanTtTimestepEmbedder | None = None,
) -> ttnn.Tensor:
    """Device-side ``scatter_distill_step_embeds`` (no host TimestepEmbedder / H2D of embeds).

    ``base_embeds_tt`` is consumed (may be deallocated); caller owns the returned tensor.
    Timestep scalars are created on device via ``ttnn.full`` (no ``torch.tensor``).
    """
    bsz = int(base_embeds_tt.shape[0])
    out = base_embeds_tt
    device = mesh_device if mesh_device is not None else timestep_emb.device

    def _tvec(scalar: float) -> ttnn.Tensor:
        # ttnn.full has no mesh_mapper kwarg; MeshDevice already replicates the fill.
        return ttnn.full(
            (1, 1, bsz, 1),
            fill_value=float(scalar),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if gen_timestep_scatter_index is not None and timestep_emb is not None:
        t_tt = _tvec(t_scalar)
        out = instantiate_continuous_tokens_tt(out, t_tt, gen_timestep_scatter_index, timestep_emb, mesh_device)
        ttnn.deallocate(t_tt, force=False)

    if guidance_scalar is not None and guidance_scatter_index is not None and guidance_emb is not None:
        g_tt = _tvec(guidance_scalar)
        out = instantiate_continuous_tokens_tt(out, g_tt, guidance_scatter_index, guidance_emb, mesh_device)
        ttnn.deallocate(g_tt, force=False)

    if t_r_scalar is not None and gen_timestep_r_scatter_index is not None and timestep_r_emb is not None:
        r_tt = _tvec(t_r_scalar)
        out = instantiate_continuous_tokens_tt(out, r_tt, gen_timestep_r_scatter_index, timestep_r_emb, mesh_device)
        ttnn.deallocate(r_tt, force=False)

    return out
