# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chunk DYNAMIC inputs of the traced chunked prefill.

A prompt is processed in chunks of a fixed C tokens per user; ONE trace of the whole chunk forward is captured and replayed for every chunk.
Everything that differs from chunk to chunk lives in persistent device tensors owned by ``DynCtx`` and refreshed with
``copy_host_to_device_tensor`` before each replay (``update``):

  * RoPE tables of the chunk's token positions (and of the latent positions of ratio-2 layers),
  * the window band mask [C, 128 + C] (the 128 halo rows of chunk 0 are masked out: there is no previous chunk),
  * ``lim`` / ``off`` of the compressed latents: the latents of all closed groups live in a FIFO buffer of fixed length ``Lmax`` per kv-source
    layer (the newest chunk's latents are appended at the END, older ones shift towards the front); buffer slot p holds latent j = p - off with
    off = Lmax - (s0 + C) / r. A query at absolute position t sees j in [0, (t + 1) / r): the latent half of the mask is computed ON the device
    (``latent_mask``) from these two small tensors, so the captured program is identical for every chunk.
"""

import os
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import h2d

NEG = -1e9
HEAD_DIM = 512
WINDOW = 128


LATENT_MASK_COLS = 32768  # column chunk of the [C, L] mask: one eltwise row wider than ~64k tiles' worth of columns needs more static L1 than exists (L > ~131k at ctx > 262144)


def _latent_mask_cols(pcol, off, lim, C, L):
    j = ttnn.subtract(pcol, off)  # [1,1,1,L] latent index of every buffer slot
    ge0 = ttnn.ge(j, 0.0)
    lt = ttnn.lt(ttnn.repeat(j, [1, 1, C, 1]), ttnn.repeat(lim, [1, 1, 1, L]))
    vis = ttnn.multiply(lt, ttnn.repeat(ge0, [1, 1, C, 1]))
    return ttnn.typecast(ttnn.multiply(ttnn.subtract(vis, 1.0), -NEG), ttnn.bfloat16)


def latent_mask_parts(pcol, off, lim, C, L):
    """The additive mask [1,1,C,L] bf16 (0 where 0 <= p - off < lim[row], else -1e9; pcol [1,1,1,L] fp32 = arange(L)) as a list of column chunks of
    at most LATENT_MASK_COLS (identical values): a concat input wider than that overflows L1 at long contexts."""
    if L <= LATENT_MASK_COLS:
        return [_latent_mask_cols(pcol, off, lim, C, L)]
    parts = []
    for c0 in range(0, L, LATENT_MASK_COLS):
        c1 = min(L, c0 + LATENT_MASK_COLS)
        pc = ttnn.slice(pcol, [0, 0, 0, c0], [1, 1, 1, c1])
        parts.append(_latent_mask_cols(pc, off, lim, C, c1 - c0))
    return parts


def latent_mask(pcol, off, lim, C, L):
    parts = latent_mask_parts(pcol, off, lim, C, L)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=3)
    for t in parts:
        ttnn.deallocate(t)
    return out


def visibility_mask(pcol, off, lim, C, L):
    """[1,1,C,L] bf16 with 1 where 0 <= p - off < lim[row] else 0 (the visibility of the sparse indexer scores), built in column chunks of at most LATENT_MASK_COLS: the fp32
    temporaries (two repeats of [C, L]) are ~4 GiB each at a 1M context, ~64 MiB per chunk here; identical values."""
    parts = []
    for c0 in range(0, L, LATENT_MASK_COLS):
        c1 = min(L, c0 + LATENT_MASK_COLS)
        pc = pcol if (c0, c1) == (0, L) else ttnn.slice(pcol, [0, 0, 0, c0], [1, 1, 1, c1])
        j = ttnn.subtract(pc, off)
        ge0 = ttnn.ge(j, 0.0)
        lt = ttnn.lt(ttnn.repeat(j, [1, 1, C, 1]), ttnn.repeat(lim, [1, 1, 1, c1 - c0]))
        parts.append(ttnn.typecast(ttnn.multiply(lt, ttnn.repeat(ge0, [1, 1, C, 1])), ttnn.bfloat16))
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=3)
    for t in parts:
        ttnn.deallocate(t)
    return out


class DynCtx:
    def __init__(self, md, C, S_pad, ratios, rope_src):
        """ratios: set of compress ratios present (0 = window only); rope_src: {compressed?: an attention object providing ``_rope_inputs``}."""
        self.md, self.C, self.S_pad = md, C, S_pad
        self.rope_src = rope_src
        self.ratios = sorted(r for r in ratios if r)
        self.L = {r: S_pad // r for r in self.ratios}
        rep = ttnn.ReplicateTensorToMesh(md)
        self._rep = rep

        def mk(shape, dtype=ttnn.bfloat16, fill=0.0):
            return ttnn.from_torch(
                torch.full(shape, fill),
                device=md,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        self.tabs = {k: tuple(mk([1, 1, C, HEAD_DIM]) for _ in range(3)) for k in rope_src}  # key: compressed?
        self.lat_tabs = {
            r: tuple(mk([1, 1, C // r, HEAD_DIM]) for _ in range(3)) for r in self.ratios if r > 1 and True
        }
        self.win_mask = mk([1, 1, C, WINDOW + C])
        self.lim = {r: mk([1, 1, C, 1], ttnn.float32) for r in self.ratios}
        self.off = {r: mk([1, 1, 1, 1], ttnn.float32) for r in self.ratios}
        self.pcol = {
            r: ttnn.from_torch(
                torch.arange(self.L[r], dtype=torch.float32).reshape(1, 1, 1, -1),
                device=md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for r in self.ratios
        }
        self.upd_hooks, self.build_hooks = (
            [],
            [],
        )  # extra per-chunk host updates (``hook(ctx, s0)``) / in-trace builders (``hook(ctx)``): tt/prefill_sparse.py
        self.masks = {}  # ratio -> full mask [1,1,C,128+C+L] built inside the traced forward by ``build_masks``

    def _host(self, t, dtype):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._rep)

    def update(self, s0):
        """Refresh every per-chunk tensor for the chunk of positions [s0, s0 + C) (host work + uploads, nothing is enqueued on the compute path)."""
        C = self.C
        up = h2d
        pos = s0 + torch.arange(C)
        for k, src in self.rope_src.items():
            c, s = src._rope_inputs(pos)
            for t, dev in zip((c, s, -s), self.tabs[k]):
                up(self._host(t.reshape(1, 1, C, HEAD_DIM), ttnn.bfloat16), dev)
        for r, tabs in self.lat_tabs.items():
            src = self.rope_src[True]
            c, s = src._rope_inputs(s0 + r * torch.arange(C // r))
            for t, dev in zip((c, s, -s), tabs):
                up(self._host(t.reshape(1, 1, C // r, HEAD_DIM), ttnn.bfloat16), dev)
        t = pos.view(-1, 1)
        kp = torch.cat([s0 - WINDOW + torch.arange(WINDOW), pos]).view(1, -1)
        up(
            self._host(
                torch.where((kp <= t) & (kp > t - WINDOW) & (kp >= 0), 0.0, NEG).reshape(1, 1, C, WINDOW + C),
                ttnn.bfloat16,
            ),
            self.win_mask,
        )
        for r in self.ratios:
            up(self._host(((pos + 1) // r).float().reshape(1, 1, C, 1), ttnn.float32), self.lim[r])
            up(self._host(torch.full((1, 1, 1, 1), float(self.L[r] - (s0 + C) // r)), ttnn.float32), self.off[r])
        t0 = time.perf_counter()
        for hk in self.upd_hooks:
            hk(self, s0)
        self.t_hooks = getattr(self, "t_hooks", 0.0) + time.perf_counter() - t0

    def build_masks(self):
        """Inside the traced forward, once per chunk: the full additive masks [1,1,C,128+C+L] of every ratio that a DENSE attention layer reads. A ratio whose layers all run the sparse path
        (``sp_win`` holds the ratio: indexer top-512 + sparse_sdpa with its own visibility mask ``sp_vis``) never reads the dense mask, so it is not built (DSV41_PF_DENSE_MASKS=1 builds all of
        them as before); a dense layer of such a ratio still gets one on demand from ``mask_for``."""
        self.masks = (
            {}
        )  # per chunk (eager chunks rebuild; a trace captures the ops once): drop the previous chunk's masks as the old overwrite did
        for r in self.ratios:
            if os.environ.get("DSV41_PF_DENSE_MASKS", "0") == "1" or r not in getattr(self, "sp_win", {}):
                self.mask_for(r)
        for hk in self.build_hooks:
            hk(self)

    def mask_for(self, r):
        """The dense mask [1,1,C,128+C+L] of ratio ``r`` (built once per forward, on first use)."""
        if r not in self.masks:
            parts = latent_mask_parts(self.pcol[r], self.off[r], self.lim[r], self.C, self.L[r])
            self.masks[r] = ttnn.concat([self.win_mask] + parts, dim=3)
            for t in parts:
                ttnn.deallocate(t)
        return self.masks[r]
