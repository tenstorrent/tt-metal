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

import torch

import ttnn

NEG = -1e9
HEAD_DIM = 512
WINDOW = 128


def latent_mask(pcol, off, lim, C, L):
    """Additive mask [1,1,C,L] bf16 on the device: 0 where 0 <= p - off < lim[row], else -1e9. pcol [1,1,1,L] fp32 = arange(L)."""
    j = ttnn.subtract(pcol, off)  # [1,1,1,L] latent index of every buffer slot
    ge0 = ttnn.ge(j, 0.0)
    lt = ttnn.lt(ttnn.repeat(j, [1, 1, C, 1]), ttnn.repeat(lim, [1, 1, 1, L]))
    vis = ttnn.multiply(lt, ttnn.repeat(ge0, [1, 1, C, 1]))
    return ttnn.typecast(ttnn.multiply(ttnn.subtract(vis, 1.0), -NEG), ttnn.bfloat16)


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
        self.masks = {}  # ratio -> full mask [1,1,C,128+C+L] built inside the traced forward by ``build_masks``

    def _host(self, t, dtype):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._rep)

    def update(self, s0):
        """Refresh every per-chunk tensor for the chunk of positions [s0, s0 + C) (host work + uploads, nothing is enqueued on the compute path)."""
        C = self.C
        up = lambda host, dev: ttnn.copy_host_to_device_tensor(host, dev)
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

    def build_masks(self):
        """Inside the traced forward, once per chunk: the full additive masks [1,1,C,128+C+L] of every ratio."""
        for r in self.ratios:
            lm = latent_mask(self.pcol[r], self.off[r], self.lim[r], self.C, self.L[r])
            self.masks[r] = ttnn.concat([self.win_mask, lm], dim=3)
            ttnn.deallocate(lm)
