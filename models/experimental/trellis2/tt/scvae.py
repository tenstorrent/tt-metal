# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 SC-VAE decoders (shape and texture) on one device, as gathered-im2col matmuls (DESIGN.md D08).

Mirrors `reference/scvae_host.functional_forward`. Per level the host builds what depends on coordinates and uploads
it once: the [N, 27] neighbour map (index N = a zero row) and, at each channel-to-space step, the child gather index
(from the shape decoder's own `to_subdiv` signs, downloaded each level, or from the shape decoder's subdivisions for
the texture decoder). On device every submanifold conv is an `ttnn.embedding` gather of the 27 neighbours into
[n, 27 C] and one matmul with W [27 C, C_out], chunked over the tokens; features live as [1, 1, N_pad, C] tiles with
N_pad = round_up(N + 1, 32), and the rows from N on are zeroed before a gather so the sentinel row is a true zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.experimental.trellis2.reference import scvae_host as host
from models.experimental.trellis2.tt.structure_flow import Precision, _hifi4, _norm_in

GATHER_ELEMS = 64 * 1024 * 1024  # bf16 elements per gathered chunk (~128 MB)


def _pad_rows(n: int) -> int:
    return -(-(n + 1) // 32) * 32


@dataclass
class Level:
    """Device-side tables of one decoder level."""

    n: int
    n_pad: int
    coords: torch.Tensor
    nbr: ttnn.Tensor  # [1, 1, 1, n_pad * 27] uint32, rows past n point at the sentinel
    row_mask: ttnn.Tensor  # [1, 1, n_pad, 1] 1 for real rows, 0 for padding


class TtSCVAEDecoder:
    def __init__(
        self, state: dict[str, torch.Tensor], *, mesh_device: ttnn.MeshDevice, precision: Precision = Precision()
    ):
        self.device = mesh_device
        self.mm = _hifi4(mesh_device) if precision.linear_hifi4 else None
        self.norm_config = _hifi4(mesh_device)
        self.s = {k: v.float() for k, v in state.items()}
        self.pred_subdiv = "blocks.0.4.to_subdiv.weight" in self.s
        self._w: dict[str, ttnn.Tensor] = {}

    # --- weights -------------------------------------------------------------------------------------------------
    def _tile(self, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _linear_w(self, p: str) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if p not in self._w:
            self._w[p] = (
                self._tile(self.s[p + ".weight"].T.contiguous()),
                self._tile(self.s[p + ".bias"].reshape(1, -1)),
            )
        return self._w[p]

    def _conv_w(self, p: str) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if p not in self._w:
            self._w[p] = (
                self._tile(host.conv_weight(self.s[p + ".weight"])),
                self._tile(self.s[p + ".bias"].reshape(1, -1)),
            )
        return self._w[p]

    def _norm_w(self, p: str) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if p not in self._w:
            self._w[p] = (
                self._tile(self.s[p + ".weight"].reshape(1, -1)),
                self._tile(self.s[p + ".bias"].reshape(1, -1)),
            )
        return self._w[p]

    # --- ops -----------------------------------------------------------------------------------------------------
    def _linear(self, x: ttnn.Tensor, p: str, dtype=None) -> ttnn.Tensor:
        w, b = self._linear_w(p)
        return ttnn.linear(_norm_in(x), w, bias=b, compute_kernel_config=self.mm, dtype=dtype)

    def _ln(self, x: ttnn.Tensor, p: str | None, eps: float = host.BLOCK_NORM_EPS) -> ttnn.Tensor:
        w, b = self._norm_w(p) if p is not None else (None, None)
        return ttnn.layer_norm(x, weight=w, bias=b, epsilon=eps, compute_kernel_config=self.norm_config)

    def _gather(self, table: ttnn.Tensor, idx: ttnn.Tensor) -> ttnn.Tensor:
        """table [1, 1, V, C] row-major bf16, idx [1, 1, 1, S] uint32 -> [1, 1, S, C] row-major."""
        out = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.reshape(out, (1, 1, idx.shape[-1], table.shape[-1]))

    def _conv(self, x: ttnn.Tensor, level: Level, p: str) -> ttnn.Tensor:
        """Submanifold 3^3 conv on [1, 1, n_pad, C] tiles: zero the padding rows, gather [n, 27 C], one matmul."""
        w, b = self._conv_w(p)
        c = x.shape[-1]
        x = ttnn.multiply(x, level.row_mask)
        table = ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        rows = max(32, GATHER_ELEMS // (27 * c) // 32 * 32)
        chunks = []
        for s in range(0, level.n_pad, rows):
            r = min(rows, level.n_pad - s)
            g = self._gather(table, ttnn.slice(level.nbr, (0, 0, 0, s * 27), (1, 1, 1, (s + r) * 27)))
            g = ttnn.to_layout(ttnn.reshape(g, (1, 1, r, 27 * c)), ttnn.TILE_LAYOUT)
            chunks.append(ttnn.linear(g, w, bias=b, compute_kernel_config=self.mm, dtype=ttnn.float32))
        return chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2)

    def _convnext(self, h: ttnn.Tensor, level: Level, p: str) -> ttnn.Tensor:
        a = self._ln(self._conv(h, level, p + "conv"), p + "norm")
        a = self._linear(ttnn.silu(self._linear(a, p + "mlp.0")), p + "mlp.2", dtype=ttnn.float32)
        return ttnn.add(h, a)

    def _level(self, coords: torch.Tensor, res: int) -> Level:
        n = coords.shape[0]
        n_pad = _pad_rows(n)
        nbr = torch.full((n_pad, 27), n, dtype=torch.int32)
        nbr[:n] = host.neighbor_map(coords, res)
        mask = torch.zeros(1, 1, n_pad, 1)
        mask[0, 0, :n] = 1
        return Level(
            n=n,
            n_pad=n_pad,
            coords=coords,
            nbr=ttnn.from_torch(
                nbr.reshape(1, 1, 1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            ),
            row_mask=self._tile(mask, ttnn.float32),
        )

    def _rows_gather(self, x: ttnn.Tensor, width: int, rows: torch.Tensor, m_pad: int) -> ttnn.Tensor:
        """x [1, 1, n_pad, 8 * width] viewed as [8 n_pad, width]; gather `rows` (padded to m_pad with row 0) -> tile
        [1, 1, m_pad, width]. Widths under 32 are padded to 32 for the gather and sliced back."""
        n_pad = x.shape[2]
        t = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), (1, 1, n_pad * 8, width)
        )
        if width % 32:
            t = ttnn.pad(t, [(0, 0), (0, 0), (0, 0), (0, 32 - width % 32)], value=0.0)
        idx = torch.zeros(m_pad, dtype=torch.int32)
        idx[: rows.shape[0]] = rows
        idx = ttnn.from_torch(
            idx.reshape(1, 1, 1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        g = self._gather(t, idx)
        if width % 32:
            g = ttnn.slice(g, (0, 0, 0, 0), (1, 1, m_pad, width))
        return ttnn.to_layout(g, ttnn.TILE_LAYOUT)

    def _c2s(self, h: ttnn.Tensor, level: Level, keep: torch.Tensor, res: int, p: str) -> tuple[ttnn.Tensor, Level]:
        c = h.shape[-1]
        c_next = self.s[p + "conv2.weight"].shape[0]
        kids = host.children(level.coords, keep)
        fine = self._level(kids.coords, res)
        a = self._conv(ttnn.silu(self._ln(h, p + "norm1")), level, p + "conv1")
        a = self._rows_gather(a, c_next, kids.rows, fine.n_pad)
        skip = self._rows_gather(h, c // 8, kids.rows, fine.n_pad)
        skip = ttnn.repeat_interleave(ttnn.typecast(skip, ttnn.float32), c_next // (c // 8), dim=3)
        a = self._conv(ttnn.silu(self._ln(a, None)), fine, p + "conv2")
        return ttnn.add(a, skip), fine

    # --- forward -------------------------------------------------------------------------------------------------
    def __call__(self, feats: torch.Tensor, coords: torch.Tensor, guide_keep: list[torch.Tensor] | None = None):
        """Torch in, torch out, as `host.functional_forward`: (out [M, C_out], coords [M, 4], subdivision logits per
        level for the shape decoder or None)."""
        res = host.START_RESOLUTION
        level = self._level(coords, res)
        x = torch.zeros(1, 1, level.n_pad, feats.shape[1])
        x[0, 0, : level.n] = feats.float()
        h = ttnn.typecast(self._linear(self._tile(x), "from_latent"), ttnn.float32)
        subs = []
        for i, nb in enumerate(host.NUM_BLOCKS):
            for j in range(nb):
                h = self._convnext(h, level, f"blocks.{i}.{j}.")
            if i == len(host.NUM_BLOCKS) - 1:
                break
            p = f"blocks.{i}.{nb}."
            if guide_keep is None:
                logits = ttnn.to_torch(ttnn.get_device_tensors(self._linear(h, p + "to_subdiv", dtype=ttnn.float32))[0])
                logits = logits.float()[0, 0, : level.n, :8]
                subs.append(logits)
                keep = logits > 0
            else:
                keep = guide_keep[i]
            res *= 2
            h, level = self._c2s(h, level, keep, res, p)
        out = self._linear(self._ln(h, None, eps=host.TAIL_NORM_EPS), "output_layer", dtype=ttnn.float32)
        out = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[
            0, 0, : level.n, : self.s["output_layer.weight"].shape[0]
        ]
        return out, level.coords, (subs if guide_keep is None else None)
