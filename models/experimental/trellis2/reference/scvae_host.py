# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side pieces of the TRELLIS.2 SC-VAE decoders (shape: `FlexiDualGridVaeDecoder`, texture: `SparseUnetVaeDecoder`;
`models/sc_vaes/`, microsoft/TRELLIS.2 75fbf018), and a torch mirror of the TT decomposition (DESIGN.md D08).

Both decoders: `from_latent` Linear 32 -> 1024, then 5 levels at channels [1024, 512, 256, 128, 64] with
[4, 16, 8, 4, 0] ConvNeXt blocks (submanifold conv 3^3, LayerNorm eps 1e-6, Linear 4x, SiLU, Linear, + residual) and a
channel-to-space (C2S) block between levels, then LayerNorm (no affine) and a Linear to 7 (shape) or 6 (texture).

The host builds, per level, what depends on coordinates: the [N, 27] neighbour map of the submanifold conv (index N
is a zero sentinel row for a missing neighbour) and the child index of the C2S step (from the shape decoder's
`to_subdiv` signs, or the shape decoder's subdivisions for the texture decoder). Everything else is a gather plus
dense math: each conv is a gather of the 27 neighbours into [N, 27 C] and one matmul with W [27 C, C_out]
(`conv_gather`), chunked over N; the C2S step gathers rows `parent * 8 + k` of the conv output viewed as [8 N, C_next].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

CHANNELS = (1024, 512, 256, 128, 64)
NUM_BLOCKS = (4, 16, 8, 4, 0)
LATENT_CHANNELS = 32
START_RESOLUTION = 32
BLOCK_NORM_EPS = 1e-6
TAIL_NORM_EPS = 1e-5
SHAPE_CKPT = "TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
TEX_CKPT = "TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
GATHER_ROWS = 65536


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


def offsets() -> torch.Tensor:
    """The 27 offsets of a 3^3 kernel in the checkpoint's (K0, K1, K2) = ij order over (x, y, z)."""
    a = torch.arange(-1, 2)
    return torch.stack(torch.meshgrid(a, a, a, indexing="ij"), dim=-1).reshape(-1, 3)


def neighbor_map(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    """[N, 27] int32: the row index of coords + offset_k, or N (a zero sentinel row) when that voxel is inactive."""
    c = coords.long()
    r = resolution
    keys = ((c[:, 0] * r + c[:, 1]) * r + c[:, 2]) * r + c[:, 3]
    sorted_keys, order = torch.sort(keys)
    n = c.shape[0]
    out = torch.empty(n, 27, dtype=torch.int32)
    for k, (dx, dy, dz) in enumerate(offsets().tolist()):
        x, y, z = c[:, 1] + dx, c[:, 2] + dy, c[:, 3] + dz
        valid = (x >= 0) & (x < r) & (y >= 0) & (y < r) & (z >= 0) & (z < r)
        q = ((c[:, 0] * r + x) * r + y) * r + z
        pos = torch.searchsorted(sorted_keys, q).clamp_(max=n - 1)
        hit = valid & (sorted_keys[pos] == q)
        out[:, k] = torch.where(hit, order[pos], torch.full_like(pos, n)).int()
    return out


@dataclass
class Children:
    """Output of a C2S step: child coords [M, 4] and the gather index into the parent rows viewed as [8 N, C]."""

    coords: torch.Tensor
    rows: torch.Tensor


def children(coords: torch.Tensor, keep: torch.Tensor) -> Children:
    """keep [N, 8] bool -> children in the reference's order (parent-major, child k ascending); child k of a parent
    sits at 2 * coords + (k & 1, k >> 1 & 1, k >> 2 & 1) (`SparseChannel2Spatial`)."""
    parent, k = keep.nonzero(as_tuple=True)
    new = coords[parent].clone()
    new[:, 1:] *= 2
    for i in range(3):
        new[:, i + 1] += (k >> i) & 1
    return Children(coords=new, rows=(parent * 8 + k).int())


def conv_weight(w: torch.Tensor) -> torch.Tensor:
    """(C_out, 3, 3, 3, C_in) -> [27 C_in, C_out] for the gathered matmul."""
    return w.reshape(w.shape[0], -1).T.contiguous()


def conv_gather(x: torch.Tensor, nbr: torch.Tensor, w: torch.Tensor, b: torch.Tensor, rows: int = GATHER_ROWS):
    """Submanifold conv as the TT path does it: gather [n, 27, C] from x plus a zero row, one matmul, chunked over N."""
    table = torch.cat([x, x.new_zeros(1, x.shape[1])])
    wm = conv_weight(w)
    out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype)
    for s in range(0, x.shape[0], rows):
        g = table[nbr[s : s + rows].long()].reshape(-1, wm.shape[0])
        out[s : s + rows] = g @ wm + b
    return out


def _ln(x: torch.Tensor, s: dict, p: str | None, eps: float = BLOCK_NORM_EPS) -> torch.Tensor:
    if p is None:
        return F.layer_norm(x, (x.shape[-1],), eps=eps)
    return F.layer_norm(x, (x.shape[-1],), s[p + ".weight"], s[p + ".bias"], eps=eps)


def _linear(x: torch.Tensor, s: dict, p: str) -> torch.Tensor:
    return F.linear(x, s[p + ".weight"], s[p + ".bias"])


def convnext(h: torch.Tensor, nbr: torch.Tensor, s: dict, p: str) -> torch.Tensor:
    a = conv_gather(h, nbr, s[p + "conv.weight"], s[p + "conv.bias"])
    a = _linear(F.silu(_linear(_ln(a, s, p + "norm"), s, p + "mlp.0")), s, p + "mlp.2")
    return h + a


def c2s(h: torch.Tensor, nbr_coarse: torch.Tensor, kids: Children, resolution: int, s: dict, p: str):
    """The C2S block given its children (coords at the fine level): returns fine features and the fine neighbour map."""
    c, c_next = h.shape[1], s[p + "conv2.weight"].shape[0]
    a = conv_gather(F.silu(_ln(h, s, p + "norm1")), nbr_coarse, s[p + "conv1.weight"], s[p + "conv1.bias"])
    a = a.reshape(-1, c_next)[kids.rows.long()]
    skip = h.reshape(-1, c // 8)[kids.rows.long()].repeat_interleave(c_next // (c // 8), dim=1)
    nbr_fine = neighbor_map(kids.coords, resolution)
    a = conv_gather(F.silu(_ln(a, s, None)), nbr_fine, s[p + "conv2.weight"], s[p + "conv2.bias"])
    return a + skip, nbr_fine


def functional_forward(
    state: dict[str, torch.Tensor],
    feats: torch.Tensor,
    coords: torch.Tensor,
    guide_keep: list[torch.Tensor] | None = None,
):
    """Torch mirror of the TT decoder in fp32. Shape decoder: guide_keep None, the children come from its own
    `to_subdiv` logits; texture decoder: guide_keep[i] = the shape decoder's logits > 0 at level i.
    Returns (output feats [M, C_out], coords [M, 4], subdivision logits per level or None)."""
    s = {k: v.float() for k, v in state.items()}
    h = _linear(feats.float(), s, "from_latent")
    res = START_RESOLUTION
    nbr = neighbor_map(coords, res)
    subs = []
    for i, nb in enumerate(NUM_BLOCKS):
        for j in range(nb):
            h = convnext(h, nbr, s, f"blocks.{i}.{j}.")
        if i == len(NUM_BLOCKS) - 1:
            break
        p = f"blocks.{i}.{nb}."
        if guide_keep is None:
            logits = _linear(h, s, p + "to_subdiv")
            subs.append(logits)
            keep = logits > 0
        else:
            keep = guide_keep[i]
        kids = children(coords, keep)
        res *= 2
        h, nbr = c2s(h, nbr, kids, res, s, p)
        coords = kids.coords
    out = _linear(_ln(h, s, None, eps=TAIL_NORM_EPS), s, "output_layer")
    return out, coords, (subs if guide_keep is None else None)
