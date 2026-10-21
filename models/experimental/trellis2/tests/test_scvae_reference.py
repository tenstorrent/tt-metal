# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU only: the building blocks of the SC-VAE decoder mirror (scvae_host, DESIGN.md D08) on small random inputs.

- conv_gather (neighbour map with a zero sentinel row, [N, 27 C] gather, one matmul, chunked) == a dense conv3d with
  zero padding evaluated at the active voxels, with inactive voxels zeroed (a submanifold conv);
- children() gives the reference SparseChannel2Spatial order and coords.

The whole-decoder check against the fp32 golden is `scripts/tt/scvae_mirror_cpu.py` in the project folder (S09: PCC
1.0 on the golden topology for both decoders and both inputs).
"""

from __future__ import annotations

import torch

from models.experimental.trellis2.reference import scvae_host as sc


def _random_voxels(res: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(res**3, generator=g)[:n]
    xyz = torch.stack([flat // (res * res), flat // res % res, flat % res], dim=1)
    return torch.cat([torch.zeros(n, 1, dtype=torch.long), xyz], dim=1).int()


def test_conv_gather_is_submanifold_conv():
    torch.manual_seed(0)
    res, n, c_in, c_out = 8, 150, 5, 7
    coords = _random_voxels(res, n)
    x = torch.randn(n, c_in)
    w, b = torch.randn(c_out, 3, 3, 3, c_in), torch.randn(c_out)
    out = sc.conv_gather(x, sc.neighbor_map(coords, res), w, b, rows=64)

    dense = torch.zeros(1, c_in, res, res, res)
    dense[0, :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()] = x.T
    y = torch.nn.functional.conv3d(dense, w.permute(0, 4, 1, 2, 3), b, padding=1)
    ref = y[0, :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()].T
    assert torch.allclose(out, ref, atol=1e-4)


def test_children_match_channel2spatial():
    coords = _random_voxels(4, 10, seed=1)
    keep = torch.rand(10, 8, generator=torch.Generator().manual_seed(2)) > 0.5
    kids = sc.children(coords, keep)
    # reference algorithm (trellis2/modules/sparse/spatial/spatial2channel.py SparseChannel2Spatial.forward)
    n_leaf = keep.sum(-1)
    subidx = keep.nonzero()[:, -1]
    ref = coords.clone()
    ref[:, 1:] *= 2
    ref = torch.repeat_interleave(ref, n_leaf, dim=0)
    for i in range(3):
        ref[:, i + 1] += subidx // 2**i % 2
    idx = torch.repeat_interleave(torch.arange(10), n_leaf)
    assert torch.equal(kids.coords, ref)
    assert torch.equal(kids.rows.long(), idx * 8 + subidx)
