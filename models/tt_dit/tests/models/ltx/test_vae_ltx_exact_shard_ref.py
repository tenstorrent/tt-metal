# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of the LTX decoder exact-shard rebalance (LTX_VAE_EXACT_SHARD=1).

Runs the real _reshard_exact_hw on a fake mesh tensor (one torch shard per chip; all_gather, slice,
reshape and mesh_partition emulated per chip). After the rebalance each chip must hold exactly its
logical/factor block of the unpadded tensor, and the per-chip conv shapes must match the blocking
keys from _compute_ltx_decoder_dims.
Run: python -m pytest --noconftest <this file>
"""

import pytest
import torch

import ttnn
from models.tt_dit.models.vae import vae_ltx
from models.tt_dit.parallel.config import VaeHWParallelConfig

# Production LTX-2 decoder_blocks (encoder order; the decoder walks them reversed).
_PROD_DECODER_BLOCKS = [
    ("res_x", {"num_layers": 4}),
    ("compress_space", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
    ("compress_all", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
]


class _FakeMeshTensor:
    def __init__(self, shards: dict[tuple[int, int], torch.Tensor]):
        self.shards = shards
        shapes = {tuple(s.shape) for s in shards.values()}
        assert len(shapes) == 1, f"ragged shards {shapes}"

    @property
    def shape(self):
        return next(iter(self.shards.values())).shape

    def __getitem__(self, idx):
        return _FakeMeshTensor({k: v[idx] for k, v in self.shards.items()})


class _FakeCCL:
    def __init__(self, mesh_shape):
        self.mesh_shape = mesh_shape
        self.gathers = 0

    def all_gather(self, x, *, dim, mesh_axis, use_hyperparams):
        assert len(x.shape) == 4
        self.gathers += 1
        out = {}
        for coord in x.shards:
            peers = []
            for i in range(self.mesh_shape[mesh_axis]):
                c = list(coord)
                c[mesh_axis] = i
                peers.append(x.shards[tuple(c)])
            out[coord] = torch.cat(peers, dim=dim)
        return _FakeMeshTensor(out)


def _fake_mesh_ops(monkeypatch, mesh_shape):
    def reshape(x, shape):
        return _FakeMeshTensor({k: v.reshape(shape) for k, v in x.shards.items()})

    def mesh_partition(x, *, dim, cluster_axis):
        n = mesh_shape[cluster_axis]
        assert x.shape[dim] % n == 0
        return _FakeMeshTensor({k: v.chunk(n, dim=dim)[k[cluster_axis]] for k, v in x.shards.items()})

    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn, "mesh_partition", mesh_partition)


def _shard(x_BTHWC, mesh_shape):
    rows, cols = mesh_shape
    h, w = x_BTHWC.shape[2] // rows, x_BTHWC.shape[3] // cols
    return _FakeMeshTensor(
        {
            (r, c): x_BTHWC[:, :, r * h : (r + 1) * h, c * w : (c + 1) * w].clone()
            for r in range(rows)
            for c in range(cols)
        }
    )


def _config(mesh_shape):
    return VaeHWParallelConfig.from_tuples(height=(mesh_shape[0], 0), width=(mesh_shape[1], 1))


@pytest.mark.parametrize(
    "mesh_shape, logical_hw, padded_hw",
    [
        ((4, 8), (68, 120), (72, 128)),  # 1080p after s0 on 4x8
        ((2, 4), (34, 60), (36, 64)),  # 544x960 after s0 on 2x4
        ((2, 4), (34, 64), (36, 64)),  # W already exact
    ],
    ids=["1080p_4x8", "544x960_2x4", "h_only"],
)
def test_reshard_matches_exact_split(monkeypatch, mesh_shape, logical_hw, padded_hw):
    _fake_mesh_ops(monkeypatch, mesh_shape)
    torch.manual_seed(0)
    lh, lw = logical_hw
    B, T, C = 1, 3, 5
    full = torch.full((B, T, *padded_hw, C), float("nan"))
    full[:, :, :lh, :lw] = torch.randn(B, T, lh, lw, C)
    ccl = _FakeCCL(mesh_shape)

    out = vae_ltx._reshard_exact_hw(_shard(full, mesh_shape), lh, lw, _config(mesh_shape), ccl)

    expected = _shard(full[:, :, :lh, :lw], mesh_shape)
    assert ccl.gathers == sum(p != l for p, l in zip(padded_hw, logical_hw))
    for coord, shard in expected.shards.items():
        assert torch.equal(out.shards[coord], shard), coord


@pytest.mark.parametrize(
    "mesh_shape, logical_hw",
    [
        ((2, 4), (34, 60)),  # already exact
        ((4, 8), (34, 60)),  # latent on 4x8: does not divide yet
    ],
    ids=["exact", "indivisible"],
)
def test_reshard_skips(monkeypatch, mesh_shape, logical_hw):
    _fake_mesh_ops(monkeypatch, mesh_shape)
    rows, cols = mesh_shape
    padded = (-(-logical_hw[0] // rows) * rows, -(-logical_hw[1] // cols) * cols)
    x = _shard(torch.randn(1, 1, *padded, 2), mesh_shape)
    ccl = _FakeCCL(mesh_shape)
    assert vae_ltx._reshard_exact_hw(x, *logical_hw, _config(mesh_shape), ccl) is x
    assert ccl.gathers == 0


def _runtime_conv_dims(mesh_shape, height, width, exact, monkeypatch):
    """Per-chip (H, W) each decoder conv site sees, walking decode_device's shape changes."""
    _fake_mesh_ops(monkeypatch, mesh_shape)
    rows, cols = mesh_shape
    lh, lw = height // 32, width // 32
    x = _shard(torch.zeros(1, 1, -(-lh // rows) * rows, -(-lw // cols) * cols, 1), mesh_shape)
    ccl = _FakeCCL(mesh_shape)
    dims = [tuple(x.shape[2:4])]  # conv_in
    for name, _ in reversed(_PROD_DECODER_BLOCKS):
        dims.append(tuple(x.shape[2:4]))
        if name in vae_ltx._DECODER_STRIDE_MAP:
            _, p2, p3 = vae_ltx._DECODER_STRIDE_MAP[name]
            x = _FakeMeshTensor(
                {k: v.repeat_interleave(p2, dim=2).repeat_interleave(p3, dim=3) for k, v in x.shards.items()}
            )
            lh, lw = lh * p2, lw * p3
            if exact:
                x = vae_ltx._reshard_exact_hw(x, lh, lw, _config(mesh_shape), ccl)
    dims.append(tuple(x.shape[2:4]))  # conv_out
    return dims


@pytest.mark.parametrize(
    "mesh_shape, height, width",
    [((4, 8), 1088, 1920), ((2, 4), 544, 960), ((2, 4), 1088, 1920)],
    ids=["1080p_4x8", "544x960_2x4", "1080p_2x4"],
)
def test_runtime_dims_match_blocking_keys(monkeypatch, mesh_shape, height, width):
    keys = vae_ltx._compute_ltx_decoder_dims(
        decoder_blocks=_PROD_DECODER_BLOCKS,
        num_frames=145,
        height=height,
        width=width,
        h_factor=mesh_shape[0],
        w_factor=mesh_shape[1],
    )
    key_hw = [(d.H, d.W) for d in keys]
    assert _runtime_conv_dims(mesh_shape, height, width, True, monkeypatch) == key_hw
    padded = _runtime_conv_dims(mesh_shape, height, width, False, monkeypatch)
    if height // 32 % mesh_shape[0] or width // 32 % mesh_shape[1]:
        # Without the rebalance the doubled padding puts every stage after s0 off its key.
        assert padded[:3] == key_hw[:3] and all(p != k for p, k in zip(padded[3:], key_hw[3:]))
    else:
        assert padded == key_hw
