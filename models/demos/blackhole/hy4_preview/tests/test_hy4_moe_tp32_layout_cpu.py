# SPDX-License-Identifier: Apache-2.0
"""CPU test of the X40 TP32 layout + kernel-dataflow emulation (small expert count, random weights; the layout has no dependence on E except strides)."""
import torch

from models.demos.blackhole.hy4_preview.tt import hy4_moe_tp32_layout as L


def test_tp32_dataflow_matches_dense():
    torch.manual_seed(0)
    E, T = 12, 3
    w0 = torch.randn(E, L.HIDDEN, L.INTER) * 0.02
    w1 = torch.randn(E, L.HIDDEN, L.INTER) * 0.02
    w2 = torch.randn(E, L.INTER, L.HIDDEN) * 0.02
    x = torch.randn(T, L.HIDDEN)
    ids = torch.stack([torch.randperm(E)[:8] for _ in range(T)])
    sc = torch.rand(T, 8)
    for c in (
        0,
        5,
        31,
    ):  # chips are independent; the dense result is the sum over the 32 slices (act is elementwise in the intermediate dim)
        w01, w2s = L.pack_chip(w0, w1, w2, c)
        part = L.emulate_chip(x, ids, sc, w01, w2s, E=E)
        sl = slice(L.ICOL * c, L.ICOL * (c + 1))
        ref = L.reference(x, ids, sc, w0[:, :, sl], w1[:, :, sl], w2[:, sl, :])
        rel = (part - ref).norm() / ref.norm()
        assert rel < 1e-5, (c, float(rel))


def test_bytes():
    assert L.bytes_per_chip(1) == 663552
