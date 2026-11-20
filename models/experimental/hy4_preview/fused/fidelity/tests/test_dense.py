# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense sums on one core against exact arithmetic on the stored operands, to separate what the multiplier does to
the operands (tests/test_probe.py: one product per sum) from what happens to sums of many products:

    layout      a SplitAcc layout (the second K face of every K tile accumulates in its own DEST rows and is merged
                at the end) against the plain one
    operands    x with 3 / 7 / 8 significant bits; weights BFP4 / BFP8 (random, scaled as a projection)
    K           one K face (16 products per sum) to 6144

and the projections piece's own operand probe (fused/projections/tests/test_probe.py: 8 K tiles, one output tile,
its weights and its one-hot rows) repeated here with FP32 readout.

One HY4_MK9 line per case: relative rms error, gain, the share of sums that are exact and the largest error in units
of the sum's 8th significant bit. Inputs and sums are stored in $HY4_PROBE_DIR as dense_*.pt. On the mock Galaxy the
programs are compiled only."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.fidelity import lab as LAB
from models.experimental.hy4_preview.fused.fidelity import probes as PR

pytestmark = pytest.mark.use_module_device
OUT_DIR = Path(os.environ.get("HY4_PROBE_DIR", "generated/hy4_preview/logs/mk9_probe"))


def x_values(rows, n, bits, seed, nonzero=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, generator=g)
    x = PR.trunc_sig(x, bits) if bits < 8 else x.to(torch.bfloat16).float()
    if nonzero is not None:
        mask = torch.zeros(n)
        mask[nonzero] = 1
        x = x * mask
    return x


def run(device, name, x, w, fmt, fidelity, layout, kb=None, ref=False):
    kt, ct = w.shape[0] // 32, w.shape[1] // 32
    passes, step = LAB.FIDELITIES[fidelity]
    op = LAB.Lab(device, kt, ct, fmt=fmt, passes=passes, step=step, layout=layout, kb=kb, ref=ref)
    try:
        op.set_w(w)
        op.set_x(x)
        op()
        if LAB.MOCK:
            return None
        out = op.read()
    finally:
        op.close()
    stored = LAB.stored(w, fmt)
    exact = x.double() @ stored.double()
    err = out.double() - exact
    rel = (err.pow(2).sum().sqrt() / exact.pow(2).sum().sqrt()).item()
    gain = ((out.double() * exact).sum() / exact.pow(2).sum()).item()
    _, e, _ = PR.decompose(exact)
    ulp8 = (err.abs() / 2.0 ** (e - 7)).max().item()
    logger.info(
        f"HY4_MK9 dense {name}: rel_rms={rel:.3e} gain={gain:.5f} exact sums {float((err == 0).float().mean()):.4f} "
        f"largest error {ulp8:.2f} units of the sum's 8th significant bit"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(name=name, x=x, w=w, stored=stored, out=out), OUT_DIR / f"dense_{name.replace(' ', '_')}.pt")
    return out


@pytest.mark.parametrize("layout", ["dense", "dense_split", "dense_split_exact"])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4", "p02"])
@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
@pytest.mark.parametrize("bits", [3, 7, 8])
@pytest.mark.parametrize("kt", [1, 8, 192])
def test_dense(device, kt, bits, fmt, fidelity, layout):
    g = torch.Generator().manual_seed(6)
    k = 32 * kt
    w = torch.randn(k, 64, generator=g) * k**-0.5
    x = x_values(8, k, bits, seed=7)
    run(device, f"K{k} x{bits} {fmt} {fidelity} {layout}", x, w, fmt, fidelity, layout, kb=min(kt, 32))


@pytest.mark.parametrize("layout", ["dense", "dense_split"])
@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
@pytest.mark.parametrize("face", [0, 1])
def test_one_face(device, face, fmt, layout):
    """16 products per sum, all in one K face of one K tile, operands the multiplier takes whole at LoFi."""
    g = torch.Generator().manual_seed(8)
    w = PR.trunc_sig(torch.randn(32, 64, generator=g) * 0.25, 3)
    x = x_values(8, 32, 7, seed=9, nonzero=slice(16 * face, 16 * face + 16))
    run(device, f"one face {face} {fmt} lofi {layout}", x, w, fmt, "lofi", layout)


@pytest.mark.parametrize("which", ["ref", "lofi"])
@pytest.mark.parametrize("layout", ["dense", "dense_split"])
@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
def test_projections_probe(device, fmt, layout, which):
    """fused/projections/tests/test_probe.py test_probe_operands with FP32 readout."""
    if which == "ref" and layout == "dense":
        pytest.skip("one reference form is enough")
    g = torch.Generator().manual_seed(4)
    wt = torch.randn(32 * 8, 32, generator=g) * 0.05
    passes, step = LAB.FIDELITIES["lofi"]
    op = LAB.Lab(device, 8, 1, fmt=fmt, passes=passes, step=step, layout=layout, ref=which == "ref")
    stored = LAB.stored(wt, fmt)
    try:
        op.set_w(wt)
        for value in (1.0, 1.99609375):
            seen, want = [], []
            for i in range(4):
                rows = [(37 * i + 29 * r + 5) % 256 for r in range(8)]
                x = torch.zeros(8, 256)
                x[torch.arange(8), rows] = value
                op.set_x(x)
                op()
                if LAB.MOCK:
                    return
                seen.append(op.read())
                want.append(stored[rows])
            dev, ref = torch.cat(seen), torch.cat(want)
            if value == 1.0:
                base = dev
                logger.info(
                    f"HY4_MK9 projections' probe {fmt} {layout} {which} weights: equal to stored "
                    f"{float((dev == ref).float().mean()):.4f}, to stored cut to 5 significant bits "
                    f"{float((dev == PR.trunc_sig(ref, 5)).float().mean()):.4f}"
                )
            else:
                nz = base != 0
                ratio = dev[nz].double() / base[nz].double()
                t7 = PR.trunc_sig(torch.tensor([value]), 7).item()
                logger.info(
                    f"HY4_MK9 projections' probe {fmt} {layout} {which} operand: x = {value} comes back as a factor "
                    f"of {ratio.median().item():.6f} (min {ratio.min().item():.6f}, max {ratio.max().item():.6f}); "
                    f"cut to 7 significant bits it is {t7}"
                )
    finally:
        op.close()


@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4", "p02"])
@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_dense_rows16(device, fmt, fidelity):
    """hifi_mm16 on the q_a-sized unit: 16 rows of BF16 x against random weights."""
    g = torch.Generator().manual_seed(6)
    w = torch.randn(6144, 64, generator=g) * 6144**-0.5
    x = x_values(16, 6144, 8, seed=7)
    passes, step = LAB.FIDELITIES[fidelity]
    op = LAB.Lab(device, 192, 2, fmt=fmt, rows=16, passes=passes, step=step, kb=32)
    try:
        op.set_w(w)
        op.set_x(x)
        op()
        if LAB.MOCK:
            return
        out = op.read()
    finally:
        op.close()
    exact = x.double() @ LAB.stored(w, fmt).double()
    err = out.double() - exact
    logger.info(
        f"HY4_MK9 dense rows16 K6144 x8 {fmt} {fidelity}: rel_rms={(err.pow(2).sum().sqrt() / exact.pow(2).sum().sqrt()).item():.3e} "
        f"gain={((out.double() * exact).sum() / exact.pow(2).sum()).item():.5f}"
    )
