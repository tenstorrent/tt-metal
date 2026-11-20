# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""hifi_mm (kernels/hifi_mm.h) on one core against exact arithmetic, on operands every fidelity must multiply
exactly: x with 3 significant bits, weights with 2, so any pass that added a wrong term, any tile that landed in the
wrong DEST rows and any partial that was not merged shows as a wrong sum.

The plain DEST layouts must give every sum exactly. The SplitAcc layouts may not: their merge moves the second K
face's accumulator through SrcB as BF16 (MOVD2B of the upper 16 bits), which cuts it to 8 significant bits; that is
custom_mm's own sequence and shows in custom_mm too ("ref"). For them the error is bounded by 2^-7 of the sum of the
products' magnitudes.

Cases: the tree's custom_mm ("ref") and every pass schedule; the DEST layouts; the K range in one or several
blocks (only the last block merges the split partials); 1 / 2 / 4 / 8 token rows; ct 1 to 8; row-major and
column-major weight tiles; BFP8 / BFP4 / BF16 weights. On the mock Galaxy the programs are compiled only.

Log lines start with HY4_MK9."""

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.fidelity import lab as LAB

FID = LAB.FIDELITIES
pytestmark = pytest.mark.use_module_device


def exact_case(kt, ct, rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-7, 8, (rows, LAB.TILE * kt), generator=g).float() / 4  # 3 significant bits
    w = torch.randint(-3, 4, (LAB.TILE * kt, LAB.TILE * ct), generator=g).float() / 4  # 2 bits: exact in BFP4 too
    w[0::16, 0::16] = 0.75
    return x, w


def run_exact(device, name, kt, ct, rows=8, fmt="bfp8", fidelity="lofi", seed=0, **kwargs):
    passes, step = FID[fidelity]
    op = LAB.Lab(device, kt, ct, fmt=fmt, rows=rows, passes=passes, step=step, **kwargs)
    x, w = exact_case(kt, ct, rows, seed)
    try:
        op.set_w(w)
        op.set_x(x)
        op()
        if LAB.MOCK:
            return
        out = op.read()
    finally:
        op.close()
    stored = LAB.stored(w, fmt)
    want = (x.double() @ stored.double()).float()
    bad = (out != want).sum().item()
    logger.info(
        f"HY4_MK9 exact {name}: {bad} of {out.numel()} sums differ from exact arithmetic "
        f"(max abs difference {(out - want).abs().max().item():.3e}, largest sum {want.abs().max().item():.3f})"
    )
    where = [(r, c, out[r, c].item(), want[r, c].item()) for r, c in (out != want).nonzero()[:8].tolist()]
    if bad:
        logger.info(f"HY4_MK9 exact {name}: (row, column, device, exact) of the first differences: {where}")
    if kwargs.get("layout", "dense").endswith("_split"):  # custom_mm's finalize
        bound = (x.double().abs() @ stored.double().abs()) * 2.0**-7
        assert (
            (out.double() - want.double()).abs() <= bound
        ).all(), f"{name}: beyond the BF16 cut of the merge: {where}"
    else:
        assert bad == 0, f"{name}: {bad} sums differ: {where}"


@pytest.mark.parametrize("which", ["ref", "lofi"])
def test_first(device, which):
    """The first board run of the kernel: custom_mm, then hifi_mm with one pass (the same instruction stream)."""
    run_exact(device, f"first {which}", 2, 2, ref=which == "ref")


@pytest.mark.parametrize("fidelity", ["hifi2", "hifi3", "hifi4", "p02"])
def test_passes(device, fidelity):
    run_exact(device, f"passes {fidelity}", 2, 2, fidelity=fidelity)


@pytest.mark.parametrize("layout", ["default", "default_split", "dense", "dense_split", "compact"])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4"])
def test_layouts(device, fidelity, layout):
    run_exact(device, f"layout {layout} {fidelity}", 6, 3, fidelity=fidelity, layout=layout, kb=2)


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4"])
def test_rows(device, fidelity, rows):
    run_exact(device, f"rows {rows} {fidelity}", 4, 2, rows=rows, fidelity=fidelity, kb=2)


@pytest.mark.parametrize("ct", [1, 2, 4, 8])
@pytest.mark.parametrize("fidelity", ["hifi2", "hifi4"])
def test_ct(device, fidelity, ct):
    run_exact(device, f"ct {ct} {fidelity}", 4, ct, fidelity=fidelity, kb=1)


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8", "bf16"])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4", "p02"])
def test_formats(device, fidelity, fmt):
    run_exact(device, f"fmt {fmt} {fidelity}", 4, 2, fmt=fmt, fidelity=fidelity, kb=2)


@pytest.mark.parametrize("layout", ["default_split", "dense_split"])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4"])
def test_column_major(device, fidelity, layout):
    run_exact(device, f"column-major {layout} {fidelity}", 6, 4, fidelity=fidelity, layout=layout, column_major=True)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("which", ["ref", "lofi", "hifi4"])
@pytest.mark.parametrize("rows", [4, 8])
@pytest.mark.parametrize("layout", ["dense", "dense_split"])
def test_seeds(device, layout, rows, which, seed):
    """More operand draws of the 4- and 8-row cases, through custom_mm and through hifi_mm, both kinds of layout."""
    fidelity = "lofi" if which == "ref" else which
    run_exact(
        device,
        f"seed {seed} rows {rows} {which} {layout}",
        4,
        2,
        rows=rows,
        fidelity=fidelity,
        kb=2,
        seed=seed,
        ref=which == "ref",
        layout=layout,
    )


def test_long_k(device):
    """K = 6144 in 6 blocks of 32 K tiles, 2 output tiles, HiFi4: the unit the projections call q_a."""
    run_exact(device, "K 6144 hifi4", 192, 2, fidelity="hifi4", kb=32)


def test_long_k_split(device):
    run_exact(device, "K 6144 hifi4 dense_split", 192, 2, fidelity="hifi4", kb=32, layout="dense_split")


@pytest.mark.parametrize("layout", ["default_split_exact", "dense_split_exact"])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4", "p02"])
@pytest.mark.parametrize(
    "shape", [(4, 2, 8, 2, 1), (4, 2, 4, 2, 1), (6, 3, 8, 2, 3), (2, 8, 8, 1, 2), (192, 2, 8, 32, 1)]
)
def test_exact_merge(device, shape, fidelity, layout):
    """SplitAcc layouts with hifi_mm_merge: every sum exact. shape: (K tiles, ct, rows, block, matmuls per launch:
    with more than one, the matmul after a merge and a pack must still be right)."""
    kt, ct, rows, kb, reps = shape
    if layout.startswith("default") and ct > 4:
        pytest.skip("4 FP32 tiles of 64 rows fit DEST")
    run_exact(
        device,
        f"merge {shape} {fidelity} {layout}",
        kt,
        ct,
        rows=rows,
        fidelity=fidelity,
        kb=kb,
        layout=layout,
        reps=reps,
    )


@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4"])
def test_exact_merge_column_major(device, fidelity):
    run_exact(
        device, f"merge column-major {fidelity}", 6, 4, fidelity=fidelity, layout="dense_split_exact", column_major=True
    )


@pytest.mark.parametrize("layout", ["dense", "default"])
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2", "hifi4", "p02"])
@pytest.mark.parametrize("shape", [(2, 2, 2, 1), (6, 3, 2, 3), (4, 1, 1, 1), (192, 2, 32, 1)])
def test_rows16(device, shape, fidelity, layout):
    """hifi_mm16: 16 token rows per pass over the weights. shape: (K tiles, ct, block, matmuls per launch)."""
    kt, ct, kb, reps = shape
    run_exact(
        device,
        f"rows 16 {shape} {fidelity} {layout}",
        kt,
        ct,
        rows=16,
        fidelity=fidelity,
        kb=kb,
        layout=layout,
        reps=reps,
    )


@pytest.mark.parametrize("fidelity", ["lofi", "hifi4"])
def test_rows16_column_major(device, fidelity):
    run_exact(device, f"rows 16 column-major {fidelity}", 6, 4, rows=16, fidelity=fidelity, column_major=True)


@pytest.mark.parametrize("fidelity", ["lofi", "p02", "hifi4"])
@pytest.mark.parametrize(
    "shape",
    [
        ("default_split_exact", 4, 2, 2, 3, False),
        ("default_split_exact", 4, 4, 2, 1, False),
        ("dense_split_exact", 2, 8, 2, 2, True),
        ("dense_split_exact", 2, 4, 2, 1, True),
    ],
)
def test_exact_merge_sync_half(device, shape, fidelity):
    """hifi_mm_merge with DEST in SyncHalf, the shapes of fused/moe: gate|up (Default layout, 2 tiles) and down
    (column-major; 4 Default or 8 Dense tiles fill the half)."""
    layout, kt, ct, kb, reps, column_major = shape
    kb = kt if column_major else kb
    run_exact(
        device, f"merge half {shape} {fidelity}", kt, ct, fidelity=fidelity, kb=kb, layout=layout, reps=reps,
        column_major=column_major, full_sync=False,
    )  # fmt: skip


@pytest.mark.parametrize("layout", ["default_split_exact", "dense_split_exact"])
def test_merge_round_bf16(device, layout):
    """hifi_mm_merge<.., round_bf16 = true>: every sum is the exact sum rounded to the nearest BF16 (halves away
    from zero)."""
    kt, ct = 6, 3
    passes, step = FID["hifi4"]
    op = LAB.Lab(device, kt, ct, passes=passes, step=step, layout=layout, kb=2, round_bf16=True, full_sync=False)
    x, w = exact_case(kt, ct, 8, seed=5)  # sums are multiples of 1/16 up to about 30: many need more than 8 bits
    try:
        op.set_w(w)
        op.set_x(x)
        op()
        if LAB.MOCK:
            return
        out = op.read()
    finally:
        op.close()
    exact = (x.double() @ LAB.stored(w, "bfp8").double()).float()
    bits = exact.view(torch.int32)
    want = ((bits + 0x8000) & ~0xFFFF).view(torch.float32)
    bad = (out != want).sum().item()
    logger.info(
        f"HY4_MK9 exact merge with BF16 rounding {layout}: {bad} of {out.numel()} sums differ from round(exact); "
        f"{int((want != exact).sum())} sums are changed by the rounding"
    )
    assert bad == 0


@pytest.mark.parametrize("full_sync", [True, False], ids=["sync_full", "sync_half"])
def test_packer_bf16(device, full_sync):
    """What the packer does to FP32 sums that go into a BF16 buffer: cut toward zero, or rounded?"""
    kt, ct = 6, 3
    x, w = exact_case(kt, ct, 8, seed=5)
    op = LAB.Lab(device, kt, ct, layout="dense_split_exact", kb=2, out_fmt="bf16", full_sync=full_sync)
    try:
        op.set_w(w)
        op.set_x(x)
        op()
        if LAB.MOCK:
            return
        out = op.read()
    finally:
        op.close()
    exact = (x.double() @ LAB.stored(w, "bfp8").double()).float()
    bits = exact.view(torch.int32)
    cut = (bits & ~0xFFFF).view(torch.float32)
    away = ((bits + 0x8000) & ~0xFFFF).view(torch.float32)
    even = exact.to(torch.bfloat16).float()
    need = cut != exact
    logger.info(
        f"HY4_MK9 packer FP32 -> BF16: of {int(need.sum())} sums that need more than 8 bits, {int((out == cut)[need].sum())} "
        f"come back cut toward zero, {int((out == away)[need].sum())} rounded to nearest (halves away from zero), "
        f"{int((out == even)[need].sum())} rounded to nearest even; the others exact: {bool((out == exact)[~need].all())}"
    )
