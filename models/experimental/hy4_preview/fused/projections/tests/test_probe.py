# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What custom_mm (LoFi) computes with, read back from the device: one unit with no finishing whose left operand has
a single non-zero value per token row, so that every output row is one row of the weight as the multiplier sees it
(times that value as the multiplier sees it).

    weights   x = 1.0: the output row is the weight row. Compared with the stored value (BFP8 / BFP4 as ttnn rounds
              it on the host) and with that value cut to 5 significant bits.
    operand   the same with x = 1.99609375 (8 significant bits, all ones): the factor the outputs come back with.

Log lines start with HY4_MK7. On the mock Galaxy the program is compiled only."""

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P

K_TILES = 8


def rel_rms(ref, dev):
    return ((ref.double() - dev.double()).pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
def test_probe_operands(device, fmt):
    g = torch.Generator().manual_seed(4)
    wt = torch.randn(L.TILE * K_TILES, L.TILE, generator=g) * 0.05
    op = P.ProjProgram(device, [P.Unit("probe", wt, 1, fin=L.FIN_NONE, rnd=P.RND_NONE)], K_TILES, 1, fmt=fmt)
    stored = L.stored(wt, fmt)
    seen, want = [], []
    for value in (1.0, 1.99609375):
        for run in range(4):
            rows = [(37 * run + 29 * r + 5) % (L.TILE * K_TILES) for r in range(L.ROWS)]
            x = torch.zeros(L.ROWS, L.TILE * K_TILES)
            x[torch.arange(L.ROWS), rows] = value
            op.set_x(x)
            op()
            if P.MOCK:
                return
            out, _, status = op.read()
            op.check_status(status)
            seen.append(out)
            want.append(stored[rows])
        dev, ref = torch.cat(seen[-4:]), torch.cat(want[-4:])
        if value == 1.0:
            same = (dev == ref).float().mean().item()
            cut5 = (dev == L.trunc_bits(ref, 5)).float().mean().item()
            cut4 = (dev == L.trunc_bits(ref, 4)).float().mean().item()
            logger.info(
                f"HY4_MK7 probe {fmt} weights: device rows equal to the stored value in {same:.4f} of the entries, to "
                f"the stored value cut to 5 significant bits in {cut5:.4f}, to 4 bits in {cut4:.4f}; relative rms "
                f"difference to the stored value {rel_rms(ref, dev):.3e}"
            )
            diff = (dev != ref).nonzero()[:16]
            pairs = [(round(ref[i, j].item(), 6), round(dev[i, j].item(), 6)) for i, j in diff.tolist()]
            logger.info(f"HY4_MK7 probe {fmt} weights: first entries that differ (stored, device): {pairs}")
            row = ref[0]
            logger.info(f"HY4_MK7 probe {fmt} weights: row 0 stored {[round(v, 6) for v in row[:16].tolist()]}")
            logger.info(f"HY4_MK7 probe {fmt} weights: row 0 device {[round(v, 6) for v in dev[0, :16].tolist()]}")
            base = dev
        else:
            nz = base != 0
            ratio = dev[nz].double() / base[nz].double()
            logger.info(
                f"HY4_MK7 probe {fmt} operand: x = {value} comes back as a factor of {ratio.median().item():.6f} "
                f"(min {ratio.min().item():.6f}, max {ratio.max().item():.6f}); 7 significant bits would be "
                f"{L.trunc_bits(torch.tensor([value]), 7).item()}"
            )


@pytest.mark.parametrize("max_block", [56 * 1024, 14 * 1024, 7 * 1024], ids=["blk56k", "blk14k", "blk7k"])
@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_probe_block_size(device, fmt, max_block):
    """A q_a-sized unit (192 K tiles, 2 output tiles, rounding only) on a left operand that already has 7 significant
    bits, against exact arithmetic on the stored weights, with the streamed blocks cut to different sizes: what is
    left is the multiplier's treatment of the weights and whatever the stream does to them."""
    g = torch.Generator().manual_seed(6)
    wt = torch.randn(L.HIDDEN, 2 * L.TILE, generator=g) * L.HIDDEN**-0.5
    x = L.round7(torch.randn(L.ROWS, L.HIDDEN, generator=g))
    unit = P.Unit("probe", wt, 2, fin=L.FIN_ROUND, rnd=P.RND_BF16)
    op = P.ProjProgram(device, [unit], L.HIDDEN // L.TILE, 2, fmt=fmt, max_block=max_block)
    op.set_x(x)
    op()
    if P.MOCK:
        return
    out, _, status = op.read()
    op.check_status(status)
    stored = L.stored(wt, fmt).double()
    exact = (x.double() @ stored).float()
    cut5 = (x.double() @ L.trunc_bits(stored.float(), 5).double()).float()
    kb = P.choose_kb(L.HIDDEN // L.TILE, 2, L.TILE_BYTES[fmt], max_block)
    block = kb * 2 * L.TILE_BYTES[fmt]
    gain = ((exact.double() * out.double()).sum() / exact.double().pow(2).sum()).item()
    logger.info(
        f"HY4_MK7 probe {fmt} blocks of {block} B ({P.packets(block)} packets of {block // P.packets(block)} B): relative "
        f"rms difference to exact arithmetic on the stored weights {rel_rms(exact, out):.3e} (gain {gain:.5f}), to the "
        f"stored weights cut to 5 significant bits {rel_rms(cut5, out):.3e}"
    )
