# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's share of the pre group (fused/projections/program.py pre_units) for 8 token rows, weights streamed from
DRAM, against the reference restricted to the chip's share (layout.pre_golden), with random weights.

On the mock Galaxy (HY4_MOCK=1) every case compiles its kernels and checks CB / L1 / kernel sizes; nothing runs.
Log lines start with HY4_MK7."""

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.tests.harness import (
    FID_KW,
    FID_TAG,
    PRE_ROUNDED,
    check,
    random_input,
    random_reference,
    tables,
)

FORMATS = {"bfp8": ("bfp8", 1), "bfp4x2": ("bfp4", 2), "bfp4": ("bfp4", 1)}
# pcc against the reference, by weight format. Measured on the board with these random weights (logs
# mk7_all_board_1, mk7_final_board_1): BFP8 0.99976 to 0.99989 (custom_mm is LoFi: the multiplier cuts a BFP8 weight
# to about 5 significant bits, gain 0.973 to 0.988), two BFP4 terms 0.9988 to 0.9996, one BFP4 term 0.987 to 0.994.
THRESHOLD = {"bfp8": 0.9995, "bfp4x2": 0.998, "bfp4": 0.98}


def stored_share(w, fmt, terms):
    """The chip's share with every weight as the program's stream holds it (layout.stored_stream: a sharded tensor
    is BF16 first, then the block format)."""
    f = lambda m: L.stored_stream(m.float().contiguous(), fmt, terms, PRE_ROUNDED)
    return {k: [f(m) for m in v] if isinstance(v, list) else f(v) for k, v in w.items()}


def run_pre(device, chip, full, fmt, terms, seed=0, seven_bits=False, **kw):
    ref, gamma, indexer = random_reference(seed, full)
    lw = L.layer_weights(ref.weights(), gamma, indexer.weights() if full else None)
    w = L.chip_weights(lw, chip)
    y, scale = random_input(seven_bits=seven_bits)
    op = P.pre_program(device, w, fmt=fmt, terms=terms, **kw)
    _, _, d_cos, d_sin = tables()
    op.set_x(torch.cat((y, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
    op.set_params(scale=scale)
    op()
    return op, w, y, scale


def check_pre(label, op, w, y, scale, fmt, terms, threshold=0.999):
    out, stat, status = op.read()
    logger.info(f"HY4_MK7 {label} status {[hex(v) for v in status[:6]]} units {op.describe()}")
    op.check_status(status)
    done, units = op.out_units_done()
    logger.info(f"HY4_MK7 {label} output units counted on the hub: {done} of {units}")
    assert done == units
    _, _, d_cos, d_sin = tables()
    exact = L.pre_golden(y, scale, w, d_cos, d_sin)
    rounded = L.pre_golden(y, scale, stored_share(w, fmt, terms), d_cos, d_sin)
    dev = {
        "q_a": out[:, : 2 * L.TILE],
        "gate": out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE],
        "lanes": L.stat_tile_lanes(stat),
    }
    if "pool" in w:
        dev["pool"] = out[:, P.PRE_POOL * L.TILE : P.PRE_GATE * L.TILE]
    if "idx_q" in w:
        dev["idx_q_dev"] = out[:, P.PRE_IDX * L.TILE :]
    results = {}
    for name, d in dev.items():
        check(f"{label} {name} vs exact arithmetic on the stored weights", rounded[name], d)
        results[name] = check(f"{label} {name} vs the reference", exact[name], d, threshold)
    return results


@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("chip,full", [(0, False), (20, True), (31, False)], ids=["chip0", "chip20full", "chip31"])
def test_pre_random(device, chip, full, fmt):
    f, terms = FORMATS[fmt]
    op, w, y, scale = run_pre(device, chip, full, f, terms, **FID_KW)
    if P.MOCK:
        return
    check_pre(
        f"pre chip {chip} {'full' if full else 'shared'} {fmt}{FID_TAG}", op, w, y, scale, f, terms, THRESHOLD[fmt]
    )
