# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's share of the post group (fused/projections/program.py post_units) for 8 token rows: kv_b's value side
on the merged attention output of the chip's 2 heads, the gate multiply, o_proj's rows of those heads; against the
reference restricted to the chip's heads (layout.post_golden: the chip's partial of the sublayer output), with
random weights.

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
    check,
    random_input,
    random_reference,
)
from models.experimental.hy4_preview.fused.projections.tests.test_pre import FORMATS, THRESHOLD, stored_share


def post_inputs(w, seed=2):
    """ctx of the chip's heads [2, 8, 512] (BF16: what a merge leaves) and its gate [8, 512] (BF16, after the
    sigmoid: what the chip's pre group left)."""
    g = torch.Generator().manual_seed(seed)
    ctx = (torch.randn(L.HEADS_PER_CHIP, L.ROWS, L.KV_LORA, generator=g)).to(torch.bfloat16).float()
    y, scale = random_input(seed)
    gate = L.pre_golden(y, scale, w)["gate"].to(torch.bfloat16).float()
    return ctx, gate


def run_post(device, chip, fmt, terms, seed=0, **kw):
    ref, gamma, _ = random_reference(seed)
    w = L.chip_weights(L.layer_weights(ref.weights(), gamma), chip)
    ctx, gate = post_inputs(w)
    op = P.post_program(device, w, fmt=fmt, terms=terms, **kw)
    op.set_x(torch.cat((ctx[0], ctx[1], gate), dim=1))
    op()
    return op, w, ctx, gate


def check_post(label, op, w, ctx, gate, fmt, terms, threshold=0.999):
    out, _, status = op.read()
    logger.info(f"HY4_MK7 {label} status {[hex(v) for v in status[:6]]} units {len(op.units)}")
    op.check_status(status)
    done, units = op.out_units_done()
    logger.info(f"HY4_MK7 {label} output units counted on the hub: {done} of {units}")
    assert done == units
    exact, _ = L.post_golden(ctx, gate, w)
    rounded, _ = L.post_golden(ctx, gate, stored_share(w, fmt, terms))
    check(f"{label} partial vs exact arithmetic on the stored weights", rounded, out)
    return check(f"{label} partial vs the reference", exact, out, threshold)


@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("chip", [0, 13], ids=["chip0", "chip13"])
def test_post_random(device, chip, fmt):
    f, terms = FORMATS[fmt]
    op, w, ctx, gate = run_post(device, chip, f, terms, **FID_KW)
    if P.MOCK:
        return
    check_post(f"post chip {chip} {fmt}{FID_TAG}", op, w, ctx, gate, f, terms, THRESHOLD[fmt])
