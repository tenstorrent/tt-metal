# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's share of the query group (fused/projections/program.py query_units) for 8 token rows: q_b of the
chip's 2 heads from the gathered q_a, q_a's deferred norm from the lane sums, RoPE on the rope dims, the absorption
with kv_b's key side; against the reference restricted to the chip's heads (layout.query_golden, which the host
test pins to reference/attention_decode.py latent_queries), with random weights.

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
    tables,
)
from models.experimental.hy4_preview.fused.projections.tests.test_pre import FORMATS, THRESHOLD, stored_share


def query_inputs(lw, seed=1):
    """What Xa leaves on a chip: q_a [8, 2048] whole, as the pre units round it (7 significant bits), and the lane
    sums of its squares."""
    y, scale = random_input(seed)
    q_a = L.round7(((y.double() * scale.double()[:, None]) @ lw.q_a.double()).float())
    return q_a, L.lane_sums(q_a)


def run_query(device, chip, fmt, terms, seed=0, **kw):
    ref, gamma, _ = random_reference(seed)
    lw = L.layer_weights(ref.weights(), gamma)
    w = L.chip_weights(lw, chip)
    q_a, lanes = query_inputs(lw)
    _, _, d_cos, d_sin = tables()
    op = P.query_program(device, w, fmt=fmt, terms=terms, **kw)
    op.set_x(torch.cat((q_a, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
    op.set_params(lanes=lanes)
    op()
    return op, w, q_a, lanes, d_cos, d_sin


def check_query(label, op, w, q_a, lanes, d_cos, d_sin, fmt, terms, threshold=0.999):
    out, _, status = op.read()
    logger.info(f"HY4_MK7 {label} status {[hex(v) for v in status[:6]]} units {op.describe()}")
    op.check_status(status)
    done, units = op.out_units_done()
    logger.info(f"HY4_MK7 {label} output units counted on the hub: {done} of {units}")
    assert done == units
    dev = out.reshape(L.ROWS, L.HEADS_PER_CHIP, L.LATENT).permute(1, 0, 2)  # [heads, 8, 576]
    exact, _ = L.query_golden(q_a, lanes, d_cos, d_sin, w)
    rounded, _ = L.query_golden(q_a, lanes, d_cos, d_sin, stored_share(w, fmt, terms))
    results = {}
    for name, cols in (("q_abs", slice(0, L.KV_LORA)), ("q_rope", slice(L.KV_LORA, L.LATENT))):
        check(f"{label} {name} vs exact arithmetic on the stored weights", rounded[..., cols], dev[..., cols])
        results[name] = check(f"{label} {name} vs the reference", exact[..., cols], dev[..., cols], threshold)
    return results


@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("chip", [0, 13], ids=["chip0", "chip13"])
def test_query_random(device, chip, fmt):
    f, terms = FORMATS[fmt]
    op, w, q_a, lanes, d_cos, d_sin = run_query(device, chip, f, terms, **FID_KW)
    if P.MOCK:
        return
    check_query(f"query chip {chip} {fmt}{FID_TAG}", op, w, q_a, lanes, d_cos, d_sin, f, terms, THRESHOLD[fmt])
