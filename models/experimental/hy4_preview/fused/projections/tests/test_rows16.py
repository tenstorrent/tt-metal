# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The projection groups with 16 token rows in ONE pass of the weights' stream (ProjProgram(rows=16): [16, 32] row
tiles through hifi_mm16, kernels/proj_compute_hifi.cpp), on one chip, against exact arithmetic on the stored weights
and against the reference, as tests/test_pre.py, test_query.py and test_post.py do for 8 rows. Every row is its own
problem, so rows 0..7 and 8..15 are also given separately. Fidelity "recommended" (16 rows need the hifi_mm path).

Board, 2026-11-20 (logs mk8a_proj16_a, mk8a_proj16_b; one chip, all six cases pass first time), relative rms error
against exact arithmetic on the stored weights, 16 rows | the 8-row group at the same fidelity (log mk8a_proj16_c8):
q_a 3.37e-3 .. 3.44e-3 | 3.35e-3 .. 3.42e-3, gate 1.62e-3 | 1.61e-3 .. 1.63e-3, lane sums 3.1e-3 .. 3.9e-3 |
3.2e-3 .. 4.4e-3, pool 1.65e-3 .. 1.84e-3 | 1.69e-3 .. 1.70e-3, q_abs 3.72e-3 .. 3.82e-3 | 3.79e-3 .. 3.94e-3, q_rope
2.04e-3 .. 2.07e-3 | 2.04e-3 .. 2.09e-3, post 3.75e-3 .. 3.76e-3 | 3.71e-3 .. 3.76e-3; rows 0..7 and 8..15 alike.

On the mock Galaxy (HY4_MOCK=1) every case compiles its kernels and checks CB / L1 / kernel sizes; nothing runs.
Log lines start with HY4_MK7."""

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.tests.harness import (
    check,
    random_input,
    random_reference,
    tables,
)
from models.experimental.hy4_preview.fused.projections.tests.test_pre import stored_share

ROWS = 2 * L.ROWS
KW = dict(fidelity="recommended", rows=ROWS)
FMT, TERMS = "bfp8", 1
# as the 8-row cases at "recommended" (program.py: against exact arithmetic on the stored weights 2.5e-3 to 3.5e-3)
REL_STORED, PCC_REF = 1.0e-2, 0.9995


def both_halves(label, want, got, results):
    """The checks on all 16 rows and on each row tile's 8."""
    rel = check(f"{label} vs exact arithmetic on the stored weights", want["stored"], got)
    ref = check(f"{label} vs the reference", want["exact"], got, PCC_REF)
    for name, rows in (("rows 0..7", slice(0, L.ROWS)), ("rows 8..15", slice(L.ROWS, ROWS))):
        check(f"{label} {name} vs exact arithmetic on the stored weights", want["stored"][rows], got[rows])
    assert rel[1] <= REL_STORED, f"{label}: rel rms {rel[1]:.3e} against exact arithmetic on the stored weights"
    results[label] = (rel, ref)


def finish(label, op):
    out, stat, status = op.read()
    logger.info(f"HY4_MK7 {label} status {[hex(v) for v in status[:6]]} units {len(op.units)}")
    op.check_status(status)
    done, units = op.out_units_done()
    assert done == units, (done, units)
    return out, stat


@pytest.mark.parametrize("chip", [13, 0], ids=["chip13", "chip0"])
def test_pre_rows16(device, chip):
    ref, gamma, _ = random_reference(0, False)
    w = L.chip_weights(L.layer_weights(ref.weights(), gamma), chip)
    y, scale = random_input(rows=ROWS)
    _, _, d_cos, d_sin = tables(ROWS)
    op = P.pre_program(device, w, fmt=FMT, terms=TERMS, **KW)
    op.set_x(torch.cat((y, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
    op.set_params(scale=scale)
    op()
    if P.MOCK:
        return
    label = f"pre 16 rows chip {chip}"
    out, stat = finish(label, op)
    exact = L.pre_golden(y, scale, w, d_cos, d_sin)
    stored = L.pre_golden(y, scale, stored_share(w, FMT, TERMS), d_cos, d_sin)
    dev = {
        "q_a": out[:, : 2 * L.TILE],
        "gate": out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE],
        "lanes": L.stat_tile_lanes(stat),
    }
    if "pool" in w:
        dev["pool"] = out[:, P.PRE_POOL * L.TILE : P.PRE_GATE * L.TILE]
    results = {}
    for name, d in dev.items():
        both_halves(f"{label} {name}", {"exact": exact[name], "stored": stored[name]}, d, results)


@pytest.mark.parametrize("chip", [13, 0], ids=["chip13", "chip0"])
def test_query_rows16(device, chip):
    ref, gamma, _ = random_reference(0)
    lw = L.layer_weights(ref.weights(), gamma)
    w = L.chip_weights(lw, chip)
    y, scale = random_input(1, rows=ROWS)
    q_a = L.round7(((y.double() * scale.double()[:, None]) @ lw.q_a.double()).float())
    lanes = L.lane_sums(q_a)
    _, _, d_cos, d_sin = tables(ROWS)
    op = P.query_program(device, w, fmt=FMT, terms=TERMS, **KW)
    op.set_x(torch.cat((q_a, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
    op.set_params(lanes=lanes)
    op()
    if P.MOCK:
        return
    label = f"query 16 rows chip {chip}"
    out, _ = finish(label, op)
    dev = out.reshape(ROWS, L.HEADS_PER_CHIP, L.LATENT)  # [16, heads, 576]
    exact, _ = L.query_golden(q_a, lanes, d_cos, d_sin, w)
    stored, _ = L.query_golden(q_a, lanes, d_cos, d_sin, stored_share(w, FMT, TERMS))
    results = {}
    for name, cols in (("q_abs", slice(0, L.KV_LORA)), ("q_rope", slice(L.KV_LORA, L.LATENT))):
        want = {"exact": exact.permute(1, 0, 2)[..., cols], "stored": stored.permute(1, 0, 2)[..., cols]}
        both_halves(f"{label} {name}", want, dev[..., cols], results)


@pytest.mark.parametrize("chip", [13, 0], ids=["chip13", "chip0"])
def test_post_rows16(device, chip):
    ref, gamma, _ = random_reference(0)
    w = L.chip_weights(L.layer_weights(ref.weights(), gamma), chip)
    g = torch.Generator().manual_seed(2)
    ctx = (torch.randn(L.HEADS_PER_CHIP, ROWS, L.KV_LORA, generator=g)).to(torch.bfloat16).float()
    y, scale = random_input(2, rows=ROWS)
    gate = L.pre_golden(y, scale, w)["gate"].to(torch.bfloat16).float()
    op = P.post_program(device, w, fmt=FMT, terms=TERMS, **KW)
    op.set_x(torch.cat((ctx[0], ctx[1], gate), dim=1))
    op()
    if P.MOCK:
        return
    label = f"post 16 rows chip {chip}"
    out, _ = finish(label, op)
    exact, _ = L.post_golden(ctx, gate, w)
    stored, _ = L.post_golden(ctx, gate, stored_share(w, FMT, TERMS))
    both_halves(f"{label} partial", {"exact": exact, "stored": stored}, out, {})
