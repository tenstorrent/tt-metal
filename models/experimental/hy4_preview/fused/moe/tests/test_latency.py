# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced latency of one chip's whole MoE share as one program (fused/moe/program.py: router -> union -> routed
experts + shared expert (two BFP4 terms) -> partial output) for 8 rows whose picks make a union of 8, 16, 32 and 64 experts: the
program captured as a trace, 20 replays back to back, one synchronize (tests/test_rate.py replay_ms).

The router is real (its scores, top-8, union and weights run on the chip); its weight is crafted so that the union
has the wanted size: expert e reads hidden channel e, and row r's 8 picks are the channels set high in x[r].
Next to the ttnn form of the same stages (PERF.md, traced, every chip): router 0.17 ms, experts 0.85 ms, shared
expert 0.10 ms (and combine 0.08 ms, the cross-chip part, which this program does not contain).

Board only. Log lines start with HY4_MK5."""

import os

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe.program import MOCK, RoutedExperts
from models.experimental.hy4_preview.fused.moe.tests.test_rate import CYCLE_NS, TRACE_REGION, replay_ms
from models.experimental.hy4_preview.fused.moe.tests.test_routed import (
    FID,
    FID_KW,
    HOST_ROUND,
    ROUNDED_PCC,
    check,
    random_share_weights,
)

E = 256
TTNN_MS = {"router": 0.17, "experts": 0.85, "shared expert": 0.10}
UNIONS = [8, 16, 32, 64]
_timing = bool(os.environ.get("HY4_LATENCY_TIMING"))


def crafted(U, seed=0):
    """(x [8, 6144] BF16-exact, the rows' picks): row r picks the 8 experts of its group, U / 8 groups in all."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(L.ROWS, L.HIDDEN, generator=g)).to(torch.bfloat16).float()
    x[:, :E] = -4.0
    picks = []
    for r in range(L.ROWS):
        group = r * (U // 8) // L.ROWS
        ids = list(range(8 * group, 8 * group + 8))
        x[r, ids] = 4.0
        picks.append(ids)
    return x, picks


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("candidates", [True, False], ids=["candidates", "scan"])
@pytest.mark.parametrize("share", ["tp32", "tp8"])
def test_chip_latency(device, share, candidates):
    if MOCK:
        pytest.skip("timing needs the board")
    sh = L.tp_share(int(share[2:]))
    gate, up, down = random_share_weights(sh, E + 1)
    defines = [("HY4_TIMING", "1")] if _timing else []
    op = RoutedExperts(
        device,
        sh,
        router=True,
        num_experts=E,
        n_plain=1,
        plain_id0=E,
        plain_terms=2,
        defines=defines,
        mode="rect",
        down_per_bank=4,
        router_candidates=candidates,
        **FID_KW,
    )
    share = f"{share} {'candidates' if candidates else 'scan'}" + ("" if FID == "custom_mm" else f" [{FID}]")
    terms = lambda w: torch.cat((w[:E], L.two_terms(w[E:])), dim=0)
    value = lambda w: torch.cat((L.bfp4_round(w[:E].transpose(1, 2)).transpose(1, 2), L.two_terms_value(w[E:])), dim=0)
    if HOST_ROUND:  # the two terms as their own BFP4 values: what layout.two_terms_value is
        q1 = lambda w: L.bfp4_round(w.transpose(1, 2)).transpose(1, 2)
        terms = lambda w: torch.cat((w[:E], q1(w[E:]), q1(w[E:] - q1(w[E:]))), dim=0)
    w_gu = L.to_device_stream(device, L.gate_up_stream(terms(gate), terms(up), sh), host_round=HOST_ROUND)
    w_dn = L.to_device_stream(device, L.down_stream(terms(down), sh), host_round=HOST_ROUND)
    g_gate, g_up, g_down = value(gate), value(up), value(down)
    op.set_router(torch.eye(E, L.HIDDEN), torch.zeros(E))
    ttnn_ms = sum(TTNN_MS.values())
    for U in UNIONS:
        x, picks = crafted(U)
        op.set_x(x)
        ms = replay_ms(device, lambda: op(w_gu, w_dn))
        out, status = op.read_output()
        assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
        _, ids, weights = op.read_router()
        assert all(sorted(ids[r].tolist()) == picks[r] for r in range(L.ROWS)), ids
        union, w = L.union_of(ids, weights)
        assert len(union) == U
        nbytes = op.bytes_read(U + 1, n_plain=1)
        logger.info(
            f"HY4_MK5 latency chip {share} U={U} (+ shared): {ms * 1e3:.1f} us per replay, {nbytes / 1e6:.1f} MB of "
            f"expert weights, {nbytes / (ms * 1e-3) / 1e9:.0f} GB/s over the whole program; ttnn form "
            f"{ttnn_ms * 1e3:.0f} us (router {TTNN_MS['router']}, experts {TTNN_MS['experts']}, shared expert "
            f"{TTNN_MS['shared expert']} ms): {ttnn_ms / ms:.1f}x"
        )
        if _timing:
            t = op.read_timing()
            us = lambda cycles: cycles * CYCLE_NS / 1e3
            logger.info(
                f"HY4_MK5 latency chip {share} U={U} hub: x sent {us(t['hub'][0]):.1f} us, scores in "
                f"{us(t['hub'][1]):.1f} us, release sent {us(t['hub'][2]):.1f} us after the hub started"
            )
        # the traced program still computes the right thing
        rounded = L.partial_golden(
            x, union + [E], torch.cat((w, torch.ones(L.ROWS, 1)), dim=1), g_gate, g_up, g_down, n_plain=1
        )
        check(f"latency chip {share} U={U} output vs golden with BFP4-rounded weights", rounded, out, ROUNDED_PCC)
