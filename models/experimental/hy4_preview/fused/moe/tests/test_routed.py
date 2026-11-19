# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's routed experts for 8 token rows, the picked experts streamed from DRAM (fused/moe/program.py), against
the golden of the chip's partial output (fused/moe/layout.py partial_golden, the per-chip term of
reference/moe.py Hy4MoE.routed), with random weights.

On the mock Galaxy (HY4_MOCK=1) every case compiles its kernels and checks CB / L1 / kernel sizes; nothing runs.
Log lines start with HY4_MK5.
"""

import os

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe.program import MOCK, TRUNCATION_CORRECTION, RoutedExperts
from models.experimental.hy4_preview.tests.common import pcc

SHARES = {"tp32": L.tp_share(32), "tp8": L.tp_share(8), "ep": L.ep_share()}
# Against exact arithmetic on the BFP4-rounded weights. The first run on the board (tp32, one expert, log
# of 2026-11-19) gave 0.9982 with gain 0.979; test_stage_accuracy shows where the difference comes from.
ROUNDED_PCC = 0.997


def random_share_weights(share, experts, seed=0):
    """gate, up [E, 32 P, 6144] and down [E, 6144, 32 P] with the reference's init (std 0.006)."""
    g = torch.Generator().manual_seed(seed)
    I = share.channels
    gate = torch.randn(experts, I, L.HIDDEN, generator=g) * 0.006
    up = torch.randn(experts, I, L.HIDDEN, generator=g) * 0.006
    down = torch.randn(experts, L.HIDDEN, I, generator=g) * 0.006
    return gate, up, down


def random_routing(experts, U, seed=0):
    """A union of U distinct local expert ids (unsorted) and [8, U] weights: every row picks about 8 of them."""
    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randperm(experts, generator=g)[:U].tolist()
    w = torch.zeros(L.ROWS, U)
    for r in range(L.ROWS):
        picks = torch.randperm(U, generator=g)[: min(8, U)]
        s = torch.rand(len(picks), generator=g) + 0.1
        w[r, picks] = s / s.sum() * 2.827
    return ids, w


def error_map(name, ref, dev):
    """Where the error sits: relative rms error per row and per 32-column tile (the worst tiles by index)."""
    rms = ref.double().pow(2).mean().sqrt().item()
    err = (dev.double() - ref.double()).pow(2)
    rows = (err.mean(dim=1).sqrt() / rms).tolist()
    tiles = err.reshape(err.shape[0], -1, 32).mean(dim=(0, 2)).sqrt() / rms
    worst = torch.argsort(tiles, descending=True)[:8].tolist()
    logger.info(
        f"HY4_MK5 {name}: relative rms error per row {[round(v, 3) for v in rows]}; median per tile "
        f"{tiles.median().item():.3f}; worst tiles {[(t, round(tiles[t].item(), 3)) for t in worst]}"
    )


def check(name, ref, dev, threshold):
    if ref.shape[-1] % 32 == 0 and ref.dim() == 2:
        error_map(name, ref, dev)
    score = pcc(ref, dev)
    err = (ref - dev).abs().max().item()
    gain = (ref.double() * dev.double()).sum().item() / (ref.double() ** 2).sum().item()
    logger.info(f"HY4_MK5 {name}: pcc={score:.6f} max_abs_err={err:.3e} gain={gain:.5f} ref_rms={ref.std().item():.3e}")
    assert score >= threshold, f"{name}: pcc {score:.6f} < {threshold}"


@pytest.mark.parametrize("U", [1, 8, 64], ids=lambda u: f"U{u}")
@pytest.mark.parametrize("share", ["tp32", "tp8"])
def test_routed_tp_random(device, share, U):
    sh = SHARES[share]
    experts = 64
    gate, up, down = random_share_weights(sh, experts)
    op = RoutedExperts(device, sh)
    w_gu = L.to_device_stream(device, L.gate_up_stream(gate, up, sh))
    w_dn = L.to_device_stream(device, L.down_stream(down, sh))
    g = torch.Generator().manual_seed(3)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g)
    ids, w = random_routing(experts, U)
    op.set_inputs(x, ids, w)
    op(w_gu, w_dn)
    if MOCK:
        return
    out, status = op.read_output()
    logger.info(f"HY4_MK5 routed {share} U={U} status {status[:6]}")
    assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
    xb = x.to(torch.bfloat16).float()
    wb = w.to(torch.bfloat16).float()
    rounded = L.partial_golden(
        xb,
        ids,
        wb,
        L.bfp4_round(gate.transpose(1, 2)).transpose(1, 2),
        L.bfp4_round(up.transpose(1, 2)).transpose(1, 2),
        L.bfp4_round(down.transpose(1, 2)).transpose(1, 2),
    )
    exact = L.partial_golden(x, ids, w, gate, up, down)
    check(f"routed {share} U={U} vs golden with BFP4-rounded weights", rounded, out, ROUNDED_PCC)
    check(f"routed {share} U={U} vs golden with exact weights", exact, out, 0.97)


@pytest.mark.parametrize("n", [0, 1, 3, 6], ids=lambda n: f"E{n}")
def test_routed_ep_random(device, n):
    """Expert parallel: the chip owns 8 whole experts and n of them were picked."""
    sh = SHARES["ep"]
    experts = 8
    gate, up, down = random_share_weights(sh, experts)
    op = RoutedExperts(device, sh, max_slots=8)
    w_gu = L.to_device_stream(device, L.gate_up_stream(gate, up, sh))
    w_dn = L.to_device_stream(device, L.down_stream(down, sh))
    g = torch.Generator().manual_seed(3)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g)
    ids, w = random_routing(experts, n)
    op.set_inputs(x, ids, w)
    op(w_gu, w_dn)
    if MOCK:
        return
    out, status = op.read_output()
    logger.info(f"HY4_MK5 routed ep picked={n} status {status[:6]}")
    assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
    if n == 0:
        assert out.abs().max().item() == 0.0
        return
    xb = x.to(torch.bfloat16).float()
    wb = w.to(torch.bfloat16).float()
    rounded = L.partial_golden(
        xb,
        ids,
        wb,
        L.bfp4_round(gate.transpose(1, 2)).transpose(1, 2),
        L.bfp4_round(up.transpose(1, 2)).transpose(1, 2),
        L.bfp4_round(down.transpose(1, 2)).transpose(1, 2),
    )
    exact = L.partial_golden(x, ids, w, gate, up, down)
    check(f"routed ep picked={n} vs golden with BFP4-rounded weights", rounded, out, ROUNDED_PCC)
    check(f"routed ep picked={n} vs golden with exact weights", exact, out, 0.97)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(defines=[("HY4_TIMING", "1")]),
        dict(defines=[("HY4_NO_READ", "1"), ("HY4_NO_MATH", "1")]),
        dict(down_per_bank=3, gate_up_per_bank=8),
        dict(down_per_bank=4, gate_up_per_bank=8),
        dict(gate_up_ring=(6, 3), mode="row"),
        dict(banks=[0, 5]),
        dict(bias_correction=TRUNCATION_CORRECTION),
    ],
    ids=["timing", "noread-nomath", "g8d3", "g8d4", "ring63-row", "banks05", "correction"],
)
def test_variants_compile(device, kwargs):
    """The measurement variants (tests/test_rate.py) compile and fit; nothing runs."""
    sh = SHARES["tp32"]
    gate, up, down = random_share_weights(sh, 2)
    op = RoutedExperts(device, sh, **kwargs)
    op.prepare(
        L.to_device_stream(device, L.gate_up_stream(gate, up, sh)), L.to_device_stream(device, L.down_stream(down, sh))
    )


def round_bits(v, bits):
    """v rounded to `bits` significant bits (to nearest)."""
    m, e = torch.frexp(v.double())
    return torch.ldexp(torch.round(m * 2**bits) / 2**bits, e).float()


@pytest.mark.parametrize("share", ["tp32", "tp8"])
def test_routed_gain(device, share):
    """The gain (least-squares scale of the output against the golden with BFP4-rounded weights) of the LoFi path:
    as is, with x rounded to 7 significant bits by the host (what its producer could do), with the finisher's
    truncation corrections, and with both. PCC does not see a uniformly smaller output."""
    sh = SHARES[share]
    experts, U = 64, 16
    gate, up, down = random_share_weights(sh, experts)
    w_gu = L.to_device_stream(device, L.gate_up_stream(gate, up, sh))
    w_dn = L.to_device_stream(device, L.down_stream(down, sh))
    g = torch.Generator().manual_seed(3)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g).to(torch.bfloat16).float()
    ids, w = random_routing(experts, U)
    rounded = [L.bfp4_round(t.transpose(1, 2)).transpose(1, 2) for t in (gate, up, down)]
    golden = L.partial_golden(x, ids, w.to(torch.bfloat16).float(), *rounded)
    for correction in ((1.0, 1.0), TRUNCATION_CORRECTION):
        op = RoutedExperts(device, sh, bias_correction=correction)
        for label, xin in (("x BF16", x), ("x rounded to 7 bits", round_bits(x, 7))):
            op.set_inputs(xin, ids, w)
            op(w_gu, w_dn)
            if MOCK:
                continue
            out, status = op.read_output()
            assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
            check(f"gain {share} correction {correction[0]:.5f}/{correction[1]:.5f}, {label}", golden, out, 0.999)


@pytest.mark.parametrize("stage", ["gate", "up", "silu", "act"])
def test_stage_accuracy(device, stage):
    """Where the device's arithmetic differs from exact arithmetic on the same BFP4 weights (tp32, one expert, row
    weights 1): the down weights are the identity on the chip's 64 intermediate channels, so output columns 0..63
    are the activation tile the finisher made, and the finisher is told (FIN_DEBUG) to pass on the gate sum, the up
    sum, silu(gate), or the full activation."""
    sh = SHARES["tp32"]
    gate, up, _ = random_share_weights(sh, 4)
    down = torch.zeros(4, L.HIDDEN, sh.channels)
    down[:, : sh.channels] = torch.eye(sh.channels)
    debug = {"gate": 1, "up": 2, "silu": 3, "act": 0}[stage]
    op = RoutedExperts(device, sh, defines=[("FIN_DEBUG", str(debug))] if debug else [])
    w_gu = L.to_device_stream(device, L.gate_up_stream(gate, up, sh))
    w_dn = L.to_device_stream(device, L.down_stream(down, sh))
    g = torch.Generator().manual_seed(3)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g).to(torch.bfloat16).float()
    ids, w = [2], torch.ones(L.ROWS, 1)
    op.set_inputs(x, ids, w)
    op(w_gu, w_dn)
    if MOCK:
        return
    out, status = op.read_output()
    assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
    gq = L.bfp4_round(gate.transpose(1, 2)).transpose(1, 2)[2].double()
    uq = L.bfp4_round(up.transpose(1, 2)).transpose(1, 2)[2].double()
    pg, pu = x.double() @ gq.T, x.double() @ uq.T
    want = {"gate": pg, "up": pu, "silu": torch.nn.functional.silu(pg), "act": L.swiglu(pg, pu)}[stage].float()
    got = out[:, : sh.channels]
    rel = ((got - want).abs() / want.abs().clamp(min=1e-3)).median().item()
    logger.info(
        f"HY4_MK5 stage {stage}: median relative error {rel:.3e}, rest of the output max {out[:, sh.channels:].abs().max().item():.3e}"
    )
    dump = os.environ.get("HY4_MOE_DUMP")
    if dump:  # for offline analysis of the error
        torch.save({"x": x, "gate": gate[2], "up": up[2], "out": out, "want": want}, f"{dump}/stage_{stage}.pt")
    logger.info(
        f"HY4_MK5 stage {stage}: row 0 device {[round(v, 4) for v in got[0, :6].tolist()]} exact {[round(v, 4) for v in want[0, :6].tolist()]}"
    )
    check(f"stage {stage} (tp32, one expert) vs exact arithmetic on the BFP4 weights", want, got, 0.9)  # a diagnostic
