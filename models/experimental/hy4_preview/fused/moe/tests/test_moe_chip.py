# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's whole MoE share as one program (fused/moe/program.py with the router on the chip): router -> union ->
routed experts + shared expert -> the chip's partial output, with random weights.

Checks: the router's scores, the picks as exact ids per row against reference/moe.py's route (layout.route_golden),
the BF16 weights, and the partial output against the golden for the device's own picks. On the mock Galaxy
(HY4_MOCK=1) the kernels are compiled and the buffers checked; nothing runs. Log lines start with HY4_MK5."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe.program import MOCK, RoutedExperts
from models.experimental.hy4_preview.fused.moe.tests.test_routed import ROUNDED_PCC, check, random_share_weights

E = 256


def random_router(seed=0):
    g = torch.Generator().manual_seed(seed + 7)
    weight = (torch.randn(E, L.HIDDEN, generator=g) * 0.006).to(torch.bfloat16).float()
    bias = torch.randn(E, generator=g) * 0.05
    return weight, bias


def check_router(op, x, weight, bias, label):
    """Device scores / picks / weights against the reference's. Returns (device ids [8, 8], weights [8, 8])."""
    scores, ids, weights = op.read_router()
    r_scores, r_ids, r_weights = L.route_golden(x, weight, bias)
    err = (scores - r_scores).abs().max().item()
    dump = os.environ.get("HY4_MOE_DUMP")
    if dump:  # for offline analysis
        torch.save(
            {"x": x, "weight": weight, "bias": bias, "scores": scores, "ids": ids, "weights": weights},
            f"{dump}/router_{label.replace(' ', '_')}.pt",
        )
    logit = lambda s: torch.log(s.clamp(1e-6, 1 - 1e-6) / (1 - s.clamp(1e-6, 1 - 1e-6)))
    lerr = (logit(scores) - logit(r_scores)).abs()
    logger.info(
        f"HY4_MK5 router {label}: logit error per row {[round(v, 3) for v in lerr.mean(dim=1).tolist()]}, per column "
        f"tile {[round(v, 3) for v in lerr.reshape(lerr.shape[0], -1, 32).mean(dim=(0, 2)).tolist()]}"
    )
    same = sum(set(ids[r].tolist()) == set(r_ids[r].tolist()) for r in range(L.ROWS))
    ordered = sum(ids[r].tolist() == r_ids[r].tolist() for r in range(L.ROWS))
    biased = r_scores + bias
    margin = (biased.gather(1, r_ids)[:, -1] - biased.topk(9, dim=-1)[0][:, -1]).min().item()
    logger.info(
        f"HY4_MK5 router {label}: score max_abs_err={err:.3e}, rows with the reference's 8 picks {same}/8 "
        f"(in the same order {ordered}/8), smallest reference margin 8th-9th {margin:.3e}"
    )
    logger.info(f"HY4_MK5 router {label}: device ids row 0 {ids[0].tolist()} reference {r_ids[0].tolist()}")
    w_err = 0.0
    for r in range(L.ROWS):
        ref = dict(zip(r_ids[r].tolist(), r_weights[r].tolist()))
        for e, w in zip(ids[r].tolist(), weights[r].tolist()):
            if e in ref:
                w_err = max(w_err, abs(w - ref[e]) / ref[e])
    logger.info(f"HY4_MK5 router {label}: weights max relative error vs FP32 reference {w_err:.3e} (BF16: 3.9e-3)")
    assert err < 2e-3, f"router scores off by {err}"
    assert same == L.ROWS, f"picks differ from the reference in {L.ROWS - same} rows"
    assert w_err < 6e-3
    return ids, weights


@pytest.mark.parametrize("args", ["sync", "launch"])
@pytest.mark.parametrize("mode", ["side", "rect"])
@pytest.mark.parametrize("candidates", [False, True], ids=["scan", "candidates"])
@pytest.mark.parametrize("share", ["tp32"])
def test_chip_router_and_experts_random(device, share, candidates, mode, args):
    """args "launch": the streams' addresses and the router's bias travel as runtime arguments of the launch
    (RoutedExperts(stream_rt, bias_rt): one program for many layers). To show that nothing else is read, the cores'
    sync words are written for zero decoy streams and the hub's bias buffer is zeroed before the run."""
    launch = args == "launch"
    sh = L.tp_share(int(share[2:]))
    gate, up, down = random_share_weights(sh, E + 1)  # the shared expert: stream entries E and E + 1 (two terms)
    weight, bias = random_router()
    op = RoutedExperts(
        device,
        sh,
        router=True,
        num_experts=E,
        n_plain=1,
        plain_id0=E,
        plain_terms=2,
        router_candidates=candidates,
        mode=mode,
        down_per_bank=4 if mode == "rect" else 6,
        stream_rt=launch,
        bias_rt=launch,
    )
    terms = lambda w: torch.cat((w[:E], L.two_terms(w[E:])), dim=0)
    s_gu, s_dn = L.gate_up_stream(terms(gate), terms(up), sh), L.down_stream(terms(down), sh)
    w_gu = L.to_device_stream(device, s_gu)
    w_dn = L.to_device_stream(device, s_dn)
    op.set_router(weight, bias)
    if launch:
        real_router = op.w_router
        op.set_router(torch.zeros_like(weight), bias)  # a decoy router stream ...
        decoys = (
            L.to_device_stream(device, torch.zeros_like(s_gu)),
            L.to_device_stream(device, torch.zeros_like(s_dn)),
        )
        op._write_core_words(*decoys)  # ... and decoy expert streams in the cores' sync words (written once)
        op.w_router = real_router
        zero = ttnn.from_torch(torch.zeros(1, E, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(zero, op.bias)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g).to(torch.bfloat16).float()
    op.set_x(x)
    op(w_gu, w_dn)
    if MOCK:
        return
    out, status = op.read_output()
    logger.info(f"HY4_MK5 chip {share} status {[hex(v) for v in status[:6]]}")
    assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
    ids, weights = check_router(op, x, weight, bias, f"{share} random")
    union, w = L.union_of(ids, weights)
    w = torch.cat((w, torch.ones(L.ROWS, 1)), dim=1)
    union = union + [E]
    logger.info(f"HY4_MK5 chip {share}: union of {len(union) - 1} routed experts + the shared expert")
    value = lambda w: torch.cat((L.bfp4_round(w[:E].transpose(1, 2)).transpose(1, 2), L.two_terms_value(w[E:])), dim=0)
    rounded = L.partial_golden(x, union, w, value(gate), value(up), value(down), n_plain=1)
    exact = L.partial_golden(x, union, w, gate, up, down, n_plain=1)
    shared_only = [E], torch.ones(L.ROWS, 1)
    s_dev = L.partial_golden(x, *shared_only, value(gate), value(up), value(down), n_plain=1)
    s_ref = L.partial_golden(x, *shared_only, gate, up, down, n_plain=1)
    check(f"chip {share}: what two BFP4 terms cost the shared expert (host arithmetic)", s_ref, s_dev, 0.999)
    check(f"chip {share} routed + shared vs golden with BFP4-rounded weights", rounded, out, ROUNDED_PCC)
    check(f"chip {share} routed + shared vs golden with exact weights", exact, out, 0.97)


@pytest.mark.parametrize("base", [40, 128], ids=lambda b: f"experts{b}")
def test_chip_router_and_experts_ep_random(device, base):
    """Expert parallel: the chip owns the 8 whole experts [base, base + 8); the router runs over all 256 and the
    chip streams those of its own that the 8 rows picked."""
    sh = L.ep_share()
    gate, up, down = random_share_weights(sh, 8)
    weight, bias = random_router()
    bias[base : base + 4] += 0.3  # some of the chip's experts get picked
    op = RoutedExperts(device, sh, router=True, num_experts=E, expert_base=base, expert_count=8)
    w_gu = L.to_device_stream(device, L.gate_up_stream(gate, up, sh))
    w_dn = L.to_device_stream(device, L.down_stream(down, sh))
    op.set_router(weight, bias)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g).to(torch.bfloat16).float()
    op.set_x(x)
    op(w_gu, w_dn)
    if MOCK:
        return
    out, status = op.read_output()
    assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
    ids, weights = check_router(op, x, weight, bias, f"ep experts {base}..{base + 7}")
    union, w = L.union_of(ids, weights, expert_base=base, expert_count=8)
    logger.info(f"HY4_MK5 chip ep: {len(union)} of the chip's 8 experts picked: {union}")
    assert len(union) > 0
    rounded = L.partial_golden(
        x,
        union,
        w,
        L.bfp4_round(gate.transpose(1, 2)).transpose(1, 2),
        L.bfp4_round(up.transpose(1, 2)).transpose(1, 2),
        L.bfp4_round(down.transpose(1, 2)).transpose(1, 2),
    )
    check(f"chip ep experts {base}.. vs golden with BFP4-rounded weights", rounded, out, ROUNDED_PCC)
