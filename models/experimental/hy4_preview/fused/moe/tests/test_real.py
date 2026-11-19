# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The streamed decode MoE with checkpoint weights (layers 1 to 5, and the dense FFN of layer 0), on ONE chip, against
reference/moe.py: the chip runs the share of each of the 32 chips of a TP 32 split in turn (their prepared weight
streams, fused/moe/prepare.py) and the host sums the 32 partial outputs, which is what the cross-chip reduce does.

The inputs are tests/test_moe.py's (32 random tokens, seed 2), 8 rows per pass, so the numbers stand next to the ttnn
form's (TTNN_ROUTED_PCC, BFP4 routed experts on the 8x4 Galaxy).

    routed   the routed experts on the reference's routing            vs Hy4MoE.routed
    shared   the shared expert alone (two BFP4 terms; BFP8 in the ttnn form) vs Hy4MoE.shared
    chip     router -> union -> experts + shared expert, one program  vs the reference following the device's picks
    dense    layer 0's FFN as 9 plain slots (two BFP4 terms)          vs Hy4FFN

Needs $HY4_HF_MODEL and the prepared streams ($HY4_MOE_STREAMS, default generated/hy4_preview/moe_streams).
Board only. Log lines start with HY4_MK5."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare
from models.experimental.hy4_preview.fused.moe.program import MOCK, RoutedExperts
from models.experimental.hy4_preview.fused.moe.tests.test_routed import check
from models.experimental.hy4_preview.tests.common import pcc

STREAMS = os.environ.get("HY4_MOE_STREAMS", prepare.DEFAULT_OUT)
SPLIT, CHIPS = "tp32", 32
E = 256
# tests/test_moe.py: routed experts, BFP4, checkpoint layers 1 to 5, the same 32 tokens (8x4 Galaxy, 2026-11-19)
TTNN_ROUTED_PCC = {1: 0.9887, 2: 0.9877, 3: 0.9850, 4: 0.9825, 5: 0.9821}
CHIP_LIST = [int(c) for c in os.environ.get("HY4_MOE_CHIPS", "").split(",") if c] or list(range(CHIPS))


def tokens(n=32, seed=2):
    """tests/test_moe.py _tokens(HY4, 32, 2)."""
    return torch.randn(n, L.HIDDEN, generator=torch.Generator().manual_seed(seed))


def _need(layer):
    if MOCK:
        pytest.skip("needs the board")
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    if not prepare.stream_paths(STREAMS, layer, SPLIT, 0)[0].is_file():
        pytest.skip(f"no prepared streams for layer {layer} under {STREAMS} (fused/moe/prepare.py)")


def reference_sum(x, ids_rows, w_rows, gate_up, down, plain=False):
    """Hy4MoE.routed for per-row picks (ids_rows [T, K] entry ids, w_rows [T, K]) with the checkpoint's weights
    (gate_up [N, 4096, 6144], down [N, 6144, 2048], BF16), in fp64. plain: no clamp (the shared expert, the FFN)."""
    out = torch.zeros(x.shape[0], L.HIDDEN, dtype=torch.float64)
    I = L.INTERMEDIATE
    xd = x.double()
    for e in sorted(set(ids_rows.flatten().tolist())):
        rows = (ids_rows == e).any(dim=-1).nonzero().flatten()
        w = (w_rows * (ids_rows == e))[rows].sum(dim=-1, keepdim=True).double()
        gu = xd[rows] @ gate_up[e].double().T
        act = L.swiglu(gu[:, :I], gu[:, I:], float("inf") if plain else 10.0) * w
        out[rows] += act @ down[e].double().T
    return out.float()


def run_shares(device, layer, passes):
    """passes: [(op, setter(op, group), label)]. Runs every pass for every chip's share and every group of 8 rows;
    returns {label: summed output [32, 6144]}."""
    sums = {label: torch.zeros(32, L.HIDDEN) for _, _, label in passes}
    for chip in CHIP_LIST:
        w_gu, w_dn = prepare.load_streams(device, STREAMS, layer, SPLIT, chip)
        for op, setter, label in passes:
            for g in range(4):
                setter(op, g)
                op(w_gu, w_dn)
                out, status = op.read_output()
                assert status[1:6] == [
                    0,
                    0,
                    0,
                    0,
                    0,
                ], f"{label} chip {chip}: a core timed out {[hex(v) for v in status[:6]]}"
                sums[label][8 * g : 8 * g + 8] += out
        ttnn.deallocate(w_gu)
        ttnn.deallocate(w_dn)
    return sums


@pytest.mark.parametrize("layer", range(1, 6), ids=lambda layer: f"layer{layer}")
def test_moe_real_tp32(device, layer):
    _need(layer)
    share = L.tp_share(32)
    gate_up, down = prepare.layer_experts(layer, shared=True)
    prefix = f"model.layers.{layer}.mlp.gate."
    router_w, router_b = prepare.checkpoint_tensors([prefix + "weight", prefix + "e_score_correction_bias"])
    x = tokens()
    r_scores, r_idx, r_w = L.route_golden(x, router_w, router_b)
    r_routed = reference_sum(x, r_idx, r_w, gate_up, down)
    shared_ids = torch.full((32, 1), E)
    r_shared = reference_sum(x, shared_ids, torch.ones(32, 1), gate_up, down, plain=True)

    op = RoutedExperts(device, share, plain_terms=2)
    op_r = RoutedExperts(device, share, router=True, num_experts=E, n_plain=1, plain_id0=E, plain_terms=2)
    op_r.set_router(router_w, router_b)
    unions = [L.union_of(r_idx[8 * g : 8 * g + 8], r_w[8 * g : 8 * g + 8]) for g in range(4)]
    logger.info(f"HY4_MK5 real L{layer}: unions of the 4 groups of 8 rows {[len(u[0]) for u in unions]}")

    def set_routed(o, g):
        o.set_inputs(x[8 * g : 8 * g + 8], *unions[g])

    def set_shared(o, g):
        o.set_inputs(x[8 * g : 8 * g + 8], [E], torch.ones(L.ROWS, 1), n_plain=1)

    passes = [(op, set_routed, "routed"), (op, set_shared, "shared")]
    sums = run_shares(device, layer, passes)
    score = pcc(r_routed, sums["routed"])
    check(f"real L{layer} routed experts (TP32 shares summed) vs reference", r_routed, sums["routed"], 0.975)
    logger.info(
        f"HY4_MK5 real L{layer} routed experts: pcc={score:.4f}, the ttnn form on the same inputs "
        f"{TTNN_ROUTED_PCC[layer]:.4f} ({'not lower' if score >= TTNN_ROUTED_PCC[layer] - 5e-4 else 'LOWER'})"
    )
    check(
        f"real L{layer} shared expert (2 BFP4 terms, TP32 shares summed) vs reference", r_shared, sums["shared"], 0.998
    )

    # the whole chip program: the router's picks come from the device, the same on every share
    d_idx = torch.zeros(32, 8, dtype=torch.long)
    d_w = torch.zeros(32, 8)
    out = torch.zeros(32, L.HIDDEN)
    agree = 0
    for chip in CHIP_LIST:
        w_gu, w_dn = prepare.load_streams(device, STREAMS, layer, SPLIT, chip)
        for g in range(4):
            op_r.set_x(x[8 * g : 8 * g + 8])
            op_r(w_gu, w_dn)
            o, status = op_r.read_output()
            assert status[1:6] == [0, 0, 0, 0, 0], f"chip {chip}: a core timed out {[hex(v) for v in status[:6]]}"
            out[8 * g : 8 * g + 8] += o
            if chip == CHIP_LIST[0]:
                scores, ids, weights = op_r.read_router()
                d_idx[8 * g : 8 * g + 8], d_w[8 * g : 8 * g + 8] = ids, weights
                err = (scores - r_scores[8 * g : 8 * g + 8]).abs().max().item()
                logger.info(f"HY4_MK5 real L{layer} router rows {8 * g}..{8 * g + 7}: score max_abs_err={err:.3e}")
        ttnn.deallocate(w_gu)
        ttnn.deallocate(w_dn)
    agree = sum(set(d_idx[r].tolist()) == set(r_idx[r].tolist()) for r in range(32))
    biased = r_scores + router_b.float()
    cut = biased.topk(8, dim=-1).values[:, -1]
    below = (cut - biased.gather(1, d_idx).min(dim=-1).values).max().item()
    margin = (cut - biased.topk(9, dim=-1).values[:, -1]).min().item()
    r_w_for = r_scores.gather(1, d_idx)
    r_w_for = r_w_for / r_w_for.sum(dim=-1, keepdim=True) * 2.827
    w_off = ((d_w - r_w_for).abs() / r_w_for).max().item()
    logger.info(
        f"HY4_MK5 real L{layer} router: {agree}/32 rows have exactly the reference's 8 picks; the worst device pick "
        f"is {below:.2e} below the reference's cut (smallest reference margin 8th-9th {margin:.2e}); weights within "
        f"{w_off:.2e} relative of the reference's for the device's picks (BF16)"
    )
    r_followed = reference_sum(x, d_idx, r_w_for, gate_up, down) + r_shared
    check(
        f"real L{layer} chip program (router + experts + shared, TP32 shares summed) vs reference",
        r_followed,
        out,
        0.975,
    )


def test_dense_real_layer0(device):
    """Layer 0's dense FFN (6144 -> 18432 -> 6144) as 9 plain slots of the TP 32 share, two BFP4 terms each."""
    _need(0)
    gate_up, down = prepare.dense_experts(0, terms=1)
    x = tokens()
    n = gate_up.shape[0]
    ids = torch.arange(n).repeat(32, 1)
    ref = reference_sum(x, ids, torch.ones(32, n), gate_up, down, plain=True)
    op = RoutedExperts(device, L.tp_share(32), plain_terms=2)

    def set_dense(o, g):  # slice j is stream entries 2 j (first term) and 2 j + 1
        o.set_inputs(x[8 * g : 8 * g + 8], [2 * j for j in range(n)], torch.ones(L.ROWS, n), n_plain=n)

    sums = run_shares(device, 0, [(op, set_dense, "dense")])
    check("real L0 dense FFN (2 BFP4 terms, TP32 shares summed) vs reference", ref, sums["dense"], 0.998)
