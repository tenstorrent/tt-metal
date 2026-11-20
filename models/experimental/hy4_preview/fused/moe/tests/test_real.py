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
from models.experimental.hy4_preview.fused.moe.tests.test_routed import FID, FID_KW, check
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


def run_shares(device, layer, passes, after=None):
    """passes: [(op, setter(op, group), label)]. Runs every pass for every chip's share and every group of 8 rows
    (the chip's streams are loaded once); returns {label: summed output [32, 6144]}. after(op, label, chip, group,
    out) is called after each run."""
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
                if after is not None:
                    after(op, label, chip, g, out)
        ttnn.deallocate(w_gu)
        ttnn.deallocate(w_dn)
    return sums


def gain(ref, dev):
    return (ref.double() * dev.double()).sum().item() / (ref.double() ** 2).sum().item()


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

    kw = dict(mode="rect", down_per_bank=4) if os.environ.get("HY4_MOE_RECT", "1") == "1" else {}
    kw.update(FID_KW)  # HY4_MOE_FIDELITY: the expert matmuls through hifi_mm (program.py FIDELITY)
    tag = "" if FID == "custom_mm" else f" [{FID}]"
    op = RoutedExperts(device, share, plain_terms=2, **kw)
    op_r = RoutedExperts(device, share, router=True, num_experts=E, n_plain=1, plain_id0=E, plain_terms=2, **kw)
    op_r.set_router(router_w, router_b)
    unions = [L.union_of(r_idx[8 * g : 8 * g + 8], r_w[8 * g : 8 * g + 8]) for g in range(4)]
    logger.info(f"HY4_MK5 real L{layer}: unions of the 4 groups of 8 rows {[len(u[0]) for u in unions]}")

    def set_routed(o, g):
        o.set_inputs(x[8 * g : 8 * g + 8], *unions[g])

    def set_shared(o, g):
        o.set_inputs(x[8 * g : 8 * g + 8], [E], torch.ones(L.ROWS, 1), n_plain=1)

    def set_chip(o, g):
        o.set_x(x[8 * g : 8 * g + 8])

    # the whole chip program: the router's picks come from the device, the same on every share
    d_idx = torch.zeros(32, 8, dtype=torch.long)
    d_w = torch.zeros(32, 8)
    d_scores = torch.zeros(32, E)
    first = CHIP_LIST[0]
    first_out = {label: torch.zeros(32, L.HIDDEN) for label in ("routed", "shared", "chip")}  # one share on its own

    def after(o, label, chip, g, out):
        if chip == first:
            first_out[label][8 * g : 8 * g + 8] = out
        if label != "chip":
            return
        scores, ids, weights = o.read_router()
        if chip == first:
            d_scores[8 * g : 8 * g + 8], d_idx[8 * g : 8 * g + 8], d_w[8 * g : 8 * g + 8] = scores, ids, weights
        else:  # every share runs the same router on the same x: the same picks
            assert torch.equal(
                ids, d_idx[8 * g : 8 * g + 8]
            ), f"chip {chip} rows {8 * g}..: picks differ from chip {first}'s"

    passes = [(op, set_routed, "routed"), (op, set_shared, "shared"), (op_r, set_chip, "chip")]
    sums = run_shares(device, layer, passes, after)

    dump = os.environ.get("HY4_MOE_DUMP")
    if dump:  # for the comparison with exact arithmetic on the BFP4 weights, done off the board
        torch.save(
            {
                "x": x, "r_idx": r_idx, "r_w": r_w, "d_idx": d_idx, "d_w": d_w, "d_scores": d_scores, **sums,
                "first_chip": first, **{f"first_{label}": v for label, v in first_out.items()},
            },  # fmt: skip
            f"{dump}/real_L{layer}.pt",
        )
    score = pcc(r_routed, sums["routed"])
    check(f"real L{layer}{tag} routed experts (TP32 shares summed) vs reference", r_routed, sums["routed"], 0.975)
    logger.info(
        f"HY4_MK5 real L{layer}{tag} routed experts: pcc={score:.4f} gain={gain(r_routed, sums['routed']):.4f}; the BFP4 "
        f"format alone (exact arithmetic) and the ttnn form on the same inputs: {TTNN_ROUTED_PCC[layer]:.4f} "
        f"({'not lower' if score >= TTNN_ROUTED_PCC[layer] - 5e-4 else 'LOWER'})"
    )
    check(
        f"real L{layer}{tag} shared expert (2 BFP4 terms, TP32 shares summed) vs reference",
        r_shared,
        sums["shared"],
        0.99,
    )
    logger.info(f"HY4_MK5 real L{layer}{tag} shared expert: gain={gain(r_shared, sums['shared']):.4f}")

    # router: exact picks per row against the host router (reference/moe.py route) on the x the device has (BF16)
    b_scores, b_idx, _ = L.route_golden(x.to(torch.bfloat16).float(), router_w, router_b)
    score_err = (d_scores - b_scores).abs().max().item()
    same = [set(d_idx[r].tolist()) == set(b_idx[r].tolist()) for r in range(32)]
    ordered = sum(d_idx[r].tolist() == b_idx[r].tolist() for r in range(32))
    unrounded = sum(set(d_idx[r].tolist()) == set(r_idx[r].tolist()) for r in range(32))
    biased = b_scores + router_b.float()
    top9 = biased.topk(9, dim=-1).values
    margins = top9[:, 7] - top9[:, 8]
    cut = top9[:, 7]
    below = (cut - biased.gather(1, d_idx).min(dim=-1).values).max().item()
    r_w_for = b_scores.gather(1, d_idx)
    r_w_for = r_w_for / r_w_for.sum(dim=-1, keepdim=True) * 2.827
    w_off = ((d_w - r_w_for).abs() / r_w_for).max().item()
    logger.info(
        f"HY4_MK5 real L{layer}{tag} router: {sum(same)}/32 rows have exactly the reference's 8 picks ({ordered}/32 in the "
        f"same order); score max_abs_err={score_err:.2e}; smallest reference margin 8th-9th {margins.min().item():.2e}; "
        f"margins of the rows that differ {[f'{margins[r].item():.1e}' for r in range(32) if not same[r]]}; the worst "
        f"device pick is {below:.2e} below the reference's cut; weights within {w_off:.2e} relative (BF16); "
        f"against the router on the unrounded x {unrounded}/32"
    )
    r_followed = reference_sum(x, d_idx, r_w_for, gate_up, down) + r_shared
    check(
        f"real L{layer}{tag} chip program (router + experts + shared, TP32 shares summed) vs reference",
        r_followed,
        sums["chip"],
        0.975,
    )
    logger.info(f"HY4_MK5 real L{layer}{tag} chip program: gain={gain(r_followed, sums['chip']):.4f}")


def test_dense_real_layer0(device):
    """Layer 0's dense FFN (6144 -> 18432 -> 6144) as 9 plain slots of the TP 32 share, two BFP4 terms each."""
    _need(0)
    gate_up, down = prepare.dense_experts(0, terms=1)
    x = tokens()
    n = gate_up.shape[0]
    ids = torch.arange(n).repeat(32, 1)
    ref = reference_sum(x, ids, torch.ones(32, n), gate_up, down, plain=True)
    op = RoutedExperts(device, L.tp_share(32), plain_terms=2, **FID_KW)

    def set_dense(o, g):  # slice j is stream entries 2 j (first term) and 2 j + 1
        o.set_inputs(x[8 * g : 8 * g + 8], [2 * j for j in range(n)], torch.ones(L.ROWS, n), n_plain=n)

    sums = run_shares(device, 0, [(op, set_dense, "dense")])
    tag = "" if FID == "custom_mm" else f" [{FID}]"
    check(f"real L0{tag} dense FFN (2 BFP4 terms, TP32 shares summed) vs reference", ref, sums["dense"], 0.99)
    logger.info(f"HY4_MK5 real L0{tag} dense FFN: gain={gain(ref, sums['dense']):.4f}")
    dump = os.environ.get("HY4_MOE_DUMP")
    if dump:
        torch.save({"x": x, "dense": sums["dense"]}, f"{dump}/real_L0.pt")
