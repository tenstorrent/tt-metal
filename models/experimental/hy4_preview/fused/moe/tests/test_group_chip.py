# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip of the "8 chips per group" layout (fused/moe/prepare.py tp8g4, fused/layer_mlp/PLAN.md section 15) as the
MLP half builds it: RoutedExperts with a 1/8 share (P 8, S 1), the router over all 256 experts, the union restricted
to the chip's group [expert_base, expert_base + 64) and numbered from expert_base, with and without the plain slot
(the shared expert as entries 64, 65 of the chip's streams, two BFP4 terms), rectangular placement, 4 down cores
per bank. Random weights; the router is crafted (expert e reads hidden channel e) so that the picks are known.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    HY4_MOE_BOARD=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x

Cases (picks of the 8 rows; what the chip of group ``base // 64`` then runs):
  mixed    random x: every row's 8 best of 256, about a quarter of them in the group
  empty    every pick outside the group: no slot (U = 0) without the plain slot: the partial must be zero, bit for bit;
           the plain slot alone (U = 1) with it
  full     8 rows x 8 distinct picks, all 64 of them the group's experts: 64 slots, 65 with the plain slot (the
           buffers' limit: max_slots 65, 520 of MAX_UNITS 640 units)
Checked: status words empty, the picks as exact ids per row against reference/moe.py's route, the partial against
the golden on the BFP4-rounded weights for the device's own picks (layout.union_of(base, count): the hub's mirror).
The output buffer is filled with 3.0 by the host before the run, so a zero partial is zeros the chip wrote.
Log lines start with HY4_TP8."""


import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe.program import MOCK, RoutedExperts
from models.experimental.hy4_preview.fused.moe.tests.test_routed import ROUNDED_PCC, check, random_share_weights

E = 256
COUNT = 64  # experts of a group
SHARE = L.tp_share(8)


def crafted_x(case, base, seed=3):
    """x [8, 6144] whose channels 0..255 make the eye router pick what the case wants."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g)
    if case == "mixed":
        return x.to(torch.bfloat16).float()
    x[:, :E] = -4.0
    for r in range(L.ROWS):
        first = base + 8 * r if case == "full" else (base + COUNT + 8 * r) % E  # empty: the next group's experts
        x[r, first : first + 8] = 4.0
    return x.to(torch.bfloat16).float()


@pytest.mark.parametrize("fidelity", ["custom_mm", "p02"])
@pytest.mark.parametrize("plain", [0, 1], ids=["routed", "plain"])
@pytest.mark.parametrize("base", [64, 192], ids=lambda b: f"base{b}")
@pytest.mark.parametrize("case", ["mixed", "empty", "full"])
def test_group_chip(device, case, base, plain, fidelity):
    host_round = fidelity != "custom_mm"  # (one rounding, what the goldens' bfp4_round is: layout.to_device_stream)
    gate, up, down = random_share_weights(SHARE, COUNT + 1)  # entry 64: the shared expert (stored as two terms)
    op = RoutedExperts(
        device,
        SHARE,
        router=True,
        num_experts=E,
        expert_base=base,
        expert_count=COUNT,
        n_plain=1,  # (the buffers are sized for the plain slot in both forms, as MlpLayer builds them)
        plain_id0=COUNT,
        plain_terms=2,
        router_candidates=True,
        mode="rect",
        down_per_bank=4,
        **({} if fidelity == "custom_mm" else {"fidelity": fidelity}),
    )
    op.n_plain = plain  # hub_router.cpp's compile-time argument 14, as MlpLayer._group_args sets it chip by chip
    assert op.max_slots == COUNT + 1 and op.max_units == (COUNT + 1) * SHARE.P
    terms = lambda w: torch.cat((w[:COUNT], L.two_terms(w[COUNT:])), dim=0)
    w_gu = L.to_device_stream(device, L.gate_up_stream(terms(gate), terms(up), SHARE), host_round=host_round)
    w_dn = L.to_device_stream(device, L.down_stream(terms(down), SHARE), host_round=host_round)
    weight, bias = torch.eye(E, L.HIDDEN), torch.zeros(E)
    op.set_router(weight, bias)
    x = crafted_x(case, base)
    op.set_x(x)
    # the output buffer holds something else before the run: a zero partial is then one the chip wrote
    stale = ttnn.from_torch(
        torch.full((L.K_TILES, 2 * 16 * L.ROWS), 3.0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.copy_host_to_device_tensor(stale, op.out)
    op(w_gu, w_dn)
    label = f"group chip {case} base {base} plain {plain} [{fidelity}]"
    if MOCK:
        logger.info(f"HY4_TP8 {label}: compiled (mock)")
        return
    out, status = op.read_output()
    logger.info(f"HY4_TP8 {label}: status {[hex(v) for v in status[:6]]}")
    assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
    _, ids, weights = op.read_router()
    r_scores, r_ids, r_w = L.route_golden(x, weight, bias)
    same = sum(set(ids[r].tolist()) == set(r_ids[r].tolist()) for r in range(L.ROWS))
    # a pick may differ from the reference's only at a near tie (random x: the 8th and 9th of 256 scores can be
    # closer than the device's scores are to the host's): the worst device pick at most 2e-3 below the host's cut
    cut = r_scores.topk(8, dim=-1).values[:, -1]
    below = (cut - r_scores.gather(1, ids.long()).min(dim=-1).values).max().item()
    logger.info(
        f"HY4_TP8 {label}: rows with the reference's 8 picks {same}/8, the worst device pick {below:.2e} below the reference's cut"
    )
    assert below <= 2e-3 and (same == L.ROWS or case == "mixed"), f"picks differ from the reference: {ids.tolist()}"
    union, w = L.union_of(ids.long(), weights, expert_base=base, expert_count=COUNT)
    routed = len(union)
    if plain:
        union, w = union + [COUNT], torch.cat((w, torch.ones(L.ROWS, 1)), dim=1)
    logger.info(f"HY4_TP8 {label}: {routed} of the group's 64 experts picked, {len(union)} slots")
    if case == "empty":
        assert routed == 0
    if case == "full":
        assert routed == COUNT
    if not union:
        nonzero = int((out != 0).sum())
        logger.info(f"HY4_TP8 {label}: empty union: {nonzero} of {out.numel()} elements of the partial are not zero")
        assert nonzero == 0
        return
    value = lambda w: torch.cat(
        (L.bfp4_round(w[:COUNT].transpose(1, 2)).transpose(1, 2), L.two_terms_value(w[COUNT:])), dim=0
    )
    rounded = L.partial_golden(x, union, w, value(gate), value(up), value(down), n_plain=plain)
    exact = L.partial_golden(x, union, w, gate, up, down, n_plain=plain)
    check(f"{label} vs golden with BFP4-rounded weights", rounded, out, ROUNDED_PCC)
    check(f"{label} vs golden with exact weights", exact, out, 0.97)
