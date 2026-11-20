# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rungs of fused/model below the whole mesh: a whole layer in this piece's shared-state form on ONE chip (the root
alone, no fabric connection, every partner's data written by the host) and on TWO chips of a column (the root and the
head of mesh row 1: the smallest mesh with a stage-2 reduce, a canary and real sends), and the senders' argument
tables of the three programs on all 32 chips, parse-only.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "solo and moe"     (then dense)
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k parse
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "pair_col and moe"

The attention half is fused/layer_attn's own one-chip case with virtual partners (its rung solo_x4: random
projections, the 7 column and 3 row partners' partials and every chip's queries in the landings); the MLP half is
checkpoint layer 1 (the root's 1/32 of every expert). Two steps:
  P1            root site pre-mix and norm of the attention site (step 2: with the post-mix of step 1's MLP output and
                its coefficients first), X1 as an own copy, pre, Xa, query, X3
  (host)        P2's inputs as the attention half's own rung writes them (they overwrite what P1 left)
  P2            attention, all of X4 against the host-written partners, post, X5: F on the site cores
  MLP           root site: post-mix of F with the coefficients P1 left + pre-mix and norm of the MLP site; X1'; router,
                experts and shared expert of the root's share; X6: F on the site cores
  POST          the post-mix of the MLP's F
Checked (lines start with HY4_MK11): both halves' words at rest after each program; the attention half's own checks
of P2 (layer_attn/tests/test_p2.py check_p2); y of both pre-mixes against the reference pre-mix of the streams the
host model gives; the streams after the MLP program's leading pass and after POST against the reference post-mix of
the device's own F (this is what proves the shared site, F, coefficients and streams); the MLP's x in its hub buffer
(an arena range here) bit-equal to the root's; picks against a host router; the root's partial against exact
arithmetic on its BFP4 share; F bit-equal to the partial (one chip: nothing to sum).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p2 import Case2, check_p2
from models.experimental.hy4_preview.fused.layer_mlp.tests.test_mlp_layer import (
    DenseWeights,
    Weights,
    gain,
    golden_layer,
    site_params,
    two_stage_sum,
)
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.moe import layout as ML
from models.experimental.hy4_preview.fused.projections.tests.harness import tables
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
TRACE_REGION = 64 << 20
FIRST_WAIT = 100_000_000
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)


RUNGS = {"solo": [(0, 0)], "pair_col": [(0, 0), (1, 0)]}


@galaxy
@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense"])
@pytest.mark.parametrize("rung", list(RUNGS))
def test_solo(mesh_device, rung, dense):
    """dense: the MLP half is layer 0's dense FFN (MlpLayer(dense=True)) in place of layer 1's MoE. On two chips the
    MoE fidelity is HY4_MOE_FIDELITY (default "p02") and F is the sum of the two chips' shares."""
    chips = RUNGS[rung]
    LAYER = 0 if dense else 1
    hf = golden_layer(LAYER)
    w = DenseWeights(0) if dense else Weights(LAYER)
    site_attn = site_params(LAYER, "hc_attn_layer")
    virtual = dict(virtual=(list(range(1, 8)), [1, 2, 3])) if len(chips) == 1 else {}
    case = Case2(mesh_device, chips=chips, wait_limit=FIRST_WAIT, sink_args=True, **virtual)
    a = case.a
    st = Stack(mesh_device, attn=a, wait_limit=FIRST_WAIT, moe_fidelity=os.environ.get("HY4_MOE_FIDELITY", "p02"))
    (gamma_attn,) = hy4_tensors(["model.layers.1.input_layernorm.weight"])  # (the attention case's norm)
    st.set_sites([site_attn, w.site], [gamma_attn, w.gamma])
    state = st.add_layer(
        LAYER,
        None,
        case.sink,
        0,
        1,
        dense=dense,
        router_weight=None if dense else w.router_w,
        router_bias=None if dense else w.router_b,
    )
    _, _, cos, sin = tables()
    st.set_rope(cos, sin)
    m, cfg = state.mlp, a.cfg
    logger.info(
        f"HY4_MK11 {rung}: L1 tensors {st.l1_bytes()} B per core; cb_audit rows {len(st.cb_audit(state, post=True))}"
    )
    post_prog = st.post_program()
    root = (0, 0)
    f_before = post_before = None
    for step, first in ((1, 0), (2, 40)):
        label = f"{rung} step {step}"
        p1, p2, mlp = st.layer_programs(state, post=step > 1, audit=True)
        streams_in = hf["streams_in"][first : first + 8].float()
        st.set_streams(streams_in)
        ttnn.synchronize_device(mesh_device)
        Stack.launch([p1])
        if is_mock():
            case.prefill()
            Stack.launch([p2, mlp, ("post", *post_prog)])
            continue
        ttnn.synchronize_device(mesh_device)
        failed = []
        status = a.status(p1_only=True)
        if status:
            failed.append(f"after P1: {status}")
        a.reset_words()
        # the host model of the root's site: (step 2) the post-mix of the F and coefficients the step before left
        s0 = streams_in if f_before is None else golden_post(cfg, streams_in, f_before, post_before)
        y1, _, post1, _ = golden_pre(cfg, *site_attn, s0)
        r1 = a.read_root()
        logger.info(
            f"HY4_MK11 {label} P1: y vs the reference pre-mix of the host model's streams max_abs={(y1 - r1['y']).abs().max():.3e}"
        )
        if (y1 - r1["y"]).abs().max() > 1e-4:
            failed.append("P1: y is not the pre-mix of the expected streams")
        assert not failed, f"{label}: " + "; ".join(failed)

        case.prefill()  # P2's inputs, as the attention half's own one-chip rung: queries, gates, partners' partials
        ttnn.synchronize_device(mesh_device)
        Stack.launch([p2])
        ttnn.synchronize_device(mesh_device)
        check_p2(case, "all", f"{label} P2")
        f_attn = a.read_root()["f"]

        Stack.launch([mlp])
        ttnn.synchronize_device(mesh_device)
        status = st.status()
        if status:
            failed.append(f"after the MLP program: {status}")
        s_mid = golden_post(cfg, s0, f_attn, post1)
        streams_mid = st.read_streams()
        y2, _, post2, _ = golden_pre(cfg, *w.site, s_mid)
        r2 = st.read_root()
        x_ref = rms_norm(y2, w.gamma.float(), cfg.norm_eps)
        x = r2["x"]
        logger.info(
            f"HY4_MK11 {label} MLP leading pass: streams vs the reference post-mix of the device's own attention F "
            f"max_abs={(s_mid - streams_mid).abs().max():.3e} (streams rms {s_mid.pow(2).mean().sqrt():.3e}); y vs the reference "
            f"pre-mix max_abs={(y2 - r2['y']).abs().max():.3e}; x vs the reference norm max_abs={(x_ref - x).abs().max():.3e} "
            f"(BF16 rounding alone {(x_ref - x_ref.to(torch.bfloat16).float()).abs().max():.3e})"
        )
        if (s_mid - streams_mid).abs().max() > 1e-3 * s_mid.abs().max() or (y2 - r2["y"]).abs().max() > 1e-4:
            failed.append("the MLP program's leading pass did not see the attention's F, coefficients or streams")
        chips_read = {c: st.read_mlp_chip(state, c) for c in st.topo.chips}
        chip = chips_read[root]
        if dense:
            same, union = 8, 9
            _, parts = w.ffn(x, rounded=True)
        else:
            ids, weights = chip["ids"].long(), chip["weights"]
            r_scores, r_ids, _ = ML.route_golden(x, w.router_w, w.router_b)
            same = sum(set(ids[r].tolist()) == set(r_ids[r].tolist()) for r in range(8))
            union = len(set(ids.flatten().tolist()))
            _, parts = w.moe(x, ids, weights, rounded=True, per_chip=True)
        f_mlp = r2["f"]
        x_ok = all(torch.equal(v["x"], x) for v in chips_read.values())
        scores = [
            (pcc(parts[st.topo.chip(c)], v["out"]), gain(parts[st.topo.chip(c)], v["out"]))
            for c, v in chips_read.items()
        ]
        f_ok = torch.equal(f_mlp, two_stage_sum(m, {c: v["out"] for c, v in chips_read.items()}))
        logger.info(
            f"HY4_MK11 {label} MLP{' (dense FFN)' if dense else ''}: x in the hubs' arena range bit-equal to the root's: {x_ok}; picks: "
            f"{same}/8 rows the host router's 8 experts (union {union}); each chip's partial vs exact arithmetic on its BFP4 "
            f"share pcc min {min(v[0] for v in scores):.5f} gain {min(v[1] for v in scores):.4f} to {max(v[1] for v in scores):.4f}; "
            f"F equal to the two-stage sum of the {len(chips_read)} partial(s): {f_ok}"
        )
        if not x_ok or same < 7 or min(v[0] for v in scores) < 0.97 or not f_ok:
            failed.append("the MLP half's x, picks, partial or F")

        Stack.launch([("post", *post_prog)])
        ttnn.synchronize_device(mesh_device)
        s_out = golden_post(cfg, s_mid, f_mlp, post2)
        streams_out = st.read_streams()
        logger.info(
            f"HY4_MK11 {label} POST: streams vs the reference post-mix of the device's own MLP F max_abs="
            f"{(s_out - streams_out).abs().max():.3e}"
        )
        if (s_out - streams_out).abs().max() > 1e-3 * s_out.abs().max():
            failed.append("POST: the streams are not the post-mix of the MLP's F")
        status = st.status()
        if status:
            failed.append(f"after POST: {status}")
        assert not failed, f"{label}: " + "; ".join(failed)
        f_before, post_before = f_mlp, post2
        canary = st.canary_summary()
        if st.canary is not None:
            logger.info(f"HY4_MK11 {label} canary on the heads: {canary}; per head {st.read_canary()}")
            for kind, v in canary.items():
                assert (
                    v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"] == step * (len(chips) - 1)
                ), f"{label}: canary {kind}: {v}"
    logger.info(
        f"HY4_MK11 {rung}: two steps of [P1, P2, MLP, POST] on {len(chips)} chip(s) in the shared-state form passed"
        if not is_mock()
        else f"HY4_MK11 {rung}: compiled"
    )


@galaxy
def test_parse(mesh_device):
    """All 32 chips, parse-only: the sender of every chip in P1, P2 and the MLP program of this piece's form (the MLP
    half's transport on the attention's arena) reads its argument table and writes it to a page, opens no fabric
    connection and sends nothing; every page must equal the table the host built. The other kernels run into their
    bounded waits (2 M polls) and end; the words are zeroed afterwards. custom_mm experts, random weights."""
    from models.experimental.hy4_preview.fused.model.tests.test_mock import build_random

    st = build_random(
        mesh_device,
        1,
        dense_first=False,
        small_streams=is_mock(),
        moe_layer0=1,
        wait_limit=2_000_000,
        parse_only=True,
        moe_fidelity="custom_mm",
        attn_kwargs=dict(parse_only=True),
    )
    programs = st.step_programs()
    wrong = {}
    for name, io, desc in programs:
        Stack.launch([(name, io, desc)])
        if is_mock():
            continue
        ttnn.synchronize_device(mesh_device)
        kind = name.split("[")[0]
        if kind != "post":
            bad = st.check_parsed(kind)
            want = (
                st.mt.parsed
                if kind == "mlp"
                else {c: st.a.parsed[(kind, c)] for c in st.topo.chips if (kind, c) in st.a.parsed}
            )
            logger.info(
                f"HY4_MK11 parse {name}: {len(want)} sender pages on {len(st.topo.chips)} chips, {len(bad)} differ from the host's "
                f"tables; words per page {sorted({len(v) for v in want.values()})}"
            )
            if bad:
                wrong[name] = bad[:4]
    if is_mock():
        return
    logger.info(
        f"HY4_MK11 parse: canary after the parse-only programs (its polls run out: nothing is served): {st.canary_summary()}"
    )
    st.reset()
    assert not wrong, f"senders parsed their tables differently: {wrong}"
