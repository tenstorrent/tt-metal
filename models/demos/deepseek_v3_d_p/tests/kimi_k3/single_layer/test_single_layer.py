# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3, one layer at a time, at real dims, through the production module stack (plan §1, §3).

Each case runs a global layer (or the two-layer tail) as its own ``TtKimiK3Transformer`` rank with
synthetic weights and a synthetic inherited AttnRes state, against a host reference of the same
layer (``reference.py``). The same ids run on LoudBox (2x4) and Galaxy (8x4).

Every comparison asserts PCC AND relative L2 (PCC is blind to scale). Two guards keep the oracles
honest on synthetic data: a counterfactual reference with each branch dropped must fail the bar by a
margin (``_assert_visible``) -- so a MISSING attention or FFN branch is caught; moderate drift inside
a branch is bounded only by the bars themselves -- and every AttnRes read must be neither one-hot
nor uniform (``_assert_entropy``).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.harness import CASES, SingleLayer
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference import (
    AttentionCarry,
    layer_queries,
    reference_layer,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.synthetic_state import inherited_state, request_tokens

pytestmark = run_for_blackhole()

SEED = 2026
# (PCC, relative L2) bars at ~2x the error measured on LB 2x4 at seed 2026 (plan §5.3); the worst
# measured value is in brackets. MLA layers get their own delta bar: device MLA alone measures PCC
# ~0.998 against the reference at these weights (in line with the 0.995 component bars), and that
# compounds with the MoE downstream of it, while KDA layers stay near 0.998 for the whole layer.
DELTA_KDA = (0.996, 0.13)  # the layer's own write, attention + FFN [0.99802 / 6.4e-2, L36]
# TODO(phase 3): tighten once the MoE reference is driven by device routing -- L27's error is ~2x L36's
# although both run MoE, which points at top-k flips amplified by MLA's input error, not MLA alone.
DELTA_MLA = (0.984, 0.26)  # [0.99202 / 1.31e-1, L27]
LIVE_KDA = (0.996, 0.13)  # live stream after the layer [0.99802 / 6.4e-2, L36]
LIVE_MLA = (0.984, 0.26)  # [0.99238 / 1.28e-1, L27]
OUT = (0.996, 0.13)  # q_out read + final norm [0.99798 / 6.4e-2, L92full]
CARRY = (0.9994, 0.08)  # KDA recurrent carry [0.99970 / 3.8e-2, L89]
CONV = (0.99995, 0.025)  # KDA convolution carry [0.999975 / 1.25e-2, L89]
KV = (0.9997, 0.035)  # MLA KV rows (bf8 cache) of a layer fed the exact input [0.99986 / 1.7e-2]
CHAINED_KV = (0.996, 0.13)  # ...of a layer fed by another device layer [0.99795 / 6.4e-2, T91_92kv L92]
# Synthetic AttnRes reads must be neither one-hot nor uniform (fraction of log(candidates)).
ENTROPY_RANGE = (0.3, 0.9)

MESHES = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(l1_small_size=KimiK3Config.L1_SMALL_SIZE),
        id="LB-2x4",
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(l1_small_size=KimiK3Config.L1_SMALL_SIZE),
        id="GH-8x4",
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="torus-8x4"),
    ),
]


def _rel_l2(want, got):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def _report(name, want, got, bar):
    """PCC and relative L2 against ``bar = (pcc, rel_l2)``."""
    pcc_bar, rel_bar = bar
    pcc = assert_accurate(want, got, name=name, pcc_threshold=pcc_bar)
    rel = _rel_l2(want, got)
    logger.info(f"{name}: PCC {pcc:.6f}, relative L2 {rel:.3e}")
    assert rel <= rel_bar, f"{name}: relative L2 {rel:.3e} > {rel_bar}"
    return pcc


def _assert_visible(name, observed, without, bar):
    """The oracle on ``observed`` must fail if ``without`` (a counterfactual with one branch dropped)
    were produced instead -- otherwise a missing branch hides under the others. The margin is 3x the
    bar's 1-PCC (about sqrt(3) in relative-L2 terms) or 3x its relative L2."""
    pcc_bar, rel_bar = bar
    gap_pcc, gap_rel = 1 - _pcc(observed, without), _rel_l2(observed, without)
    logger.info(f"{name}: dropping it moves 1-PCC by {gap_pcc:.3e}, relative L2 by {gap_rel:.3e}")
    assert gap_pcc >= 3 * (1 - pcc_bar) or gap_rel >= 3 * rel_bar, f"{name} is invisible to the oracle"


def _assert_entropy(name, result):
    for read, fraction in result.entropy.items():
        logger.info(f"{name} {read}-read entropy {fraction:.2f} of max")
        assert (
            ENTROPY_RANGE[0] <= fraction <= ENTROPY_RANGE[1]
        ), f"{name} {read}-read softmax entropy {fraction:.2f} of max is outside {ENTROPY_RANGE}"


def _inputs(harness, case, tokens, start):
    """Device input for the case's first layer, and the natural-order state it encodes."""
    if case.first == 0:
        ids = request_tokens(tokens, seed=SEED)
        live = harness.state["embed_weight"][ids].float()
        return harness.place_tokens(ids, start), live, torch.zeros(tokens, 0, KimiK3Config.EMB_SIZE)
    live, sealed = inherited_state(case.first, tokens, seed=SEED)
    return harness.place_state(live, sealed, start), live, sealed


def _reference(harness, case, live, sealed, rows, carries, drop=None, drop_at=None):
    """Run the case's layers on host. Every layer but the last needs all rows: the next one mixes them.
    ``drop`` removes that branch from local layer ``drop_at`` (default: the last)."""
    results = []
    drop_at = len(case.layers) - 1 if drop_at is None else drop_at
    for local, layer in enumerate(case.layers):
        last = local == len(case.layers) - 1
        result = reference_layer(
            layer,
            harness.state["layers"][local],
            layer_queries(harness.state["attn_res_weights"], layer, harness.state["attn_res_prefix"]),
            live,
            sealed,
            rows=rows if last else torch.arange(live.shape[0]),
            carry=carries[local],
            experts=harness.routed_experts(local),
            kv_only=case.kv_only and last,
            final_norm=harness.state.get("norm_weight") if last else None,
            drop=drop if local == drop_at else None,
        )
        results.append(result)
        live, sealed = result.live, result.sealed
    return results


def _check_attention_state(harness, case, results, tokens, label):
    for local, (layer, result) in enumerate(zip(case.layers, results, strict=True)):
        if result.kvpe is not None:
            bar = KV if local == 0 else CHAINED_KV
            _report(f"{label} L{layer} KV", result.kvpe, harness.mla_kv(local, tokens), bar)
        else:
            recurrent, convolution = harness.kda_carry(layer)
            want = result.carry.kda
            want_conv = torch.cat([want.q_convolution, want.k_convolution, want.v_convolution], dim=-1)[0]
            for rank, (got_r, got_c) in enumerate(zip(recurrent, convolution, strict=True)):
                _report(f"{label} L{layer} recurrent carry sp{rank}", want.recurrent[0], got_r, CARRY)
                _report(f"{label} L{layer} conv carry sp{rank}", want_conv, got_c, CONV)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
@pytest.mark.parametrize("case", [pytest.param(c, id=f"{c.name}-{c.priority}") for c in CASES])
def test_single_layer(mesh_device, device_params, case):
    harness = SingleLayer(mesh_device, case, seed=SEED)
    try:
        tokens, start = harness.chunk, 0
        inputs, live, sealed = _inputs(harness, case, tokens, start)
        got = harness.run_chunk(inputs, start)
        ttnn.deallocate(inputs)
        rows = torch.arange(tokens)
        carries = [AttentionCarry() for _ in case.layers]
        results = _reference(harness, case, live, sealed, rows, carries)
        want = results[-1]
        label = f"S1 {case.name}"
        for layer, result in zip(case.layers, results, strict=True):
            _assert_entropy(f"{label} L{layer}", result)

        _check_attention_state(harness, case, results, tokens, label)
        if case.kv_only:
            assert got.live is None and got.model_out is None, "a kv_only rank must return nothing"
            # A layer before the kv_only one is observed only through the last layer's KV rows.
            for local in range(len(case.layers) - 1):
                for branch in ("attn", "ffn"):
                    without = _reference(harness, case, live, sealed, rows, carries, drop=branch, drop_at=local)
                    _assert_visible(f"{label} L{case.layers[local]} {branch}", want.kvpe, without[-1].kvpe, CHAINED_KV)
            return

        if case.full_tail:
            _report(f"{label} model out", want.model_out, got.model_out, OUT)
            assert torch.isfinite(got.model_out).all(), "non-finite model output"
            for branch in ("attn", "ffn"):
                without = _reference(harness, case, live, sealed, rows, carries, drop=branch)[-1]
                _assert_visible(f"{label} {branch}", want.model_out, without.model_out, OUT)
            return

        layer = case.first
        # Sealed planes: inherited ones pass through untouched; a seal adds the incoming live.
        assert got.sealed.shape[1] == want.sealed.shape[1], f"{got.sealed.shape} vs {want.sealed.shape}"
        assert torch.equal(got.sealed[:, : sealed.shape[1]], sealed), "inherited planes changed"
        sealing = layer % KimiK3Config.ATTN_RES_BLOCK_SIZE == 0
        if sealing:
            assert torch.equal(got.sealed[:, -1], live), "the sealed plane is not the incoming live stream"
        # The layer's own contribution, so a large inherited residual cannot hide a broken branch.
        base = torch.zeros_like(live) if sealing else live
        is_mla = layer in KimiK3Config.mla_layer_ids()
        delta_bar, live_bar = (DELTA_MLA, LIVE_MLA) if is_mla else (DELTA_KDA, LIVE_KDA)
        delta = want.attn_delta + want.ffn_delta
        for branch in ("attn", "ffn"):
            without = _reference(harness, case, live, sealed, rows, carries, drop=branch)[-1]
            _assert_visible(f"{label} {branch}", delta, without.attn_delta + without.ffn_delta, delta_bar)
        _report(f"{label} layer delta", delta, got.live - base, delta_bar)
        _report(f"{label} live", want.live, got.live, live_bar)
        assert torch.isfinite(got.live).all(), "non-finite live stream"
    finally:
        harness.close()
