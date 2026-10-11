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

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.harness import CASES, LOCAL_ROWS, SingleLayer
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference import (
    AttentionCarry,
    layer_queries,
    reference_layer,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import moe_forward, moe_route
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
# MoE alone, reference driven by the DEVICE's routing, so gate tie-breaks are out of the error.
MOE_ROUTED = (0.9983, 0.12)  # [0.99913 / 6.0e-2, L12, L87]; a mis-dispatch on one chip moves 2e-2 / 0.22
# Share of (token, slot) picks the device gate makes that the reference gate does not (plan §5.2).
# Same bf16 input both sides; the rest is fp32 accumulation order on near-tied scores (896 sigmoid
# scores leave the 16th/17th close).
MAX_FLIP_RATE = 0.0175  # [0.864%, L87]
# Weights of tokens whose picks agree; they sit near 1/16 with little spread, so PCC is a weak
# measure here and relative L2 is the binding bar.
GATE_WEIGHTS = (0.994, 0.004)  # [0.99701 / 1.94e-3]
# A request of a few tokens: one or two rows, so a single top-k flip on one token moves the whole
# comparison. These requests exercise plumbing (T=1/2 through KDA, MLA, MoE padding); the full-chunk
# scenarios carry the accuracy bars.
SHORT = (0.98, 0.25)  # [0.99073 / 1.40e-1, 1- and 2-token requests]
SHORT_ROWS = 32
# Synthetic AttnRes reads must be neither one-hot nor uniform (fraction of log(candidates)).
ENTROPY_RANGE = (0.3, 0.9)

TRACE_REGION_SIZE = 256 * 1024 * 1024  # S6; the runner's default (PREFILL_TRACE_REGION_SIZE)
MESHES = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(l1_small_size=KimiK3Config.L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE),
        id="LB-2x4",
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(l1_small_size=KimiK3Config.L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE),
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


def _check_moe(harness, result, start, label, local=0):
    """The layer's MoE on the reference's own FFN input, eagerly with intermediates (plan §5.2, S9).

    * the device gate's top-k against the reference gate's: flip rate bounded;
    * dispatch: no chip's dispatched-token sum exceeds the buffer (overflow drops tokens silently);
    * the MoE output against the reference driven by the device's routing -- tight, since gate
      tie-breaks are taken out;
    * S9(a): the same oracle must FAIL when the reference's expert ids are shifted by one, or when
      only the experts living on one chip are swapped -- proof it can see a mis-dispatch.
    """
    state = harness.state["layers"][local]
    experts = harness.routed_experts(local)
    diag = harness.moe_diagnostic(result.ffn_input, start, local)
    dev_idx, dev_w = diag["indices"], diag["weights"]
    k = dev_idx.shape[1]
    # The reference gate on the bf16 input the device gate actually received.
    gate = state["gate_weights"]
    ref_idx, ref_w = moe_route(result.ffn_input.bfloat16().float(), gate["weight"], gate["e_score_correction_bias"], k)
    missing = sum(len(set(d.tolist()) - set(r.tolist())) for d, r in zip(dev_idx, ref_idx))
    flip_rate = missing / dev_idx.numel()
    logger.info(
        f"{label} gate: {missing} of {dev_idx.numel()} device picks differ from the reference ({flip_rate:.3%})"
    )
    assert flip_rate <= MAX_FLIP_RATE, f"{label} gate flip rate {flip_rate:.3%} > {MAX_FLIP_RATE:.1%}"
    assert diag["max_per_chip_dispatch"] <= diag["dispatch_capacity"], f"{label}: dispatch overflow {diag}"
    agree = (dev_idx.sort(-1).values == ref_idx.sort(-1).values).all(-1)
    _report(
        f"{label} gate weights (tokens with identical picks)",
        ref_w.gather(1, ref_idx.argsort(-1))[agree],
        dev_w.gather(1, dev_idx.argsort(-1))[agree],
        GATE_WEIGHTS,
    )

    eps = KimiK3Config.RMS_NORM_EPS
    # The device ran every row; the reference MoE (and the controls below) evaluates the sample.
    rows = _sample_rows(harness, dev_idx.shape[0], start)
    x, dev_idx, dev_w, dev_out = result.ffn_input[rows], dev_idx[rows], dev_w[rows], diag["output"][rows]
    want, _ = moe_forward(x, state, experts, top_k=k, eps=eps, routing=(dev_idx, dev_w))
    _report(f"{label} MoE (device-routed)", want, dev_out, MOE_ROUTED)

    num_experts = KimiK3Config.NUM_ROUTED_EXPERTS
    per_chip = num_experts // harness.mesh_device.get_num_devices()
    shifted = (dev_idx + 1) % num_experts
    chip0 = dev_idx < per_chip  # experts held by global device 0 only: shift within that chip
    one_chip = torch.where(chip0, (dev_idx + 1) % per_chip, dev_idx)
    for name, idx in (("all ids shifted", shifted), ("chip-0 ids permuted", one_chip)):
        wrong, _ = moe_forward(x, state, experts, top_k=k, eps=eps, routing=(idx, dev_w))
        gap_pcc, gap_rel = 1 - _pcc(wrong, dev_out), _rel_l2(wrong, dev_out)
        logger.info(f"{label} negative control ({name}): 1-PCC {gap_pcc:.3e}, relative L2 {gap_rel:.3e}")
        assert (
            gap_pcc >= 3 * (1 - MOE_ROUTED[0]) or gap_rel >= 3 * MOE_ROUTED[1]
        ), f"{label}: MoE oracle blind to {name}"


@dataclass
class Request:
    """One request's per-position input in natural order, long enough for any chunk placed in it
    (the tail past a chunk's real end is padding the device must ignore)."""

    live: torch.Tensor  # [P, d]
    sealed: torch.Tensor  # [P, S, d]
    ids: torch.Tensor | None = None  # [P] token ids (layer 0 only)


def _request(harness, case, seed):
    positions = harness.max_seq_len + harness.chunk
    if case.first == 0:
        ids = request_tokens(positions, seed=seed)
        live = harness.state["embed_weight"][ids].float()
        return Request(live, torch.zeros(positions, 0, KimiK3Config.EMB_SIZE), ids)
    live, sealed = inherited_state(case.first, positions, seed=seed)
    return Request(live, sealed)


def _run(harness, request, start, end, slot=0):
    """One chunk at ``start`` whose real tokens end at ``end``."""
    chunk = harness.chunk
    if request.ids is not None:
        inputs = harness.place_tokens(request.ids[start : start + chunk], start)
    else:
        inputs = harness.place_state(request.live[start : start + chunk], request.sealed[start : start + chunk], start)
    out = harness.run_chunk(inputs, start, end, slot)
    ttnn.deallocate(inputs)
    return out


SAMPLE_ROWS = 512


def _sample_rows(harness, valid, start=0):
    """Rows the reference evaluates after token mixing (exact for those rows, plan §5.5): all of a
    short chunk; otherwise an even spread plus the rows at every place the device layout breaks
    natural order -- each SP shard's first and last row and, at a rotated ``start``, both sides of
    each rotation seam -- where an off-by-one in the row rotation would show first."""
    if valid <= SAMPLE_ROWS:
        return torch.arange(valid)
    perm = harness.permutation(start)  # device row -> natural row of the chunk
    shard_edges = torch.tensor(
        [r for shard in range(harness.sp) for r in (shard * harness.local_rows, (shard + 1) * harness.local_rows - 1)]
    )
    seams = torch.nonzero(perm[1:] - perm[:-1] != 1).flatten()
    edges = torch.cat([perm[shard_edges], perm[seams], perm[seams + 1]]).unique()
    edges = edges[edges < valid]
    spread = torch.linspace(0, valid - 1, SAMPLE_ROWS - len(edges)).long()
    return torch.cat([spread, edges]).unique()


def _reference_request(harness, case, request, spans, sample=False):
    """The case's layers over a request's chunks, carries flowing; one result list per chunk.

    Only the LAST chunk's output is compared, so earlier chunks evaluate a single row past token
    mixing (the carries and KV still cover every token). With ``sample`` the last chunk evaluates
    ``_sample_rows`` too; ``results[-1].rows`` says which rows each result holds."""
    carries = [AttentionCarry() for _ in case.layers]
    per_chunk = []
    for index, (start, end) in enumerate(spans):
        last = index == len(spans) - 1
        rows = (
            (_sample_rows(harness, end - start, start) if sample else torch.arange(end - start))
            if last
            else torch.tensor([0])
        )
        results = _reference(harness, case, request.live[start:end], request.sealed[start:end], rows, carries)
        for result in results:
            result.rows = rows
        carries = [result.carry for result in results]
        per_chunk.append(results)
    return per_chunk


def _snapshot(harness, case, outs, spans, slot=0):
    """Everything a request leaves behind, on host: every chunk's output (its real rows) and the
    attention state -- KV rows, and the KDA carries of every SP rank."""
    snap = {}
    for index, (out, (start, end)) in enumerate(zip(outs, spans, strict=True)):
        for key in ("live", "sealed", "model_out"):
            value = getattr(out, key)
            snap[f"{key}{index}"] = None if value is None else value[: end - start]
    end = spans[-1][1]
    for local, layer in enumerate(case.layers):
        if harness.model.schedule.local_is_mla(local):
            snap[f"kv{local}"] = harness.mla_kv(local, end, slot)
        else:
            recurrent, convolution = harness.kda_carry(layer, slot)
            snap[f"carry{local}"] = (*recurrent, *convolution)
    return snap


def _assert_snapshots_equal(a, b, label):
    assert a.keys() == b.keys(), f"{label}: {sorted(a)} != {sorted(b)}"
    for key in a:
        x, y = a[key], b[key]
        if x is None:
            assert y is None, f"{label}: {key} present in one run only"
            continue
        if isinstance(x, tuple):
            assert all(torch.equal(u, v) for u, v in zip(x, y, strict=True)), f"{label}: {key} not bitwise"
            continue
        assert torch.equal(x, y), f"{label}: {key} not bitwise"


def _run_spans(harness, request, spans, slot=0):
    return [_run(harness, request, start, end, slot) for start, end in spans]


def _check_chunk(harness, case, request, start, end, got, results, label, guard_rows=None):
    """A chunk's output against its reference, over the rows the reference holds (``rows``); the
    pass-through checks are bitwise over every real row. ``guard_rows``: also hold the oracle to its
    bar on exactly the rows the S1 guards proved it sensitive on (``want`` must hold every row)."""
    valid = end - start
    want = results[-1]
    rows = getattr(want, "rows", torch.arange(valid))
    if case.kv_only:
        assert got.live is None and got.model_out is None, "a kv_only rank must return nothing"
        return
    short = valid < SHORT_ROWS
    if case.full_tail:
        _report(f"{label} model out", want.model_out, got.model_out[rows], SHORT if short else OUT)
        if guard_rows is not None:
            _report(f"{label} model out (guard rows)", want.model_out[guard_rows], got.model_out[guard_rows], OUT)
        assert torch.isfinite(got.model_out[:valid]).all(), f"{label}: non-finite model output"
        return
    assert torch.isfinite(got.live[:valid]).all(), f"{label}: non-finite live stream"
    live, sealed = request.live[start:end], request.sealed[start:end]
    assert got.sealed.shape[1] == want.sealed.shape[1], f"{label}: {got.sealed.shape[1]} sealed planes"
    assert torch.equal(got.sealed[:valid, : sealed.shape[1]], sealed), f"{label}: inherited planes changed"
    sealing = case.first % KimiK3Config.ATTN_RES_BLOCK_SIZE == 0
    if sealing:
        assert torch.equal(got.sealed[:valid, -1], live), f"{label}: the sealed plane is not the incoming live stream"
    live, got_live = live[rows], got.live[rows]
    base = torch.zeros_like(live) if sealing else live
    delta_bar, live_bar = (DELTA_MLA, LIVE_MLA) if case.first in KimiK3Config.mla_layer_ids() else (DELTA_KDA, LIVE_KDA)
    if short:
        delta_bar = live_bar = SHORT
    _report(f"{label} layer delta", want.attn_delta + want.ffn_delta, got_live - base, delta_bar)
    _report(f"{label} live", want.live, got_live, live_bar)
    if guard_rows is not None:
        delta = want.attn_delta + want.ffn_delta
        _report(f"{label} layer delta (guard rows)", delta[guard_rows], (got_live - base)[guard_rows], delta_bar)


def _check_state(harness, case, per_chunk, end, label, slot=0):
    """KV rows of the whole request so far, and the KDA carries after its last chunk."""
    for local, layer in enumerate(case.layers):
        if harness.model.schedule.local_is_mla(local):
            want = torch.cat([results[local].kvpe for results in per_chunk])
            _report(f"{label} L{layer} KV", want, harness.mla_kv(local, end, slot), KV if local == 0 else CHAINED_KV)
        else:
            want = per_chunk[-1][local].carry.kda
            want_conv = torch.cat([want.q_convolution, want.k_convolution, want.v_convolution], dim=-1)[0]
            recurrent, convolution = harness.kda_carry(layer, slot)
            for rank, (got_r, got_c) in enumerate(zip(recurrent, convolution, strict=True)):
                _report(f"{label} L{layer} recurrent carry sp{rank}", want.recurrent[0], got_r, CARRY)
                _report(f"{label} L{layer} conv carry sp{rank}", want_conv, got_c, CONV)


def _observed(case, result, rows=None):
    """What the device test compares for a case: layer delta, model output, or the last layer's KV
    (which always covers every token). ``rows`` picks a sample out of a result holding all rows."""
    if case.kv_only:
        return result.kvpe
    value = result.model_out if case.full_tail else result.attn_delta + result.ffn_delta
    return value if rows is None else value[rows]


def _bar(case):
    if case.kv_only:
        return CHAINED_KV
    if case.full_tail:
        return OUT
    return DELTA_MLA if case.first in KimiK3Config.mla_layer_ids() else DELTA_KDA


def _guards(harness, case, request, results, label):
    """S1 only: entropy of every read; every branch of every layer visible to its oracle; and (S9b) a
    corrupted inherited AttnRes state visible too -- the live stream swapped with a sealed plane, and
    one sealed plane replaced by another. (Reordering the sealed set is not a control: the read's
    softmax is a function of the candidate SET.)"""
    for layer, result in zip(case.layers, results, strict=True):
        _assert_entropy(f"{label} L{layer}", result)
    tokens = harness.chunk
    live, sealed = request.live[:tokens], request.sealed[:tokens]
    # Counterfactuals on the row sample (exact for those rows); `want` holds every row of the chunk.
    rows = _sample_rows(harness, tokens)
    fresh = [AttentionCarry() for _ in case.layers]
    want = results[-1]
    for local in range(len(case.layers)):
        last = local == len(case.layers) - 1
        for branch in ("attn", "ffn"):
            if last and case.kv_only:
                continue  # a kv_only layer has no FFN, and its attention output is never read
            without = _reference(harness, case, live, sealed, rows, fresh, drop=branch, drop_at=local)[-1]
            _assert_visible(
                f"{label} L{case.layers[local]} {branch}",
                _observed(case, want, rows),
                _observed(case, without),
                _bar(case),
            )
    if sealed.shape[1] == 0:
        return
    corruptions = []
    # At a sealing layer the live stream joins the sealed set straight after the pre-read, so
    # swapping it with a sealed plane leaves every read's candidate set unchanged -- not a control.
    # A kv_only layer's observable (its KV) depends on the pre-read alone, which is a function of the
    # candidate SET -- so for it only a changed set is a control, not a swap.
    if case.first % KimiK3Config.ATTN_RES_BLOCK_SIZE and not case.kv_only:
        swapped_live, swapped_sealed = sealed[:, 0].clone(), sealed.clone()
        swapped_sealed[:, 0] = live
        corruptions.append(("live <-> sealed plane 0", swapped_live, swapped_sealed))
    if sealed.shape[1] >= 2:
        duplicated = sealed.clone()
        duplicated[:, 1] = sealed[:, 0]
        corruptions.append(("sealed plane 1 := plane 0", live, duplicated))
    for name, bad_live, bad_sealed in corruptions:
        without = _reference(harness, case, bad_live, bad_sealed, rows, fresh)[-1]
        _assert_visible(f"{label} S9b {name}", _observed(case, want, rows), _observed(case, without), _bar(case))


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


# ------------------------------------------------------------------------------------- scenarios


def _s1_and_s5(harness, case):
    """S1: one chunk at 0 against the reference, with every guard and the MoE checks.
    S5: the same request twice more on the same model -- bitwise identical each time."""
    request = _request(harness, case, SEED)
    chunk = harness.chunk
    got = _run(harness, request, 0, chunk)
    first = _snapshot(harness, case, [got], [(0, chunk)])
    (results,) = _reference_request(harness, case, request, [(0, chunk)])
    label = f"S1 {case.name}"
    _guards(harness, case, request, results, label)
    _check_chunk(harness, case, request, 0, chunk, got, results, label, guard_rows=_sample_rows(harness, chunk))
    _check_state(harness, case, [results], chunk, label)
    for local, result in enumerate(results):
        if result.routing is not None:
            _check_moe(harness, result, 0, f"{label} L{case.layers[local]}", local)
    for repeat in (1, 2):
        again = _snapshot(harness, case, [_run(harness, request, 0, chunk)], [(0, chunk)])
        _assert_snapshots_equal(first, again, f"S5 {case.name} repeat {repeat}")
    logger.info(f"S5 {case.name}: three runs bitwise identical")


def _s2(harness, case):
    """S2: a request continued into a second chunk -- chunk-aligned, and from rotated starts."""
    chunk = harness.chunk
    for split in (chunk, 32, harness.chunk // harness.sp + 32):
        request = _request(harness, case, SEED + split)
        spans = [(0, split), (split, split + chunk)]
        for start, end in spans:
            got = _run(harness, request, start, end)
        per_chunk = _reference_request(harness, case, request, spans, sample=True)
        label = f"S2 {case.name} split {split}"
        start, end = spans[-1]
        _check_chunk(harness, case, request, start, end, got, per_chunk[-1], label)
        _check_state(harness, case, per_chunk, end, label)


def _s4(harness, case):
    """S4: a chunk that ends mid-chunk, and 1- and 2-token requests; padding rows are ignored."""
    chunk = harness.chunk
    for end in (chunk - 37, 1, 2):
        request = _request(harness, case, SEED + 1000 + end)
        got = _run(harness, request, 0, end)
        per_chunk = _reference_request(harness, case, request, [(0, end)], sample=True)
        label = f"S4 {case.name} end {end}"
        _check_chunk(harness, case, request, 0, end, got, per_chunk[-1], label)
        _check_state(harness, case, per_chunk, end, label)


def _s3(harness, case):
    """S3: a request is unaffected by whatever request ran before it on the same slot.

    B (two chunks) -> A (two chunks) -> B again must be bitwise identical to the first B --
    outputs, carries, KV rows -- and continuing B's second chunk exposes a wrong final carry. Short
    restarts: a 1- and a 2-token B after two different A's must also be identical."""
    chunk = harness.chunk
    spans = [(0, chunk), (chunk, 2 * chunk)]
    b, a = _request(harness, case, SEED + 7), _request(harness, case, SEED + 8)

    def run(request, spans):
        return _snapshot(harness, case, _run_spans(harness, request, spans), spans)

    first_b = run(b, spans)
    run(a, spans)
    _assert_snapshots_equal(first_b, run(b, spans), f"S3 {case.name} B after A")
    other = _request(harness, case, SEED + 9)
    for tokens in (1, 2):
        run(a, spans)
        after_a = run(b, [(0, tokens)])
        run(other, spans)
        after_other = run(b, [(0, tokens)])
        _assert_snapshots_equal(after_a, after_other, f"S3 {case.name} {tokens}-token restart")
    logger.info(f"S3 {case.name}: restarts bitwise identical")


def _kda_layers(case):
    return [layer for layer in case.layers if layer not in KimiK3Config.mla_layer_ids()]


def _one_shot(mesh_device, case):
    """The S8 one-shot side: a two-chunk request run as ONE chunk of twice the size, on its own model
    (built and closed before any other model of the case; see ``test_single_layer``)."""
    one_shot = SingleLayer(mesh_device, case, seed=SEED, chunks=1, local_rows=2 * LOCAL_ROWS)
    try:
        request = _request(one_shot, case, SEED + 21)
        whole = _run(one_shot, request, 0, one_shot.chunk)
        carries = {layer: one_shot.kda_carry(layer) for layer in _kda_layers(case)}
    finally:
        one_shot.close()
    return request, whole, carries


def _s8_one_shot(harness, case, one_shot):
    """S8: the two-chunk request matches the one-shot run of the same tokens -- both chunks' outputs
    and every SP rank's carries."""
    request, whole, whole_carries = one_shot
    chunk = harness.chunk
    outs = _run_spans(harness, request, [(0, chunk), (chunk, 2 * chunk)])
    for layer in _kda_layers(case):
        chunked, one = harness.kda_carry(layer), whole_carries[layer]
        for rank, (r1, c1, r2, c2) in enumerate(zip(*chunked, *one, strict=True)):
            _report(f"S8 {case.name} L{layer} recurrent sp{rank}: chunked vs one-shot", r2, r1, CARRY)
            _report(f"S8 {case.name} L{layer} conv sp{rank}: chunked vs one-shot", c2, c1, CONV)
    if whole.live is not None:
        for index, out in enumerate(outs):
            rows = slice(index * chunk, (index + 1) * chunk)
            _report(f"S8 {case.name} chunk {index + 1} live: chunked vs one-shot", whole.live[rows], out.live, LIVE_KDA)
    logger.info(f"S8 {case.name}: chunked matches one-shot")


def _s7_s8_slots(harness, case):
    """S7: two requests interleaved chunk by chunk on two slots (eager) are bitwise what each produces
    alone, and slot 1's request matches the reference (a slot-1 offset bug would be self-consistent).
    S8 (KDA layers): carry buffers never move across requests and the migration slab round-trips
    bit-exactly. Needs a two-slot model (traced K3 is single-slot, so S6 cannot share it)."""
    chunk = harness.chunk
    spans = [(0, chunk), (chunk, 2 * chunk)]
    a, b = _request(harness, case, SEED + 21), _request(harness, case, SEED + 22)
    addresses = {slot: harness.carry_addresses(slot) for slot in (0, 1)} if _kda_layers(case) else None

    alone = {}
    for slot, request in ((0, a), (1, b)):
        alone[slot] = _snapshot(harness, case, _run_spans(harness, request, spans, slot), spans, slot)

    outs = {0: [], 1: []}
    for start, end in spans:
        for slot, request in ((0, a), (1, b)):
            outs[slot].append(_run(harness, request, start, end, slot))
    for slot in (0, 1):
        interleaved = _snapshot(harness, case, outs[slot], spans, slot)
        _assert_snapshots_equal(alone[slot], interleaved, f"S7 {case.name} slot {slot}")
    logger.info(f"S7 {case.name}: interleaved slots bitwise equal to each request alone")
    per_chunk = _reference_request(harness, case, b, spans, sample=True)
    label = f"S7 {case.name} slot 1"
    _check_chunk(harness, case, b, chunk, 2 * chunk, outs[1][-1], per_chunk[-1], label)
    _check_state(harness, case, per_chunk, 2 * chunk, label, slot=1)

    if addresses is None:
        return
    for slot in (0, 1):
        assert harness.carry_addresses(slot) == addresses[slot], f"S8 {case.name}: slot {slot} carries moved"
        for layer in _kda_layers(case):
            assert harness.slab_round_trip(layer, slot), f"S8 {case.name} L{layer} slot {slot}: slab not exact"
    logger.info(f"S8 {case.name}: carry addresses stable across requests, slabs round-trip exactly")


def _s6_traced(harness, case):
    """S6: one capture with on-device (slot, start, end) metadata, replayed for continuation chunks,
    a dirty restart, an unaligned end, 1- and 2-token requests and a rotated start. Traced and eager
    are different programs (eager KDA takes host bounds, the capture reads device metadata), yet they
    measure bitwise equal on LB, so every replay is held bitwise to the eager run of the same chunks."""
    chunk = harness.chunk

    def place(request, start):
        if request.ids is not None:
            return harness.place_tokens(request.ids[start : start + chunk], start)
        return harness.place_state(request.live[start : start + chunk], request.sealed[start : start + chunk], start)

    def eager(request, spans):
        return _snapshot(harness, case, _run_spans(harness, request, spans), spans)

    def traced(request, spans):
        outs = []
        for start, end in spans:
            inputs = place(request, start)
            outs.append(harness.replay(inputs, start, end))
            ttnn.deallocate(inputs)
        return _snapshot(harness, case, outs, spans)

    r, other = _request(harness, case, SEED + 31), _request(harness, case, SEED + 32)
    harness.prepare_trace(place(r, 0))
    harness.capture_trace()
    try:
        spans = [(0, chunk), (chunk, 2 * chunk)]
        first = traced(r, spans)
        _assert_snapshots_equal(eager(r, spans), first, f"S6 {case.name} traced vs eager")
        _assert_snapshots_equal(first, traced(r, spans), f"S6 {case.name} replay vs replay")
        traced(other, spans)
        _assert_snapshots_equal(first, traced(r, spans), f"S6 {case.name} traced restart")
        for spans in ([(0, chunk - 37)], [(0, 1)], [(0, 2)], [(0, 32), (32, 32 + chunk)]):
            _assert_snapshots_equal(eager(r, spans), traced(r, spans), f"S6 {case.name} traced {spans}")
        logger.info(f"S6 {case.name}: traced replays bitwise equal to eager and to each other")
    finally:
        harness.release_trace()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
@pytest.mark.parametrize("case", [pytest.param(c, id=f"{c.name}-{c.priority}") for c in CASES])
def test_single_layer(mesh_device, device_params, case):
    """Every scenario of a case in one test, so each model is built once (plan §5.3); building is
    most of a case's device cost, and the synthetic weights are shared by all of them.

    Models, in build order (each closed before the next):
    - P0 KDA cases: the one-shot model of S8 (twice the chunk). It comes first: built AFTER a
      two-slot model that had run a forward, a model of a different chunk size computed a wrong MoE
      (L36: MoE alone PCC 0.954 vs 0.999) -- reproduced, not a program-cache or stale-DRAM effect;
      plan §7.1.
    - The case's model: S1-S5 against the reference, then S8 (chunked vs one-shot) and S6 (traced,
      last: the capture pins the model's buffers) on P0.
    - P0 cases: a two-slot model for S7/S8 (traced K3 is single-slot)."""
    p0 = case.priority == "P0"
    one_shot = _one_shot(mesh_device, case) if p0 and _kda_layers(case) else None
    harness = SingleLayer(mesh_device, case, seed=SEED)
    try:
        _s1_and_s5(harness, case)
        _s2(harness, case)
        _s4(harness, case)
        _s3(harness, case)
        if one_shot is not None:
            _s8_one_shot(harness, case, one_shot)
        if p0:
            _s6_traced(harness, case)
    finally:
        harness.close()
    if p0:
        slots = SingleLayer(mesh_device, case, seed=SEED, slot_num=2)
        try:
            _s7_s8_slots(slots, case)
        finally:
            slots.close()
