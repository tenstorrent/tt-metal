# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the fused T = K + 1 KDA launch and the fused commit mode (SP2, ``GLM53_SPEC_COMMIT=fused``,
``_agents/spec_sp2/STATUS.md``) -- written device-free, NOT run (the chips are owned by the validation queue);
importorskip-safe.

1. Chip 0 (``TT_VISIBLE_DEVICES=0``): ``test_kda_decode_fused_op_vs_sequential`` -- the fused program at H = 16 (the
   TP4 shard), B = 4 users x K + 1 = 4 candidates (R = 16) and B = 8 x 4 (R = 32, full-tile SFPU): the DRY launch's
   ``[R, H V]`` output equals the K + 1 sequential plain launches' outputs (bits) and leaves the recurrent state and
   history byte-identical; the COMMIT launch with the step page of ``m = [0, 3, 1, 2]`` (``n_u = m_u + 1``) ends every
   user at the plain state after ``n_u`` launches (bits), its applied output rows equal the sequential ones (bits); the
   all-T page equals the plain launches' end state, a zero page leaves the state; both base variants (pre/col,
   act/rect); then the commit program captured in a trace and replayed with refreshed step pages (== eager).
2. Chip 0: ``test_fused_launch_us_table`` -- traced replays of one launch: the plain B-row launch x (K + 1), the fused
   dry launch, the fused commit launch (all T / all 1 / mixed) at B = 4 and 8, K = 1 .. 3; the us per launch logged.
3. (2,4) block: ``test_fused_commit_vs_blend_commit`` -- two ``SpecDecoder``s on one model (``blend`` = today's path,
   ``fused``): the fused verify's ids equal the blend's, the live KDA states stay byte-identical to the pre-verify
   snapshot, ``commit(m)`` (a per-user mix) ends bit-identical to the blend commit and matches the plain state after
   ``m_u + 1`` steps (PCC); the fused graphs under trace reproduce the eager fused commit (bits).
4. (2,4) block: ``test_spec_decode_lossless_fused`` -- ``SpecDecoder(commit_mode="fused").generate`` vs the plain traced
   greedy decode over 64 tokens (token-identical up to the first near-tie), acceptance and tok/s logged.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_mtp_spec_fused.py -s -k "fused_op or us_table"
    export W=/home/ttuser/ashai/glm53/tt-metal
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-7 GLM53_MODEL_B=2 GLM53_SPEC_K=3 \\
      GLM53_SPEC_COMMIT=fused pytest models/demos/glm53_flash/tests/test_mtp_spec_fused.py -s --timeout 10800 \\
      -k "commit_vs_blend or lossless_fused"
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.kda_decode.op import GATE_WORDS, KdaDecode  # noqa: E402
from models.demos.glm53_flash.tests.test_kda_decode_v2_mesh import _dev, _inputs, _to_device  # noqa: E402
from models.demos.glm53_flash.tests.test_mtp_spec import (  # noqa: E402
    DEVICE_PARAMS,
    K_ENV,
    MAX_SEQ_LEN,
    MESH,
    N_TOKENS,
    NEAR_TIE_GAP,
    B,
    _build,
    _kda_state_host,
    _plain_steps,
    _prefill_all,
    _state_close,
)
from models.demos.glm53_flash.tests.test_mtp_spec_gated import _gate_dev, _set_gate, _slabs  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.spec_decode import (  # noqa: E402
    SpecDecoder,
    commit_steps_page,
    kda_commit_steps,
    spec_draft_len,
)

K = V = 128
LB, L2_EPS, NORM_EPS = -5.0, 1e-6, 1e-5
STATE_PCC = 0.9999


# ================================================================================================ op level
def _cat_slabs(slabs):
    """The K + 1 per-candidate slabs -> the candidate-major ``[1, 1, R, W]`` qkv and gates host tensors."""
    qkv = torch.cat([q for q, _ in slabs], dim=-2)
    gates = torch.cat([g for _, g in slabs], dim=-2)
    return qkv, gates


def _fused_inputs(device, x, B_, H, T, slabs, gates_act: bool):
    """The device tensors of one fused launch: the B-slot state / history + the R-row slabs and output."""
    d = _to_device(device, x, B_, H, gates_act)  # qkv / gates / out are the B-row ones: replaced below
    qkv, gates = _cat_slabs(slabs)
    if gates_act:
        gates = KdaDecode.activate_gates(gates[0, 0], x["dt_bias"], x["a_exp"], H * K).reshape(1, 1, B_ * T, -1)
    d["qkv"] = _dev(device, qkv.to(torch.bfloat16), ttnn.bfloat16)
    d["gates"] = _dev(device, gates.contiguous())
    d["out"] = _dev(device, torch.zeros(1, 1, B_ * T, H * V).to(torch.bfloat16), ttnn.bfloat16)
    return d


def _run_fused(op, d, page):
    return op(d["qkv"], d["gates"], d["conv"], d["S"], d["taps"], d["cc"], d["onw"], d["sel"], d["out"], steps=page)


def _snap(d):
    return ttnn.to_torch(d["out"]).clone(), ttnn.to_torch(d["S"]).clone(), ttnn.to_torch(d["conv"]).clone()


def _set_slab(d, qkv_j, gates_j, gates_act, x, H):
    g = KdaDecode.activate_gates(gates_j, x["dt_bias"], x["a_exp"], H * K) if gates_act else gates_j
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(qkv_j.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), d["qkv"]
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(g.reshape(1, 1, -1, g.shape[-1]).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        d["gates"],
    )


def _sequential_reference(device, x, B_, H, slabs, gates_act, plain):
    """The plain program's launches 0 .. K on a copy of the state; snapshots (out, S, conv) after each."""
    d = _to_device(device, x, B_, H, gates_act)
    ref = []
    for qkv, g in slabs:
        _set_slab(d, qkv[0, 0], g[0, 0], gates_act, x, H)
        plain(d["qkv"], d["gates"], d["conv"], d["S"], d["taps"], d["cc"], d["onw"], d["sel"], d["out"])
        ref.append(_snap(d))
    return ref


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("gates, placement", [("pre", "col"), ("act", "rect")])
@pytest.mark.parametrize("B_, Kd", [(4, 3), (8, 3), (4, 1)])
def test_kda_decode_fused_op_vs_sequential(mesh_device, gates, placement, B_, Kd):
    H, T = 16, Kd + 1
    act = gates == "act"
    plain = KdaDecode(
        num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS, gates=gates, placement=placement
    )
    dry, commit = plain.fused_twin(T, True), plain.fused_twin(T, False)
    x = _inputs(B_, H, seed=41 + B_ + Kd)
    slabs = _slabs(x, B_, H, T, seed=42)
    ref = _sequential_reference(mesh_device, x, B_, H, slabs, act, plain)
    # ---- the dry launch: every candidate row's output == the sequential launch's row; the state untouched (bits)
    d = _fused_inputs(mesh_device, x, B_, H, T, slabs, act)
    before = (ttnn.to_torch(d["S"]).clone(), ttnn.to_torch(d["conv"]).clone())
    page = _gate_dev(mesh_device, torch.zeros(1, GATE_WORDS, dtype=torch.int32))  # never read by the dry launch
    _run_fused(dry, d, page)
    o, S, conv = _snap(d)
    assert torch.equal(S, before[0]) and torch.equal(conv, before[1]), "the dry launch wrote the state / history"
    for j in range(T):
        assert torch.equal(o[0, 0, j * B_ : (j + 1) * B_], ref[j][0][0, 0, :B_]), f"dry: candidate {j} output rows"
    logger.info(f"fused dry B={B_} K={Kd} gates={gates} placement={placement}: == {T} sequential launches (bits)")
    # ---- the commit launch with the step page of m: user u ends at the plain state after m_u + 1 launches (bits)
    m = torch.tensor([(u * 3) % T for u in range(B_)])
    n = kda_commit_steps(m)
    d = _fused_inputs(mesh_device, x, B_, H, T, slabs, act)
    _set_gate(page, commit_steps_page(m, B_, 1))
    _run_fused(commit, d, page)
    o, S, conv = _snap(d)
    for u in range(B_):
        nu = int(n[u])
        assert torch.equal(S[u], ref[nu - 1][1][u]), f"user {u}: state != plain after {nu} launches (bits)"
        assert torch.equal(conv[:, u], ref[nu - 1][2][:, u]), f"user {u}: history != plain after {nu} launches"
        for j in range(nu):
            assert torch.equal(o[0, 0, j * B_ + u], ref[j][0][0, 0, u]), f"user {u} step {j}: output row"
    # the all-T page == the plain launches' end state; the zero page leaves the state bytes
    d = _fused_inputs(mesh_device, x, B_, H, T, slabs, act)
    _set_gate(page, torch.full((1, GATE_WORDS), T, dtype=torch.int32))
    _run_fused(commit, d, page)
    _, S, conv = _snap(d)
    assert torch.equal(S, ref[-1][1]) and torch.equal(conv, ref[-1][2]), "all-T page != the sequential end state"
    d = _fused_inputs(mesh_device, x, B_, H, T, slabs, act)
    _set_gate(page, torch.zeros(1, GATE_WORDS, dtype=torch.int32))
    _run_fused(commit, d, page)
    _, S, conv = _snap(d)
    assert torch.equal(S, before[0]) and torch.equal(conv, before[1]), "the zero page changed the state"
    assert not torch.equal(ref[0][1], before[0]), "the launches change the state (the checks are not vacuous)"
    logger.info(f"fused commit B={B_} K={Kd} gates={gates} placement={placement}: per-user bit-exact vs sequential")
    # ---- the commit program in a captured trace: replays with refreshed step pages == eager launches
    if (B_, Kd) != (4, 3):
        return
    d_tr = _fused_inputs(mesh_device, x, B_, H, T, slabs, act)
    d_eg = _fused_inputs(mesh_device, x, B_, H, T, slabs, act)
    p_tr = _gate_dev(mesh_device, commit_steps_page(m, B_, 1))
    p_eg = _gate_dev(mesh_device, commit_steps_page(m, B_, 1))
    _run_fused(commit, d_tr, p_tr)  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    _run_fused(commit, d_tr, p_tr)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    _run_fused(commit, d_eg, p_eg)
    _run_fused(commit, d_eg, p_eg)
    for m2 in (torch.tensor([3, 0, 2, 1]), torch.tensor([1, 1, 1, 1]), torch.tensor([0, 0, 3, 3])):
        _set_gate(p_tr, commit_steps_page(m2, B_, 1))
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        _set_gate(p_eg, commit_steps_page(m2, B_, 1))
        _run_fused(commit, d_eg, p_eg)
        a, b = _snap(d_tr), _snap(d_eg)
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]), f"replay m={m2.tolist()}: traced != eager"
    ttnn.release_trace(mesh_device, tid)
    logger.info("fused commit: traced replays with refreshed step pages == eager")


def _traced_us(mesh_device, fn, n: int = 20) -> float:
    """Wall-clock microseconds per replay of one launch captured in a trace (compile + capture first)."""
    fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / n * 1e6
    ttnn.release_trace(mesh_device, tid)
    return us


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_launch_us_table(mesh_device):
    """us per launch (traced replays, wall clock): the plain B-row launch x (K + 1) vs the fused dry / commit launches."""
    H = 16
    rows = []
    for B_ in (4, 8):
        for Kd in (1, 2, 3):
            T = Kd + 1
            plain = KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS)
            dry, commit = plain.fused_twin(T, True), plain.fused_twin(T, False)
            x = _inputs(B_, H, seed=B_ * 10 + Kd)
            slabs = _slabs(x, B_, H, T, seed=5)
            d_plain = _to_device(mesh_device, x, B_, H, False)
            _set_slab(d_plain, slabs[0][0][0, 0], slabs[0][1][0, 0], False, x, H)
            us_plain = _traced_us(
                mesh_device,
                lambda: plain(
                    d_plain["qkv"], d_plain["gates"], d_plain["conv"], d_plain["S"], d_plain["taps"], d_plain["cc"],
                    d_plain["onw"], d_plain["sel"], d_plain["out"],
                ),  # fmt: skip
            )
            d = _fused_inputs(mesh_device, x, B_, H, T, slabs, False)
            page = _gate_dev(mesh_device, torch.full((1, GATE_WORDS), T, dtype=torch.int32))
            us_dry = _traced_us(mesh_device, lambda: _run_fused(dry, d, page))
            us_all = _traced_us(mesh_device, lambda: _run_fused(commit, d, page))
            _set_gate(page, torch.ones(1, GATE_WORDS, dtype=torch.int32))
            us_one = _traced_us(mesh_device, lambda: _run_fused(commit, d, page))
            _set_gate(page, commit_steps_page(torch.tensor([u % T for u in range(B_)]), B_, 1))
            us_mix = _traced_us(mesh_device, lambda: _run_fused(commit, d, page))
            rows.append((B_, Kd, us_plain, us_plain * T, us_dry, us_all, us_one, us_mix))
    lines = [
        "| B | K | plain launch us | (K+1) plain us | fused dry us | fused commit all-T us | commit all-1 us | commit mixed us |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r) + " |")
    logger.info("PERF-FUSED-KDA (chip 0, H=16, traced replays, wall clock):\n" + "\n".join(lines))


# ============================================================================================ block level
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_fused_commit_vs_blend_commit(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    Kd = spec_draft_len(B, K_ENV)
    assert Kd >= 1
    blend = SpecDecoder(model, mtp, Kd, commit_mode="blend")
    fused = SpecDecoder(model, mtp, Kd, commit_mode="fused")
    assert all(st.fused for st in fused.states.values() if hasattr(st, "fused")) and fused.commit_mode == "fused"
    tok, pos0 = _prefill_all(model, mtp, prompt)
    blend.warmup([pos0] * U)  # every program compiled before any capture (2026-10-26 rule)
    fused.warmup([pos0] * U)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    trace_plain, outs = model.capture(torch.full((U,), tok), [pos0] * U)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    toks, gaps, kda_plain, _ = _plain_steps(model, trace_plain, outs, tok, pos0, Kd + 1, snapshots=True)
    tokens = toks[: Kd + 1].t().contiguous()  # [U, K + 1]
    m = torch.tensor([u % (Kd + 1) for u in range(U)])
    # ---- blend (today's path), eager
    tok, pos0 = _prefill_all(model, mtp, prompt)
    pre = _kda_state_host(model)
    ids_b, _ = blend.verify_forward(tokens, [pos0] * U)
    blend.commit(m, [pos0] * U)
    after_blend = _kda_state_host(model)
    # ---- fused, eager: the verify leaves the live state untouched (bits), the commit equals the blend's (bits)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    _state_close(_kda_state_host(model), pre, "re-prefill reproduces the pre-verify state", bits=True)
    ids_f, _ = fused.verify_forward(tokens, [pos0] * U)
    assert torch.equal(ids_f, ids_b), "the verify's ids do not depend on the commit mode"
    _state_close(_kda_state_host(model), pre, "fused verify: live KDA state untouched", bits=True)
    fused.commit(m, [pos0] * U)
    after_fused = _kda_state_host(model)
    _state_close(after_fused, after_blend, "commit(mixed m): fused == blend", bits=True)
    for layer, (rec, conv) in after_fused.items():
        for u in range(U):
            rec_ref, conv_ref = kda_plain[int(m[u])][layer]
            p = min(pcc(rec[u], rec_ref[u]), pcc(conv[:, u].float(), conv_ref[:, u].float()))
            assert p > STATE_PCC, f"fused commit user {u} (m={int(m[u])}) layer {layer}: pcc {p:.6f} vs plain"
    logger.info(f"fused commit(m={m.tolist()}) == blend commit bit for bit; == plain after m_u + 1 steps (pcc)")
    # ---- the fused graphs under trace: replayed V + C reproduce the eager fused commit bit for bit
    fused.capture()
    tok, pos0 = _prefill_all(model, mtp, prompt)
    ids_t, _ = fused.verify_forward(tokens, [pos0] * U)
    assert torch.equal(ids_t, ids_f)
    _state_close(_kda_state_host(model), pre, "traced fused verify: live KDA state untouched", bits=True)
    fused.commit(m, [pos0] * U)
    _state_close(_kda_state_host(model), after_fused, "traced fused commit == eager fused commit", bits=True)
    m2 = torch.tensor([(u + 1) % (Kd + 1) for u in range(U)])
    pos2 = [p + int(mu) + 1 for p, mu in zip([pos0] * U, m)]
    fused.verify_forward(tokens, pos2)
    fused.commit(m2, pos2)
    assert all(torch.isfinite(rec).all() for rec, _ in _kda_state_host(model).values())
    fused.release()
    ttnn.release_trace(mesh_device, trace_plain)


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_spec_decode_lossless_fused(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    Kd = spec_draft_len(B, K_ENV)
    assert Kd >= 1
    spec = SpecDecoder(model, mtp, Kd, commit_mode="fused")
    tok, pos0 = _prefill_all(model, mtp, prompt)
    spec.warmup([pos0] * U)  # every program compiled before any capture
    tok, pos0 = _prefill_all(model, mtp, prompt)
    trace_plain, outs = model.capture(torch.full((U,), tok), [pos0] * U)
    spec.capture()
    n = N_TOKENS
    assert pos0 + n + Kd + 1 < MAX_SEQ_LEN
    tok, pos0 = _prefill_all(model, mtp, prompt)
    t0 = time.perf_counter()
    toks, gaps, _, _ = _plain_steps(model, trace_plain, outs, tok, pos0, n)
    plain_s = time.perf_counter() - t0
    tok, pos0 = _prefill_all(model, mtp, prompt)
    outs_spec, stats = spec.generate([tok] * U, [pos0] * U, n)
    matched = []
    for u in range(U):
        ref = toks[1:, u].tolist()
        got = outs_spec[u]
        assert len(got) == n, f"user {u}: {len(got)} tokens"
        i = next((i for i in range(n) if got[i] != ref[i]), n)
        matched.append(i)
        if i < n:
            gap = float(gaps[i, u])
            logger.info(f"user {u}: first difference at token {i} (plain top-2 gap {gap:.3f}): {got[i]} vs {ref[i]}")
            assert (
                gap < NEAR_TIE_GAP
            ), f"user {u}: fused spec diverges from plain greedy at token {i} with gap {gap:.3f}"
    plain_tps = n / plain_s
    spec_tps = n / stats.seconds if stats.seconds else float("nan")
    logger.info(
        f"SPEC-FUSED mesh {MESH} layers {len(model.layers)} users {U} K {Kd}: {stats.summary()}; plain {n} steps "
        f"{plain_s:.2f} s; tokens identical to plain greedy per user: {matched} of {n} (n = all)"
    )
    logger.info(
        f"PERF-SPEC-FUSED mesh {MESH} layers {len(model.layers)} users {U} K {Kd}: plain {plain_tps:.1f} tok/s per "
        f"user ({plain_tps * U:.0f} aggregate); spec {spec_tps:.1f} tok/s per user ({spec_tps * U:.0f} aggregate), "
        f"x{spec_tps / plain_tps:.2f}, {stats.tokens_per_iteration / U:.2f} tokens per user per iteration"
    )
    spec.release()
    ttnn.release_trace(mesh_device, trace_plain)
