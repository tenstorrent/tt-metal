# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the gated KDA commit (SP1, ``GLM53_SPEC_COMMIT=gated``, ``_agents/spec_sp1/STATUS.md``) -- written
device-free, NOT run (the chips are owned by the validation queue); importorskip-safe.

1. Chip 0 (``TT_VISIBLE_DEVICES=0``): ``test_kda_decode_gated_op_vs_plain`` -- the gated ``kda_decode`` program at
   B = 4, H = 16 (the TP4 shard) over K + 1 = 4 launches with the commit gates of ``m = [0, 3, 1, 2]``
   (``gate_j[u] = m_u >= j``) on ONE state vs the plain program's launches on a copy: every user's recurrent state and
   conv history after the 4 gated launches equal the plain state after its own ``m_u + 1`` launches, bit for bit, and
   the gated users' output rows equal the plain ones per launch; the all-ones gate is the plain program bit for bit
   (output included), the all-zeros gate leaves the state bytes; both gate modes / placements of the base program;
   then the gated program captured in a trace and replayed with refreshed gate pages (== the eager launches).
2. (2,4) block: ``test_gated_commit_vs_blend_commit`` -- two ``SpecDecoder``s on the same model (``blend`` = today's
   path, ``gated``): the gated verify leaves the live KDA states byte-identical to the pre-verify snapshot while its
   scratch equals the blend path's post-verify state; after ``commit(m)`` (a per-user mix) the live states of both
   modes are bit-identical and match the plain state after ``m_u + 1`` steps (PCC); the gated graphs captured in
   traces reproduce the eager gated commit bit for bit.
3. (2,4) block: ``test_spec_decode_lossless_gated`` -- ``SpecDecoder(commit_mode="gated").generate`` vs the plain
   traced greedy decode over 64 tokens (token-identical up to the first near-tie), acceptance and tok/s logged.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_mtp_spec_gated.py -s -k gated_op
    export W=/home/ttuser/ashai/glm53/tt-metal
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-7 GLM53_MODEL_B=2 GLM53_SPEC_K=3 \\
      GLM53_SPEC_COMMIT=gated pytest models/demos/glm53_flash/tests/test_mtp_spec_gated.py -s --timeout 10800 \\
      -k "commit_vs_blend or lossless_gated"
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.kda_decode.op import GATE_WORDS, KdaDecode  # noqa: E402
from models.demos.glm53_flash.tests.test_kda_decode_v2_mesh import _golden, _inputs, _to_device  # noqa: E402
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
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.spec_decode import (  # noqa: E402
    SpecDecoder,
    commit_gate_rows,
    kda_commit_gates,
    spec_draft_len,
)

K = V = 128
LB, L2_EPS, NORM_EPS = -5.0, 1e-6, 1e-5
STATE_PCC = 0.9999


# ================================================================================================ op level
def _gate_dev(device, page: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        page.to(torch.int32).contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _set_gate(dev: ttnn.Tensor, page: torch.Tensor):
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(page.to(torch.int32).contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), dev
    )


def _set_slab(d, qkv: torch.Tensor, gates: torch.Tensor, gates_act: bool, x, H: int):
    g = KdaDecode.activate_gates(gates, x["dt_bias"], x["a_exp"], H * K) if gates_act else gates
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(qkv.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), d["qkv"]
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(g.reshape(1, 1, -1, g.shape[-1]).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        d["gates"],
    )


def _snap(d):
    return (
        ttnn.to_torch(d["out"]).clone(),
        ttnn.to_torch(d["S"]).clone(),
        ttnn.to_torch(d["conv"]).clone(),
    )


def _slabs(x, B_, H, n, seed):
    gen = torch.Generator().manual_seed(seed)
    W = 3 * H * K
    Hb = -(-H // 32) * 32
    out = []
    for _ in range(n):
        qkv = torch.randn(1, 1, B_, W, generator=gen).to(torch.bfloat16).float()
        gates = torch.zeros(1, 1, B_, H * K + H * V + Hb)
        gates[..., : H * K] = torch.randn(B_, H * K, generator=gen) * 2.0
        gates[..., H * K : H * K + H * V] = torch.randn(B_, H * V, generator=gen) * 2.0
        gates[..., H * K + H * V : H * K + H * V + H] = torch.randn(B_, H, generator=gen) * 2.0
        out.append((qkv, gates))
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("gates, placement", [("pre", "col"), ("act", "rect")])
def test_kda_decode_gated_op_vs_plain(mesh_device, gates, placement):
    B_, H, Kd = 4, 16, 3
    plain = KdaDecode(
        num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS, gates=gates, placement=placement
    )
    gated = plain.gated_twin()
    act = gates == "act"
    x = _inputs(B_, H, seed=31)
    slabs = _slabs(x, B_, H, Kd + 1, seed=32)
    m = torch.tensor([0, 3, 1, 2])
    pages = kda_commit_gates(m, Kd)  # [K+1, B]
    # the plain reference: launches 0 .. K on a copy, snapshots after each
    d_ref = _to_device(mesh_device, x, B_, H, act)
    ref = []
    for qkv, g in slabs:
        _set_slab(d_ref, qkv[0, 0], g[0, 0], act, x, H)
        plain(
            d_ref["qkv"],
            d_ref["gates"],
            d_ref["conv"],
            d_ref["S"],
            d_ref["taps"],
            d_ref["cc"],
            d_ref["onw"],
            d_ref["sel"],
            d_ref["out"],
        )
        ref.append(_snap(d_ref))
    o_g, S_g, conv_g = _golden(dict(x, qkv=slabs[0][0], gates=slabs[0][1]), B_, H)
    assert pcc(S_g, ref[0][1]) > STATE_PCC, "the plain program vs its golden (sanity)"
    # the gated program on ONE state with the commit gates
    d = _to_device(mesh_device, x, B_, H, act)
    gate_dev = _gate_dev(mesh_device, commit_gate_rows(pages[0], B_, 1))
    outs = []
    for j, (qkv, g) in enumerate(slabs):
        _set_slab(d, qkv[0, 0], g[0, 0], act, x, H)
        _set_gate(gate_dev, commit_gate_rows(pages[j], B_, 1))
        gated(d["qkv"], d["gates"], d["conv"], d["S"], d["taps"], d["cc"], d["onw"], d["sel"], d["out"], gate=gate_dev)
        outs.append(_snap(d))
        # the gated users' output rows equal the plain launch's (their live state equals the reference's)
        for u in range(B_):
            if int(pages[j, u]):
                assert torch.equal(outs[j][0][0, 0, u], ref[j][0][0, 0, u]), f"launch {j} user {u}: output row"
    S_fin, conv_fin = outs[-1][1], outs[-1][2]
    for u in range(B_):
        mu = int(m[u])
        assert torch.equal(S_fin[u], ref[mu][1][u]), f"user {u}: state != plain after {mu + 1} launches (bits)"
        assert torch.equal(conv_fin[:, u], ref[mu][2][:, u]), f"user {u}: conv history != plain after {mu + 1} launches"
    # all-ones gate = the plain program, bit for bit (output included); all-zeros = the input bytes
    for ones in (True, False):
        d1 = _to_device(mesh_device, x, B_, H, act)
        d2 = _to_device(mesh_device, x, B_, H, act)
        _set_slab(d1, slabs[1][0][0, 0], slabs[1][1][0, 0], act, x, H)
        _set_slab(d2, slabs[1][0][0, 0], slabs[1][1][0, 0], act, x, H)
        _set_gate(gate_dev, torch.full((1, GATE_WORDS), 1 if ones else 0, dtype=torch.int32))
        gated(
            d1["qkv"],
            d1["gates"],
            d1["conv"],
            d1["S"],
            d1["taps"],
            d1["cc"],
            d1["onw"],
            d1["sel"],
            d1["out"],
            gate=gate_dev,
        )
        before = (ttnn.to_torch(d2["S"]).clone(), ttnn.to_torch(d2["conv"]).clone())
        plain(d2["qkv"], d2["gates"], d2["conv"], d2["S"], d2["taps"], d2["cc"], d2["onw"], d2["sel"], d2["out"])
        a, b = _snap(d1), _snap(d2)
        if ones:
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]), "all-ones != plain"
        else:
            assert torch.equal(a[1], before[0]) and torch.equal(a[2], before[1]), "all-zeros changed the state"
            assert not torch.equal(b[1], before[0]), "the plain launch changes the state (the check is not vacuous)"
    logger.info(
        f"gated kda_decode gates={gates} placement={placement}: per-user commit bit-exact vs the plain launches"
    )
    # the gated program in a captured trace: replays with refreshed gate pages == eager launches
    d_tr = _to_device(mesh_device, x, B_, H, act)
    d_eg = _to_device(mesh_device, x, B_, H, act)
    gate_tr = _gate_dev(mesh_device, commit_gate_rows(pages[0], B_, 1))
    gate_eg = _gate_dev(mesh_device, commit_gate_rows(pages[0], B_, 1))
    run = lambda dd, gg: gated(
        dd["qkv"], dd["gates"], dd["conv"], dd["S"], dd["taps"], dd["cc"], dd["onw"], dd["sel"], dd["out"], gate=gg
    )
    _set_slab(d_tr, slabs[0][0][0, 0], slabs[0][1][0, 0], act, x, H)
    run(d_tr, gate_tr)  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run(d_tr, gate_tr)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    # the eager twin takes the same three launches (compile + capture-run + replays) on the same gates / slabs
    _set_slab(d_eg, slabs[0][0][0, 0], slabs[0][1][0, 0], act, x, H)
    run(d_eg, gate_eg)
    run(d_eg, gate_eg)
    for j in (1, 2, 3):
        _set_slab(d_tr, slabs[j][0][0, 0], slabs[j][1][0, 0], act, x, H)
        _set_gate(gate_tr, commit_gate_rows(pages[j], B_, 1))
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        _set_slab(d_eg, slabs[j][0][0, 0], slabs[j][1][0, 0], act, x, H)
        _set_gate(gate_eg, commit_gate_rows(pages[j], B_, 1))
        run(d_eg, gate_eg)
        a, b = _snap(d_tr), _snap(d_eg)
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]), f"replay {j}: traced gated launch != eager"
    ttnn.release_trace(mesh_device, tid)
    logger.info("gated kda_decode: traced replays with refreshed gate pages == eager")


# ============================================================================================ block level
def _bk_state_host(spec: SpecDecoder) -> dict:
    """{layer: (recurrent, conv)} of the KDA scratch / backup tensors (column 0's chips)."""
    out = {}
    rows = spec.mesh.rows
    for layer, st in spec.states.items():
        if hasattr(st, "bk"):
            rec = spec.mesh.to_host(st.bk.recurrent, rows=0 if rows > 1 else None)
            conv = spec.mesh.to_host(st.bk.convolution, rows=1 if rows > 1 else None)
            out[layer] = (rec.float().clone(), conv.clone())
    return out


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_gated_commit_vs_blend_commit(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    Kd = spec_draft_len(B, K_ENV)
    assert Kd >= 1
    blend = SpecDecoder(model, mtp, Kd, commit_mode="blend")
    gated = SpecDecoder(model, mtp, Kd, commit_mode="gated")
    assert all(st.gated for st in gated.states.values() if hasattr(st, "gated")) and gated.commit_mode == "gated"
    tok, pos0 = _prefill_all(model, mtp, prompt)
    # every program compiled before any capture (2026-10-26 rule): both decoders' graphs, then the plain trace
    blend.warmup([pos0] * U)
    gated.warmup([pos0] * U)
    tok, pos0 = _prefill_all(model, mtp, prompt)
    trace_plain, outs = model.capture(torch.full((U,), tok), [pos0] * U)
    # the plain reference states after 1 .. K + 1 steps
    tok, pos0 = _prefill_all(model, mtp, prompt)
    toks, gaps, kda_plain, _ = _plain_steps(model, trace_plain, outs, tok, pos0, Kd + 1, snapshots=True)
    tokens = toks[: Kd + 1].t().contiguous()  # [U, K + 1]
    m = torch.tensor([u % (Kd + 1) for u in range(U)])
    # ---- blend (today's path), eager
    tok, pos0 = _prefill_all(model, mtp, prompt)
    pre = _kda_state_host(model)
    ids_b, _ = blend.verify_forward(tokens, [pos0] * U)
    post_v_blend = _kda_state_host(model)
    blend.commit(m, [pos0] * U)
    after_blend = _kda_state_host(model)
    # ---- gated, eager: the verify leaves the live state untouched, its scratch holds the post-verify state
    tok, pos0 = _prefill_all(model, mtp, prompt)
    _state_close(_kda_state_host(model), pre, "re-prefill reproduces the pre-verify state", bits=True)
    ids_g, _ = gated.verify_forward(tokens, [pos0] * U)
    assert torch.equal(ids_g, ids_b), "the verify's ids do not depend on the commit mode"
    _state_close(_kda_state_host(model), pre, "gated verify: live KDA state untouched", bits=True)
    _state_close(_bk_state_host(gated), post_v_blend, "gated verify: scratch == blend's post-verify state", bits=True)
    gated.commit(m, [pos0] * U)
    after_gated = _kda_state_host(model)
    _state_close(after_gated, after_blend, "commit(mixed m): gated == blend", bits=True)
    for layer, (rec, conv) in after_gated.items():
        for u in range(U):
            rec_ref, conv_ref = kda_plain[int(m[u])][layer]
            p = min(pcc(rec[u], rec_ref[u]), pcc(conv[:, u].float(), conv_ref[:, u].float()))
            assert p > STATE_PCC, f"gated commit user {u} (m={int(m[u])}) layer {layer}: pcc {p:.6f} vs plain"
    logger.info(f"gated commit(m={m.tolist()}) == blend commit bit for bit; == plain after m_u + 1 steps (pcc)")
    # ---- the gated graphs under trace: replayed V + C reproduce the eager gated commit bit for bit
    gated.capture()
    tok, pos0 = _prefill_all(model, mtp, prompt)
    ids_t, _ = gated.verify_forward(tokens, [pos0] * U)
    assert torch.equal(ids_t, ids_g)
    _state_close(_kda_state_host(model), pre, "traced gated verify: live KDA state untouched", bits=True)
    gated.commit(m, [pos0] * U)
    _state_close(_kda_state_host(model), after_gated, "traced gated commit == eager gated commit", bits=True)
    # a second commit from the committed state with another m (no backup to spend in the gated mode)
    m2 = torch.tensor([(u + 1) % (Kd + 1) for u in range(U)])
    gated.verify_forward(tokens, [p + int(mu) + 1 for p, mu in zip([pos0] * U, m)])
    gated.commit(m2, [p + int(mu) + 1 for p, mu in zip([pos0] * U, m)])
    assert all(torch.isfinite(rec).all() for rec, _ in _kda_state_host(model).values())
    gated.release()
    ttnn.release_trace(mesh_device, trace_plain)


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_spec_decode_lossless_gated(mesh_device):
    model, mtp, cfg, prompt = _build(mesh_device)
    U = model.total_users
    Kd = spec_draft_len(B, K_ENV)
    assert Kd >= 1
    spec = SpecDecoder(model, mtp, Kd, commit_mode="gated")
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
            ), f"user {u}: gated spec diverges from plain greedy at token {i} with gap {gap:.3f}"
    plain_tps = n / plain_s
    spec_tps = n / stats.seconds if stats.seconds else float("nan")
    logger.info(
        f"SPEC-GATED mesh {MESH} layers {len(model.layers)} users {U} K {Kd}: {stats.summary()}; plain {n} steps "
        f"{plain_s:.2f} s; tokens identical to plain greedy per user: {matched} of {n} (n = all)"
    )
    logger.info(
        f"PERF-SPEC-GATED mesh {MESH} layers {len(model.layers)} users {U} K {Kd}: plain {plain_tps:.1f} tok/s per "
        f"user ({plain_tps * U:.0f} aggregate); spec {spec_tps:.1f} tok/s per user ({spec_tps * U:.0f} aggregate), "
        f"x{spec_tps / plain_tps:.2f}, {stats.tokens_per_iteration / U:.2f} tokens per user per iteration"
    )
    spec.release()
    ttnn.release_trace(mesh_device, trace_plain)
