# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the ``kda_decode`` roofline knobs (_agents/kda_roofline/STATUS.md) -- written device-free, NOT run
(the chips are owned by the validation queue); importorskip-safe.

Chip 0 (real layer-4 KDA weights, the TP4 shard's 16 heads unless ``GLM53_PERF_H=64``):

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_kda_decode_v2_mesh.py -s

* ``test_kda_decode_op_variants_vs_golden``: the program with ``gates="act"`` (host-activated gates) and with the
  rectangle placement vs ``KdaDecode.golden`` (B = 4 / 32 / 40 at 16 heads).
* ``test_kda_decode_placement_bit_identity``: column vs rectangle placement on identical inputs, 3 steps: output and
  state byte-identical (the same per-user math on the same core-local data; the gather sums disjoint rows).
* ``test_kda_layer_v2_vs_v1``: ``TtKDA`` block path v1 (today) vs v2 (``gates=act`` + ``placement=rect``) on real
  weights, 10 decode steps from the same carry: output PCC per step and the final state PCC (bar 0.9999; the g2
  weight re-rounding ``bf16(a w)`` makes the gates differ at the bf16-ulp level), HF user 0 as the absolute anchor;
  ``placement=rect`` alone must be bit-identical to v1.
* ``test_kda_layer_v2_trace_replay``: the v2 layer captured in a trace, 3 replays == 3 eager steps.
* ``test_kda_layer_variants_perf``: traced ms/step of v1, v2, v2 + ``fidelity=hifi2`` at ``GLM53_PERF_B`` (the
  tracy run gives the ``kda_decode`` row: ``python -m tracy -r -p -v -m pytest ... -k perf``).
* ``test_kda_mm_override_perf``: traced ms/step of the layer under a few ``GLM53_KDA_MM_QKV`` / ``_O`` configs
  (the sweep lever; no assertion beyond finiteness).

(2,4) block decoder test and the e2e chain with the bundle knob (the decoder test feeds every prompt through the
decode step against HF rows; the chain = 16 greedy steps vs HF):

    export W=/home/ttuser/ashai/glm53/tt-metal
    GLM53_KDA_DECODE_V2=1 TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_mesh_decoder.py -s
    GLM53_KDA_DECODE_V2=1 TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_mesh_model.py -s -k chain
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.kda_decode.op import KdaDecode, kda_selector_tiles
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.kda import TtKDA

K = V = 128
LB, L2_EPS, NORM_EPS = -5.0, 1e-6, 1e-5
LAYER = int(os.environ.get("GLM53_PERF_LAYER", "4"))
H_LOCAL = int(os.environ.get("GLM53_PERF_H", "16"))  # 16 = the TP4 shard
PERF_B = int(os.environ.get("GLM53_PERF_B", "32"))
STEPS = int(os.environ.get("GLM53_PERF_DECODE_STEPS", "10"))
V1 = dict(gates="pre", placement="col", fidelity="hifi4")
V2 = dict(gates="act", placement="rect", fidelity="hifi4")


# ------------------------------------------------------------------------------------------------ op level
def _dev(device, t, dtype=ttnn.float32):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _inputs(B, H, seed):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    W = 3 * H * K
    Hb = -(-H // 32) * 32
    qkv = r(1, 1, B, W).to(torch.bfloat16).float()
    gates = torch.zeros(1, 1, B, H * K + H * V + Hb)
    gates[..., : H * K] = r(B, H * K) * 2.0
    gates[..., H * K : H * K + H * V] = r(B, H * V) * 2.0
    gates[..., H * K + H * V : H * K + H * V + H] = r(B, H) * 2.0
    conv = (r(B, 3, W) * 0.8).to(torch.bfloat16).float()
    S = 0.05 * r(B, H, K, V)
    taps = (r(4, W) * 0.5).to(torch.bfloat16).float()
    dt_bias = r(H * K) * 0.5
    a_exp = torch.rand(H, generator=gen).repeat_interleave(K) * 4.0 + 0.5
    onw = torch.rand(V, generator=gen) * 0.5 + 0.75
    return dict(qkv=qkv, gates=gates, conv=conv, S=S, taps=taps, dt_bias=dt_bias, a_exp=a_exp, onw=onw)


def _to_device(device, x, B, H, gates_act: bool):
    gates = x["gates"][0, 0]
    if gates_act:  # what the pre-arranged g2 matmul + fused sigmoid hand the program
        gates = KdaDecode.activate_gates(gates, x["dt_bias"], x["a_exp"], H * K)
    return dict(
        qkv=_dev(device, x["qkv"].to(torch.bfloat16), ttnn.bfloat16),
        gates=_dev(device, gates.reshape(1, 1, B, -1).contiguous()),
        conv=_dev(device, x["conv"].permute(1, 0, 2).contiguous().to(torch.bfloat16), ttnn.bfloat16),
        S=_dev(device, x["S"].clone()),
        taps=_dev(device, x["taps"].reshape(1, 1, 4, -1).to(torch.bfloat16), ttnn.bfloat16),
        cc=_dev(device, torch.stack([x["dt_bias"], x["a_exp"]]).reshape(1, 1, 2, -1)),
        onw=_dev(device, x["onw"].reshape(1, 1, 1, -1)),
        sel=_dev(device, kda_selector_tiles().to(torch.bfloat16), ttnn.bfloat16),
        out=_dev(device, torch.zeros(1, 1, B, H * V).to(torch.bfloat16), ttnn.bfloat16),
    )


def _run(op, d):
    return op(d["qkv"], d["gates"], d["conv"], d["S"], d["taps"], d["cc"], d["onw"], d["sel"], d["out"])


def _golden(x, B, H):
    return KdaDecode.golden(
        x["qkv"][0, 0, :B], x["gates"][0, 0, :B], x["conv"], x["S"], x["taps"], x["dt_bias"], x["a_exp"], x["onw"],
        num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS,
    )  # fmt: skip


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("gates, placement", [("act", "col"), ("pre", "rect"), ("act", "rect")])
@pytest.mark.parametrize("B, H", [(4, 16), (32, 16), (40, 16)])
def test_kda_decode_op_variants_vs_golden(mesh_device, gates, placement, B, H):
    op = KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS, gates=gates, placement=placement)
    x = _inputs(B, H, seed=B * 3 + H)
    o_ref, S_ref, conv_ref = _golden(x, B, H)
    d = _to_device(mesh_device, x, B, H, gates == "act")
    _run(op, d)
    o = ttnn.to_torch(d["out"]).float()[0, 0]
    S = ttnn.to_torch(d["S"])
    conv = ttnn.to_torch(d["conv"]).float().permute(1, 0, 2)
    p_o, p_s = pcc(o_ref, o), pcc(S_ref, S)
    logger.info(f"gates={gates} placement={placement} B={B} H={H}: o pcc {p_o:.6f} state pcc {p_s:.6f}")
    assert p_o > 0.9999 and p_s > 0.9999, (p_o, p_s)
    torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)
    assert (o - o_ref).abs().max() < 0.05 and (S - S_ref).abs().max() < 0.05


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_kda_decode_placement_bit_identity(mesh_device, B):
    """Column vs rectangle placement: the same per-user math on the same data, so the outputs and the state must be
    byte-identical over 3 steps (a difference = a multicast / gather defect of the rectangle layout)."""
    H = 16
    ops = [
        KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS, placement=p)
        for p in ("col", "rect")
    ]
    x = _inputs(B, H, seed=11)
    ds = [_to_device(mesh_device, x, B, H, False) for _ in ops]
    for step in range(3):
        gen = torch.Generator().manual_seed(100 + step)
        qkv = torch.randn(1, 1, B, 3 * H * K, generator=gen).to(torch.bfloat16)
        for d in ds:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(qkv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), d["qkv"]
            )
        outs = []
        for op, d in zip(ops, ds):
            _run(op, d)
            outs.append((ttnn.to_torch(d["out"]), ttnn.to_torch(d["S"]), ttnn.to_torch(d["conv"])))
        for a, b in zip(outs[0], outs[1]):
            assert torch.equal(a, b), f"step {step}: column vs rectangle placement differ"


# ------------------------------------------------------------------------------------------------ layer level
def _layer_weights():
    from models.demos.glm53_flash.tests.test_kda_perf import slice_heads

    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=lambda n: n.startswith("self_attn."))
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    if H_LOCAL != cfg.kda_num_heads:
        attn, cfg = slice_heads(attn, cfg, H_LOCAL)
    return snapshot, cfg, raw, attn


def _seed_state(tt, B, S0, conv0):
    st = tt.allocate_state(B)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), st.recurrent)
    tt.upload_conv_history(st, conv0)
    return st


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [4, 32])
def test_kda_layer_v2_vs_v1(mesh_device, batch):
    snapshot, cfg, raw, attn = _layer_weights()
    layers = {
        "v1": TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=V1),
        "rect": TtKDA(
            mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=dict(V1, placement="rect")
        ),
        "v2": TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=V2),
    }
    assert layers["v2"].kda_decode.gates_activated and layers["v2"].g2_bias is not None and layers["v1"].g2_bias is None
    torch.manual_seed(1)
    B, H = batch, layers["v1"].H
    S0 = 0.05 * torch.randn(B, H, K, V)
    conv0 = (0.5 * torch.randn(B, 3, layers["v1"].conv_width)).to(torch.bfloat16)
    states = {k: _seed_state(tt, B, S0, conv0) for k, tt in layers.items()}
    hf = None
    if H_LOCAL == cfg.kda_num_heads:  # the HF module needs the whole layer
        from transformers import DynamicCache

        from models.demos.glm53_flash.reference.hf_layer_reference import load_config
        from models.demos.glm53_flash.tests.test_kda import _hf_linear_attention

        _, text = load_config(snapshot)
        hf = _hf_linear_attention(snapshot, text, raw)
        cache = DynamicCache(config=text)
        with torch.no_grad():
            hf(
                torch.zeros(1, 4, cfg.hidden_size, dtype=torch.bfloat16),
                cache_params=cache,
                attention_mask=torch.ones(1, 4, dtype=torch.bool),
            )
        lc = cache.layers[LAYER]
        lc.recurrent_states[0] = S0[:1].clone()
        lc.conv_states[0][:, :, -3:] = conv0[:1].float().transpose(1, 2)
    worst = dict(v2=1.0, rect=1.0, hf=1.0)
    for step in range(10):
        xt = (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16)
        x_dev = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        ys = {}
        for k, tt in layers.items():
            y, states[k] = tt.decode_forward(x_dev, states[k])
            ys[k] = ttnn.to_torch(y).float()[0, 0]
        assert torch.equal(ys["rect"], ys["v1"]), f"step {step}: rectangle placement is not bit-identical to v1"
        p = pcc(ys["v1"], ys["v2"])
        worst["v2"] = min(worst["v2"], p)
        msg = f"  step {step}: v2 vs v1 pcc {p:.6f}"
        if hf is not None:
            with torch.no_grad():
                y_hf = hf(xt[:, :, 0], cache_params=cache, attention_mask=torch.ones(1, 1, dtype=torch.bool))
            p_hf = pcc(y_hf[0], ys["v2"][:1])
            worst["hf"] = min(worst["hf"], p_hf)
            msg += f"; v2 vs HF user 0 {p_hf:.6f} (v1 {pcc(y_hf[0], ys['v1'][:1]):.6f})"
        logger.info(msg)
    s = {k: ttnn.to_torch(st.recurrent) for k, st in states.items()}
    assert torch.equal(s["rect"], s["v1"]), "rectangle placement: state not bit-identical to v1"
    p_state = pcc(s["v1"], s["v2"])
    p_conv = pcc(
        layers["v1"].conv_history_torch(states["v1"]).float(), layers["v2"].conv_history_torch(states["v2"]).float()
    )
    logger.info(
        f"KDA layer {LAYER} B={B} H={H}: v2 vs v1 worst output pcc {worst['v2']:.6f}, state {p_state:.6f}, conv {p_conv:.6f}"
    )
    assert worst["v2"] > 0.9999 and p_state > 0.9999 and p_conv > 0.9999
    if hf is not None:
        assert worst["hf"] > 0.999 and pcc(lc.recurrent_states[0], s["v2"][:1]) > 0.999


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_layer_v2_trace_replay(mesh_device):
    """The v2 layer inside a captured trace (the bias and the pre-arranged weights are persistent, the program's
    io addresses baked): 3 replays on a persistent input == 3 eager steps of a twin layer from the same carry."""
    _, cfg, _, attn = _layer_weights()
    B = 4
    eager = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=V2)
    traced = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=V2)
    torch.manual_seed(2)
    S0 = 0.05 * torch.randn(B, eager.H, K, V)
    conv0 = (0.5 * torch.randn(B, 3, eager.conv_width)).to(torch.bfloat16)
    st_e, st_t = _seed_state(eager, B, S0, conv0), _seed_state(traced, B, S0, conv0)
    x_dev = ttnn.from_torch(
        torch.zeros(1, 1, B, cfg.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    x0 = (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(x0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x_dev)
    traced.decode_forward(x_dev, st_t)  # compile pass (advances st_t once)
    eager.decode_forward(x_dev, st_e)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_t, _ = traced.decode_forward(x_dev, st_t)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for step in range(3):
        xt = (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x_dev)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        y_e, _ = eager.decode_forward(x_dev, st_e)
        a, b = ttnn.to_torch(y_t).float(), ttnn.to_torch(y_e).float()
        p = pcc(a, b)
        logger.info(f"  replay {step}: trace vs eager pcc {p:.6f}")
        assert p > 0.9999
    assert torch.equal(ttnn.to_torch(st_t.recurrent), ttnn.to_torch(st_e.recurrent)), "traced and eager states differ"
    ttnn.release_trace(mesh_device, tid)


def _traced_ms_per_step(mesh_device, tt, B, cfg):
    torch.manual_seed(0)
    state = tt.allocate_state(B)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(0.05 * torch.randn(B, tt.H, tt.K, tt.V), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        state.recurrent,
    )
    x = ttnn.from_torch(
        (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    tt.decode_forward(x, state)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y, _ = tt.decode_forward(x, state)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) / STEPS
    assert torch.isfinite(ttnn.to_torch(y).float()).all()
    ttnn.release_trace(mesh_device, tid)
    return dt * 1e3


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_layer_variants_perf(mesh_device):
    _, cfg, _, attn = _layer_weights()
    variants = {
        "v1": V1,
        "rect": dict(V1, placement="rect"),
        "act": dict(V1, gates="act"),
        "v2": V2,
        "v2+hifi2": dict(V2, fidelity="hifi2"),
    }
    rows = []
    for name, opts in variants.items():
        tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=opts)
        ms = _traced_ms_per_step(mesh_device, tt, PERF_B, cfg)
        rows.append((name, ms))
        logger.info(
            f"KDA layer {LAYER} H={tt.H} B={PERF_B} {name}: {ms:.3f} ms/step ({PERF_B / ms * 1e3:.0f} tok/s aggregate)"
        )
    logger.info("| variant | ms/step |\n|---|---|\n" + "\n".join(f"| {n} | {ms:.3f} |" for n, ms in rows))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_mm_override_perf(mesh_device, monkeypatch):
    """The projection-config lever: a few 1D mcast configs of the qkv (4096 x 6144 bfp8) and o_proj (2048 x 4096)
    matmuls through ``GLM53_KDA_MM_*`` (read at layer construction), traced ms/step of the whole layer."""
    _, cfg, _, attn = _layer_weights()
    configs = [
        ("table", None, None),
        ("qkv 12x10/2/4", "12,10,2,4", None),
        ("o 12x6/2/8", None, "12,6,2,8"),
        ("o 12x4/3/8", None, "12,4,3,8"),
    ]
    for name, qkv, o in configs:
        for env, val in (("GLM53_KDA_MM_QKV", qkv), ("GLM53_KDA_MM_O", o)):
            if val is None:
                monkeypatch.delenv(env, raising=False)
            else:
                monkeypatch.setenv(env, val)
        tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block", decode_opts=V1)
        ms = _traced_ms_per_step(mesh_device, tt, PERF_B, cfg)
        logger.info(f"KDA layer {LAYER} B={PERF_B} mm {name}: {ms:.3f} ms/step")
