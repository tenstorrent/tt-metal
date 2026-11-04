# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MLP_GU_FUSED on the TP mesh: the fused [gate|up] decode MLP against today's (r5) decode path.

* test_swiglu_split_exact: the split-SwiGLU kernel against torch on the device's own gate/up values (fp32 silu * mul,
  one bf16 rounding), interleaved and width-sharded inputs, 32 and 8 rows.
* test_mlp_gu_fused_vs_r5[1d|dsmc]: Qwen36MLP with the flag on vs off on layer-0 weights, decode shapes (32 = verify
  rows, 8 = drafter rows), input in the ff-norm's act_shard_hidden layout as in the model: PCC of the MLP output and of
  the per-device hidden (silu(gate) * up) against the r5 path and torch; then the fused chain captured in a trace and
  replayed == eager, bit for bit.
* test_mlp_gu_fused_bench: replay time of the per-device decode MLP chain (no CCL) inside a trace, r5 vs 1d vs dsmc,
  K3B_REPS chains per trace (default 32) x K3B_ITERS replays (default 20). Logs [k3bench] lines.

Run (8-chip TP mesh, e.g. /tmp/penv.sh):
    pytest models/demos/blackhole/qwen36/tests/test_mlp_gu_fused.py -s
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_mlp_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.mlp_swiglu import SwiGLUSplit
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

TRACE_REGION = 64 * 1024 * 1024


def _silu_mul_bf16(gu, pair=True):
    """torch mirror of the kernel on [.., W] gate/up values (pair: [g0 u0 g1 u1 ...] tiles)."""
    gu = gu.float()
    k = gu.shape[-1] // 64
    t = gu.reshape(*gu.shape[:-1], k, 2, 32)
    g, u = t[..., 0, :], t[..., 1, :]
    return (torch.nn.functional.silu(g) * u).reshape(*gu.shape[:-1], k * 32).to(torch.bfloat16)


def _per_device(t, mesh_device):
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _ulps(a, b):
    """|a - b| in units of b's bf16 ulp (b != 0); 0 where both are equal."""
    a, b = a.float(), b.float()
    ulp = torch.where(b == 0, torch.full_like(b, 2.0**-133), b.abs() * 2.0**-7)
    return (a - b).abs() / ulp


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("rows", [32, 8])
@pytest.mark.parametrize("layout", ["interleaved", "width_sharded"])
def test_swiglu_split_exact(mesh_device, rows, layout, reset_seeds):
    nd = mesh_device.get_num_devices()
    width = 2 * 2176  # one device's [gate|up] at TP=8
    x = (torch.randn(nd, 1, rows, width) * 4).to(torch.bfloat16)
    mc = ttnn.L1_MEMORY_CONFIG
    if layout == "width_sharded":
        mc = ttnn.create_sharded_memory_config(
            shape=(32, 4 * 32),
            core_grid=ttnn.num_cores_to_corerangeset(width // 128, mesh_device.compute_with_storage_grid_size(), True),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    gu = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=mc,
    )
    out = SwiGLUSplit(mesh_device)(gu)
    assert tuple(out.shape) == (1, 1, rows, width // 2), out.shape
    got = _per_device(out, mesh_device)
    worst, exact = 0.0, []
    for d in range(nd):
        exp = _silu_mul_bf16(x[d : d + 1])
        u = _ulps(got[d][..., :rows, :], exp)
        worst = max(worst, u.max().item())
        exact.append((u == 0).float().mean().item())
    logger.info(f"[k3op] swiglu_split rows={rows} {layout}: max {worst:.3f} bf16 ulp, exact {min(exact):.4f} (min over devices)")
    assert worst <= 1.0, f"split-SwiGLU off by {worst} ulp"
    ttnn.deallocate(out)
    ttnn.deallocate(gu)


def _mlp(mesh_device, monkeypatch, mode, mlp_state, tt_ccl):
    monkeypatch.setenv("QWEN36_MLP_GU_FUSED", mode)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=256)
    assert args.mlp_gu_fused == (None if mode == "0" else mode)
    mlp = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)
    assert mlp.weights.gu_fused == (None if mode == "0" else mode)
    if mode != "0":
        assert mlp.weights.w1 is None and mlp.weights.w3 is None, "fused decode must not load w1/w3"
    return args, mlp


def _hidden_chain(mlp, x_sh):
    """The fused MLP's per-device hidden (silu(gate) * up), [M, hidden/tp], as its decode forward computes it."""
    args, w = mlp.args, mlp.weights
    ckc = mlp.compute_kernel_config_decode
    if w.gu_fused == "dsmc":
        gu = ttnn.linear(
            x_sh,
            w.w_gu_dec,
            program_config=args.mlp_gu_dsmc_progcfg,
            compute_kernel_config=ckc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
    else:
        xi = ttnn.to_memory_config(x_sh, ttnn.L1_MEMORY_CONFIG)
        gu = ttnn.linear(
            xi, w.w_gate_up, program_config=args.mlp_gu_decode_1d_progcfg, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(xi)
    h = mlp._swiglu(gu)
    ttnn.deallocate(gu)
    return h


def _hidden_r5(mlp, x_sh):
    """r5's decode hidden: 1D gate (SiLU fused) and up matmuls, then ttnn.mul (mlp._forward_tp's 1D branch)."""
    args, w = mlp.args, mlp.weights
    ckc = mlp.compute_kernel_config_decode
    xi = ttnn.to_memory_config(x_sh, ttnn.L1_MEMORY_CONFIG)
    g = ttnn.linear(xi, w.w1, program_config=args.mlp_w1_decode_1d_progcfg, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
    u = ttnn.linear(xi, w.w3, program_config=args.mlp_w3_decode_1d_progcfg, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.mul(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
    for t in (xi, g, u):
        ttnn.deallocate(t)
    return h


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=TRACE_REGION)
@pytest.mark.parametrize("mode", ["1d", "dsmc"])
def test_mlp_gu_fused_vs_r5(mesh_device, mode, monkeypatch, reset_seeds, ensure_gc):
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    tt_ccl = TT_CCL(mesh_device)
    args0 = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=256)
    mlp_state = load_mlp_layer(args0.CKPT_DIR, 0)
    _, off = _mlp(mesh_device, monkeypatch, "0", mlp_state, tt_ccl)
    args, on = _mlp(mesh_device, monkeypatch, mode, mlp_state, tt_ccl)
    g = mlp_state["gate_proj.weight"].float()
    u = mlp_state["up_proj.weight"].float()
    d = mlp_state["down_proj.weight"].float()
    hd = args.hidden_dim // nd
    for rows in (32, 8):
        x = torch.randn(1, 1, rows, args.dim, dtype=torch.bfloat16)
        xf = x.float()[0, 0]
        h_ref = torch.nn.functional.silu(xf @ g.T) * (xf @ u.T)  # [rows, hidden]
        ref = h_ref @ d.T
        x_tt = replicate_to_device(mesh_device, x)
        x_sh = ttnn.to_memory_config(x_tt, args.act_shard_hidden)
        outs = {}
        for tag, m in (("off", off), ("on", on)):
            o = m.forward(x_sh)
            outs[tag] = ttnn.to_torch(o, mesh_composer=tp_composer(mesh_device))[0, 0, :rows].float()
            ttnn.deallocate(o)
        h_on = _per_device(_hidden_chain(on, x_sh), mesh_device)
        h_off = _per_device(_hidden_r5(off, x_sh), mesh_device)
        hp = min(comp_pcc(h_off[i][0, 0, :rows].float(), h_on[i][0, 0, :rows].float(), 0.0)[1] for i in range(nd))
        ht = min(comp_pcc(h_ref[:, i * hd : (i + 1) * hd], h_on[i][0, 0, :rows].float(), 0.0)[1] for i in range(nd))
        _, p_r5 = comp_pcc(outs["off"], outs["on"], 0.0)
        _, p_ref_on = comp_pcc(ref, outs["on"], 0.0)
        _, p_ref_off = comp_pcc(ref, outs["off"], 0.0)
        logger.info(
            f"[k3op] mode={mode} rows={rows}: out pcc on-vs-r5 {p_r5:.6f}, vs torch on {p_ref_on:.6f} / r5 {p_ref_off:.6f}; "
            f"hidden pcc on-vs-r5 {hp:.6f}, on-vs-torch {ht:.6f} (min over devices)"
        )
        assert p_r5 >= 0.9999 and hp >= 0.9999, f"{mode} rows={rows}: fused path drifted from r5 ({p_r5}, {hp})"
        assert p_ref_on >= p_ref_off - 1e-4, f"{mode} rows={rows}: fused path less accurate than r5 vs torch"
        # The fused chain inside a trace == eager, bit for bit.
        eager = _per_device(_hidden_chain(on, x_sh), mesh_device)
        h = _hidden_chain(on, x_sh)  # compile outside the trace
        ttnn.deallocate(h)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        h_tr = _hidden_chain(on, x_sh)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            got = _per_device(h_tr, mesh_device)
            assert all(torch.equal(got[i], eager[i]) for i in range(nd)), f"{mode} rows={rows}: trace replay != eager"
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(x_sh)
        ttnn.deallocate(x_tt)


def _replay_us(mesh_device, tid, iters, reps):
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1e6 / (iters * reps)


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=TRACE_REGION)
def test_mlp_gu_fused_bench(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ.setdefault("HF_MODEL", model_path())
    reps = int(os.environ.get("K3B_REPS", "32"))
    iters = int(os.environ.get("K3B_ITERS", "20"))
    tt_ccl = TT_CCL(mesh_device)
    args0 = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=256)
    mlp_state = load_mlp_layer(args0.CKPT_DIR, 0)
    arms = {"r5": _mlp(mesh_device, monkeypatch, "0", mlp_state, tt_ccl)[1]}
    for mode in os.environ.get("K3B_MODES", "1d dsmc").split():
        arms[mode] = _mlp(mesh_device, monkeypatch, mode, mlp_state, tt_ccl)[1]
    x = replicate_to_device(mesh_device, torch.randn(1, 1, 32, args0.dim, dtype=torch.bfloat16))
    x_sh = ttnn.to_memory_config(x, args0.act_shard_hidden)

    def chain(m):
        h = _hidden_r5(m, x_sh) if m.weights.gu_fused is None else _hidden_chain(m, x_sh)
        p = ttnn.linear(
            h,
            m.weights.w2,
            program_config=m.args.mlp_w2_decode_1d_progcfg,
            compute_kernel_config=m.compute_kernel_config_decode,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return p

    res = {}
    for rnd in range(2):  # interleaved rounds
        for tag, m in arms.items():
            ttnn.deallocate(chain(m))
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(reps):
                ttnn.deallocate(chain(m))
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            res.setdefault(tag, []).append(_replay_us(mesh_device, tid, iters, reps))
            ttnn.release_trace(mesh_device, tid)
    base = min(res["r5"])
    for tag, v in res.items():
        logger.info(
            f"[k3bench] {tag}: {min(v):.2f} us per decode MLP chain (gate/up + split + down, no CCL; rounds "
            f"{', '.join(f'{t:.2f}' for t in v)}) | delta vs r5 {min(v) - base:+.2f} us = {64 * (min(v) - base) / 1e3:+.3f} ms "
            f"per 64-layer verify"
        )
