# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for the Qwen3.5/3.6 SwiGLU MLP on a Blackhole mesh.

Loads just one layer's gate/up/down weights from the FP8 checkpoint (fast,
RAM-light), runs the tensor-parallel Qwen36MLP forward, and compares against a
torch SwiGLU reference. Output is fractured along the hidden dim (reduce-scatter)
so it is gathered with ConcatMeshToTensor(dim=3).

test_mlp_tp_decode_ds runs the TP=8 decode MLP of the MTP verify (RES40 input, residual-adding all-reduce) on the
DRAM-sharded matmuls (mlp_decode_ds) and on the 1D matmuls they replace, at the gate/up weight dtypes and math
fidelities in question, against an exact reference, and logs the trace replay time of each.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
        pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
"""
import os
import time

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    get_pcc_threshold,
    load_mlp_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.mlp import MLPWeights, Qwen36MLP, _build_gate_up_ds
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
def test_mlp_tp(mesh_device, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    logger.info(f"devices={nd} dim={args.dim} hidden_dim={args.hidden_dim}")

    # args.CKPT_DIR is the resolved local snapshot dir (Qwen36ModelArgs downloads the hub id).
    mlp_state = load_mlp_layer(args.CKPT_DIR, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    mlp = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)

    # Torch reference: down(silu(gate(x)) * up(x))
    T = 32
    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    g = mlp_state["gate_proj.weight"].to(torch.float32)
    u = mlp_state["up_proj.weight"].to(torch.float32)
    d = mlp_state["down_proj.weight"].to(torch.float32)
    xf = x.to(torch.float32)[0, 0]  # [T, dim]
    ref = (torch.nn.functional.silu(xf @ g.T) * (xf @ u.T)) @ d.T  # [T, dim]

    x_tt = replicate_to_device(mesh_device, x)
    out = mlp.forward(x_tt)
    out_torch = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].to(torch.float32)  # [T, dim]

    passing, pcc = comp_pcc(ref, out_torch, get_pcc_threshold(request))
    logger.info(f"MLP TP PCC = {pcc}")
    assert passing, f"MLP TP PCC too low: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_mlp_tp_prefill(mesh_device, reset_seeds, ensure_gc, request):
    """Prefill-path (S>32) TP MLP vs torch SwiGLU. Exercises the 2D prefill matmul for w1/w3
    and the (now default) explicit w2 down-proj progcfg."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    nd = mesh_device.get_num_devices()
    logger.info(f"devices={nd} dim={args.dim} hidden_dim={args.hidden_dim}")

    mlp_state = load_mlp_layer(args.CKPT_DIR, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    mlp = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)

    # Torch reference: down(silu(gate(x)) * up(x)) at prefill seq length.
    T = 2048
    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    g = mlp_state["gate_proj.weight"].to(torch.float32)
    u = mlp_state["up_proj.weight"].to(torch.float32)
    d = mlp_state["down_proj.weight"].to(torch.float32)
    xf = x.to(torch.float32)[0, 0]  # [T, dim]
    ref = (torch.nn.functional.silu(xf @ g.T) * (xf @ u.T)) @ d.T  # [T, dim]

    # Prefill fused gate/up AGMM expects a K-sharded input (ff_norm skips its
    # post-norm all-gather in the real model); the op gathers it back internally.
    x_tt = shard_to_device(mesh_device, x, dim=-1) if nd > 1 else replicate_to_device(mesh_device, x)
    out = mlp.forward(x_tt)
    out_torch = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].to(torch.float32)  # [T, dim]

    passing, pcc = comp_pcc(ref, out_torch, get_pcc_threshold(request, default=0.97))
    logger.info(f"MLP TP PREFILL PCC (T={T}) = {pcc}")
    assert passing, f"MLP TP prefill PCC too low: {pcc}"


def _ckc(fidelity):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, fp32_dest_acc_en=True, packer_l1_acc=True)


def _replay_us(mesh, tid, n=50):
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n * 1e6


@torch.no_grad()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
            "trace_region_size": 64 << 20,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8")], indirect=True)
def test_mlp_tp_decode_ds(mesh_device, reset_seeds, ensure_gc, request):
    """The verify's decode MLP on DRAM-sharded matmuls against the 1D matmuls it replaces.

    Arms: gate/up bfloat4_b on the 1D matmuls at LoFi (the replaced path), on the DRAM-sharded matmuls at LoFi (the
    default) and HiFi4; gate/up bfloat8_b on the 1D matmuls at HiFi2 and on the DRAM-sharded ones at HiFi4. down is
    bfloat8_b in every arm. Each arm takes the same RES40 input and a zero residual through DecodeResidualAllReduce, so
    its output is the all-reduced MLP output, compared with an fp32 reference on the checkpoint weights. The default
    must not be less accurate than the replaced path, and HiFi4 not less accurate than LoFi. Trace replays of every
    arm compile no program after the eager warm-up runs; their time per MLP (with its all-reduce) is logged."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.demos.blackhole.qwen36.tt.decode_allreduce import get_decode_allreduce
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=256)
    if not args.mlp_decode_ds:
        pytest.skip("mlp_decode_ds is off on this mesh")
    tt_ccl = TT_CCL(mesh_device)
    dar = get_decode_allreduce(args, tt_ccl)
    if dar is None:
        pytest.skip("this mesh keeps the fractured decode residual")

    state = load_mlp_layer(args.CKPT_DIR, 0)
    ds = Qwen36MLP(mesh_device, state, None, args=args, tt_ccl=tt_ccl)
    args.mlp_decode_ds = False
    one_d = Qwen36MLP(mesh_device, state, None, args=args, tt_ccl=tt_ccl)
    args.mlp_decode_ds = True
    assert ds._decode_ds and ds.weights.w1 is None and not one_d._decode_ds

    gate, up = state["gate_proj.weight"], state["up_proj.weight"]
    bf8_1d = MLPWeights(
        w1=tpc.shard_w(gate, mesh_device, -1, ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat8_b),
        w3=tpc.shard_w(up, mesh_device, -1, ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat8_b),
        w2=one_d.weights.w2,
    )
    bf8_ds = MLPWeights(
        w1=None,
        w3=None,
        w2=ds.weights.w2,
        w_gu_ds=_build_gate_up_ds(
            gate, up, mesh_device, args.num_devices, None, args.mlp_ds_gateup_weight_memcfg, dtype=ttnn.bfloat8_b
        ),
    )
    lofi, hifi2, hifi4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
    arms = {
        "1d bf4 LoFi": (one_d, one_d.weights, lofi),
        "ds bf4 LoFi": (ds, ds.weights, lofi),
        "ds bf4 HiFi4": (ds, ds.weights, hifi4),
        "1d bf8 HiFi2": (one_d, bf8_1d, hifi2),
        "ds bf8 HiFi4": (ds, bf8_ds, hifi4),
    }

    def select(arm):
        mlp, weights, fidelity = arms[arm]
        mlp.weights = weights
        if mlp is ds:
            mlp.compute_kernel_config_ds = _ckc(fidelity)
        else:
            mlp.compute_kernel_config_decode = _ckc(fidelity)
        return mlp

    def to_res40(t):
        dram = ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out = ttnn.to_memory_config(dram, dar.res_mem)
        ttnn.deallocate(dram)
        return out

    g, u, d = gate.float(), up.float(), state["down_proj.weight"].float()
    threshold = get_pcc_threshold(request, default=0.97)
    for rows in (4, 32):
        x = torch.randn(1, 1, rows, args.dim, dtype=torch.bfloat16)
        xf = x.float()[0, 0]
        ref = (torch.nn.functional.silu(xf @ g.T) * (xf @ u.T)) @ d.T
        x_tt, zero = to_res40(x), to_res40(torch.zeros(1, 1, rows, args.dim, dtype=torch.bfloat16))
        outs, err = {}, {}
        for arm in arms:
            out = select(arm).forward(x_tt, decode_ar=dar.with_residual(zero))
            dev = [ttnn.to_torch(t)[0, 0].float() for t in ttnn.get_device_tensors(out)]
            ttnn.deallocate(out)
            assert all(torch.equal(dev[0], t) for t in dev[1:]), f"{arm}: the all-reduced output differs per device"
            outs[arm] = dev[0]
            err[arm] = ((dev[0] - ref).norm() / ref.norm()).item()
            passing, pcc = comp_pcc(ref, dev[0], threshold)
            logger.info(f"[MLP_DS] rows={rows} {arm}: PCC {pcc}, error against the fp32 reference {err[arm]:.3e}")
            assert passing, f"rows={rows} {arm}: PCC {pcc} below {threshold}"
        base = outs["1d bf4 LoFi"]
        for arm in arms:
            same = (outs[arm] == base).float().mean().item()
            logger.info(f"[MLP_DS] rows={rows} {arm}: {same:.1%} of the outputs equal those of 1d bf4 LoFi")
        assert err["ds bf4 LoFi"] <= 1.05 * err["1d bf4 LoFi"], err
        assert err["ds bf4 HiFi4"] <= 1.01 * err["ds bf4 LoFi"], err
        ttnn.deallocate(zero)
        if rows != 32:
            ttnn.deallocate(x_tt)

    # Trace replays: 4 MLPs per trace, the warm-up above compiled every arm's programs at 32 rows.
    zero = to_res40(torch.zeros(1, 1, 32, args.dim, dtype=torch.bfloat16))
    programs = mesh_device.num_program_cache_entries()
    ttnn.synchronize_device(mesh_device)
    tids, held = {}, []
    with trace_allocation_tracker.corruptible_allocation_scope(mesh_device):
        for arm in arms:
            mlp = select(arm)
            tids[arm] = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for i in range(4):
                out = mlp.forward(x_tt, decode_ar=dar.with_residual(zero))
                if i < 3:
                    ttnn.deallocate(out)
            ttnn.end_trace_capture(mesh_device, tids[arm], cq_id=0)
            held.append(out)
    us = {arm: _replay_us(mesh_device, tid) / 4 for arm, tid in tids.items()}
    assert mesh_device.num_program_cache_entries() == programs, "a trace capture or replay compiled a program"
    for arm in arms:
        logger.info(f"[MLP_DS] trace replay {arm}: {us[arm]:.1f} us per MLP with its all-reduce")
    for tid in tids.values():
        ttnn.release_trace(mesh_device, tid)
    for t in held + [x_tt, zero]:
        ttnn.deallocate(t)
