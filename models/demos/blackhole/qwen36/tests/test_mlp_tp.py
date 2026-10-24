# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for the Qwen3.5/3.6 SwiGLU MLP on a Blackhole mesh.

Loads just one layer's gate/up/down weights from the FP8 checkpoint (fast,
RAM-light), runs the tensor-parallel Qwen36MLP forward, and compares against a
torch SwiGLU reference. Output is fractured along the hidden dim (reduce-scatter)
so it is gathered with ConcatMeshToTensor(dim=3).

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
        pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
"""
import gc
import os

import pytest
import torch
from loguru import logger

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
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


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


@torch.no_grad()
@pytest.mark.parametrize("batch", [1, 32], ids=["B1", "B32"])
@parametrize_mesh_tp()
def test_mlp_tp_decode_dshard(mesh_device, batch, monkeypatch, reset_seeds, ensure_gc, request):
    """QWEN36_MLP_DECODE_DSHARD=1 (DRAM-sharded 3-reader gate/up, padded N) vs the interleaved 1D-mcast decode path (flag set explicitly for both arms):
    same module inputs, outputs must agree to PCC >= 0.9999 (and both match torch). The DRAM-sharded layout is TP=4
    only (mlp_decode_dshard_active): at any other TP the flag must leave the interleaved layout, so both arms match."""
    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    if nd < 2:
        pytest.skip("DRAM-sharded decode gate/up is TP-only")
    args = Qwen36ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=256)
    mlp_state = load_mlp_layer(args.CKPT_DIR, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    x = torch.randn(1, 1, batch, args.dim, dtype=torch.bfloat16)
    g = mlp_state["gate_proj.weight"].to(torch.float32)
    u = mlp_state["up_proj.weight"].to(torch.float32)
    d = mlp_state["down_proj.weight"].to(torch.float32)
    xf = x.to(torch.float32)[0, 0]
    ref = (torch.nn.functional.silu(xf @ g.T) * (xf @ u.T)) @ d.T

    outs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("QWEN36_MLP_DECODE_DSHARD", flag)
        mlp = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)
        assert mlp._dsh_decode == (flag == "1" and nd == 4)
        if mlp._dsh_decode:
            assert mlp.weights.w1.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            assert mlp.weights.w1.shape[-1] == args.mlp_dsh_n_padded
        out = mlp.forward(replicate_to_device(mesh_device, x))
        outs[flag] = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0, :batch].to(torch.float32)
        del mlp
        gc.collect()

    _, pcc_ref = comp_pcc(ref, outs["1"], 0.0)
    passing, pcc = comp_pcc(outs["0"], outs["1"], 0.9999)
    logger.info(f"MLP decode dshard B={batch}: PCC vs flag-off = {pcc}, vs torch = {pcc_ref}")
    assert passing, f"dshard MLP differs from default decode path: PCC {pcc}"
    assert pcc_ref > get_pcc_threshold(request), f"dshard MLP vs torch PCC too low: {pcc_ref}"
