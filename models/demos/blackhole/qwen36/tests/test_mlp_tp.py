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
@pytest.mark.parametrize(
    "in_dtype",
    [
        pytest.param(ttnn.bfloat16, id="in_bf16"),
        pytest.param(ttnn.bfloat8_b, id="in_bf8"),
        pytest.param(ttnn.bfloat4_b, id="in_bf4"),
    ],
)
def test_mlp_tp_prefill(mesh_device, in_dtype, reset_seeds, ensure_gc, request):
    """Prefill-path (S>32) TP MLP vs torch SwiGLU. Exercises the 2D prefill matmul for w1/w3
    and the (now default) explicit w2 down-proj progcfg.

    in_dtype is the dtype ff_norm hands the MLP, and it is model-dependent on Wormhole:
    the 27B narrows its post-norm all-gather to bf8 because that collective is bytes-bound on one
    ETH link (layer.py ``_ff_gather_dtype`` / tt/prefill_norm_tuned.py), while the 9B gathers
    before the norm and stays bf16. Both are covered here so the bf8 arm's accuracy is measured
    against the fp32 torch reference rather than against another device path that shares the same
    quantisation -- the model-level TP tests compare two paths that BOTH carry it, so they cannot
    see this.

    MEASURED (T3K, TP=8, 27B, T=2048): bf16 in 0.9989536, bf8 in 0.9989485 -- a 5th-decimal
    difference. The MLP's error budget is dominated by the bfp4 gate/up weights + LoFi either way,
    which is why halving the activation's bytes is close to free here.

    in_bf4 is NOT shipped -- it is the row that rejected it. Narrowing the gather again (bf8 -> bfp4)
    was tried on 2026-08-20 and dropped: it bought only -13us on the gather (that collective is on a
    ~680us latency floor, not a bytes floor -- see layer.py _ff_gather_dtype) while costing 130x the
    bf8 step's accuracy. MEASURED here, same T=2048, vs the same fp32 reference:
        bf16 0.9989521 | bf8 0.9989442 (-8e-6) | bfp4 0.9979378 (-1.0e-3)
    Kept as a parametrization because this is the only place that trade is measured against fp32
    rather than against another device path carrying the same quantisation, and because the number is
    what makes the rejection checkable rather than a claim. The MLP floors gate/up's output at bf8
    regardless (mlp.py), so this row prices the ACTIVATION narrowing alone -- which was the claim.
    """
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

    # Prefill fused gate/up AGMM (Blackhole only) expects a K-sharded input (ff_norm skips its
    # post-norm all-gather in the real model); the op gathers it back internally. When the fusion
    # is disabled (Wormhole; see tp_common.mlp_gateup_agmm_enabled), ff_norm does its own gather in
    # the real model, so MLP expects an already-full-width input here instead.
    if nd > 1 and mlp._fuse_gateup_agmm:
        x_tt = shard_to_device(mesh_device, x, dim=-1, dtype=in_dtype)
    else:
        x_tt = replicate_to_device(mesh_device, x, dtype=in_dtype)
    out = mlp.forward(x_tt)
    out_torch = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].to(torch.float32)  # [T, dim]

    passing, pcc = comp_pcc(ref, out_torch, get_pcc_threshold(request, default=0.97))
    logger.info(f"MLP TP PREFILL PCC (T={T}, in={in_dtype}) = {pcc}")
    assert passing, f"MLP TP prefill PCC too low: {pcc}"
