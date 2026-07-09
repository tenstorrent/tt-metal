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
