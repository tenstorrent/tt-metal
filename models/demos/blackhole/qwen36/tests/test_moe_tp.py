# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for the Qwen3.5-MoE sparse MLP on a Blackhole mesh.

Loads one layer's router + fused experts + shared-expert weights, runs the
tensor-parallel Qwen36MoE forward (decode seq_len=1 and prefill seq_len=32), and
compares against the torch MoE reference. Output is fractured along the hidden dim
(reduce-scatter, matching Qwen36MLP), so it is gathered with ConcatMeshToTensor(dim=3).

Only runs on a MoE checkpoint; auto-skips on the dense 27B. Run:

    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-35B-A3B \
        pytest models/demos/blackhole/qwen36/tests/test_moe_tp.py -v -s
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    get_pcc_threshold,
    load_moe_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    torch_moe_reference,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("seq_len", [1, 32], ids=["decode", "prefill"])
def test_moe_tp(mesh_device, seq_len, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    if args.moe_num_experts <= 0:
        pytest.skip("not a MoE checkpoint (moe_num_experts == 0)")

    from models.demos.blackhole.qwen36.tt.moe import MoEConfig, Qwen36MoE

    nd = mesh_device.get_num_devices()
    logger.info(f"devices={nd} dim={args.dim} experts={args.moe_num_experts} top_k={args.moe_top_k}")

    moe_state = load_moe_layer(args.CKPT_DIR, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    moe = Qwen36MoE(mesh_device, MoEConfig.from_args(args), moe_state, None, args=args, tt_ccl=tt_ccl)

    x = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)
    ref = torch_moe_reference(moe_state, x[0, 0].float(), args.moe_top_k, args.moe_norm_topk_prob)  # [S, dim]

    x_tt = replicate_to_device(mesh_device, x)
    out = moe.forward(x_tt)
    out_torch = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()  # [S, dim]

    passing, pcc = comp_pcc(ref, out_torch, get_pcc_threshold(request))
    logger.info(f"MoE TP ({request.node.callspec.id}) PCC = {pcc}")
    assert passing, f"MoE TP PCC too low: {pcc}"
