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
    torch_routed_experts_reference,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "seq_len, mode",
    [(1, "decode"), (8, "decode"), (32, "prefill"), (256, "prefill"), (512, "prefill")],
    ids=["decode", "decode_batch8", "prefill32", "prefill256", "prefill512"],
)
def test_moe_tp(mesh_device, seq_len, mode, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=1024)
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
    out = moe.forward(x_tt, mode=mode)
    out_torch = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()  # [S, dim]

    passing, pcc = comp_pcc(ref, out_torch, get_pcc_threshold(request))
    logger.info(f"MoE TP ({request.node.callspec.id}) PCC = {pcc}")
    assert passing, f"MoE TP PCC too low: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_moe_routed_experts_batch(mesh_device, reset_seeds, ensure_gc, request):
    """Routed-expert path only (no shared expert), multi-user decode.

    Multi-user decode collapses the per-user routing into a per-expert union mask, so every
    selected expert is computed for every user and each user's own top-k is reapplied
    afterwards. That makes the (expert, user) axis order of the sparse_matmul output
    load-bearing, and a mix-up is invisible to an aggregate PCC that includes the shared
    expert. Two checks pin it down:

      1. batch independence -- user b's output must equal the same input run on its own
         (B=1), so one user's activation can never reach another user's down-projection;
      2. accuracy against a routed-only torch reference.

    The routing is supplied explicitly (bypassing the router) with *disjoint* expert sets
    per user, which is the arrangement a cross-user leak corrupts most visibly.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=1024)
    if args.moe_num_experts <= 0:
        pytest.skip("not a MoE checkpoint (moe_num_experts == 0)")

    from models.demos.blackhole.qwen36.tt.moe import MoEConfig, Qwen36MoE

    nd = mesh_device.get_num_devices()
    batch, top_k, num_experts = 8, args.moe_top_k, args.moe_num_experts
    if batch * top_k > num_experts:
        pytest.skip(f"need batch*top_k <= num_experts for disjoint routing ({batch}*{top_k} > {num_experts})")

    moe_state = load_moe_layer(args.CKPT_DIR, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    moe = Qwen36MoE(mesh_device, MoEConfig.from_args(args), moe_state, None, args=args, tt_ccl=tt_ccl)

    x = torch.randn(1, 1, batch, args.dim, dtype=torch.bfloat16)

    # Disjoint top-k per user: user b owns experts [b*top_k, (b+1)*top_k), weights summing to 1.
    routing = torch.zeros(batch, num_experts, dtype=torch.float32)
    for b in range(batch):
        w = torch.rand(top_k) + 0.5
        routing[b, b * top_k : (b + 1) * top_k] = w / w.sum()

    def run_experts(x_bf16, routing_f32):
        """Routed experts only -- moe.experts(), skipping moe.forward()'s shared-expert add."""
        x_tt = replicate_to_device(mesh_device, x_bf16)
        r_tt = replicate_to_device(mesh_device, routing_f32.reshape(1, 1, *routing_f32.shape).to(torch.bfloat16))
        out = moe.experts(x_tt, r_tt, mode="decode")
        return ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()

    batched = run_experts(x, routing)[:batch]  # [batch, dim]

    # 1. batch independence: each user re-run alone must reproduce its batched row. The union
    # mask makes this exact (same weights, same per_core_M, only the M padding differs), so it
    # is held to ~bit-identical rather than the accuracy threshold -- that is what makes it a
    # sharp guard for an axis mix-up instead of a loose accuracy check.
    for b in range(batch):
        single = run_experts(x[:, :, b : b + 1, :], routing[b : b + 1])[:1]
        passing, pcc = comp_pcc(single[0], batched[b], 0.9999)
        logger.info(f"routed experts user {b}: B=1 vs B={batch} PCC = {pcc}")
        assert passing, f"user {b} output depends on the other users in the batch: PCC {pcc}"

    # 2. accuracy of the routed path against torch.
    ref = torch_routed_experts_reference(moe_state, x[0, 0].float(), routing)
    threshold = get_pcc_threshold(request)
    passing, pcc = comp_pcc(ref, batched, threshold)
    logger.info(f"routed experts (B={batch}) vs torch PCC = {pcc}")
    assert passing, f"routed-expert PCC too low: {pcc}"
