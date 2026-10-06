# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.moe import (
    Qwen38MoE,
    Qwen38MoEWeights,
    admit_moe_compute_global_batch,
    weighted_routed_reduce,
)

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")


def _load_reference_module():
    if not TRANSFORMERS_SRC:
        pytest.skip("set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
    source = Path(TRANSFORMERS_SRC)
    sys.path.insert(0, str(source))
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp

        return modeling_qwen4_exp
    finally:
        sys.path.remove(str(source))


class _LazyExpertAxis:
    def __init__(self, weights, projection):
        self.weights = weights
        self.projection = projection

    def __getitem__(self, expert_index):
        expert = self.weights.expert(int(expert_index))
        return getattr(expert, self.projection)


def test_exact_moe_tp4_weight_placement_has_no_routed_replication():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    weights = Qwen38MoEWeights(checkpoint, placement, layer_index=0)

    assert weights.expert_ranges == ((0, 128), (128, 256), (256, 384), (384, 512))
    assert weights.bf4_routed_payload_bytes_per_device == 353_894_400
    assert torch.equal(torch.cat([weights.router_shard(device) for device in range(4)]), weights.router_weight)

    shared = [weights.shared_shard(device) for device in range(4)]
    assert all(shard.gate_proj.shape == (160, 2560) for shard in shared)
    assert all(shard.up_proj.shape == (160, 2560) for shard in shared)
    assert all(shard.down_proj.shape == (2560, 160) for shard in shared)
    assert all(shard.scalar_gate.shape == (1, 640) for shard in shared)
    assert torch.equal(torch.cat([shard.gate_proj for shard in shared], dim=0), weights.shared_gate_proj)
    assert torch.equal(torch.cat([shard.up_proj for shard in shared], dim=0), weights.shared_up_proj)
    assert torch.equal(torch.cat([shard.down_proj for shard in shared], dim=1), weights.shared_down_proj)
    assert torch.equal(torch.cat([shard.scalar_gate for shard in shared], dim=1), weights.shared_scalar_gate)

    assert weights.owner(127) == 0
    assert weights.owner(128) == 1
    assert weights.expert(127).gate.shape == (640, 2560)
    assert weights.expert(128).down.shape == (2560, 640)


def test_exact_checkpoint_moe_and_ep4_decomposition_match_pinned_transformers():
    reference = _load_reference_module()
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    weights = Qwen38MoEWeights(checkpoint, placement, layer_index=0)
    model = Qwen38MoE(weights)

    generator = torch.Generator().manual_seed(38)
    hidden = (torch.randn((1, 1, 2560), generator=generator) * 0.02).to(torch.bfloat16)
    flat = hidden.view(-1, 2560)

    router_proxy = SimpleNamespace(
        hidden_dim=2560,
        num_experts=512,
        top_k=10,
        norm_topk_prob=True,
        weight=weights.router_weight,
    )
    reference_logits, reference_scores, reference_indices = reference.Qwen4ExpTextTopKRouter.forward(router_proxy, flat)

    expert_proxy = SimpleNamespace(
        config=SimpleNamespace(_experts_implementation="eager"),
        num_experts=512,
        gate_up_proj=_LazyExpertAxis(weights, "gate_up"),
        down_proj=_LazyExpertAxis(weights, "down"),
        act_fn=F.silu,
    )
    reference_routed = reference.Qwen4ExpTextExperts.forward(expert_proxy, flat, reference_indices, reference_scores)
    reference_shared = F.linear(
        F.silu(F.linear(flat, weights.shared_gate_proj)) * F.linear(flat, weights.shared_up_proj),
        weights.shared_down_proj,
    )
    reference_shared *= torch.sigmoid(F.linear(flat, weights.shared_scalar_gate))
    reference_output = (reference_routed + reference_shared).view_as(hidden)

    output, routing = model(hidden)
    assert torch.equal(routing.logits, reference_logits)
    assert torch.equal(routing.scores, reference_scores)
    assert torch.equal(routing.indices, reference_indices)
    torch.testing.assert_close(output, reference_output, atol=0.0, rtol=0.0)

    ep4 = model.expert_parallel_forward(hidden)
    assert torch.equal(ep4.routing.indices, reference_indices)
    assert sorted(expert for local in ep4.local_selected_experts for expert in local) == sorted(
        reference_indices.flatten().tolist()
    )
    assert sum(len(local) for local in ep4.local_selected_experts) == 10
    torch.testing.assert_close(torch.cat(ep4.hidden_shards, dim=-1), reference_output, atol=0.05, rtol=0.02)


def test_weighted_reduce_requires_exact_scores_and_preserves_topk_order(expect_error):
    expert_outputs = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    scores = torch.softmax(torch.arange(20, dtype=torch.float32).reshape(2, 10), dim=-1)
    expected = (expert_outputs * scores.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(weighted_routed_reduce(expert_outputs, scores), expected)
    with expect_error(ValueError, "real top-k scores"):
        weighted_routed_reduce(expert_outputs, None)


def test_moe_compute_dispatch_path_fails_closed_for_true_global_batch_one(expect_error):
    assert admit_moe_compute_global_batch(global_batch=4, dispatch_devices=4) == 1
    with expect_error(ValueError, "true global batch 1"):
        admit_moe_compute_global_batch(global_batch=1, dispatch_devices=4)
