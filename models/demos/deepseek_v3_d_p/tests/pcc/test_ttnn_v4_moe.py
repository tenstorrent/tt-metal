# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V4-Flash MoE (build_v4_moe over TtMoe) vs the reference DeepseekV4SparseMoeBlock with random weights: a learned
top-k layer (sqrtsoftplus, route scale 1.5) and a hash-routed layer (tid2eid on device). 2x4 runs 64 experts
(8 per chip, the galaxy's per-chip load), 8x4 the full 256. The reference applies swiglu_limit; the device runs the
fused Silu without the clamp (plan M8) -- the PCC here measures that deviation too."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.moe import build_v4_moe, reference_moe_weights

_SEED = 42
_PCC = 0.98

_MESH_CONFIGS = [
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        64,
        id="fabric2d-mesh-2x4-64experts",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        256,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4-256experts",
    ),
]


def _cfg(n_experts):
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    cfg.n_routed_experts = n_experts
    return cfg


def _ref(cfg, layer_idx):
    torch.manual_seed(_SEED)
    ref = DeepseekV4SparseMoeBlock(cfg, layer_idx).eval()
    with torch.no_grad():
        ref.gate.weight.normal_(0.0, 0.02)
        if hasattr(ref.gate, "e_score_correction_bias"):
            ref.gate.e_score_correction_bias.normal_(0.0, 0.1)
        if hasattr(ref.gate, "tid2eid"):
            ref.gate.tid2eid.copy_(torch.randint(0, cfg.n_routed_experts, tuple(ref.gate.tid2eid.shape)))
        ref.experts.gate_up_proj.normal_(0.0, 0.02)
        ref.experts.down_proj.normal_(0.0, 0.02)
        for m in (ref.shared_experts.gate_proj, ref.shared_experts.up_proj, ref.shared_experts.down_proj):
            m.weight.normal_(0.0, 0.02)
    return ref


@pytest.mark.parametrize("layer_idx", [3, 0], ids=["topk-layer3", "hash-layer0"])
@pytest.mark.parametrize(
    "mesh_device, device_params, n_experts", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_v4_moe(mesh_device, device_params, n_experts, layer_idx):
    cfg = _cfg(n_experts)
    ref = _ref(cfg, layer_idx)
    sp, tp = mesh_device.shape
    seq_len_per_chip = 640  # the engine's 5120-token chunk over SP 8; the same per-chip load on 2x4
    total = sp * seq_len_per_chip
    torch.manual_seed(_SEED + 1)
    x = torch.randn(sp, seq_len_per_chip, cfg.hidden_size).to(torch.bfloat16)
    input_ids = torch.randint(0, cfg.vocab_size, (total,))
    with torch.no_grad():
        ref_bf = ref.to(torch.bfloat16)
        out_ref = ref_bf(x.view(1, total, cfg.hidden_size), input_ids=input_ids.view(1, total)).view(
            sp, seq_len_per_chip, -1
        )

    w = reference_moe_weights(ref)
    moe = build_v4_moe(
        mesh_device,
        cfg,
        layer_idx,
        seq_len_per_chip=seq_len_per_chip,
        gate_weight=w["gate_weight"].float(),
        gate_bias=None if w["gate_bias"] is None else w["gate_bias"].float(),
        tid2eid=w["tid2eid"],
        routed_expert_weights=[{k: v.float() for k, v in e.items()} for e in w["routed_expert_weights"]],
        shared_expert_weights={k: v.float() for k, v in w["shared_expert_weights"].items()},
        num_routed_experts=n_experts,
    )
    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, -1)),
    )
    ttnn.synchronize_device(mesh_device)
    out_tt, _ = moe(x_tt, input_ids=input_ids if layer_idx < 3 else None)
    ttnn.synchronize_device(mesh_device)
    out = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, -1))
    )
    out = out.reshape(out_ref.shape) if out.numel() == out_ref.numel() else out
    ok, pcc = comp_pcc(out_ref.float(), out.float(), _PCC)
    logger.info(
        f"V4 MoE layer {layer_idx} ({'hash' if layer_idx < 3 else 'topk'}) {n_experts} experts on {tuple(mesh_device.shape)}: PCC {pcc} (shapes ref {tuple(out_ref.shape)} tt {tuple(out.shape)})"
    )
    assert ok, pcc
