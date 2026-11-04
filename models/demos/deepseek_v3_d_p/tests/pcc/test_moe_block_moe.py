# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MoE block (ttnn.experimental.moe_block through TtK3MoEBlockFfn, GLM-5.3's default MoE) vs the CPU MoE reference
on random weights: the block-path twin of test_ttnn_moe.py::test_glm_moe (which tests the TtMoe module itself).

One 5120-token chunk on the 8x4 XY torus (S/8 rows, H/4 columns a chip), the weights given as host tensors (the
TtPrefillBlock state dict the synthetic-weight tests use), the router non-degenerate on random weights. Also Kimi-K3
the same way (test_k3_moe_block).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3.moe import glm_moe_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_block_mesh_params
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = 5120
FINAL_PCC = 0.982  # (test_ttnn_moe.run_model's final-output bar)
_PAYLOAD, _L1_SMALL = moe_block_mesh_params(GLM53Config, moe_fabric_payload(GLM53Config), 0, moe_impl="block")


def _random_moe_weights(hidden, inter, n_routed, seed):
    """GLM-5.3 MoE weights as test_prefill_block's _glm_random_moe_weights makes them (bf16, 1/sqrt(fan-in))."""
    g = torch.Generator().manual_seed(seed)
    hs, ds = hidden**-0.5, inter**-0.5

    def expert():
        return {
            "gate_proj": (torch.randn(inter, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(inter, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, inter, generator=g) * ds).to(torch.bfloat16),
        }

    gate = {
        "weight": (torch.randn(n_routed, hidden, generator=g) * hs).to(torch.bfloat16),
        "e_score_correction_bias": (torch.randn(n_routed, generator=g) * 0.01).to(torch.float32),
    }
    return gate, [expert() for _ in range(n_routed)], expert()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=_PAYLOAD,
                worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
                l1_small_size=_L1_SMALL,
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1200)
def test_glm_moe_block(mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import TtK3MoEBlockFfn

    R, C = tuple(mesh_device.shape)
    H = GLM53Config.EMB_SIZE
    gate, routed, shared = _random_moe_weights(H, GLM53Config.MOE_INTERMEDIATE_SIZE, GLM53Config.NUM_ROUTED_EXPERTS, 3)
    sd = {"gate_weights": gate, "routed_expert_weights": routed, "shared_expert_weights": shared}
    ffn = TtK3MoEBlockFfn(mesh_device, GLM53Config, layer_idx=6, state_dict=sd)

    x = (torch.randn(1, 1, CHUNK, H, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3)),
    )
    y_tt = ffn(x_tt)  # (the build and the weights, then one call)
    ttnn.synchronize_device(mesh_device)
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    y = ttnn.to_torch(y_tt, mesh_composer=comp).float()

    want = glm_moe_reference(
        x[0],
        gate_weights=gate,
        routed_expert_weights=routed,
        shared_expert_weights=shared,
        emb_dim=H,
        num_experts_per_tok=GLM53Config.NUM_EXPERTS_PER_TOKEN,
        n_group=GLM53Config.NUM_EXPERT_GROUPS,
        topk_group=GLM53Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=GLM53Config.ROUTE_SCALE,
    ).float()
    ok, msg = comp_pcc(want.reshape(-1, H), y.reshape(-1, H), FINAL_PCC)
    logger.info(f"MoE block (GLM-5.3, random weights, {CHUNK} tokens) vs the CPU MoE: {msg}")
    ttnn.deallocate(y_tt)
    assert ok, f"MoE block vs the CPU MoE: {msg} (need {FINAL_PCC})"


def _k3_moe_reference(x, gate, routed, shared, latent):
    """Kimi-K3's MoE in fp32 on the host - what TorchMoe computes for test_kimi_k3_moe: the noaux_tc router (sigmoid
    scores, + the correction bias for the pick, one group), top-16 weights renormalized x ROUTE_SCALE; the routed
    experts at the latent width (down_proj in, SiTU-GLU, the sum, the latent RMSNorm, up_proj out); the SiTU-GLU shared
    expert at emb_dim."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config as K

    b, lb = K.ACTIVATION_SITU_BETA, K.ACTIVATION_SITU_LINEAR_BETA
    situ = lambda g, u: b * torch.tanh(g / b) * torch.sigmoid(g) * lb * torch.tanh(u / lb)  # noqa: E731
    x = x.float()
    scores = torch.sigmoid(x @ gate["weight"].float().T)
    ids = (scores + gate["e_score_correction_bias"].float()).topk(K.NUM_EXPERTS_PER_TOKEN, dim=-1).indices
    w = scores.gather(-1, ids)
    w = w / w.sum(-1, keepdim=True) * K.ROUTE_SCALE
    h = x @ latent["down_proj"].float().T
    acc = torch.zeros(x.shape[0], h.shape[1])
    for e in torch.unique(ids).tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        r = routed[e]
        a = situ(h[tok] @ r["gate_proj"].float().T, h[tok] @ r["up_proj"].float().T)
        acc.index_add_(0, tok, (a @ r["down_proj"].float().T) * w[tok, kk].unsqueeze(1))
    acc = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + K.RMS_NORM_EPS) * latent["norm"].float()
    y = acc @ latent["up_proj"].float().T
    return y + situ(x @ shared["gate_proj"].float().T, x @ shared["up_proj"].float().T) @ shared["down_proj"].float().T


from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config  # noqa: E402

_K3_PAYLOAD, _K3_L1_SMALL = moe_block_mesh_params(KimiK3Config, moe_fabric_payload(KimiK3Config), 0, moe_impl="block")
K3_FINAL_PCC = 0.965  # (test_kimi_k3_moe's final-output bar)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=_K3_PAYLOAD,
                worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
                l1_small_size=_K3_L1_SMALL,
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_k3_moe_block(mesh_device, device_params):
    """Kimi-K3 on the MoE block (TtK3MoEBlockFfn from host weights) vs the K3 MoE math on the host: test_kimi_k3_moe's
    weight creators and seeds, its 0.965 bar. The routed experts: one chip's worth (28) of create_torch_expert_weights
    repeated on every chip (896 distinct experts would take the host ~120 GB to lay out); the router, latent pair and
    shared expert as test_kimi_k3_moe's."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import TtK3MoEBlockFfn
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        create_gate_weights,
        create_latent_weights,
        create_shared_expert_weights,
        create_torch_expert_weights,
    )

    R, C = tuple(mesh_device.shape)
    K = KimiK3Config
    H, Hr, E = K.EMB_SIZE, K.ROUTED_EXPERT_HIDDEN_SIZE, K.NUM_ROUTED_EXPERTS
    epc = E // (R * C)
    pool = create_torch_expert_weights(epc, Hr, K.MOE_INTERMEDIATE_SIZE, seed=1234)
    routed = [pool[e % epc] for e in range(E)]
    shared = create_shared_expert_weights(H, K.SHARED_EXPERT_INTERMEDIATE_SIZE, seed=5678)
    gate = create_gate_weights(E, H, seed=9012)
    latent = create_latent_weights(H, Hr, seed=3456)
    sd = {
        "gate_weights": gate,
        "routed_expert_weights": routed,
        "shared_expert_weights": shared,
        "latent_weights": latent,
    }
    ffn = TtK3MoEBlockFfn(mesh_device, KimiK3Config, layer_idx=1, state_dict=sd)

    x = torch.randn(1, 1, CHUNK, H, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3)),
    )
    y_tt = ffn(x_tt)
    ttnn.synchronize_device(mesh_device)
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    y = ttnn.to_torch(y_tt, mesh_composer=comp).float()
    ttnn.deallocate(y_tt)

    want = _k3_moe_reference(x[0, 0], gate, routed, shared, latent)
    ok, msg = comp_pcc(want.reshape(-1, H), y.reshape(-1, H), K3_FINAL_PCC)
    logger.info(f"MoE block (Kimi-K3, random weights, {CHUNK} tokens) vs the host K3 MoE: {msg}")
    assert ok, f"MoE block vs the host K3 MoE: {msg} (need {K3_FINAL_PCC})"
