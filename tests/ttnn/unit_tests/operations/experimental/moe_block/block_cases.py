# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The MoE block's test models and meshes, shared by the block tests (models/demos/moe_block/tests) and the op-level
checks that run the block's stages (block_stages.py): a model config (BlockModel, MODELS) is what a model brings - its
block dims, seeded random weights, the host reference on them; MESHES the stages it runs on."""

from dataclasses import dataclass
from typing import Callable

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3.moe import glm_moe_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_block_fabric_payload, model_l1_small_with_block

# ---------------------------------------------------------------------------------------------------- model configs
class DeepSeekV41FlashConfig:
    """DeepSeek V4.1-Flash's MoE as the block runs it (the old block matrix's dsv41_flash): DeepSeek V4's router and
    activation - sqrtsoftplus scores, top-6 over all experts, x 1.5; clamped SwiGLU (limit 10) - confirm against the V4.1
    checkpoint. Hash-routed layers are out of scope. Synthetic weights only."""

    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 6
    EMB_SIZE = 5120
    MOE_INTERMEDIATE_SIZE = 2304
    SHARED_EXPERT_INTERMEDIATE_SIZE = 2304
    ROUTE_SCALE = 1.5
    SWIGLU_LIMIT = 10.0

    @staticmethod
    def moe_block_dims():
        mb = ttnn.experimental.moe_block
        act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=DeepSeekV41FlashConfig.SWIGLU_LIMIT)
        dims = mb.MoEModelDims(
            num_experts=DeepSeekV41FlashConfig.NUM_ROUTED_EXPERTS,
            top_k=DeepSeekV41FlashConfig.NUM_EXPERTS_PER_TOKEN,
            emb_dim=DeepSeekV41FlashConfig.EMB_SIZE,
            routed_expert_hidden_dim=DeepSeekV41FlashConfig.MOE_INTERMEDIATE_SIZE,
            routed_expert_activation=act,
            router=mb.RouterSpec(
                score_func=mb.RouterScoreFunc.SqrtSoftplus, route_scale=DeepSeekV41FlashConfig.ROUTE_SCALE
            ),
            shared_expert_hidden_dim=DeepSeekV41FlashConfig.SHARED_EXPERT_INTERMEDIATE_SIZE,
            shared_expert_activation=act,
        )
        return dims, "clamped_silu"


class MiniMaxM3BlockConfig(MiniMaxM3Config):
    """MiniMax-M3's MoE as the block runs it (M3.1 has the same MoE: they differ in attention and checkpoint format
    only): 128 experts, top-4 of sigmoid scores + correction bias, renormalized x 2.0, no groups; the routed experts and
    the one shared expert (I 3072) clamped swigluoai (alpha 1.702, limit 7 - the kernels' fixed values)."""

    SWIGLU_ALPHA = 1.702
    SWIGLU_LIMIT = 7.0

    @staticmethod
    def moe_block_dims():
        mb = ttnn.experimental.moe_block
        K = MiniMaxM3BlockConfig
        act = mb.ExpertActivation(kind=mb.ExpertActivationKind.SwigluOai, limit=K.SWIGLU_LIMIT, alpha=K.SWIGLU_ALPHA)
        dims = mb.MoEModelDims(
            num_experts=K.NUM_ROUTED_EXPERTS,
            top_k=K.NUM_EXPERTS_PER_TOKEN,
            emb_dim=K.EMB_SIZE,
            routed_expert_hidden_dim=K.MOE_INTERMEDIATE_SIZE,
            routed_expert_activation=act,
            router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=K.ROUTE_SCALE),
            shared_expert_hidden_dim=K.SHARED_INTERMEDIATE_SIZE,
            shared_expert_activation=act,
        )
        return dims, "swigluoai"


def _bf16_experts(n, hidden, inter, g):
    """n experts, bf16, 1/sqrt(fan-in) (test_prefill_block's random MoE weights)."""
    hs, ds = hidden**-0.5, inter**-0.5
    return [
        {
            "gate_proj": (torch.randn(inter, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(inter, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, inter, generator=g) * ds).to(torch.bfloat16),
        }
        for _ in range(n)
    ]


def _bf16_gate(E, hidden, g):
    return {
        "weight": (torch.randn(E, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "e_score_correction_bias": (torch.randn(E, generator=g) * 0.01).to(torch.float32),
    }


def _glm_weights(chips):
    K = GLM53Config
    g = torch.Generator().manual_seed(3)
    gate = _bf16_gate(K.NUM_ROUTED_EXPERTS, K.EMB_SIZE, g)
    routed = _bf16_experts(K.NUM_ROUTED_EXPERTS, K.EMB_SIZE, K.MOE_INTERMEDIATE_SIZE, g)
    shared = _bf16_experts(1, K.EMB_SIZE, K.MOE_INTERMEDIATE_SIZE, g)[0]
    return {"gate_weights": gate, "routed_expert_weights": routed, "shared_expert_weights": shared}


def _glm_reference(x, sd):
    K = GLM53Config
    return glm_moe_reference(
        x.reshape(1, *x.shape),  # ([1, seq, hidden])
        gate_weights=sd["gate_weights"],
        routed_expert_weights=sd["routed_expert_weights"],
        shared_expert_weights=sd["shared_expert_weights"],
        emb_dim=K.EMB_SIZE,
        num_experts_per_tok=K.NUM_EXPERTS_PER_TOKEN,
        n_group=K.NUM_EXPERT_GROUPS,
        topk_group=K.NUM_LIMITED_GROUPS,
        routed_scaling_factor=K.ROUTE_SCALE,
    ).float().reshape(x.shape)


def _k3_weights(chips):
    """test_kimi_k3_moe's creators and seeds; the routed experts one chip's worth repeated on every chip (896 distinct
    experts would take the host ~120 GB)."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        create_gate_weights,
        create_latent_weights,
        create_shared_expert_weights,
        create_torch_expert_weights,
    )

    K = KimiK3Config
    H, Hr, E = K.EMB_SIZE, K.ROUTED_EXPERT_HIDDEN_SIZE, K.NUM_ROUTED_EXPERTS
    epc = E // chips
    pool = create_torch_expert_weights(epc, Hr, K.MOE_INTERMEDIATE_SIZE, seed=1234)
    return {
        "gate_weights": create_gate_weights(E, H, seed=9012),
        "routed_expert_weights": [pool[e % epc] for e in range(E)],
        "shared_expert_weights": create_shared_expert_weights(H, K.SHARED_EXPERT_INTERMEDIATE_SIZE, seed=5678),
        "latent_weights": create_latent_weights(H, Hr, seed=3456),
    }


def _k3_reference(x, sd):
    """Kimi-K3's MoE in fp32: noaux_tc router (sigmoid + bias for the pick, one group), top-16 renormalized x
    ROUTE_SCALE; the routed experts at the latent width (down_proj in, SiTU-GLU, sum, latent RMSNorm, up_proj out); the
    SiTU-GLU shared expert at emb_dim."""
    K = KimiK3Config
    gate, routed, shared, latent = (
        sd["gate_weights"],
        sd["routed_expert_weights"],
        sd["shared_expert_weights"],
        sd["latent_weights"],
    )
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


def _dsv41_weights(chips):
    K = DeepSeekV41FlashConfig
    g = torch.Generator().manual_seed(11)
    gate = _bf16_gate(K.NUM_ROUTED_EXPERTS, K.EMB_SIZE, g)
    routed = _bf16_experts(K.NUM_ROUTED_EXPERTS, K.EMB_SIZE, K.MOE_INTERMEDIATE_SIZE, g)
    shared = _bf16_experts(1, K.EMB_SIZE, K.SHARED_EXPERT_INTERMEDIATE_SIZE, g)[0]
    return {"gate_weights": gate, "routed_expert_weights": routed, "shared_expert_weights": shared}


def _dsv41_reference(x, sd):
    """DeepSeek V4's MoE in fp32 (reference/deepseek_v4/modeling_deepseek_v4.py DeepseekV4TopKRouter, Experts): scores
    sqrt(softplus(logits)), top-6 of scores + bias, the scores renormalized x 1.5; experts down(silu(min(gate, L)) *
    clamp(up, -L, L)); the shared expert the same."""
    K = DeepSeekV41FlashConfig
    L = K.SWIGLU_LIMIT
    glu = lambda g, u: torch.nn.functional.silu(g.clamp(max=L)) * u.clamp(-L, L)  # noqa: E731
    gate, routed, shared = sd["gate_weights"], sd["routed_expert_weights"], sd["shared_expert_weights"]
    x = x.float()
    scores = torch.sqrt(torch.nn.functional.softplus(x @ gate["weight"].float().T))
    ids = (scores + gate["e_score_correction_bias"].float()).topk(K.NUM_EXPERTS_PER_TOKEN, dim=-1).indices
    w = scores.gather(-1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * K.ROUTE_SCALE
    acc = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        r = routed[e]
        a = glu(x[tok] @ r["gate_proj"].float().T, x[tok] @ r["up_proj"].float().T)
        acc.index_add_(0, tok, (a @ r["down_proj"].float().T) * w[tok, kk].unsqueeze(1))
    return acc + glu(x @ shared["gate_proj"].float().T, x @ shared["up_proj"].float().T) @ shared["down_proj"].float().T


def _m3_weights(chips):
    """128 distinct experts (bf16 ~14 GB on the host, less than GLM-5.3's 256)."""
    K = MiniMaxM3BlockConfig
    g = torch.Generator().manual_seed(13)
    gate = _bf16_gate(K.NUM_ROUTED_EXPERTS, K.EMB_SIZE, g)
    routed = _bf16_experts(K.NUM_ROUTED_EXPERTS, K.EMB_SIZE, K.MOE_INTERMEDIATE_SIZE, g)
    shared = _bf16_experts(1, K.EMB_SIZE, K.SHARED_INTERMEDIATE_SIZE, g)[0]
    return {"gate_weights": gate, "routed_expert_weights": routed, "shared_expert_weights": shared}


def _m3_reference(x, sd):
    """MiniMax-M3's MoE in fp32 (minimax_m3/reference/model.py's sparse MoE): scores sigmoid(logits), top-4 of scores +
    bias, the scores renormalized x 2.0; experts down(swiglu_oai(gate, up)), swiglu_oai(g, u) = (clamp(u, +-7) + 1) *
    g' sigmoid(1.702 g'), g' = min(g, 7); plus the shared expert the same."""
    K = MiniMaxM3BlockConfig
    a, L = K.SWIGLU_ALPHA, K.SWIGLU_LIMIT

    def glu(g, u):
        g = g.clamp(max=L)
        return (u.clamp(-L, L) + 1.0) * g * torch.sigmoid(a * g)

    gate, routed, shared = sd["gate_weights"], sd["routed_expert_weights"], sd["shared_expert_weights"]
    x = x.float()
    scores = torch.sigmoid(x @ gate["weight"].float().T)
    ids = (scores + gate["e_score_correction_bias"].float()).topk(K.NUM_EXPERTS_PER_TOKEN, dim=-1).indices
    w = scores.gather(-1, ids)
    w = w / w.sum(-1, keepdim=True) * K.ROUTE_SCALE
    acc = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        r = routed[e]
        h = glu(x[tok] @ r["gate_proj"].float().T, x[tok] @ r["up_proj"].float().T)
        acc.index_add_(0, tok, (h @ r["down_proj"].float().T) * w[tok, kk].unsqueeze(1))
    return acc + glu(x @ shared["gate_proj"].float().T, x @ shared["up_proj"].float().T) @ shared["down_proj"].float().T


BARRIER_L1_SMALL = 4096  # (test_moe_block_perf's barrier: L1_SMALL a bank for its all-gathers)


@dataclass(frozen=True)
class BlockModel:
    """What a model brings to the block tests."""

    name: str
    cfg: type  # (the model config TtK3MoEBlockFfn takes)
    weights: Callable[[int], dict]  # chips -> the state dict (seeded)
    reference: Callable[[torch.Tensor, dict], torch.Tensor]  # x [S, H], state dict -> y [S, H] fp32
    pcc: float  # the accuracy floor on random weights
    layer: int = 1  # (the layer index the random-weight block is built as)
    x_scale: float = 1.0  # (the random input's scale)
    weka: bool = False  # real layers + golden activations on Weka

    def mesh_params(self, trace_region: int = 64 << 20) -> dict:
        width = getattr(self.cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or self.cfg.EMB_SIZE
        return dict(
            fabric_payload_size=moe_block_fabric_payload(width),
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            # (+ the perf test's barrier: its all-gathers' semaphores go in L1_SMALL - DSv4.1's mesh opened with exactly
            # the block's 2368 B had none left)
            l1_small_size=model_l1_small_with_block(self.cfg, 0) + BARRIER_L1_SMALL,
            trace_region_size=trace_region,
        )


MODELS = {
    "glm53": BlockModel("glm53", GLM53Config, _glm_weights, _glm_reference, pcc=0.982, layer=6, x_scale=0.5),
    "kimi_k3": BlockModel("kimi_k3", KimiK3Config, _k3_weights, _k3_reference, pcc=0.965, weka=True),
    "dsv41_flash": BlockModel("dsv41_flash", DeepSeekV41FlashConfig, _dsv41_weights, _dsv41_reference, pcc=0.98),
    "minimax_m3": BlockModel("minimax_m3", MiniMaxM3BlockConfig, _m3_weights, _m3_reference, pcc=0.98, layer=3),
}


# ----------------------------------------------------------------------------------------------------- test configs
MESHES = {  # id -> (mesh shape, device params maker, requires_mesh_topology's topology)
    "8x4": ((8, 4), torus_xy_device_params, "mesh-8x4"),
    "4x4-y": ((4, 4), torus_y_device_params, "mesh-4x4"),  # (SP ring, TP line)
    "4x4-xy": ((4, 4), torus_xy_device_params, "mesh-4x4"),  # (both rings)
    # MiniMax-M3's production stage: one tray, SP 2 x TP 4, plain Fabric2D - no wrap on either axis (run with
    # TT_MESH_GRAPH_DESC_PATH = experimental_descriptors/single_bh_galaxy_sub_2x4_graph_descriptor.textproto and
    # TT_VISIBLE_DEVICES = one tray's 8 chips, e.g. 0-7)
    "2x4": ((2, 4), fabric2d_device_params, "mesh-2x4"),
    # The 4x4 prefill stage as #61481 runs it: one tray half, SP on the axis-1 ring, TP on the axis-0 line (run with
    # TT_MESH_GRAPH_DESC_PATH = experimental_descriptors/single_bh_galaxy_tray_half_sp1_graph_descriptor.textproto and
    # TT_VISIBLE_DEVICES = the half's 16 chips, gen_pipeline_binding.py --layout 4x4) - the block on the transposed mesh
    "4x4-sp1": ((4, 4), torus_x_device_params, "mesh-4x4"),
    # MiniMax-M3 on a 4x2 stage (SP 4 x TP 2, one tray's 8 chips, plain Fabric2D: run with TT_MESH_GRAPH_DESC_PATH =
    # experimental_descriptors/single_bh_galaxy_sub_4x2_graph_descriptor.textproto and TT_VISIBLE_DEVICES = 0-7)
    "4x2": ((4, 2), fabric2d_device_params, "mesh-4x2"),
}
MESH_SP_AXIS = {"4x4-sp1": 1}  # (the mesh axis SP runs on; 0 elsewhere)
