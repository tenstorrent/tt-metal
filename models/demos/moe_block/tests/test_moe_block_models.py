# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The whole MoE block (fused head -> sparse dispatch -> flat experts + combine -> fused tail) for every model it runs:
accuracy, determinism and per-op device time, each its own test function, all parametrized by the same test cases.

* A model config (BlockModel, MODELS) is what a model brings: its block dims (through TtK3MoEBlockFfn), seeded random
  weights, the host reference on them, and - Kimi-K3 - its real layers and golden activations on Weka.
* A test config (BlockCase, CASES) is one run of a model: the mesh (8x4 torus, 4x4 SP ring / TP line, 4x4 torus), the
  chunk, random or Weka weights, and the expected numbers - the accuracy floor and each op's traced device time.

test_moe_block_accuracy: the block's output vs the model's host reference (random weights) or the golden trace's MoE
    output (Weka), PCC at least the case's floor.
test_moe_block_determinism: `calls` eager calls, then `calls` replays of a captured trace, every output bit-identical to
    the first - compared on the device, one scalar a chip read back (op_checks.DeterminismMarker). Runs with the
    combine's snap (q 2^-14): without it the combine's fp32 sums depend on the rows' arrival order.
test_moe_block_perf: `calls` traced replays on the real-time program profiler, each op's device time (its slowest chip,
    the median over the replays) no slower than the case's number + margin; a case without numbers logs what it measured
    and skips.

4x4 cases run on half a Galaxy: TT_MESH_GRAPH_DESC_PATH (the sub-torus descriptor) and TT_VISIBLE_DEVICES set as for
test_k3_moe_block's 4x4 runs. Kimi-K3 Weka cases need $KIMI_K3_CKPT and the golden trace ($KIMI_K3_GOLDEN_TRACE or the
default) - they skip without them, and fail in CI."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3.moe import glm_moe_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_block_fabric_payload, model_l1_small_with_block
from tests.ttnn.utils_for_testing import comp_pcc

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from op_checks import (  # noqa: E402
    BLOCK_OPS,
    DeterminismMarker,
    MeshBarrier,
    assert_not_slower,
    median_us,
    op_device_times,
    traced_op_device_times,
)


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
}


@dataclass(frozen=True)
class BlockCase:
    """One run of a model and its expected numbers."""

    model: str
    mesh: str = "8x4"
    chunk: int = 5120
    weights: str = "random"  # "random" | "weka" (Kimi-K3: the layer's real weights, the golden trace's activations)
    layer: int = 1  # (weka: the model layer)
    pcc: Optional[float] = None  # (None: the model's floor; weka: K3_GOLDEN_PCC)
    perf_us: dict = field(default_factory=dict)  # op -> expected traced device us (slowest chip); empty: report only
    perf_margin: float = 0.05
    calls: int = 10
    # a padded chunk (actual_start, actual_end): main's chunked-prefill metadata - the chunk's real tokens, in main's
    # rotated layout over the SP rows (chunk_layout.rotated_chunk_positions); None: every row real
    pad: Optional[tuple] = None
    packing: str = ""  # (the id's label of the production packing the chunk stands for)

    @property
    def id(self) -> str:
        w = "" if self.weights == "random" else f"-weka-L{self.layer}"
        p = f"-{self.packing}" if self.packing else ""
        pad = f"-real{self.pad[1] - self.pad[0]}@{self.pad[0]}" if self.pad else ""
        return f"{self.model}-{self.mesh}-{self.chunk}{w}{p}{pad}"


# MiniMax-M3 / M3.1 in production: the 16-stage pipeline (4 Galaxies) / 4-stage intra-Galaxy pipeline, every stage a
# [2, 4] tray (SP 2 x TP 4, Fabric2D, no wrap). The packings its runner runs (models/demos/minimax_m3:
# scripts/prefill_matrix, tt/runners/manifests):
# - one 5120-token chunk a forward (every binding and the perf matrix: PREFILL_CHUNK_SIZE 5120), full;
# - 5120 with the last chunk partial: the perf matrix's new-token sizes 640 / 1600 / 3072 (p25 / p50 / p75) and
#   6900's (p90) second chunk 1780, after a chunk-aligned cached prefix (61440, its p25);
# - a multi-turn resume mid-chunk (any 32-token boundary: PREFILL_PRODUCER_MULTI_TURN_PROB) - the chunk starts mid-slab,
#   its tokens rotated over the SP rows: full and partial;
# - four 1024-token chunks packed in one 4096-token forward (manifests/*_packed_*: PREFILL_FORWARD_TOKENS_BUDGET 4096),
#   all real (each SP row holds every chunk's 512 rows back to back - for the MoE one 4096-token chunk). A packed forward
#   with a partial chunk is not expressible: the block takes one padded range a call;
# - 2048-token chunks (the perf matrix's smallest CHUNK).
M3_PREFIX = 61440
M3_RESUME = M3_PREFIX + 1600  # (a resume after the p50 request: mid-slab, 32-aligned)
M3_PRODUCTION = [
    BlockCase("minimax_m3", "2x4", 5120, packing="c5120"),
    *(
        BlockCase("minimax_m3", "2x4", 5120, packing="c5120", pad=(M3_PREFIX, M3_PREFIX + n))
        for n in (640, 1600, 3072, 1780)
    ),
    BlockCase("minimax_m3", "2x4", 5120, packing="c5120-resume", pad=(M3_RESUME, M3_RESUME + 5120)),
    BlockCase("minimax_m3", "2x4", 5120, packing="c5120-resume", pad=(M3_RESUME, M3_RESUME + 1780)),
    BlockCase("minimax_m3", "2x4", 4096, packing="4xc1024"),
    BlockCase("minimax_m3", "2x4", 2048, packing="c2048"),
]

CASES = [
    BlockCase("glm53", "8x4", 5120),
    BlockCase("glm53", "8x4", 10240),
    BlockCase("glm53", "4x4-y", 5120),
    BlockCase("glm53", "4x4-y", 10240),  # (local baselines: the SP-ring 10240 point, K3 past the old head L1 limit)
    BlockCase("kimi_k3", "8x4", 10240),
    BlockCase("kimi_k3", "4x4-y", 5120),
    BlockCase("kimi_k3", "4x4-y", 7680),
    BlockCase("kimi_k3", "4x4-y", 10240),
    BlockCase("glm53", "4x4-xy", 7680),
    BlockCase("kimi_k3", "8x4", 5120),
    BlockCase("kimi_k3", "4x4-y", 2560),
    BlockCase("kimi_k3", "8x4", 5120, weights="weka", layer=1),
    BlockCase("dsv41_flash", "8x4", 5120),
    *M3_PRODUCTION,
]


def _params(cases):
    out = []
    for c in cases:
        shape, make_params, topology = MESHES[c.mesh]
        out.append(
            pytest.param(
                c,
                shape,
                make_params(**MODELS[c.model].mesh_params()),
                marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology),
                id=c.id,
            )
        )
    return out


PARAMS = pytest.mark.parametrize("case, mesh_device, device_params", _params(CASES), indirect=["mesh_device", "device_params"])


# ---------------------------------------------------------------------------------------------------------- the runs
COMBINE_SNAP = 2.0**-14  # (the determinism test's combine snap q)
K3_GOLDEN_PCC = float(os.environ.get("K3_GOLDEN_PCC", "0.97"))
K3_GOLDEN_START = int(os.environ.get("K3_GOLDEN_START", str(4 * 5120)))  # (the chunk's first token in the trace)


def _require(ok: bool, what: str):
    if ok:
        return
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(f"CI runs this case: {what}")
    pytest.skip(what)


@dataclass
class _Run:
    ffn: object
    x_tt: object
    want: Optional[torch.Tensor]  # [S, H] fp32: the reference or the golden output
    pcc: float
    pad_words: Optional[tuple] = None  # (a padded chunk: the block's actual_start / actual_end device words)

    def __call__(self):
        """One block call on the case's input (a padded chunk with its metadata words: traceable)."""
        if self.pad_words is None:
            return self.ffn(self.x_tt)
        block = self.ffn._block(int(self.x_tt.shape[-2]) * tuple(self.ffn.mesh.shape)[0])
        return block(self.x_tt, self.ffn._weights(block), actual_start=self.pad_words[0], actual_end=self.pad_words[1])


def _real_rows(case: BlockCase, R: int) -> torch.Tensor:
    """[S] bool: the case's real rows in the host x's order (SP row r's rows r S/R ..: a prefix of each, as main's
    rotated layout puts them)."""
    from models.demos.common.prefill.chunk_layout import rotated_chunk_real_counts

    local = case.chunk // R
    real = torch.ones(case.chunk, dtype=torch.bool)
    if case.pad is not None:
        start, end = case.pad
        for r, n in enumerate(rotated_chunk_real_counts(start, end - start, R, local)):
            real[r * local + n : (r + 1) * local] = False
    return real


def _setup(case: BlockCase, mesh_device, combine_snap: Optional[float] = None) -> _Run:
    """The block for the case (weights loaded on its first call) and its input on the device. combine_snap: the
    combine's snap q (None: production's, off)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import MoEBlockFfnOptions, TtK3MoEBlockFfn

    opts = MoEBlockFfnOptions(combine_snap=combine_snap)

    m = MODELS[case.model]
    R, C = tuple(mesh_device.shape)
    H = m.cfg.EMB_SIZE
    if case.weights == "weka":
        assert m.weka, f"{m.name}: no Weka weights / goldens in these tests"
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace

        trace = resolve_trace(TRACE_100K)
        L = case.layer
        _require(trace is not None, f"no K3 golden trace at {TRACE_100K} ($KIMI_K3_GOLDEN_TRACE)")
        _require(
            trace.has("moe_io", f"moe_input_layer_{L}") and trace.has("moe_io", f"moe_output_layer_{L}"),
            f"{trace.path.name} has no moe_io for layer {L}",
        )
        _require(resolve_checkpoint() is not None, "no K3 checkpoint ($KIMI_K3_CKPT)")
        s0, s1 = K3_GOLDEN_START, K3_GOLDEN_START + case.chunk
        x = trace.rows("moe_io", f"moe_input_layer_{L}", s0, s1).to(torch.bfloat16)
        want = trace.rows("moe_io", f"moe_output_layer_{L}", s0, s1).float()
        ffn = TtK3MoEBlockFfn(mesh_device, m.cfg, layer_idx=L, options=opts)
        pcc = case.pcc if case.pcc is not None else K3_GOLDEN_PCC
    else:
        sd = m.weights(R * C)
        x = (torch.randn(case.chunk, H, generator=torch.Generator().manual_seed(7)) * m.x_scale).to(torch.bfloat16)
        want = None  # (computed only by the accuracy test: the host reference costs minutes)
        ffn = TtK3MoEBlockFfn(mesh_device, m.cfg, layer_idx=m.layer, state_dict=sd, options=opts)
        ffn._test_sd = sd
        pcc = case.pcc if case.pcc is not None else m.pcc
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, case.chunk, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3)),
    )
    ffn._test_x = x
    run = _Run(ffn, x_tt, want, pcc)
    if case.pad is not None:
        assert case.weights == "random", "padded cases: random weights only"
        block = ffn._block(case.chunk)
        run.pad_words = block.chunk_scalars(*case.pad)  # (written now, before any trace: replays read them)
    return run


def _to_host(y_tt, mesh_device, chunk, H):
    R, C = tuple(mesh_device.shape)
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    return ttnn.to_torch(y_tt, mesh_composer=comp).float().reshape(chunk, H)


# ---------------------------------------------------------------------------------------------------------- the tests
@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_accuracy(case, mesh_device, device_params):
    m = MODELS[case.model]
    run = _setup(case, mesh_device)
    H = m.cfg.EMB_SIZE
    y = _to_host(run(), mesh_device, case.chunk, H)
    # (a padded chunk: its real rows against the reference; the padding rows carry the shared expert's output only and
    # are discarded by the model - finite is all they need be)
    real = _real_rows(case, tuple(mesh_device.shape)[0])
    assert torch.isfinite(y).all(), f"{case.id}: non-finite outputs"
    x = run.ffn._test_x if run.want is None else None
    want = run.want[real] if run.want is not None else m.reference(x[real], run.ffn._test_sd)
    ok, msg = comp_pcc(want.reshape(-1, H), y[real], run.pcc)
    logger.info(f"ACCURACY {case.id}: {msg} over {int(real.sum())} real rows (floor {run.pcc})")
    assert ok, f"{case.id}: the block vs the reference: {msg} (need {run.pcc})"


@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_determinism(case, mesh_device, device_params):
    # (with the combine's snap: every routed term rounded to a multiple of q, so its fp32 sums do not depend on the order
    # the rows land in - without it they do, and runs differ in the last bits)
    run = _setup(case, mesh_device, combine_snap=COMBINE_SNAP)
    ttnn.deallocate(run())  # (weights, programs)
    det = DeterminismMarker(mesh_device)
    for _ in range(case.calls):
        y = run()
        det.add({"y": y})
        ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(case.calls):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        det.add({"y": y_tr})
    ttnn.release_trace(mesh_device, tid)
    det.assert_identical(f"{case.id}: {case.calls} eager calls + {case.calls} traced replays")


def _spread(t):
    """{op: [ns]} -> 'op median (min-max) us | ...'"""
    return " | ".join(
        f"{op} {median_us(v):.0f} ({min(v) / 1e3:.0f}-{max(v) / 1e3:.0f}) us x{len(v)}"
        for op, v in t.items()
        if v and op != "other"
    ) + (f" | other {len(t['other'])} programs {sum(t['other']) / 1e3:.0f} us" if t.get("other") else "")


@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_perf(case, mesh_device, device_params):
    run = _setup(case, mesh_device)
    ttnn.deallocate(run())  # (weights, programs)

    def eager():
        for _ in range(case.calls):
            ttnn.deallocate(run())

    # (eager, logged only: a chip's program that starts late on host dispatch shows the wait as device time)
    te = op_device_times(mesh_device, eager)
    logger.info(f"PERF {case.id} eager: " + _spread(te))
    # A barrier ahead of the block in the trace (op_checks.MeshBarrier): every chip starts the head together - with it
    # the head is 202 (199-204) us, with or without a long device lead ahead. MOE_PERF_BARRIER=0 measures without it.
    barrier = MeshBarrier(mesh_device) if os.environ.get("MOE_PERF_BARRIER", "1") == "1" else None

    def sync():
        if barrier is not None:
            barrier()

    sync()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    sync()
    run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # (a warm replay)
    t = traced_op_device_times(mesh_device, tid, case.calls)
    ttnn.release_trace(mesh_device, tid)
    if barrier is not None:
        barrier.close()
    measured = {op: median_us(t[op]) for op in BLOCK_OPS}
    logger.info(f"PERF {case.id} traced: " + _spread(t))
    if not case.perf_us:
        pytest.skip(f"{case.id}: no expected device times yet - measured {measured}")
    for op, expected in case.perf_us.items():
        assert_not_slower(f"{case.id} {op}", measured[op], expected, case.perf_margin)
