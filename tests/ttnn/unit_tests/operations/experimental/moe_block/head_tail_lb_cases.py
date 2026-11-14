# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The LoudBox cases the MoE head's and the MoE tail's op tests share (test_moe_head_lb.py,
test_moe_tail_lb.py): SP 2 x TP 4 (FABRIC_2D), seeded random weights, each op on its own (no dispatch, no experts,
no combine).

A case is a model config (MODELS: GLM-5.3, Kimi-K3 - the block's dims, the host reference's activation) at a test
config (the tokens a mesh row, 640 or 384: chunk = SP x that - a multiple of 32 x TP, the head's rows rule).
Determinism: CALLS calls on the same inputs, float outputs folded on the device (op_checks.DeterminismMarker), the
small integer outputs (routing rows, route words, plan tensors) compared on the host.
Perf: op_checks.PERF_RUNS (5) traced replays on the real-time program profiler, the op's slowest chip, median, against
its baseline (per machine: the LoudBox's, or a Galaxy's 2x4 tray; MOE_PERF_RECORD=1 records it)."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

mb = ttnn.experimental.moe_block

SP, TP = 2, 4
CALLS = 10  # (determinism calls; perf: op_checks.PERF_RUNS)


def act_ref(g, u, act="silu"):
    """The gate / up activation's reference (the compute kernels' variants)."""
    F = torch.nn.functional
    if act == "silu":
        return F.silu(g) * u
    if act == "situ":  # Kimi K3: beta 4 (gate), 25 (up)
        return 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)
    if act == "clamped_silu":  # DeepSeek V4: limit 10
        return F.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0)
    raise ValueError(act)


# ---------------------------------------------------------------------------------------------------- model configs
@dataclass(frozen=True)
class Model:
    E: int
    K: int
    H: int
    I: int  # routed expert hidden (the head plans the combine for it)
    Is: int  # shared expert hidden
    act: str  # act_ref name (the shared expert's, here also the routed experts')
    score: str  # "Sigmoid" | "SqrtSoftplus"
    route_scale: float
    latent: int | None = None

    def dims(self):
        if self.act == "situ":
            act = mb.ExpertActivation(kind=mb.ExpertActivationKind.Situ, beta=4.0, linear_beta=25.0)
        elif self.act == "clamped_silu":
            act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=10.0)
        else:
            act = mb.ExpertActivation(kind=mb.ExpertActivationKind.Silu)
        kw = dict(latent_dim=self.latent, latent_norm=True) if self.latent else {}
        return mb.MoEModelDims(
            num_experts=self.E,
            top_k=self.K,
            emb_dim=self.H,
            routed_expert_hidden_dim=self.I,
            routed_expert_activation=act,
            router=mb.RouterSpec(score_func=getattr(mb.RouterScoreFunc, self.score), route_scale=self.route_scale),
            shared_expert_hidden_dim=self.Is,
            shared_expert_activation=act,
            **kw,
        )

    @property
    def routed_width(self):
        return self.latent or self.H


MODELS = {
    "glm53": Model(E=256, K=8, H=6144, I=2048, Is=2048, act="silu", score="Sigmoid", route_scale=2.5),
    "k3": Model(E=896, K=16, H=7168, I=3072, Is=6144, act="situ", score="Sigmoid", route_scale=1.0, latent=3584),
}


# ---------------------------------------------------------------------------------------------------- test configs
@dataclass(frozen=True)
class Case:
    model: str
    row_tokens: int  # tokens a mesh row (chunk = SP x row_tokens)
    margin: float = 0.05  # (the baselines: each op test's perf baselines file)
    seed: int = 3

    @property
    def chunk(self):
        return SP * self.row_tokens

    @property
    def id(self):
        return f"{self.model}-row{self.row_tokens}"


# (Kimi-K3 is not a case: a latent model needs a torus axis (MoEBlockResources fails fast - on the plain 2D fabric its
# first call hangs), and the LoudBox's 2x4 has none)
CASES = [Case("glm53", 640), Case("glm53", 384)]


def _payload(model):
    return (2 * model.routed_width + 32 + 63) // 64 * 64  # (moe_impl.moe_block_fabric_payload)


def _device_params(model):
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = _payload(model)
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": rc,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 16384,
        "trace_region_size": 64 << 20,
    }


def _params():
    return [pytest.param((SP, TP), _device_params(MODELS[c.model]), c, id=c.id) for c in CASES]


PARAMS = pytest.mark.parametrize("mesh_device, device_params, case", _params(), indirect=["mesh_device", "device_params"])


def _skip_unless_lb(mesh_device):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    if tuple(mesh_device.shape) != (SP, TP):
        pytest.skip(f"needs a {SP}x{TP} mesh")


# ---------------------------------------------------------------------------------------------------- host side
def _q(w, dtype):
    """w on a device dtype's grid (the reference's weights as the op holds them)."""
    return ttnn.to_torch(ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


class Weights:
    """Seeded random weights of a case, torch Linear layouts ([out, in])."""

    def __init__(self, model: Model, seed: int):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
        H, Is = model.H, model.Is
        self.gate, self.up, self.down = r(Is, H, sc=H**-0.5), r(Is, H, sc=H**-0.5), r(H, Is, sc=Is**-0.5)
        self.router, self.bias = r(model.E, H, sc=H**-0.5), r(model.E, sc=0.01)
        self.latent_down = self.latent_up = self.latent_norm = None
        if model.latent:
            L = model.latent
            self.latent_down = r(L, H, sc=H**-0.5)
            self.latent_up = r(H, L, sc=L**-0.5)
            self.latent_norm = 1.0 + r(L, sc=0.1)


def _shared_ref(model, w, x):
    """x [T, H] (bf16 values) -> the shared expert's output [T, H], fp32, on the bf8 weights."""
    b8 = ttnn.bfloat8_b
    g, u, d = _q(w.gate, b8), _q(w.up, b8), _q(w.down, b8)
    return act_ref(x @ g.T, x @ u.T, model.act) @ d.T


def _router_ref(model, w, x):
    """x [T, H] -> (ids [T, K] sorted, weights [T, K] in the ids' order), fp32 on the bf16 router weights."""
    logits = x @ _q(w.router, ttnn.bfloat16).T
    if model.score == "Sigmoid":
        s = torch.sigmoid(logits)
    else:
        s = torch.sqrt(torch.nn.functional.softplus(logits))
    sel = torch.topk(s + w.bias, model.K, dim=-1).indices
    ids, order = torch.sort(sel, dim=-1)
    ws = torch.gather(s, 1, ids)
    ws = ws / ws.sum(-1, keepdim=True) * model.route_scale
    return ids, ws


def _rmsnorm(z, gamma, eps):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * gamma


def _x_host(case, model):
    g = torch.Generator().manual_seed(1000 + case.seed)
    return torch.randn(case.chunk, model.H, generator=g).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------- mesh layouts
def _to_residual(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """[chunk, W] -> [1, 1, chunk / SP, W / TP] a chip (the residual's layout)."""
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP, TP), dims=(2, 3)),
    )


def _to_owner(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    """[chunk, W] -> [1, 1, S, W] a chip, the owners' rows (chip (r, c) owns tokens (r TP + c) S ..)."""
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )


def _from_residual(mesh, t, chunk):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(SP, TP), dims=(2, 3))).reshape(
        chunk, -1
    )


def _from_owner(t, S):
    """per-chip [.., S', W] (the first S rows the chip's own tokens) -> [chunk, W] in token order."""
    return torch.cat([ttnn.to_torch(d).reshape(-1, d.shape[-1])[:S] for d in ttnn.get_device_tensors(t)], dim=0)


def _host_ints(t):
    """an integer output's per-chip host copies (determinism on the host: small tensors)."""
    return [ttnn.to_torch(d).clone() for d in ttnn.get_device_tensors(t)]


def _pcc(name, ref, got, floor):
    ok, pcc = comp_pcc(ref.float(), got.float(), floor)
    logger.info(f"  {name}: pcc {pcc}")
    assert ok, f"{name}: pcc {pcc} < {floor}"
