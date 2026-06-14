"""Phase 2: ZAYA1 MoE block (MLP router + 16 swiglu experts, top-1, MoD skip).

Correctness-first dense implementation, fully on-device:
  router -> softmax(17) -> +balancing_bias -> top-1 one-hot gate
  expert_term = sum_{e=0..15} expert_e(hidden) * gate[:, e]
  skip_term   = hidden * gate[:, 16]              # MoD: skip expert == identity
  out         = expert_term + skip_term

Matches HF ZayaBlock output (expert_output weighted by route_prob, no residual).
EDA (cross-layer router state) is added in Phase 4; this validates layer 1 where
EDA is disabled (layer_number == 1).
"""
import os
import torch
import ttnn

from .model_args import ZayaConfig
from .standard import to_dev

ROUTER_NORM_EPS = 1e-6  # ZayaRouter uses layernorm_epsilon default (1e-6), NOT norm_epsilon

# Expert weights dominate decode DRAM bandwidth (16 experts x 402MB/layer x 40 layers
# ~16GB/token read densely). bfp8_b halves the bytes -> ~2x the MoE read, at some
# accuracy cost (validate token-exactness). Toggle via ZAYA_EXPERT_DTYPE=bf16|bfp8.
_EXPERT_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("ZAYA_EXPERT_DTYPE", "bf16")]


def _lin_w(w, name, device, dtype=ttnn.bfloat16):
    """Load an nn.Linear weight [out,in] transposed to [in,out] for ttnn.linear."""
    return to_dev(w.get(name).t().contiguous(), device, dtype=dtype)


def _bias(w, name, device, dtype=ttnn.bfloat16):
    return to_dev(w.get(name).reshape(1, -1), device, dtype=dtype)


class ExpertMLP:
    """swiglu expert: fc1 2048->4096, swiglu->2048, fc2 2048->2048 (no bias)."""

    def __init__(self, device, w, layer, e):
        p = f"model.layers.{layer}.zaya_block.experts.local_experts.{e}"
        self.fc1 = _lin_w(w, f"{p}.linear_fc1.weight", device)   # [2048,4096]
        self.fc2 = _lin_w(w, f"{p}.linear_fc2.weight", device)   # [2048,2048]
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def __call__(self, x):
        h = ttnn.linear(x, self.fc1, compute_kernel_config=self.ckc)  # [.,4096]
        d = ZayaConfig.dim
        a = ttnn.slice(h, [0, 0, 0], [h.shape[0], h.shape[1], d])
        b = ttnn.slice(h, [0, 0, d], [h.shape[0], h.shape[1], 2 * d])
        g = ttnn.mul(ttnn.silu(a), b)                                 # [.,2048]
        return ttnn.linear(g, self.fc2, compute_kernel_config=self.ckc)


class Router:
    # NOTE: bf16 router (validated in Phase 2). Top-1 routing is a discrete cliff:
    # occasionally a borderline token picks a different expert than the fp32 reference,
    # which can flip the final token on close calls. fp32 router is a Phase-6 accuracy
    # item (fp32 decision ops misbehave on this build; needs investigation).
    def __init__(self, device, w, layer):
        p = f"model.layers.{layer}.zaya_block.router"
        self.device = device
        self.down_w = _lin_w(w, f"{p}.down_proj.weight", device)     # [2048,256]
        self.down_b = _bias(w, f"{p}.down_proj.bias", device)
        self.norm_w = to_dev(w.get(f"{p}.rmsnorm_eda.weight").reshape(1, -1), device)
        self.m0_w = _lin_w(w, f"{p}.router_mlp.0.weight", device)
        self.m0_b = _bias(w, f"{p}.router_mlp.0.bias", device)
        self.m2_w = _lin_w(w, f"{p}.router_mlp.2.weight", device)
        self.m2_b = _bias(w, f"{p}.router_mlp.2.bias", device)
        self.m4_w = _lin_w(w, f"{p}.router_mlp.4.weight", device)    # [256,17], no bias
        self.bal = to_dev(w.get(f"{p}.balancing_biases").reshape(1, -1), device, dtype=ttnn.float32)
        eda_name = f"{p}.router_states_scale"
        self.eda_scale = to_dev(w.get(eda_name).reshape(1, -1), device) if eda_name in w else None
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def forward(self, hidden, prev_router=None):
        """hidden: [1,S,2048] (post input_norm). Returns (gate[1,S,17], router_next[1,S,256])."""
        hs = ttnn.linear(hidden, self.down_w, bias=self.down_b, compute_kernel_config=self.ckc)  # [1,S,256]
        if self.eda_scale is not None and prev_router is not None:
            hs = ttnn.add(hs, ttnn.mul(prev_router, self.eda_scale))  # EDA cross-layer state
        router_next = hs                                  # pre-norm, carried for EDA
        hs_n = ttnn.rms_norm(hs, epsilon=ROUTER_NORM_EPS, weight=self.norm_w)
        x = ttnn.linear(hs_n, self.m0_w, bias=self.m0_b, compute_kernel_config=self.ckc)
        x = ttnn.gelu(x, fast_and_approximate_mode=False)
        x = ttnn.linear(x, self.m2_w, bias=self.m2_b, compute_kernel_config=self.ckc)
        x = ttnn.gelu(x, fast_and_approximate_mode=False)
        logits = ttnn.linear(x, self.m4_w, compute_kernel_config=self.ckc)  # [1,S,17]
        prob = ttnn.softmax(logits, dim=-1)
        biased = ttnn.add(prob, self.bal)
        mx = ttnn.max(biased, dim=-1, keepdim=True)       # [1,S,1]
        onehot = ttnn.eq(biased, mx)                      # [1,S,17] (ties -> rare)
        gate = ttnn.mul(onehot, prob)                     # route_prob in chosen slot
        return gate, router_next


class ZayaMoEBlock:
    """All 16 experts run as ONE batched matmul (weights stacked on a batch dim),
    then gate-weighted-summed. Fully on-device, no host sync; ~13x fewer op
    dispatches than the per-expert loop (the decode dispatch bottleneck)."""

    def __init__(self, device, w, layer):
        self.device = device
        self.router = Router(device, w, layer)
        n_e = ZayaConfig.n_experts
        p = f"model.layers.{layer}.zaya_block.experts.local_experts"
        fc1 = torch.stack([w.get(f"{p}.{e}.linear_fc1.weight").t().contiguous() for e in range(n_e)], 0)  # [16,2048,4096]
        fc2 = torch.stack([w.get(f"{p}.{e}.linear_fc2.weight").t().contiguous() for e in range(n_e)], 0)  # [16,2048,2048]
        self.fc1 = to_dev(fc1, device, dtype=_EXPERT_DTYPE)
        self.fc2 = to_dev(fc2, device, dtype=_EXPERT_DTYPE)
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def forward(self, hidden, prev_router=None, return_gate=False):
        """hidden: [1,S,2048] post input_norm. Returns (out[1,S,2048], router_next[1,S,256]
        [, gate[1,S,17] if return_gate])."""
        gate, router_next = self.router.forward(hidden, prev_router)       # [1,S,17]
        n_e = ZayaConfig.n_experts
        S, d = hidden.shape[1], ZayaConfig.dim
        h16 = ttnn.repeat(hidden, ttnn.Shape([n_e, 1, 1]))                 # [16,S,2048]
        hh = ttnn.matmul(h16, self.fc1, compute_kernel_config=self.ckc)    # [16,S,4096]
        a = ttnn.slice(hh, [0, 0, 0], [n_e, S, d])
        b = ttnn.slice(hh, [0, 0, d], [n_e, S, 2 * d])
        g = ttnn.mul(ttnn.silu(a), b)                                      # [16,S,2048] swiglu
        eo = ttnn.matmul(g, self.fc2, compute_kernel_config=self.ckc)      # [16,S,2048]
        gate_e = ttnn.permute(ttnn.slice(gate, [0, 0, 0], [1, S, n_e]), [2, 1, 0])  # [16,S,1]
        out = ttnn.sum(ttnn.mul(eo, gate_e), dim=0, keepdim=True)          # [1,S,2048]
        gskip = ttnn.slice(gate, [0, 0, n_e], [1, S, n_e + 1])             # MoD skip == identity
        out = ttnn.add(out, ttnn.mul(hidden, gskip))
        if return_gate:
            return out, router_next, gate
        return out, router_next

    def decode_forward(self, hidden, prev_router=None):
        """Decode uses the same batched dense path (fully on-device, no host sync)."""
        return self.forward(hidden, prev_router)[:2]
