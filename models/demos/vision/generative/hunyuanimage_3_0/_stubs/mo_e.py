# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mo_e` (HunyuanMoE) of tencent/HunyuanImage-3.0.

HunyuanMoE is a mixed shared/routed MoE:

    shared = shared_mlp(x)                       # SwiGLU
    logits = gate.wg(x)                          # [tokens, num_experts]
    gates  = softmax(logits, dim=experts)
    top-k (=8) routing, weights normalized by the sum of the top-k gates
    routed = sum_e router_weight[:, e] * expert_e(x)   # each expert a SwiGLU
    out    = shared + routed

The canonical Mixtral `TtMoeLayer` models a different routing (8 experts,
top-2, no shared expert, no gate-and-up fusion), so this component is ported
directly with TTNN ops. With `moe_drop_tokens=False` the reference uses an
expert capacity equal to the max tokens-per-expert, so NO token is ever
dropped -- making this dense per-expert formulation numerically identical to
the reference dispatch/combine einsum.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0._stubs import top_k_gate as _top_k_gate

HF_MODEL_ID = "tencent/HunyuanImage-3.0"


def _to_ttnn(t, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(torch.float32),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_weight(w, device, dtype=ttnn.bfloat16):
    # nn.Linear stores [out, in]; ttnn.linear(x, W) = x @ W needs [in, out].
    return _to_ttnn(w.t().contiguous(), device, dtype=dtype)


class _TtMoE:
    def __init__(self, device, torch_module):
        self.device = device
        cfg = torch_module.config
        layer_idx = getattr(torch_module, "layer_idx", 0) or 0

        self.use_shared = bool(getattr(cfg, "use_mixed_mlp_moe", False))
        topk = torch_module.gate.moe_topk
        self.moe_topk = int(topk if isinstance(topk, int) else topk[layer_idx])
        self.num_experts = int(torch_module.num_experts)

        # Composed graduated gate: HunyuanMoE.gate == HunyuanTopKGate. The gate
        # stub owns the router (softmax + top-k) AND the load-balance l_aux; its
        # router weights feed the expert-combine below (main forward path).
        self.gate = _top_k_gate.build(device, torch_module.gate)
        # Gate 2 real-invocation counter.
        self.num_calls = 0
        # NOTE: the router weight `wg` now lives in the composed gate stub;
        # mo_e no longer holds a duplicate copy.

        def _mlp_weights(mlp):
            return (
                _linear_weight(mlp.gate_and_up_proj.weight, device),
                _linear_weight(mlp.down_proj.weight, device),
                int(mlp.gate_and_up_proj.weight.shape[0] // 2),  # intermediate (post-split)
            )

        if self.use_shared:
            self.shared_gu, self.shared_down, self.shared_inter = _mlp_weights(torch_module.shared_mlp)
        self.experts = [_mlp_weights(e) for e in torch_module.experts]

    def _swiglu(self, x, gu_w, down_w, inter):
        gu = ttnn.linear(x, gu_w)
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], inter])
        x2 = ttnn.slice(gu, [0, 0, inter], [gu.shape[0], gu.shape[1], 2 * inter])
        ttnn.deallocate(gu)
        act = ttnn.multiply(x1, ttnn.silu(x2))
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        out = ttnn.linear(act, down_w)
        ttnn.deallocate(act)
        return out

    def __call__(self, hidden_states, return_l_aux=False, **kwargs):
        x = hidden_states

        # --- routing via the composed graduated gate (top_k_gate) ---
        # The gate returns the load-balance l_aux AND the normalized top-k
        # router weights [1, S, E] the experts are combined with below.
        l_aux, router = self.gate(x, return_router=True)
        self.num_calls += 1

        # --- shared expert ---
        combined = None
        if self.use_shared:
            combined = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_inter)

        # --- routed experts (dense; no token dropping in this config) ---
        for e in range(self.num_experts):
            gu_w, down_w, inter = self.experts[e]
            y = self._swiglu(x, gu_w, down_w, inter)
            w = ttnn.slice(router, [0, 0, e], [router.shape[0], router.shape[1], e + 1])
            y = ttnn.multiply(y, w)
            ttnn.deallocate(w)
            if combined is None:
                combined = y
            else:
                combined = ttnn.add(combined, y)
                ttnn.deallocate(y)
        ttnn.deallocate(router)
        if return_l_aux:
            return combined, l_aux
        ttnn.deallocate(l_aux)
        return combined


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("mo_e native port requires the HF torch_module to extract weights.")
    return _TtMoE(device, torch_module)


def mo_e(device, torch_module=None):
    return build(device, torch_module)
