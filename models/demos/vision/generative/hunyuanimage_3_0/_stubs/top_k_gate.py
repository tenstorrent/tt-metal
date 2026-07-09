# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `top_k_gate` (HunyuanTopKGate) of tencent/HunyuanImage-3.0.

HunyuanTopKGate routes tokens to experts:

    logits = wg(x)                         # [tokens, num_experts], fp32 router
    gates  = softmax(logits, dim=experts)
    top-k (=8) selection per token
    l_aux  = num_experts^2 * mean_e( (fraction of tokens routed to e)
                                     * (mean gate prob for e) )     # load-balance aux

The reference `forward` returns `([l_aux, exp_capacity_rate], combine_weights,
dispatch_mask, exp_counts)`; the per-component harness reduces this via
`_normalize_out` to the scalar `l_aux`, which is what the PCC test compares.
This port computes that routing (softmax + top-k via the kth-value threshold)
natively in TTNN and returns `l_aux`.
"""

from __future__ import annotations

import torch

import ttnn

HF_MODEL_ID = "tencent/HunyuanImage-3.0"


def _to_ttnn(t, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(torch.float32),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class _TtTopKGate:
    def __init__(self, device, torch_module):
        self.device = device
        layer_idx = getattr(torch_module, "layer_idx", 0) or 0
        topk = torch_module.moe_topk
        self.moe_topk = int(topk if isinstance(topk, int) else topk[layer_idx])
        # nn.Linear stores [out, in]; ttnn.linear(x, W) = x @ W needs [in, out].
        self.wg = _to_ttnn(torch_module.wg.weight.t().contiguous(), device)
        self.num_experts = int(torch_module.wg.weight.shape[0])
        # Gate 2 real-invocation counter (bumped only on the real forward path).
        self.num_calls = 0

    def __call__(self, hidden_states, return_router=False, **kwargs):
        """Graduated load-balance `l_aux`, plus (optionally) the normalized
        top-k router weights the enclosing MoE combines experts with.

        Composed inside `mo_e`, the router weights returned here FEED the
        expert-combine on the main forward path (Gate 2); `l_aux` is the real
        load-balance co-output the HF `HunyuanTopKGate`/`topkgating` computes on
        every forward. Returns `l_aux` alone (per-component contract) unless
        `return_router=True`, then `(l_aux, router)`."""
        self.num_calls += 1
        x = hidden_states  # [1, T, hidden]
        T = x.shape[1]
        E = self.num_experts

        logits = ttnn.linear(x, self.wg)  # [1, T, E]
        gates = ttnn.softmax(logits, dim=-1)
        ttnn.deallocate(logits)

        # top-k membership mask via the kth-largest gate value per token
        topk_vals, _ = ttnn.topk(gates, self.moe_topk, dim=-1)
        kth = ttnn.slice(
            topk_vals,
            [0, 0, self.moe_topk - 1],
            [topk_vals.shape[0], topk_vals.shape[1], self.moe_topk],
        )
        ttnn.deallocate(topk_vals)
        mask = ttnn.ge(gates, kth)  # [1, T, E]  1.0 if expert in top-k
        ttnn.deallocate(kth)

        # --- normalized top-k router weights (norm_topk_prob=True) ---
        # router[:, :, e] = gates[e] / sum_{e' in top-k}(gates[e'])  if e in top-k
        # else 0. Identical to the reference `router_probs = gates / gates_s`
        # with the top-k dispatch mask applied; this is what the enclosing MoE
        # uses to combine expert outputs.
        router = None
        if return_router:
            masked_r = ttnn.multiply(gates, mask)  # [1, T, E]
            denom_r = ttnn.sum(masked_r, dim=-1, keepdim=True)  # [1, T, 1]
            router = ttnn.multiply(masked_r, ttnn.reciprocal(denom_r))
            ttnn.deallocate(masked_r)
            ttnn.deallocate(denom_r)

        # per-expert stats (reduce over the token dim=1)
        inv_T = 1.0 / float(T)
        tpe = ttnn.multiply(ttnn.sum(mask, dim=1, keepdim=True), inv_T)  # [1,1,E]
        gpe = ttnn.multiply(ttnn.sum(gates, dim=1, keepdim=True), inv_T)  # [1,1,E]
        ttnn.deallocate(mask)
        ttnn.deallocate(gates)

        prod = ttnn.multiply(tpe, gpe)  # [1,1,E]
        ttnn.deallocate(tpe)
        ttnn.deallocate(gpe)
        se = ttnn.sum(prod, dim=-1, keepdim=True)  # [1,1,1] = sum_e(tpe*gpe)
        ttnn.deallocate(prod)
        # l_aux = E^2 * mean_e(prod) = E^2 * (se / E) = E * se
        l_aux = ttnn.multiply(se, float(E))
        ttnn.deallocate(se)
        if return_router:
            return l_aux, router
        return l_aux


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("top_k_gate native port requires the HF torch_module to extract weights.")
    return _TtTopKGate(device, torch_module)


def top_k_gate(device, torch_module=None):
    return build(device, torch_module)
