# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation (torch golden) of the GPT-OSS prefill MoE experts — the
correctness oracle for the sparse_matmul + SwiGLU + routing pattern in
``tt/experts/prefill.py`` and ``tt/experts/operations.py``.

This mirrors ``models/demos/minimax_m3/reference/sparse_gqa_prefill.py`` in style:
pure-torch, no ttnn, no HF — drop it in, run it, get shapes and numbers.

WHAT MAKES THIS OP NON-TRIVIAL vs STANDARD DENSE FFN
-----------------------------------------------------
1. **Sparse dispatch**: Only ``top_k`` (=4) of 128 experts process each token.
   The TT implementation uses ``ttnn.sparse_matmul`` with a sparsity mask derived
   from the router. The reference implements this as a dense matmul + sparse mask
   (correct, not efficient — that's the reference contract).

2. **Custom SwiGLU with clamping**: GPT-OSS uses a modified SwiGLU with:
   - Gate:   clamp(gate, max=LIMIT) * sigmoid(alpha * clamp(gate, max=LIMIT))
   - Up:     clamp(up, -LIMIT, LIMIT) + 1
   - Result: gate_act * up_act
   Standard SwiGLU (gate * silu(gate) * up) is NOT used here.
   GPT-OSS constants: alpha=1.702, swiglu_limit=7.0.

3. **Routing weight application**: After the down projection, each expert's output
   is multiplied by that expert's routing weight for that token (a scalar from the
   router's per-token probability distribution), then all experts are summed.

4. **Bias on every projection**: gate, up, and down all have learned per-expert
   biases. The TT code adds them after each matmul.

ARCHITECTURE VERIFIED (GPT-OSS 120B, single chip, TP=1, EP=1)
--------------------------------------------------------------
  hidden_size        = 2880
  num_experts        = 128   (E)
  num_experts_per_tok = 4    (top_k)
  intermediate_size  = 2880  (I, per expert)
  alpha              = 1.702
  swiglu_limit       = 7.0   (LIMIT)

REAL SHAPES (single chip, TP=1, EP=1, batch=1):
  hidden_states     [1, 1,  S, 2880]    (batch_dim=1, seq_dim=2)
  routing_weights   [S, 128]            (full-expert probabilities; 0 for inactive experts)
  expert_indices    [S, 4]              (which 4 experts are active per token)
  gate_proj_weight  [128, 2880, 2880]   (per-expert)
  gate_proj_bias    [128, 2880]
  up_proj_weight    [128, 2880, 2880]
  up_proj_bias      [128, 2880]
  down_proj_weight  [128, 2880, 2880]
  down_proj_bias    [128, 2880]
  -> out            [1, 1,  S, 2880]

NOTE ON WEIGHT LAYOUT IN TT CODE
---------------------------------
The TT weight tensors are shaped as ``[1, E, hidden, intermediate]`` (gate/up) and
``[1, E, intermediate, hidden]`` (down) because the sparse_matmul kernel expects
the expert dimension at index 1. The reference uses the logical layout
``[E, hidden, intermediate]`` directly; callers should reshape/permute from the TT
layout when building inputs for this reference.

NOTE ON ROUTING WEIGHTS FORMAT
-------------------------------
``routing_weights`` here is the FULL [S, E] sparse tensor (0 for inactive experts),
not the compact [S, top_k] from the router. This matches the ``routing_weights``
argument to ``prefill_forward`` in ``tt/experts/prefill.py``.
"""

import torch

# Real GPT-OSS 120B defaults
HIDDEN_SIZE = 2880
NUM_EXPERTS = 128
TOP_K = 4
INTERMEDIATE_SIZE = 2880  # per expert
ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def swiglu_clamped(
    gate: torch.Tensor,  # [..., I]
    up: torch.Tensor,  # [..., I]
    alpha: float = ALPHA,
    limit: float = SWIGLU_LIMIT,
) -> torch.Tensor:
    """GPT-OSS custom SwiGLU with clamping.

    Unlike standard SwiGLU (gate * silu(gate) * up), GPT-OSS uses:
        gate_act  = clamp(gate, max=limit)
        glu       = gate_act * sigmoid(alpha * gate_act)
        up_act    = clamp(up, -limit, limit) + 1
        result    = glu * up_act
    """
    gate_c = gate.clamp(max=limit)
    glu = gate_c * torch.sigmoid(alpha * gate_c)
    up_c = up.clamp(min=-limit, max=limit) + 1.0
    return glu * up_c


def moe_prefill(
    hidden_states: torch.Tensor,  # [1, 1, S, hidden]
    routing_weights: torch.Tensor,  # [S, E]  (0 for inactive experts)
    gate_proj_weight: torch.Tensor,  # [E, hidden, I]
    gate_proj_bias: torch.Tensor,  # [E, I]
    up_proj_weight: torch.Tensor,  # [E, hidden, I]
    up_proj_bias: torch.Tensor,  # [E, I]
    down_proj_weight: torch.Tensor,  # [E, I, hidden]
    down_proj_bias: torch.Tensor,  # [E, hidden]
    alpha: float = ALPHA,
    limit: float = SWIGLU_LIMIT,
) -> torch.Tensor:  # [1, 1, S, hidden]
    """
    GPT-OSS MoE prefill forward: sparse gate/up/down projections with clamped SwiGLU.

    Computes the same result as the ``_process_prefill_chunk`` path in
    ``tt/experts/prefill.py`` (EP=1, TP=1, no sequence chunking).

    Routing weights are the full [S, E] sparse tensor; 0 entries are inactive
    experts and contribute 0 to the output (no explicit mask needed after multiply).
    """
    h = hidden_states.squeeze(0).squeeze(0)  # [S, hidden]

    # Gate and up projections for all experts: [S, hidden] x [E, hidden, I] -> [E, S, I]
    # (dense reference — the TT kernel uses sparse_matmul)
    gate = torch.einsum("sh,ehi->esi", h, gate_proj_weight) + gate_proj_bias.unsqueeze(1)  # [E, S, I]
    up = torch.einsum("sh,ehi->esi", h, up_proj_weight) + up_proj_bias.unsqueeze(1)  # [E, S, I]

    # SwiGLU activation
    down_input = swiglu_clamped(gate, up, alpha, limit)  # [E, S, I]

    # Down projection: [E, S, I] x [E, I, hidden] -> [E, S, hidden]
    down = torch.einsum("esi,eio->eso", down_input, down_proj_weight) + down_proj_bias.unsqueeze(1)  # [E, S, hidden]

    # Apply routing weights and reduce across experts
    # routing_weights: [S, E] -> [E, S, 1] for broadcast
    rw = routing_weights.T.unsqueeze(-1)  # [E, S, 1]
    out = (down * rw).sum(dim=0)  # [S, hidden]

    return out.unsqueeze(0).unsqueeze(0)  # [1, 1, S, hidden]


def moe_prefill_sparse(
    hidden_states: torch.Tensor,  # [1, 1, S, hidden]
    expert_indices: torch.Tensor,  # [S, top_k]  (int64)
    expert_weights: torch.Tensor,  # [S, top_k]  (compact, not full-E)
    gate_proj_weight: torch.Tensor,  # [E, hidden, I]
    gate_proj_bias: torch.Tensor,  # [E, I]
    up_proj_weight: torch.Tensor,  # [E, hidden, I]
    up_proj_bias: torch.Tensor,  # [E, I]
    down_proj_weight: torch.Tensor,  # [E, I, hidden]
    down_proj_bias: torch.Tensor,  # [E, hidden]
    alpha: float = ALPHA,
    limit: float = SWIGLU_LIMIT,
) -> torch.Tensor:  # [1, 1, S, hidden]
    """Sparse-dispatch golden: only run the top-k active experts per token.

    Unlike ``moe_prefill`` (which is dense and uses the full [S,E] routing-weight
    tensor), this variant takes the compact router outputs ``(expert_indices,
    expert_weights)`` and dispatches each token to exactly top_k experts.

    Equivalent to ``moe_prefill`` when routing_weights is built from
    ``scatter(zeros(S,E), expert_indices, expert_weights)``.  Use as an
    independent cross-check.
    """
    h = hidden_states.squeeze(0).squeeze(0)  # [S, hidden]
    S, hidden = h.shape
    top_k = expert_indices.shape[1]

    out = torch.zeros(S, hidden, dtype=h.dtype, device=h.device)
    for k in range(top_k):
        exp_ids = expert_indices[:, k]  # [S]
        weights_k = expert_weights[:, k]  # [S]

        # Gather per-token expert weights
        gate_w = gate_proj_weight[exp_ids]  # [S, hidden, I]
        gate_b = gate_proj_bias[exp_ids]  # [S, I]
        up_w = up_proj_weight[exp_ids]
        up_b = up_proj_bias[exp_ids]
        down_w = down_proj_weight[exp_ids]  # [S, I, hidden]
        down_b = down_proj_bias[exp_ids]

        gate = torch.einsum("sh,shi->si", h, gate_w) + gate_b  # [S, I]
        up = torch.einsum("sh,shi->si", h, up_w) + up_b

        act = swiglu_clamped(gate, up, alpha, limit)  # [S, I]

        down = torch.einsum("si,sio->so", act, down_w) + down_b  # [S, hidden]
        out = out + down * weights_k.unsqueeze(-1)

    return out.unsqueeze(0).unsqueeze(0)  # [1, 1, S, hidden]


def make_moe_inputs(
    S: int,
    E: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    hidden: int = HIDDEN_SIZE,
    intermediate: int = INTERMEDIATE_SIZE,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Build random MoE inputs matching the GPT-OSS prefill producer contract.

    Returns a dict with keys: hidden_states, routing_weights, expert_indices,
    expert_weights, gate_proj_weight, gate_proj_bias, up_proj_weight,
    up_proj_bias, down_proj_weight, down_proj_bias.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    hidden_states = torch.randn(1, 1, S, hidden, generator=g, dtype=dtype)

    # Simulate router: uniform top-k selection, random weights
    expert_indices = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(S)]).to(
        torch.int64
    )  # [S, top_k]
    raw_weights = torch.rand(S, top_k, generator=g, dtype=dtype)
    expert_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)  # [S, top_k]

    # Scatter to full [S, E] routing weight tensor (0 for inactive experts)
    routing_weights = torch.zeros(S, E, dtype=dtype)
    routing_weights.scatter_(1, expert_indices, expert_weights)

    # Expert projection weights (small scale for numerical stability in tests)
    gate_proj_weight = torch.randn(E, hidden, intermediate, generator=g, dtype=dtype) * 0.02
    gate_proj_bias = torch.zeros(E, intermediate, dtype=dtype)
    up_proj_weight = torch.randn(E, hidden, intermediate, generator=g, dtype=dtype) * 0.02
    up_proj_bias = torch.zeros(E, intermediate, dtype=dtype)
    down_proj_weight = torch.randn(E, intermediate, hidden, generator=g, dtype=dtype) * 0.02
    down_proj_bias = torch.zeros(E, hidden, dtype=dtype)

    return dict(
        hidden_states=hidden_states,
        routing_weights=routing_weights,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        gate_proj_weight=gate_proj_weight,
        gate_proj_bias=gate_proj_bias,
        up_proj_weight=up_proj_weight,
        up_proj_bias=up_proj_bias,
        down_proj_weight=down_proj_weight,
        down_proj_bias=down_proj_bias,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    for S, label in [(32, "S=32"), (128, "S=128")]:
        # Use smaller E/I for speed in the self-test
        inputs = make_moe_inputs(S, E=8, top_k=2, hidden=64, intermediate=64)

        out_dense = moe_prefill(
            inputs["hidden_states"],
            inputs["routing_weights"],
            inputs["gate_proj_weight"],
            inputs["gate_proj_bias"],
            inputs["up_proj_weight"],
            inputs["up_proj_bias"],
            inputs["down_proj_weight"],
            inputs["down_proj_bias"],
        )

        out_sparse = moe_prefill_sparse(
            inputs["hidden_states"],
            inputs["expert_indices"],
            inputs["expert_weights"],
            inputs["gate_proj_weight"],
            inputs["gate_proj_bias"],
            inputs["up_proj_weight"],
            inputs["up_proj_bias"],
            inputs["down_proj_weight"],
            inputs["down_proj_bias"],
        )

        max_diff = (out_dense - out_sparse).abs().max().item()
        print(f"[{label}] dense vs sparse max_diff={max_diff:.2e}  shape={tuple(out_dense.shape)}")
        assert max_diff < 1e-4, f"moe_prefill dense vs sparse mismatch: {max_diff}"

    print("all checks passed")
