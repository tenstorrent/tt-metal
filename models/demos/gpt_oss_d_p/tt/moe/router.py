# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS MoE router (top-k gate) for prefill.

Ported from ``models/demos/gpt_oss/tt/topk.py::TopKRouter`` — the PLAIN, model-agnostic
routing path only. GPT-OSS routing differs from DeepSeek/MiniMax (which score ALL experts
with sigmoid + an ``e_score_correction_bias`` and then top-k). GPT-OSS instead:

    logits = hidden @ router.weight.T + router.bias      # Linear (+bias) over all experts
    weights, indices = topk(logits, k=num_experts_per_tok, sorted=True)   # top-k FIRST
    weights = softmax(weights, dim=-1)                   # softmax over the k SELECTED logits

so the router cannot reuse DeepSeek's ``TtMoEGatePrefill``. We return the SPARSE ``[tokens, k]``
output (``use_throughput_experts=True`` in the reference) that the EP dispatch/combine machinery
consumes — NOT the decode-only fused ``topk_router_gpt`` kernel, and NOT the dense scatter path.

Reference: models/demos/gpt_oss/tt/topk.py (TopKRouter.__call__ plain path + topk_router).
"""

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name


def route_tokens_to_experts(router_logits, experts_per_token, softmax_compute_config):
    """Apply GPT-OSS routing to gate logits ``[tokens, num_experts]``.

    top-k FIRST (sorted), then softmax over the k selected logits (dim=-1). Returns the sparse
    ``(expert_indices [tokens, k], expert_weights [tokens, k])`` pair. ``expert_indices`` is
    converted to uint16 / ROW_MAJOR — the layout ``TtMoERoutingSetup`` requires for its
    height-sharded ``ttnn_top_k_experts_indices`` input (see tt_moe_routing_setup.py).
    """
    if router_logits.dtype != ttnn.bfloat16:
        router_logits = ttnn.typecast(router_logits, dtype=ttnn.bfloat16)

    # top-k over experts (last dim). ttnn.topk returns (values, indices), values sorted desc.
    expert_weights, expert_indices = ttnn.topk(router_logits, k=experts_per_token, dim=-1, sorted=True)

    # Softmax over the k selected logits. dim=1 == last dim of the [tokens, k] tensor (matches the
    # gpt_oss reference, which also softmaxes the SELECTED logits, not the full expert axis).
    expert_weights = ttnn.softmax(
        expert_weights, dim=1, numeric_stable=True, compute_kernel_config=softmax_compute_config
    )

    # routing_setup wants uint16, ROW_MAJOR, height-sharded [tokens, k] indices.
    expert_indices = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    if expert_indices.dtype != ttnn.uint16:
        expert_indices = ttnn.typecast(expert_indices, ttnn.uint16)

    return expert_indices, expert_weights


class TtGptOssRouter:
    """GPT-OSS top-k router: Linear(+bias) -> top-k(sorted) -> softmax over the k selected."""

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path

        # HF gate Linear: weight [num_experts, hidden] -> [hidden, num_experts]; bias [num_experts].
        torch_weight = state_dict["weight"].transpose(0, 1) if state_dict else None
        torch_bias = state_dict["bias"].unsqueeze(0) if state_dict else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            torch_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Keep compute_config=None for the linear (quality-safe default, per gpt_oss).
        self.compute_config = None
        self.softmax_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, hidden_states):
        """Route ``hidden_states`` -> ``(expert_indices [tokens, k], expert_weights [tokens, k])``.

        ``hidden_states`` may be any shape whose trailing dim is ``hidden_dim`` (e.g. [1, 1, S, H]);
        it is flattened to ``[tokens, hidden_dim]`` first. Token count is derived from the tensor
        volume (shape[0] after reshape is tile-padded in TILE_LAYOUT)."""
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # L1 for small (decode) token counts, DRAM for large (prefill) sequences.
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        expert_indices, expert_weights = route_tokens_to_experts(router_logits, self.top_k, self.softmax_compute_config)
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights
