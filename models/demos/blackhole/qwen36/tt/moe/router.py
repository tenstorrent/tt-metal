# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE router: linear -> softmax -> topk -> (sum-normalize) -> scatter.

Fully on-device, trace-compatible. Returns dense routing weights [1,1,S,E] on device
(weights at the selected experts, zeros elsewhere) for sparse_matmul.

Simpler than gemma4's router: Qwen3-Next/Qwen3.5-MoE has NO router RMSNorm, NO input
pre-scale, and NO per-expert scale. Matmul + softmax accumulate in fp32 to match HF's
routing precision.
"""

import torch

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.blackhole.qwen36.tt import tp_common as tpc

# k values the fused generalized_moe_gate kernel supports (its C++ validation covers exactly these);
# anything else falls back to the ttnn.topk path below.
_FUSED_GATE_TOPK = (4, 6, 8)


# TTMoEGate's four persistent sharded L1 buffers are layer-INDEPENDENT, so share one set: per-layer copies reserve enough L1 to clash with the GDN prefill CBs.
_SHARED_GATE_BUFS = {}


def _share_gate_buffers(gate, mesh_device, config):
    """Point ``gate`` at the process-wide buffer set for its shape, freeing its own copies."""
    # The four buffers and the two key attributes are named literally rather than reached through
    # getattr/setattr over a name list: they are a fixed part of TTMoEGate's contract, so naming
    # them keeps every use greppable and makes an upstream rename an AttributeError here instead of
    # a silent None that would let two different shapes collide on one cache key.
    key = (
        id(mesh_device),
        config.num_experts,
        config.top_k,
        config.hidden_size,
        gate._buffer_rows,
        gate.num_blocks,
    )
    shared = _SHARED_GATE_BUFS.get(key)
    if shared is None:
        _SHARED_GATE_BUFS[key] = (gate.tt_bias, gate.tt_input_indices, gate.tt_output, gate.tt_output_indices)
        return
    own = (gate.tt_bias, gate.tt_input_indices, gate.tt_output, gate.tt_output_indices)
    gate.tt_bias, gate.tt_input_indices, gate.tt_output, gate.tt_output_indices = shared
    for buf in own:
        if buf is not None:
            ttnn.deallocate(buf)


class Qwen36Router:
    def __init__(self, mesh_device, config, state_dict, tensor_cache_path=None, dtype=ttnn.bfloat16):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_topk_prob = config.norm_topk_prob

        is_mesh = hasattr(mesh_device, "shape")
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        # HF [E,H] -> [1,1,H,E] for ttnn.linear, replicated and kept bf16; the transform runs as an as_tensor preprocess (cache-miss only).
        self.proj_weight = ttnn.as_tensor(
            state_dict["weight"] if state_dict else None,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=(str(tensor_cache_path / "moe.router.weight") if tensor_cache_path else None),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            preprocess=lambda t: t.to(torch.bfloat16).transpose(-2, -1).unsqueeze(0).unsqueeze(0),
        )
        self.compute_kernel_config = tpc.COMPUTE_HIFI2  # fp32 accumulate (see module docstring)
        # Blackhole keeps the original linear -> softmax -> topk -> scatter router (_call_bh); the
        # tuned decode matmul, the fused gate and the threshold routing below are Wormhole-only.
        self._wh = not tpc.is_blackhole()
        self.decode_gate = None
        if not self._wh:
            return
        # ttnn-auto picks a poor program for this skinny [S,H] x [H,E] decode matmul (measured
        # 24 us, and 64 us once the activation is in L1). The explicit small-grid 1D config —
        # one core per output N-tile — runs it in 14. Prefill (M > 1 tile) keeps ttnn-auto.
        n_tiles = (config.num_experts + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        self.decode_progcfg = tpc.create_matmul_1d_decode_progcfg(
            ttnn.TILE_SIZE, config.hidden_size, config.num_experts, num_cores=n_tiles
        )

        # Fused decode gate: one height-sharded op at one token per core, replacing the single-core topk and its normalize tail.
        self.mesh_device = mesh_device
        # Zeroed scatter bases cached per seq_len: ttnn.zeros is a host upload, so per-forward it would break tracing; zeros_like is a device fill.
        self._scatter_base = {}
        if (
            state_dict  # the op builds its own weights from the torch tensor; no ttnn disk-cache path
            and self.norm_topk_prob  # the op always renormalizes over the selected experts
            and config.num_experts <= 512
            and config.top_k in _FUSED_GATE_TOPK
        ):
            self.decode_gate = TTMoEGate(
                mesh_device,
                TTMoEGateConfig(
                    num_routed_experts=config.num_experts,
                    select_experts_k=config.top_k,
                    hidden_size=config.hidden_size,
                    batch_per_device=ttnn.TILE_SIZE,
                    n_group=1,
                    score_func="softmax",
                    softmax_position="pre",
                    routed_scaling_factor=1.0,
                ),
                state_dict["weight"].to(torch.float32).transpose(-2, -1).contiguous(),
            )
            _share_gate_buffers(self.decode_gate, mesh_device, config)

    def __call__(self, hidden_states):
        """hidden_states: [1,1,S,H] (replicated full hidden). Returns [1,1,S,E]."""
        if not self._wh:
            return self._call_bh(hidden_states)
        if self.decode_gate is not None and hidden_states.shape[-2] <= ttnn.TILE_SIZE:
            return self._dense_from_fused_gate(hidden_states)
        decode = hidden_states.shape[-2] <= ttnn.TILE_SIZE
        expert_scores = ttnn.linear(
            hidden_states,
            self.proj_weight,
            compute_kernel_config=self.compute_kernel_config,
            program_config=self.decode_progcfg if decode else None,
            memory_config=ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG,
        )
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)
        top_k_indices.deallocate(True)

        # Build the dense routing by thresholding at the k-th largest probability: ttnn.scatter is ROW_MAJOR-only, thresholding stays in TILE.
        kth = ttnn.slice(
            top_k_values,
            [0, 0, 0, self.top_k - 1],
            [1, 1, top_k_values.shape[-2], self.top_k],
        )  # [1,1,S,1] — topk returns descending, so element k-1 is the cutoff
        top_k_values.deallocate(True)
        above = ttnn.ge(router_probs, kth)
        kth.deallocate(True)
        dense_routing = ttnn.mul(router_probs, above)
        above.deallocate(True)
        router_probs.deallocate(True)

        # Sum-normalize the selected weights so they sum to 1 per token (HF norm_topk_prob).
        if self.norm_topk_prob:
            denom = ttnn.sum(dense_routing, dim=-1, keepdim=True)
            dense_routing = ttnn.div(dense_routing, denom)
            denom.deallocate(True)
        return dense_routing

    def _call_bh(self, hidden_states):
        """Blackhole router, unchanged: linear -> softmax -> topk -> sum-normalize -> scatter."""
        expert_scores = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)

        # Sum-normalize the top-k weights so they sum to 1 per token (HF norm_topk_prob).
        if self.norm_topk_prob:
            top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
            top_k_values = ttnn.div(top_k_values, top_k_sum)
            top_k_sum.deallocate(True)

        dense_routing = ttnn.scatter(
            ttnn.zeros_like(router_probs),
            dim=-1,
            index=top_k_indices,
            src=top_k_values,
        )
        router_probs.deallocate(True)
        top_k_values.deallocate(True)
        top_k_indices.deallocate(True)
        return dense_routing

    def _dense_from_fused_gate(self, hidden_states):
        """Fused gate -> compact (weights, indices) -> the dense [1,1,S,E] sparse_matmul wants.

        The gate emits the top-k compactly; scattering it back into a zeroed row is the price of
        keeping the dense-routing expert path, and still leaves the fused chain far ahead.
        """
        seq_len = hidden_states.shape[-2]
        weights, indices = self.decode_gate.forward(hidden_states)  # [S,1,1,k] each
        # batch-in-dim-0 -> batch-in-dim-2 is the metadata-only inverse of the gate's own view.
        weights = ttnn.reshape(weights, (1, 1, seq_len, self.top_k))
        indices = ttnn.reshape(indices, (1, 1, seq_len, self.top_k))
        base = self._scatter_base.get(seq_len)
        if base is None:
            base = ttnn.zeros(
                [1, 1, seq_len, self.num_experts],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                # DRAM: nothing reads this template, and one is cached per decode width per MoE layer -- in L1 that tips the GDN decode CBs over.
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._scatter_base[seq_len] = base
        dense_routing = ttnn.scatter(
            ttnn.zeros_like(base, memory_config=ttnn.L1_MEMORY_CONFIG), dim=-1, index=indices, src=weights
        )
        weights.deallocate(True)
        indices.deallocate(True)
        return dense_routing
