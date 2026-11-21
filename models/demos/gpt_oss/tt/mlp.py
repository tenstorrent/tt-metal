# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""

import ttnn
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .batched_prefill import gather_users
from .experts import ExpertConfig, Experts
from .experts.prefill_routed import RoutedPrefillExperts, routed_prefill_supported
from .experts_throughput import (
    DeepSeekPrefillConfig,
    ThroughputExpertConfig,
    ThroughputExperts,
    create_fused_moe_gpt_config,
)
from .fused_decode.config import DENSE_MOE_TOKENS, dense_moe_decode_supported, fused_decode_supported
from .topk import TopKRouter

# dtype of the expert input rows that MLP.prefill_sequence_parallel all-gathers. bf8 measured TTFT 8K -3.3%, 32K -3.0%
# together with the local-rows router (L11c); bf16 makes the sequence-parallel prefill bit-identical to the
# replicated one.
PREFILL_SP_EXPERT_INPUT_DTYPE = ttnn.bfloat8_b


class MLP:
    """Streamlined MoE MLP combining router and experts"""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        use_throughput_experts=True,
        tokens_per_device=32,
        layer_idx=None,
    ):
        # Split state dict
        router_state_dict = substate(state_dict, "router")
        experts_state_dict = substate(state_dict, "experts")

        # Throughput experts rely on all_to_all_dispatch/combine across a mesh axis,
        # which has no meaning on a single device and would require fabric.
        if use_throughput_experts and mesh_device.get_num_devices() == 1:
            use_throughput_experts = False

        # One decode token per device under TP (no expert parallelism): decode routes and computes only the token's
        # top-k experts (router.decode_indexed + Experts streamed decode).
        self.indexed_decode = fused_decode_supported(
            mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device
        )

        # Initialize components with mesh_config
        self.router = TopKRouter(
            mesh_device,
            hf_config,
            router_state_dict,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
            indexed_decode=self.indexed_decode,
            ccl_manager=ccl_manager,
        )

        self.use_throughput_experts = use_throughput_experts
        self.dense_decode = False
        if self.use_throughput_experts:
            # Create TT config
            throughput_expert_config = ThroughputExpertConfig(
                intermediate_size=hf_config.intermediate_size,
                num_experts=hf_config.num_local_experts,
                hidden_size=hf_config.hidden_size,
                num_experts_per_tok=hf_config.num_experts_per_tok,
                num_devices=mesh_device.get_num_devices(),
            )

            # Create fused MoE config if requested
            fused_config = None
            # moe_gpt and its packed weights require twelve DRAM-aligned cores.
            # Blackhole uses the generic gather/matmul/reduce decode path.
            if not ttnn.device.is_blackhole(mesh_device):
                fused_config = create_fused_moe_gpt_config(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    state_dict=experts_state_dict,
                    tokens_per_device=tokens_per_device,
                    num_links=ccl_manager.num_links,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                )

            # DeepSeek prefill config: always created when throughput experts are
            # enabled. The two were previously gated by separate flags but were always
            # set together by every caller; now bundled.
            prefill_config = None
            deepseek_permuted_weights = None
            if use_throughput_experts:
                import torch as _torch

                from .experts_throughput.prefill import _compute_weight_permutation

                prefill_config = DeepSeekPrefillConfig(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    dispatch_group_size=mesh_device.shape[0],
                    num_dispatch_groups=mesh_device.shape[1],
                    capacity_factor=2.0,
                    seq_len_per_chip=1024,
                    num_links=ccl_manager.num_links,
                )
                # Permute expert state_dict to GROUP-BASED ordering before loading
                perm = _compute_weight_permutation(
                    mesh_device.shape[0],
                    mesh_device.shape[1],
                    throughput_expert_config.num_experts // (mesh_device.shape[0] * mesh_device.shape[1]),
                )
                perm_t = _torch.tensor(perm, dtype=_torch.long)
                permuted_sd = {
                    k: v.index_select(0, perm_t) if v.shape[0] == throughput_expert_config.num_experts else v
                    for k, v in experts_state_dict.items()
                }
                from .experts_throughput.weights import load_throughput_expert_weights

                deepseek_permuted_weights = load_throughput_expert_weights(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    state_dict=permuted_sd,
                    weight_dtype=ttnn.bfloat4_b,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts_ds_perm"),
                )
                prefill_config.permuted_weights = deepseek_permuted_weights

            # Create TT experts module
            self.experts = ThroughputExperts(
                mesh_device=mesh_device,
                config=throughput_expert_config,
                state_dict=experts_state_dict,
                weight_dtype=ttnn.bfloat4_b,
                dispatch_cluster_axis=0,
                decode_memory_config=ttnn.L1_MEMORY_CONFIG,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                fused_config=fused_config,
                prefill_config=prefill_config,
            )
        else:
            # Create expert config from HF config
            expert_config = ExpertConfig(
                intermediate_size=hf_config.intermediate_size,
                num_experts=hf_config.num_local_experts,
                hidden_size=hf_config.hidden_size,
                num_experts_per_tok=hf_config.num_experts_per_tok,
                swiglu_limit=hf_config.swiglu_limit,
            )

            # Use GPT-OSS specific program config
            program_config = GPTOSSProgramConfig()
            # One 32-token tile per device (batch 32 on Blackhole 1x4): decode routes into dense [tokens, experts]
            # weights and computes every expert (router.dense_decode + Experts decode_forward_dense).
            self.dense_decode = dense_moe_decode_supported(
                mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device
            )

            # Create experts with new modular implementation
            self.experts = Experts(
                mesh_device=mesh_device,
                config=expert_config,
                state_dict=experts_state_dict,
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                program_config=program_config,
                weight_dtype=ttnn.bfloat4_b,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                indexed_decode=self.indexed_decode,
                dense_decode_tokens=DENSE_MOE_TOKENS if self.dense_decode else None,
            )

        # Routed top-k prefill experts (experts/prefill_routed.py): only the routed (token, expert) pairs are computed,
        # with local expert parallelism over the mesh row. On by default on Blackhole 1xN meshes;
        # GPT_OSS_ROUTED_PREFILL=0 selects the dense prefill path. Prefills shorter than GPT_OSS_ROUTED_PREFILL_MIN_SEQ
        # (default 128) or not a multiple of 64 still run dense. A batched prefill (B users flattened to [1, 1, B*S, H])
        # takes the routed path too: the experts are per token, the closing reduction is per user (batch_size).
        # With the flag on, missing routed weights (no HF expert weights and an incomplete routed cache) raise
        # RoutedWeightsUnavailable instead of silently falling back to dense.
        self.routed_prefill = None
        if mesh_config is not None and routed_prefill_supported(
            mesh_device, mesh_config, hf_config, use_throughput_experts
        ):
            self.routed_prefill = RoutedPrefillExperts(
                mesh_device,
                hf_config,
                experts_state_dict,
                ccl_manager,
                mesh_config,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                layer_idx=layer_idx,
            )
        # The routed experts may hold the experts in a per-layer calibrated order (device-load balance,
        # experts/prefill_routed_placement.py); their router then emits slot ids (expert order[v] as id v).
        self.routed_router = self.router
        if self.routed_prefill is not None and self.routed_prefill.expert_order is not None:
            self.routed_router = self.router.with_expert_order(self.routed_prefill.expert_order)

    def routed_prefill_supports(self, shape):
        """Whether a prefill input of this shape takes the routed experts (else the dense prefill path)."""
        return self.routed_prefill is not None and self.routed_prefill.supports(shape)

    def prefill_sequence_parallel(self, x_rows, cluster_axis, batch_size=1):
        """Routed MoE of a sequence-parallel prefill. x_rows: this device's S / N rows of the normed input
        ([1, 1, S / N, H] bf16, device d holding rows d * S / N onward). Returns the reduce-scattered MoE output rows
        [1, 1, S / N, H] (bf8).

        The router is row-wise, so it runs on the local rows and only its top-k ids / weights ([S / N, k]) are
        all-gathered. The experts need all S rows: those are gathered in PREFILL_SP_EXPERT_INPUT_DTYPE (bf8, half the
        bytes of bf16). The expert op packs its bf16 input to bf8 itself, but this typecast rounds differently from
        that pack, so the output is not bit-identical to the replicated path (gate v4 neutral); with the constant at
        bf16 it is (tests/unit/test_prefill_sequence_parallel.py).

        batch_size > 1 (batched prefill): x_rows holds rows d * S / N onward of each of the B users (stacked by
        user); the gathers return all rows stacked by user (batched_prefill.gather_users)."""
        idx_rows, w_rows = self.routed_router(x_rows, True, batch_size)
        n = x_rows.shape[-2]
        k = idx_rows.shape[-1]

        def gather_rows(t):
            t4 = ttnn.reshape(t, (1, 1, n, k))
            full = gather_users(t4, batch_size, cluster_axis)
            t.deallocate(True)
            return ttnn.reshape(full, (full.shape[-2], k))

        expert_indices = gather_rows(idx_rows)
        expert_weights = gather_rows(w_rows)
        dtype = PREFILL_SP_EXPERT_INPUT_DTYPE
        x_in = x_rows if x_rows.dtype == dtype else ttnn.typecast(x_rows, dtype)
        x_full = gather_users(x_in, batch_size, cluster_axis, self.routed_prefill.ccl_manager)
        if x_in is not x_rows:
            x_in.deallocate(True)
        assert self.routed_prefill_supports(x_full.shape), "SP prefill needs routed MoE"
        out = self.routed_prefill(x_full, expert_indices, expert_weights, scatter=True, batch_size=batch_size)
        x_full.deallocate(True)
        ttnn.deallocate(expert_indices)
        ttnn.deallocate(expert_weights)
        return out

    def __call__(self, hidden_states, is_decode, batch_size=1):
        """Forward pass: route -> experts
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
        Returns:
            Expert output tensor [batch, seq_len, hidden_size]
        """
        if is_decode and self.indexed_decode:
            # Fused decode (fused_decode/stream.py): the router (top-k + softmax fused) and the experts read the flat
            # norm output of the layer boundary; the routed ids / weights live in shared buffers. Returns this
            # device's flat MoE partial sum (the next boundary all-reduces it).
            # hidden_states may be the pending post-attention boundary, which then runs inside the router op
            # (fused_decode/boundary.py consumer_parts); the experts read its normed output.
            expert_indices, expert_weights = self.router.decode_indexed(hidden_states)
            x = hidden_states if isinstance(hidden_states, ttnn.Tensor) else hidden_states.x
            return self.experts(
                x, topk_expert_indices=expert_indices, topk_expert_weights=expert_weights, is_decode=True
            )
        if not is_decode and self.routed_prefill_supports(hidden_states.shape):
            expert_indices, expert_weights = self.routed_router(hidden_states, True, batch_size)
            expert_output = self.routed_prefill(hidden_states, expert_indices, expert_weights, batch_size=batch_size)
            ttnn.deallocate(expert_indices)
            ttnn.deallocate(expert_weights)
            return expert_output
        if is_decode and self.dense_decode:
            # Batched decode (dense MoE): dense [tokens, experts] routing weights straight from the router.
            expert_weights = self.router.dense_decode(hidden_states)
            return self.experts(hidden_states, topk_expert_weights=expert_weights, is_decode=True)
        expert_indices, expert_weights = self.router(hidden_states, self.use_throughput_experts)
        expert_output = self.experts(
            hidden_states, topk_expert_indices=expert_indices, topk_expert_weights=expert_weights, is_decode=is_decode
        )
        return expert_output
