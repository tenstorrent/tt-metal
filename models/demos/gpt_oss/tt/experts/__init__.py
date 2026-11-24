# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generic MoE Experts Module.

This module provides a model-agnostic implementation of MoE experts.
Models provide their own ProgramConfig implementations for customization.

Usage:
    from models.demos.gpt_oss.tt.experts import Experts, ExpertConfig
    from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig

    config = ExpertConfig(...)
    program_config = GPTOSSProgramConfig()

    experts = Experts(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=program_config,
    )

    output = experts(hidden_states, routing_weights)
"""

import ttnn

from models.demos.gpt_oss.config import MeshConfig, ModeConfig

from .config import ExpertConfig, ProgramConfig
from .decode import (
    DenseDecodePlan,
    IndexedDecodeTensors,
    decode_forward,
    decode_forward_dense,
    decode_forward_indexed,
    decode_forward_stream,
)

from .. import opt_switches

# opt_switches: DECODE_TOPK = decode MoE over the top-k routed experts only (decode_forward_indexed); PREFILL_FUSEBIAS =
# EP=1 prefill experts with the per-expert bias fused into the (indexed, all experts) sparse matmuls, on the M-blocked
# prefill path (prefill._process_prefill_chunk_mblock).
from .operations import apply_tensor_parallel_allreduce
from .prefill import prefill_forward
from .weights import load_decode_expert_weights, load_dense_decode_expert_weights, load_expert_weights

__all__ = ["Experts", "ExpertConfig", "ProgramConfig"]


class Experts:
    """
    Generic MoE Expert implementation with automatic decode/prefill dispatch.

    This class provides a clean interface for expert layers. Models provide
    their own ProgramConfig implementations to customize behavior.
    """

    def __init__(
        self,
        mesh_device,
        config: ExpertConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        weight_dtype=ttnn.bfloat4_b,
        tensor_cache_path=None,
        indexed_decode=False,
        dense_decode_tokens=None,
        down_weight_dtype=None,
    ):
        """
        Initialize expert layers.

        Args:
            mesh_device: TTNN mesh device
            config: Expert configuration
            state_dict: Expert weights dictionary
            ccl_manager: Communication manager
            mesh_config: Mesh parallelization configuration
            program_config: Model-specific program configurations
            weight_dtype: Data type for weights (default: bfloat4_b)
            tensor_cache_path: Optional path for weight caching
            indexed_decode: Fused decode: only the routed experts are computed (DRAM-streaming ops, one token per
                device, TP without expert parallelism); requires the router's decode_indexed outputs.
            dense_decode_tokens: Batched decode with this many tokens per device (fused_decode.config
                dense_moe_decode_supported): every expert is computed for every token by two DRAM-sharded matmuls
                over all experts side by side (decode_forward_dense).
        """
        self.config = config
        self.mesh_config = mesh_config
        self.weight_dtype, self.tensor_cache_path = weight_dtype, tensor_cache_path
        # down_proj's dtype (default weight_dtype; the ND experts can keep down in another dtype than gate / up)
        self.down_weight_dtype = down_weight_dtype or weight_dtype
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config

        # Load weights. The batched dense path (dense_decode_tokens) serves prefill too, from its own layout, so the
        # per-expert weights are not loaded there (they are ~2.5 GB per device for gpt-oss-20b at TP=4, the margin a
        # 128k-token context needs at batch 32).
        self.weights = None
        self.prefill_sparsity = None
        # TP=1 fused decode (one chip) and gpt-oss-120b at TP=4: the experts are stored once, ND-sharded, and both
        # prefill and the fused decode read them (fused_decode/nd.py); there is no room for a decode-only copy next to
        # them (fused_decode/config.py nd_experts_on).
        from ..bfp4_quant import expert_bfp4_rounding
        from ..fused_decode.config import nd_experts_on

        nd = bool(indexed_decode) and nd_experts_on(mesh_config.decode.tp, config.num_experts)
        # BFP4 rounding (../bfp4_quant.py): the fused decode path's (GPT_OSS_EXPERT_BFP4); the original path keeps ttnn's
        self.bfp4_rounding = expert_bfp4_rounding(nd) if indexed_decode else "max"
        if not dense_decode_tokens:
            self.weights = load_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                mesh_config=mesh_config,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
                nd=nd,
                down_weight_dtype=self.down_weight_dtype,
                rounding=self.bfp4_rounding,
            )
            # Cache prefill sparsity (created once, reused for all prefill calls)
            self.prefill_sparsity = self._create_prefill_sparsity()

        # Fused decode (TP over mesh columns, no expert parallelism): only the routed experts are computed, by the
        # DRAM-streaming ops of fused_decode/stream.py (ops and buffers shared by every layer through the CCL manager).
        self.decode_weights = None
        self.decode_stream = None
        self.decode_nd = None
        if nd:
            assert mesh_config.decode.ep == 1, "fused decode needs EP == 1"
            self._init_decode_nd(state_dict)
        elif indexed_decode:
            assert mesh_config.decode.tp > 1 and mesh_config.decode.ep == 1, "fused decode needs TP > 1, EP == 1"
            from ..fused_decode.config import expert_decode_weight_dtype

            # The decode copies have their own dtype (fused_decode/config.py: GPT_OSS_FUSED_EXPERT_DTYPE, BFP4 by
            # default, K-grouped); prefill keeps weight_dtype.
            self.decode_weight_dtype = expert_decode_weight_dtype()
            self.decode_weights = load_decode_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                mesh_config=mesh_config,
                weight_dtype=self.decode_weight_dtype,
                tensor_cache_path=tensor_cache_path,
                rounding=self.bfp4_rounding,
            )
            self.decode_stream = ccl_manager.get_decode_expert_stream(
                config.hidden_size,
                self.decode_weights.intermediate_padded,
                config.num_experts_per_tok,
                config.swiglu_limit,
                config.alpha,
                weight_dtype=self.decode_weight_dtype,
                k_grouped=self.decode_weights.k_grouped,
            )

        self.dense_decode_weights = None
        if dense_decode_tokens:
            self.dense_decode_weights = load_dense_decode_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                mesh_config=mesh_config,
                tokens=dense_decode_tokens,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
            )
            self.dense_decode_plan = DenseDecodePlan(
                mesh_device,
                config.hidden_size,
                self.dense_decode_weights.intermediate_local,
                config.num_experts,
                dense_decode_tokens,
            )

        # GPT_OSS_OPT_DECODE_TOPK / PREFILL_FUSEBIAS: per-expert bias tables for the indexed sparse matmuls, see
        # decode_forward_indexed.
        self.indexed_decode_tensors = None
        if (
            (opt_switches.on("DECODE_TOPK") or opt_switches.on("PREFILL_FUSEBIAS"))
            and self.weights is not None
            and (not indexed_decode or nd)
            and mesh_config.decode.ep == 1
        ):
            # ND experts at TP > 1 (gpt-oss-120b): only prefill uses the tables; built per prefill call (32x padded
            # rows: ~1.3 GB per device for 36 layers of 128 experts).
            lazy = nd and mesh_config.decode.tp > 1
            self.indexed_decode_tensors = IndexedDecodeTensors(mesh_device, self.weights, config, lazy=lazy)

        # For backward compatibility
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

    def _init_decode_nd(self, state_dict):
        """Fused decode MoE over the ND-sharded experts (fused_decode/nd.py; TP=1, or TP > 1 for gpt-oss-120b): the
        routed-expert ops streaming self.weights' ND-sharded gate / up / down tensors, plus their decode-only bias tiles
        (per device: the packed [E, 2 I_pad] = [gate | up] shard biases and the [E, H] down bias, real on the first TP
        device only (row-parallel)), in the experts' weight dtype."""
        import torch

        from ..fused_decode.nd import ND_BIAS_PACK, nd_bias_tensor
        from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

        c = self.config
        E, H, I = c.num_experts, c.hidden_size, c.intermediate_size
        tp = self.mesh_config.decode.tp
        il = I // tp
        ip = self.weights.intermediate_size_per_device  # il padded to whole tiles (ND experts at TP > 1)
        names = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}
        dt, ddt = names[self.weight_dtype], names[self.down_weight_dtype]
        # TP > 1: bias tiles packed 8 columns to a tile (fused_decode/nd.py ND_BIAS_PACK)
        pack = 1 if tp == 1 else ND_BIAS_PACK
        tag = "" if tp == 1 else f"_tp{tp}_i{ip}_p{pack}"
        gu_bias = dn_bias = None
        if state_dict:
            b = state_dict["gate_up_proj_bias"].float().reshape(E, 2 * I)  # gate / up interleaved

            def shard(v, d):
                return torch.nn.functional.pad(v[:, d * il : (d + 1) * il], (0, ip - il))

            # per device d: [gate shard d | up shard d], each padded to ip; devices side by side (sharded on dim -1)
            gu_bias = torch.cat([torch.cat([shard(b[..., ::2], d), shard(b[..., 1::2], d)], -1) for d in range(tp)], -1)
            db = state_dict["down_proj_bias"].float().reshape(E, H)
            dn_bias = torch.cat([db] + [torch.zeros_like(db)] * (tp - 1), -1)
        self.nd_gate_up_bias = nd_bias_tensor(
            self.mesh_device,
            gu_bias,
            E,
            get_cache_file_name(self.tensor_cache_path, f"nd_gate_up_bias_{dt}{tag}"),
            dtype=self.weight_dtype,
            tp=tp,
            parts=2,
            pack=pack,
        )
        self.nd_down_bias = nd_bias_tensor(
            self.mesh_device,
            dn_bias,
            E,
            get_cache_file_name(self.tensor_cache_path, f"nd_down_bias_{ddt}{tag}"),
            dtype=self.down_weight_dtype,
            tp=tp,
            pack=pack,
        )
        self.decode_nd = self.ccl_manager.get_decode_expert_nd(
            H, ip, c.num_experts_per_tok, c.swiglu_limit, c.alpha, self.weight_dtype, self.down_weight_dtype, pack
        )

    def _create_prefill_sparsity(self):
        """Create prefill sparsity mask once and reuse."""
        import torch
        from models.demos.gpt_oss.config import Mode

        prefill_config = self.mesh_config.get_config(Mode.PREFILL)
        prefill_ep = prefill_config.ep
        tokens_per_ep = self.config.num_experts // prefill_ep

        sparsity = torch.zeros(1, 1, prefill_ep, self.config.num_experts)
        for i in range(prefill_ep):
            sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)

        return ttnn.from_torch(
            sparsity,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None) if prefill_ep > 1 else (None, None),
                mesh_shape=self.mesh_device.shape,
                mesh_device=self.mesh_device,
            ),
        )

    def __call__(
        self,
        hidden_states,
        topk_expert_weights: ttnn.Tensor,
        is_decode: bool = True,
        topk_expert_indices: ttnn.Tensor = None,
    ):
        """
        Forward pass - automatically dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            topk_expert_weights: Sparse router scores output [seq_len, num_experts]
            is_decode: Decode mode
            topk_expert_indices: Top-k expert indices per token (unused in this version of experts)

        Returns:
            Expert output tensor [1, batch, seq_len, hidden_size]
        """
        if is_decode and self.decode_nd is not None:
            # ND experts (TP=1, gpt-oss-120b at TP=4): the routed experts streamed from the ND-sharded prefill tensors
            # (fused_decode/nd.py)
            nd, w = self.decode_nd, self.weights
            act = nd["gate_up"](
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                w.gate_proj,
                w.up_proj,
                self.nd_gate_up_bias,
                nd["act"],
            )
            return nd["down"](
                act,
                topk_expert_indices,
                topk_expert_weights,
                w.down_proj,
                self.nd_down_bias,
                nd["partial"],
                send=self.ccl_manager.decode_boundary_send(self.config.hidden_size, "moe"),
            )
        if is_decode and self.decode_stream is not None:
            return decode_forward_stream(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                weights=self.decode_weights,
                stream=self.decode_stream,
                send=self.ccl_manager.decode_boundary_send(self.config.hidden_size, "moe"),
            )
        if is_decode and self.dense_decode_weights is not None:
            partial = decode_forward_dense(
                hidden_states, topk_expert_weights, self.dense_decode_weights, self.dense_decode_plan, self.config
            )
            if getattr(self, "decode_all_reduce", None) is not None:
                # Batched decode layer (batched_decode.py): one-op all-reduce into the sharded residual layout.
                return self.decode_all_reduce(partial, "moe")
            out = apply_tensor_parallel_allreduce(
                partial, self.mesh_config, self.mesh_device, partial.shape[-2], self.ccl_manager
            )
            tokens = out.shape[-2]
            return ttnn.reshape(
                out, (1, 1, tokens, self.config.hidden_size), (1, 1, max(32, tokens), self.config.hidden_size)
            )
        if self.dense_decode_weights is not None:
            return self._prefill_forward_dense(hidden_states, topk_expert_weights)
        if (
            is_decode
            and opt_switches.on("DECODE_TOPK")
            and self.indexed_decode_tensors is not None
            and topk_expert_indices is not None
            and hidden_states.shape[2] == 1
        ):
            return decode_forward_indexed(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                weights=self.weights,
                tensors=self.indexed_decode_tensors,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
            )
        # Determine mode based on sequence length. Decode with several tokens per device (batch > 1 without expert
        # parallelism) takes the prefill path: each token's experts are the same computation either way, and the
        # one-token decode ops below only accept a single token.
        if is_decode and hidden_states.shape[2] == 1:
            return decode_forward(
                hidden_states=hidden_states,
                routing_weights=topk_expert_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
            )
        else:
            fuse_bias = opt_switches.on("PREFILL_FUSEBIAS") and self.indexed_decode_tensors is not None
            if fuse_bias:
                self.indexed_decode_tensors.materialize()
            out = prefill_forward(
                hidden_states=hidden_states,
                routing_weights=topk_expert_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
                prefill_sparsity=self.prefill_sparsity,
                **({"indexed_tensors": self.indexed_decode_tensors} if fuse_bias else {}),
            )
            if fuse_bias:
                self.indexed_decode_tensors.release()
            return out

    def _prefill_forward_dense(self, hidden_states, routing_weights):
        """Prefill MoE of the batched dense path: the sequence in tiles of plan tokens, each through
        decode_forward_dense (every expert, from the side-by-side weights), then one TP all-reduce.

        hidden_states: [1, 1, S, hidden] with S a multiple of the plan's tokens; routing_weights: [S, num_experts]."""
        plan, hidden = self.dense_decode_plan, self.config.hidden_size
        tokens = plan.tokens
        seq_len = hidden_states.shape[-2]
        assert seq_len % tokens == 0, f"prefill length {seq_len} is not a multiple of {tokens}"
        experts = routing_weights.shape[-1]
        routing_weights = ttnn.reshape(routing_weights, (seq_len, experts))
        parts, groups = [], []
        for start in range(0, seq_len, tokens):
            x = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, start + tokens, hidden])
            w = ttnn.slice(routing_weights, [start, 0], [start + tokens, experts])
            parts.append(decode_forward_dense(x, w, self.dense_decode_weights, plan, self.config))
            x.deallocate(True)
            if len(parts) == 64:
                groups.append(ttnn.concat(parts, dim=2))
                for t in parts:
                    t.deallocate(True)
                parts = []
        if parts:
            groups.append(ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0])
            if len(parts) > 1:
                for t in parts:
                    t.deallocate(True)
        partial = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)
        if len(groups) > 1:
            for t in groups:
                t.deallocate(True)
        out = apply_tensor_parallel_allreduce(partial, self.mesh_config, self.mesh_device, seq_len, self.ccl_manager)
        return ttnn.reshape(out, (1, 1, seq_len, hidden), (1, 1, max(32, seq_len), hidden))
