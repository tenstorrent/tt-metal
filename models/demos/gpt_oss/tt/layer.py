# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import GPTOSSAttentionProgramConfig
from .decode_boundary import PendingBoundary
from .fused_decode import DECODE_BOUNDARY_FUSE_CONSUMER
from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
        create_kv_cache=True,
        transformation_mats=None,
        max_seq_len=1024,
        max_local_batch_size=1,
        users_row_sharded=False,
        use_throughput_experts=False,
        tokens_per_device=32,
    ):
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
        )
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            ccl_manager,
            dtype=dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            mesh_config=mesh_config,
            use_throughput_experts=use_throughput_experts,
            tokens_per_device=tokens_per_device,
        )

        self.attention_type = hf_config.layer_types[layer_idx]

        # Create attention configuration
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=hf_config.sliding_window,
            max_seq_len=max_seq_len,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
        )

        # Create attention program config
        attention_program_config = GPTOSSAttentionProgramConfig()

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attention_program_config,
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=transformation_mats,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
            create_kv_cache=create_kv_cache,
            fused_decode=self.mlp.indexed_decode,
        )
        self.mesh_device = mesh_device
        # Fused decode (one token per device, TP over mesh columns): between the projections the residual stream is
        # the flat replicated vector of decode_boundary.py, and each boundary all-reduces the row-parallel partial
        # sums, adds them to the residual and applies the next RMSNorm in one op; the MoE computes only the routed
        # experts.
        self.fused_decode = self.mlp.indexed_decode
        if self.fused_decode:
            self.boundary = ccl_manager.get_decode_boundary(
                hf_config.hidden_size, hf_config.rms_norm_eps, mesh_config.tp_axis
            )
            # The producing o_proj / MoE down ops send their partial sums themselves, and each boundary runs inside the
            # op consuming its output (fused_decode.py).
            self.boundary_sent = ccl_manager.decode_boundary_send(hf_config.hidden_size, "attn") is not None
            self.boundary_fuse_consumer = DECODE_BOUNDARY_FUSE_CONSUMER and self.boundary_sent
            self.input_norm_gamma = self.boundary.flat_gamma(self.input_layernorm)
            self.post_norm_gamma = self.boundary.flat_gamma(self.post_attention_layernorm)
            self.next_norm_gamma = None  # set_decode_next_norm (the model knows the next layer)
            self.is_last_layer = False
            self.lm_head_fused = False
            # The persistent buffers the streamed decode ops share across layers are allocated here, at model
            # construction, so none of them is created while a trace is live.
            attn_config = self.self_attn.config
            ccl_manager.get_decode_qkv_heads(
                mesh_config.shard_size(attn_config.num_heads),
                mesh_config.shard_size(attn_config.num_kv_heads),
                attn_config.head_dim,
            )
            ccl_manager.get_decode_partial(hf_config.hidden_size)
            ccl_manager.get_decode_router_out()

    def set_decode_next_norm(self, norm, is_last_layer, lm_head_fused=False):
        """Fused decode: the layer's last boundary applies the next RMSNorm (the next layer's input norm, or the
        model's final norm after the last layer). lm_head_fused: the last layer returns its pending boundary, which the
        streamed LM head runs (decode_terminal.py)."""
        self.next_norm_gamma = self.boundary.flat_gamma(norm)
        self.is_last_layer = is_last_layer
        self.lm_head_fused = lm_head_fused

    def _consumer_input(self, pending):
        """The input of a boundary's consumer op: the pending boundary itself when it runs fused into that op
        (fused_decode.DECODE_BOUNDARY_FUSE_CONSUMER), else its normed output after running it on its own."""
        if self.boundary_fuse_consumer:
            return pending
        return pending.run()[1]

    def _decode_forward(self, hidden_states, position_embeddings, position_idx, page_table, kv_cache):
        """One decode token through the inter-layer contract of decode_boundary.py.

        hidden_states: the previous layer's pending boundary (all-reduce of its MoE partial sums + residual add + this
        layer's input norm; decode_boundary.PendingBoundary), or for the first layer the [1, 1, 1, hidden] BF16
        row-major embedding. Runs [boundary] -> attention -> [boundary: all-reduce of the o_proj partial sums + residual
        add + post-attention norm] -> MoE, each boundary inside the op consuming its normed output (QKV, router).
        Returns the pending MoE boundary for the next layer (for the last layer: with the final norm; the model's
        streamed LM head runs it when lm_head_fused, else the layer runs it and returns the normed hidden in row 0 of
        the width-sharded [1, 1, 32, hidden] layout)."""
        if isinstance(hidden_states, PendingBoundary):
            pending = hidden_states
        else:
            pending = PendingBoundary(self.boundary, hidden_states, self.input_norm_gamma, entry=True)
        attn_partial = self.self_attn(
            self._consumer_input(pending),
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
        )
        residual = pending.residual_out
        pending.x.deallocate(True)
        pending.residual.deallocate(True)

        pending = PendingBoundary(
            self.boundary, residual, self.post_norm_gamma, attn_partial, "attn", sent=self.boundary_sent
        )
        mlp_partial = self.mlp(self._consumer_input(pending), is_decode=True)
        residual = pending.residual_out
        pending.x.deallocate(True)
        pending.residual.deallocate(True)

        pending = PendingBoundary(
            self.boundary, residual, self.next_norm_gamma, mlp_partial, "moe", sent=self.boundary_sent
        )
        if not self.is_last_layer or self.lm_head_fused:
            return pending
        residual, x = pending.run()
        pending.residual.deallocate(True)
        residual.deallocate(True)
        rows = self.boundary.to_rows(x)
        x.deallocate(True)
        return rows

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        batch_size=1,
    ):
        if is_decode and self.fused_decode:
            return self._decode_forward(hidden_states, position_embeddings, position_idx, page_table, kv_cache)

        seqlen = hidden_states.shape[-2]
        if seqlen > 32 * 1024:
            # Reallocate hidden states to prevent memory fragmentation.
            hidden_states = ttnn.move(hidden_states)

        # hidden_states: [1, 1, tokens/num_rows, hidden_size/num_columns]
        # residual: [1, 1, tokens/num_rows, hidden_size/num_columns]
        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)

        # additional all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]
        # hidden_states_post_norm: [1, 1, tokens/num_rows, hidden_size]
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            batch_size=batch_size,
        )
        hidden_states_post_norm.deallocate(True)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode)  # diff with llama: router scores
        hidden_states_post_norm.deallocate(True)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states
