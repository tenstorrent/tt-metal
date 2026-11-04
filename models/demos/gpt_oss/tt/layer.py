# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention.prefill import prefill_sequence_parallel_supported
from .attention_configs import GPTOSSAttentionProgramConfig
from .batched_prefill import gather_users
from .fused_decode.boundary import PendingBoundary
from .fused_decode.config import DECODE_BOUNDARY_FUSE_CONSUMER
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
            layer_idx=layer_idx,
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
        # the flat replicated vector of fused_decode/boundary.py, and each boundary all-reduces the row-parallel partial
        # sums, adds them to the residual and applies the next RMSNorm in one op; the MoE computes only the routed
        # experts.
        self.fused_decode = self.mlp.indexed_decode
        if self.fused_decode:
            self.boundary = ccl_manager.get_decode_boundary(
                hf_config.hidden_size, hf_config.rms_norm_eps, mesh_config.tp_axis
            )
            # The producing o_proj / MoE down ops send their partial sums themselves, and each boundary runs inside the
            # op consuming its output (fused_decode/config.py).
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
        # Batched decode (one 32-token tile per device, dense MoE; batched_decode.py): BF16 residual width-sharded in L1,
        # sharded RMSNorms, one-op all-reduces of the attention and MoE partial sums.
        self.batched_decode = not self.fused_decode and self.mlp.dense_decode
        if self.batched_decode:
            from .batched_decode import residual_memory_config

            self.residual_memory_config = residual_memory_config(mesh_device, hf_config.hidden_size)
            self.decode_all_reduce = ccl_manager.get_batched_decode_all_reduce(
                hf_config.hidden_size, mesh_config.tp_axis
            )
            self.self_attn.decode_returns_partial = True
            from .batched_decode import BatchedOProj, BatchedQKV

            self.self_attn.batched_qkv = BatchedQKV(
                mesh_device, self.self_attn.weights.wqkv, self.self_attn.config.head_dim
            )
            self.self_attn.batched_oproj = BatchedOProj(
                mesh_device, self.self_attn.weights.o_proj, self.self_attn.weights.o_proj_bias, hf_config.hidden_size
            )
            self.mlp.experts.decode_all_reduce = self.decode_all_reduce

    def _batched_decode_forward(self, hidden_states, position_embeddings, position_idx, page_table, kv_cache):
        """Batched decode: sharded norm -> attention -> one-op all-reduce -> residual add -> sharded norm -> MoE
        (all-reduced inside) -> residual add. Returns the BF16 residual in the width-sharded layout."""
        if hidden_states.memory_config() != self.residual_memory_config or hidden_states.dtype != ttnn.bfloat16:
            # First layer: the embedding output enters the residual layout.
            embeddings = hidden_states
            hidden_states = ttnn.to_memory_config(embeddings, self.residual_memory_config, dtype=ttnn.bfloat16)
            embeddings.deallocate(True)
        residual = hidden_states
        # The DRAM-sharded QKV (batched_decode.BatchedQKV) reshards the normed shards straight into its input layout.
        attn_in = self.input_layernorm.forward_sharded(hidden_states)
        attn_partial = self.self_attn(
            attn_in,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
        )
        attn_in.deallocate(True)
        attn_out = self.decode_all_reduce(attn_partial, "attn")
        hidden_states = ttnn.add(residual, attn_out, memory_config=self.residual_memory_config, dtype=ttnn.bfloat16)
        attn_out.deallocate(True)
        residual.deallocate(True)

        residual = hidden_states
        normed = self.post_attention_layernorm.forward_sharded(hidden_states)
        mlp_in = ttnn.to_memory_config(normed, ttnn.L1_MEMORY_CONFIG)
        normed.deallocate(True)
        mlp_out = self.mlp(mlp_in, is_decode=True)
        mlp_in.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, memory_config=self.residual_memory_config, dtype=ttnn.bfloat16)
        mlp_out.deallocate(True)
        residual.deallocate(True)
        return hidden_states

    def set_decode_next_norm(self, norm, is_last_layer, lm_head_fused=False):
        """Fused decode: the layer's last boundary applies the next RMSNorm (the next layer's input norm, or the
        model's final norm after the last layer). lm_head_fused: the last layer returns its pending boundary, which the
        streamed LM head runs (fused_decode/terminal.py)."""
        self.next_norm_gamma = self.boundary.flat_gamma(norm)
        self.is_last_layer = is_last_layer
        self.lm_head_fused = lm_head_fused

    # DFlash (models/demos/gpt_oss/tt/dflash/capture.py): callable(tensor, rows_split=False) receiving this layer's
    # prefill input residual (the previous layer's output); rows_split: the sequence-parallel prefill's residual, each
    # device holding its S / TP rows.
    capture_input = None

    def _consumer_input(self, pending):
        """The input of a boundary's consumer op: the pending boundary itself when it runs fused into that op
        (fused_decode.config.DECODE_BOUNDARY_FUSE_CONSUMER), else its normed output after running it on its own."""
        if self.boundary_fuse_consumer:
            return pending
        return pending.run()[1]

    def _decode_forward(self, hidden_states, position_embeddings, position_idx, page_table, kv_cache):
        """One decode token through the inter-layer contract of fused_decode/boundary.py.

        hidden_states: the previous layer's pending boundary (all-reduce of its MoE partial sums + residual add + this
        layer's input norm; fused_decode.boundary.PendingBoundary), or for the first layer the [1, 1, 1, hidden] BF16
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

    def prefill_sequence_parallel_supported(self, x, batch_size=1):
        """Whether this layer can run the sequence-parallel prefill (_prefill_forward_sp) for the replicated prefill
        input x [1, 1, B*S, H] (B users stacked by user): a single mesh row, the unpadded o_proj + all_reduce
        attention path and the routed MoE experts, with S % (32 * TP) == 0 (where both all-reduces scatter rows;
        with B > 1 each user's rows are scattered, batched_prefill.per_user_reduce)."""
        attn = self.self_attn
        rows = x.padded_shape[-2]
        return (
            self.mesh_device.shape[0] == 1
            and rows % batch_size == 0
            and (rows // batch_size) % (ttnn.TILE_SIZE * attn.mesh_config.tp) == 0
            and prefill_sequence_parallel_supported(attn.mesh_config, attn.weights, x)
            and self.mlp.routed_prefill_supports(x.shape)
        )

    def _prefill_forward_sp(self, residual, position_embeddings, page_table, kv_cache, user_id, batch_size=1):
        """Sequence-parallel prefill of one layer. residual: this device's S / TP rows of the residual stream
        ([1, 1, S / TP, H], device d holding rows d * S / TP onward). Returns the next residual rows.

        The replicated path all-reduces the o_proj and MoE partial sums (ttnn.all_reduce = reduce_scatter over the
        rows + all_gather) and then runs the residual add and the next RMSNorm on all S rows on every device. Here
        each all-reduce stops after its reduce-scatter, the add and the norm run on the local S / TP rows, and the
        all-gather moves the normed rows instead (the next projection needs all S rows). The input norm output is
        bf8 because ttnn.rms_norm keeps its input's dtype and the residual entering a layer is bf8 (the bf8 prefill
        embedding, then the in-place add into the bf8 MoE output); so the QKV projection reads the same bf8 rows as
        on the replicated path, and this gather moves as many bytes as the MoE all-reduce's gather it replaces. After
        the post-attention norm the router runs on the local rows and the experts' input is gathered as bf8
        (MLP.prefill_sequence_parallel).

        batch_size > 1 (batched prefill): residual holds rows d * S / TP onward of each of the B users, stacked by
        user ([1, 1, B * S / TP, H]); the gathers and reduce-scatters keep that layout (batched_prefill.py)."""
        cluster_axis = self.self_attn.mesh_config.tp_axis
        if self.capture_input is not None:  # DFlash: the previous layer's output, split by rows over the mesh row
            self.capture_input(residual, rows_split=True)
        if residual.shape[-2] * self.self_attn.mesh_config.tp > 32 * 1024:
            # As in __call__: reallocate the residual to limit DRAM fragmentation on long prefills.
            residual = ttnn.move(residual)
        x = self.input_layernorm(residual)
        x_full = gather_users(x, batch_size, cluster_axis, self.self_attn.ccl_manager)
        x.deallocate(True)
        seqlen = x_full.shape[-2] // batch_size  # per-user length, as in __call__
        attn_rows = self.self_attn(
            x_full,
            rope_mats=position_embeddings,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=False,
            user_id=user_id,
            batch_size=batch_size,
            sequence_parallel=True,
        )
        # x_full is consumed (deallocated) by the attention's QKV projection.
        # Post-attention residual add as in __call__: bf16 output up to 32K, bf8 in place above.
        if seqlen <= 32 * 1024 and attn_rows.dtype == ttnn.bfloat8_b:
            hidden = ttnn.add(residual, attn_rows, dtype=ttnn.bfloat16)
            attn_rows.deallocate(True)
        else:
            hidden = ttnn.add(residual, attn_rows, output_tensor=attn_rows)
        residual.deallocate(True)
        residual = hidden
        x = self.post_attention_layernorm(residual)
        moe_rows = self.mlp.prefill_sequence_parallel(x, cluster_axis, batch_size)
        x.deallocate(True)
        hidden = ttnn.add(residual, moe_rows, output_tensor=moe_rows)
        residual.deallocate(True)
        return hidden

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
        sequence_parallel=False,
    ):
        if is_decode and self.fused_decode:
            return self._decode_forward(hidden_states, position_embeddings, position_idx, page_table, kv_cache)
        if sequence_parallel:
            assert not is_decode
            return self._prefill_forward_sp(
                hidden_states, position_embeddings, page_table, kv_cache, user_id, batch_size
            )
        if is_decode and self.batched_decode:
            return self._batched_decode_forward(hidden_states, position_embeddings, position_idx, page_table, kv_cache)

        seqlen = hidden_states.shape[-2]
        if seqlen > 32 * 1024:
            # Reallocate hidden states to prevent memory fragmentation.
            hidden_states = ttnn.move(hidden_states)

        # hidden_states: [1, 1, tokens/num_rows, hidden_size/num_columns]
        # residual: [1, 1, tokens/num_rows, hidden_size/num_columns]
        if self.capture_input is not None:  # DFlash: the previous layer's output, one row per token
            self.capture_input(hidden_states)
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
        # Per-user length, as attention/prefill.py gates its activation dtype on: with batch > 1 the total token count
        # can exceed 32K while each user's sequence does not, and the total would select the bf8 in-place add.
        user_seqlen = seqlen // batch_size
        if not is_decode and user_seqlen <= 32 * 1024 and hidden_states.dtype == ttnn.bfloat8_b:
            # Prefill attention output arrives in bf8 (the all-reduce payload); up to 32K keep the post-attention
            # residual in bf16 (the bf8 in-place add failed the v4 per-prompt guard). Above 32K it stays bf8 in place,
            # as before, to bound DRAM ([S, H] bf16 is 755 MB per device at 128K).
            attn_out = hidden_states
            hidden_states = ttnn.add(residual, attn_out, dtype=ttnn.bfloat16)
            attn_out.deallocate(True)
        else:
            hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        # diff with llama: router scores
        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode, batch_size=batch_size)
        hidden_states_post_norm.deallocate(True)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states
