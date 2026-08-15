# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Four-chip tensor-parallel Gemma-4 26B A4B decoder.

The implementation deliberately inherits the selected single-chip
``OptimizedDecoder`` orchestration and numerical policy.  Setup fractures all
material projection and active-expert tensors over the 1x4 P300 ring.  Runtime
keeps the residual stream replicated and reduces only row-parallel contraction
partials, so expanded MLP/expert/head activations never cross the fabric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_KIND,
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    PREFILL_FULL_CHUNK_SIZE,
    PREFILL_SLIDING_CHUNK_SIZE,
    TILE_SIZE,
    _bounded_cache_fill_plan,
    _DecoderWeights,
    _detect_layer_prefix,
    _layer_kind,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
    _make_single_user_cache_update_memory_config,
    _prefill_attention_path,
    _text_config,
    _validate_text_config,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder
from models.demos.gemma4.tt.experts.weights import ExpertWeights

TP_SIZE = 4
LOCAL_Q_HEADS = 4
PADDED_MLP_INTERMEDIATE_SIZE = 2176
LOCAL_MLP_INTERMEDIATE_SIZE = PADDED_MLP_INTERMEDIATE_SIZE // TP_SIZE
PADDED_MOE_INTERMEDIATE_SIZE = 768
LOCAL_MOE_INTERMEDIATE_SIZE = PADDED_MOE_INTERMEDIATE_SIZE // TP_SIZE


def _require_target_mesh(mesh_device: Any) -> None:
    if not isinstance(mesh_device, ttnn.MeshDevice):
        raise ValueError("MultichipDecoder requires a TTNN MeshDevice")
    if tuple(mesh_device.shape) != (1, TP_SIZE):
        raise ValueError(f"Gemma-4 multichip decoder targets a 1x{TP_SIZE} mesh, got {tuple(mesh_device.shape)}")


def _pad_last(torch_tensor: Any, padded: int) -> Any:
    import torch

    if torch_tensor.shape[-1] == padded:
        return torch_tensor
    if torch_tensor.shape[-1] > padded:
        raise ValueError(f"cannot pad width {torch_tensor.shape[-1]} to smaller width {padded}")
    return torch.nn.functional.pad(torch_tensor, (0, padded - torch_tensor.shape[-1]))


def _pad_penultimate(torch_tensor: Any, padded: int) -> Any:
    import torch

    if torch_tensor.shape[-2] == padded:
        return torch_tensor
    if torch_tensor.shape[-2] > padded:
        raise ValueError(f"cannot pad K {torch_tensor.shape[-2]} to smaller K {padded}")
    return torch.nn.functional.pad(torch_tensor, (0, 0, 0, padded - torch_tensor.shape[-2]))


class MultichipDecoder(OptimizedDecoder):
    """TP=4 optimized decoder with a replicated stack-compatible residual."""

    tp_size = TP_SIZE
    topology = ttnn.Topology.Ring
    cluster_axis = 1

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        attention_weight_dtype: ttnn.DataType | None = None,
        mlp_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        mlp_down_weight_dtype: ttnn.DataType | None = None,
        prefill_expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "MultichipDecoder":
        import torch

        _require_target_mesh(mesh_device)
        text_config = _text_config(hf_config)
        _validate_text_config(text_config)
        kind = _layer_kind(text_config.layer_types[layer_idx])
        prefix = _detect_layer_prefix(state_dict, layer_idx)
        cache_root = Path(tensor_cache_path) if tensor_cache_path is not None else None
        attention_weight_dtype = attention_weight_dtype or (
            ttnn.bfloat16 if kind.name == "sliding_attention" else ttnn.bfloat8_b
        )
        mlp_down_weight_dtype = mlp_down_weight_dtype or mlp_weight_dtype

        def get(name: str) -> Any:
            return state_dict[f"{prefix}.{name}"]

        def upload(
            name: str,
            source: Any,
            *,
            dtype: ttnn.DataType,
            mapper: Any,
            layout: ttnn.Layout = ttnn.TILE_LAYOUT,
        ) -> ttnn.Tensor:
            upload_kwargs = {
                "device": mesh_device,
                "layout": layout,
                "dtype": dtype,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "mesh_mapper": mapper,
            }
            if cache_root is not None:
                upload_kwargs["cache_file_name"] = str(cache_root / "multichip" / f"layer_{layer_idx}" / name)
            return ttnn.as_tensor(source, **upload_kwargs)

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shard_n = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        shard_k = ttnn.ShardTensorToMesh(mesh_device, dim=-2)

        q = get("self_attn.q_proj.weight").transpose(-2, -1).contiguous()
        k = get("self_attn.k_proj.weight").transpose(-2, -1).contiguous()
        v = k if kind.uses_k_as_v else get("self_attn.v_proj.weight").transpose(-2, -1).contiguous()
        q_shards = q.chunk(TP_SIZE, dim=-1)
        if kind is FULL_KIND:
            # One full 512-wide KV head per device.  Each GQA head is paired
            # with the two Q shards that consume it.
            k_heads = k.chunk(2, dim=-1)
            v_heads = v.chunk(2, dim=-1)
            k_shards = (k_heads[0], k_heads[0], k_heads[1], k_heads[1])
            v_shards = (v_heads[0], v_heads[0], v_heads[1], v_heads[1])
        else:
            k_shards = k.chunk(TP_SIZE, dim=-1)
            v_shards = v.chunk(TP_SIZE, dim=-1)
        qkv_mesh_source = (
            torch.cat([torch.cat((q_shards[i], k_shards[i], v_shards[i]), dim=-1) for i in range(TP_SIZE)], dim=-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        mlp_gate = _pad_last(get("mlp.gate_proj.weight").transpose(-2, -1).contiguous(), PADDED_MLP_INTERMEDIATE_SIZE)
        mlp_up = _pad_last(get("mlp.up_proj.weight").transpose(-2, -1).contiguous(), PADDED_MLP_INTERMEDIATE_SIZE)
        mlp_down = _pad_penultimate(
            get("mlp.down_proj.weight").transpose(-2, -1).contiguous(), PADDED_MLP_INTERMEDIATE_SIZE
        )
        gate_up = get("experts.gate_up_proj")
        expert_gate = _pad_last(
            gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous(), PADDED_MOE_INTERMEDIATE_SIZE
        )
        expert_up = _pad_last(
            gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous(), PADDED_MOE_INTERMEDIATE_SIZE
        )
        expert_down = _pad_penultimate(
            get("experts.down_proj").transpose(-2, -1).contiguous(), PADDED_MOE_INTERMEDIATE_SIZE
        )

        def replicated(name: str, source: Any, dtype: ttnn.DataType = weight_dtype, layout=ttnn.TILE_LAYOUT):
            return upload(name, source, dtype=dtype, mapper=replicate, layout=layout)

        weights = _DecoderWeights(
            layer_scalar=replicated("layer_scalar", get("layer_scalar").reshape(1, 1, 1, 1), ttnn.bfloat16),
            input_ln=replicated("input_ln", get("input_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_attn_ln=replicated(
                "post_attn_ln", get("post_attention_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            pre_ff_ln=replicated("pre_ff_ln", get("pre_feedforward_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_ff_ln=replicated("post_ff_ln", get("post_feedforward_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_ff_ln_1=replicated(
                "post_ff_ln_1", get("post_feedforward_layernorm_1.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            post_ff_ln_2=replicated(
                "post_ff_ln_2", get("post_feedforward_layernorm_2.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            pre_ff_ln_2=replicated(
                "pre_ff_ln_2", get("pre_feedforward_layernorm_2.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            q_norm=replicated("q_norm", get("self_attn.q_norm.weight").reshape(1, 1, 1, kind.head_dim)),
            k_norm=replicated("k_norm", get("self_attn.k_norm.weight").reshape(1, 1, 1, kind.head_dim)),
            qkv=upload("qkv_tp4", qkv_mesh_source, dtype=attention_weight_dtype, mapper=shard_n),
            o_proj=upload(
                "o_proj_tp4",
                get("self_attn.o_proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                dtype=attention_weight_dtype,
                mapper=shard_k,
            ),
            mlp_gate=upload("mlp_gate_tp4", mlp_gate.unsqueeze(0).unsqueeze(0), dtype=mlp_weight_dtype, mapper=shard_n),
            mlp_up=upload("mlp_up_tp4", mlp_up.unsqueeze(0).unsqueeze(0), dtype=mlp_weight_dtype, mapper=shard_n),
            mlp_down=upload(
                "mlp_down_tp4", mlp_down.unsqueeze(0).unsqueeze(0), dtype=mlp_down_weight_dtype, mapper=shard_k
            ),
            router_scale=replicated("router_scale", get("router.scale").reshape(1, 1, 1, HIDDEN_SIZE), ttnn.float32),
            router_proj=replicated(
                "router_proj",
                get("router.proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                ttnn.float32,
            ),
            router_per_expert_scale=replicated(
                "router_per_expert_scale", get("router.per_expert_scale").reshape(1, NUM_EXPERTS), ttnn.float32
            ),
            expert_gate=upload("expert_gate_tp4", expert_gate.unsqueeze(0), dtype=expert_weight_dtype, mapper=shard_n),
            expert_up=upload("expert_up_tp4", expert_up.unsqueeze(0), dtype=expert_weight_dtype, mapper=shard_n),
            expert_down=upload("expert_down_tp4", expert_down.unsqueeze(0), dtype=expert_weight_dtype, mapper=shard_k),
        )
        sparsity = replicated(
            "expert_prefill_sparsity",
            torch.ones(1, 1, 1, NUM_EXPERTS, dtype=torch.bfloat16),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        decoder = cls(
            hf_config=text_config,
            layer_idx=layer_idx,
            layer_kind=kind,
            mesh_device=mesh_device,
            weights=weights,
            expert_prefill_sparsity=sparsity,
            activation_dtype=activation_dtype,
            eps=text_config.rms_norm_eps,
            dense_decode_dram_sharded=False,
            dram_sharded_roles=(),
            residual_shard_cores=0,
            # TP-local expert down K is 192 = 6 tiles; the single-chip block
            # width 11 cannot divide it. Six consumes the complete local K.
            expert_down_in0_block_w=6,
            prefill_expert_down_in0_block_w=6,
            **kwargs,
        )
        decoder.expert_weights = ExpertWeights(
            gate_proj=weights.expert_gate,
            up_proj=weights.expert_up,
            down_proj=weights.expert_down,
            intermediate_size_per_device=LOCAL_MOE_INTERMEDIATE_SIZE,
        )
        decoder.packed_mlp_gate_up = ttnn.concat(
            [weights.mlp_gate, weights.mlp_up], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        decoder.decode_dram_weights = {}
        decoder.decode_dram_configs = {}
        decoder.decode_dram_input_configs = {}
        decoder.decode_dram_output_configs = {}
        decoder.multichip_path_counters = {"all_reduce": 0, "attention_tp": 0, "dense_tp": 0, "expert_tp": 0}
        return decoder

    def _all_reduce_hidden(self, partial: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["all_reduce"] += 1
        return ttnn.all_reduce(
            partial,
            cluster_axis=self.cluster_axis,
            topology=self.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _linear(self, x: ttnn.Tensor, weight_name: str, *, compute_kernel_config: Any) -> ttnn.Tensor:
        result = ttnn.linear(
            x,
            getattr(self.weights, weight_name),
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        if weight_name in {"o_proj", "mlp_down"}:
            result = self._all_reduce_hidden(result)
        return result

    def _cache_view_kwargs(self, *, prefill: bool) -> dict[str, int]:
        if self.layer_kind.name != "full_attention":
            return {}
        kwargs = {"block_size": self.layer_kind.block_size}
        if not prefill:
            kwargs["num_kv_heads"] = 1
        return kwargs

    def _fill_prefill_cache(
        self,
        key_cache,
        value_cache,
        k_heads,
        v_heads,
        page_table,
        *,
        user_id,
        logical_seq_len,
        cache_position_modulo,
        fill_kwargs,
    ) -> None:
        """Modulo-safe cache fill using TP-local, rather than global, KV heads."""
        if cache_position_modulo is None or logical_seq_len % TILE_SIZE == 0:
            modulo = {"cache_position_modulo": cache_position_modulo} if cache_position_modulo is not None else {}
            ttnn.experimental.paged_fill_cache(
                key_cache, k_heads, page_table, batch_idx=user_id, **fill_kwargs, **modulo
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, v_heads, page_table, batch_idx=user_id, **fill_kwargs, **modulo
            )
            return
        aligned_prefix, tail_positions = _bounded_cache_fill_plan(logical_seq_len)
        if aligned_prefix:
            k_prefix = ttnn.slice(k_heads, [0, 0, 0, 0], [1, k_heads.shape[1], aligned_prefix, k_heads.shape[3]])
            v_prefix = ttnn.slice(v_heads, [0, 0, 0, 0], [1, v_heads.shape[1], aligned_prefix, v_heads.shape[3]])
            ttnn.experimental.paged_fill_cache(
                key_cache,
                k_prefix,
                page_table,
                batch_idx=user_id,
                cache_position_modulo=cache_position_modulo,
                **fill_kwargs,
            )
            ttnn.experimental.paged_fill_cache(
                value_cache,
                v_prefix,
                page_table,
                batch_idx=user_id,
                cache_position_modulo=cache_position_modulo,
                **fill_kwargs,
            )
            k_prefix.deallocate(True)
            v_prefix.deallocate(True)
        page_table_row = page_table
        owns_page_table_row = False
        if page_table.shape[0] > 1:
            page_table_row = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            owns_page_table_row = True
        update_mem = _make_single_user_cache_update_memory_config(self.mesh_device, self.layer_kind.head_dim)
        update_kwargs = self._cache_view_kwargs(prefill=False)
        update_kwargs["cache_position_modulo"] = cache_position_modulo
        local_kv_heads = k_heads.shape[1]
        for position in tail_positions:
            k_token = ttnn.slice(k_heads, [0, 0, position, 0], [1, local_kv_heads, position + 1, k_heads.shape[3]])
            v_token = ttnn.slice(v_heads, [0, 0, position, 0], [1, local_kv_heads, position + 1, v_heads.shape[3]])
            k_token = ttnn.to_memory_config(ttnn.transpose(k_token, 1, 2), update_mem, dtype=k_token.dtype)
            v_token = ttnn.to_memory_config(ttnn.transpose(v_token, 1, 2), update_mem, dtype=v_token.dtype)
            position_tensor = ttnn.full(
                (1,),
                position,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.experimental.paged_update_cache(
                key_cache,
                k_token,
                update_idxs_tensor=position_tensor,
                page_table=page_table_row,
                **update_kwargs,
            )
            ttnn.experimental.paged_update_cache(
                value_cache,
                v_token,
                update_idxs_tensor=position_tensor,
                page_table=page_table_row,
                **update_kwargs,
            )
            k_token.deallocate(True)
            v_token.deallocate(True)
            position_tensor.deallocate(True)
        if owns_page_table_row:
            page_table_row.deallocate(True)

    def _attention_prefill(self, x: ttnn.Tensor, **call: Any) -> ttnn.Tensor:
        self.optimized_path_counters["prefill_attention"] += 1
        self.multichip_path_counters["attention_tp"] += 1
        kind = self.layer_kind
        seq_len = x.shape[-2]
        local_kv_heads = 1 if kind.name == "full_attention" else 2
        xqkv = ttnn.linear(
            x,
            self.weights.qkv,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_config,
        )
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=LOCAL_Q_HEADS,
            num_kv_heads=local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        q_heads = ttnn.experimental.rotary_embedding_hf(
            q_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
        )
        k_heads = ttnn.experimental.rotary_embedding_hf(
            k_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
        )
        key_cache, value_cache = call["kv_cache"]
        fill_table = call.get("chunk_page_table")
        if fill_table is None:
            fill_table = call["page_table"]
        self._fill_prefill_cache(
            key_cache,
            value_cache,
            k_heads,
            v_heads,
            fill_table,
            user_id=call["user_id"],
            logical_seq_len=call["logical_seq_len"],
            cache_position_modulo=call.get("cache_position_modulo"),
            fill_kwargs=self._cache_view_kwargs(prefill=True),
        )
        attention_path = _prefill_attention_path(
            seq_len,
            is_sliding=kind.sliding_window is not None,
            has_paged_cache=fill_table is not None,
        )
        if attention_path == "sliding_chunked":
            attn_out = self._sliding_chunked_prefill_attention(q_heads, k_heads, v_heads)
        elif attention_path == "full_chunked":
            attn_out = self._full_chunked_prefill_attention(
                q_heads, key_cache, value_cache, fill_table, user_id=call["user_id"]
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                is_causal=True,
                sliding_window_size=kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_out = ttnn.reshape(attn_out, [1, LOCAL_Q_HEADS, seq_len, kind.head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._all_reduce_hidden(
            ttnn.linear(
                attn_out,
                self.weights.o_proj,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.attention_compute_config,
            )
        )

    def _full_chunked_prefill_attention(
        self, q_heads, key_cache, value_cache, page_table, *, user_id: int
    ) -> ttnn.Tensor:
        """Run the baseline paged long-prefill algorithm with four local Q heads."""
        num_pages = page_table.shape[-1]
        user_page_table = page_table
        owns_user_page_table = False
        if page_table.shape[0] > 1:
            user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, num_pages])
            owns_user_page_table = True
        outputs = []
        seq_len = q_heads.shape[-2]
        for start in range(0, seq_len, PREFILL_FULL_CHUNK_SIZE):
            chunk_len = min(PREFILL_FULL_CHUNK_SIZE, seq_len - start)
            q_chunk = ttnn.slice(
                q_heads, [0, 0, start, 0], [1, LOCAL_Q_HEADS, start + chunk_len, self.layer_kind.head_dim]
            )
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_chunk,
                key_cache,
                value_cache,
                user_page_table,
                chunk_start_idx=start,
                scale=1.0,
                compute_kernel_config=self.correctness_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_chunk.deallocate(True)
            outputs.append(output)
        if owns_user_page_table:
            user_page_table.deallocate(True)
        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        return result

    def _sliding_chunked_prefill_attention(self, q_heads, k_heads, v_heads) -> ttnn.Tensor:
        """Run the baseline windowed long-prefill algorithm with TP-local heads."""
        seq_len = q_heads.shape[-2]
        history = ((self.layer_kind.sliding_window + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        outputs = []
        for start in range(0, seq_len, PREFILL_SLIDING_CHUNK_SIZE):
            output_len = min(PREFILL_SLIDING_CHUNK_SIZE, seq_len - start)
            slice_start = max(0, start - history)
            slice_end = start + output_len
            q_slice = ttnn.slice(
                q_heads, [0, 0, slice_start, 0], [1, LOCAL_Q_HEADS, slice_end, self.layer_kind.head_dim]
            )
            k_slice = ttnn.slice(
                k_heads, [0, 0, slice_start, 0], [1, k_heads.shape[1], slice_end, self.layer_kind.head_dim]
            )
            v_slice = ttnn.slice(
                v_heads, [0, 0, slice_start, 0], [1, v_heads.shape[1], slice_end, self.layer_kind.head_dim]
            )
            output = ttnn.transformer.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                is_causal=True,
                sliding_window_size=self.layer_kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_slice.deallocate(True)
            k_slice.deallocate(True)
            v_slice.deallocate(True)
            drop = start - slice_start
            if drop:
                full_output = output
                output = ttnn.slice(
                    full_output, [0, 0, drop, 0], [1, LOCAL_Q_HEADS, slice_end - slice_start, self.layer_kind.head_dim]
                )
                full_output.deallocate(True)
            outputs.append(output)
        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        return result

    def _attention_decode(self, x: ttnn.Tensor, **call: Any) -> ttnn.Tensor:
        self.optimized_path_counters["decode_attention"] += 1
        self.multichip_path_counters["attention_tp"] += 1
        kind = self.layer_kind
        batch = x.shape[-2]
        local_kv_heads = 1 if kind.name == "full_attention" else 2
        xqkv = ttnn.linear(
            x,
            self.weights.qkv,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_config,
        )
        head_mem = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=LOCAL_Q_HEADS, num_kv_heads=local_kv_heads, memory_config=head_mem
        )
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(
                q_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
            )
            k_heads = ttnn.experimental.rotary_embedding_hf(
                k_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
            )
            q_heads = ttnn.to_memory_config(ttnn.transpose(q_heads, 1, 2), head_mem, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(ttnn.transpose(k_heads, 1, 2), head_mem, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, head_mem, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, head_mem, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
            rope_mem = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            cos = ttnn.interleaved_to_sharded(call["position_cos"], rope_mem)
            sin = ttnn.interleaved_to_sharded(call["position_sin"], rope_mem)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, cos, sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, cos, sin, is_decode_mode=True)
        key_cache, value_cache = call["kv_cache"]
        update_kwargs = self._cache_view_kwargs(prefill=False)
        if call.get("cache_position_modulo") is not None:
            update_kwargs["cache_position_modulo"] = call["cache_position_modulo"]
        for cache, value in ((key_cache, k_heads), (value_cache, v_heads)):
            ttnn.experimental.paged_update_cache(
                cache,
                value,
                update_idxs_tensor=call["current_pos"],
                page_table=call["page_table"],
                **update_kwargs,
            )
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=call["page_table"],
            cur_pos_tensor=call["current_pos"],
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._cache_view_kwargs(prefill=False),
        )
        attn_out = ttnn.to_memory_config(attn_out, head_mem, dtype=attn_out.dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=LOCAL_Q_HEADS)
        attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
        partial = ttnn.linear(
            attn_out,
            self.weights.o_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_config,
        )
        result = self._all_reduce_hidden(partial)
        if result.shape[-2] != batch:
            result = ttnn.slice(result, [0, 0, 0, 0], [1, 1, batch, HIDDEN_SIZE])
        return result

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["dense_tp"] += 1
        return super()._dense_mlp(x)

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["expert_tp"] += 1
        return self._all_reduce_hidden(super()._moe_decode(hidden_states, routing_weights))

    def _moe_prefill(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["expert_tp"] += 1
        return self._all_reduce_hidden(super()._moe_prefill(hidden_states, routing_weights))


__all__ = ["MultichipDecoder"]
