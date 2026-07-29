# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized North-Mini decoder layer.

The functional decoder owns the public/cache semantics.  This subclass keeps
that contract while replacing material mathematical subgraphs.  In particular,
the dense SwiGLU input projections are packed into one device matmul and split
on device; runtime dispatch therefore reaches this class's optimized methods
through both prefill and decode.
"""

from __future__ import annotations

import math

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    DEFAULT_PAGE_SIZE,
    PREFILL_MOE_CHUNK,
    FunctionalDecoder,
    _as_device_tensor,
    _load_expert_weights,
    _require_tensor,
    _rope_output_permutation,
)


class OptimizedDecoder(FunctionalDecoder):
    """North-Mini decoder with phase-aware packed dense projections."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_cache_len=ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        dense_decode_variant="auto",
        dense_gate_up_in0_block_w=8,
        dense_down_in0_block_w=12,
        sparse_weight_dtype="auto",
        sparse_cores=11,
        sparse_gate_up_in0_block_w=32,
        sparse_down_in0_block_w=24,
        attention_weight_dtype="bfp8",
        attention_decode_variant="auto",
        sparse_compute_fidelity="lofi",
        sparse_gate_up_out_subblock_w=1,
        sparse_down_out_subblock_w=1,
        prefill_program_variant="auto",
        kv_cache_dtype="bf16",
        sdpa_decode_variant="default",
        attention_compute_fidelity="lofi",
        **kwargs,
    ):
        import torch

        decoder = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            page_size=page_size,
            **kwargs,
        )
        # Attention projections are bandwidth-bound at decode.  BFP8 retains
        # BF16 activations/cache while halving projection-weight traffic.
        q = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        q = q.index_select(0, _rope_output_permutation(decoder.num_heads, decoder.head_dim))
        k = k.index_select(0, _rope_output_permutation(decoder.num_kv_heads, decoder.head_dim))
        attention_dtype = ttnn.bfloat4_b if attention_weight_dtype == "bfp4" else ttnn.bfloat8_b
        qkv_host = torch.cat((q, k, v), dim=0).transpose(-2, -1).to(torch.bfloat16)
        o_host = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").transpose(-2, -1).to(torch.bfloat16)
        decoder.weights["qkv"] = _as_device_tensor(
            qkv_host,
            mesh_device=mesh_device,
            dtype=attention_dtype,
        )
        decoder.weights["o"] = _as_device_tensor(
            o_host,
            mesh_device=mesh_device,
            dtype=attention_dtype,
        )
        decoder.attention_decode_variant = (
            "dram_sharded" if attention_decode_variant == "auto" and batch == 1 else attention_decode_variant
        )
        decoder._init_advisor_attention_configs()
        decoder.dense_decode_variant = (
            "advisor_dram_sharded_bfp4_all"
            if dense_decode_variant == "auto" and batch == 1
            else "packed_interleaved"
            if dense_decode_variant == "auto"
            else dense_decode_variant
        )
        decoder.prefill_program_variant = prefill_program_variant
        decoder.kv_cache_dtype = kv_cache_dtype
        decoder.sdpa_decode_variant = sdpa_decode_variant
        decoder.attention_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=(
                ttnn.MathFidelity.HiFi2 if attention_compute_fidelity == "hifi2" else ttnn.MathFidelity.LoFi
            ),
            math_approx_mode=False,
            fp32_dest_acc_en=attention_compute_fidelity == "hifi2",
            packer_l1_acc=True,
        )
        dram_banks = mesh_device.dram_grid_size().x
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))})

        def dram_weight_memcfg(k, n):
            padded_n = math.ceil(n / (ttnn.TILE_SIZE * dram_banks)) * ttnn.TILE_SIZE * dram_banks
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (k, padded_n // dram_banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

        decoder.weights["qkv_dram_sharded"] = _as_device_tensor(
            qkv_host,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            memory_config=dram_weight_memcfg(decoder.hidden_size, 5120),
        )
        decoder.weights["o_dram_sharded"] = _as_device_tensor(
            o_host,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            memory_config=dram_weight_memcfg(4096, decoder.hidden_size),
        )
        if decoder.mlp_type == "dense":
            gate = _require_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
            up = _require_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
            packed = torch.cat((gate, up), dim=0).transpose(-2, -1).to(torch.bfloat16)
            decoder.weights["gate_up"] = _as_device_tensor(
                packed,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
            )
            decoder.weights["gate_proj_bfp8"] = _as_device_tensor(
                gate.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
            )
            decoder.weights["up_proj_bfp8"] = _as_device_tensor(
                up.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
            )
            decoder.weights["down_proj_bfp8"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
            )
            decoder.weights["gate_up_dram_sharded"] = _as_device_tensor(
                packed,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
                memory_config=dram_weight_memcfg(decoder.hidden_size, 2 * decoder.intermediate_size),
            )
            decoder.weights["down_dram_sharded"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
                memory_config=dram_weight_memcfg(decoder.intermediate_size, decoder.hidden_size),
            )
            decoder.weights["gate_up_dram_sharded_bfp4"] = _as_device_tensor(
                packed,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat4_b,
                memory_config=dram_weight_memcfg(decoder.hidden_size, 2 * decoder.intermediate_size),
            )
            decoder.weights["down_dram_sharded_bfp4"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat4_b,
                memory_config=dram_weight_memcfg(decoder.intermediate_size, decoder.hidden_size),
            )
            decoder.weights["gate_dram_sharded_bfp4"] = _as_device_tensor(
                gate.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat4_b,
                memory_config=dram_weight_memcfg(decoder.hidden_size, decoder.intermediate_size),
            )
            decoder.weights["up_dram_sharded_bfp4"] = _as_device_tensor(
                up.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat4_b,
                memory_config=dram_weight_memcfg(decoder.hidden_size, decoder.intermediate_size),
            )
            decoder._init_dense_decode_configs()
            decoder.dense_gate_up_dram_program.in0_block_w = dense_gate_up_in0_block_w
            decoder.dense_down_dram_program.in0_block_w = dense_down_in0_block_w
        else:
            import torch

            gate, up, down = _load_expert_weights(
                state_dict,
                layer_idx,
                decoder.num_experts,
                decoder.intermediate_size,
            )
            decoder.weights["expert_gate_sparse"] = _as_device_tensor(
                gate.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat4_b
            )
            decoder.weights["expert_up_sparse"] = _as_device_tensor(
                up.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat4_b
            )
            decoder.weights["expert_down_sparse"] = _as_device_tensor(
                down.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat4_b
            )
            decoder.weights["expert_gate_sparse_bfp8"] = _as_device_tensor(
                gate.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat8_b
            )
            decoder.weights["expert_up_sparse_bfp8"] = _as_device_tensor(
                up.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat8_b
            )
            decoder.weights["expert_down_sparse_bfp8"] = _as_device_tensor(
                down.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat8_b
            )
            decoder.weights["expert_gate_sparse_bf16"] = _as_device_tensor(
                gate.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat16
            )
            decoder.weights["expert_up_sparse_bf16"] = _as_device_tensor(
                up.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat16
            )
            decoder.weights["expert_down_sparse_bf16"] = _as_device_tensor(
                down.unsqueeze(0), mesh_device=mesh_device, dtype=ttnn.bfloat16
            )
            decoder.weights["router_bfp8"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.gate.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
            )
            decoder._init_sparse_decode_configs(
                cores=sparse_cores,
                gate_up_in0_block_w=sparse_gate_up_in0_block_w,
                down_in0_block_w=sparse_down_in0_block_w,
                gate_up_out_subblock_w=sparse_gate_up_out_subblock_w,
                down_out_subblock_w=sparse_down_out_subblock_w,
            )
            decoder.sparse_weight_dtype = (
                "bfp4"
                if sparse_weight_dtype == "auto" and batch == 1
                else "bfp8"
                if sparse_weight_dtype == "auto"
                else sparse_weight_dtype
            )
            decoder.sparse_decode_compute = (
                decoder.dense_hifi2 if sparse_compute_fidelity == "hifi2" else decoder.dense_lofi
            )
        return decoder

    def create_paged_kv_cache(self, *, num_blocks=None):
        if self.kv_cache_dtype == "bf16":
            return super().create_paged_kv_cache(num_blocks=num_blocks)
        min_blocks = self.batch * math.ceil(self.max_cache_len / self.page_size)
        num_blocks = min_blocks if num_blocks is None else int(num_blocks)
        if num_blocks < min_blocks:
            raise ValueError(f"num_blocks={num_blocks} cannot cover required {min_blocks} blocks")
        shape = (num_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        dtype = ttnn.bfloat8_b if self.kv_cache_dtype == "bfp8" else ttnn.bfloat4_b
        return (
            ttnn.zeros(
                shape,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.zeros(
                shape,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    def _width_sharded_memcfg(self, *, width, cores, height=ttnn.TILE_SIZE):
        storage_grid = self.mesh_device.compute_with_storage_grid_size()
        core_grid = ttnn.num_cores_to_corerangeset(cores, storage_grid, row_wise=True)
        shard_width = math.ceil(width / (ttnn.TILE_SIZE * cores)) * ttnn.TILE_SIZE
        return ttnn.create_sharded_memory_config(
            shape=(height, shard_width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _init_advisor_attention_configs(self):
        self.advisor_qkv_input_memcfg = self._width_sharded_memcfg(width=2048, cores=32)
        self.advisor_qkv_output_memcfg = self._width_sharded_memcfg(width=5120, cores=80)
        self.advisor_o_output_memcfg = self._width_sharded_memcfg(width=2048, cores=64)
        self.advisor_qkv_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=2,
            out_block_h=1,
            out_block_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.advisor_o_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 6),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.attention_qkv_ds_input_memcfg = self._width_sharded_memcfg(width=2048, cores=8)
        self.attention_qkv_ds_output_memcfg = self._width_sharded_memcfg(width=5120, cores=8)
        self.attention_o_ds_input_memcfg = self._width_sharded_memcfg(width=4096, cores=8)
        self.attention_o_ds_output_memcfg = self._width_sharded_memcfg(width=2048, cores=8)
        self.attention_qkv_ds_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=8, per_core_M=1, per_core_N=20, fused_activation=None
        )
        self.attention_o_ds_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=16, per_core_M=1, per_core_N=8, fused_activation=None
        )

    def _qkv_decode(self, normalized, position_cos, position_sin):
        if self.attention_decode_variant not in ("advisor_1d", "dram_sharded") or self.batch != 1:
            return super()._qkv_decode(normalized, position_cos, position_sin)
        is_ds = self.attention_decode_variant == "dram_sharded"
        working = ttnn.to_memory_config(
            normalized,
            self.attention_qkv_ds_input_memcfg if is_ds else self.advisor_qkv_input_memcfg,
        )
        fused = ttnn.linear(
            working,
            self.weights["qkv_dram_sharded" if is_ds else "qkv"],
            dtype=ttnn.bfloat16,
            memory_config=(self.attention_qkv_ds_output_memcfg if is_ds else self.advisor_qkv_output_memcfg),
            program_config=self.attention_qkv_ds_program if is_ds else self.advisor_qkv_program,
            compute_kernel_config=self.attention_compute,
        )
        fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.reshape(fused, (1, 1, self.batch, -1))
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        if self.use_rope:
            position_cos = ttnn.interleaved_to_sharded(position_cos, self.decode_rope_memory_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, self.decode_rope_memory_config)
            query = ttnn.experimental.rotary_embedding_hf(
                query,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            key = ttnn.experimental.rotary_embedding_hf(
                key,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
        return query, key, value

    def _attention_decode(self, normalized, **kwargs):
        if self.attention_decode_variant not in ("advisor_1d", "dram_sharded") or self.batch != 1:
            return super()._attention_decode(normalized, **kwargs)
        key_cache = kwargs["key_cache"]
        value_cache = kwargs["value_cache"]
        page_table = kwargs["page_table"]
        current_positions = kwargs["current_positions"]
        query, key, value = self._qkv_decode(normalized, kwargs["position_cos"], kwargs["position_sin"])
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
        )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_positions,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=(
                None
                if self.sdpa_decode_variant == "default"
                else ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    q_chunk_size=32,
                    k_chunk_size=int(self.sdpa_decode_variant[1:]),
                    exp_approx_mode=False,
                )
            ),
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended, num_heads=self.num_heads, sub_core_grids=self.decode_sub_core_grids
        )
        attended = ttnn.to_memory_config(attended, ttnn.DRAM_MEMORY_CONFIG)
        is_ds = self.attention_decode_variant == "dram_sharded"
        if is_ds:
            attended = ttnn.to_memory_config(attended, self.attention_o_ds_input_memcfg)
        projected = ttnn.linear(
            attended,
            self.weights["o_dram_sharded" if is_ds else "o"],
            dtype=ttnn.bfloat16,
            memory_config=(self.attention_o_ds_output_memcfg if is_ds else self.advisor_o_output_memcfg),
            program_config=self.attention_o_ds_program if is_ds else self.advisor_o_program,
            compute_kernel_config=self.attention_compute,
        )
        projected = ttnn.to_memory_config(projected, ttnn.L1_MEMORY_CONFIG)
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.permute(projected, (0, 2, 1, 3))

    def _attention_prefill(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos,
        position_sin,
        seq_len,
    ):
        query, key, value = self._qkv_prefill(normalized, seq_len, position_cos, position_sin)
        for user in range(self.batch):
            key_user = ttnn.slice(key, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            value_user = ttnn.slice(
                value,
                (user, 0, 0, 0),
                (user + 1, self.num_kv_heads, seq_len, self.head_dim),
            )
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._large_prefill_program(attended, self.hidden_size, in0_block_w=8),
        )

    def _init_dense_decode_configs(self):
        # Exact batch-1 candidates emitted by shard-advise, generalized to the
        # real physical M height for larger decode batches. North's public
        # [1,batch,1,K] shape has batch tile faces, so batch 32 has 32 M tiles.
        physical_m = self.batch * ttnn.TILE_SIZE
        # The DRAM-sharded matmul contract sees logical M=1 even though the
        # [1,batch,1,K] tensor's tiled physical height is batch*32.
        per_core_m = 1
        self.dense_gate_up_input_memcfg = self._width_sharded_memcfg(width=self.hidden_size, cores=8, height=physical_m)
        self.dense_gate_up_output_memcfg = self._width_sharded_memcfg(
            width=2 * self.intermediate_size, cores=96, height=physical_m
        )
        self.dense_down_input_memcfg = self._width_sharded_memcfg(
            width=self.intermediate_size, cores=8, height=physical_m
        )
        self.dense_down_output_memcfg = self._width_sharded_memcfg(width=self.hidden_size, cores=64, height=physical_m)
        program_type = (
            ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
            if self.batch == 1
            else ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig
        )
        self.dense_gate_up_dram_program = program_type(
            in0_block_w=8,
            per_core_M=per_core_m,
            per_core_N=2,
            fused_activation=None,
        )
        self.dense_down_dram_program = program_type(
            in0_block_w=12,
            per_core_M=per_core_m,
            per_core_N=1,
            fused_activation=None,
        )
        self.dense_separate_gate_up_output_memcfg = self._width_sharded_memcfg(
            width=self.intermediate_size, cores=8, height=physical_m
        )
        self.dense_separate_gate_up_program = program_type(
            in0_block_w=8,
            per_core_M=per_core_m,
            per_core_N=12,
            fused_activation=None,
        )
        self.dense_hifi2 = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.dense_lofi = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _sparse_matmul_program(self, *, n, k, cores, in0_block_w, out_subblock_w):
        out_block_w = 1
        if out_subblock_w <= 0 or out_subblock_w > out_block_w or out_block_w % out_subblock_w != 0:
            raise ValueError(
                f"sparse out_subblock_w ({out_subblock_w}) must be positive, no greater than, "
                f"and divide out_block_w ({out_block_w})"
            )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cores, 1),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=out_block_w,
            per_core_M=1,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * cores)),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def _init_sparse_decode_configs(
        self,
        *,
        cores=11,
        gate_up_in0_block_w=32,
        down_in0_block_w=24,
        gate_up_out_subblock_w=1,
        down_out_subblock_w=1,
    ):
        self.dense_hifi2 = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.dense_lofi = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.sparse_gate_up_program = self._sparse_matmul_program(
            n=self.intermediate_size,
            k=self.hidden_size,
            cores=cores,
            in0_block_w=gate_up_in0_block_w,
            out_subblock_w=gate_up_out_subblock_w,
        )
        self.sparse_down_program = self._sparse_matmul_program(
            n=self.hidden_size,
            k=self.intermediate_size,
            cores=cores,
            in0_block_w=down_in0_block_w,
            out_subblock_w=down_out_subblock_w,
        )

    def _dense_mlp(self, normalized):
        if normalized.shape[2] == 1 and self.dense_decode_variant == "separate_dram_sharded_bfp4":
            return self._dense_mlp_decode_separate_dram_sharded(normalized)
        if normalized.shape[2] == 1 and self.dense_decode_variant.startswith("advisor_dram_sharded"):
            return self._dense_mlp_decode_dram_sharded(normalized)
        if normalized.shape[2] > 1:
            # The generic split kernel currently fails to compile for this
            # large BFP8 packed output on Blackhole prefill.  Separate BFP8
            # projections avoid that compiler defect while still halving
            # weight movement versus the functional BF16 path.
            program_gate_up = self._large_prefill_program(normalized, self.intermediate_size, in0_block_w=8)
            gate = ttnn.linear(
                normalized,
                self.weights["gate_proj_bfp8"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=program_gate_up,
            )
            up = ttnn.linear(
                normalized,
                self.weights["up_proj_bfp8"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=program_gate_up,
            )
            activated = ttnn.multiply(ttnn.silu(gate), up)
            program_down = self._large_prefill_program(activated, self.hidden_size, in0_block_w=12)
            return ttnn.linear(
                activated,
                self.weights["down_proj_bfp8"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=program_down,
            )
        packed = ttnn.linear(
            normalized,
            self.weights["gate_up"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate, up = ttnn.split(packed, self.intermediate_size, dim=-1)
        activated = ttnn.multiply(ttnn.silu(gate), up)
        return ttnn.linear(
            activated,
            self.weights["down_proj_bfp8"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _dense_mlp_decode_separate_dram_sharded(self, normalized):
        working = ttnn.to_memory_config(normalized, self.dense_gate_up_input_memcfg)
        gate = ttnn.linear(
            working,
            self.weights["gate_dram_sharded_bfp4"],
            dtype=ttnn.bfloat16,
            memory_config=self.dense_separate_gate_up_output_memcfg,
            program_config=self.dense_separate_gate_up_program,
            compute_kernel_config=self.dense_lofi,
        )
        up = ttnn.linear(
            working,
            self.weights["up_dram_sharded_bfp4"],
            dtype=ttnn.bfloat16,
            memory_config=self.dense_separate_gate_up_output_memcfg,
            program_config=self.dense_separate_gate_up_program,
            compute_kernel_config=self.dense_lofi,
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        activated = ttnn.to_memory_config(activated, self.dense_down_input_memcfg)
        return ttnn.linear(
            activated,
            self.weights["down_dram_sharded_bfp4"],
            dtype=ttnn.bfloat16,
            memory_config=self.dense_down_output_memcfg,
            program_config=self.dense_down_dram_program,
            compute_kernel_config=self.dense_lofi,
        )

    def _large_prefill_program(self, activation, output_width, *, in0_block_w):
        activation_shape = tuple(activation.shape)
        return self._large_prefill_program_for_m(
            math.prod(activation_shape[:-1]), output_width, in0_block_w=in0_block_w
        )

    def _large_prefill_program_for_m(self, m, output_width, *, in0_block_w):
        if self.prefill_program_variant == "default":
            return None
        m_tiles = math.ceil(m / ttnn.TILE_SIZE)
        n_tiles = math.ceil(output_width / ttnn.TILE_SIZE)
        grid_x = 8
        grid_y = min(10, m_tiles)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=math.ceil(m_tiles / grid_y),
            per_core_N=math.ceil(n_tiles / grid_x),
            transpose_mcast=False,
            fused_activation=None,
        )

    def _dense_mlp_decode_dram_sharded(self, normalized):
        gate_up_bfp4 = self.dense_decode_variant in (
            "advisor_dram_sharded_bfp4_gate_up",
            "advisor_dram_sharded_bfp4_all",
        )
        down_bfp4 = self.dense_decode_variant == "advisor_dram_sharded_bfp4_all"
        working = ttnn.to_memory_config(normalized, self.dense_gate_up_input_memcfg)
        packed = ttnn.linear(
            working,
            self.weights["gate_up_dram_sharded_bfp4" if gate_up_bfp4 else "gate_up_dram_sharded"],
            dtype=ttnn.bfloat16,
            memory_config=self.dense_gate_up_output_memcfg,
            program_config=self.dense_gate_up_dram_program,
            compute_kernel_config=self.dense_lofi if gate_up_bfp4 else self.dense_hifi2,
        )
        packed = ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG)
        gate, up = ttnn.split(packed, self.intermediate_size, dim=-1)
        activated = ttnn.multiply(ttnn.silu(gate), up)
        activated = ttnn.to_memory_config(activated, self.dense_down_input_memcfg)
        return ttnn.linear(
            activated,
            self.weights["down_dram_sharded_bfp4" if down_bfp4 else "down_dram_sharded"],
            dtype=ttnn.bfloat16,
            memory_config=self.dense_down_output_memcfg,
            program_config=self.dense_down_dram_program,
            compute_kernel_config=self.dense_lofi if down_bfp4 else self.dense_hifi2,
        )

    def _sparse_moe(self, normalized, seq_len):
        total_tokens = self.batch * seq_len
        flat = ttnn.reshape(normalized, (1, 1, total_tokens, self.hidden_size))
        if total_tokens == 1:
            return ttnn.reshape(
                self._sparse_moe_decode(flat, token_count=1),
                (1, self.batch, seq_len, self.hidden_size),
            )
        # Grouped sparse_matmul was adapted through DRAM outputs, 32-token
        # chunking, BF16/HiFi2, and token-specific routing. It remains slower
        # than batched expert matmul because sparse-down cannot represent a
        # token-specific [T,E] sparse-A mask. Retain the measured device-only
        # batched path for grouped tokens.
        dense_group = PREFILL_MOE_CHUNK
        if total_tokens <= dense_group:
            result = self._sparse_moe_chunk(flat, total_tokens)
        else:
            chunks = ttnn.split(flat, dense_group, dim=2)
            outputs = [self._sparse_moe_chunk(chunk, chunk.shape[2]) for chunk in chunks]
            result = ttnn.concat(outputs, dim=0)
        return ttnn.reshape(result, (1, self.batch, seq_len, self.hidden_size))

    def _sparse_moe_chunk(self, normalized, token_count):
        """Device-resident broadcast expert path for prefill/serving decode.

        The batch-1 sparse kernels avoid all inactive expert compute.  For
        larger token groups their full expert output is too large for L1, so
        this path uses BF16 expert weights and TTNN batched matmuls in DRAM.
        It remains wholly device resident and does not call the functional
        decoder implementation.
        """
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        expert_input = ttnn.reshape(flat, (1, token_count, self.hidden_size))
        expert_input = ttnn.repeat(expert_input, ttnn.Shape((self.num_experts, 1, 1)))
        # Synthetic and real prefill sweeps show that quantizing all expert
        # weights drops token-wise routing PCC below the functional floor.
        # Retain BF16 experts for grouped tokens; batch-1 decode uses the
        # separately validated routed BFP8 sparse kernels.
        gate_weight = self.weights["expert_gate"]
        up_weight = self.weights["expert_up"]
        down_weight = self.weights["expert_down"]
        expert_gate_up_program = (
            self._large_prefill_program_for_m(token_count * self.num_experts, self.intermediate_size, in0_block_w=8)
            if self.prefill_program_variant == "expert_2d"
            else None
        )
        gate = ttnn.matmul(
            expert_input,
            gate_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=expert_gate_up_program,
        )
        up = ttnn.matmul(
            expert_input,
            up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=expert_gate_up_program,
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        expert_down_program = (
            self._large_prefill_program_for_m(token_count * self.num_experts, self.hidden_size, in0_block_w=12)
            if self.prefill_program_variant == "expert_2d"
            else None
        )
        expert_output = ttnn.matmul(
            activated,
            down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=expert_down_program,
        )
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.num_experts, token_count, 1))
        expert_output = ttnn.multiply(expert_output, routing)
        return ttnn.sum(expert_output, dim=0)

    def _sparse_moe_decode(self, normalized, token_count=None, grouped=False):
        token_count = self.batch if token_count is None else token_count
        sparse_suffix = "_bf16" if grouped else "" if self.sparse_weight_dtype == "bfp4" else "_bfp8"
        # A batch-32 expert intermediate is ~192 MiB (32 tokens x 32
        # experts x 3072 channels x BF16), so it cannot reside in aggregate
        # worker L1.  Keep the low-latency batch-1 path in L1 and use DRAM for
        # the serving-batch intermediates without changing the routed kernel.
        expert_memory_config = ttnn.L1_MEMORY_CONFIG if token_count == 1 else ttnn.DRAM_MEMORY_CONFIG
        sparse_compute = self.dense_hifi2 if grouped else self.sparse_decode_compute
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router" if grouped else "router_bfp8"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing), ttnn.ROW_MAJOR_LAYOUT)
        hidden = ttnn.reshape(normalized, (1, token_count, 1, self.hidden_size))
        output_tile = ttnn.Tile([32, 32])
        gate = ttnn.sparse_matmul(
            hidden,
            self.weights["expert_gate_sparse" + sparse_suffix],
            sparsity=sparsity,
            nnz=None,
            memory_config=expert_memory_config,
            output_tile=output_tile,
            program_config=self.sparse_gate_up_program,
            compute_kernel_config=sparse_compute,
            dtype=ttnn.bfloat16,
        )
        up = ttnn.sparse_matmul(
            hidden,
            self.weights["expert_up_sparse" + sparse_suffix],
            sparsity=sparsity,
            nnz=None,
            memory_config=expert_memory_config,
            output_tile=output_tile,
            program_config=self.sparse_gate_up_program,
            compute_kernel_config=sparse_compute,
            dtype=ttnn.bfloat16,
        )
        gate = ttnn.reshape(gate, (token_count, self.num_experts, 1, self.intermediate_size))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (token_count, self.num_experts, self.intermediate_size))
        up = ttnn.reshape(up, (token_count, self.num_experts, 1, self.intermediate_size))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (token_count, self.num_experts, self.intermediate_size))
        activated = ttnn.multiply(ttnn.silu(gate), up)
        if grouped:
            # sparse_matmul's sparse-A contract models only expert batch
            # dimensions: a token-specific [T,E] sparsity tensor is rejected
            # because it requires volume T*E while the kernel requires E.
            # Gate/up (the two dominant projections) remain token-routed; use
            # one BF16 batched down projection after inactive activations have
            # already been zeroed by those sparse kernels.
            activated = ttnn.transpose(activated, 1, 0)
            down = ttnn.matmul(
                activated,
                self.weights["expert_down"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            down = ttnn.transpose(down, 1, 0)
        else:
            activated = ttnn.reshape(activated, (1, self.num_experts, token_count, self.intermediate_size))
            down = ttnn.sparse_matmul(
                activated,
                self.weights["expert_down_sparse" + sparse_suffix],
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=True,
                memory_config=expert_memory_config,
                output_tile=output_tile,
                program_config=self.sparse_down_program,
                compute_kernel_config=sparse_compute,
                dtype=ttnn.bfloat16,
            )
            down = ttnn.permute(down, (0, 2, 1, 3))
            down = ttnn.reshape(down, (token_count, self.num_experts, self.hidden_size))
        routing = ttnn.reshape(routing, (token_count, self.num_experts, 1))
        down = ttnn.multiply(down, routing)
        reduced = ttnn.sum(down, dim=1)
        return ttnn.reshape(reduced, (token_count, self.hidden_size))
