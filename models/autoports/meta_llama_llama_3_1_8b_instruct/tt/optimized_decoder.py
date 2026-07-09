# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    LlamaDecoderConfig,
    _lookup,
    _require_config,
)
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import get_rot_transformation_mat

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT
BF16 = ttnn.bfloat16
BFP8 = ttnn.bfloat8_b
BFP4 = ttnn.bfloat4_b


def _wormhole_hifi2() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _wormhole_lofi() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _wormhole_hifi4() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _core_coord(mesh_device, requested_x: int = 8, requested_y: int = 8) -> ttnn.CoreCoord:
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(int(grid.x), requested_x), min(int(grid.y), requested_y))


def _height_sharded_memcfg(mesh_device, logical_rows: int, logical_width: int) -> ttnn.MemoryConfig:
    grid = _core_coord(mesh_device)
    core_count = min(max(1, logical_rows), int(grid.x) * int(grid.y))
    core_grid = ttnn.num_cores_to_corerangeset(core_count, grid, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(max(32, logical_rows), logical_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_memcfg(mesh_device, logical_rows: int, logical_width: int) -> ttnn.MemoryConfig:
    grid = _core_coord(mesh_device)
    max_cores = int(grid.x) * int(grid.y)
    width_tiles = math.ceil(logical_width / 32)
    core_count = min(max_cores, max(1, width_tiles))
    while width_tiles % core_count != 0 and core_count > 1:
        core_count -= 1
    core_grid = ttnn.num_cores_to_corerangeset(core_count, grid, row_wise=True)
    shard_width = math.ceil(logical_width / core_count)
    shard_width = math.ceil(shard_width / 32) * 32
    return ttnn.create_sharded_memory_config(
        shape=(max(32, logical_rows), shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _matmul_1d_config(mesh_device, out_width: int, *, in0_block_w: int, fused_activation=None):
    grid = _core_coord(mesh_device)
    core_count = int(grid.x) * int(grid.y)
    out_tiles = math.ceil(out_width / 32)
    active_cores = min(core_count, out_tiles)
    while out_tiles % active_cores != 0 and active_cores > 1:
        active_cores -= 1
    per_core_n = max(1, out_tiles // active_cores)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_n,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


@dataclass(frozen=True)
class OptimizedDecoderPolicy:
    activation_dtype: ttnn.DataType = BF16
    attention_weight_dtype: ttnn.DataType = BFP8
    mlp_weight_dtype: ttnn.DataType = BFP4
    cache_dtype: ttnn.DataType = BFP8
    norm_compute_kernel_config: ttnn.WormholeComputeKernelConfig = _wormhole_hifi4()
    matmul_compute_kernel_config: ttnn.WormholeComputeKernelConfig = _wormhole_hifi2()
    mlp_compute_kernel_config: ttnn.WormholeComputeKernelConfig = _wormhole_lofi()
    sdpa_compute_kernel_config: ttnn.WormholeComputeKernelConfig = _wormhole_hifi2()


class OptimizedDecoder(LightweightModule):
    """Single-device optimized Llama-3.1 decoder layer.

    This is intentionally independent of ``FunctionalDecoder`` so tests cannot
    pass by falling back to the functional prefill implementation. Runtime uses
    packed QKV, TTNN head creation/concat helpers, paged KV cache ops, SDPA
    decode, explicit compute configs, and no torch host fallback in forward.
    """

    def __init__(
        self,
        *,
        config: LlamaDecoderConfig,
        layer_idx: int,
        mesh_device,
        weights: dict[str, ttnn.Tensor],
        batch: int,
        page_block_size: int = 32,
        max_num_blocks: int = 128,
        policy: OptimizedDecoderPolicy | None = None,
    ):
        self.config = config
        self.layer_idx = int(layer_idx)
        self.mesh_device = mesh_device
        self.weights = weights
        self.batch = int(batch)
        self.page_block_size = int(page_block_size)
        self.max_num_blocks = int(max_num_blocks)
        self.policy = policy or OptimizedDecoderPolicy()
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.decode_qkv_memcfg = _width_sharded_memcfg(
            mesh_device,
            self.batch,
            config.num_attention_heads * config.head_dim + 2 * config.num_key_value_heads * config.head_dim,
        )
        self.decode_head_memcfg = _height_sharded_memcfg(mesh_device, self.batch, config.head_dim)
        self.decode_rope_memcfg = _height_sharded_memcfg(mesh_device, self.batch, config.head_dim)
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_core_coord(mesh_device),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        self.decode_mlp_gate_program_config = _matmul_1d_config(
            mesh_device,
            config.intermediate_size,
            in0_block_w=32,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        self.decode_mlp_up_program_config = _matmul_1d_config(
            mesh_device,
            config.intermediate_size,
            in0_block_w=32,
        )
        self.decode_mlp_down_program_config = _matmul_1d_config(
            mesh_device,
            config.hidden_size,
            in0_block_w=32,
        )
        batch_grid = ttnn.num_cores_to_corerangeset(self.batch, _core_coord(mesh_device), row_wise=True)
        trans_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat().repeat(1, 1, self.batch, 1),
            dtype=BF16,
            layout=TILE,
            device=mesh_device,
            memory_config=trans_memcfg,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=32,
        page_block_size=32,
        max_num_blocks=128,
        policy: OptimizedDecoderPolicy | None = None,
        **kwargs,
    ):
        del kwargs
        config = _require_config(hf_config)
        policy = policy or OptimizedDecoderPolicy()

        q_proj = _lookup(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_proj = _lookup(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_proj = _lookup(state_dict, layer_idx, "self_attn.v_proj.weight")

        weights = {
            "input_layernorm": cls._to_tt_weight(
                _lookup(state_dict, layer_idx, "input_layernorm.weight"), mesh_device, BF16
            ),
            "post_attention_layernorm": cls._to_tt_weight(
                _lookup(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device, BF16
            ),
            "qkv": cls._to_tt_weight(
                torch.cat([q_proj.T, k_proj.T, v_proj.T], dim=1), mesh_device, policy.attention_weight_dtype
            ),
            "o_proj": cls._to_tt_weight(
                _lookup(state_dict, layer_idx, "self_attn.o_proj.weight").T,
                mesh_device,
                policy.attention_weight_dtype,
            ),
            "gate_proj": cls._to_tt_weight(
                _lookup(state_dict, layer_idx, "mlp.gate_proj.weight").T, mesh_device, policy.mlp_weight_dtype
            ),
            "up_proj": cls._to_tt_weight(
                _lookup(state_dict, layer_idx, "mlp.up_proj.weight").T, mesh_device, policy.mlp_weight_dtype
            ),
            "down_proj": cls._to_tt_weight(
                _lookup(state_dict, layer_idx, "mlp.down_proj.weight").T, mesh_device, policy.mlp_weight_dtype
            ),
        }
        return cls(
            config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weights=weights,
            batch=batch,
            page_block_size=page_block_size,
            max_num_blocks=max_num_blocks,
            policy=policy,
        )

    @staticmethod
    def _to_tt_weight(tensor: torch.Tensor, mesh_device, dtype: ttnn.DataType) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.detach().to(torch.bfloat16),
            dtype=dtype,
            layout=TILE,
            device=mesh_device,
            memory_config=DRAM,
        )

    @staticmethod
    def prepare_inputs(hidden_states: torch.Tensor, mesh_device) -> ttnn.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        return ttnn.from_torch(
            hidden_states.detach().to(torch.bfloat16).unsqueeze(0),
            dtype=BF16,
            layout=TILE,
            device=mesh_device,
            memory_config=DRAM,
        )

    @staticmethod
    def prepare_decode_inputs(hidden_states: torch.Tensor, mesh_device) -> ttnn.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(f"decode hidden_states must be [batch, 1, hidden], got {tuple(hidden_states.shape)}")
        return ttnn.from_torch(
            hidden_states.detach().to(torch.bfloat16).reshape(1, 1, hidden_states.shape[0], hidden_states.shape[2]),
            dtype=BF16,
            layout=TILE,
            device=mesh_device,
            memory_config=DRAM,
        )

    @staticmethod
    def prepare_rope(
        position_cos: torch.Tensor, position_sin: torch.Tensor, mesh_device
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position_cos.ndim == 3:
            position_cos = position_cos[:, None, :, :]
        if position_sin.ndim == 3:
            position_sin = position_sin[:, None, :, :]
        if position_cos.shape[0] != 1:
            position_cos = position_cos[:1]
            position_sin = position_sin[:1]
        return (
            ttnn.from_torch(position_cos.detach().to(torch.bfloat16), dtype=BF16, layout=TILE, device=mesh_device),
            ttnn.from_torch(position_sin.detach().to(torch.bfloat16), dtype=BF16, layout=TILE, device=mesh_device),
        )

    def prepare_decode_rope(
        self, position_cos: torch.Tensor, position_sin: torch.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position_cos.ndim != 3 or position_cos.shape[1] != 1:
            raise ValueError("decode position_cos must be [batch, 1, head_dim]")
        cos = position_cos.detach().to(torch.bfloat16).reshape(1, self.batch, 1, self.config.head_dim)
        sin = position_sin.detach().to(torch.bfloat16).reshape(1, self.batch, 1, self.config.head_dim)
        return (
            ttnn.from_torch(
                cos, dtype=BF16, layout=TILE, device=self.mesh_device, memory_config=self.decode_rope_memcfg
            ),
            ttnn.from_torch(
                sin, dtype=BF16, layout=TILE, device=self.mesh_device, memory_config=self.decode_rope_memcfg
            ),
        )

    def prepare_current_pos(self, current_pos: torch.Tensor) -> ttnn.Tensor:
        if current_pos.ndim != 1 or current_pos.shape[0] != self.batch:
            raise ValueError(f"current_pos must be [{self.batch}], got {tuple(current_pos.shape)}")
        return ttnn.from_torch(
            current_pos.detach().to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
        )

    def prepare_page_table(self, page_table: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            page_table.detach().to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
        )

    def allocate_paged_kv_cache(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        shape = (self.max_num_blocks, self.config.num_key_value_heads, self.page_block_size, self.config.head_dim)
        zeros = torch.zeros(shape, dtype=torch.bfloat16)
        key_cache = ttnn.from_torch(
            zeros,
            dtype=self.policy.cache_dtype,
            layout=TILE,
            device=self.mesh_device,
            memory_config=DRAM,
        )
        value_cache = ttnn.from_torch(
            zeros,
            dtype=self.policy.cache_dtype,
            layout=TILE,
            device=self.mesh_device,
            memory_config=DRAM,
        )
        return key_cache, value_cache

    def _attention_prefill(
        self, normed, batch: int, seq_len: int, position_cos, position_sin, attn_mask, kv_cache=None, page_table=None
    ):
        cfg = self.config
        qkv = ttnn.linear(
            normed,
            self.weights["qkv"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            compute_kernel_config=self.policy.matmul_compute_kernel_config,
        )
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        q_width = cfg.num_attention_heads * cfg.head_dim
        query = ttnn.slice(qkv, [0, 0, 0, 0], [1, batch, seq_len, q_width], [1, 1, 1, 1], memory_config=DRAM)
        key = ttnn.slice(
            qkv,
            [0, 0, 0, q_width],
            [1, batch, seq_len, q_width + kv_width],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )
        value = ttnn.slice(
            qkv,
            [0, 0, 0, q_width + kv_width],
            [1, batch, seq_len, q_width + kv_width + kv_width],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )
        query = ttnn.reshape(query, [batch, seq_len, cfg.num_attention_heads, cfg.head_dim])
        key = ttnn.reshape(key, [batch, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        value = ttnn.reshape(value, [batch, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        query = ttnn.permute(query, [0, 2, 1, 3], memory_config=DRAM)
        key = ttnn.permute(key, [0, 2, 1, 3], memory_config=DRAM)
        value = ttnn.permute(value, [0, 2, 1, 3], memory_config=DRAM)
        query = ttnn.experimental.rotary_embedding(query, position_cos, position_sin, None, memory_config=DRAM)
        key = ttnn.experimental.rotary_embedding(key, position_cos, position_sin, None, memory_config=DRAM)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [batch, cfg.num_attention_heads, seq_len, cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )
        key = ttnn.slice(
            key, [0, 0, 0, 0], [batch, cfg.num_key_value_heads, seq_len, cfg.head_dim], [1, 1, 1, 1], memory_config=DRAM
        )

        if kv_cache is not None and page_table is not None:
            key_cache, value_cache = kv_cache
            key_fill = ttnn.typecast(key, dtype=key_cache.dtype) if key.dtype != key_cache.dtype else key
            value_fill = ttnn.typecast(value, dtype=value_cache.dtype) if value.dtype != value_cache.dtype else value
            for batch_idx in range(batch):
                ttnn.experimental.paged_fill_cache(key_cache, key_fill, page_table, batch_idx=batch_idx)
                ttnn.experimental.paged_fill_cache(value_cache, value_fill, page_table, batch_idx=batch_idx)

        if attn_mask is None:
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=self.scale,
                memory_config=DRAM,
                compute_kernel_config=self.policy.sdpa_compute_kernel_config,
            )
        else:
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self.scale,
                memory_config=DRAM,
                compute_kernel_config=self.policy.sdpa_compute_kernel_config,
            )
        attn = ttnn.transformer.concatenate_heads(attn, memory_config=DRAM)
        return ttnn.linear(
            attn,
            self.weights["o_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            compute_kernel_config=self.policy.matmul_compute_kernel_config,
        )

    def prefill_forward(
        self, hidden_states, *, position_cos, position_sin, attn_mask=None, kv_cache=None, page_table=None
    ):
        cfg = self.config
        batch = hidden_states.shape[1]
        seq_len = hidden_states.shape[2]
        if hidden_states.shape[0] != 1 or hidden_states.shape[3] != cfg.hidden_size:
            raise ValueError("hidden_states must have shape [1, batch, seq, hidden_size]")
        if batch != self.batch:
            raise ValueError(f"runtime batch {batch} does not match configured batch {self.batch}")

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["input_layernorm"],
            memory_config=DRAM,
            compute_kernel_config=self.policy.norm_compute_kernel_config,
        )
        attn = self._attention_prefill(
            normed, batch, seq_len, position_cos, position_sin, attn_mask, kv_cache, page_table
        )
        hidden = ttnn.add(attn, residual, dtype=BF16, memory_config=DRAM)

        mlp_input = ttnn.rms_norm(
            hidden,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["post_attention_layernorm"],
            memory_config=DRAM,
            compute_kernel_config=self.policy.norm_compute_kernel_config,
        )
        gate = ttnn.linear(
            mlp_input,
            self.weights["gate_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            activation="silu",
            compute_kernel_config=self.policy.mlp_compute_kernel_config,
        )
        up = ttnn.linear(
            mlp_input,
            self.weights["up_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            compute_kernel_config=self.policy.mlp_compute_kernel_config,
        )
        gated = ttnn.multiply(gate, up, memory_config=DRAM)
        mlp = ttnn.linear(
            gated,
            self.weights["down_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            compute_kernel_config=self.policy.mlp_compute_kernel_config,
        )
        return ttnn.add(mlp, hidden, dtype=BF16, memory_config=DRAM)

    def decode_forward(self, hidden_states, *, current_pos, position_cos, position_sin, kv_cache, page_table):
        cfg = self.config
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1 or hidden_states.shape[2] != self.batch:
            raise ValueError("decode hidden_states must have shape [1, 1, batch, hidden_size]")
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["input_layernorm"],
            memory_config=DRAM,
            compute_kernel_config=self.policy.norm_compute_kernel_config,
        )
        qkv = ttnn.linear(
            normed,
            self.weights["qkv"],
            dtype=self.policy.activation_dtype,
            memory_config=self.decode_qkv_memcfg,
            compute_kernel_config=self.policy.matmul_compute_kernel_config,
        )
        qkv = ttnn.reshape(qkv, (1, 1, self.batch, qkv.shape[3]), (1, 1, 32, qkv.shape[3]))
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            memory_config=self.decode_head_memcfg,
        )
        query = ttnn.experimental.rotary_embedding_llama(
            query,
            position_cos,
            position_sin,
            self.decode_transformation_mat,
            is_decode_mode=True,
        )
        key = ttnn.experimental.rotary_embedding_llama(
            key,
            position_cos,
            position_sin,
            self.decode_transformation_mat,
            is_decode_mode=True,
        )

        key_cache, value_cache = kv_cache
        ttnn.experimental.paged_update_cache(key_cache, key, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(value_cache, value, update_idxs_tensor=current_pos, page_table=page_table)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.policy.sdpa_compute_kernel_config,
            memory_config=DRAM,
        )
        attn = ttnn.to_memory_config(attn, self.decode_head_memcfg)
        attn = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=cfg.num_attention_heads)
        attn = ttnn.linear(
            attn,
            self.weights["o_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            compute_kernel_config=self.policy.matmul_compute_kernel_config,
        )
        hidden = ttnn.add(attn, residual, dtype=BF16, memory_config=DRAM)

        mlp_input = ttnn.rms_norm(
            hidden,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["post_attention_layernorm"],
            memory_config=DRAM,
            compute_kernel_config=self.policy.norm_compute_kernel_config,
        )
        gate = ttnn.linear(
            mlp_input,
            self.weights["gate_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            program_config=self.decode_mlp_gate_program_config,
            compute_kernel_config=self.policy.mlp_compute_kernel_config,
        )
        up = ttnn.linear(
            mlp_input,
            self.weights["up_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            program_config=self.decode_mlp_up_program_config,
            compute_kernel_config=self.policy.mlp_compute_kernel_config,
        )
        mlp = ttnn.linear(
            ttnn.multiply(gate, up, memory_config=DRAM),
            self.weights["down_proj"],
            dtype=self.policy.activation_dtype,
            memory_config=DRAM,
            program_config=self.decode_mlp_down_program_config,
            compute_kernel_config=self.policy.mlp_compute_kernel_config,
        )
        return ttnn.add(mlp, hidden, dtype=BF16, memory_config=DRAM)

    def forward(self, hidden_states, *, mode="prefill", **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported OptimizedDecoder mode: {mode}")
