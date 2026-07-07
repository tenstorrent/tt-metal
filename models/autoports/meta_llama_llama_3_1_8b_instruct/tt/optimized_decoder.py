# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    Llama31DecoderConfig,
    _get_layer_tensor,
    _to_device_tensor,
    build_causal_mask,
    build_rope_tables,
)
from models.common.lightweightmodule import LightweightModule

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional outside profiling runs.

    def signpost(header):
        return None


def _decode_head_core_grid(mesh_device, batch_size: int) -> ttnn.CoreRangeSet:
    from models.tt_transformers.tt.model_config import num_to_corerange

    compute_grid = mesh_device.compute_with_storage_grid_size()
    grid_x = min(batch_size, compute_grid.x)
    if batch_size >= grid_x and batch_size % grid_x != 0:
        divisors = [x for x in range(grid_x, 0, -1) if batch_size % x == 0 and batch_size // x <= compute_grid.y]
        if not divisors:
            return ttnn.num_cores_to_corerangeset(batch_size, compute_grid, row_wise=True)
        grid_x = divisors[0]
    return ttnn.CoreRangeSet({num_to_corerange(batch_size, grid_x=grid_x, grid_y=compute_grid.y)})


def _decode_head_sub_core_grids(mesh_device, batch_size: int) -> ttnn.CoreRangeSet | None:
    compute_grid = mesh_device.compute_with_storage_grid_size()
    core_grid = _decode_head_core_grid(mesh_device, batch_size)
    ranges = core_grid.ranges()
    if len(ranges) == 1 and ranges[0].start == ttnn.CoreCoord(0, 0):
        return None
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1),
            )
        }
    )


@dataclass(frozen=True)
class PagedKVConfig:
    max_num_blocks: int = 4
    block_size: int = 32
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    def __post_init__(self) -> None:
        if self.block_size % 32 != 0:
            raise ValueError("Paged KV cache block_size must be a multiple of 32 for TTNN paged ops")

    @property
    def max_seq_len(self) -> int:
        return self.max_num_blocks * self.block_size


DEFAULT_PAGED_KV_CONFIG = PagedKVConfig()


@dataclass(frozen=True)
class OptimizedDecoderTimings:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    traced_decode_ms: float | None = None


class OptimizedDecoder(LightweightModule):
    """Llama-3.1-8B-Instruct single-layer optimized decoder path."""

    optimization_profile = {
        "name": "llama31_8b_instruct_optimized_decoder_single_chip_v1",
        "attention_weight_dtype": "bfloat4_b",
        "mlp_weight_dtype": "bfloat4_b",
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat8_b",
        "attention_math_fidelity": "LoFi",
        "mlp_math_fidelity": "LoFi",
        "auxiliary_math_fidelity": "LoFi",
        "prefill_layout": "DRAM interleaved activations with separate Q/K/V projections, tuned K/V prefill config for tile-padded <=32-token prefill, and explicit compute configs",
        "decode_layout": "packed QKV, paged KV cache, L1 height-sharded Q/K/V heads, L1 width-sharded residual/norm/MLP input/final residual, tuned gate/up decode matmuls, DRAM-sharded down projection, traced paged SDPA decode",
        "sdpa": "TTNN scaled_dot_product_attention prefill and paged_scaled_dot_product_attention_decode",
    }

    def __init__(
        self,
        *,
        cfg: Llama31DecoderConfig,
        layer_idx: int,
        hf_config,
        mesh_device,
        q_proj_prefill_weight: ttnn.Tensor,
        k_proj_prefill_weight: ttnn.Tensor,
        v_proj_prefill_weight: ttnn.Tensor,
        qkv_decode_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        down_proj_weight: ttnn.Tensor,
        down_proj_weight_dram_sharded: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        max_seq_len: int,
        paged_kv_config: PagedKVConfig,
        attention_math_fidelity: ttnn.MathFidelity,
        mlp_math_fidelity: ttnn.MathFidelity,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.q_proj_prefill_weight = q_proj_prefill_weight
        self.k_proj_prefill_weight = k_proj_prefill_weight
        self.v_proj_prefill_weight = v_proj_prefill_weight
        self.qkv_decode_weight = qkv_decode_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.down_proj_weight_dram_sharded = down_proj_weight_dram_sharded
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.position_cos = position_cos
        self.position_sin = position_sin
        self.attention_mask = attention_mask
        self.max_seq_len = max_seq_len
        self.paged_kv_config = paged_kv_config
        self.attention_math_fidelity = attention_math_fidelity
        self.mlp_math_fidelity = mlp_math_fidelity
        self.timings = OptimizedDecoderTimings()

        self.compute_kernel_config_hifi2 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_lofi = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.mlp_compute_kernel_config = (
            self.compute_kernel_config_lofi
            if mlp_math_fidelity == ttnn.MathFidelity.LoFi
            else self.compute_kernel_config_hifi2
        )
        self.attention_compute_kernel_config = (
            self.compute_kernel_config_lofi
            if attention_math_fidelity == ttnn.MathFidelity.LoFi
            else self.compute_kernel_config_hifi2
        )
        self.auxiliary_compute_kernel_config = self.compute_kernel_config_lofi
        self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=32,
            k_chunk_size=32,
        )
        self.prefill_kv_program_config = self._prefill_kv_projection_program_config()
        self.decode_head_memcfg = self._decode_head_memory_config(1)

    @staticmethod
    def _pad_to(value: int, multiple: int) -> int:
        remainder = value % multiple
        return value if remainder == 0 else value + multiple - remainder

    @staticmethod
    def _decode_dram_matmul_num_cores(mesh_device) -> int:
        return min(8, mesh_device.compute_with_storage_grid_size().x)

    @classmethod
    def _decode_dram_core_range(cls, mesh_device) -> ttnn.CoreRangeSet:
        num_cores = cls._decode_dram_matmul_num_cores(mesh_device)
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    @classmethod
    def _decode_dram_weight_memory_config(cls, mesh_device, k: int, n: int) -> ttnn.MemoryConfig:
        dram_banks = mesh_device.dram_grid_size().x * mesh_device.dram_grid_size().y
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
                )
            }
        )
        shard_shape = [k, cls._pad_to(n, 32 * dram_banks) // dram_banks]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _decode_dram_input_memory_config(self, k: int) -> ttnn.MemoryConfig:
        num_cores = self._decode_dram_matmul_num_cores(self.mesh_device)
        shard_shape = [32, k // num_cores]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self._decode_dram_core_range(self.mesh_device), shard_shape, ttnn.ShardOrientation.ROW_MAJOR
            ),
        )

    def _decode_dram_output_memory_config(self, n: int) -> ttnn.MemoryConfig:
        num_cores = self._decode_dram_matmul_num_cores(self.mesh_device)
        shard_shape = [32, self._pad_to(n, 32 * num_cores) // num_cores]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self._decode_dram_core_range(self.mesh_device), shard_shape, ttnn.ShardOrientation.ROW_MAJOR
            ),
        )

    def _decode_dram_matmul_program_config(
        self, k: int, n: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        num_cores = self._decode_dram_matmul_num_cores(self.mesh_device)
        shard_k_tiles = k // (32 * num_cores)
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=min(shard_k_tiles, 14),
            per_core_M=1,
            per_core_N=math.ceil(n / (32 * num_cores)),
            fused_activation=None,
        )

    def _decode_gate_up_program_config(self) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        compute_grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = min(8, compute_grid.x)
        grid_y = min(8, compute_grid.y)
        num_cores = grid_x * grid_y
        per_core_n = math.ceil((self.cfg.intermediate_size // 32) / num_cores)
        out_subblock_w = min(8, per_core_n)
        while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
            out_subblock_w -= 1

        residual_grid = self._decode_residual_core_grid()
        residual_block_w = self.cfg.hidden_size // (residual_grid.x * residual_grid.y * 32)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=residual_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _prefill_kv_projection_program_config(self) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        compute_grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = min(8, compute_grid.x)
        grid_y = min(2, compute_grid.y)
        num_cores = grid_x * grid_y
        kv_width_tiles = (self.cfg.num_key_value_heads * self.cfg.head_dim) // 32
        per_core_n = math.ceil(kv_width_tiles / num_cores)
        out_subblock_w = 2 if per_core_n >= 2 and per_core_n % 2 == 0 else 1
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _decode_head_memory_config(self, batch_size: int) -> ttnn.MemoryConfig:
        if batch_size <= 0:
            raise ValueError(f"decode batch_size must be positive, got {batch_size}")
        return ttnn.create_sharded_memory_config(
            shape=(32, self.cfg.head_dim),
            core_grid=_decode_head_core_grid(self.mesh_device, batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_residual_core_grid(self) -> ttnn.CoreGrid:
        compute_grid = self.mesh_device.compute_with_storage_grid_size()
        for grid_y in range(min(4, compute_grid.y), 0, -1):
            for grid_x in range(min(8, compute_grid.x), 0, -1):
                num_cores = grid_x * grid_y
                shard_width = self.cfg.hidden_size // num_cores
                if self.cfg.hidden_size % num_cores == 0 and shard_width % 32 == 0:
                    return ttnn.CoreGrid(x=grid_x, y=grid_y)
        raise ValueError(
            f"cannot create decode residual sharding for hidden_size={self.cfg.hidden_size} "
            f"on compute grid {compute_grid}"
        )

    def _decode_residual_memory_config(self, batch_size: int) -> ttnn.MemoryConfig:
        if batch_size <= 0:
            raise ValueError(f"decode batch_size must be positive, got {batch_size}")
        core_grid = self._decode_residual_core_grid()
        num_cores = core_grid.x * core_grid.y
        return ttnn.create_sharded_memory_config(
            shape=(self._pad_to(batch_size, 32), self.cfg.hidden_size // num_cores),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_residual_norm_program_config(self, batch_size: int) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
        core_grid = self._decode_residual_core_grid()
        num_cores = core_grid.x * core_grid.y
        block_w = self.cfg.hidden_size // num_cores // 32
        subblock_w = min(4, block_w)
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[core_grid.x, core_grid.y],
            subblock_w=subblock_w,
            block_h=self._pad_to(batch_size, 32) // 32,
            block_w=block_w,
            inplace=False,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_seq_len: int = DEFAULT_PAGED_KV_CONFIG.max_seq_len,
        paged_kv_config: PagedKVConfig | None = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        attention_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        mlp_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        **kwargs,
    ) -> "OptimizedDecoder":
        if kwargs:
            raise TypeError(f"unsupported OptimizedDecoder kwargs: {sorted(kwargs)}")
        cfg = Llama31DecoderConfig.from_hf_config(hf_config)
        if max_seq_len <= 0 or max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {cfg.max_position_embeddings}], got {max_seq_len}")
        paged_kv_config = paged_kv_config or DEFAULT_PAGED_KV_CONFIG
        if max_seq_len > paged_kv_config.max_seq_len:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds paged KV capacity {paged_kv_config.max_seq_len}; "
                "increase PagedKVConfig.max_num_blocks or lower max_seq_len"
            )

        q_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        qkv_decode = torch.cat((q_proj.transpose(0, 1), k_proj.transpose(0, 1), v_proj.transpose(0, 1)), dim=1)
        attention_weight_dtype = attention_weight_dtype or weight_dtype
        mlp_weight_dtype = mlp_weight_dtype or weight_dtype

        position_cos, position_sin = build_rope_tables(hf_config, max_seq_len, mesh_device)
        attention_mask = build_causal_mask(max_seq_len, mesh_device)

        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            hf_config=hf_config,
            mesh_device=mesh_device,
            q_proj_prefill_weight=_to_device_tensor(q_proj.transpose(0, 1), mesh_device, dtype=attention_weight_dtype),
            k_proj_prefill_weight=_to_device_tensor(k_proj.transpose(0, 1), mesh_device, dtype=attention_weight_dtype),
            v_proj_prefill_weight=_to_device_tensor(v_proj.transpose(0, 1), mesh_device, dtype=attention_weight_dtype),
            qkv_decode_weight=_to_device_tensor(qkv_decode, mesh_device, dtype=attention_weight_dtype),
            o_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight"),
                mesh_device,
                dtype=attention_weight_dtype,
            ),
            gate_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight"), mesh_device, dtype=mlp_weight_dtype
            ),
            up_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight"), mesh_device, dtype=mlp_weight_dtype
            ),
            down_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight"), mesh_device, dtype=mlp_weight_dtype
            ),
            down_proj_weight_dram_sharded=ttnn.from_torch(
                _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(0, 1).detach().contiguous(),
                dtype=mlp_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=cls._decode_dram_weight_memory_config(
                    mesh_device, cfg.intermediate_size, cfg.hidden_size
                ),
            ),
            input_layernorm_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "input_layernorm.weight"), mesh_device
            ),
            post_attention_layernorm_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device
            ),
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
            max_seq_len=max_seq_len,
            paged_kv_config=paged_kv_config,
            attention_math_fidelity=attention_math_fidelity,
            mlp_math_fidelity=mlp_math_fidelity,
        )

    def init_paged_kv_cache(self) -> list[ttnn.Tensor]:
        cache_shape = (
            self.paged_kv_config.max_num_blocks,
            self.cfg.num_key_value_heads,
            self.paged_kv_config.block_size,
            self.cfg.head_dim,
        )
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)
        return [
            ttnn.from_torch(
                zeros,
                dtype=self.paged_kv_config.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

    def make_identity_page_table(self, batch_size: int = 1) -> ttnn.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.paged_kv_config.max_num_blocks % batch_size != 0:
            raise ValueError(
                f"max_num_blocks={self.paged_kv_config.max_num_blocks} must divide evenly across batch_size={batch_size}"
            )
        blocks_per_user = self.paged_kv_config.max_num_blocks // batch_size
        pages = torch.arange(self.paged_kv_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_size, blocks_per_user
        )
        return ttnn.from_torch(
            pages,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def make_current_pos(self, positions: list[int] | torch.Tensor) -> ttnn.Tensor:
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.int32)
        return ttnn.from_torch(
            positions.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _fill_paged_kv_cache(self, k, v, kv_cache, page_table, *, user_id: int = 0) -> None:
        k_cache, v_cache = kv_cache
        if k.dtype != k_cache.dtype:
            k = ttnn.typecast(k, k_cache.dtype)
        if v.dtype != v_cache.dtype:
            v = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_cache, v, page_table, batch_idx=user_id)

    def _prefill_kv_program_kwargs(self, seq_len: int) -> dict:
        if self._pad_to(seq_len, 32) <= 32:
            return {"program_config": self.prefill_kv_program_config}
        return {}

    def position_tables_for_decode(self, position: int, *, batch_size: int = 1) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(f"decode position must be in [0, {self.max_seq_len}), got {position}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        position_ids = torch.full((1, batch_size), position, dtype=torch.long)
        dummy = torch.empty(batch_size, 1, self.cfg.head_dim, dtype=torch.bfloat16)
        cos, sin = LlamaRotaryEmbedding(self.hf_config)(dummy, position_ids)
        return (
            _to_device_tensor(cos.reshape(1, 1, batch_size, self.cfg.head_dim), self.mesh_device),
            _to_device_tensor(sin.reshape(1, 1, batch_size, self.cfg.head_dim), self.mesh_device),
        )

    def _prefill_attention(self, hidden_states, seq_len, cos, sin, mask, kv_cache=None, page_table=None, user_id=0):
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        q = ttnn.matmul(
            normed,
            self.q_proj_prefill_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        k = ttnn.matmul(
            normed,
            self.k_proj_prefill_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._prefill_kv_program_kwargs(seq_len),
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        v = ttnn.matmul(
            normed,
            self.v_proj_prefill_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._prefill_kv_program_kwargs(seq_len),
            compute_kernel_config=self.attention_compute_kernel_config,
        )

        q = ttnn.reshape(q, [1, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(
            q,
            [0, 0, 0, 0],
            [1, self.cfg.num_attention_heads, seq_len, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        k = ttnn.reshape(k, [1, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(
            k,
            [0, 0, 0, 0],
            [1, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        v = ttnn.reshape(v, [1, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim])
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if kv_cache is not None:
            if page_table is None:
                raise ValueError("page_table is required when prefill_forward fills paged kv_cache")
            self._fill_paged_kv_cache(k, v, kv_cache, page_table, user_id=user_id)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        attn = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(
            attn,
            [1, 1, seq_len, self.cfg.num_attention_heads * self.cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp(
        self,
        post_norm,
        *,
        use_dram_sharded_down: bool = False,
        input_memory_config: ttnn.MemoryConfig | None = ttnn.L1_MEMORY_CONFIG,
        output_memory_config: ttnn.MemoryConfig | None = ttnn.DRAM_MEMORY_CONFIG,
    ):
        if input_memory_config is not None:
            post_norm = ttnn.to_memory_config(post_norm, input_memory_config)
        gate_up_program_config = self._decode_gate_up_program_config() if use_dram_sharded_down else None
        gate = ttnn.matmul(
            post_norm,
            self.gate_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gate_up_program_config,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(
            post_norm,
            self.up_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gate_up_program_config,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if use_dram_sharded_down:
            gated = ttnn.to_memory_config(gated, self._decode_dram_input_memory_config(self.cfg.intermediate_size))
            down = ttnn.matmul(
                gated,
                self.down_proj_weight_dram_sharded,
                dtype=ttnn.bfloat16,
                memory_config=self._decode_dram_output_memory_config(self.cfg.hidden_size),
                program_config=self._decode_dram_matmul_program_config(
                    self.cfg.intermediate_size, self.cfg.hidden_size
                ),
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            if output_memory_config is None:
                return down
            return ttnn.to_memory_config(down, output_memory_config)
        return ttnn.matmul(
            gated,
            self.down_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor | None = None,
        position_sin: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
        kv_cache: list[ttnn.Tensor] | None = None,
        page_table: ttnn.Tensor | None = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        signpost("PERF_PREFILL")
        start = time.perf_counter()
        seq_len = hidden_states.shape[-2]
        if seq_len > self.max_seq_len and (position_cos is None or position_sin is None):
            raise ValueError(
                f"prefill seq_len {seq_len} exceeds setup max_seq_len {self.max_seq_len}; "
                "provide matching RoPE tables or rebuild the decoder"
            )
        cos = position_cos if position_cos is not None else self.position_cos
        sin = position_sin if position_sin is not None else self.position_sin
        mask = attention_mask
        attn = self._prefill_attention(hidden_states, seq_len, cos, sin, mask, kv_cache, page_table, user_id)
        attn_out = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        output = ttnn.add(
            self._mlp(post_norm), attn_residual, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_PREFILL_END")
        self.timings = OptimizedDecoderTimings(
            prefill_ms=elapsed_ms,
            decode_ms=self.timings.decode_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output

    def _decode_qkv(self, hidden_states, position_cos, position_sin, batch_size):
        decode_head_memcfg = self._decode_head_memory_config(batch_size)
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        qkv = ttnn.matmul(
            hidden_states,
            self.qkv_decode_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch_size, 4096], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(
            qkv, [0, 0, 0, 4096], [1, 1, batch_size, 5120], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v = ttnn.slice(
            qkv, [0, 0, 0, 5120], [1, 1, batch_size, 6144], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        q = ttnn.reshape(q, [1, batch_size, self.cfg.num_attention_heads, self.cfg.head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(
            q, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(q, [0, 0, 0, 0], [1, batch_size, self.cfg.num_attention_heads, self.cfg.head_dim], [1, 1, 1, 1])
        q = ttnn.to_memory_config(q, decode_head_memcfg)

        k = ttnn.reshape(k, [1, batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(
            k, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(k, [0, 0, 0, 0], [1, batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim], [1, 1, 1, 1])
        k = ttnn.to_memory_config(k, decode_head_memcfg)

        v = ttnn.reshape(v, [1, batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim])
        v = ttnn.to_memory_config(v, decode_head_memcfg)
        return q, k, v

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        use_persistent_ccl: bool = True,
    ) -> ttnn.Tensor:
        del use_persistent_ccl
        signpost("PERF_DECODE")
        start = time.perf_counter()
        batch_size = hidden_states.shape[-2]
        if hidden_states.shape[-3] != 1:
            raise ValueError(f"decode expects one logical token per user, got shape {hidden_states.shape}")
        q, k, v = self._decode_qkv(hidden_states, position_cos, position_sin, batch_size)
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(
            k_cache,
            k,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.paged_kv_config.block_size,
            num_kv_heads=self.cfg.num_key_value_heads,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.paged_kv_config.block_size,
            num_kv_heads=self.cfg.num_key_value_heads,
        )

        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            program_config=self.sdpa_decode_program_config,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            block_size=self.paged_kv_config.block_size,
            num_kv_heads=self.cfg.num_key_value_heads,
        )
        sdpa = ttnn.to_memory_config(sdpa, self._decode_head_memory_config(batch_size))
        attn = ttnn.experimental.nlp_concat_heads_decode(
            sdpa,
            num_heads=self.cfg.num_attention_heads,
            sub_core_grids=_decode_head_sub_core_grids(self.mesh_device, batch_size),
        )
        attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, batch_size, 4096], [1, 1, 1, 1])
        attn_out = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        residual_memcfg = self._decode_residual_memory_config(batch_size)
        hidden_states = ttnn.to_memory_config(hidden_states, residual_memcfg)
        attn_out = ttnn.to_memory_config(attn_out, residual_memcfg)
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=residual_memcfg)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=residual_memcfg,
            program_config=self._decode_residual_norm_program_config(batch_size),
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        mlp_out = self._mlp(
            post_norm,
            use_dram_sharded_down=True,
            input_memory_config=None,
            output_memory_config=None,
        )
        output = ttnn.add(
            mlp_out,
            attn_residual,
            dtype=ttnn.bfloat16,
            memory_config=residual_memcfg,
        )
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_DECODE_END")
        self.timings = OptimizedDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=elapsed_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output

    def trace_decode_once(self, *args, **kwargs) -> tuple[int, ttnn.Tensor]:
        self.decode_forward(*args, **kwargs)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        output = self.decode_forward(*args, **kwargs)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        signpost("PERF_TRACE_DECODE")
        start = time.perf_counter()
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_TRACE_DECODE_END")
        ttnn.release_trace(self.mesh_device, trace_id)
        self.timings = OptimizedDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=self.timings.decode_ms,
            traced_decode_ms=elapsed_ms,
        )
        return trace_id, output

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str = "prefill", **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported OptimizedDecoder mode: {mode!r}")
