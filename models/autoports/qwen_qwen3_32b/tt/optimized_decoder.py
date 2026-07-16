# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Qwen3-32B decoder layer.

The functional decoder remains the semantic oracle and construction-time source
for validated Qwen constants.  Every measured prefill/decode method is owned by
this class: packed QKV, composite SDPA, fused SwiGLU elementwise work, explicit
phase-specific program/compute configs, and a width-sharded decode residual
stream backed by DRAM-sharded projection weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    FunctionalDecoder,
    _state_tensor,
)


@dataclass
class DecodePositionBuffers:
    """Fixed-address position inputs shared by warm compile and trace replay."""

    cos: ttnn.Tensor
    sin: ttnn.Tensor
    cos_embedding_output: ttnn.Tensor
    sin_embedding_output: ttnn.Tensor
    rope_index: ttnn.Tensor
    update_indices: ttnn.Tensor
    current_pos: int

    def tensors(self) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        return self.cos, self.sin, self.rope_index, self.update_indices


PRECISION_POLICIES = {
    "bf16_hifi4": {
        "attention": ttnn.bfloat16,
        "mlp_gate_up": ttnn.bfloat16,
        "mlp_down": ttnn.bfloat16,
        "kv_cache": ttnn.bfloat16,
        "attention_fidelity": ttnn.MathFidelity.HiFi4,
        "mlp_fidelity": ttnn.MathFidelity.HiFi4,
    },
    "bfp8_hifi2": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat8_b,
        "mlp_down": ttnn.bfloat8_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
        "mlp_fidelity": ttnn.MathFidelity.HiFi2,
    },
    "bfp8_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat8_b,
        "mlp_down": ttnn.bfloat8_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "attention_bfp4_lofi": {
        "attention": ttnn.bfloat4_b,
        "mlp_gate_up": ttnn.bfloat8_b,
        "mlp_down": ttnn.bfloat8_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.HiFi2,
    },
    "mlp_bfp4_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "mlp_bfp4_hifi2": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
        "mlp_fidelity": ttnn.MathFidelity.HiFi2,
    },
    "mlp_bfp4_lofi_kv_bf16": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat16,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "mlp_gate_up_bfp4_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat8_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "mlp_bfp4_attention_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "all_bfp4_lofi": {
        "attention": ttnn.bfloat4_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "all_bfp4_attention_hifi2": {
        "attention": ttnn.bfloat4_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
        "mlp_fidelity": ttnn.MathFidelity.LoFi,
    },
    "all_bfp4_mlp_hifi2": {
        "attention": ttnn.bfloat4_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.HiFi2,
    },
}


def _largest_divisor(value: int, limit: int | None = None) -> int:
    upper = value if limit is None else min(value, limit)
    for candidate in range(upper, 0, -1):
        if value % candidate == 0:
            return candidate
    raise ValueError(f"Expected a positive value, got {value}")


def _core_grid_for_tiles(k_tiles: int, n_tiles: int, *, target_cores: int, device) -> ttnn.CoreGrid:
    grid_size = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid_size.x), int(grid_size.y)
    candidates = []
    for cores in range(1, max_x * max_y + 1):
        if k_tiles % cores or n_tiles % cores:
            continue
        for x in range(min(max_x, cores), 0, -1):
            if cores % x == 0 and cores // x <= max_y:
                candidates.append((abs(cores - target_cores), -cores, -x, x, cores // x))
                break
    if not candidates:
        raise ValueError(f"No exact core grid for Kt={k_tiles}, Nt={n_tiles} on {max_x}x{max_y}")
    _, _, _, x, y = min(candidates)
    return ttnn.CoreGrid(x=x, y=y)


def _width_sharded_memory_config(rows: int, width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
    if width % grid.num_cores:
        raise ValueError(f"width={width} is not divisible by {grid.num_cores} cores")
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // grid.num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _matmul_output_memory_config(rows: int, width: int, grid: ttnn.CoreGrid, device) -> ttnn.MemoryConfig:
    worker_grid = ttnn.num_cores_to_corerangeset(
        grid.num_cores,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // grid.num_cores),
        core_grid=worker_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _padded_width_sharded_memory_config(rows: int, width: int, num_cores: int, device) -> ttnn.MemoryConfig:
    """Width-shard a tiled tensor, padding the final shard when necessary."""

    worker_grid = ttnn.num_cores_to_corerangeset(
        num_cores,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    shard_width = 32 * math.ceil(width / 32 / num_cores)
    return ttnn.create_sharded_memory_config(
        shape=(rows, shard_width),
        core_grid=worker_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _height_sharded_memory_config(rows_per_core: int, width: int, num_cores: int, device) -> ttnn.MemoryConfig:
    worker_grid = ttnn.num_cores_to_corerangeset(
        num_cores,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        shape=(rows_per_core, width),
        core_grid=worker_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_sharded_memory_config(device, k: int, n: int) -> ttnn.MemoryConfig:
    dram_grid_size = device.dram_grid_size()
    dram_banks = int(dram_grid_size.x) * int(dram_grid_size.y)
    if n % (32 * dram_banks):
        raise ValueError(f"N={n} must be divisible by {32 * dram_banks} for DRAM sharding")
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(int(dram_grid_size.x) - 1, int(dram_grid_size.y) - 1),
            )
        }
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, n // dram_banks), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _device_weight(host: torch.Tensor, *, dtype, device, memory_config) -> ttnn.Tensor:
    return ttnn.from_torch(
        host.detach().to(torch.bfloat16).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _compute_config(device, fidelity):
    config_class = (
        ttnn.types.BlackholeComputeKernelConfig
        if device.arch() == ttnn.Arch.BLACKHOLE
        else ttnn.WormholeComputeKernelConfig
    )
    return config_class(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _dram_matmul_program_config(
    m: int,
    k: int,
    n: int,
    grid: ttnn.CoreGrid,
    *,
    fused_activation=None,
    in0_block_w_limit: int | None = None,
):
    k_tiles_per_core = k // 32 // grid.num_cores
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_divisor(k_tiles_per_core, limit=in0_block_w_limit),
        per_core_M=math.ceil(m / 32),
        per_core_N=n // 32 // grid.num_cores,
        fused_activation=fused_activation,
    )


def _advisor_matmul_program_config(*, grid: tuple[int, int], in0_block_w: int, per_core_n: int, out_subblock_w: int):
    """Materialize the exact 1-D multicast config emitted by ttnn-advise."""

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet(set()),
        num_global_cb_receivers=0,
        untilize_out=False,
    )


def _prefill_matmul_program_config(device, m: int, k: int, n: int, *, in0_block_w: int = 4):
    grid_size = device.compute_with_storage_grid_size()
    grid_x = min(10, int(grid_size.x))
    grid_y = min(10, int(grid_size.y))
    per_core_m = math.ceil(math.ceil(m / 32) / grid_y)
    per_core_n = math.ceil(math.ceil(n / 32) / grid_x)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=_largest_divisor(k // 32, limit=in0_block_w),
        out_subblock_h=1,
        out_subblock_w=_largest_divisor(per_core_n, limit=4),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Qwen3 decoder with implementation-owned optimized runtime paths."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        precision_policy: str = "all_bfp4_lofi",
        use_packed_mlp: bool = False,
        decode_matmul_mode: str = "dram_sharded",
        decode_target_cores: int | None = None,
        decode_mlp_target_cores: int | None = None,
        decode_down_target_cores: int | None = None,
        decode_in0_block_w_limit: int | None = None,
        decode_gate_in0_block_w_limit: int | None = None,
        decode_down_in0_block_w_limit: int | None = None,
        decode_sdpa_grid: tuple[int, int] = (8, 8),
        decode_sdpa_exp_approx: bool = False,
        prefill_in0_block_w: int = 10,
        advisor_head_layouts: bool = True,
    ) -> "OptimizedDecoder":
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(f"Unknown precision policy {precision_policy!r}")
        if decode_matmul_mode not in ("dram_sharded", "shard_advisor"):
            raise ValueError("decode_matmul_mode must be 'dram_sharded' or 'shard_advisor'")
        if decode_target_cores is None:
            decode_target_cores = 40 if PRECISION_POLICIES[precision_policy]["mlp_gate_up"] == ttnn.bfloat4_b else 80
        if decode_target_cores not in (16, 20, 32, 40, 80):
            raise ValueError("decode_target_cores must be 16, 20, 32, 40, or 80")
        if decode_mlp_target_cores is None:
            decode_mlp_target_cores = decode_target_cores
        if decode_mlp_target_cores not in (16, 20, 32, 40, 80):
            raise ValueError("decode_mlp_target_cores must be 16, 20, 32, 40, or 80")
        if decode_down_target_cores is None:
            decode_down_target_cores = 32 if PRECISION_POLICIES[precision_policy]["mlp_down"] == ttnn.bfloat4_b else 80
        if decode_down_target_cores not in (16, 20, 32, 40, 80):
            raise ValueError("decode_down_target_cores must be 16, 20, 32, 40, or 80")
        for name, limit in (
            ("decode_in0_block_w_limit", decode_in0_block_w_limit),
            ("decode_gate_in0_block_w_limit", decode_gate_in0_block_w_limit),
            ("decode_down_in0_block_w_limit", decode_down_in0_block_w_limit),
        ):
            if limit is not None and limit not in (1, 2, 4, 5, 8, 10, 20, 25, 40, 50):
                raise ValueError(f"unsupported {name}")
        if decode_gate_in0_block_w_limit is None:
            decode_gate_in0_block_w_limit = decode_in0_block_w_limit
        if decode_down_in0_block_w_limit is None:
            decode_down_in0_block_w_limit = decode_in0_block_w_limit
        if decode_sdpa_grid not in ((8, 4), (8, 6), (8, 8), (7, 10), (10, 10)):
            raise ValueError("unsupported decode_sdpa_grid")
        if prefill_in0_block_w not in (1, 2, 4, 5, 8, 10, 16):
            raise ValueError("unsupported prefill_in0_block_w")
        if decode_matmul_mode == "shard_advisor" and use_packed_mlp:
            raise ValueError("shard_advisor reproduces the emitted separate gate/up graph; packed MLP is unsupported")

        base = FunctionalDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
        )
        self = object.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.precision_policy_name = precision_policy
        self.precision_policy = dict(PRECISION_POLICIES[precision_policy])
        self.use_packed_mlp = use_packed_mlp
        self.decode_matmul_mode = decode_matmul_mode
        self.decode_target_cores = decode_target_cores
        self.decode_mlp_target_cores = decode_mlp_target_cores
        self.decode_down_target_cores = decode_down_target_cores
        self.decode_in0_block_w_limit = decode_in0_block_w_limit
        self.decode_gate_in0_block_w_limit = decode_gate_in0_block_w_limit
        self.decode_down_in0_block_w_limit = decode_down_in0_block_w_limit
        self.decode_sdpa_grid = decode_sdpa_grid
        self.decode_sdpa_exp_approx = decode_sdpa_exp_approx
        self.prefill_in0_block_w = prefill_in0_block_w
        self.advisor_head_layouts = advisor_head_layouts

        for name in ("qkv_weight", "output_weight", "gate_weight", "up_weight", "down_weight"):
            ttnn.deallocate(getattr(self, name), True)

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
        qkv_host = torch.cat((q.T, k.T, v.T), dim=-1)

        attention_dtype = self.precision_policy["attention"]
        mlp_dtype = self.precision_policy["mlp_gate_up"]
        down_dtype = self.precision_policy["mlp_down"]
        self.qkv_weight = _device_weight(
            qkv_host, dtype=attention_dtype, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.output_weight = _device_weight(
            o.T, dtype=attention_dtype, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if use_packed_mlp:
            self.gate_up_weight = _device_weight(
                torch.cat((gate.T, up.T), dim=-1),
                dtype=mlp_dtype,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_weight = self.up_weight = None
        else:
            self.gate_weight = _device_weight(
                gate.T, dtype=mlp_dtype, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.up_weight = _device_weight(
                up.T, dtype=mlp_dtype, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.gate_up_weight = None
        self.down_weight = _device_weight(
            down.T, dtype=down_dtype, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        if decode_matmul_mode == "dram_sharded":
            self.qkv_decode_weight = _device_weight(
                qkv_host,
                dtype=attention_dtype,
                device=mesh_device,
                memory_config=_dram_sharded_memory_config(mesh_device, self.hidden_size, qkv_host.shape[-1]),
            )
            self.output_decode_weight = _device_weight(
                o.T,
                dtype=attention_dtype,
                device=mesh_device,
                memory_config=_dram_sharded_memory_config(mesh_device, self.attention_width, self.hidden_size),
            )
            if use_packed_mlp:
                self.gate_up_decode_weight = _device_weight(
                    torch.cat((gate.T, up.T), dim=-1),
                    dtype=mlp_dtype,
                    device=mesh_device,
                    memory_config=_dram_sharded_memory_config(
                        mesh_device, self.hidden_size, 2 * self.intermediate_size
                    ),
                )
                self.gate_decode_weight = self.up_decode_weight = None
            else:
                self.gate_decode_weight = _device_weight(
                    gate.T,
                    dtype=mlp_dtype,
                    device=mesh_device,
                    memory_config=_dram_sharded_memory_config(mesh_device, self.hidden_size, self.intermediate_size),
                )
                self.up_decode_weight = _device_weight(
                    up.T,
                    dtype=mlp_dtype,
                    device=mesh_device,
                    memory_config=_dram_sharded_memory_config(mesh_device, self.hidden_size, self.intermediate_size),
                )
                self.gate_up_decode_weight = None
            self.down_decode_weight = _device_weight(
                down.T,
                dtype=down_dtype,
                device=mesh_device,
                memory_config=_dram_sharded_memory_config(mesh_device, self.intermediate_size, self.hidden_size),
            )
        else:
            self.qkv_decode_weight = self.output_decode_weight = None
            self.gate_decode_weight = self.up_decode_weight = self.gate_up_decode_weight = None
            self.down_decode_weight = None

        self.attention_compute_config = _compute_config(mesh_device, self.precision_policy["attention_fidelity"])
        self.mlp_compute_config = _compute_config(mesh_device, self.precision_policy["mlp_fidelity"])
        self.norm_compute_config = _compute_config(mesh_device, ttnn.MathFidelity.HiFi2)
        self.kv_cache_dtype = self.precision_policy["kv_cache"]
        self._eager_position_buffers = None
        # Embedding accepts a preallocated output, unlike partial-tile slice.
        # These bounded row-major lookup tables let position refresh run while
        # a trace is live without allocating temporary device tensors.
        self.rotary_cos_row_major = ttnn.to_layout(self.rotary_cos, ttnn.ROW_MAJOR_LAYOUT)
        self.rotary_sin_row_major = ttnn.to_layout(self.rotary_sin, ttnn.ROW_MAJOR_LAYOUT)
        self._build_optimized_configs()
        return self

    def _build_optimized_configs(self) -> None:
        padded_rows = 32 * math.ceil(self.batch / 32)
        hidden_tiles = self.hidden_size // 32
        attention_tiles = self.attention_width // 32
        qkv_tiles = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim // 32
        mlp_tiles = self.intermediate_size // 32
        target = self.decode_target_cores
        mlp_target = self.decode_mlp_target_cores

        self.residual_grid = _core_grid_for_tiles(
            hidden_tiles, hidden_tiles, target_cores=target, device=self.mesh_device
        )
        self.qkv_grid = _core_grid_for_tiles(hidden_tiles, qkv_tiles, target_cores=target, device=self.mesh_device)
        self.o_grid = _core_grid_for_tiles(attention_tiles, hidden_tiles, target_cores=target, device=self.mesh_device)
        self.mlp_gate_grid = _core_grid_for_tiles(
            hidden_tiles, mlp_tiles, target_cores=mlp_target, device=self.mesh_device
        )
        self.mlp_down_grid = _core_grid_for_tiles(
            mlp_tiles,
            hidden_tiles,
            target_cores=self.decode_down_target_cores,
            device=self.mesh_device,
        )
        self.packed_mlp_grid = _core_grid_for_tiles(
            hidden_tiles, 2 * mlp_tiles, target_cores=max(mlp_target, 40), device=self.mesh_device
        )

        self.residual_memory_config = _width_sharded_memory_config(padded_rows, self.hidden_size, self.residual_grid)
        self.qkv_output_memory_config = _matmul_output_memory_config(
            padded_rows, qkv_tiles * 32, self.qkv_grid, self.mesh_device
        )
        self.o_input_memory_config = _width_sharded_memory_config(padded_rows, self.attention_width, self.o_grid)
        self.o_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.o_grid, self.mesh_device
        )
        self.mlp_gate_memory_config = _matmul_output_memory_config(
            padded_rows, self.intermediate_size, self.mlp_gate_grid, self.mesh_device
        )
        self.mlp_gate_input_memory_config = _width_sharded_memory_config(
            padded_rows,
            self.hidden_size,
            self.packed_mlp_grid if self.use_packed_mlp else self.mlp_gate_grid,
        )
        self.mlp_down_input_memory_config = _width_sharded_memory_config(
            padded_rows, self.intermediate_size, self.mlp_down_grid
        )
        self.mlp_down_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.mlp_down_grid, self.mesh_device
        )
        self.packed_mlp_memory_config = _matmul_output_memory_config(
            padded_rows, 2 * self.intermediate_size, self.packed_mlp_grid, self.mesh_device
        )

        block_w = hidden_tiles // self.residual_grid.num_cores
        self.norm_memory_config = self.residual_memory_config
        self.matmul_input_memory_config = self.residual_memory_config
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.residual_grid.x, self.residual_grid.y],
            subblock_w=_largest_divisor(block_w, limit=4),
            block_h=padded_rows // 32,
            block_w=block_w,
            inplace=False,
        )
        self.qkv_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            qkv_tiles * 32,
            self.qkv_grid,
            in0_block_w_limit=self.decode_in0_block_w_limit,
        )
        self.o_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.attention_width,
            self.hidden_size,
            self.o_grid,
            in0_block_w_limit=self.decode_in0_block_w_limit,
        )
        self.gate_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            self.intermediate_size,
            self.mlp_gate_grid,
            in0_block_w_limit=self.decode_gate_in0_block_w_limit,
        )
        self.packed_gate_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            2 * self.intermediate_size,
            self.packed_mlp_grid,
            in0_block_w_limit=self.decode_gate_in0_block_w_limit,
        )
        self.down_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.intermediate_size,
            self.hidden_size,
            self.mlp_down_grid,
            in0_block_w_limit=self.decode_down_in0_block_w_limit,
        )
        if self.decode_matmul_mode == "shard_advisor":
            # Legal projection/layout plan from shard_advise/final_ir.mlir.
            # nlp_concat_heads_decode is the sole adaptation: its source-level
            # contract forces an L1 width-sharded result, so the following
            # slice materializes the advisor's L1-interleaved O-projection input.
            self.residual_memory_config = _padded_width_sharded_memory_config(
                padded_rows, self.hidden_size, 80, self.mesh_device
            )
            self.matmul_input_memory_config = self.residual_memory_config
            self.mlp_gate_input_memory_config = self.residual_memory_config
            self.norm_memory_config = ttnn.create_sharded_memory_config(
                shape=(padded_rows, 15 * 32),
                core_grid=ttnn.CoreGrid(x=11, y=1),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_q_norm_memory_config = ttnn.create_sharded_memory_config(
                shape=(7 * 32, 32),
                core_grid=ttnn.CoreGrid(x=4, y=10),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_k_norm_memory_config = ttnn.create_sharded_memory_config(
                shape=(4 * 32, 32),
                core_grid=ttnn.CoreGrid(x=4, y=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_q_rope_memory_config = _height_sharded_memory_config(32, self.head_dim, 64, self.mesh_device)
            self.advisor_k_rope_memory_config = _height_sharded_memory_config(32, self.head_dim, 32, self.mesh_device)
            self.advisor_rope_row_memory_config = _height_sharded_memory_config(32, self.head_dim, 1, self.mesh_device)
            self.qkv_output_memory_config = _padded_width_sharded_memory_config(
                padded_rows, qkv_tiles * 32, 107, self.mesh_device
            )
            self.o_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.o_output_memory_config = self.residual_memory_config
            self.mlp_gate_memory_config = _padded_width_sharded_memory_config(
                padded_rows, self.intermediate_size, 100, self.mesh_device
            )
            self.mlp_down_input_memory_config = _padded_width_sharded_memory_config(
                padded_rows, self.intermediate_size, 80, self.mesh_device
            )
            self.mlp_down_memory_config = self.residual_memory_config
            self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[11, 1],
                subblock_w=1,
                block_h=1,
                block_w=15,
                inplace=False,
            )
            self.advisor_q_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[4, 10],
                subblock_w=1,
                block_h=7,
                block_w=1,
                inplace=False,
            )
            self.advisor_k_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[4, 8],
                subblock_w=1,
                block_h=4,
                block_w=1,
                inplace=False,
            )
            self.qkv_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 10), in0_block_w=2, per_core_n=3, out_subblock_w=3
            )
            self.o_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2
            )
            self.gate_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 10), in0_block_w=2, per_core_n=8, out_subblock_w=8
            )
            self.down_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2
            )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.decode_sdpa_grid,
            exp_approx_mode=self.decode_sdpa_exp_approx,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    def allocate_kv_cache(self, max_cache_len: int | None = None):
        cache_len = self.max_cache_len if max_cache_len is None else max_cache_len
        shape = (self.batch, self.num_kv_heads, cache_len, self.head_dim)
        return (
            ttnn.zeros(
                shape,
                dtype=self.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.zeros(
                shape,
                dtype=self.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    def optimization_summary(self) -> dict:
        """Return JSON-safe construction evidence for candidate comparisons."""

        def grid_summary(grid) -> dict[str, int]:
            return {"x": int(grid.x), "y": int(grid.y), "cores": int(grid.num_cores)}

        def program_summary(config) -> dict:
            fields = {}
            for name in (
                "in0_block_w",
                "per_core_M",
                "per_core_N",
                "out_subblock_h",
                "out_subblock_w",
            ):
                if hasattr(config, name):
                    fields[name] = int(getattr(config, name))
            fields["class"] = type(config).__name__
            fields["repr"] = repr(config)
            return fields

        return {
            "precision_policy": self.precision_policy_name,
            "activation_dtype": "bfloat16",
            "attention_weight_dtype": str(self.precision_policy["attention"]),
            "attention_fidelity": str(self.precision_policy["attention_fidelity"]),
            "mlp_gate_up_weight_dtype": str(self.precision_policy["mlp_gate_up"]),
            "mlp_down_weight_dtype": str(self.precision_policy["mlp_down"]),
            "mlp_fidelity": str(self.precision_policy["mlp_fidelity"]),
            "kv_cache_dtype": str(self.kv_cache_dtype),
            "decode_matmul_mode": self.decode_matmul_mode,
            "decode_target_cores": self.decode_target_cores,
            "decode_mlp_target_cores": self.decode_mlp_target_cores,
            "decode_down_target_cores": self.decode_down_target_cores,
            "decode_in0_block_w_limit": self.decode_in0_block_w_limit,
            "decode_gate_in0_block_w_limit": self.decode_gate_in0_block_w_limit,
            "decode_down_in0_block_w_limit": self.decode_down_in0_block_w_limit,
            "use_packed_mlp": self.use_packed_mlp,
            "prefill_in0_block_w": self.prefill_in0_block_w,
            "decode_sdpa_grid": list(self.decode_sdpa_grid),
            "decode_sdpa_exp_approx": self.decode_sdpa_exp_approx,
            "advisor_head_layouts": self.advisor_head_layouts,
            "dram_sharded_program_config_contract": {
                "user_fields": ["in0_block_w", "per_core_M", "per_core_N", "fused_activation"],
                "output_subblock": "factory-derived; not exposed by this TTNN program-config class",
            },
            "roles": {
                "residual": {
                    "grid": grid_summary(self.residual_grid),
                    "memory_config": str(self.residual_memory_config),
                },
                "qkv": {
                    "grid": grid_summary(self.qkv_grid),
                    "program_config": program_summary(self.qkv_decode_program_config),
                    "weight_memory_config": (
                        str(self.qkv_decode_weight.memory_config())
                        if self.qkv_decode_weight is not None
                        else str(self.qkv_weight.memory_config())
                    ),
                    "output_memory_config": str(self.qkv_output_memory_config),
                },
                "o": {
                    "grid": grid_summary(self.o_grid),
                    "program_config": program_summary(self.o_decode_program_config),
                    "weight_memory_config": (
                        str(self.output_decode_weight.memory_config())
                        if self.output_decode_weight is not None
                        else str(self.output_weight.memory_config())
                    ),
                    "output_memory_config": str(self.o_output_memory_config),
                },
                "gate_up": {
                    "grid": grid_summary(self.packed_mlp_grid if self.use_packed_mlp else self.mlp_gate_grid),
                    "program_config": program_summary(
                        self.packed_gate_decode_program_config
                        if self.use_packed_mlp
                        else self.gate_decode_program_config
                    ),
                    "output_memory_config": str(
                        self.packed_mlp_memory_config if self.use_packed_mlp else self.mlp_gate_memory_config
                    ),
                    "input_memory_config": str(self.mlp_gate_input_memory_config),
                },
                "down": {
                    "grid": grid_summary(self.mlp_down_grid),
                    "program_config": program_summary(self.down_decode_program_config),
                    "weight_memory_config": (
                        str(self.down_decode_weight.memory_config())
                        if self.down_decode_weight is not None
                        else str(self.down_weight.memory_config())
                    ),
                    "output_memory_config": str(self.mlp_down_memory_config),
                },
            },
            "advisor_legal_layouts": (
                {
                    "initial_and_final_residual": str(ttnn.L1_MEMORY_CONFIG),
                    "residual_matmul_stream": str(self.residual_memory_config),
                    "q_norm": str(self.advisor_q_norm_memory_config),
                    "k_norm": str(self.advisor_k_norm_memory_config),
                    "q_rope": str(self.advisor_q_rope_memory_config),
                    "k_rope": str(self.advisor_k_rope_memory_config),
                    "rope_rows": str(self.advisor_rope_row_memory_config),
                    "concat_adaptation": (
                        "nlp_concat_heads_decode has a fixed L1 width-sharded output; "
                        "slice directly materializes the advisor L1-interleaved O input"
                    ),
                }
                if self.decode_matmul_mode == "shard_advisor"
                else None
            ),
        }

    def roofline_summary(self, current_pos: int, *, peak_dram_gb_s: float = 512.0) -> dict:
        """Return a conservative weights-plus-KV DRAM roofline for one layer step."""

        tile_bytes = {
            ttnn.bfloat16: 2048,
            ttnn.bfloat8_b: 1088,
            ttnn.bfloat4_b: 576,
        }

        def matrix_bytes(k: int, n: int, dtype) -> int:
            return (k // 32) * (n // 32) * tile_bytes[dtype]

        attention_dtype = self.precision_policy["attention"]
        gate_up_dtype = self.precision_policy["mlp_gate_up"]
        down_dtype = self.precision_policy["mlp_down"]
        weights = {
            "qkv": matrix_bytes(
                self.hidden_size,
                (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                attention_dtype,
            ),
            "o": matrix_bytes(self.attention_width, self.hidden_size, attention_dtype),
            "gate": matrix_bytes(self.hidden_size, self.intermediate_size, gate_up_dtype),
            "up": matrix_bytes(self.hidden_size, self.intermediate_size, gate_up_dtype),
            "down": matrix_bytes(self.intermediate_size, self.hidden_size, down_dtype),
        }
        rounded_kv_positions = 32 * math.ceil((current_pos + 1) / 32)
        cache_tiles_per_tensor = self.batch * self.num_kv_heads * (rounded_kv_positions // 32) * (self.head_dim // 32)
        kv_bytes = 2 * cache_tiles_per_tensor * tile_bytes[self.kv_cache_dtype]
        lower_bound_bytes = sum(weights.values()) + kv_bytes
        return {
            "scope": "one batch-32 decoder-layer step",
            "current_pos": current_pos,
            "rounded_kv_positions": rounded_kv_positions,
            "physical_tile_bytes": {
                "bfloat16": tile_bytes[ttnn.bfloat16],
                "bfloat8_b": tile_bytes[ttnn.bfloat8_b],
                "bfloat4_b": tile_bytes[ttnn.bfloat4_b],
            },
            "weight_bytes": weights,
            "two_kv_cache_read_bytes": kv_bytes,
            "weights_plus_kv_lower_bound_bytes": lower_bound_bytes,
            "peak_dram_gb_s_decimal": peak_dram_gb_s,
            "theoretical_lower_bound_ms": lower_bound_bytes / (peak_dram_gb_s * 1e9) * 1e3,
            "excluded": "activation, cache-update, norm, SDPA scratch, layout, and output traffic",
        }

    def _move_owned(self, tensor, memory_config):
        if tensor.memory_config() == memory_config:
            return tensor
        moved = ttnn.to_memory_config(tensor, memory_config)
        ttnn.deallocate(tensor, True)
        return moved

    def _decode_norm(self, residual, weight):
        norm_input = residual
        if norm_input.memory_config() != self.norm_memory_config:
            norm_input = ttnn.to_memory_config(residual, self.norm_memory_config)
        return ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=weight,
            program_config=self.norm_program_config,
            compute_kernel_config=self.norm_compute_config,
            memory_config=self.norm_memory_config,
        )

    def allocate_decode_position_buffers(self, current_pos: int) -> DecodePositionBuffers:
        """Allocate one bounded set of trace-stable position inputs."""

        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        rope_index = ttnn.full(
            [1, 32],
            current_pos,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos_embedding_output = ttnn.embedding(
            rope_index,
            self.rotary_cos_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_embedding_output = ttnn.embedding(
            rope_index,
            self.rotary_sin_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.decode_matmul_mode == "shard_advisor" and self.advisor_head_layouts:
            cos_embedding_output = self._move_owned(cos_embedding_output, self.advisor_rope_row_memory_config)
            sin_embedding_output = self._move_owned(sin_embedding_output, self.advisor_rope_row_memory_config)
        cos = ttnn.reshape(cos_embedding_output, [1, 1, 32, self.head_dim])
        sin = ttnn.reshape(sin_embedding_output, [1, 1, 32, self.head_dim])
        update_indices = ttnn.full(
            [self.batch],
            current_pos,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return DecodePositionBuffers(
            cos=cos,
            sin=sin,
            cos_embedding_output=cos_embedding_output,
            sin_embedding_output=sin_embedding_output,
            rope_index=rope_index,
            update_indices=update_indices,
            current_pos=current_pos,
        )

    def prepare_decode_position_buffers(
        self, buffers: DecodePositionBuffers, current_pos: int
    ) -> DecodePositionBuffers:
        """Refresh stable device buffers before trace replay without reallocating them."""

        if current_pos == buffers.current_pos:
            return buffers
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        rope_index_host = ttnn.Tensor(
            [current_pos] * 32,
            [1, 32],
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        update_indices_host = ttnn.Tensor(
            [current_pos] * self.batch,
            [self.batch],
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(rope_index_host, buffers.rope_index)
        ttnn.copy_host_to_device_tensor(update_indices_host, buffers.update_indices)
        ttnn.embedding(
            buffers.rope_index,
            self.rotary_cos_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=buffers.cos_embedding_output.memory_config(),
            output_tensor=buffers.cos_embedding_output,
        )
        ttnn.embedding(
            buffers.rope_index,
            self.rotary_sin_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=buffers.sin_embedding_output.memory_config(),
            output_tensor=buffers.sin_embedding_output,
        )
        buffers.current_pos = current_pos
        return buffers

    def _eager_decode_position_buffers(self, current_pos: int) -> DecodePositionBuffers:
        if self._eager_position_buffers is None:
            self._eager_position_buffers = self.allocate_decode_position_buffers(current_pos)
        else:
            self.prepare_decode_position_buffers(self._eager_position_buffers, current_pos)
        return self._eager_position_buffers

    def _prefill_mlp_chunk(self, residual):
        rows = int(residual.shape[-2])
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_packed_mlp:
            packed = ttnn.matmul(
                normed,
                self.gate_up_weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    rows,
                    self.hidden_size,
                    2 * self.intermediate_size,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, rows, self.intermediate_size])
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.intermediate_size],
                [1, 1, rows, 2 * self.intermediate_size],
            )
        else:
            program_config = _prefill_matmul_program_config(
                self.mesh_device,
                rows,
                self.hidden_size,
                self.intermediate_size,
                in0_block_w=self.prefill_in0_block_w,
            )
            gate = ttnn.matmul(
                normed,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                program_config=program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.matmul(
                normed,
                self.up_weight,
                dtype=ttnn.bfloat16,
                program_config=program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down = ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=_prefill_matmul_program_config(
                self.mesh_device,
                rows,
                self.intermediate_size,
                self.hidden_size,
                in0_block_w=self.prefill_in0_block_w,
            ),
            compute_kernel_config=self.mlp_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.add(residual, down, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _prefill_linear(self, tensor, weight, *, k: int, n: int, compute_kernel_config):
        """Run a prefill projection without making sequence alignment public.

        Large flattened sequences are internally sliced so program-config
        circular buffers stay bounded.  The last chunk may be non-aligned;
        TTNN owns its tile padding and the logical row count is restored by
        concat.
        """

        rows = int(tensor.shape[-2])
        # Gate/up produce 80 output tiles per X core on the 10x10 prefill
        # grid.  Keeping each chunk at <=640 rows bounds per_core_M at two,
        # the largest configuration proven to fit Blackhole L1.
        max_chunk_rows = 640
        if rows <= max_chunk_rows:
            return ttnn.matmul(
                tensor,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    rows,
                    k,
                    n,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        chunks = []
        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            chunk = ttnn.slice(
                tensor,
                [0, 0, start, 0],
                [1, 1, end, k],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            projected = ttnn.matmul(
                chunk,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    end - start,
                    k,
                    n,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(chunk, True)
            chunks.append(projected)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _prefill_mlp(self, residual):
        rows = int(residual.shape[-2])
        max_chunk_rows = 640
        if rows <= max_chunk_rows:
            return self._prefill_mlp_chunk(residual)
        chunks = []
        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            chunk = ttnn.slice(
                residual,
                [0, 0, start, 0],
                [1, 1, end, self.hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            chunks.append(self._prefill_mlp_chunk(chunk))
            ttnn.deallocate(chunk, True)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache)
        rows = self.batch * seq_len
        residual = ttnn.reshape(hidden_states, [1, 1, rows, self.hidden_size])
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = self._prefill_linear(
            normed,
            self.qkv_weight,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            compute_kernel_config=self.attention_compute_config,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query = ttnn.rms_norm(
            query,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.rms_norm(
            key,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for user_id in range(self.batch):
            key_user = ttnn.slice(
                key,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            value_user = ttnn.slice(
                value,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            if self.kv_cache_dtype != ttnn.bfloat16:
                key_user = ttnn.typecast(key_user, self.kv_cache_dtype)
                value_user = ttnn.typecast(value_user, self.kv_cache_dtype)
            ttnn.fill_cache(key_cache, key_user, user_id)
            ttnn.fill_cache(value_cache, value_user, user_id)
        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=64,
                k_chunk_size=64,
            ),
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, 1, rows, self.attention_width])
        attention = self._prefill_linear(
            attention,
            self.output_weight,
            k=self.attention_width,
            n=self.hidden_size,
            compute_kernel_config=self.attention_compute_config,
        )
        residual = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual = self._prefill_mlp(residual)
        return ttnn.reshape(residual, [1, self.batch, seq_len, self.hidden_size])

    def _decode_mlp(self, residual):
        normed = self._decode_norm(residual, self.post_attention_norm)
        # Gate/up core-count sweeps need a coherent K shard. Keeping the
        # residual's 40-core shard would artificially cap in0_block_w at four
        # tiles even when a 16/20/32-core gate grid admits 10/8/5 tiles.
        normed = self._move_owned(normed, self.mlp_gate_input_memory_config)
        if self.use_packed_mlp:
            packed = ttnn.matmul(
                normed,
                self.gate_up_decode_weight if self.gate_up_decode_weight is not None else self.gate_up_weight,
                dtype=ttnn.bfloat16,
                program_config=self.packed_gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.packed_mlp_memory_config,
            )
            packed = self._move_owned(packed, ttnn.DRAM_MEMORY_CONFIG)
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, self.batch, self.intermediate_size])
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.intermediate_size],
                [1, 1, self.batch, 2 * self.intermediate_size],
            )
            elementwise_memory_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            gate = ttnn.matmul(
                normed,
                self.gate_decode_weight if self.gate_decode_weight is not None else self.gate_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_gate_memory_config,
            )
            up = ttnn.matmul(
                normed,
                self.up_decode_weight if self.up_decode_weight is not None else self.up_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_gate_memory_config,
            )
            elementwise_memory_config = self.mlp_gate_memory_config
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=elementwise_memory_config,
        )
        gated = self._move_owned(gated, self.mlp_down_input_memory_config)
        down = ttnn.matmul(
            gated,
            self.down_decode_weight if self.down_decode_weight is not None else self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=self.down_decode_program_config,
            compute_kernel_config=self.mlp_compute_config,
            memory_config=self.mlp_down_memory_config,
        )
        down = self._move_owned(down, self.residual_memory_config)
        return ttnn.add(residual, down, dtype=ttnn.bfloat16, memory_config=self.residual_memory_config)

    def decode_forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        current_pos: int,
        position_buffers: DecodePositionBuffers | None = None,
    ):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        if self.decode_matmul_mode == "shard_advisor":
            residual = ttnn.reshape(
                hidden_states,
                [1, 1, self.batch, self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            residual = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
            residual = ttnn.to_memory_config(residual, self.residual_memory_config)
        normed = self._decode_norm(residual, self.input_norm)
        normed = self._move_owned(normed, self.matmul_input_memory_config)
        qkv_weight = self.qkv_decode_weight if self.qkv_decode_weight is not None else self.qkv_weight
        fused_qkv = ttnn.matmul(
            normed,
            qkv_weight,
            dtype=ttnn.bfloat16,
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.qkv_output_memory_config,
        )
        fused_qkv = self._move_owned(fused_qkv, ttnn.L1_MEMORY_CONFIG)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_heads_mem_config,
        )
        value = self._move_owned(value, self.decode_kv_mem_config)
        if self.decode_matmul_mode == "shard_advisor" and self.advisor_head_layouts:
            query = self._move_owned(query, self.advisor_q_norm_memory_config)
            query = ttnn.rms_norm(
                query,
                epsilon=self.rms_norm_eps,
                weight=self.q_norm,
                program_config=self.advisor_q_norm_program_config,
                compute_kernel_config=self.norm_compute_config,
                memory_config=self.advisor_q_norm_memory_config,
            )
            query = self._move_owned(query, self.advisor_q_rope_memory_config)
            key = self._move_owned(key, self.advisor_k_norm_memory_config)
            key = ttnn.rms_norm(
                key,
                epsilon=self.rms_norm_eps,
                weight=self.k_norm,
                program_config=self.advisor_k_norm_program_config,
                compute_kernel_config=self.norm_compute_config,
                memory_config=self.advisor_k_norm_memory_config,
            )
            key = self._move_owned(key, self.advisor_k_rope_memory_config)
            rotary_memory_config_q = self.advisor_q_rope_memory_config
            rotary_memory_config_k = self.advisor_k_rope_memory_config
        else:
            query = self._move_owned(query, ttnn.DRAM_MEMORY_CONFIG)
            key = self._move_owned(key, ttnn.DRAM_MEMORY_CONFIG)
            query = ttnn.rms_norm(
                query,
                epsilon=self.rms_norm_eps,
                weight=self.q_norm,
                compute_kernel_config=self.norm_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            key = ttnn.rms_norm(
                key,
                epsilon=self.rms_norm_eps,
                weight=self.k_norm,
                compute_kernel_config=self.norm_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rotary_memory_config_q = ttnn.DRAM_MEMORY_CONFIG
            rotary_memory_config_k = ttnn.DRAM_MEMORY_CONFIG
        if position_buffers is None:
            position_buffers = self._eager_decode_position_buffers(current_pos)
        elif position_buffers.current_pos != current_pos:
            raise ValueError(
                f"position_buffers hold position {position_buffers.current_pos}, expected current_pos={current_pos}"
            )
        cos, sin = position_buffers.cos, position_buffers.sin
        query = ttnn.experimental.rotary_embedding(query, cos, sin, 0, memory_config=rotary_memory_config_q)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, 0, memory_config=rotary_memory_config_k)
        key = self._move_owned(key, self.decode_kv_mem_config)
        update_indices = position_buffers.update_indices
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=update_indices, share_cache=False, page_table=None
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=update_indices, share_cache=False, page_table=None
        )
        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            cur_pos_tensor=update_indices,
            scale=self.scale,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_concat_input_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        if self.decode_matmul_mode == "shard_advisor":
            attention = ttnn.slice(
                attention,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.attention_width],
                [1, 1, 1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            attention = self._move_owned(attention, ttnn.DRAM_MEMORY_CONFIG)
            attention = ttnn.slice(
                attention,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.attention_width],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attention = ttnn.to_memory_config(attention, self.o_input_memory_config)
        attention = ttnn.matmul(
            attention,
            self.output_decode_weight if self.output_decode_weight is not None else self.output_weight,
            dtype=ttnn.bfloat16,
            program_config=self.o_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.o_output_memory_config,
        )
        attention = self._move_owned(attention, self.residual_memory_config)
        residual = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=self.residual_memory_config,
        )
        residual = self._decode_mlp(residual)
        # Reshape the public output in its final DRAM allocation.  Reshaping
        # an L1-interleaved tensor and copying that result to DRAM created a
        # separate ~41 us CopyDeviceOperation on every traced decode replay.
        if self.decode_matmul_mode == "shard_advisor":
            residual = self._move_owned(residual, ttnn.L1_MEMORY_CONFIG)
            residual = ttnn.reshape(
                residual,
                [1, self.batch, 1, self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return self._move_owned(residual, ttnn.DRAM_MEMORY_CONFIG)
        residual = self._move_owned(residual, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(residual, [1, self.batch, 1, self.hidden_size])

    def forward(self, hidden_states, key_cache, value_cache, *, mode: str, current_pos: int | None = None):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, key_cache, value_cache)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(hidden_states, key_cache, value_cache, current_pos=current_pos)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
