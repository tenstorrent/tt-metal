# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Falcon3 decoder layer.

The functional decoder remains the semantic reference.  This implementation
owns every measured prefill/decode hot-path method and adds explicit Blackhole-
aware layouts, program configs, compute configs, weight precision, and
DRAM-sharded decode matmuls.  Reusing the functional loader for validation and
RoPE construction is intentionally limited to construction time.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.functional_decoder import (
    CURRENT_SUPPORTED_CONTEXT,
    EMITTED_BATCH,
    FunctionalDecoder,
    _resolve_layer_tensor,
)

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
    "attention_bfp8_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat8_b,
        "mlp_down": ttnn.bfloat8_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
        "mlp_fidelity": ttnn.MathFidelity.HiFi2,
    },
    "mlp_bfp8_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat8_b,
        "mlp_down": ttnn.bfloat8_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.HiFi2,
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
    "mlp_bfp4_attention_lofi": {
        "attention": ttnn.bfloat8_b,
        "mlp_gate_up": ttnn.bfloat4_b,
        "mlp_down": ttnn.bfloat4_b,
        "kv_cache": ttnn.bfloat8_b,
        "attention_fidelity": ttnn.MathFidelity.LoFi,
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


def _core_grid_for_tiles(
    k_tiles: int,
    n_tiles: int,
    *,
    target_cores: int,
    max_x: int,
    max_y: int,
) -> ttnn.CoreGrid:
    """Choose a rectangular grid that exactly divides both tiled dimensions."""
    candidates = []
    for cores in range(1, max_x * max_y + 1):
        if k_tiles % cores or n_tiles % cores:
            continue
        for x in range(min(max_x, cores), 0, -1):
            if cores % x == 0 and cores // x <= max_y:
                candidates.append((abs(cores - target_cores), -cores, -x, x, cores // x))
                break
    if not candidates:
        raise ValueError(f"No exact core grid for Kt={k_tiles}, Nt={n_tiles} within {max_x}x{max_y}")
    _, _, _, x, y = min(candidates)
    return ttnn.CoreGrid(x=x, y=y)


def _largest_divisor(value: int, limit: int | None = None) -> int:
    if value <= 0:
        raise ValueError(f"Expected a positive tiled shard width, got {value}")
    upper = value if limit is None else min(value, limit)
    for candidate in range(upper, 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("one always divides a positive integer")


def _sharded_memory_config(rows: int, width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
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
    """Match Blackhole DRAM-matmul's round-robin worker-core selection."""
    if width % grid.num_cores:
        raise ValueError(f"width={width} is not divisible by {grid.num_cores} cores")
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


def _device_weight(host: torch.Tensor, *, dtype, device, memory_config=None) -> ttnn.Tensor:
    host = host.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=(
            _dram_sharded_memory_config(device, int(host.shape[-2]), int(host.shape[-1]))
            if memory_config is None
            else memory_config
        ),
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
    in0_block_w: int | None = None,
):
    cores = grid.num_cores
    k_tiles_per_core = k // 32 // cores
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_divisor(k_tiles_per_core) if in0_block_w is None else in0_block_w,
        per_core_M=math.ceil(m / 32),
        per_core_N=n // 32 // cores,
        fused_activation=fused_activation,
    )


def _prefill_matmul_program_config(
    device,
    m: int,
    k: int,
    n: int,
    *,
    fused_activation=None,
    grid_x_limit: int = 8,
    in0_block_w: int = 1,
):
    grid_size = device.compute_with_storage_grid_size()
    grid_x = min(grid_x_limit, int(grid_size.x))
    grid_y = min(10 if device.arch() == ttnn.Arch.BLACKHOLE else 8, int(grid_size.y))
    per_core_m = math.ceil(math.ceil(m / 32) / grid_y)
    per_core_n = math.ceil(math.ceil(n / 32) / grid_x)
    out_subblock_w = _largest_divisor(per_core_n, limit=4)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def _advisor_matmul_program_config(*, grid, in0_block_w, per_core_n, out_subblock_w):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _width_sharded_output_memory_config(rows: int, width: int, num_cores: int, device):
    if rows <= 0 or width <= 0 or num_cores <= 0:
        raise ValueError(f"invalid output sharding rows={rows}, width={width}, cores={num_cores}")
    worker_grid = ttnn.num_cores_to_corerangeset(
        num_cores,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        shape=(rows, 32 * math.ceil(width / 32 / num_cores)),
        core_grid=worker_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _advisor_norm_memory_config(rows: int, width: int):
    """Exact 11-core block-sharded RMSNorm layout from final_ir.mlir."""
    grid = ttnn.CoreGrid(x=11, y=1)
    shard_width = 32 * math.ceil(width / 32 / grid.num_cores)
    return ttnn.create_sharded_memory_config(
        shape=(rows, shard_width),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Falcon3 decoder with an optimized implementation-owned runtime path."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = CURRENT_SUPPORTED_CONTEXT,
        precision_policy: str = "all_bfp4_lofi",
        use_packed_mlp: bool = False,
        use_explicit_decode_mask: bool = False,
        decode_matmul_mode: str = "dram_sharded",
        dram_mlp_target_cores: int | None = None,
        advisor_mlp_geometry: str = "report",
        advisor_residual_mode: str = "legacy_32core",
        advisor_matmul_input_mode: str = "interleaved",
        prefill_grid_x: int = 11,
        prefill_in0_block_w: int = 8,
        align_dram_mlp_down_input: bool = True,
    ) -> "OptimizedDecoder":
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(f"Unknown precision policy {precision_policy!r}; choose from {sorted(PRECISION_POLICIES)}")
        if decode_matmul_mode not in ("dram_sharded", "shard_advisor"):
            raise ValueError("decode_matmul_mode must be 'dram_sharded' or 'shard_advisor'")
        if dram_mlp_target_cores is None:
            dram_mlp_target_cores = 24 if PRECISION_POLICIES[precision_policy]["mlp_gate_up"] == ttnn.bfloat4_b else 48
        if dram_mlp_target_cores not in (6, 12, 16, 24, 48):
            raise ValueError("dram_mlp_target_cores must be 6, 12, 16, 24, or 48")
        if advisor_mlp_geometry not in ("report", "wide_blocks", "wider_blocks"):
            raise ValueError("advisor_mlp_geometry must be 'report', 'wide_blocks', or 'wider_blocks'")
        if advisor_residual_mode not in ("legacy_32core", "report"):
            raise ValueError("advisor_residual_mode must be 'legacy_32core' or 'report'")
        if advisor_matmul_input_mode not in ("interleaved", "report_sharded"):
            raise ValueError("advisor_matmul_input_mode must be 'interleaved' or 'report_sharded'")
        if prefill_grid_x not in (8, 11):
            raise ValueError("prefill_grid_x must be 8 or 11")
        if prefill_in0_block_w not in (1, 2, 4, 8, 16):
            raise ValueError("prefill_in0_block_w must be 1, 2, 4, 8, or 16")

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
        self.use_explicit_decode_mask = use_explicit_decode_mask
        self.decode_matmul_mode = decode_matmul_mode
        self.dram_mlp_target_cores = dram_mlp_target_cores
        self.advisor_mlp_geometry = advisor_mlp_geometry
        self.advisor_residual_mode = advisor_residual_mode
        self.advisor_matmul_input_mode = advisor_matmul_input_mode
        self.prefill_grid_x = prefill_grid_x
        self.prefill_in0_block_w = prefill_in0_block_w
        self.align_dram_mlp_down_input = align_dram_mlp_down_input

        # The functional constructor derives the decode-head shard height from
        # the logical batch.  Tile-layout tensors require a tile-aligned
        # physical shard, so batches below one tile otherwise produce an
        # invalid ``[batch, head_dim]`` shard (for example ``[1, 256]``).
        # Preserve the logical batch while padding only the physical shard.
        if batch < 32:
            self.decode_head_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, self.head_dim),
                core_grid=ttnn.CoreGrid(x=1, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        for name in ("qkv_weight", "o_weight", "gate_weight", "up_weight", "down_weight"):
            ttnn.deallocate(getattr(self, name), True)

        q = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate = _resolve_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
        up = _resolve_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
        down = _resolve_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight")

        qkv_host = torch.cat([q.T, k.T, v.T], dim=-1)
        # Large-M prefill matmuls require conventional interleaved DRAM
        # weights.  The decode-only DRAM-sharded program has a different
        # weight-layout contract, so keep dedicated copies for that path.
        # This is deliberately independent of dtype: BFP8 is a control for
        # the same layout contract, while the selected DRAM path intentionally
        # owns phase-specific duplicate weights.
        split_decode_weights = decode_matmul_mode == "dram_sharded"
        prefill_weight_memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.qkv_weight = _device_weight(
            qkv_host,
            dtype=self.precision_policy["attention"],
            device=mesh_device,
            memory_config=prefill_weight_memory_config,
        )
        self.qkv_decode_weight = (
            _device_weight(qkv_host, dtype=self.precision_policy["attention"], device=mesh_device)
            if split_decode_weights
            else None
        )
        self.o_weight = _device_weight(
            o.T,
            dtype=self.precision_policy["attention"],
            device=mesh_device,
            memory_config=prefill_weight_memory_config,
        )
        self.o_decode_weight = (
            _device_weight(o.T, dtype=self.precision_policy["attention"], device=mesh_device)
            if split_decode_weights
            else None
        )
        if use_packed_mlp:
            self.gate_up_weight = _device_weight(
                torch.cat([gate.T, up.T], dim=-1),
                dtype=self.precision_policy["mlp_gate_up"],
                device=mesh_device,
                memory_config=prefill_weight_memory_config,
            )
            self.gate_up_decode_weight = (
                _device_weight(
                    torch.cat([gate.T, up.T], dim=-1),
                    dtype=self.precision_policy["mlp_gate_up"],
                    device=mesh_device,
                )
                if split_decode_weights
                else None
            )
            self.gate_weight = None
            self.up_weight = None
            self.gate_decode_weight = None
            self.up_decode_weight = None
        else:
            self.gate_weight = _device_weight(
                gate.T,
                dtype=self.precision_policy["mlp_gate_up"],
                device=mesh_device,
                memory_config=prefill_weight_memory_config,
            )
            self.up_weight = _device_weight(
                up.T,
                dtype=self.precision_policy["mlp_gate_up"],
                device=mesh_device,
                memory_config=prefill_weight_memory_config,
            )
            self.gate_decode_weight = (
                _device_weight(gate.T, dtype=self.precision_policy["mlp_gate_up"], device=mesh_device)
                if split_decode_weights
                else None
            )
            self.up_decode_weight = (
                _device_weight(up.T, dtype=self.precision_policy["mlp_gate_up"], device=mesh_device)
                if split_decode_weights
                else None
            )
            self.gate_up_weight = None
            self.gate_up_decode_weight = None
        self.down_weight = _device_weight(
            down.T,
            dtype=self.precision_policy["mlp_down"],
            device=mesh_device,
            memory_config=prefill_weight_memory_config,
        )
        self.down_decode_weight = (
            _device_weight(down.T, dtype=self.precision_policy["mlp_down"], device=mesh_device)
            if split_decode_weights
            else None
        )

        self.attention_compute_config = _compute_config(mesh_device, self.precision_policy["attention_fidelity"])
        self.mlp_compute_config = _compute_config(mesh_device, self.precision_policy["mlp_fidelity"])
        self.norm_compute_config = _compute_config(mesh_device, ttnn.MathFidelity.HiFi2)
        self.kv_cache_dtype = self.precision_policy["kv_cache"]
        self._build_optimized_configs()
        return self

    def _build_optimized_configs(self) -> None:
        grid_size = self.mesh_device.compute_with_storage_grid_size()
        max_x = min(8, int(grid_size.x))
        max_y = min(10 if self.mesh_device.arch() == ttnn.Arch.BLACKHOLE else 8, int(grid_size.y))
        hidden_tiles = self.hidden_size // 32
        qkv_tiles = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim // 32
        mlp_tiles = self.intermediate_size // 32
        packed_mlp_tiles = 2 * mlp_tiles
        padded_batch_rows = 32 * math.ceil(self.batch / 32)

        self.residual_grid = _core_grid_for_tiles(hidden_tiles, hidden_tiles, target_cores=32, max_x=max_x, max_y=max_y)
        self.qkv_grid = _core_grid_for_tiles(hidden_tiles, qkv_tiles, target_cores=32, max_x=max_x, max_y=max_y)
        self.mlp_gate_grid = _core_grid_for_tiles(
            hidden_tiles,
            mlp_tiles,
            target_cores=self.dram_mlp_target_cores,
            max_x=max_x,
            max_y=max_y,
        )
        self.mlp_down_grid = _core_grid_for_tiles(
            mlp_tiles,
            hidden_tiles,
            target_cores=self.dram_mlp_target_cores,
            max_x=max_x,
            max_y=max_y,
        )
        self.packed_mlp_grid = _core_grid_for_tiles(
            hidden_tiles, packed_mlp_tiles, target_cores=48, max_x=max_x, max_y=max_y
        )

        self.residual_memory_config = _sharded_memory_config(padded_batch_rows, self.hidden_size, self.residual_grid)
        self.qkv_output_memory_config = _matmul_output_memory_config(
            padded_batch_rows,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            self.qkv_grid,
            self.mesh_device,
        )
        self.mlp_gate_input_memory_config = _sharded_memory_config(
            padded_batch_rows, self.hidden_size, self.mlp_gate_grid
        )
        self.mlp_gate_output_memory_config = _matmul_output_memory_config(
            padded_batch_rows, self.intermediate_size, self.mlp_gate_grid, self.mesh_device
        )
        self.mlp_down_input_memory_config = _sharded_memory_config(
            padded_batch_rows, self.intermediate_size, self.mlp_down_grid
        )
        if self.decode_matmul_mode == "dram_sharded" and self.align_dram_mlp_down_input:
            self.mlp_down_input_memory_config = self.mlp_gate_output_memory_config
        self.mlp_down_output_memory_config = _matmul_output_memory_config(
            padded_batch_rows, self.hidden_size, self.mlp_down_grid, self.mesh_device
        )
        self.packed_mlp_output_memory_config = _matmul_output_memory_config(
            padded_batch_rows, 2 * self.intermediate_size, self.packed_mlp_grid, self.mesh_device
        )
        self.o_output_memory_config = _matmul_output_memory_config(
            padded_batch_rows, self.hidden_size, self.residual_grid, self.mesh_device
        )

        block_w = self.hidden_size // 32 // self.residual_grid.num_cores
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.residual_grid.x, self.residual_grid.y],
            subblock_w=_largest_divisor(block_w, limit=4),
            block_h=padded_batch_rows // 32,
            block_w=block_w,
            inplace=False,
        )
        self.norm_memory_config = self.residual_memory_config
        self.decode_input_memory_config = self.residual_memory_config
        self.qkv_input_memory_config = self.residual_memory_config
        self.gate_input_memory_config = self.mlp_gate_input_memory_config
        self.down_input_memory_config = self.mlp_down_input_memory_config
        self.qkv_decode_program_config = _dram_matmul_program_config(
            padded_batch_rows,
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            self.qkv_grid,
        )
        self.o_decode_program_config = _dram_matmul_program_config(
            padded_batch_rows, self.hidden_size, self.hidden_size, self.residual_grid
        )
        self.gate_decode_program_config = _dram_matmul_program_config(
            padded_batch_rows,
            self.hidden_size,
            self.intermediate_size,
            self.mlp_gate_grid,
        )
        self.packed_gate_decode_program_config = _dram_matmul_program_config(
            padded_batch_rows,
            self.hidden_size,
            2 * self.intermediate_size,
            self.packed_mlp_grid,
            in0_block_w=1,
        )
        self.down_decode_program_config = _dram_matmul_program_config(
            padded_batch_rows,
            self.intermediate_size,
            self.hidden_size,
            self.mlp_down_grid,
        )
        if self.decode_matmul_mode == "shard_advisor":
            # Exact program configs from shard_advise/final_ir.mlir. The
            # report-sharded inputs are available below as a measured A/B;
            # L1-interleaved inputs are the faster real-weight default.
            self.qkv_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2
            )
            self.o_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.gate_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 10), in0_block_w=2, per_core_n=7, out_subblock_w=7
            )
            self.down_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 9), in0_block_w=8, per_core_n=1, out_subblock_w=1
            )
            self.qkv_output_memory_config = _width_sharded_output_memory_config(
                padded_batch_rows,
                (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                80,
                self.mesh_device,
            )
            self.o_output_memory_config = _width_sharded_output_memory_config(
                padded_batch_rows, self.hidden_size, 96, self.mesh_device
            )
            self.mlp_gate_output_memory_config = _width_sharded_output_memory_config(
                padded_batch_rows, self.intermediate_size, 103, self.mesh_device
            )
            self.mlp_down_output_memory_config = _width_sharded_output_memory_config(
                padded_batch_rows, self.hidden_size, 96, self.mesh_device
            )
            self.packed_gate_decode_program_config = _advisor_matmul_program_config(
                grid=(11, 10), in0_block_w=2, per_core_n=14, out_subblock_w=7
            )
            self.packed_mlp_output_memory_config = _width_sharded_output_memory_config(
                padded_batch_rows, 2 * self.intermediate_size, 103, self.mesh_device
            )
            self.qkv_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.gate_input_memory_config = ttnn.L1_MEMORY_CONFIG
            self.down_input_memory_config = ttnn.L1_MEMORY_CONFIG
            if self.advisor_matmul_input_mode == "report_sharded":
                self.qkv_input_memory_config = _width_sharded_output_memory_config(
                    padded_batch_rows, self.hidden_size, 48, self.mesh_device
                )
                self.gate_input_memory_config = self.qkv_input_memory_config
                self.down_input_memory_config = _width_sharded_output_memory_config(
                    padded_batch_rows, self.intermediate_size, 90, self.mesh_device
                )
            if self.advisor_residual_mode == "report":
                self.decode_input_memory_config = ttnn.L1_MEMORY_CONFIG
                self.residual_memory_config = _width_sharded_output_memory_config(
                    padded_batch_rows, self.hidden_size, 96, self.mesh_device
                )
                self.norm_memory_config = _advisor_norm_memory_config(padded_batch_rows, self.hidden_size)
                norm_block_w = math.ceil(hidden_tiles / 11)
                self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[11, 1],
                    subblock_w=_largest_divisor(norm_block_w, limit=4),
                    block_h=padded_batch_rows // 32,
                    block_w=norm_block_w,
                    inplace=False,
                )
            if self.advisor_mlp_geometry == "wide_blocks":
                self.gate_decode_program_config = _advisor_matmul_program_config(
                    grid=(10, 9), in0_block_w=8, per_core_n=8, out_subblock_w=8
                )
                self.down_decode_program_config = _advisor_matmul_program_config(
                    grid=(8, 6), in0_block_w=16, per_core_n=2, out_subblock_w=2
                )
                self.mlp_gate_output_memory_config = _sharded_memory_config(
                    padded_batch_rows, self.intermediate_size, ttnn.CoreGrid(x=10, y=9)
                )
                self.mlp_down_output_memory_config = _sharded_memory_config(
                    padded_batch_rows, self.hidden_size, ttnn.CoreGrid(x=8, y=6)
                )
            elif self.advisor_mlp_geometry == "wider_blocks":
                self.gate_decode_program_config = _advisor_matmul_program_config(
                    grid=(10, 8), in0_block_w=16, per_core_n=9, out_subblock_w=3
                )
                self.down_decode_program_config = _advisor_matmul_program_config(
                    grid=(8, 3), in0_block_w=16, per_core_n=4, out_subblock_w=4
                )
                self.mlp_gate_output_memory_config = _sharded_memory_config(
                    padded_batch_rows, self.intermediate_size, ttnn.CoreGrid(x=10, y=8)
                )
                self.mlp_down_output_memory_config = _sharded_memory_config(
                    padded_batch_rows, self.hidden_size, ttnn.CoreGrid(x=8, y=3)
                )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    def allocate_kv_cache(self, max_cache_len: int | None = None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cache_len = self.max_cache_len if max_cache_len is None else max_cache_len
        if cache_len <= 0:
            raise ValueError(f"max_cache_len must be positive, got {cache_len}")
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

    def _rotary_slice(self, start: int, length: int):
        end = start + length
        if start < 0 or end > self.max_cache_len:
            raise ValueError(f"rotary range [{start}, {end}) exceeds cache length {self.max_cache_len}")
        starts = [0, 0, start, 0]
        ends = [1, 1, end, self.head_dim]
        return ttnn.slice(self.cos_cache, starts, ends), ttnn.slice(self.sin_cache, starts, ends)

    def _unpad_prefill_sequence(self, tensor, *, batch, heads, seq_len):
        if int(tensor.shape[2]) == seq_len:
            return tensor
        padded = tensor
        tensor = ttnn.slice(padded, [0, 0, 0, 0], [batch, heads, seq_len, self.head_dim])
        ttnn.deallocate(padded, True)
        return tensor

    def _prepare_decode_heads(self, tensor, num_heads, *, memory_config):
        interleaved = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tensor, True)
        unpadded = ttnn.slice(interleaved, [0, 0, 0, 0], [1, self.batch, num_heads, self.head_dim])
        ttnn.deallocate(interleaved, True)
        prepared = ttnn.to_memory_config(unpadded, memory_config)
        ttnn.deallocate(unpadded, True)
        return prepared

    def _decode_attention_mask(self, cache_position):
        scalar_position = ttnn.slice(cache_position, [0], [1])
        valid_positions = ttnn.le(
            self.decode_positions,
            scalar_position,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(scalar_position, False)
        single_head_mask = ttnn.where(
            valid_positions,
            0.0,
            -3.3895313892515355e38,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(valid_positions, True)
        attention_mask = ttnn.repeat(
            single_head_mask,
            ttnn.Shape([1, 1, self.num_heads, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(single_head_mask, True)
        return attention_mask

    def _prefill_attention(self, residual, *, batch, seq_len, key_cache, value_cache):
        m = batch * seq_len
        # Explicit multicast configs are valuable for the measured prompt
        # sizes, but their per-core M scales linearly and eventually makes the
        # circular buffers exceed L1.  TTNN's automatic large-M selection is
        # bounded and is the same device-only mechanism used by the functional
        # capacity baseline.
        large_m = m > 1024
        qkv_program_config = None
        o_program_config = None
        if not large_m:
            qkv_program_config = _prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.hidden_size,
                (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                grid_x_limit=self.prefill_grid_x,
                in0_block_w=self.prefill_in0_block_w,
            )
            o_program_config = _prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.hidden_size,
                self.hidden_size,
                grid_x_limit=self.prefill_grid_x,
                in0_block_w=self.prefill_in0_block_w,
            )
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm_weight,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.matmul(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            program_config=qkv_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(normed, True)
        fused_qkv_matmul = fused_qkv
        fused_qkv = ttnn.reshape(
            fused_qkv_matmul, (batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim)
        )
        ttnn.deallocate(fused_qkv_matmul, False)
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused_qkv, True)
        cos, sin = self._rotary_slice(0, seq_len)
        key_rotated = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query_rotated = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(key, True)
        ttnn.deallocate(query, True)
        ttnn.deallocate(cos, False)
        ttnn.deallocate(sin, False)
        key_rotated = self._unpad_prefill_sequence(key_rotated, batch=batch, heads=self.num_kv_heads, seq_len=seq_len)
        query_rotated = self._unpad_prefill_sequence(query_rotated, batch=batch, heads=self.num_heads, seq_len=seq_len)

        for user_id in range(batch):
            value_user = ttnn.slice(value, [user_id, 0, 0, 0], [user_id + 1, self.num_kv_heads, seq_len, self.head_dim])
            key_user = ttnn.slice(
                key_rotated, [user_id, 0, 0, 0], [user_id + 1, self.num_kv_heads, seq_len, self.head_dim]
            )
            if self.kv_cache_dtype != ttnn.bfloat16:
                value_fill = ttnn.typecast(value_user, self.kv_cache_dtype)
                key_fill = ttnn.typecast(key_user, self.kv_cache_dtype)
                # For batch=1 the full-user slice aliases the parent V/K
                # allocation, which SDPA still consumes below.
                ttnn.deallocate(value_user, False)
                ttnn.deallocate(key_user, False)
            else:
                value_fill = value_user
                key_fill = key_user
            ttnn.fill_cache(value_cache, value_fill, batch_idx=user_id)
            ttnn.fill_cache(key_cache, key_fill, batch_idx=user_id)
            ttnn.deallocate(value_fill, self.kv_cache_dtype != ttnn.bfloat16)
            ttnn.deallocate(key_fill, self.kv_cache_dtype != ttnn.bfloat16)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query_rotated,
            key_rotated,
            value,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.attention_compute_config,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=64,
                k_chunk_size=64,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(query_rotated, True)
        ttnn.deallocate(key_rotated, True)
        ttnn.deallocate(value, True)
        attention_heads = attention
        attention = ttnn.transformer.concatenate_heads(attention_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attention_heads, True)
        concatenated = attention
        attention = ttnn.reshape(concatenated, (m, self.hidden_size))
        ttnn.deallocate(concatenated, False)
        projected = ttnn.matmul(
            attention,
            self.o_weight,
            dtype=ttnn.bfloat16,
            program_config=o_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attention, True)
        projected_matmul = projected
        projected = ttnn.reshape(projected_matmul, (1, 1, m, self.hidden_size))
        ttnn.deallocate(projected_matmul, False)
        output = ttnn.add(residual, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(projected, True)
        return output

    def _decode_norm(self, residual, weight):
        norm_input = residual
        converted = False
        if norm_input.memory_config() != self.norm_memory_config:
            norm_input = ttnn.to_memory_config(residual, self.norm_memory_config)
            converted = True
        output = ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=weight,
            program_config=self.norm_program_config,
            compute_kernel_config=self.norm_compute_config,
            memory_config=self.norm_memory_config,
        )
        if converted:
            ttnn.deallocate(norm_input, True)
        return output

    def _move_owned(self, tensor, memory_config):
        """Move an owned tensor only when needed and retire the old allocation."""
        if tensor.memory_config() == memory_config:
            return tensor
        moved = ttnn.to_memory_config(tensor, memory_config)
        ttnn.deallocate(tensor, True)
        return moved

    def _decode_qkv(self, residual, *, position_index):
        normed = self._decode_norm(residual, self.input_norm_weight)
        qkv_input = self._move_owned(normed, self.qkv_input_memory_config)
        qkv_weight = self.qkv_decode_weight if self.qkv_decode_weight is not None else self.qkv_weight
        fused_qkv_sharded = ttnn.matmul(
            qkv_input,
            qkv_weight,
            dtype=ttnn.bfloat16,
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.qkv_output_memory_config,
        )
        ttnn.deallocate(qkv_input, True)
        fused_qkv = ttnn.sharded_to_interleaved(fused_qkv_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(fused_qkv_sharded, True)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused_qkv, True)
        cos_dram, sin_dram = self._rotary_slice(position_index, 1)
        cos = ttnn.to_memory_config(cos_dram, ttnn.L1_MEMORY_CONFIG)
        sin = ttnn.to_memory_config(sin_dram, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(cos_dram, False)
        ttnn.deallocate(sin_dram, False)
        key_rotated = ttnn.experimental.rotary_embedding(
            key, cos, sin, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        query_rotated = ttnn.experimental.rotary_embedding(
            query, cos, sin, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(key, True)
        ttnn.deallocate(query, True)
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        key_rotated = self._prepare_decode_heads(
            key_rotated, self.num_kv_heads, memory_config=self.decode_head_memory_config
        )
        query_rotated = self._prepare_decode_heads(query_rotated, self.num_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return query_rotated, key_rotated, value

    def _decode_attention(self, residual, *, key_cache, value_cache, cache_position, position_index):
        query_rotated, key_rotated, value = self._decode_qkv(residual, position_index=position_index)
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=cache_position, share_cache=False, page_table=None
        )
        ttnn.experimental.paged_update_cache(
            key_cache, key_rotated, update_idxs_tensor=cache_position, share_cache=False, page_table=None
        )
        ttnn.deallocate(value, True)
        ttnn.deallocate(key_rotated, True)
        if self.use_explicit_decode_mask:
            attention_mask = self._decode_attention_mask(cache_position)
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query_rotated,
                key_cache,
                value_cache,
                attn_mask=attention_mask,
                cur_pos_tensor=None,
                is_causal=False,
                scale=self.scale,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(attention_mask, True)
        else:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query_rotated,
                key_cache,
                value_cache,
                cur_pos_tensor=cache_position,
                is_causal=True,
                scale=self.scale,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(query_rotated, True)
        attention_sharded = ttnn.to_memory_config(attention, self.decode_head_memory_config)
        ttnn.deallocate(attention, True)
        concatenated = ttnn.experimental.nlp_concat_heads_decode(attention_sharded, num_heads=self.num_heads)
        ttnn.deallocate(attention_sharded, True)
        projected_input = ttnn.to_memory_config(
            concatenated,
            ttnn.L1_MEMORY_CONFIG if self.decode_matmul_mode == "shard_advisor" else self.residual_memory_config,
        )
        ttnn.deallocate(concatenated, True)
        projected = ttnn.matmul(
            projected_input,
            self.o_decode_weight if self.o_decode_weight is not None else self.o_weight,
            dtype=ttnn.bfloat16,
            program_config=self.o_decode_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=self.o_output_memory_config,
        )
        ttnn.deallocate(projected_input, True)
        projected_residual_layout = self._move_owned(projected, self.residual_memory_config)
        output = ttnn.add(residual, projected_residual_layout, memory_config=self.residual_memory_config)
        ttnn.deallocate(residual, True)
        ttnn.deallocate(projected_residual_layout, True)
        return output

    def _prefill_mlp_chunk(self, residual):
        m = int(residual.shape[-2])
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm_weight,
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
                    m,
                    self.hidden_size,
                    2 * self.intermediate_size,
                    grid_x_limit=self.prefill_grid_x,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=self.mlp_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, m, self.intermediate_size])
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.intermediate_size],
                [1, 1, m, 2 * self.intermediate_size],
            )
            ttnn.deallocate(packed, True)
        else:
            program_config = _prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.hidden_size,
                self.intermediate_size,
                grid_x_limit=self.prefill_grid_x,
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
        ttnn.deallocate(normed, True)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate, True)
        ttnn.deallocate(up, True)
        down = ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=_prefill_matmul_program_config(
                self.mesh_device,
                m,
                self.intermediate_size,
                self.hidden_size,
                grid_x_limit=self.prefill_grid_x,
                in0_block_w=self.prefill_in0_block_w,
            ),
            compute_kernel_config=self.mlp_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gated, True)
        output = ttnn.add(residual, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(residual, True)
        ttnn.deallocate(down, True)
        return output

    def _prefill_mlp(self, residual):
        """Run the dense MLP in bounded row chunks to keep large prefill CBs in L1."""
        rows = int(residual.shape[-2])
        max_chunk_rows = 1024
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
        ttnn.deallocate(residual, True)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _decode_mlp(self, residual):
        normed = self._decode_norm(residual, self.post_attention_norm_weight)
        if self.use_packed_mlp:
            packed_input = self._move_owned(normed, self.gate_input_memory_config)
            gate_up_weight = (
                self.gate_up_decode_weight if self.gate_up_decode_weight is not None else self.gate_up_weight
            )
            packed = ttnn.matmul(
                packed_input,
                gate_up_weight,
                dtype=ttnn.bfloat16,
                program_config=self.packed_gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.packed_mlp_output_memory_config,
            )
            ttnn.deallocate(packed_input, True)
            packed_dram = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(packed, True)
            gate = ttnn.slice(packed_dram, [0, 0, 0, 0], [1, 1, self.batch, self.intermediate_size])
            up = ttnn.slice(
                packed_dram,
                [0, 0, 0, self.intermediate_size],
                [1, 1, self.batch, 2 * self.intermediate_size],
            )
            ttnn.deallocate(packed_dram, True)
        else:
            mlp_input = self._move_owned(normed, self.gate_input_memory_config)
            gate_weight = self.gate_decode_weight if self.gate_decode_weight is not None else self.gate_weight
            up_weight = self.up_decode_weight if self.up_decode_weight is not None else self.up_weight
            gate = ttnn.matmul(
                mlp_input,
                gate_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_gate_output_memory_config,
            )
            up = ttnn.matmul(
                mlp_input,
                up_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_gate_output_memory_config,
            )
            ttnn.deallocate(mlp_input, True)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=(ttnn.DRAM_MEMORY_CONFIG if self.use_packed_mlp else self.mlp_gate_output_memory_config),
        )
        ttnn.deallocate(gate, True)
        ttnn.deallocate(up, True)
        down_input = self._move_owned(gated, self.down_input_memory_config)
        down = ttnn.matmul(
            down_input,
            self.down_decode_weight if self.down_decode_weight is not None else self.down_weight,
            dtype=ttnn.bfloat16,
            program_config=self.down_decode_program_config,
            compute_kernel_config=self.mlp_compute_config,
            memory_config=self.mlp_down_output_memory_config,
        )
        ttnn.deallocate(down_input, True)
        down_residual_layout = self._move_owned(down, self.residual_memory_config)
        output = ttnn.add(residual, down_residual_layout, memory_config=self.residual_memory_config)
        ttnn.deallocate(residual, True)
        ttnn.deallocate(down_residual_layout, True)
        return output

    def prefill_forward(self, hidden_states, *, key_cache, value_cache):
        batch, seq_len = self._validate_hidden(hidden_states, decode=False)
        self._validate_caches(key_cache, value_cache)
        residual = ttnn.reshape(hidden_states, (1, 1, batch * seq_len, self.hidden_size))
        residual = self._prefill_attention(
            residual,
            batch=batch,
            seq_len=seq_len,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        residual = self._prefill_mlp(residual)
        output = ttnn.reshape(residual, (1, batch, seq_len, self.hidden_size))
        ttnn.deallocate(residual, False)
        return output

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        cache_position,
        position_index,
    ):
        batch, _ = self._validate_hidden(hidden_states, decode=True)
        self._validate_caches(key_cache, value_cache)
        self._validate_cache_position(cache_position)
        if not 0 <= position_index < int(key_cache.shape[2]):
            raise ValueError(f"position_index={position_index} is outside the cache")
        residual = ttnn.reshape(hidden_states, (1, 1, batch, self.hidden_size))
        residual = ttnn.to_memory_config(residual, self.decode_input_memory_config)
        residual = self._decode_attention(
            residual,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            position_index=position_index,
        )
        residual = self._decode_mlp(residual)
        if int(residual.shape[-2]) != batch:
            padded_residual = residual
            residual = ttnn.slice(
                padded_residual,
                [0, 0, 0, 0],
                [1, 1, batch, self.hidden_size],
            )
            ttnn.deallocate(padded_residual, False)
        residual_interleaved = self._move_owned(residual, ttnn.L1_MEMORY_CONFIG)
        output_l1 = ttnn.reshape(residual_interleaved, (1, batch, 1, self.hidden_size))
        ttnn.deallocate(residual_interleaved, False)
        output = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(output_l1, True)
        return output
