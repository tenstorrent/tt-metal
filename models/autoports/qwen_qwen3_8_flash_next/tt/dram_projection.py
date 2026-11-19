# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Setup-selected TP4 DRAM projection groups for decoder experiments.

Weights and memory/program configs are prepared in construction. Runtime only
reshards the activation, executes local matmuls and restores the interleaved
BF16 consumer contract; it performs no host conversion or CCL. Each rank keeps
its own checkpoint partition. A flat policy selects Q1 only. Explicit ``rows``
selects Q2..32, including non-aligned logical counts. With both phases selected,
native weights are released and larger logical inputs are processed in internal
32-row chunks. Partial phase policies retain the native path for other shapes.
This phase choice is static shape policy, not exception fallback. Prepare every
used shape before trace capture.
"""

import math
from dataclasses import dataclass

import ttnn

from .fused_decoder import FusedPreciseProjection


@dataclass(frozen=True)
class DramProjectionConfig:
    cores: int = 8
    readers: int = 1
    block: int = 10
    dtype: str = "bfloat8_b"
    fidelity: str = "HiFi4"
    packed: bool = True
    activation_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.cores < 1 or self.readers not in (1, 2, 3) or self.block < 1:
            raise ValueError("Positive cores/block and readers in 1..3 required")
        if self.dtype not in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            raise ValueError("Unsupported projection dtype")
        if self.activation_dtype not in ("bfloat16", "bfloat8_b"):
            raise ValueError("Activation dtype must be bfloat16 or bfloat8_b")
        if self.fidelity not in ("LoFi", "HiFi2", "HiFi4"):
            raise ValueError("Unsupported projection fidelity")
        if type(self.packed) is not bool:
            raise ValueError("packed must be bool")


def phase_configs(config):
    """Flat mapping configures Q1; nested decode/rows mappings control both."""
    if isinstance(config, DramProjectionConfig):
        return config, None
    if not isinstance(config, dict):
        raise ValueError("Expected config dataclass or mapping")
    if "decode" in config or "rows" in config:
        if set(config) - {"decode", "rows"}:
            raise ValueError("Nested phase policy permits only decode and rows")

        def parse(value):
            return value if isinstance(value, DramProjectionConfig) or value is None else DramProjectionConfig(**value)

        return parse(config.get("decode")), parse(config.get("rows"))
    return DramProjectionConfig(**config), None


def weight_memory_plan(input_width, output_widths, config, *, native_dtype, native_packed, dram_banks=8):
    """Host arithmetic for per-rank tiled weight allocations (not measured peak).

    Includes per-bank shard padding and BFP exponent storage. Native interleaved
    pages are rounded equally over banks. The setup overlap is native + prepared
    persistent weights; upload staging, allocator/program metadata, activation,
    and kernel L1 scratch are separate reservations, so this is not a total
    device high-water measurement.
    """
    tile_bytes = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}
    if input_width % 32 or dram_banks < 1 or not output_widths or any(n < 1 for n in output_widths):
        raise ValueError("Expected tile-aligned K, positive widths and bank count")
    decode, rows = phase_configs(config)
    policies = [policy for policy in dict.fromkeys((decode, rows)) if policy is not None]
    native_columns = [sum(output_widths)] if native_packed else output_widths
    native = sum(
        math.ceil((input_width // 32) * math.ceil(n / 32) / dram_banks) * dram_banks * tile_bytes[native_dtype]
        for n in native_columns
    )
    weight_groups = {}
    for policy in policies:
        key = (policy.dtype, policy.packed)
        if key in weight_groups:
            continue
        widths = [math.ceil(n / 32) * 32 for n in output_widths]
        columns = [sum(widths)] if policy.packed else widths
        matrices = []
        for n in columns:
            per_bank_columns = math.ceil(n / (32 * dram_banks)) * 32
            per_bank_bytes = (input_width // 32) * (per_bank_columns // 32) * tile_bytes[policy.dtype]
            matrices.append(
                {
                    "logical_padded_columns": n,
                    "per_bank_columns": per_bank_columns,
                    "per_bank_bytes": per_bank_bytes,
                    "all_banks_bytes": per_bank_bytes * dram_banks,
                }
            )
        weight_groups[key] = {"dtype": policy.dtype, "packed": policy.packed, "matrices": matrices}
    candidate = sum(matrix["all_banks_bytes"] for group in weight_groups.values() for matrix in group["matrices"])
    retained_native = 0 if decode is not None and rows is not None else native
    return {
        "scope": "per-device persistent weight byte arithmetic, not measured total allocation high-water",
        "dram_banks": dram_banks,
        "native_bytes": native,
        "prepared_weight_groups": list(weight_groups.values()),
        "prepared_bytes": candidate,
        "retained_native_bytes": retained_native,
        "steady_weight_bytes": candidate + retained_native,
        "steady_delta_vs_native_bytes": candidate + retained_native - native,
        "construction_weight_overlap_bytes": native + candidate,
    }


class _PreparedGroup:
    def __init__(self, loader, parts, config, weight_cache=None):
        import torch

        self.config = config
        self.mesh = loader.mesh_device
        if tuple(self.mesh.shape) != (1, 4) or self.mesh.get_num_devices() != 4:
            raise ValueError("This projection policy requires the target 1x4 TP mesh")
        self.widths = [w.shape[0] for w in parts[0]]
        self.input_width = parts[0][0].shape[1]
        k = self.input_width
        if len(parts) != 4 or any(
            len(rank) != len(self.widths) or any(tuple(w.shape) != (width, k) for w, width in zip(rank, self.widths))
            for rank in parts
        ):
            raise ValueError("Expected four equal-shape rank partitions")
        if k % (32 * config.cores):
            raise ValueError("Input width must tile-divide over activation storage cores")
        shard_tiles = k // (32 * config.cores)
        if config.readers == 1 and (
            (k // 32) % config.block or (shard_tiles % config.block and config.block % shard_tiles)
        ):
            raise ValueError("Single-reader K block and activation shard must divide one another")
        grid_size = self.mesh.compute_with_storage_grid_size()
        grid = ttnn.num_cores_to_corerangeset(config.cores, grid_size, row_wise=True)
        self.padded_widths = [math.ceil(n / 32) * 32 for n in self.widths]
        self.compute = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, config.fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def l1(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (32, width), ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.activation_memory = l1(k // config.cores)
        rank_matrices = []
        for rank in parts:
            matrices = [
                torch.nn.functional.pad(w.T.contiguous(), (0, n - w.shape[0])) for w, n in zip(rank, self.padded_widths)
            ]
            rank_matrices.append([torch.cat(matrices, dim=-1)] if config.packed else matrices)
        banks = self.mesh.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        self.weights, self.specs = [], []
        weight_key = (config.dtype, config.packed)
        cached_weights = weight_cache.get(weight_key) if weight_cache is not None else None
        for index in range(len(rank_matrices[0])):
            matrices = [rank[index] for rank in rank_matrices]
            n = matrices[0].shape[-1]
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bank_grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.weights.append(
                cached_weights[index]
                if cached_weights is not None
                else ttnn.from_torch(
                    torch.stack(matrices).unsqueeze(1),
                    device=self.mesh,
                    dtype=getattr(ttnn, config.dtype),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                )
            )
            out_width = math.ceil(n / (32 * config.cores)) * 32
            self.specs.append(
                dict(
                    memory_config=l1(out_width),
                    program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=config.block,
                        per_core_M=1,
                        per_core_N=out_width // 32,
                        fused_activation=None,
                        num_workers_per_dram_bank=config.readers,
                        use_multicore_pipeline=config.readers >= 2,
                    ),
                )
            )
        if weight_cache is not None and cached_weights is None:
            # Reader count, input shard grid, K block and fidelity change the
            # programs, not the immutable per-bank weight geometry.
            weight_cache[weight_key] = self.weights

    def __call__(self, x):
        count = x.shape[-2]
        if tuple(x.shape) != (1, 1, count, self.input_width) or not 1 <= count <= 32:
            raise ValueError("Expected logical Q1..32 projection input")
        # Quantization is part of the timed device group, before activation
        # sharding. The public input and consumer output remain BF16.
        activation = ttnn.typecast(x, ttnn.bfloat8_b) if self.config.activation_dtype == "bfloat8_b" else x
        activation = ttnn.to_memory_config(activation, self.activation_memory)
        values = []
        for weight, spec in zip(self.weights, self.specs):
            value = ttnn.linear(activation, weight, dtype=ttnn.bfloat16, compute_kernel_config=self.compute, **spec)
            values.append(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG))
        if self.config.packed:
            result, offset = [], 0
            for width, padded in zip(self.widths, self.padded_widths):
                result.append(values[0][..., offset : offset + width])
                offset += padded
            return result
        return [value[..., :width] for value, width in zip(values, self.widths)]


class PhaseDramProjectionGroup:
    """Multiple local projections with explicit Q1/Q2..32 setup policies."""

    def __init__(self, loader, parts, config, native_forward=None):
        decode, rows = phase_configs(config)
        weight_cache = {}
        prepared = {
            policy: _PreparedGroup(loader, parts, policy, weight_cache)
            for policy in dict.fromkeys((decode, rows))
            if policy is not None
        }
        self.decode = prepared.get(decode)
        self.rows = prepared.get(rows)
        self.fully_selected = decode is not None and rows is not None
        self.native_forward = None if self.fully_selected else native_forward
        self.input_width = parts[0][0].shape[1]
        self.output_widths = [w.shape[0] for w in parts[0]]
        self.phase_policy = {"decode": decode, "rows": rows}

    def selected(self, x):
        count = x.shape[-2]
        return self.decode if count == 1 else self.rows if 2 <= count <= 32 else None

    def __call__(self, x):
        selected = self.selected(x)
        if selected is not None:
            return selected(x)
        if self.fully_selected and x.shape[-2] > 32:
            chunks = [self(x[..., start : min(start + 32, x.shape[-2]), :]) for start in range(0, x.shape[-2], 32)]
            return [ttnn.concat([chunk[i] for chunk in chunks], dim=2) for i in range(len(self.output_widths))]
        if self.native_forward is None:
            raise ValueError("Unselected phase requires its original projection callable")
        return self.native_forward(x)


class PhaseDramProjection(FusedPreciseProjection):
    """Subtype recognized by decoder_components.linear; returns one packed tensor.

    Internal per-projection tile padding is compacted before the existing
    consumer slices Z/A/B or Q/K/V. This extra compaction is part of integrated
    timing; standalone group numbers must not be substituted for that result.
    """

    def __init__(self, group, native_forward):
        self.group = group
        self.native_forward = None if group.fully_selected else native_forward
        self.input_width = group.input_width
        self.output_width = sum(group.output_widths)

    def __call__(self, x):
        if not self.group.fully_selected and self.group.selected(x) is None:
            return self.native_forward(x)
        values = self.group(x)
        return values[0] if len(values) == 1 else ttnn.concat(values, dim=-1)
