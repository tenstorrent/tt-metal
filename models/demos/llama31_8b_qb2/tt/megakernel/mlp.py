# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-program local gate/up, SwiGLU and down projection with layer addresses.

This experimental body preserves the checkpoint's selected quantized weights.
It has eight projection workers and sixteen independent SFPU workers. The
same compiled body and scratch allocations are reused for every layer; a
device DRAM table selects weights. Optional four-chip reduction shares the
program. Optional stages include native RMSNorm, all-gather and the output
projection; attention remains outside this body.
"""

from pathlib import Path

import torch
import ttnn
from .tuning import ProjectionTuning


def _grid(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _width_memory(cores, width):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_grid(cores), [32, width // len(cores)], ttnn.ShardOrientation.ROW_MAJOR),
    )


class FusedMLP:
    """Own weights/address table/scratch through the lifetime of every trace.

    Calls are sequential. The returned output aliases reusable scratch and is
    valid only until the next invocation, matching the decoder workspace rule.
    """

    def __init__(
        self,
        layers,
        *,
        reuse_scratch=False,
        fuse_reduce=False,
        gu_workers=8,
        fuse_norm=False,
        fuse_gather=False,
        fuse_output=False,
        fuse_attention=False,
        fuse_prepare=False,
        tuning=None,
    ):
        if not layers or len(layers) > 32:
            raise ValueError("Provide one to 32 decoder layers")
        if gu_workers not in (8, 16):
            raise ValueError("gu_workers must be 8 or 16")
        self.tuning = tuning or ProjectionTuning()
        if self.tuning.compact_activations != "off" and not fuse_prepare:
            raise ValueError("Compact transport requires the complete decoder body")
        if self.tuning.split_gu_bank_rows and gu_workers != 16:
            raise ValueError("Split GU bank rows currently require sixteen compute workers")
        if reuse_scratch and (self.tuning.reader != "original" or self.tuning.buffers != 2):
            raise ValueError("Tuned readers require independent projection buffers")
        if (self.tuning.prefetch_gu_blocks or self.tuning.prefetch_down_blocks) and (
            not fuse_prepare or gu_workers != 8 or self.tuning.reader == "original"
        ):
            raise ValueError("Weight staging requires a complete GU8 loop with a tuned reader")
        if self.tuning.alias_projection_cbs and (not fuse_prepare or gu_workers != 8 or reuse_scratch):
            raise ValueError("Static projection aliasing requires the complete GU8 layer loop")
        self.gu_workers = gu_workers
        self.layers = tuple(layers)
        self.reuse_scratch = reuse_scratch
        self.fuse_reduce = fuse_reduce
        self.fuse_norm = fuse_norm
        self.fuse_gather = fuse_gather
        self.fuse_output = fuse_output
        self.fuse_prepare = fuse_prepare
        if fuse_prepare and not fuse_attention:
            raise ValueError("Complete layer composition requires attention composition")
        self.fuse_attention = fuse_attention
        if fuse_attention and not fuse_output:
            raise ValueError("Attention composition requires the full post-attention body")
        if fuse_output and not fuse_gather:
            raise ValueError("Output projection composition requires the gathered MLP tail")
        if fuse_gather and not (fuse_norm and fuse_reduce):
            raise ValueError("All-gather composition requires norm and reduction")
        self.gather_output = layers[0].decode_workspace.buffers["mlp"] if fuse_gather else None
        self.mesh = layers[0].mesh_device
        if self.mesh.arch() != ttnn.device.Arch.BLACKHOLE or self.mesh.dram_grid_size().x != 8:
            raise ValueError("Fused MLP requires Blackhole with eight DRAM banks")
        grid = self.mesh.compute_with_storage_grid_size()
        if grid.x < 8 or grid.y < (6 if fuse_norm else 5 if fuse_reduce else 4):
            raise ValueError("Fused MLP needs eight columns and four worker rows (five with reduction)")
        from .placement import ProjectionPlacement
        self.placement = ProjectionPlacement(self.mesh, self.tuning.projection_placement, gu_workers,
            ttnn.corerange_to_cores(layers[0].decode_inputs["gate_up"].shard_spec.grid, row_wise=True))
        self.projection_cores = self.placement.map(
            [ttnn.CoreCoord(x, y) for y in range(4 - gu_workers // 8, 4) for x in range(8)])
        self.sfpu_cores = self.placement.map(
            ttnn.corerange_to_cores(layers[0].decode_inputs["down"].shard_spec.grid, row_wise=True), row_major=True)
        if len(self.sfpu_cores) != 16 or any(c in self.sfpu_cores for c in self.projection_cores):
            raise ValueError("Projection and sixteen-core down-input grids must be disjoint")
        down_shard = layers[0].decode_inputs["down"].shard_spec
        if down_shard.shape != [32, 224] or down_shard.orientation != ttnn.ShardOrientation.ROW_MAJOR:
            raise ValueError("Expected sixteen row-major down-input shards of shape [32,224]")
        self.projection_grid = _grid(self.projection_cores)
        self.sfpu_grid = _grid(self.sfpu_cores)
        self.communication_cores = [ttnn.CoreCoord(x, 4) for x in range(2)] if fuse_reduce else []
        self.norm_cores = [ttnn.CoreCoord(x, 5) for x in range(8)] if fuse_norm else []
        self.all_grid = _grid(self.projection_cores + self.sfpu_cores + self.communication_cores + self.norm_cores)

        for layer in layers:
            policy = layer.precision_policy
            if (
                policy["accumulation"]["matmul_fp32"]
                or policy["accumulation"]["math_approx_mode"]
                or policy["activation_dtype"] != "bfloat16"
                or policy["activation_output_dtype"] != "bfloat16"
            ):
                raise ValueError(
                    "Fused MLP requires BF16 activations, BF16 destination accumulation and exact math mode"
                )
            if layer.decode_inputs["down"] != layers[0].decode_inputs["down"]:
                raise ValueError("All layers must share the same down-input layout")
            if layer.mesh_device is not self.mesh or layer.decode_workspace.batch != 1:
                raise ValueError("All layers must share this mesh and a batch-one decode workspace")
            roles = (
                ("gate_up", (4096, 7168), ttnn.bfloat4_b),
                ("down", (3584, 4096), ttnn.bfloat8_b),
            ) + ((("o", (1024, 4096), ttnn.bfloat8_b),) if fuse_output else ())
            if fuse_prepare:
                roles += (("qkv", (4096, 1536), ttnn.bfloat8_b),)
            for role, shape, dtype in roles:
                weight = layer.decode_weights[role]
                if tuple(weight.shape) != shape or weight.dtype != dtype:
                    raise ValueError(f"Unsupported {role} weight shape/precision")
                memory = weight.memory_config()
                if (
                    memory.buffer_type != ttnn.BufferType.DRAM
                    or memory.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
                    or memory.shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR
                    or memory.shard_spec.shape != [shape[0], shape[1] // 8]
                ):
                    raise ValueError(f"Unsupported {role} DRAM layout")
                if layer.precision_policy["compute_fidelities"]["decode"][role] != "LoFi":
                    raise ValueError("This body requires the selected LoFi projection policy")

        def empty(width, memory):
            return ttnn.empty(
                (1, 1, 1, width),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=memory,
            )

        if self.tuning.projection_placement == "dram":
            # Bank-ordered compute cores are not row-major tensor shards.
            # Writers scatter logical GU columns onto the SFPU storage grid.
            self.packed = empty(7168, _width_memory(self.sfpu_cores, 7168))
            self.product = empty(3584, _width_memory(self.sfpu_cores, 3584))
        else:
            self.packed = empty(7168, _width_memory(self.projection_cores, 7168))
            self.product = empty(3584, layers[0].decode_inputs["down"])
        # Complete decode replaces every native use of these workspace slots.
        # Reuse them so persistent decode storage leaves the native prefill
        # head's static CB region available at longer contexts.
        workspace = layers[0].decode_workspace.buffers
        self.output = workspace["attn"] if fuse_prepare else empty(4096, layers[0].residual_memcfg)
        self.normalizer = None
        if fuse_norm:
            from .norm import FusedNorm

            self.normalizer = FusedNorm(self.mesh, layers[0].decode_inputs["gate_up"], layers[0].eps, compact_output=self.tuning.compact_activations != "off")
        self.attention_stage = None
        if fuse_attention:
            from .attention import FusedAttention

            self.attention_stage = FusedAttention(layers[0], output=workspace["o"] if fuse_prepare else None,
                cores=self.placement.map([ttnn.CoreCoord(x, y) for y in range(6, 10) for x in range(8)], row_major=True), compact_output=self.tuning.compact_activations == "all")
        self.preparation = None
        if fuse_prepare:
            from .prepare import FusedPreparation

            self.preparation = FusedPreparation(self)
        self.reduction = None
        if fuse_reduce:
            from .reduce_scatter import CompactReduceScatter

            self.reduction = CompactReduceScatter(
                self.mesh, layers[0].local_residual_memcfg, output=workspace["down"] if fuse_prepare else None
            )
        self.scratch_storage = []
        self.scratch_by_cb = {}
        if reuse_scratch:
            # Separate CB views can have different page counts/formats while
            # sharing a backing allocation. A single multi-format CB would
            # force one total size, breaking the 224/112-page block wrap rules.
            for indices, per_core_bytes in (((0, 4), 16 * 2048), ((1, 3), 224 * 576), ((24, 25), 28 * 2048)):
                width = ((per_core_bytes + 4095) // 4096) * 32 * len(self.projection_cores)
                storage = ttnn.empty(
                    (1, 1, 32, width),
                    dtype=ttnn.uint32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=_width_memory(self.projection_cores, width),
                )
                self.scratch_storage.append(storage)
                for index in indices:
                    self.scratch_by_cb[index] = storage
            # A single weight block keeps persistent storage small enough for
            # the native prefill RMSNorm buffers. This sacrifices double-buffer
            # overlap and must be measured separately from the default body.
            # The down reader waits for phase 3. That release follows all
            # gate/up packs and every SFPU read, so the previous input, weight
            # and partial CB contents have no remaining consumers.

        self.gu_weights = [layer.decode_weights["gate_up"] for layer in layers]
        if self.tuning.split_gu_bank_rows:
            from .weight_layout import split_gu_bank_rows
            self.gu_weights = [split_gu_bank_rows(weight, self.mesh) for weight in self.gu_weights]
            ttnn.synchronize_device(self.mesh)

        # Each 128-byte row is an independently addressable DRAM page. The
        # reserved fields are zero, not uninitialized kernel arguments.
        addresses = torch.zeros((len(layers), 32), dtype=torch.int64)
        for i, layer in enumerate(layers):
            addresses[i, 0] = self.gu_weights[i].buffer_address()
            addresses[i, 1] = layer.decode_weights["down"].buffer_address()
            if fuse_output:
                addresses[i, 2] = layer.decode_weights["o"].buffer_address()
            if fuse_prepare:
                addresses[i, 3] = layer.decode_weights["qkv"].buffer_address()
        self.address_rows = addresses
        self.address_table = ttnn.from_torch(
            addresses,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def descriptor(self, normalized, layer_index, attention=None):
        if not 0 <= layer_index < len(self.layers):
            raise ValueError("Layer index outside the address table")
        if tuple(normalized.shape) != (1, 1, 1, 4096) or normalized.dtype != ttnn.bfloat16:
            raise ValueError("Expected batch-one BF16 normalized hidden state")
        if normalized.memory_config() != self.layers[0].decode_inputs["gate_up"]:
            raise ValueError("Expected the original eight-core normalized input layout")
        tensors = (
            normalized,
            self.gu_weights[0],
            self.layers[0].decode_weights["down"],
            self.packed,
            self.product,
            self.output,
            self.address_table,
        )
        if self.fuse_output:
            if tuple(attention.shape) != (1, 1, 1, 1024) or attention.dtype != ttnn.bfloat16:
                raise ValueError("Expected batch-one concatenated BF16 attention")
            tensors = (*tensors, self.layers[0].decode_weights["o"], attention)
        ct = []
        for tensor in tensors:
            ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        physical = [
            self.mesh.worker_core_from_logical_core(c)
            for c in self.projection_cores + self.sfpu_cores + self.communication_cores
        ]
        coord_args = [v for c in physical for v in (c.x, c.y)]
        prefetch_coordinates = []
        if self.tuning.prefetch_gu_blocks or self.tuning.prefetch_down_blocks:
            prefetch_coordinates = [v for core in self.norm_cores + self.preparation.norm_cores
                for c in [self.mesh.worker_core_from_logical_core(core)] for v in (c.x, c.y)]
        rt_projection, rt_sfpu = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        common = [
            normalized.buffer_address(),
            self.packed.buffer_address(),
            self.product.buffer_address(),
            self.output.buffer_address(),
            self.address_table.buffer_address(),
            layer_index,
        ]
        if self.fuse_output:
            common.append(attention.buffer_address())
        for bank, core in enumerate(self.projection_cores):
            rt_projection[core.x][core.y] = [bank, *common, *coord_args, *prefetch_coordinates]
        for rank, core in enumerate(self.sfpu_cores):
            rt_sfpu[core.x][core.y] = [rank, *common, *coord_args]
        source = str(Path(__file__).with_name("kernels") / "mlp.cpp")
        kernels = []
        for role, grid, rt in (
            ("PROJECTION", self.projection_grid, rt_projection),
            ("SWIGLU", self.sfpu_grid, rt_sfpu),
        ):
            for risc, config in (
                (
                    "READER",
                    ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                ),
                (
                    "WRITER",
                    ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                ),
                (
                    "COMPUTE",
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi if role == "PROJECTION" else ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                    ),
                ),
            ):
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=source,
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=grid,
                        compile_time_args=ct,
                        runtime_args=rt,
                        defines=[(role, "1"), (risc, "1"), ("GU_WORKERS", str(self.gu_workers)), *self.tuning.defines,
                                 ("PREFETCH_COORD_OFFSET", str(1 + len(common) + len(coord_args)))]
                        + ([("FUSE_REDUCE", "1")] if self.fuse_reduce else [])
                        + ([("FUSE_NORM", "1")] if self.fuse_norm else [])
                        + ([("FUSE_OUTPUT", "1")] if self.fuse_output else [])
                        + ([("FUSE_ATTENTION", "1")] if self.fuse_attention else []),
                        config=config,
                    )
                )

        cbs = []

        def cb(index, tiles, dtype, grid):
            page = ttnn.Tile([32, 32]).get_tile_size(dtype)
            formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
            if index in self.scratch_by_cb and grid == self.projection_grid:
                descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    index,
                    self.scratch_by_cb[index],
                    total_size=tiles * page,
                )
                descriptor.format_descriptors = formats
            else:
                descriptor = ttnn.CBDescriptor(total_size=tiles * page, core_ranges=grid, format_descriptors=formats)
            cbs.append(descriptor)

        for index, tiles, dtype in (
            (0, 8 * self.tuning.buffers, ttnn.bfloat16),
            (1, (224 if self.reuse_scratch else 224 * self.tuning.buffers) * 8 // self.gu_workers, ttnn.bfloat4_b),
            (3, 112 if self.reuse_scratch else 112 * self.tuning.buffers, ttnn.bfloat8_b),
            (4, 7 * self.tuning.buffers, ttnn.bfloat16),
            (17, 16, ttnn.bfloat16),
            (24, 224 // self.gu_workers, ttnn.bfloat16),
            (25, 16, ttnn.bfloat16),
        ):
            cb(index, tiles, dtype, self.projection_grid)
        if self.fuse_output:
            cb(6, 4 * self.tuning.buffers, ttnn.bfloat16, self.projection_grid)
            cb(7, 64 * self.tuning.buffers, ttnn.bfloat8_b, self.projection_grid)
        if self.tuning.share_qkv_workers:
            cb(8, 16 * self.tuning.buffers, ttnn.bfloat16, self.projection_grid)
            cb(9, 96 * self.tuning.buffers, ttnn.bfloat8_b, self.projection_grid)
            cb(26, 6, ttnn.bfloat16, self.projection_grid)
            cb(27, 6, ttnn.bfloat16, self.projection_grid)
        if self.tuning.alias_projection_cbs:
            # The dependent O/GU/down phases have disjoint lifetimes. Reserve
            # static storage once; the loop installs each index's exact ring
            # capacity before its first use. No persistent prefill allocation.
            groups = ((0, 4, 6, 8), (1, 3, 7, 9), (24, 25, 26)) if self.tuning.share_qkv_workers else ((0, 4, 6), (1, 3, 7), (24, 25))
            for group in groups:
                shared = [item for item in cbs if item.core_ranges == self.projection_grid
                          and item.format_descriptors[0].buffer_index in group]
                if len(shared) != len(group):
                    raise ValueError("Incomplete projection CB alias group")
                from math import lcm
                alignment = lcm(*(item.format_descriptors[0].page_size for item in shared))
                size = max(item.total_size for item in shared)
                cbs = [item for item in cbs if item not in shared]
                cbs.append(ttnn.CBDescriptor(total_size=((size + alignment - 1) // alignment) * alignment,
                    core_ranges=self.projection_grid,
                    format_descriptors=[item.format_descriptors[0] for item in shared]))
        if self.tuning.projection_placement == "dram":
            cb(16, 224 // self.gu_workers, ttnn.bfloat16, self.projection_grid)
        cb(31, 1, ttnn.uint32, self.projection_grid)
        for index in (0, 1, 2):
            cb(index, 7 if self.tuning.batch_swiglu else 4, ttnn.bfloat16, self.sfpu_grid)
        cbs.extend(
            (
                ttnn.cb_descriptor_from_sharded_tensor(16, self.packed),
                ttnn.cb_descriptor_from_sharded_tensor(18, self.product),
            )
        )
        if self.tuning.projection_tile_height == 16:
            for item in cbs:
                if item.core_ranges == self.projection_grid:
                    formats = list(item.format_descriptors)
                    for fmt in formats:
                        if fmt.buffer_index in (0, 4, 6, 16, 17, 24, 25):
                            fmt.tile = ttnn.TileDescriptor(16, 32)
                    item.format_descriptors = formats
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=i,
                core_ranges=(
                    self.all_grid
                    if i < 6 or (self.fuse_output and i in (11, 13))
                    else _grid(self.projection_cores + self.sfpu_cores + self.norm_cores)
                ),
                initial_value=0,
            )
            for i in range(
                14
                if self.fuse_output
                else 13 if self.fuse_gather else 10 if self.fuse_norm else 6 if self.fuse_reduce else 4
            )
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)

    def __call__(
        self,
        normalized,
        layer_index,
        residual=None,
        attention_inputs=None,
        cache_inputs=None,
        layer_loop=None,
        embedding_tokens=None,
    ):
        if residual is not None and self.reduction is None:
            raise ValueError("Residual fusion requires four-chip reduction")
        if self.preparation is not None:
            attention_inputs = (self.preparation.heads[0], *cache_inputs[:4])
        attention = self.attention_stage.output if self.fuse_attention else normalized if self.fuse_output else None
        norm_input = self.gather_output if self.fuse_gather else normalized
        if self.normalizer is not None:
            normalized = self.normalizer.output
        if self.reduction is None:
            descriptor = self.descriptor(normalized, layer_index, attention)
            if self.normalizer is not None:
                descriptor = self.normalizer.append(descriptor, norm_input, self.projection_cores)
        else:
            descriptor = ttnn.MeshProgramDescriptor()
            for rank in range(4):
                coord = ttnn.MeshCoordinate(0, rank)
                local = self.descriptor(normalized, layer_index, attention)
                if self.normalizer is not None:
                    local = self.normalizer.append(
                        local, norm_input, self.projection_cores, wait_for_gather=self.fuse_gather
                    )
                if self.attention_stage is not None:
                    local = self.attention_stage.append(
                        local, attention_inputs, self.projection_cores, wait_for_kv=self.fuse_prepare
                    )
                if self.preparation is not None:
                    local = self.preparation.append(local, layer_index, cache_inputs)
                local = self.reduction.append(
                    local,
                    self.output,
                    rank,
                    wait_for_mlp=True,
                    residual=residual,
                    gathered=self.gather_output,
                    norm_cores=self.norm_cores,
                    fuse_output=self.fuse_output,
                    pre_norm_cores=self.preparation.norm_cores if self.preparation is not None else (),
                )
                if layer_loop is not None:
                    local = layer_loop.append(local, embedding_tokens)
                descriptor[ttnn.MeshCoordinateRange(coord, coord)] = local
        # Keep every table-referenced weight resident; the returned scratch is
        # consumed by reduce-scatter before the next decoder invokes this body.
        io = [normalized, self.address_table, self.packed, self.product, *self.scratch_storage]
        if layer_loop is not None:
            io.extend(layer_loop.tensors())
            if embedding_tokens is not None:
                io.append(embedding_tokens)
        if self.preparation is not None:
            io.extend(self.preparation.tensors(cache_inputs))
        if self.attention_stage is not None:
            io.extend([*attention_inputs, self.attention_stage.heads])
        if self.fuse_output:
            io.append(attention)
            io.extend(layer.decode_weights["o"] for layer in self.layers)
        if self.normalizer is not None:
            io.append(norm_input)
        if residual is not None:
            io.append(residual)
        io.extend(self.gu_weights)
        io.extend(layer.decode_weights["down"] for layer in self.layers)
        if self.reduction is None:
            ttnn.generic_op([*io, self.output], descriptor)
            return self.output
        if layer_loop is not None and layer_loop.head is not None:
            return ttnn.generic_op([*io, self.output, self.reduction.output, layer_loop.head.output], descriptor)
        return ttnn.generic_op([*io, self.output, self.reduction.output], descriptor)
