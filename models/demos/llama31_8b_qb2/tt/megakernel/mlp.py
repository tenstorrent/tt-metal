# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-program local gate/up, SwiGLU and down projection with layer addresses.

This experimental body preserves the checkpoint's selected quantized weights.
It has eight projection workers and sixteen independent SFPU workers. The
same compiled body and scratch allocations are reused for every layer; a
device DRAM table selects weights. Optional four-chip reduction shares the
program; normalization and attention remain outside this body.
"""

from pathlib import Path

import torch
import ttnn


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

    def __init__(self, layers, *, reuse_scratch=False, fuse_reduce=False):
        if not layers or len(layers) > 32:
            raise ValueError("Provide one to 32 decoder layers")
        self.layers = tuple(layers)
        self.reuse_scratch = reuse_scratch
        self.fuse_reduce = fuse_reduce
        self.mesh = layers[0].mesh_device
        if self.mesh.arch() != ttnn.device.Arch.BLACKHOLE or self.mesh.dram_grid_size().x != 8:
            raise ValueError("Fused MLP requires Blackhole with eight DRAM banks")
        grid = self.mesh.compute_with_storage_grid_size()
        if grid.x < 8 or grid.y < (5 if fuse_reduce else 4):
            raise ValueError("Fused MLP needs eight columns and four worker rows (five with reduction)")
        self.projection_cores = [ttnn.CoreCoord(x, 3) for x in range(8)]
        self.sfpu_cores = ttnn.corerange_to_cores(layers[0].decode_inputs["down"].shard_spec.grid, row_wise=True)
        if len(self.sfpu_cores) != 16 or any(c in self.sfpu_cores for c in self.projection_cores):
            raise ValueError("Projection and sixteen-core down-input grids must be disjoint")
        down_shard = layers[0].decode_inputs["down"].shard_spec
        if down_shard.shape != [32, 224] or down_shard.orientation != ttnn.ShardOrientation.ROW_MAJOR:
            raise ValueError("Expected sixteen row-major down-input shards of shape [32,224]")
        self.projection_grid = _grid(self.projection_cores)
        self.sfpu_grid = _grid(self.sfpu_cores)
        self.communication_cores = [ttnn.CoreCoord(x, 4) for x in range(2)] if fuse_reduce else []
        self.all_grid = _grid(self.projection_cores + self.sfpu_cores + self.communication_cores)

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
            for role, shape, dtype in (
                ("gate_up", (4096, 7168), ttnn.bfloat4_b),
                ("down", (3584, 4096), ttnn.bfloat8_b),
            ):
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

        self.packed = empty(7168, _width_memory(self.projection_cores, 7168))
        self.product = empty(3584, layers[0].decode_inputs["down"])
        self.output = empty(4096, layers[0].residual_memcfg)
        self.reduction = None
        if fuse_reduce:
            from .reduce_scatter import CompactReduceScatter

            self.reduction = CompactReduceScatter(self.mesh, layers[0].local_residual_memcfg)
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

        # Each 128-byte row is an independently addressable DRAM page. The
        # reserved fields are zero, not uninitialized kernel arguments.
        addresses = torch.zeros((len(layers), 32), dtype=torch.int64)
        for i, layer in enumerate(layers):
            addresses[i, 0] = layer.decode_weights["gate_up"].buffer_address()
            addresses[i, 1] = layer.decode_weights["down"].buffer_address()
        self.address_table = ttnn.from_torch(
            addresses,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def descriptor(self, normalized, layer_index):
        if not 0 <= layer_index < len(self.layers):
            raise ValueError("Layer index outside the address table")
        if tuple(normalized.shape) != (1, 1, 1, 4096) or normalized.dtype != ttnn.bfloat16:
            raise ValueError("Expected batch-one BF16 normalized hidden state")
        if normalized.memory_config() != self.layers[0].decode_inputs["gate_up"]:
            raise ValueError("Expected the original eight-core normalized input layout")
        tensors = (
            normalized,
            self.layers[0].decode_weights["gate_up"],
            self.layers[0].decode_weights["down"],
            self.packed,
            self.product,
            self.output,
            self.address_table,
        )
        ct = []
        for tensor in tensors:
            ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        physical = [
            self.mesh.worker_core_from_logical_core(c)
            for c in self.projection_cores + self.sfpu_cores + self.communication_cores
        ]
        coord_args = [v for c in physical for v in (c.x, c.y)]
        rt_projection, rt_sfpu = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        common = [
            normalized.buffer_address(),
            self.packed.buffer_address(),
            self.product.buffer_address(),
            self.output.buffer_address(),
            self.address_table.buffer_address(),
            layer_index,
        ]
        for bank, core in enumerate(self.projection_cores):
            rt_projection[core.x][core.y] = [bank, *common, *coord_args]
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
                        defines=[(role, "1"), (risc, "1")] + ([("FUSE_REDUCE", "1")] if self.fuse_reduce else []),
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
            (0, 16, ttnn.bfloat16),
            (1, 224 if self.reuse_scratch else 448, ttnn.bfloat4_b),
            (3, 112 if self.reuse_scratch else 224, ttnn.bfloat8_b),
            (4, 14, ttnn.bfloat16),
            (17, 16, ttnn.bfloat16),
            (24, 28, ttnn.bfloat16),
            (25, 16, ttnn.bfloat16),
        ):
            cb(index, tiles, dtype, self.projection_grid)
        cb(31, 1, ttnn.uint32, self.projection_grid)
        for index in (0, 1, 2):
            cb(index, 4, ttnn.bfloat16, self.sfpu_grid)
        cbs.extend(
            (
                ttnn.cb_descriptor_from_sharded_tensor(16, self.packed),
                ttnn.cb_descriptor_from_sharded_tensor(18, self.product),
            )
        )
        semaphores = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_grid, initial_value=0)
            for i in range(6 if self.fuse_reduce else 4)
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)

    def __call__(self, normalized, layer_index):
        if self.reduction is None:
            descriptor = self.descriptor(normalized, layer_index)
        else:
            descriptor = ttnn.MeshProgramDescriptor()
            for rank in range(4):
                coord = ttnn.MeshCoordinate(0, rank)
                local = self.descriptor(normalized, layer_index)
                descriptor[ttnn.MeshCoordinateRange(coord, coord)] = self.reduction.append(
                    local,
                    self.output,
                    rank,
                    wait_for_mlp=True,
                )
        # Keep every table-referenced weight resident; the returned scratch is
        # consumed by reduce-scatter before the next decoder invokes this body.
        io = [normalized, self.address_table, self.packed, self.product, *self.scratch_storage]
        io.extend(w for layer in self.layers for w in (layer.decode_weights["gate_up"], layer.decode_weights["down"]))
        if self.reduction is None:
            ttnn.generic_op([*io, self.output], descriptor)
            return self.output
        return ttnn.generic_op([*io, self.output, self.reduction.output], descriptor)
