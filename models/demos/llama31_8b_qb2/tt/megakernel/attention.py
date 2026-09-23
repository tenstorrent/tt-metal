# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compose native paged SDPA with a BF16 head concatenation and ready flag.

Attention retains the native 32-core grouping and HiFi4/FP32 compute policy.
Its workers occupy rows 6..9, disjoint from projection, norm and fabric workers.
"""

from pathlib import Path

import ttnn

from .mlp import _grid


class FusedAttention:
    def __init__(self, layer, *, output=None, cores=None, compact_output=False):
        self.compact_output = compact_output
        self.mesh = layer.mesh_device
        if self.mesh.compute_with_storage_grid_size().y < 10:
            raise ValueError("Attention composition requires ten worker rows")
        self.cores = list(cores) if cores is not None else [ttnn.CoreCoord(x, y) for y in range(6, 10) for x in range(8)]
        if len(self.cores) != 32 or self.cores != sorted(self.cores, key=lambda c: (c.y, c.x)):
            raise ValueError("Attention requires32 workers in row-major native grouping order")
        self.grid = _grid(self.cores)
        self.compute = layer.decode_sdpa_compute
        self.config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            sub_core_grids=self.grid,
            q_chunk_size=32,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        self.heads = ttnn.empty(
            (1, 1, 8, 128),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if output is not None and (
            tuple(output.shape) != (1, 1, 1, 1024)
            or output.dtype != ttnn.bfloat16
            or output.memory_config() != layer.decode_inputs["o"]
        ):
            raise ValueError("Borrowed attention output must match the native O-input layout")
        self.output = (
            output
            if output is not None
            else ttnn.empty(
                (1, 1, 1, 1024),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=layer.decode_inputs["o"],
            )
        )

    def append(self, program, inputs, projection_cores, *, wait_for_kv=False):
        q, k, v, position, table = inputs
        if tuple(q.shape) != (1, 1, 8, 128) or q.dtype != ttnn.bfloat16:
            raise ValueError("Attention composition requires one BF16 query with eight heads")
        if not q.is_sharded() or q.memory_config().shard_spec.grid.num_cores() != 1:
            raise ValueError("The batch-one query must reside on exactly one core")
        native = ttnn._ttnn.operations.transformer._create_paged_sdpa_decode_descriptor(
            q, k, v, table, position, self.heads, self.config, self.compute
        )
        kernels = list(native.kernels)
        reader, writer, compute = kernels
        # Native sharded-Q readers use the output-core map as the Q source map.
        # The original Q stays on its original core; only attention workers move.
        q_core = ttnn.corerange_to_cores(q.memory_config().shard_spec.grid, row_wise=True)[0]
        q_physical = self.mesh.worker_core_from_logical_core(q_core)
        coordinator = self.mesh.worker_core_from_logical_core(self.cores[0])
        projections = [self.mesh.worker_core_from_logical_core(c) for c in projection_cores[:8]]
        reader_rt, writer_rt = reader.runtime_args, writer.runtime_args
        offsets = set()
        for core in self.cores:
            args = list(reader_rt[core.x][core.y])
            # Native read_q treats the output worker as the local Q owner.
            # Relocation separates those cores, so every reader must use the
            # explicit source coordinates, including the output worker.
            args[9] = 0
            args[20:22] = [q_physical.x, q_physical.y]
            reader_rt[core.x][core.y] = args
            args = list(writer_rt[core.x][core.y])
            offsets.add(len(args))
            writer_rt[core.x][core.y] = [
                *args,
                self.output.buffer_address(),
                coordinator.x,
                coordinator.y,
                *[value for c in projections for value in (c.x, c.y)],
            ]
        if len(offsets) != 1:
            raise ValueError("Unexpected native SDPA writer runtime argument layout")
        reader.runtime_args = reader_rt
        if wait_for_kv:
            reader.kernel_source = str(Path(__file__).with_name("kernels") / "attention_reader.cpp")
        writer.runtime_args = writer_rt
        writer.kernel_source = str(Path(__file__).with_name("kernels") / "attention_writer.cpp")
        writer.defines = [
            *writer.defines,
            ("COMPACT_ATTENTION_OUTPUT", str(int(self.compact_output))),
            ("CONCAT_CT_OFFSET", str(len(writer.compile_time_args))),
            ("CONCAT_RT_OFFSET", str(offsets.pop())),
        ]
        writer.compile_time_args = [
            *writer.compile_time_args,
            *ttnn.TensorAccessorArgs(self.output).get_compile_time_args(),
        ]
        program.kernels = [*program.kernels, reader, writer, compute]
        program.cbs = [
            *program.cbs,
            *native.cbs,
            ttnn.CBDescriptor(
                total_size=32 * 2048,
                core_ranges=_grid(self.cores[:1]),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=32, data_format=ttnn.bfloat16, page_size=2048)
                ],
            ),
        ]
        program.semaphores = [
            *program.semaphores,
            *native.semaphores,
            ttnn.SemaphoreDescriptor(id=3, core_ranges=self.grid, initial_value=0),
            ttnn.SemaphoreDescriptor(id=14, core_ranges=_grid(projection_cores), initial_value=0),
        ]
        if wait_for_kv:
            program.semaphores = [
                *program.semaphores,
                ttnn.SemaphoreDescriptor(id=4, core_ranges=self.grid, initial_value=0),
            ]
        return program
