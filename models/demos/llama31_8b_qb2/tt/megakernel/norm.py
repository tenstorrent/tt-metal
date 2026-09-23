# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed eight-core native RMSNorm arithmetic for composition in the QB2 body."""

from pathlib import Path
import struct
import ttnn
from .mlp import _grid


class FusedNorm:
    def __init__(self, mesh, output_memory_config, epsilon, *, debug=False, cores=None, output=None, compact_output=False, tile_height=32, full_dst=False, stats_face=False):
        self.mesh = mesh
        self.compact_output = compact_output
        if tile_height not in (16, 32): raise ValueError("Norm supports16 or32 rows")
        self.tile_height = tile_height
        self.full_dst = full_dst
        self.stats_face = stats_face
        if stats_face and tile_height != 16: raise ValueError("Short statistics need16-row geometry")
        if debug and compact_output:
            raise ValueError("Compact norm output is a private projection transport")
        self.epsilon = struct.unpack("I", struct.pack("f", epsilon))[0]
        self.cores = list(cores) if cores is not None else [ttnn.CoreCoord(x, 5) for x in range(8)]
        self.grid = _grid(self.cores)
        self.debug = {}
        if debug:
            from .mlp import _width_memory

            for index in (0, 5, 7, 8, 11):
                self.debug[index] = ttnn.empty(
                    (1, 1, 32, 4096 if index in (0, 5) else 256),
                    dtype=ttnn.bfloat16 if index == 0 else ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=_width_memory(self.cores, 4096 if index in (0, 5) else 256),
                )
        self.output = (
            output
            if output is not None
            else ttnn.empty(
                (1, 1, 1, 4096),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=output_memory_config,
            )
        )

    def append(self, program, input_tensor, projection_cores=(), *, wait_for_gather=False, ready_semaphore=6):
        if tuple(input_tensor.shape) != (1, 1, 1, 4096) or input_tensor.dtype != ttnn.bfloat16:
            raise ValueError("Norm requires batch-one BF16 width4096")
        coords = [self.mesh.worker_core_from_logical_core(c) for c in self.cores]
        common = [v for c in coords for v in (c.x, c.y)]
        if projection_cores:
            c = self.mesh.worker_core_from_logical_core(projection_cores[0])
            common.extend([c.x, c.y])
        rt = ttnn.RuntimeArgs()
        for rank, c in enumerate(self.cores):
            rt[c.x][c.y] = [rank, input_tensor.buffer_address(), self.output.buffer_address(), self.epsilon, *common]
        ct = (
            ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.output).get_compile_time_args()
        )
        source = Path(__file__).with_name("kernels")
        kernels = list(program.kernels)
        for role, config in [("READER", ttnn.ReaderConfigDescriptor()), ("WRITER", ttnn.WriterConfigDescriptor())]:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(source / "norm_dataflow.cpp"),
                    core_ranges=self.grid,
                    compile_time_args=ct,
                    runtime_args=rt,
                    defines=[(role, "1"), ("NORM_STATS_FACE", str(int(self.stats_face))), ("NORM_READY_SEMAPHORE", str(ready_semaphore)), ("COMPACT_NORM_OUTPUT", str(int(self.compact_output))), ("TINY_NORM_M", str(int(self.tile_height == 16)))]
                    + ([("FUSE_NORM", "1")] if projection_cores else [])
                    + ([("FUSE_GATHER", "1")] if wait_for_gather else []),
                    config=config,
                )
            )
        for cores, defines in [(self.cores[:1], [("IS_ALLGATHER_WORKER", "1")]), (self.cores[1:], [])]:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(source / "norm_compute.cpp"),
                    core_ranges=_grid(cores),
                    defines=[*defines, ("NORM_SUBBLOCK", str(8 if self.full_dst else 4))],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=self.full_dst
                    ),
                )
            )
        cbs = list(program.cbs)
        for index, tiles, dtype in [
            (0, 16, ttnn.bfloat16),
            (2, 1, ttnn.bfloat16),
            (3, 1, ttnn.bfloat16),
            (4, 1, ttnn.float32),
            (5, 16, ttnn.float32),
            (6, 16, ttnn.float32),
            (7, 1, ttnn.float32),
            (8, 1, ttnn.float32),
            (9, 8, ttnn.float32),
            (10, 1, ttnn.float32),
            (11, 1, ttnn.float32),
            (16, 16, ttnn.bfloat16),
        ]:
            if index in self.debug:
                cbs.append(ttnn.cb_descriptor_from_sharded_tensor(index, self.debug[index]))
                continue
            size = ttnn.Tile([32, 32]).get_tile_size(dtype)
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=tiles * size,
                    core_ranges=self.grid,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size, tile=ttnn.TileDescriptor(self.tile_height, 32))],
                )
            )
        program.kernels = kernels
        program.cbs = cbs
        return program

    def __call__(self, input_tensor):
        program = ttnn.ProgramDescriptor(
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid, initial_value=0) for i in range(10)]
        )
        return ttnn.generic_op([input_tensor, *self.debug.values(), self.output], self.append(program, input_tensor))
