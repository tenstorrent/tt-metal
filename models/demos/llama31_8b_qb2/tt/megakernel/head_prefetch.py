# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Use terminal-head readers and their idle weight buffers for layer look-ahead."""
from pathlib import Path
import ttnn
from .mlp import _grid


class HeadPrefetch:
    def __init__(self, loop):
        self.loop = loop
        self.body = loop.body
        self.head = loop.head
        self.qkv = self.body.tuning.head_prefetch_targets in ("both", "qkv")
        self.o = self.body.tuning.head_prefetch_targets in ("both", "o")
        self.consumers = self.body.preparation.projection_cores + self.body.projection_cores
        if len(self.consumers) != 16 or len(self.head.cores) != 16:
            raise ValueError("Head staging expects eight QKV, eight O and sixteen helper workers")
        grid = _grid(self.consumers + self.head.cores)
        # Per-core fields: last produced sense, staging pointer, last consumed
        # sense. These persist across invocations. Every handshake toggles;
        # there is no growing arrival count or reset racing a peer publication.
        self.semaphores = [ttnn.create_global_semaphore(self.body.mesh, grid, 0, ttnn.BufferType.L1_SMALL)
                           for _ in range(3)]
        self.addresses = [ttnn.get_global_semaphore_address(s) for s in self.semaphores]

    def coordinates(self, cores):
        return [v for core in cores for c in [self.body.mesh.worker_core_from_logical_core(core)] for v in (c.x, c.y)]

    def append_consumers(self, program):
        kernels = list(program.kernels)
        found = 0
        for kernel in kernels:
            name, defines = Path(kernel.kernel_source).name, dict(kernel.defines)
            if "READER" not in defines:
                continue
            if name == "qkv.cpp" and self.qkv:
                cores, helpers = self.consumers[:8], self.head.cores[:8]
            elif name == "mlp.cpp" and "PROJECTION" in defines and self.o:
                cores, helpers = self.consumers[8:], self.head.cores[8:]
            else:
                continue
            found += 1
            rt, offsets = kernel.runtime_args, set()
            extra = [*self.addresses, *self.coordinates(helpers)]
            for core in cores:
                args = list(rt[core.x][core.y]); offsets.add(len(args))
                rt[core.x][core.y] = [*args, *extra]
            assert len(offsets) == 1
            kernel.runtime_args = rt
            kernel.defines = [*kernel.defines, ("HEAD_PREFETCH_RECEIVER", "1"),
                              ("HEAD_PREFETCH_CONSUMER_RT", str(offsets.pop()))]
        assert found == int(self.qkv) + int(self.o)
        program.kernels = kernels
        return program

    def append_helpers(self, program):
        kernels = list(program.kernels)
        found = 0
        extra = [self.body.address_table.buffer_address(), self.loop.first, self.loop.count,
                 *self.addresses, *self.coordinates(self.consumers)]
        for kernel in kernels:
            if Path(kernel.kernel_source).name != "head.cpp" or "READER" not in dict(kernel.defines):
                continue
            found += 1
            rt, offsets = kernel.runtime_args, set()
            for core in self.head.cores:
                args = list(rt[core.x][core.y]); offsets.add(len(args))
                rt[core.x][core.y] = [*args, *extra]
            assert len(offsets) == 1
            kernel.runtime_args = rt
            kernel.defines = [*kernel.defines, ("HEAD_PREFETCH_HELPER", "1"),
                              ("HEAD_PREFETCH_HELPER_RT", str(offsets.pop())),
                              ("HEAD_PREFETCH_HELPER_CT", str(len(kernel.compile_time_args))),
                              ("HEAD_PREFETCH_QKV", str(int(self.qkv))),
                              ("HEAD_PREFETCH_O", str(int(self.o)))]
            kernel.compile_time_args = [*kernel.compile_time_args,
                *[v for tensor in (self.body.address_table, self.body.layers[0].decode_weights["qkv"],
                                   self.body.layers[0].decode_weights["o"])
                  for v in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]]
        assert found == 1
        program.kernels = kernels
        program.cbs = [*program.cbs, ttnn.CBDescriptor(total_size=4096, core_ranges=self.head.grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=4096)])]
        return program
