# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host side of minimal_matmul's transposed-grid dataflow, for model-local matmuls with custom epilogues.

The fused matmuls here keep ttnn's in0 kernel (``dm_in0_sender.cpp``, unmodified) and replace
the in1 reader / output writer and the compute kernel. This module reproduces what
``minimal_matmul_program_factory.cpp`` sets up for them when M > N (the "transposed" grid):

* M is split over the grid's x axis, N over its y axis; every core gets the same number of
  tiles on each (``padded`` up), so the multicast handshakes see the same block counts.
* in0 (activations, M x K) is read by the y = 0 row and forwarded down each column;
  in1 (weights, K x N) is read by the x = 0 column and forwarded along each row.
* semaphores 0-2 are in0's (sender, receiver, valid), 3-5 in1's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import TILE_BYTES

KERNEL_ROOT = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels"
IN0_KERNEL = f"{KERNEL_ROOT}/dm_in0_sender.cpp"
IN0_NOC = ttnn.NOC.NOC_0
IN1_NOC = ttnn.NOC.NOC_1


@dataclass
class Grid:
    gx: int
    gy: int
    M: int  # tiles
    K: int
    N: int
    M_block: int
    K_block: int
    N_block: int
    padded_M: int = 0
    M_per_core: int = 0
    N_per_core: int = 0
    M_blocks_per_core: int = 0
    N_blocks_per_core: int = 0

    def __post_init__(self):
        self.padded_M = -(-self.M // self.gx) * self.gx
        self.M_per_core = self.padded_M // self.gx
        self.N_per_core = -(-self.N // self.gy)
        self.M_blocks_per_core = -(-self.M_per_core // self.M_block)
        self.N_blocks_per_core = -(-self.N_per_core // self.N_block)
        if self.K % self.K_block:
            raise ValueError(f"K block {self.K_block} must divide K {self.K} tiles")

    @property
    def K_blocks(self):
        return self.K // self.K_block

    def all_cores(self):
        return cores(0, 0, self.gx - 1, self.gy - 1)

    def m_range(self, x):
        return self.M_per_core * x, self.M_per_core * (x + 1)

    def n_range(self, y):
        return self.N_per_core * y, self.N_per_core * (y + 1)

    def defer_write_k_block(self, y):
        return min(y * math.ceil(self.K_blocks / self.gy), self.K_blocks - 1)


def cores(x0, y0, x1, y1) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def physical(device, x, y):
    return device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))


def chain(device, x, y, length, axis, noc):
    """minimal_matmul's build_core_order_for_axis: the forwarding order along one axis and this core's place."""
    values = [0] + (list(range(1, length)) if noc == ttnn.NOC.NOC_0 else list(range(length - 1, 0, -1)))
    cur = x if axis == "x" else y
    order = [physical(device, v if axis == "x" else x, y if axis == "x" else v) for v in values]
    i = values.index(cur)
    return order, i


def accessor_args(*tensors):
    args = []
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    return args


def cb(index, tiles, dtype, core_set, page_size=None):
    page = page_size or TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def matmul_semaphores(core_set, extra=0):
    vals = [0, 0, 1, 0, 0, 1] + [0] * extra
    return [ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=v) for i, v in enumerate(vals)]


def in0_kernels(device, g: Grid, x_tensor, out_tensor):
    """The two in0 KernelDescriptors (senders on y = 0, receivers elsewhere); ``out_tensor`` only
    fills the accessor slot the stock kernel expects (it does not write when in1 is the writer)."""
    x_bytes = TILE_BYTES[x_tensor.dtype]
    out_bytes = TILE_BYTES[out_tensor.dtype]
    prefix = [g.M, g.padded_M, g.K, g.K, g.N, g.gy * g.N_per_core, g.M_block, g.K_block, g.N_block]
    prefix += [g.M_blocks_per_core, g.N_blocks_per_core, x_bytes, out_bytes, out_bytes, 0, 1, 2, 0]
    send_ct = prefix + [1, 1, g.N, x_bytes] + accessor_args(x_tensor, out_tensor)
    recv_ct = prefix + [0, 1, g.N, x_bytes] + accessor_args(x_tensor, out_tensor)
    send_rt, recv_rt = [], []
    for x in range(g.gx):
        for y in range(g.gy):
            order, i = chain(device, x, y, g.gy, "y", IN0_NOC)
            prev, nxt = order[max(0, i - 1)], order[min(len(order) - 1, i + 1)]
            m0, m1 = g.m_range(x)
            n0, n1 = g.n_range(y)
            rt = [x_tensor.buffer_address(), 0, 0, int(i == len(order) - 1), nxt.x, nxt.y, prev.x, prev.y]
            rt += [m0, m1, n0, n1, g.defer_write_k_block(y), g.K_blocks - 1, out_tensor.buffer_address()]
            (send_rt if y == 0 else recv_rt).append(((x, y), rt))

    def kd(ct, rt, core_set):
        return ttnn.KernelDescriptor(
            kernel_source=IN0_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=IN0_NOC),
        )

    kernels = [kd(send_ct, send_rt, cores(0, 0, g.gx - 1, 0))]
    if g.gy > 1:
        kernels.append(kd(recv_ct, recv_rt, cores(0, 1, g.gx - 1, g.gy - 1)))
    return kernels


def in1_chain_args(device, g: Grid, x, y):
    """[is_sink, next x, next y, prev x, prev y] for core (x, y)'s place in its row's in1 chain."""
    order, i = chain(device, x, y, g.gx, "x", IN1_NOC)
    prev, nxt = order[max(0, i - 1)], order[min(len(order) - 1, i + 1)]
    return [int(i == len(order) - 1), nxt.x, nxt.y, prev.x, prev.y]


def in1_mcast_args(device, g: Grid, y):
    """[rect x0, y0, x1, y1, n_dests, sender x, sender y] for multicasting in1 from (0, y) to (1..gx-1, y)
    on NOC 1, whose multicast rectangle runs from the far end to the near one."""
    if g.gx < 2:
        s = physical(device, 0, y)
        return [s.x, s.y, s.x, s.y, 0, s.x, s.y]
    near, far = physical(device, 1, y), physical(device, g.gx - 1, y)
    s = physical(device, 0, y)
    return [far.x, far.y, near.x, near.y, g.gx - 1, s.x, s.y]
