# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host side of minimal_matmul's dataflow, for model-local matmuls with custom epilogues.

The fused matmuls here keep ttnn's in0 kernel (``dm_in0_sender.cpp``, unmodified) and replace
the in1 reader / output writer and the compute kernel. This module reproduces what
``minimal_matmul_program_factory.cpp`` sets up for them:

* ``transposed`` (minimal_matmul's choice when M > N): M is split over the grid's x axis and N
  over its y axis; otherwise M over y and N over x. Every core gets the same number of tiles on
  each (``padded`` up), so the forwarding handshakes see the same block counts.
* in0 (activations, M x K) is read by the cores at N index 0 and forwarded along the N axis (the
  cores sharing those rows); in1 (weights, K x N) is read by the cores at M index 0 and forwarded
  along the M axis.
* semaphores 0-2 are in0's (sender, receiver, valid), 3-5 in1's.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import TILE_BYTES

KERNEL_ROOT = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels"
# Kernel sources resolve against the tt-metal root, but compiler include directories resolve against the
# process's working directory, so includes take the absolute path (servers do not run from the tt-metal root).
_TT_METAL_ROOT = os.environ.get("TT_METAL_HOME") or str(Path(__file__).resolve().parents[6])
KERNEL_INCLUDE = os.path.join(_TT_METAL_ROOT, KERNEL_ROOT)
IN0_KERNEL = f"{KERNEL_ROOT}/dm_in0_sender.cpp"
IN0_MCAST_KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "dm_in0_mcast.cpp")
IN0_NOC = ttnn.NOC.NOC_0
IN1_NOC = ttnn.NOC.NOC_1


def m_tiles_per_core(mt, n_m, min_block=4):
    """Tile rows per core along M for mt tile rows over n_m cores: ceil(mt / n_m), padded up when that count is
    above 8 and has no divisor from 4 to 8 (an op's M block must divide a core's rows, so it would fall to 1 or
    2 and re-read the weight block for every tile row), to the next count with a divisor of at least
    ``min_block``. E.g. 10 x 512 tokens: 160 tile rows over 13 cores is 13 per core (prime); the resident
    residual and its fused ops pad to 16 (M block 8), other matmuls to 14 (M block 7). Counts that already
    split into blocks of 4-8 (8 x 512: 10, 32 x 512: 40) are unchanged."""
    best = lambda m: max(d for d in range(1, 9) if m % d == 0)
    mpc = -(-mt // n_m)
    if mpc > 8 and best(mpc) < 4:
        while best(mpc) < min_block:
            mpc += 1
    return mpc


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
    transposed: bool = True
    m_min_block: int = 4  # see m_tiles_per_core
    padded_M: int = 0
    M_per_core: int = 0
    N_per_core: int = 0
    M_blocks_per_core: int = 0
    N_blocks_per_core: int = 0

    def __post_init__(self):
        self.n_m = self.gx if self.transposed else self.gy  # cores along the M axis
        self.n_n = self.gy if self.transposed else self.gx  # cores along the N axis
        self.padded_M = m_tiles_per_core(self.M, self.n_m, self.m_min_block) * self.n_m
        self.M_per_core = self.padded_M // self.n_m
        self.N_per_core = -(-self.N // self.n_n)
        self.M_blocks_per_core = -(-self.M_per_core // self.M_block)
        self.N_blocks_per_core = -(-self.N_per_core // self.N_block)
        if self.K % self.K_block:
            raise ValueError(f"K block {self.K_block} must divide K {self.K} tiles")

    @property
    def K_blocks(self):
        return self.K // self.K_block

    def all_cores(self):
        return cores(0, 0, self.gx - 1, self.gy - 1)

    def m_index(self, x, y):
        return x if self.transposed else y

    def n_index(self, x, y):
        return y if self.transposed else x

    def m_range(self, x, y):
        i = self.m_index(x, y)
        return self.M_per_core * i, self.M_per_core * (i + 1)

    def n_range(self, x, y):
        i = self.n_index(x, y)
        return self.N_per_core * i, self.N_per_core * (i + 1)

    def defer_write_k_block(self, x, y):
        return min(self.n_index(x, y) * math.ceil(self.K_blocks / self.n_n), self.K_blocks - 1)

    def in1_sender_cores(self):
        """The cores at M index 0 (they read in1 from DRAM)."""
        return cores(0, 0, 0, self.gy - 1) if self.transposed else cores(0, 0, self.gx - 1, 0)

    def in1_receiver_cores(self):
        return cores(1, 0, self.gx - 1, self.gy - 1) if self.transposed else cores(0, 1, self.gx - 1, self.gy - 1)

    def has_in1_receivers(self):
        return self.n_m > 1


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


def in0_kernels(device, g: Grid, x_tensor, out_tensor, kernel=IN0_KERNEL):
    """The two in0 KernelDescriptors (senders at N index 0, receivers elsewhere); ``out_tensor`` only
    fills the accessor slot the stock kernel expects (it does not write when in1 is the writer)."""
    x_bytes = TILE_BYTES[x_tensor.dtype]
    out_bytes = TILE_BYTES[out_tensor.dtype]
    prefix = [g.M, g.padded_M, g.K, g.K, g.N, g.n_n * g.N_per_core, g.M_block, g.K_block, g.N_block]
    prefix += [g.M_blocks_per_core, g.N_blocks_per_core, x_bytes, out_bytes, out_bytes, 0, 1, 2, 0]
    send_ct = prefix + [1, 1, g.N, x_bytes] + accessor_args(x_tensor, out_tensor)
    recv_ct = prefix + [0, 1, g.N, x_bytes] + accessor_args(x_tensor, out_tensor)
    axis = "y" if g.transposed else "x"
    send_rt, recv_rt = [], []
    for x in range(g.gx):
        for y in range(g.gy):
            order, i = chain(device, x, y, g.n_n, axis, IN0_NOC)
            prev, nxt = order[max(0, i - 1)], order[min(len(order) - 1, i + 1)]
            m0, m1 = g.m_range(x, y)
            n0, n1 = g.n_range(x, y)
            rt = [x_tensor.buffer_address(), 0, 0, int(i == len(order) - 1), nxt.x, nxt.y, prev.x, prev.y]
            rt += [m0, m1, n0, n1, g.defer_write_k_block(x, y), g.K_blocks - 1, out_tensor.buffer_address()]
            (send_rt if g.n_index(x, y) == 0 else recv_rt).append(((x, y), rt))

    def kd(ct, rt, core_set):
        return ttnn.KernelDescriptor(
            kernel_source=kernel,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=IN0_NOC),
            compiler_include_paths=[KERNEL_INCLUDE],
        )

    if g.transposed:
        senders, receivers = cores(0, 0, g.gx - 1, 0), cores(0, 1, g.gx - 1, g.gy - 1)
    else:
        senders, receivers = cores(0, 0, 0, g.gy - 1), cores(1, 0, g.gx - 1, g.gy - 1)
    kernels = [kd(send_ct, send_rt, senders)]
    if g.n_n > 1:
        kernels.append(kd(recv_ct, recv_rt, receivers))
    return kernels


def in0_mcast_kernels(device, g: Grid, x_tensor):
    """The two in0 KernelDescriptors of ``dm_in0_mcast.cpp`` (transposed grid): the core at N index 0
    (y = 0) of each column reads in0 and multicasts each block to the rest of its column."""
    if not g.transposed:
        raise ValueError("in0 multicast is implemented for the transposed grid only")
    x_bytes = TILE_BYTES[x_tensor.dtype]
    prefix = [g.M, g.padded_M, g.K, g.K, g.M_block, g.K_block, g.M_blocks_per_core, g.N_blocks_per_core, x_bytes, 0, 1]
    send_ct = prefix + [1] + accessor_args(x_tensor)
    recv_ct = prefix + [0] + accessor_args(x_tensor)
    send_rt, recv_rt = [], []
    for x in range(g.gx):
        s = physical(device, x, 0)
        near, far = physical(device, x, 1 if g.gy > 1 else 0), physical(device, x, g.gy - 1)
        for y in range(g.gy):
            m0, m1 = g.m_range(x, y)
            rt = [x_tensor.buffer_address(), m0, m1, near.x, near.y, far.x, far.y, g.gy - 1, s.x, s.y]
            (send_rt if y == 0 else recv_rt).append(((x, y), rt))

    def kd(ct, rt, core_set):
        return ttnn.KernelDescriptor(
            kernel_source=IN0_MCAST_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=IN0_NOC),
            compiler_include_paths=[KERNEL_INCLUDE],
        )

    kernels = [kd(send_ct, send_rt, cores(0, 0, g.gx - 1, 0))]
    if g.gy > 1:
        kernels.append(kd(recv_ct, recv_rt, cores(0, 1, g.gx - 1, g.gy - 1)))
    return kernels


def in1_chain_args(device, g: Grid, x, y):
    """[is_sink, next x, next y, prev x, prev y] for core (x, y)'s place in its in1 chain (along M)."""
    order, i = chain(device, x, y, g.n_m, "x" if g.transposed else "y", IN1_NOC)
    prev, nxt = order[max(0, i - 1)], order[min(len(order) - 1, i + 1)]
    return [int(i == len(order) - 1), nxt.x, nxt.y, prev.x, prev.y]


def in1_mcast_args(device, g: Grid, y):
    """[rect x0, y0, x1, y1, n_dests, sender x, sender y] for multicasting in1 from (0, y) to (1..gx-1, y)
    on NOC 1, whose multicast rectangle runs from the far end to the near one (transposed grid only)."""
    if g.gx < 2 or not g.transposed:
        s = physical(device, 0, y)
        return [s.x, s.y, s.x, s.y, 0, s.x, s.y]
    near, far = physical(device, 1, y), physical(device, g.gx - 1, y)
    s = physical(device, 0, y)
    return [far.x, far.y, near.x, near.y, g.gx - 1, s.x, s.y]
