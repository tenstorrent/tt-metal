# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the weight-streaming expert (ws_expert) generic_op programs and benches."""

import os

import torch
import ttnn

KDIR = "models/demos/ws_expert/kernels"
TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}
PEAK_CYC_PER_TILE_LOFI = 16  # 2 * 32^3 FLOP / 4096 FLOP a cycle


def kpath(name):
    return os.path.join(KDIR, name)


def crs(cores):
    """CoreRangeSet of single-core ranges for a list of (x, y)."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def cb(index, dtype, n_tiles, core_ranges, page=None):
    page = page or TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=n_tiles * page,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def stamps_tensor(device, cores, words=64):
    """An L1 HEIGHT_SHARDED uint32 [len(cores) * 8, words] tensor: 8 x words stamp words per core (same address)."""
    rows = 8
    grid = crs(cores)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [rows, words], ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = torch.zeros([len(cores) * rows, words], dtype=torch.int32)
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)


def read_stamps(st, n_cores, words=64):
    t = ttnn.to_torch(st).to(torch.int64) & 0xFFFFFFFF
    return t.reshape(n_cores, 8 * words)


def to_dev(t, dtype, device, mem=None):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG
    )


def quantize(t, dtype):
    """Round-trip t through a ttnn dtype on the host (what the device sees)."""
    if dtype == ttnn.bfloat16:
        return t.to(torch.bfloat16).float()
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def aiclk_mhz(dev_id=0):
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{dev_id}/tt_aiclk") as f:
            return int(f.read().strip())
    except OSError:
        return -1
