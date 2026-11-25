# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the GLM K-split expert prototype (generic_op programs and benches)."""

import os

import torch

import ttnn

KDIR = "models/demos/combine_lab/kernels"
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
PEAK_CYC_PER_TILE_LOFI = 16  # 2 * 32^3 FLOP / 4096 FLOP a cycle
# GLM-5.3 routed expert
GLM = dict(H=6144, I=2048, E=256, topk=8)
# the expert activation's named compile-time args (common/kernels/expert_activation.hpp): GLM-5.3 = Silu (kind 0)
ACT_ARGS = [
    ("expert_activation", 0),
    ("expert_activation_beta_bits", 0x3F800000),
    ("expert_activation_linear_beta_bits", 0x3F800000),
    ("expert_activation_limit_bits", 0x40E00000),
    ("expert_activation_alpha_bits", 0x3FD9DB23),
]


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


def l1_sharded(device, cores, shard_shape, dtype, layout, torch_dtype=torch.float32):
    """A HEIGHT_SHARDED L1 tensor of shard_shape a core on cores, zeroed from the host."""
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(cores), list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )
    shape = [len(cores) * shard_shape[0], shard_shape[1]]
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=layout, device=device, memory_config=mem
    )


def stamps_tensor(device, cores, words=64):
    """8 x words uint32 stamp words per core (the same L1 address on every core)."""
    return l1_sharded(device, cores, [8, words], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)


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
    if dtype == ttnn.float32:
        return t.float()
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel_err(a, b):
    return float((a.double() - b.double()).norm() / (b.double().norm() + 1e-30))


def aiclk_mhz(dev_id=0):
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{dev_id}/tt_aiclk") as f:
            return int(f.read().strip())
    except OSError:
        return -1


def tiles_to_rows(t, rows_t, cols_t):
    """[rows_t * 32, cols_t * 32] from a TILE tensor read back (already row-major from to_torch)."""
    return t.reshape(rows_t * 32, cols_t * 32)
