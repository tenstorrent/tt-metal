# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import struct

import torch
import ttnn

READER = """
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    cb_reserve_back(0, num_tiles);
    cb_push_back(0, num_tiles);
    cb_reserve_back(1, num_tiles);
    cb_push_back(1, num_tiles);
}
"""

COMPUTE = """
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 1, 16);
    OP_INIT(0, 1, false);
    pack_reconfig_l1_acc(get_compile_time_arg_val(1));
    cb_wait_front(0, num_tiles);
    cb_wait_front(1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        OP_TILES(0, 1, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(16, 1);
        pack_tile(0, 16);
        cb_push_back(16, 1);
        tile_regs_release();
    }
}
"""

CORE = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

ADD_CASES = [(1.0, 2**-8), (1.0078125, 2**-8), (-1.0, -(2**-8)), (1.0, 3 * 2**-9)]
MUL_CASES = [(1.0078125, 1.5), (-1.0078125, 1.5)]
L1_ACC_CASES = [(1.0, 2**-8), (1.0078125, 2**-8), (-1.0, -(2**-8))]


def tiles(device, values, dtype):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.float32).repeat_interleave(32 * 32).reshape(-1, 32),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(CORE, (32 * len(values), 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


def run(device, op, a, b, out, fp32_dest_acc_en, l1_acc):
    io = [tiles(device, a, ttnn.bfloat16), tiles(device, b, ttnn.bfloat16), out]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=CORE,
            compile_time_args=[len(a)],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=CORE,
            compile_time_args=[len(a), l1_acc],
            defines=[("OP_INIT", f"{op}_init"), ("OP_TILES", f"{op}_tiles")],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en
            ),
        ),
    ]
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb, t) for cb, t in zip([0, 1, 16], io)]
    return ttnn.to_torch(ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)))


def fp32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def matching_modes(exact, observed):
    bits = fp32_bits(exact)
    modes = {
        "RNE": (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000,
        "ties-away": (bits + 0x8000) & 0xFFFF0000,
        "truncate": bits & 0xFFFF0000,
    }
    return "/".join(mode for mode, mode_bits in modes.items() if mode_bits == fp32_bits(observed)) or "none"


def test_bf16_rounding_probe(device, capsys):
    rows = []
    for op, cases, combine in (("add", ADD_CASES, float.__add__), ("mul", MUL_CASES, float.__mul__)):
        a, b = zip(*cases)
        out = run(device, op, a, b, tiles(device, [0.0] * len(a), ttnn.float32), False, 0)
        rows += [(f"{op} dest16 {x!r}, {y!r}", combine(x, y), tile) for x, y, tile in zip(a, b, out.split(32))]
    seed, addend = zip(*L1_ACC_CASES)
    out = run(device, "add", addend, [0.0] * len(seed), tiles(device, seed, ttnn.bfloat16), True, 1)
    rows += [(f"pack l1acc bf16 {s!r} + {d!r}", s + d, tile) for s, d, tile in zip(seed, addend, out.split(32))]

    with capsys.disabled():
        print(f"\n{'case':<40} {'exact':>14} {'observed':>14} {'bits':>10} {'uniform':>7}  modes")
        for name, exact, tile in rows:
            observed = tile[0, 0].float().item()
            print(
                f"{name:<40} {exact:>14.10g} {observed:>14.10g} {fp32_bits(observed):>#10x} "
                f"{str(bool((tile == tile[0, 0]).all())):>7}  {matching_modes(exact, observed)}"
            )
