# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal repro: FPU dest-reuse binary corrupts DEST slot 7 (Blackhole, 16-bit DEST).

    scripts/tt-probe.sh rms_norm < dest_reuse_slot7_minimal.py

Single core, sharded L1 input/output, no DRAM, no kernel_lib, no chain, no broadcast.

The kernel runs the SAME three instructions twice, differing only in the DEST slot:

    tile_regs_acquire()
    copy_tile(cb_a, 0, slot)                            # DEST[slot] = a = 1.0
    mul_reuse_dest_tiles<DEST_TO_SRCA>(cb_b, 0, slot)   # DEST[slot] = DEST[slot] * b
    pack_tile(slot, cb_out, <tile>)

a = b = 1.0 everywhere, so every output element must be 1.0.

  slot 0 -> all 1.0                     correct
  slot 7 -> 1.0 in faces 0,1,2
            2.0 in face 3 (rows 16-31, cols 16-31)

2.0 == a + a*b, i.e. the DEST clear that `mul_reuse_dest_tiles` issues before its
accumulating ELWMUL was dropped, so the product accumulated on top of the operand
still sitting in DEST.

Why slot 7 face 3: the LLK clears one face at a time and the math DEST pointer
(DEST_TARGET_REG_CFG_MATH_Offset + dst_rwc) advances 16 rows per face, so slot 7's four
clears are issued from DEST rows 448, 464, 480, 496. ZEROACC(CLR_16) is silently dropped
when that pointer is near the top of a DEST half, and 496 is the only one of the four that
is. Slots 0..6 never get there (slot 6 tops out at 480).

Requires fp32_dest_acc_en=False (16-bit DEST). With fp32_dest_acc_en=True the pointer
never reaches the dead zone and the bug does not appear.
"""

import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import torch
import ttnn

TILE = 32
CB_A, CB_B, CB_OUT = 0, 1, 16
SLOTS = (0, 7)  # one output tile per slot, same op, same data

COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;
constexpr auto REUSE = ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA;

void kernel_main() {
    constexpr uint32_t num_trials = get_compile_time_arg_val(0);
    constexpr uint32_t slot0 = get_compile_time_arg_val(1);
    constexpr uint32_t slot1 = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, num_trials);

    for (uint32_t t = 0; t < num_trials; ++t) {
        const uint32_t slot = (t == 0) ? slot0 : slot1;

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_a);
        copy_tile(cb_a, 0, slot);                       // DEST[slot] = a
        mul_reuse_dest_init<REUSE>(cb_b);
        mul_reuse_dest_tiles<REUSE>(cb_b, 0, slot);     // DEST[slot] = DEST[slot] * b
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(slot, cb_out, t);
        tile_regs_release();
    }

    cb_push_back(cb_out, num_trials);
}
"""

# The CBs are bound directly to the resident L1 shards, so nothing has to be moved -- this
# only makes the pages available to the compute kernel.
PUBLISH_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    cb_reserve_back(0, 1);
    cb_push_back(0, 1);
    cb_reserve_back(1, 1);
    cb_push_back(1, 1);
}
"""


def single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded(shape):
    """Whole tensor resident in one core's L1."""
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def build(device):
    a = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    b = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded(tuple(t.shape)),
        )

    a_dev, b_dev = to_dev(a), to_dev(b)
    out_shape = (TILE, TILE * len(SLOTS))
    out_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, sharded(out_shape)
    )

    crs = single_core()
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PUBLISH_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=[],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=[len(SLOTS), SLOTS[0], SLOTS[1]],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=False,  # 16-bit DEST -- required to reproduce
                    math_approx_mode=False,
                ),
            ),
        ],
        semaphores=[],
        cbs=[
            ttnn.cb_descriptor_from_sharded_tensor(CB_A, a_dev),
            ttnn.cb_descriptor_from_sharded_tensor(CB_B, b_dev),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_dev),
        ],
    )
    return a_dev, b_dev, out_dev, descriptor


def report(actual):
    print("\n  a = b = 1.0, so DEST[slot] = a * b = 1.0 is expected everywhere.\n")
    failed = False
    for t, slot in enumerate(SLOTS):
        tile = actual[:, t * TILE : (t + 1) * TILE]
        values = sorted(set(tile.flatten().tolist()))
        faces = {}
        for fr in (0, 1):
            for fc in (0, 1):
                face = tile[fr * 16 : (fr + 1) * 16, fc * 16 : (fc + 1) * 16]
                faces[fr * 2 + fc] = sorted(set(face.flatten().tolist()))
        bad = values != [1.0]
        failed = failed or bad
        print(f"  DEST slot {slot}: distinct values = {values}   {'<-- WRONG' if bad else 'correct'}")
        for f, vals in faces.items():
            rows = f"rows {(f // 2) * 16:>2}-{(f // 2) * 16 + 15:>2}"
            cols = f"cols {(f % 2) * 16:>2}-{(f % 2) * 16 + 15:>2}"
            print(f"      face {f} ({rows}, {cols}): {vals}")
        print()

    if failed:
        print("  REPRODUCED: the same three instructions are correct on DEST 0 and wrong on DEST 7.")
        print("  2.0 == a + a*b -> the pre-multiply DEST clear was dropped on that face.")
    else:
        print("  Not reproduced on this configuration.")
    return failed


def main():
    device = ttnn.open_device(device_id=0)
    try:
        a_dev, b_dev, out_dev, descriptor = build(device)
        ttnn.generic_op([a_dev, b_dev, out_dev], descriptor)
        ttnn.synchronize_device(device)
        actual = ttnn.to_torch(out_dev).to(torch.float32)
        reproduced = report(actual)
        for t in (a_dev, b_dev, out_dev):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    return reproduced


if __name__ == "__main__":
    main()
