"""Minimal ZEROACC repro: does CLR_16 clear 16-row group `where` when the math DEST
pointer (DEST_TARGET_REG_CFG_MATH_Offset + dst_rwc) already sits inside that group?

Per tile i: copy x into DEST slot 7 (all 4 faces = x), advance dst_rwc to RWC via INCRWC,
issue ONE TT_ZEROACC(CLR_16, 0, 0, ADDR_MOD_1, 28 + i), reset the counter, pack slot 7.
A working clear zeroes exactly face (28+i)&3.

Both simulators model this as always clearing the addressed group.
"""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn

TILE = 32
CB_X, CB_OUT = 0, 9

KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common_globals.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ckernel_ops.h"
#include "ckernel_instr_params.h"

constexpr uint32_t cb_x = 0, cb_out = 9;

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);       // trials (= output tiles)
    constexpr uint32_t RWC = get_compile_time_arg_val(1);     // dst_rwc to establish, in rows
    constexpr uint32_t SLOT = 7;

    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    cb_wait_front(cb_x, N);
    cb_reserve_back(cb_out, N);

    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t W = 28 + i;   // slot 7, face i
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, i, SLOT);                      // sets Offset = 7*64 = 448, rwc = 0
        // advance dst_rwc to RWC rows (INCRWC dest field is 4 bits, so step by 12)
        for (uint32_t r = 0; r < RWC; r += 12) {
            MATH(TTI_INCRWC(0, 12, 0, 0));
        }
        MATH(TT_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, 0, ADDR_MOD_1, W));
        MATH(ckernel::math::clear_dst_reg_addr());     // put the counter back before packing
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(SLOT, cb_out, i);
        tile_regs_release();
    }
    cb_push_back(cb_out, N);
}
"""

PUBLISH = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    cb_reserve_back(0, N); cb_push_back(0, N);
}
"""


def crs():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def shard(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape, core_grid=crs(), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


def run(device, rwc):
    N = 4
    width = N * TILE
    torch.manual_seed(5)
    x = (torch.rand(TILE, width) + 1.0).to(torch.bfloat16)   # all >= 1 so 0 is unambiguous
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                         memory_config=shard((TILE, width)))
    od = ttnn.allocate_tensor_on_device(ttnn.Shape([TILE, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                        device, shard((TILE, width)))
    c = crs()
    pd = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(kernel_source=PUBLISH,
                                  source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=c, compile_time_args=[N],
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=KERNEL,
                                  source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=c, compile_time_args=[N, rwc],
                                  config=ttnn.ComputeConfigDescriptor(
                                      math_fidelity=ttnn.MathFidelity.HiFi2,
                                      fp32_dest_acc_en=False, math_approx_mode=False)),
        ],
        semaphores=[],
        cbs=[ttnn.cb_descriptor_from_sharded_tensor(CB_X, xd),
             ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, od)],
    )
    ttnn.generic_op([xd, od], pd)
    ttnn.synchronize_device(device)
    a = ttnn.to_torch(od).to(torch.float32)
    print(f"\n  dst_rwc = {rwc} rows  (Offset = 448, so dst_offset = {448 + rwc})")
    for i in range(N):
        W = 28 + i
        t = a[:, i * TILE:(i + 1) * TILE]
        zf = [bool((t[r * 16:(r + 1) * 16, cc * 16:(cc + 1) * 16] == 0).all())
              for r in (0, 1) for cc in (0, 1)]
        got = [k for k, z in enumerate(zf) if z]
        want = W & 3
        print(f"    where={W} (slot 7 face {want}) -> faces zeroed {got}   "
              f"{'ok' if got == [want] else 'CLEAR LOST' if got == [] else 'WRONG TARGET'}")
    ttnn.deallocate(xd); ttnn.deallocate(od)


device = ttnn.open_device(device_id=0)
try:
    for rwc in (0, 48):
        run(device, rwc)
finally:
    ttnn.close_device(device)
