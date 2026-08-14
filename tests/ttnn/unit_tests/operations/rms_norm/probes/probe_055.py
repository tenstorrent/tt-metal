"""Where exactly does ZEROACC CLR_16 stop working, and does clear_mode 0 survive it?

Per trial: copy x into DEST slot 7, advance dst_rwc to RWC (Offset is 448, so the math DEST
pointer = 448 + RWC), then clear one face and pack.

MODE 0: one TT_ZEROACC(CLR_16, ..., where = 28 + i)              -> should zero slot 7 face i
MODE 1: 16x TT_ZEROACC(clear_mode 0 "1 row", ..., where = 0..15)  -> should zero the face AT the
        pointer (clear_mode 0 is pointer-relative: dst += dst_offset)
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
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t RWC = get_compile_time_arg_val(1);
    constexpr uint32_t MODE = get_compile_time_arg_val(2);
    constexpr uint32_t SLOT = 7;

    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    cb_wait_front(cb_x, N);
    cb_reserve_back(cb_out, N);

    for (uint32_t i = 0; i < N; ++i) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, i, SLOT);                  // Offset = 448, rwc = 0
        for (uint32_t r = 0; r < RWC; ++r) {
            MATH(TTI_INCRWC(0, 1, 0, 0));          // dst_rwc += 1 row
        }
        if constexpr (MODE == 0) {
            MATH(TT_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, 0, ADDR_MOD_1, 28 + i));
        } else {
#pragma GCC unroll 0
            for (uint32_t r = 0; r < 16; ++r) {
                MATH(TT_ZEROACC(ckernel::p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, r));
            }
        }
        MATH(ckernel::math::clear_dst_reg_addr());
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


def run(device, rwc, mode):
    N = 4
    width = N * TILE
    torch.manual_seed(5)
    x = (torch.rand(TILE, width) + 1.0).to(torch.bfloat16)
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
                                  core_ranges=c, compile_time_args=[N, rwc, mode],
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
    off = 448 + rwc
    zeroed = []
    for i in range(N):
        t = a[:, i * TILE:(i + 1) * TILE]
        zf = [k for k, z in enumerate(
            [bool((t[r * 16:(r + 1) * 16, cc * 16:(cc + 1) * 16] == 0).all())
             for r in (0, 1) for cc in (0, 1)]) if z]
        zeroed.append(zf)
    if mode == 0:
        ok = all(zeroed[i] == [i] for i in range(N))
        detail = " ".join(f"w{28+i}->{zeroed[i]}" for i in range(N))
    else:
        want = (rwc // 16) & 3
        ok = all(z == [want] for z in zeroed)
        detail = f"want face {want}; got " + " ".join(str(z) for z in zeroed)
    last_group = 496 <= off <= 511 or 1008 <= off <= 1023
    print(f"  mode={mode} rwc={rwc:<3} dst_offset={off:<4} "
          f"{'[LAST GROUP of half]' if last_group else '                    '} "
          f"{'ok  ' if ok else 'LOST'} {detail}")
    ttnn.deallocate(xd); ttnn.deallocate(od)


device = ttnn.open_device(device_id=0)
try:
    print("\n=== MODE 0: ZEROACC CLR_16, sweeping the math DEST pointer ===")
    for rwc in (33, 40, 44, 45, 46, 47):
        run(device, rwc, 0)
    pass
finally:
    ttnn.close_device(device)
