"""Hand-rolled dest-reuse run loop == the LLK's, plus candidate fixes for the lost ZEROACC.

V0 faithful copy of eltwise_binary_run_with_dest_reuse (control: must reproduce)
V1 + TTI_STALLWAIT(STALL_MATH, MATH) before the ZEROACC   (drain the FPU first)
V2 clear_zero_flags = 1 instead of 0
V3 ZEROACC issued twice
V4 TTI_ZEROACC (immediate operand, unrolled) instead of TT_ZEROACC (runtime instrn_buffer)
V9 stock mul_reuse_dest_tiles (reference)
"""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn

TILE = 32
CB_X, CB_G, CB_OUT = 0, 1, 9

KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

constexpr uint32_t cb_x = 0, cb_g = 1, cb_out = 9;
constexpr auto REUSE = ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t SLOT = get_compile_time_arg_val(1);
    constexpr uint32_t V = get_compile_time_arg_val(2);
    constexpr uint32_t LOCAL = SLOT & 7u;

    compute_kernel_hw_startup(cb_x, cb_g, cb_out);
    cb_wait_front(cb_x, S);
    cb_wait_front(cb_g, S);
    cb_reserve_back(cb_out, S);
    for (uint32_t i = 0; i < S; ++i) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, i, SLOT);
        mul_reuse_dest_init<REUSE>(cb_g);

        if constexpr (V == 9) {
            mul_reuse_dest_tiles<REUSE>(cb_g, i, SLOT);
        } else {
            // --- the two halves of detail::binary_reuse_dest_tiles, hand-rolled ---
            UNPACK((llk_unpack_A<ckernel::BroadcastType::NONE, true /*acc_to_dest*/, REUSE>(cb_g, i)));
            MATH((ckernel::math::set_dst_write_addr<
                      ckernel::DstTileShape::Tile32x32, ckernel::UnpackDestination::SrcRegs>(SLOT)));
#pragma GCC unroll 4
            for (uint32_t n = 0; n < 4; ++n) {
                MATH((eltwise_binary_reuse_dest_as_src<REUSE>()));
                if constexpr (V == 1) {
                    MATH(TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::MATH));
                }
                constexpr uint32_t ZF = (V == 2) ? 1u : 0u;
                if constexpr (V == 4) {
                    // immediate-operand ZEROACC, one per face, fully constant-folded
                    switch (n) {
                        case 0: MATH(TTI_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, ZF, ADDR_MOD_1, LOCAL * 4 + 0)); break;
                        case 1: MATH(TTI_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, ZF, ADDR_MOD_1, LOCAL * 4 + 1)); break;
                        case 2: MATH(TTI_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, ZF, ADDR_MOD_1, LOCAL * 4 + 2)); break;
                        default: MATH(TTI_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, ZF, ADDR_MOD_1, LOCAL * 4 + 3)); break;
                    }
                } else {
                    MATH(TT_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, ZF, ADDR_MOD_1,
                                     ckernel::math::get_dest_index_in_faces(LOCAL, n)));
                    if constexpr (V == 3) {
                        MATH(TT_ZEROACC(ckernel::p_zeroacc::CLR_16, 0, ZF, ADDR_MOD_1,
                                         ckernel::math::get_dest_index_in_faces(LOCAL, n)));
                    }
                }
                MATH((ckernel::ckernel_template::run()));
            }
            MATH((ckernel::math::clear_dst_reg_addr()));
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(SLOT, cb_out, i);
        tile_regs_release();
    }
    cb_push_back(cb_out, S);
}
"""

PUBLISH = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    cb_reserve_back(0, S); cb_push_back(0, S);
    cb_reserve_back(1, S); cb_push_back(1, S);
}
"""


def crs():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def shard(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape, core_grid=crs(), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


NAMES = {0: "V0 faithful", 1: "V1 +STALLWAIT(MATH)", 2: "V2 clear_zero_flags=1",
         3: "V3 ZEROACC twice", 4: "V4 TTI_ZEROACC imm", 9: "V9 stock helper"}


def run(device, s, slot, v):
    width = s * TILE
    torch.manual_seed(3)
    x = (torch.rand(TILE, width) * 2 - 1).to(torch.bfloat16)
    g = (torch.rand(TILE, width) + 0.5).to(torch.bfloat16)
    q = lambda t: t.to(torch.bfloat16).to(torch.float32)
    xf, gf = q(x), q(g)
    expected = xf * gf
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                    memory_config=shard((TILE, width)))
    xd, gd = dev(x), dev(g)
    od = ttnn.allocate_tensor_on_device(ttnn.Shape([TILE, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                        device, shard((TILE, width)))
    c = crs()
    pd = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(kernel_source=PUBLISH,
                                  source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=c, compile_time_args=[s],
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=KERNEL,
                                  source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=c, compile_time_args=[s, slot, v],
                                  config=ttnn.ComputeConfigDescriptor(
                                      math_fidelity=ttnn.MathFidelity.HiFi2,
                                      fp32_dest_acc_en=False, math_approx_mode=False)),
        ],
        semaphores=[],
        cbs=[ttnn.cb_descriptor_from_sharded_tensor(CB_X, xd),
             ttnn.cb_descriptor_from_sharded_tensor(CB_G, gd),
             ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, od)],
    )
    ttnn.generic_op([xd, gd, od], pd)
    ttnn.synchronize_device(device)
    a = ttnn.to_torch(od).to(torch.float32)
    rn = expected.pow(2).mean().sqrt().item() + 1e-9
    rel = (a - expected).abs().max().item() / rn
    aa, ee = a[:, :TILE], expected[:, :TILE]
    fl = [(aa[r*16:(r+1)*16, cc*16:(cc+1)*16] - ee[r*16:(r+1)*16, cc*16:(cc+1)*16]).abs().max().item()/rn
          for r in (0, 1) for cc in (0, 1)]
    print(f"  {'BAD ' if rel > 0.1 else 'ok  '}SLOT=D{slot:<2} {NAMES[v]:<22} rel={rel:.3f}  "
          f"faces=" + " ".join(f"{q:.2f}" for q in fl))
    for t in (xd, gd, od):
        ttnn.deallocate(t)


device = ttnn.open_device(device_id=0)
try:
    print("\n--- SLOT D7 (the broken slot): does any candidate fix it?")
    for v in (9, 0, 1, 2, 3, 4):
        run(device, 4, 7, v)
    print("\n--- SLOT D6 (control: must stay correct under every variant)")
    for v in (9, 0, 1, 2, 3, 4):
        run(device, 4, 6, v)
finally:
    ttnn.close_device(device)
