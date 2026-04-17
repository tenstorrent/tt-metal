// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wormhole B0 variant: UNPACK/PACK instruction sequences match wormhole_b0/max_util_compute.cpp;
// TRISC_MATH follows tests/didt/max_util_workload/kernels/slow_cos_compute.cpp (SFPU cosine).
//
// Compile-time args:
//   0: l1_buffer0_addr
//   1: l1_buffer1_addr
//   2: l1_buffer2_addr
//   3: num_tiles
//   4: num_loops

#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/trigonometry.h"

#ifdef TRISC_UNPACK
ALWI void slow_cos_unpack(uint32_t num_loops, uint32_t num_tiles, uint32_t l1_buffer0_addr, uint32_t l1_buffer1_addr) {
    constexpr bool is_fp32_dest_acc_en = false;
    constexpr uint32_t face_r_dim = 16;
    constexpr uint32_t num_faces_A = 4;
    constexpr uint32_t num_faces_B = 4;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        (uint32_t)DataFormat::Float16_b,
        (uint32_t)DataFormat::Float16_b,
        (uint32_t)DataFormat::Float16_b,
        (uint32_t)DataFormat::Float16_b,
        face_r_dim,
        face_r_dim,
        num_faces_A,
        num_faces_B,
        128 /* tile size for float16_b >> 4 */,
        128 /* tile size for float16_b >> 4 */);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TT_SETADCXX(p_setadc::UNP_A, 1024 - 1, 0x0);

    volatile uint32_t* cfg = get_cfg_pointer();

    for (uint32_t i = 0; i < num_loops; i++) {
        for (uint32_t j = 0; j < num_tiles; j++) {
            uint32_t address_a = L1_ADDRESS(l1_buffer0_addr + j * 2048);

            wait_for_next_context(2);
            if (0 == unp_cfg_context) {
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            } else {
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            }
            semaphore_post(semaphore::UNPACK_SYNC);
            TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
            TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
            TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
            TTI_UNPACR(
                SrcA,
                0,
                0,
                0,
                0,
                1 /*Set OvrdThreadId*/,
                1 /*Set Dvalid*/,
                p_unpacr::RAREFYB_DISABLE,
                0,
                0 /* Set ContextIdInc */,
                0,
                0,
                1);

            t6_semaphore_get(semaphore::UNPACK_SYNC);
            switch_config_context(unp_cfg_context);
        }
    }
}
#endif  // TRISC_UNPACK

#ifdef TRISC_MATH
#include "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h"
ALWI void slow_cos_math(uint32_t num_loops, uint32_t num_tiles) {
    constexpr bool is_fp32_dest_acc_en = false;
    _llk_math_hw_configure_<is_fp32_dest_acc_en>((uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);
    ckernel_template tmp(1, 8, TT_OP_MOVA2D(0, 0, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD));
    tmp.program();
    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::cosine>();
    ckernel::sfpu::cosine_init<false>();

    for (uint32_t i = 0; i < num_loops; i++) {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (uint32_t j = 0; j < num_tiles; j++) {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(j);
            ckernel_template::run();

            _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(j);
            ckernel::sfpu::calculate_cosine<false, is_fp32_dest_acc_en, 32>();
            _llk_math_eltwise_unary_sfpu_done_();
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif  // TRISC_MATH

#ifdef TRISC_PACK
ALWI void slow_cos_pack(uint32_t num_loops, uint32_t num_tiles, uint32_t l1_buffer2_addr) {
    constexpr bool is_fp32_dest_acc_en = false;
    _llk_pack_hw_configure_<is_fp32_dest_acc_en>(
        (uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b, 128 /* tile size for float16_b >> 4 */);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 0, .cr = 0},
        .y_dst = {.incr = 0, .clr = 0, .cr = 0},
        .z_src = {.incr = 4, .clr = 0},
        .z_dst = {.incr = 4, .clr = 0},
    }
        .set(ADDR_MOD_0);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 0, .clr = 1},
        .z_dst = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);

    const std::uint32_t MOP_INNER_LOOP = 1;
    const std::uint32_t MOP_OUTER_LOOP = 1;
    ckernel::ckernel_template tmp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_PACR(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(4), 0, 1, 0, 1));
    tmp.set_last_outer_loop_instr(TT_OP_PACR(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(4), 0, 1, 0, 1));
    tmp.program();
    set_dst_write_addr(0);
    program_packer_destination(L1_ADDRESS(l1_buffer2_addr));

    for (uint32_t i = 0; i < num_loops; i++) {
        _llk_packer_wait_for_math_done_();
        for (uint32_t j = 0; j < num_tiles; j++) {
            mop_run(1, 1);
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif  // TRISC_PACK

void kernel_main() {
    constexpr uint32_t l1_buffer0_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_buffer1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_buffer2_addr = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_loops = get_compile_time_arg_val(4);

    UNPACK((slow_cos_unpack(num_loops, num_tiles, l1_buffer0_addr, l1_buffer1_addr)));
    MATH((slow_cos_math(num_loops, num_tiles)));
    PACK((slow_cos_pack(num_loops, num_tiles, l1_buffer2_addr)));
}
