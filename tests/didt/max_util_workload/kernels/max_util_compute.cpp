// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul.h"
#include "api/compute/pack.h"

// Compute kernel for max-utilization workload.
// Uses pre-loaded L1 buffers directly - completely decoupled from data movement.
// No CB dependencies, no waits - runs compute at full speed.
//
// Compile-time args:
//   0: l1_buffer0_addr   – L1 address of pre-filled buffer 0 (float16_b)
//   1: l1_buffer1_addr  – L1 address of pre-filled buffer 1 (float16_b)
//   2: l1_buffer2_addr  – L1 output buffer for compute results (8 float16_b tiles)
//   3: num_tiles        – number of tiles to process per iteration (8)
//   4: num_iterations   – number of workload repetitions (stress-test loop)

#ifdef TRISC_UNPACK
ALWI void max_util_unpack(
    uint32_t num_iterations, uint32_t num_tiles, uint32_t l1_buffer0_addr, uint32_t l1_buffer1_addr) {
    // init
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

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // set address counters for z and w dimensions for all channels
    TT_SETADCXX(p_setadc::UNP_A, 1024 - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, 1024 - 1, 0x0);

    load_replay_buf(
        0,
        12,
        // Lambda function to set up replay buffer
        [] {
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
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
            TTI_NOP;
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
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
            TTI_NOP;
        });

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                    // src B
        false,                    // halo - just used for 4 unpacks
        lltt::replay_insn(0, 6),  // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(6, 6),  // runs when context is 1
        0,
        0);

    tmp.program();

    // compute loop
    volatile uint32_t* cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_tiles; j++) {
            uint32_t address_a = L1_ADDRESS(l1_buffer0_addr);
            uint32_t address_b = L1_ADDRESS(l1_buffer1_addr + j * 2048);

            // Wait for free context
            wait_for_next_context(2);

            // Validate and configure addresses
            _llk_unpack_configure_addresses_(address_a, address_b, cfg);

            semaphore_post(semaphore::UNPACK_SYNC);  // Trisc::SEMPOST for context acquire

            // Stall unpacker until pending CFG writes from Trisc have completed
            TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

            TTI_UNPACR(
                SrcB,
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
            TT_MOP(0, num_tiles - 1, unp_cfg_context == 0 ? 0 : 0xff);  // Run the MOP

            // T6::SEMGET for context release
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // Switch unpacker config context
            switch_config_context(unp_cfg_context);
        }
    }
}
#endif

#ifdef TRISC_MATH
ALWI void max_util_math(uint32_t num_iterations, uint32_t num_tiles) {
    // init
    constexpr bool is_fp32_dest_acc_en = false;
    constexpr uint32_t face_rc_dim = 16;
    constexpr uint32_t tile_rc_dim = 32;
    constexpr MathFidelity math_fidelity = MathFidelity::LoFi;
    constexpr int THROTTLE_LEVEL = 0;
    constexpr uint32_t num_faces_A = 4;
    constexpr uint32_t num_faces_B = 4;
    _llk_math_hw_configure_<is_fp32_dest_acc_en>((uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);
    addr_mod_t{
        //.srca = {.incr = srca_increment, .clr = 0, .cr = 0},
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 32, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        //.srca = {.incr = srca_set, .clr = 0, .cr = 1},
        .srcb = {.incr = 48, .clr = 0, .cr = 1},  // cr=32 before, cr+48=16 after wrapping
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        // .bias = {.incr = 1},
    }
        .set(ADDR_MOD_4);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 0},
    }
        .set(ADDR_MOD_5);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 0},
    }
        .set(ADDR_MOD_6);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        15,
        // Lambda function to load reply buffer
        [] {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0A0 // srca=srca, srcb+=8,  dest+=8
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_1,
                0);  // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_0,
                0);  // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_2,
                0);  // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B2A0 // srca=srca, srcb+=8,  dest+=8
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_1,
                0);  // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_0,
                0);  // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_4,
                0);  // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed

            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_0,
                0);  // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_1,
                0);  // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B1A3 // srca=srca, srcb+=8,  dest+=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);  // B1A3 // srca=32,   srcb=48,  dest+=8

            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_0,
                0);  // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(
                p_setrwc::CLR_NONE,
                0,
                ADDR_MOD_1,
                0);  // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B3A3 // srca=srca, srcb+=8,  dest+=8

            // TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase
            // (addr_mod_5), clear src A
        });
    ckernel_template tmp(
        8, 1, lltt::replay_insn(ckernel::math::replay_buf_offset, 15), TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0));
    tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_6, 0));
    tmp.program();

    math::reset_counters(p_setrwc::SET_ABD_F);

    // compute loop
    for (uint32_t i = 0; i < num_iterations; i++) {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (uint32_t j = 0; j < num_tiles; j++) {
            ckernel_template::run();
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

#ifdef TRISC_PACK
ALWI void max_util_pack(uint32_t num_iterations, uint32_t num_tiles, uint32_t l1_buffer2_addr) {
    // init
    constexpr bool is_fp32_dest_acc_en = false;
    _llk_pack_hw_configure_<is_fp32_dest_acc_en>(
        (uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b, 128 /* tile size for float16_b >> 4 */);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    addr_mod_pack_t{
        .y_src = {.incr = 4},
        .y_dst = {.incr = 4},
    }
        .set(ADDR_MOD_0);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 0, .clr = 1},
        .z_dst = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 4, .clr = 0, .cr = 0},
        .z_src = {.incr = 1, .clr = 0},
    }
        .set(ADDR_MOD_2);

    const std::uint32_t MOP_INNER_LOOP = 4;              // face_r_dim >> 2;
    const std::uint32_t MOP_OUTER_LOOP = 4 * num_tiles;  // num_faces * num_tiles;
    ckernel::ckernel_template tmp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            0,
            0,
            0));
    tmp.set_last_inner_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        0));
    tmp.set_last_outer_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        1));
    tmp.program();
    set_dst_write_addr(0);
    program_packer_destination(L1_ADDRESS(l1_buffer2_addr));

    // compute loop
    for (uint32_t i = 0; i < num_iterations; i++) {
        _llk_packer_wait_for_math_done_();
        for (uint32_t j = 0; j < num_tiles; j++) {
            ckernel::ckernel_template::run();
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

void kernel_main() {
    constexpr uint32_t l1_buffer0_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_buffer1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_buffer2_addr = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(4);

    // TRISC0: perform unpack A (float16_b) and unpack B (float16_b) to SRC_A and SRC_B
    UNPACK((max_util_unpack(num_iterations, num_tiles, l1_buffer0_addr, l1_buffer1_addr)));

    // TRISC1: perform MVMUL to DST
    MATH((max_util_math(num_iterations, num_tiles)));

    // TRISC2: perform pack to output L1 addr + SFPU programming
    PACK((max_util_pack(num_iterations, num_tiles, l1_buffer2_addr)));
}
