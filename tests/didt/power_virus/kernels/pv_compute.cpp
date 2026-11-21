// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h"
#include "pv_common.hpp"

// Power-virus TRISC kernel: the MaxUtil MVMUL loop (tests/didt/max_util_workload)
// running from pre-filled L1 buffers with no CB dependencies, until the host
// sets the STOP word in the core's control block.
//
// Stopping: UNPACK checks STOP every kCheckLoops loops. When it is set, UNPACK
// publishes FINAL = (current loop + kCheckLoops) and runs to it. MATH and PACK
// read FINAL at the same loop boundaries. They can never be past UNPACK, and by
// the time they reach loop FINAL, UNPACK wrote it before producing the data
// they consumed, so all three stop at the same loop without deadlock.
//
// Throttle (see pv_common.hpp). Smooth mode pads every tile in MATH with a
// fixed number of NOPs. Phase, burst and the duty cycle gate UNPACK; MATH and
// PACK stall behind it. Burst counts loops of num_tiles tiles.
//
// Determinism check (check_det == 1): UNPACK alternates between two A-input
// buffers (in0_a on even loops, in0_b on odd loops) so output changes every
// loop. PACK waits for each pack to land in L1 (hidden in MATH's shadow while
// MATH computes the other DST half), captures golden_a/golden_b across a
// rolling 64-word slice over the first 128 loops, then compares the rolling
// slice against golden_a/golden_b on every subsequent loop.
//
// Compile-time args:
//   0: l1_in0_a_addr    – pre-filled input A0 (8 float16_b tiles)
//   1: l1_in0_b_addr    – pre-filled input A1 (8 float16_b tiles)
//   2: l1_in1_addr      – pre-filled input B (8 float16_b tiles)
//   3: l1_out_addr      – output (8 float16_b tiles)
//   4: l1_golden_a_addr – golden output for A0 (8 float16_b tiles)
//   5: l1_golden_b_addr – golden output for A1 (8 float16_b tiles)
//   6: num_tiles        – tiles per loop (8)
//   7: ctrl_base        – L1 control block (pv_common.hpp)
//   8: super_sync       – wait for GO from the BRISC barrier
//   9: mode             – pv::kMode*
//  10: a                – smooth: NOPs per tile; phase: period; burst: loops
//  11: b                – phase: active units; burst: idle units
//  12: duty_period      – 0 = no duty cycle
//  13: duty_active
//  14: check_det        – 1 = continuous determinism check

constexpr uint32_t kCheckLoops = 16;
constexpr uint32_t kDetSliceWords = 64;

ALWI volatile tt_l1_ptr uint32_t* ctrl_word(uint32_t ctrl_base, uint32_t offset) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_base + offset);
}

// MATH and PACK: true when this loop boundary is the agreed final loop.
ALWI bool reached_final(uint32_t i, volatile tt_l1_ptr uint32_t* final_ptr) {
    if ((i & (kCheckLoops - 1)) != 0) {
        return false;
    }
    invalidate_l1_cache();
    return *final_ptr == i;
}

#ifdef TRISC_UNPACK
// Only the low wall-clock word is read: reading WALL_CLOCK_L latches
// WALL_CLOCK_H, so 64-bit reads here could disturb other RISCs' reads.
ALWI uint32_t wall_clock_l() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

template <
    uint32_t num_tiles,
    bool super_sync,
    uint32_t mode,
    uint32_t a,
    uint32_t b,
    uint32_t duty_period,
    uint32_t duty_active,
    bool check_det>
ALWI void pv_unpack(
    uint32_t l1_in0_a_addr, uint32_t l1_in0_b_addr, uint32_t l1_in1_addr, uint32_t ctrl_base) {
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

    // Clear unpacker transpose and stochastic rounding that a prior workload may have left set.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    _llk_unpack_configure_stoch_rnd_<StochRndType::None>();

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

    volatile tt_l1_ptr uint32_t* stop_ptr = ctrl_word(ctrl_base, pv::kStop);
    volatile tt_l1_ptr uint32_t* final_ptr = ctrl_word(ctrl_base, pv::kFinal);
    volatile tt_l1_ptr uint32_t* go_ptr = ctrl_word(ctrl_base, pv::kGo);
    if constexpr (super_sync) {
        *ctrl_word(ctrl_base, pv::kReady) = 1;
        do {
            invalidate_l1_cache();
        } while (*go_ptr != 1);
    }

    uint32_t prev_clk = wall_clock_l();
    uint64_t elapsed = 0;
    uint64_t loops = 0;
    // Smooth mode is handled in MATH, so UNPACK only sees the clock gates.
    using T = pv::Throttle<mode == pv::kModeSmooth ? pv::kModeFull : mode, a, b, duty_period, duty_active>;
    uint32_t burst_left = a;
    bool final_set = false;

    for (uint32_t i = 0;; i++) {
        if ((i & (kCheckLoops - 1)) == 0) {
            invalidate_l1_cache();
            if (final_set && *final_ptr == i) {
                break;
            }
            if (!final_set && *stop_ptr != 0) {
                *final_ptr = i + kCheckLoops;
                asm volatile("fence" ::: "memory");
                final_set = true;
            }
            const uint32_t now = wall_clock_l();
            elapsed += now - prev_clk;
            prev_clk = now;
        }
        if constexpr (T::kGated) {
            while (!T::open(wall_clock_l())) {
            }
        }
        if constexpr (mode == pv::kModeBurst) {
            if (burst_left-- == 0) {
                burst_left = a - 1;
                const uint32_t start = wall_clock_l();
                while (wall_clock_l() - start < T::pause_after(0)) {
                }
            }
        }
        loops++;
        const uint32_t cur_in0 = (check_det && (i & 1u)) ? l1_in0_b_addr : l1_in0_a_addr;
        for (uint32_t j = 0; j < num_tiles; j++) {
            uint32_t address_a = L1_ADDRESS(cur_in0);
            uint32_t address_b = L1_ADDRESS(l1_in1_addr + j * 2048);

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

    elapsed += wall_clock_l() - prev_clk;
    volatile tt_l1_ptr uint32_t* record = ctrl_word(ctrl_base, pv::kFpuRecord);
    record[0] = static_cast<uint32_t>(elapsed);
    record[1] = static_cast<uint32_t>(elapsed >> 32);
    record[2] = static_cast<uint32_t>(loops);
    record[3] = static_cast<uint32_t>(loops >> 32);
}
#endif

#ifdef TRISC_MATH
template <uint32_t num_tiles, uint32_t nops_per_tile>
ALWI void pv_math(uint32_t ctrl_base) {
    volatile tt_l1_ptr uint32_t* final_ptr = ctrl_word(ctrl_base, pv::kFinal);
    // init
    constexpr bool is_fp32_dest_acc_en = false;
    _llk_math_hw_configure_<is_fp32_dest_acc_en>((uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);
    addr_mod_t{
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
        .srcb = {.incr = 48, .clr = 0, .cr = 1},  // cr=32 before, cr+48=16 after wrapping
        .dest = {.incr = 0, .clr = 0, .cr = 1},
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
    math::_configure_default_zero_flag_state_();

    // compute loop
    for (uint32_t i = 0; !reached_final(i, final_ptr); i++) {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (uint32_t j = 0; j < num_tiles; j++) {
            ckernel_template::run();
            if constexpr (nops_per_tile != 0) {
#pragma GCC unroll nops_per_tile
                for (uint32_t k = 0; k < nops_per_tile; k++) {
                    TTI_NOP;
                }
            }
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif

#ifdef TRISC_PACK
// PACK stage from max_util_pack_common.hpp: the MOP drains all num_tiles
// tiles from DST to the fixed L1 output in one run.
template <uint32_t num_tiles, bool check_det>
ALWI void pv_pack(
    uint32_t l1_output_addr, uint32_t l1_golden_a_addr, uint32_t l1_golden_b_addr, uint32_t ctrl_base) {
    volatile tt_l1_ptr uint32_t* final_ptr = ctrl_word(ctrl_base, pv::kFinal);
    constexpr bool is_fp32_dest_acc_en = false;
    _llk_pack_hw_configure_<is_fp32_dest_acc_en>(
        (uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b, 128 /* tile size for float16_b >> 4 */);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    set_packer_strides<PackMode::Default>((uint32_t)DataFormat::Float16_b, TILE_C_DIM);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);

    addr_mod_pack_t{
        .y_src = {.incr = 4},
    }
        .set(ADDR_MOD_0);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 1, .clr = 0},
    }
        .set(ADDR_MOD_2);

    const uint32_t mop_inner_loop = 4;              // face_r_dim >> 2
    const uint32_t mop_outer_loop = 4 * num_tiles;  // num_faces * num_tiles
    ckernel::ckernel_template pack_mop(
        mop_outer_loop,
        mop_inner_loop,
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
    pack_mop.set_last_inner_loop_instr(TT_OP_PACR(
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
    pack_mop.set_last_outer_loop_instr(TT_OP_PACR(
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
    pack_mop.program();
    set_dst_write_addr(0);
    program_packer_destination(L1_ADDRESS(l1_output_addr));

    constexpr uint32_t kOutWords = num_tiles * 512;  // float16_b: 32*32*2 / 4 = 512 words/tile
    constexpr uint32_t kNumSlices = kOutWords / kDetSliceWords;
    static_assert(kOutWords % kDetSliceWords == 0 && (kNumSlices & (kNumSlices - 1)) == 0);

    volatile tt_l1_ptr uint32_t* out_base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_output_addr);
    volatile tt_l1_ptr uint32_t* gold_a_base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_golden_a_addr);
    volatile tt_l1_ptr uint32_t* gold_b_base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_golden_b_addr);
    pv::ErrTracker err;

    for (uint32_t i = 0; !reached_final(i, final_ptr); i++) {
        _llk_packer_wait_for_math_done_();
        ckernel::ckernel_template::run();
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        if constexpr (check_det) {
            // Drain PACK L1 writes before reading l1_output_addr.
            TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
            TTI_DMANOP;
            tensix_sync();
            invalidate_l1_cache();

            const uint32_t pair_idx = i >> 1;
            const uint32_t off = (pair_idx & (kNumSlices - 1)) * kDetSliceWords;
            volatile tt_l1_ptr uint32_t* out_p = out_base + off;
            volatile tt_l1_ptr uint32_t* gold_p = ((i & 1u) ? gold_b_base : gold_a_base) + off;
            if (pair_idx < kNumSlices) {
                for (uint32_t w = 0; w < kDetSliceWords; w++) {
                    gold_p[w] = out_p[w];
                }
            } else {
                uint32_t diff = 0;
                for (uint32_t w = 0; w < kDetSliceWords; w += 4) {
                    diff |= out_p[w] ^ gold_p[w];
                    diff |= out_p[w + 1] ^ gold_p[w + 1];
                    diff |= out_p[w + 2] ^ gold_p[w + 2];
                    diff |= out_p[w + 3] ^ gold_p[w + 3];
                }
                err.checks++;
                if (__builtin_expect(diff != 0, 0)) {
                    for (uint32_t w = 0; w < kDetSliceWords; w++) {
                        const uint32_t act = out_p[w];
                        const uint32_t exp = gold_p[w];
                        if (act != exp) {
                            err.record_mismatch(i, off + w, exp, act);
                        }
                    }
                }
            }
        }
    }

    if constexpr (check_det) {
        err.store(ctrl_word(ctrl_base, pv::kFpuErrRecord));
        asm volatile("fence" ::: "memory");
    }
}
#endif

void kernel_main() {
    constexpr uint32_t l1_in0_a_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_in0_b_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_in1_addr = get_compile_time_arg_val(2);
    constexpr uint32_t l1_out_addr = get_compile_time_arg_val(3);
    constexpr uint32_t l1_golden_a_addr = get_compile_time_arg_val(4);
    constexpr uint32_t l1_golden_b_addr = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t ctrl_base = get_compile_time_arg_val(7);
    constexpr bool super_sync = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t mode = get_compile_time_arg_val(9);
    constexpr uint32_t a = get_compile_time_arg_val(10);
    constexpr uint32_t b = get_compile_time_arg_val(11);
    constexpr uint32_t duty_period = get_compile_time_arg_val(12);
    constexpr uint32_t duty_active = get_compile_time_arg_val(13);
    constexpr bool check_det = get_compile_time_arg_val(14) != 0;

    // TRISC0: unpack A and B (float16_b) to SRC_A and SRC_B
    UNPACK((pv_unpack<num_tiles, super_sync, mode, a, b, duty_period, duty_active, check_det>(
        l1_in0_a_addr, l1_in0_b_addr, l1_in1_addr, ctrl_base)));

    // TRISC1: MVMUL to DST
    MATH((pv_math < num_tiles, mode == pv::kModeSmooth ? a : 0 > (ctrl_base)));

    // TRISC2: pack results to the output L1 address
    PACK((pv_pack<num_tiles, check_det>(l1_out_addr, l1_golden_a_addr, l1_golden_b_addr, ctrl_base)));
}
