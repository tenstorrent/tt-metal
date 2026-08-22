
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Elementwise binary test whose MATH thread drives the compute through the
 * compiler-managed Tensix compute intrinsics (__builtin_rvtt_*_elwmul)
 * instead of the LLK's llk_math_eltwise_binary_* API.  The one-time ALU
 * hw_configure baseline is issued by the LLK's _llk_math_hw_configure_ through
 * the config-write intrinsics (stallwait/rmwciB*), which pass_rvtt_config
 * consumes and coalesces; the per-compute reconfig for each elwmul is derived
 * by the compiler from the intrinsic's format operands.  The synchronization
 * primitives (_llk_math_pack_sync_init_, wait/done) own the semaphores and
 * dest-section coordination.
 *
 * The compute is one 16x16 face per call (a single TTELWMUL), which matches
 * the LLK's LoFi MOP for a single-face tile.  Multi-face tiles need the
 * deferred addr-mod / INCRWC dest-walking work.
 */

#include <cstdint>

#include "build.h"
#include "llk_defs.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

// Compiler-managed unpack data-op, arch-prefixed like INTR_ELWMUL.
#if defined(ARCH_WORMHOLE)
#define INTR_UNPACR __builtin_rvtt_wh_unpacr
#else
#define INTR_UNPACR __builtin_rvtt_bh_unpacr
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    const ckernel::Transpose transpose      = params.UNPACK_TRANSPOSE_FACES
                                                  ? (params.UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::Both : ckernel::Transpose::InterFace)
                                                  : (params.UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::IntraFace : ckernel::Transpose::None);

    // Configure hardware for unpacking, no broadcast, no transpose.  The LLK's
    // _llk_unpack_hw_configure_ now issues the config through the config-write
    // intrinsics (rmwciB*/setdmareg) which pass_rvtt_config consumes and
    // coalesces; the compile-time formats/geometry drive constant-folded
    // immediates.  The 16x16 one-face shape matches the unpack-config
    // declaration the compiler used to derive from these six operands.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_A_IN,
        UNPACK_B_IN,
        UNPACK_A_OUT,
        UNPACK_B_OUT,
        /*unpA_face_r_dim=*/16,
        /*unpB_face_r_dim=*/16,
        /*unpA_num_faces=*/1,
        /*unpB_num_faces=*/1);

    // Must come after _llk_unpack_hw_configure_, otherwise the ALU stoch-rnd
    // bits programmed here are overwritten by configure_unpack_AB().
    _llk_unpack_configure_stoch_rnd_<StochRndType::None>();

    _llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, transpose);

    const std::uint32_t num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (std::uint32_t i = 0; i < num_total_tiles; ++i)
    {
        // Author-owned data-op: the same per-tile address writes + sync as
        // _llk_unpack_AB_, but the MOP run is replaced by two inline unpacr
        // intrinsics (one 16x16 face = one SrcA + one SrcB read).  The unpacr
        // words match the LLK MOP's (AddrMode=1, OvrdThreadId=1,
        // SetDatValid=1, Last=1); the remaining fields are the TT_OP_UNPACR
        // defaults.  The operand list is the full 13-field UNPACR word:
        // (Unpack_block_selection, AddrMode, CfgContextCntInc, CfgContextId,
        //  AddrCntContextId, OvrdThreadId, SetDatValid, srcb_bcast,
        //  ZeroWrite2, AutoIncContextID, RowSearch, SearchCacheFlush, Last).
        volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset addr counters
        wait_for_next_context(2);
        _llk_unpack_configure_addresses_(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]), cfg);
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        INTR_UNPACR(0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1); // SrcA read
        INTR_UNPACR(1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1); // SrcB read
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

// Compute intrinsic, arch-prefixed (the Tensix mnemonics are identical; the
// J-format field widths differ per arch).  Selected by the harness's
// -DARCH_* define.
#if defined(ARCH_WORMHOLE)
#define INTR_ELWMUL __builtin_rvtt_wh_elwmul
#else
#define INTR_ELWMUL __builtin_rvtt_bh_elwmul
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Semaphore + dest-section sync, plus the one-time ALU baseline.  The LLK's
    // _llk_math_hw_configure_ now issues the config through the config-write
    // intrinsics (stallwait/rmwciB*) which pass_rvtt_config consumes and
    // coalesces; the compile-time MATH_FORMAT drives constant-folded immediates.
    // _llk_math_pack_sync_init_ also programs the dest section base (SETC16).
    // The elwmul intrinsic is the bare TTELWMUL instruction; the ALU formats /
    // INT8 / zero-flag state it runs on is established by the hw_configure
    // below (and, on the runtime-formats path, re-derived per call).
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FORMAT, MATH_FORMAT);

    // One 16x16 tile (one 16-row face) at dest index 0.  A single TTELWMUL
    // computes 8 rows (MAX_FPU_ROWS); a 16-row face needs two ELWMULs with an
    // INCRWC row-advance between, mirroring the LLK's partial-face MOP
    // (eltwise_binary_configure_mop_standard's loop_op1 INCRWC).  The
    // intrinsic's dst operand is a compile-time constant (dest-walking /
    // runtime dst is deferred work), so this oracle is scoped to the one-tile
    // case.  The INCRWC is a hand-written row advance between two identical
    // intrinsic calls; the compiler emits config once (state tracking) and
    // TTELWMUL twice.
    _llk_math_wait_for_dest_available_<dest_sync>();
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    _llk_math_reconfig_data_format_<is_fp32_dest_acc_en>(
        ckernel::to_underlying(formats.math), ckernel::to_underlying(formats.math));
    INTR_ELWMUL(
        0 /*clr_src*/, 0 /*acc_to_dest*/, 0 /*broadcast*/, 0 /*addr_mod*/, 0 /*dst*/);
    TTI_INCRWC(0 /*cr*/, 8 /*dest*/, 8 /*srcb*/, 8 /*srca*/);
    INTR_ELWMUL(
        0 /*clr_src*/, 0 /*acc_to_dest*/, 0 /*broadcast*/, 0 /*addr_mod*/, 0 /*dst*/);
#else
    INTR_ELWMUL(0, 0, 0, 0, 0);
    TTI_INCRWC(0 /*cr*/, 8 /*dest*/, 8 /*srcb*/, 8 /*srca*/);
    INTR_ELWMUL(0, 0, 0, 0, 0);
#endif
    // Leave the simulator in a clean state for the next kernel (ttsim persists
    // across tests in one process): clear the SrcA/B valid bits so a following
    // kernel's ELWMUL source-valid stall waits for its own unpack, and reset
    // the RWC A/B/D counters that sit at 8 after the face.  The stock LLK's
    // MOP end_op does the same (CLR_AB + clear_dst_reg_addr).
    TTI_SETRWC(p_setrwc::CLR_AB, 0 /*cr*/, 0 /*dest*/, 0 /*srcb*/, 0 /*srca*/, p_setrwc::SET_ABD);
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

// Compiler-managed pack data-op.  BH-only this slice (the WH pack intrinsics +
// config geometry are deferred, like WH unpack).
// The builtin takes all 12 PACR fields; this keeps the six the test varies
// and zeroes the rest, which is what it always meant.
#define INTR_PACR(cfg_context, addr_mode, addr_cnt_context, read_intf_sel, zero_write, last) \
    __instrn_buffer[0] = __builtin_rvtt_bh_pacr(cfg_context, 0, 0, addr_mode, addr_cnt_context, zero_write, read_intf_sel, 0, 0, 0, 0, last)

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};

    // The compiler-managed pack hw_configure is now the LLK's configure_pack,
    // which issues the config through the config-write intrinsics (rmwciB*/
    // setdmareg/wrcfg) that pass_rvtt_config consumes.  Those immediates must be
    // constant-foldable, so the call uses the compile-time PACK_IN/PACK_OUT and
    // this oracle's fixed 16x16 one-face geometry (mirroring the pack-config
    // declaration the compiler used to derive from these operands), NOT the
    // runtime tensor_shape values.  _llk_pack_dest_init_wrapper_ stays
    // (dest-offset GPRs + DEST_TARGET + counter init -- author-owned
    // dest-section state).
    [[maybe_unused]] const std::uint32_t tile_size = tensor_shape.total_tensor_size();

    [[maybe_unused]] const std::uint32_t num_faces = tensor_shape.total_num_faces();
    [[maybe_unused]] const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    const bool narrow_tile = (tensor_shape.num_faces_c_dim == 1);

    // tile_size is the TILE_HEADER datum count (total_tensor_size() of the
    // 16x16 one-face tile = 16*16 datums).
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        PACK_IN, PACK_OUT, /*tile_size=*/16 * 16, /*face_r_dim=*/16, /*tile_c_dim=*/16, /*num_faces=*/1, /*partial_face=*/false, /*narrow_tile=*/false);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        PACK_OUT, /*face_r_dim=*/16, /*tile_c_dim=*/16, /*num_faces=*/1, /*partial_face=*/false, /*narrow_tile=*/false);

    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    const std::uint32_t output_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t output_num_blocks     = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < output_num_blocks; block++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
        {
            std::uint32_t res_tile_idx = (block * output_tiles_in_block) + tile;
            LLK_ASSERT(
                (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            // Author-owned data-op: the per-tile dest select + L1 dest address
            // + sync, then the MOP's PACR stream inlined -- one 16-row face is
            // 4 PACRs (4 dest rows each via ALL_INTF_ACTIVE) with addr-mod 2 on
            // the last, then the outer end (addr-mod 1, Last=1).  The pacr
            // words match the LLK MOP's (_llk_pack_mop_config_, Default mode).
            set_dst_write_addr(tile);
            program_packer_destination(L1_ADDRESS(params.buffer_Res[res_tile_idx]));
            INTR_PACR(0, 0, 0, 0, 0, 0); // rows 0-3   (addr-mod 0)
            INTR_PACR(0, 0, 0, 0, 0, 0); // rows 4-7
            INTR_PACR(0, 0, 0, 0, 0, 0); // rows 8-11
            INTR_PACR(0, 2, 0, 0, 0, 0); // rows 12-15 (last inner: addr-mod 2)
            INTR_PACR(0, 1, 0, 0, 0, 1); // outer end  (addr-mod 1, Last=1)
            TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
