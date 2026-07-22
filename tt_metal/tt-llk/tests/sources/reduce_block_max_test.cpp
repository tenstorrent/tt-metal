// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone block-based MAX-pool-along-ROW LLK test (experimental).
//
// Exercises the experimental reduce_block_max_row LLKs
// (experimental/llk_{math,unpack_AB}_reduce_custom{,_runtime}.h):
//
//   out[row] = max over the whole block width (REDUCE_BLOCK_CT_DIM tiles, 32 cols each)
//              of operand A, with the row-max landing in column [0] of the tile.
//
// The scaler B is a single face of 1.0 in F0 (the op's contract). The op does a
// pure MAX pool, so the scaler is a no-op multiplier here.
//
// This expands the Compute API (api/compute/reduce_custom.h) into its underlying
// _llk_* calls so the kernel runs inside the tt-llk harness. Compile-time
// (constexpr) switches, supplied by the Python driver, select the covered path:
//
//   * USE_RUNTIME   - the runtime (dynamic block_ct_dim) LLK family.
//   * CLOBBER_OP    - op run between init and reinit that overwrites the reduce
//                     MOP/addrmods: 0 = none, 1 = eltwise binary init (reprograms
//                     ALL addrmods + MOP, as matmul / sub_exp do in the SDPA inner
//                     loop), 2 = scramble ADDR_MOD_1/2/6 only (preserving ADDR_MOD_3
//                     + MOP, the narrow escape reinit_minimal expects).
//   * REINIT_MODE   - re-arm the reduce config after the clobber, matching the SDPA
//                     inner-loop reinit paths: 0 = none, 1 = reinit_short (reprogram
//                     MOP + addrmods), 2 = reinit_minimal (ADDR_MOD_1/2/6 only).
//
// The compile-time short/minimal reinit lib fns are Blackhole-only, so the driver
// only requests those on Blackhole; the runtime reinit fns exist on both arches
// (runtime reinit_minimal is Blackhole-only).
//
// respect_trigger / overlap_first_half (the SDPA MOP-split producer/consumer
// handshake) are out of scope here — they are owned by the packer layer two levels
// above the LLK and need the SDPA compute-kernel scaffolding to drive faithfully.
//
// Reference: on-silicon compute kernels
//   tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row{,_runtime}/compute.cpp

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// 32x32 operand tile: 4 faces (2x2 face grid).
static constexpr std::uint32_t NUM_FACES = 4;

// respect_trigger is a producer/consumer handshake owned by the packer layer above
// the LLK; the standalone correctness sweep exercises the untriggered block reduce.
static constexpr bool RESPECT_TRIGGER    = false;
static constexpr bool OVERLAP_FIRST_HALF = false;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_reduce_custom.h"
#include "experimental/llk_unpack_AB_reduce_custom_runtime.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    // compute_kernel_hw_startup
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, NUM_FACES, NUM_FACES);

    if constexpr (USE_RUNTIME)
    {
        // reduce_block_max_row_init_runtime
        _llk_unpack_AB_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(REDUCE_BLOCK_CT_DIM, RESPECT_TRIGGER, tensor_shape);

        // Operand A is REDUCE_BLOCK_CT_DIM contiguous tiles; scaler B is a single tile of 1.0.
        _llk_unpack_AB_reduce_block_max_row_runtime_(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), RESPECT_TRIGGER, OVERLAP_FIRST_HALF);

        _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(RESPECT_TRIGGER, OVERLAP_FIRST_HALF);
    }
    else
    {
        // reduce_block_max_row_init
        _llk_unpack_AB_reduce_block_max_row_init_<REDUCE_BLOCK_CT_DIM, is_fp32_dest_acc_en, RESPECT_TRIGGER>(tensor_shape);

        _llk_unpack_AB_reduce_block_max_row_<RESPECT_TRIGGER>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));

        _llk_unpack_AB_reduce_block_max_row_uninit_<RESPECT_TRIGGER>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_reduce_custom.h"
#include "experimental/llk_math_reduce_runtime_custom.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"

// CLOBBER_OP selector: an op run between reduce init and reinit that overwrites the
// reduce config, so the reinit path must actually restore it (reconfig-escape guard).
//   0 = none.
//   1 = eltwise binary (ELWADD) init: reprograms ALL of ADDR_MOD_0..3 + the MOP — the
//       full escape that reinit_short (reprogram MOP + reconfigure all addrmods) restores.
//   2 = scramble ONLY ADDR_MOD_1/2/6, preserving ADDR_MOD_3 + the MOP — the narrow escape
//       reinit_minimal expects (it restores 1/2/6 and relies on 3 + MOP being intact, as
//       the real SDPA predecessors matmul / sub_exp / copy_tile_custom leave them).
static inline void clobber_reduce_config([[maybe_unused]] const ckernel::TensorShape& tensor_shape)
{
    if constexpr (CLOBBER_OP == 1)
    {
        // ELWADD requires LoFi (HiFi is multiply-only); fidelity is irrelevant here since
        // the op is never executed, only its init reconfigures the clobbered registers.
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
            tensor_shape, 0 /* acc_to_dest */);
    }
    else if constexpr (CLOBBER_OP == 2)
    {
        // Overwrite ADDR_MOD_1/2/6 with plainly wrong (zeroed) values; leave ADDR_MOD_3 and
        // the reduce MOP intact, matching reinit_minimal's contract. reinit_minimal must then
        // restore 1/2/6 for the reduce to produce the correct row-max.
        const ckernel::addr_mod_t zeroed = {
            .srca = {.incr = 0, .clr = 0, .cr = 0}, .srcb = {.incr = 0, .clr = 0, .cr = 0}, .dest = {.incr = 0, .clr = 0, .cr = 0}};
        zeroed.set(ckernel::ADDR_MOD_1);
        zeroed.set(ckernel::ADDR_MOD_2);
        zeroed.set(ckernel::ADDR_MOD_6);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    if constexpr (USE_RUNTIME)
    {
        _llk_math_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(REDUCE_BLOCK_CT_DIM, tensor_shape);

        // Overwrite the reduce MOP/addrmods so the reinit below must restore them.
        clobber_reduce_config(tensor_shape);

        if constexpr (REINIT_MODE == 1)
        {
            // reduce_block_max_row_reinit_short_runtime: reprogram MOP + restore addrmods (both arches).
            _llk_math_reduce_block_max_row_reinit_short_runtime_<is_fp32_dest_acc_en>(REDUCE_BLOCK_CT_DIM, tensor_shape);
        }
#ifdef ARCH_BLACKHOLE
        else if constexpr (REINIT_MODE == 2)
        {
            // reduce_block_max_row_reinit_minimal_runtime: restore ADDR_MOD_1/2/6 only (Blackhole-only lib fn).
            _llk_math_reduce_block_max_row_reinit_minimal_runtime_();
        }
#endif

        _llk_math_wait_for_dest_available_<DST_SYNC>();
        _llk_math_reduce_block_max_row_runtime_<is_fp32_dest_acc_en>(0 /* dst_index */, tensor_shape);
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

        _llk_math_reduce_block_max_row_uninit_runtime_<is_fp32_dest_acc_en>();
    }
    else
    {
        _llk_math_reduce_block_max_row_init_<REDUCE_BLOCK_CT_DIM, is_fp32_dest_acc_en>(tensor_shape);

        // Overwrite the reduce MOP/addrmods so the reinit below must restore them.
        clobber_reduce_config(tensor_shape);

#ifdef ARCH_BLACKHOLE
        if constexpr (REINIT_MODE == 1)
        {
            // reduce_block_max_row_reinit_short == reduce_max_row_configure_addrmod + mop_reprogram_only
            // (matches llk_math_reduce_block_max_row_reinit_with_mop). Blackhole-only lib path.
            reduce_max_row_configure_addrmod();
            _llk_math_reduce_block_max_row_mop_reprogram_only_<REDUCE_BLOCK_CT_DIM>(tensor_shape);
        }
        else if constexpr (REINIT_MODE == 2)
        {
            // reduce_block_max_row_reinit_minimal: restore ADDR_MOD_1/2/6 only (Blackhole-only lib fn).
            reduce_max_row_configure_addrmod_reinit_minimal();
        }
#endif

        _llk_math_wait_for_dest_available_<DST_SYNC>();
        _llk_math_reduce_block_max_row_<REDUCE_BLOCK_CT_DIM, is_fp32_dest_acc_en>(0 /* dst_index */, tensor_shape);
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

        _llk_math_reduce_block_max_row_uninit_<is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;

    // compute_kernel_hw_startup
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    // reduce_block_max_row_init: mask so only the reduced row-max column [0] is packed.
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_ROW>(tensor_shape.face_r_dim);

    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    // Single output tile: the per-row maxima live in column [0] of DEST[0].
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* dst_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    // reduce_block_max_row_uninit
    _llk_pack_reduce_mask_clear_();
}

#endif
