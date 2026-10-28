// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the SFPU activations (llk2::vec_*) on Tensors, one kernel body for all three threads. The
// activation is a compile-time parameter (VEC_FUNCTION, the value of llk2::VecFunction; VEC_FAST).
// Getting the input into Dest (unpack A + datacopy), the math-to-pack Dest sync and the pack still use today's LLKs
// until llk2::copy and the pack + Dest sync API land.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/vec.h"

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#endif
#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#endif

constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
constexpr std::uint32_t tiles    = (ROWS / llk2::default_tile_dim) * (COLS / llk2::default_tile_dim);

// The scalars each activation takes, as FP32 bit patterns. The goldens in test_llk2_vec.py use the same values.
constexpr std::uint32_t relu_max_max       = 0x40a00000u; // 5.0 (RELU_MAX_THRESHOLD)
constexpr std::uint32_t hardtanh_min       = 0xbf800000u; // -1.0 (CLAMP_MIN)
constexpr std::uint32_t hardtanh_max       = 0x3f800000u; // 1.0 (CLAMP_MAX)
constexpr std::uint32_t selu_scale         = 0x3f867d5fu; // 1.0507 (torch selu)
constexpr std::uint32_t selu_alpha         = 0x3fd62d7du; // 1.6733 (torch selu)
constexpr std::uint32_t softplus_beta      = 0x3f800000u; // 1.0 (SOFTPLUS_BETA)
constexpr std::uint32_t softplus_beta_recip = 0x3f800000u; // 1 / 1.0
constexpr std::uint32_t softplus_threshold = 0x41a00000u; // 20.0 (SOFTPLUS_THRESHOLD)

template <const auto& Acc>
inline void apply(const llk2::Tensor<Acc>& acc)
{
    using llk2::VecFunction;
    constexpr llk2::vec_modifiers mods {.fast = VEC_FAST};
    constexpr VecFunction f = static_cast<VecFunction>(VEC_FUNCTION);
    if constexpr (f == VecFunction::Silu)
    {
        llk2::vec_silu<mods>(acc);
    }
    else if constexpr (f == VecFunction::Tanh)
    {
        llk2::vec_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::Gelu)
    {
        llk2::vec_gelu<mods>(acc);
    }
    else if constexpr (f == VecFunction::GeluTanh)
    {
        llk2::vec_gelu_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::ReluMax)
    {
        llk2::vec_relu_max<mods>(acc, relu_max_max);
    }
    else if constexpr (f == VecFunction::Sigmoid)
    {
        llk2::vec_sigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardsigmoid)
    {
        llk2::vec_hardsigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardtanh)
    {
        llk2::vec_hardtanh<mods>(acc, hardtanh_min, hardtanh_max);
    }
    else if constexpr (f == VecFunction::Selu)
    {
        llk2::vec_selu<mods>(acc, selu_scale, selu_alpha);
    }
    else if constexpr (f == VecFunction::Softplus)
    {
        llk2::vec_softplus<mods>(acc, softplus_beta, softplus_beta_recip, softplus_threshold);
    }
    else
    {
        static_assert(f != f, "unknown VEC_FUNCTION");
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] const Operand& buffer_A   = params.buffer_A;
    [[maybe_unused]] const Operand& buffer_Res = params.buffer_Res;

    const auto acc = llk2::dst_tensor<ROWS, COLS, dest_format>(0); // the whole tensor in Dest from row 0

    // Old style: L1 -> SrcA (or straight to Dest) for every tile.
#ifdef LLK_TRISC_UNPACK
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t i = 0; i < tiles; i++)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
#endif

    // Old style: SrcA -> Dest for every tile, inside today's Dest sync.
#ifdef LLK_TRISC_MATH
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<dest_sync>();
    for (std::uint32_t i = 0; i < tiles; i++)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math);
    }
#endif

    apply(acc); // acc = f(acc), every tile

#ifdef LLK_TRISC_MATH
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
#endif

#ifdef LLK_TRISC_PACK
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < tiles; i++)
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
#endif
}
