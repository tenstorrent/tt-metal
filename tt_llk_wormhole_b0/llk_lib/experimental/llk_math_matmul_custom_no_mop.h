// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_matmul.h"

using namespace ckernel;

inline void matmul_validate_no_mop_contract(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    LLK_ASSERT(
        in0_tile_r_dim == TILE_R_DIM && in0_tile_c_dim == TILE_C_DIM && in1_tile_r_dim == TILE_R_DIM && in1_tile_c_dim == TILE_C_DIM && !partial_face,
        "Wormhole custom no-mop matmul currently supports only full 32x32 tiles with partial_face disabled");
}

inline std::uint32_t matmul_get_replay_buf_len_no_mop(
    [[maybe_unused]] const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    [[maybe_unused]] const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    [[maybe_unused]] const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    [[maybe_unused]] const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    [[maybe_unused]] const bool partial_face            = false)
{
    // The narrowed WH full-tile path uses a fixed replay image.
    return 16;
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void matmul_configure_addrmod_no_mop(
    const bool transpose,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    // The current Wormhole no-mop path is intentionally narrowed to the
    // full-tile use case. Keep the contract explicit so generic LLK harnesses
    // do not silently exercise unsupported tiny-tile or partial-face variants.
    static_assert(THROTTLE_LEVEL == 0, "Wormhole custom no-mop matmul only supports THROTTLE_LEVEL == 0");
    matmul_validate_no_mop_contract(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    // Reuse the regular Wormhole matmul addrmods for the tile traversal itself.
    // The no-mop-specific part below only fixes up the dvalid contract for
    // reentry. Unlike BH, this WH path has to make the A/B reuse policy explicit
    // so repeated replays keep the right source valid across ct/rt shapes.
    matmul_configure_addrmod<math_fidelity, THROTTLE_LEVEL>(transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    // When the replay reuses one operand across multiple output tiles, keep the
    // opposite source disabled until the replay sequence explicitly clears it.
    if (t_dim > 1)
    {
        if (reuse_a)
        {
            TTI_SETC16(CLR_DVALID_SrcB_Disable_ADDR32, CLR_DVALID_SrcB_Disable_MASK);
        }
        else
        {
            TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, CLR_DVALID_SrcA_Disable_MASK);
        }
    }
    else
    {
        TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    }
}

template <MathFidelity math_fidelity>
inline void matmul_emit_replay_program_no_mop(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    [[maybe_unused]] const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    [[maybe_unused]] const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    [[maybe_unused]] const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    [[maybe_unused]] const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    [[maybe_unused]] const bool partial_face            = false)
{
    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    // This is the fixed full-tile replay image for the current no-mop path.
    //
    // in0 is loaded to SrcB, in1 is loaded to SrcA.
    // Unpacker always loads faces in f0,f1,f2,f3 order. If in1 is transposed,
    // faces 1 and 2 are effectively swapped by the addrmod increments.
    //
    // The first 15 instructions are the repeated MVMUL schedule for:
    //   B0*A0, B0*A1, B2*A0, B2*A1, B1*A2, B1*A3, B3*A2, B3*A3
    // with the usual transpose-sensitive face remap handled by ADDR_MOD_1/3.
    //
    // The final instruction leaves A/B counters in the state expected by the
    // next replay call. For LoFi this depends on whether A or B is being reused
    // across the outer loop.
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    // B0A0
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    // B0A0
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    // B0A1
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    // B0A1
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);

    // B2A0
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    // B2A0
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    // B2A1
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
    // B2A1
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);

    // B1A2
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    // B1A2
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    // B1A3
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    // B1A3
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);

    // B3A2
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    // B3A2
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    // B3A3
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);

    if constexpr (high_fidelity)
    {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    }
    else
    {
        if (reuse_a)
        {
            if (t_dim > 1)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_1, 0);
            }
        }
        else
        {
            if (t_dim > 1)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_1, 0);
            }
        }
    }
}

template <MathFidelity math_fidelity>
inline void matmul_load_replay_no_mop(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    const std::uint32_t replay_buf_len = matmul_get_replay_buf_len_no_mop(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    // WH records the replay image explicitly at init/reinit time rather than
    // assuming it persists like the BH path does.
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, replay_buf_len);
    matmul_emit_replay_program_no_mop<math_fidelity>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
}

template <MathFidelity math_fidelity>
inline void matmul_execute_replay_no_mop(const std::uint32_t replay_buf_len, const bool reuse_a, const std::uint32_t t_dim)
{
    if constexpr (!is_high_fidelity(math_fidelity))
    {
        lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
        return;
    }

    // HiFi paths replay the same full-tile program multiple times, then repair
    // the A/B/F counter state to match what the next outer-loop iteration
    // expects.
    constexpr std::uint32_t num_replay = to_underlying(math_fidelity);
    for (std::uint32_t replay = 0; replay < num_replay; replay++)
    {
        lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
    }

    if (t_dim > 1)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_F);
    }
    else if (reuse_a)
    {
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
    else
    {
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
}

template <MathFidelity math_fidelity>
inline void matmul_run_no_mop_tdim1_reuse_a(
    const std::uint32_t dst_index, [[maybe_unused]] const std::uint32_t ct_dim, const std::uint32_t rut_dim, const std::uint32_t replay_buf_len)
{
    for (std::uint32_t rut = 0; (rut + 1) < rut_dim; rut++)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + rut);
        matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, true, 1);
    }

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + rut_dim - 1);
    matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, true, 1);
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <MathFidelity math_fidelity>
inline void matmul_run_no_mop_tdim1_reuse_b(
    const std::uint32_t dst_index, const std::uint32_t ct_dim, const std::uint32_t rut_dim, const std::uint32_t replay_buf_len)
{
    for (std::uint32_t rut = 0; (rut + 1) < rut_dim; rut++)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + rut * ct_dim);
        matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, false, 1);
    }

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (rut_dim - 1) * ct_dim);
    matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, false, 1);
    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <MathFidelity math_fidelity>
inline void matmul_run_no_mop_tdim_gt1_reuse_a(
    const std::uint32_t dst_index, const std::uint32_t ct_dim, const std::uint32_t t_dim, const std::uint32_t rut_dim, const std::uint32_t replay_buf_len)
{
    // reuse_a means A stays fixed while we walk B/DEST across the row-tiling
    // dimension. The loop handles two t-slices at a time so the second slice
    // can reuse the same B state before resetting both source counters.
    for (std::uint32_t t = 0; t < t_dim; t += 2)
    {
        for (std::uint32_t rut = 0; (rut + 1) < rut_dim; rut++)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + ct_dim * t + rut);
            matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, true, t_dim);

            if ((t + 1) < t_dim)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + ct_dim * (t + 1) + rut);
                matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, true, t_dim);
            }

            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }

        const std::uint32_t rut = rut_dim - 1;
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + ct_dim * t + rut);
        matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, true, t_dim);

        if ((t + 1) < t_dim)
        {
            TTI_CLEARDVALID(p_setrwc::CLR_B, 0);
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + ct_dim * (t + 1) + rut);
            matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, true, t_dim);
        }

        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
        TTI_CLEARDVALID(p_setrwc::CLR_B, 0);
    }
}

template <MathFidelity math_fidelity>
inline void matmul_run_no_mop_tdim_gt1_reuse_b(
    const std::uint32_t dst_index, const std::uint32_t ct_dim, const std::uint32_t t_dim, const std::uint32_t rut_dim, const std::uint32_t replay_buf_len)
{
    // Mirror of the reuse_a case: B stays fixed while A/DEST advance across the
    // row-tiling dimension. The counter repair is symmetric, but now A is the
    // source that gets reused and later released.
    for (std::uint32_t t = 0; t < t_dim; t += 2)
    {
        for (std::uint32_t rut = 0; (rut + 1) < rut_dim; rut++)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + t + rut * ct_dim);
            matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, false, t_dim);

            if ((t + 1) < t_dim)
            {
                TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + t + 1 + rut * ct_dim);
                matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, false, t_dim);
            }

            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }

        const std::uint32_t rut = rut_dim - 1;
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + t + rut * ct_dim);
        matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, false, t_dim);

        if ((t + 1) < t_dim)
        {
            TTI_CLEARDVALID(p_setrwc::CLR_A, 0);
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + t + 1 + rut * ct_dim);
            matmul_execute_replay_no_mop<math_fidelity>(replay_buf_len, false, t_dim);
        }

        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        TTI_CLEARDVALID(p_setrwc::CLR_A, 0);
    }
}

template <MathFidelity math_fidelity = MathFidelity::LoFi, int THROTTLE_LEVEL = 0>
inline void matmul_reinit_no_mop(
    const bool transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    static_assert(THROTTLE_LEVEL == 0, "Wormhole custom no-mop matmul only supports THROTTLE_LEVEL == 0");
    // Reinit restores the addrmod + counter contract for the next replay call.
    // Replay bytes are managed separately by the higher-level WH LLK API.
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(
        transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, ct_dim, rt_dim);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_no_mop_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    static_assert(THROTTLE_LEVEL == 0, "Wormhole custom no-mop matmul only supports THROTTLE_LEVEL == 0");
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(
        transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, ct_dim, rt_dim);
    // Initial entry records the replay image once; later calls just replay it
    // after restoring the addrmod/counter contract.
    matmul_load_replay_no_mop<math_fidelity>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_matmul_uninit_no_mop_()
{
    _llk_math_matmul_uninit_();
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_no_mop_(
    std::uint32_t dst_index,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    static_assert(THROTTLE_LEVEL == 0, "Wormhole custom no-mop matmul only supports THROTTLE_LEVEL == 0");
    matmul_validate_no_mop_contract(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    // rut_dim is the outer dimension we iterate over while the other source is
    // reused. This matches the BH helper naming and is effectively the
    // "reuse-dim" for the no-mop path.
    const std::uint32_t rut_dim        = reuse_a ? ct_dim : rt_dim;
    const std::uint32_t replay_buf_len = matmul_get_replay_buf_len_no_mop(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    if (t_dim == 1)
    {
        if (reuse_a)
        {
            matmul_run_no_mop_tdim1_reuse_a<math_fidelity>(dst_index, ct_dim, rut_dim, replay_buf_len);
        }
        else
        {
            matmul_run_no_mop_tdim1_reuse_b<math_fidelity>(dst_index, ct_dim, rut_dim, replay_buf_len);
        }
        return;
    }

    if (reuse_a)
    {
        matmul_run_no_mop_tdim_gt1_reuse_a<math_fidelity>(dst_index, ct_dim, t_dim, rut_dim, replay_buf_len);
    }
    else
    {
        matmul_run_no_mop_tdim_gt1_reuse_b<math_fidelity>(dst_index, ct_dim, t_dim, rut_dim, replay_buf_len);
    }
}
