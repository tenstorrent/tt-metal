// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_dest_chain.h"
#include "llk_memory_checks.h"

#ifdef LLK_TRISC_UNPACK
#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#define UNPACK(...) __VA_ARGS__
#else
#define UNPACK(...)
#endif

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#define MATH(...) __VA_ARGS__
#else
#define MATH(...)
#endif

#ifdef LLK_TRISC_ISOLATE_SFPU
#include "experimental/ckernel_sfpu_fill.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#define SFPU(...) __VA_ARGS__
#else
#define SFPU(...)
#endif

#ifdef LLK_TRISC_PACK
#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#define PACK(...) __VA_ARGS__
#else
#define PACK(...)
#endif

#include "params.h"

namespace
{

using ckernel::dest_dvalid_client;

constexpr std::uint32_t BITS_PER_SECTION  = 4;
constexpr std::uint32_t STEP_BITS         = 3;
constexpr std::uint32_t STEPS_PER_SECTION = 5;

inline float fill_value(const std::uint32_t i)
{
    return static_cast<float>(i + 1) / 8.0f;
}

inline std::uint32_t nibble(const std::uint32_t word, const std::uint32_t section)
{
    return (word >> (BITS_PER_SECTION * section)) & 0xF;
}

inline std::uint32_t step_bit(const std::uint32_t order_half, const std::uint32_t k)
{
    const std::uint32_t code = (order_half >> (STEP_BITS * k)) & 0x7;
    return code == 0 ? 0 : (1u << (code - 1));
}

UNPACK(std::uint32_t bfd_a;)
MATH(std::uint32_t datacopy_rows;)
PACK(std::uint32_t bfd_pack;)

struct UnpackToDestOp
{
    static constexpr dest_dvalid_client UNIT = dest_dvalid_client::UNPACK;

    static void init()
    {
        UNPACK((_llk_unpack_unary_operand_to_dest_init_(bfd_a)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        ckernel::_llk_dest_chain_take_<UNIT, dest_sync, is_fp32_dest_acc_en>();
        UNPACK((_llk_unpack_unary_operand_to_dest_tile_(i, 0 /*dest tile*/)));
    }
};

struct DatacopyOp
{
    static constexpr dest_dvalid_client UNIT = dest_dvalid_client::FPU;

    static void init()
    {
        UNPACK((
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(bfd_a, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/)));
        MATH((_llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(datacopy_rows, 1 /*num_matrices*/)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        ckernel::_llk_dest_chain_take_<UNIT, dest_sync, is_fp32_dest_acc_en>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, ckernel::DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0 /*dest tile*/)));
    }
};

struct SquareOp
{
    static constexpr dest_dvalid_client UNIT = dest_dvalid_client::SFPU;

    static void init()
    {
        SFPU((sfpu::init_square()));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, const bool produced)
    {
        ckernel::_llk_dest_chain_take_<UNIT, dest_sync, is_fp32_dest_acc_en>();
        if (produced)
        {
            SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), 0 /*dest tile*/, VectorMode::RC)));
        }
        else
        {
            SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, _calculate_fill_, (SFPU_ITERATIONS), 0 /*dest tile*/, VectorMode::RC, fill_value(i))));
        }
    }
};

struct PackOp
{
    static constexpr dest_dvalid_client UNIT = dest_dvalid_client::PACK;

    static void init()
    {
        PACK((_llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        ckernel::_llk_dest_chain_take_<UNIT, dest_sync, is_fp32_dest_acc_en>();
        PACK((_llk_pack_(0 /*dest tile*/, i, ckernel::DEFAULT_TENSOR_SHAPE)));
    }
};

template <typename Op>
inline void run_op(const std::uint32_t step, const bool produced, const std::uint32_t i, std::uint32_t& inited, dest_dvalid_client& configured_unpacker)
{
    const std::uint32_t bit = ckernel::dest_dvalid::bit_of(Op::UNIT);
    if (step != bit)
    {
        return;
    }
    constexpr bool USES_UNPACKER = Op::UNIT == dest_dvalid_client::UNPACK || Op::UNIT == dest_dvalid_client::FPU;
    if ((inited & bit) == 0 || (USES_UNPACKER && configured_unpacker != Op::UNIT))
    {
        Op::init();
        inited |= bit;
    }
    if constexpr (USES_UNPACKER)
    {
        configured_unpacker = Op::UNIT;
    }
    Op::exec(i, produced);
}

template <typename Op>
inline void init_op(const std::uint32_t members, std::uint32_t& inited)
{
    const std::uint32_t bit = ckernel::dest_dvalid::bit_of(Op::UNIT);
    if (members & bit)
    {
        ckernel::_llk_dest_chain_join_<Op::UNIT>();
    }
    else
    {
        ckernel::_llk_dest_chain_leave_<Op::UNIT>();
        inited &= ~bit;
    }
}

inline void run_section(const std::uint32_t order_half, const std::uint32_t i, std::uint32_t& inited, dest_dvalid_client& configured_unpacker)
{
    bool produced = false;
    for (std::uint32_t k = 0; k < STEPS_PER_SECTION; ++k)
    {
        const std::uint32_t step = step_bit(order_half, k);
        if (step == 0)
        {
            break;
        }
        run_op<UnpackToDestOp>(step, produced, i, inited, configured_unpacker);
        run_op<DatacopyOp>(step, produced, i, inited, configured_unpacker);
        run_op<SquareOp>(step, produced, i, inited, configured_unpacker);
        run_op<PackOp>(step, produced, i, inited, configured_unpacker);
        produced = produced || step != p_cleardvalid::PACK;
    }
}

inline void run_schedule(const std::uint32_t tile_cnt, const std::uint32_t* orders, const std::uint32_t members_word)
{
    ckernel::_llk_dest_chain_join_<dest_dvalid_client::FPU>();
    ckernel::_llk_dest_chain_join_<dest_dvalid_client::PACK>();
    std::uint32_t inited = 0;
    // Both producer operations program the same unpack MOP bank.
    dest_dvalid_client configured_unpacker = ckernel::dest_dvalid::NONE;
    for (std::uint32_t i = 0; i < tile_cnt; ++i)
    {
        const std::uint32_t members    = nibble(members_word, i);
        const std::uint32_t order_half = (orders[i / 2] >> (16 * (i % 2))) & 0xFFFF;
        init_op<UnpackToDestOp>(members, inited);
        init_op<SquareOp>(members, inited);
        ckernel::_llk_dest_chain_section_begin_();
        run_section(order_half, i, inited, configured_unpacker);
        ckernel::_llk_dest_chain_section_end_<dest_sync, is_fp32_dest_acc_en>();
    }
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    [[maybe_unused]] const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    [[maybe_unused]] const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    [[maybe_unused]] const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TILE_CNT                         = params.TILE_CNT;
    const std::uint32_t ORDERS[4]                        = {params.ORDER0, params.ORDER1, params.ORDER2, params.ORDER3};
    const std::uint32_t MEMBERS                          = params.MEMBERS;
    UNPACK(const Operand& buffer_A = params.buffer_A;)
    PACK(const Operand& buffer_Res = params.buffer_Res;)
#endif
    using namespace ckernel;

    UNPACK(
        (bfd_a = trisc::bfd_alloc_and_program<trisc::BfdResource::Unp0>(
             tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src)));
    UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_A_dst))));
    UNPACK((_llk_unpack_dest_init_()));

    MATH((_llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en>(
        static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math))));
    MATH((_reset_dest_register_offset_()));
    MATH((_set_dest_section_base_<TRISC_ID>(_get_dest_buffer_base_())));
    MATH(datacopy_rows = num_faces * TEST_FACE_R_DIM;)

    SFPU((_llk_math_eltwise_sfpu_init_()));

    PACK(bfd_pack = trisc::bfd_alloc_and_program<trisc::BfdResource::Pack0>(
             tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);)
    PACK((_llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ReluConfig::none())));
    PACK((_llk_pack_dest_init_<p_pacr::PACK0, dest_sync>()));

    run_schedule(TILE_CNT, ORDERS, MEMBERS);
}
