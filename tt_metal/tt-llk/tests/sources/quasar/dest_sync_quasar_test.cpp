// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_dest_sync.h"
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

using ckernel::DestClient;

constexpr std::uint32_t STEP_BITS         = 3;
constexpr std::uint32_t STEPS_PER_SECTION = 5;
constexpr std::uint32_t NO_STEP           = 0;

inline float fill_value(const std::uint32_t i)
{
    return static_cast<float>(i + 1) / 8.0f;
}

// Step code k + 1 is client k, 0 ends the section.
inline std::uint32_t step_code(const std::uint32_t order_half, const std::uint32_t k)
{
    return (order_half >> (STEP_BITS * k)) & 0x7;
}

constexpr std::uint32_t code_of(const DestClient client)
{
    return ckernel::to_underlying(client) + 1;
}

constexpr std::uint32_t member_bit(const DestClient client)
{
    return 1u << ckernel::to_underlying(client);
}

UNPACK(std::uint32_t bfd_a;)
UNPACK(DataFormat to_dest_format;)
UNPACK(DataFormat to_src_format;)
MATH(std::uint32_t datacopy_rows;)
PACK(std::uint32_t bfd_pack;)

struct UnpackToDestOp
{
    static constexpr DestClient CLIENT = DestClient::UNPACK;

    // Both producers share the unpacker, so each programs its own output format, after the unpacks in flight.
    static void init()
    {
        ckernel::_llk_dest_sync_join_<CLIENT>();
        UNPACK((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::UNPACK0)));
        UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(to_dest_format)));
        UNPACK((_llk_unpack_unary_operand_to_dest_init_(bfd_a)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        // The unpacker writes the L1 format, Float32 when dest is 32-bit, so it addresses dest like every other client.
        ckernel::_llk_dest_sync_take_<CLIENT, is_fp32_dest_acc_en>();
        UNPACK((_llk_unpack_unary_operand_to_dest_tile_(i, 0 /*dest tile*/)));
    }
};

struct DatacopyOp
{
    static constexpr DestClient CLIENT = DestClient::FPU;

    // The unpacker goes back to unpacking to SrcA, so it leaves the dest sync chain.
    static void init()
    {
        ckernel::_llk_dest_sync_leave_<DestClient::UNPACK>();
        UNPACK((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::UNPACK0)));
        UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(to_src_format)));
        UNPACK((
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(bfd_a, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/)));
        MATH((_llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(datacopy_rows, 1 /*num_matrices*/)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        ckernel::_llk_dest_sync_take_<CLIENT, is_fp32_dest_acc_en>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, ckernel::DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0 /*dest tile*/)));
    }
};

struct SquareOp
{
    static constexpr DestClient CLIENT = DestClient::SFPU;

    static void init()
    {
        ckernel::_llk_dest_sync_join_<CLIENT>();
        SFPU((sfpu::init_square()));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, const bool produced)
    {
        ckernel::_llk_dest_sync_take_<CLIENT, is_fp32_dest_acc_en>();
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
    static constexpr DestClient CLIENT = DestClient::PACK;

    static void init()
    {
        PACK((_llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        ckernel::_llk_dest_sync_take_<CLIENT, is_fp32_dest_acc_en>();
        PACK((_llk_pack_(0 /*dest tile*/, i, ckernel::DEFAULT_TENSOR_SHAPE)));
    }
};

#if defined(LLK_TRISC_MATH) || defined(LLK_TRISC_ISOLATE_SFPU)
// Run on T1 and T3, each before its first dest access. srcAB configure disables implied source formats only for the
// calling thread. The Dstacc override pins the dest format: without it, sfpmem::DEFAULT loads and stores take the
// format from the per-bank saved dest format, and without dvalid nothing keeps that right for bank 1.
inline void configure_alu(const DataFormat math_format)
{
    const DataFormat dest_format = is_fp32_dest_acc_en ? DataFormat::Float32 : math_format;
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    _configure_alu_formats_<is_fp32_dest_acc_en>(math_format, math_format, false /*en_int32_dest_format*/, dest_format);
}
#endif

struct Schedule
{
    std::uint32_t inited    = 0;
    DestClient unpacker_mop = DestClient::PACK; // the producer the unpack MOP is set up for, PACK while none
};

template <typename Op>
inline void run_op(const std::uint32_t code, const bool produced, const std::uint32_t i, Schedule& schedule)
{
    if (code != code_of(Op::CLIENT))
    {
        return;
    }
    // Unpack to dest and the datacopy program the same unpack MOP.
    constexpr bool USES_UNPACKER = Op::CLIENT == DestClient::UNPACK || Op::CLIENT == DestClient::FPU;
    if ((schedule.inited & member_bit(Op::CLIENT)) == 0 || (USES_UNPACKER && schedule.unpacker_mop != Op::CLIENT))
    {
        Op::init();
        schedule.inited |= member_bit(Op::CLIENT);
    }
    if constexpr (USES_UNPACKER)
    {
        schedule.unpacker_mop = Op::CLIENT;
    }
    Op::exec(i, produced);
}

inline void run_section(const std::uint32_t order_half, const std::uint32_t i, Schedule& schedule)
{
    bool produced = false;
    for (std::uint32_t k = 0; k < STEPS_PER_SECTION; ++k)
    {
        const std::uint32_t code = step_code(order_half, k);
        if (code == NO_STEP)
        {
            break;
        }
        run_op<UnpackToDestOp>(code, produced, i, schedule);
        run_op<DatacopyOp>(code, produced, i, schedule);
        run_op<SquareOp>(code, produced, i, schedule);
        run_op<PackOp>(code, produced, i, schedule);
        // A take after PACK opens a use on a cleared bank, so the golden starts that section without a product.
        produced = code != code_of(DestClient::PACK);
    }
}

inline void run_schedule(const std::uint32_t tile_cnt, const std::uint32_t* orders)
{
    Schedule schedule;
    for (std::uint32_t i = 0; i < tile_cnt; ++i)
    {
        const std::uint32_t order_half = (orders[i / 2] >> (16 * (i % 2))) & 0xFFFF;
        run_section(order_half, i, schedule);
    }
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] const FormatConfig& formats         = params.formats;
    [[maybe_unused]] const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    [[maybe_unused]] const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TILE_CNT                         = params.TILE_CNT;
    const std::uint32_t ORDERS[4]                        = {params.ORDER0, params.ORDER1, params.ORDER2, params.ORDER3};
    UNPACK(const Operand& buffer_A = params.buffer_A;)
    PACK(const Operand& buffer_Res = params.buffer_Res;)
    using namespace ckernel;

    UNPACK(
        (bfd_a = trisc::bfd_alloc_and_program<trisc::BfdResource::Unp0>(
             tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src)));
    // Unpack to dest writes the L1 format unconverted, the datacopy unpacks to the inferred SrcA format.
    UNPACK(to_dest_format = static_cast<DataFormat>(formats.unpack_A_src);)
    UNPACK(to_src_format = static_cast<DataFormat>(formats.unpack_A_dst);)

    MATH((configure_alu(static_cast<DataFormat>(formats.math))));
    MATH(datacopy_rows = num_faces * TEST_FACE_R_DIM;)

    SFPU((configure_alu(static_cast<DataFormat>(formats.math))));
    SFPU((_llk_math_eltwise_sfpu_init_()));

    PACK(bfd_pack = trisc::bfd_alloc_and_program<trisc::BfdResource::Pack0>(
             tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);)
    PACK((_llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ReluConfig::none())));

    // Sets every client's dest base, so no per-thread dest init is needed.
    _llk_dest_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    run_schedule(TILE_CNT, ORDERS);
    _llk_dest_sync_finish_();
}
