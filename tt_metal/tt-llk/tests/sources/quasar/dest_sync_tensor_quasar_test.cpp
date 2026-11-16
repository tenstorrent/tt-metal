// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
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

#include "llk_dest_sync.h"
#include "params.h"

namespace
{

using ckernel::DestClient;

using T0 = ckernel::DestTensor<64, 2>;
using T1 = ckernel::DestTensor<64, 2>;
using T2 = ckernel::DestTensor<64, 2>;
using T3 = ckernel::DestTensor<64, 2>;

constexpr std::uint32_t TENSOR_CNT = 4;

constexpr std::uint32_t STEP_BITS         = 5;
constexpr std::uint32_t STEPS_PER_SECTION = 3;
constexpr std::uint32_t NO_STEP           = 0;
constexpr std::uint32_t CLIENT_MASK       = 0x7;
constexpr std::uint32_t TENSOR_SHIFT      = 3;

constexpr std::uint32_t code_of(const DestClient client)
{
    return ckernel::to_underlying(client) + 1;
}

inline std::uint32_t step_code(const std::uint32_t order_half, const std::uint32_t k)
{
    return (order_half >> (STEP_BITS * k)) & 0x1F;
}

inline float fill_value(const std::uint32_t i)
{
    return static_cast<float>(i + 1) / 8.0f;
}

UNPACK(std::uint32_t bfd_a;)
UNPACK(DataFormat to_dest_format;)
UNPACK(DataFormat to_src_format;)
MATH(std::uint32_t datacopy_rows;)
PACK(std::uint32_t bfd_pack;)
PACK(std::uint32_t pack_cnt = 0;)

template <typename Tensor>
struct UnpackToDestOp
{
    static constexpr DestClient CLIENT = DestClient::UNPACK;

    static void init()
    {
        UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(to_dest_format)));
        UNPACK((_llk_unpack_unary_operand_to_dest_init_(bfd_a)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        Tensor::template acquire<CLIENT>();
        UNPACK((_llk_unpack_unary_operand_to_dest_tile_(i, 0)));
    }
};

template <typename Tensor>
struct DatacopyOp
{
    static constexpr DestClient CLIENT = DestClient::FPU;

    static void init()
    {
        UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(to_src_format)));
        UNPACK((_llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, is_fp32_dest_acc_en>(bfd_a, ckernel::DEFAULT_TENSOR_SHAPE, 1)));
        MATH((_llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(datacopy_rows, 1)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        Tensor::template acquire<CLIENT>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, ckernel::DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0)));
    }
};

template <typename Tensor>
struct SquareOp
{
    static constexpr DestClient CLIENT = DestClient::SFPU;

    static void init()
    {
        SFPU((sfpu::init_square()));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, const bool produced)
    {
        Tensor::template acquire<CLIENT>();
        if (produced)
        {
            SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), 0, VectorMode::RC)));
        }
        else
        {
            SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, _calculate_fill_, (SFPU_ITERATIONS), 0, VectorMode::RC, fill_value(i))));
        }
    }
};

template <typename Tensor>
struct PackOp
{
    static constexpr DestClient CLIENT = DestClient::PACK;

    static void init()
    {
        PACK((_llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, const bool produced)
    {
        Tensor::template acquire<CLIENT>();
        if (produced)
        {
            PACK((_llk_pack_(0, pack_cnt, ckernel::DEFAULT_TENSOR_SHAPE)));
            PACK(++pack_cnt;)
        }
    }
};

#if defined(LLK_TRISC_MATH) || defined(LLK_TRISC_ISOLATE_SFPU)
inline void configure_alu(const DataFormat math_format)
{
    const DataFormat dest_format = is_fp32_dest_acc_en ? DataFormat::Float32 : math_format;
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    _configure_alu_formats_<is_fp32_dest_acc_en>(math_format, math_format, false, dest_format);
}
#endif

struct Schedule
{
    std::uint32_t inited    = 0;
    DestClient unpacker_mop = DestClient::PACK;
};

template <typename Op>
inline void run_op(const std::uint32_t code, const std::uint32_t i, const bool produced, Schedule& schedule)
{
    if (code != code_of(Op::CLIENT))
    {
        return;
    }
    constexpr bool USES_UNPACKER = Op::CLIENT == DestClient::UNPACK || Op::CLIENT == DestClient::FPU;
    if ((schedule.inited & (1u << ckernel::to_underlying(Op::CLIENT))) == 0 || (USES_UNPACKER && schedule.unpacker_mop != Op::CLIENT))
    {
        Op::init();
        schedule.inited |= 1u << ckernel::to_underlying(Op::CLIENT);
    }
    if constexpr (USES_UNPACKER)
    {
        schedule.unpacker_mop = Op::CLIENT;
    }
    Op::exec(i, produced);
}

template <typename Tensor>
inline void run_steps(const std::uint32_t client, const std::uint32_t i, const bool produced, Schedule& schedule)
{
    Tensor::allocate();
    run_op<UnpackToDestOp<Tensor>>(client, i, produced, schedule);
    run_op<DatacopyOp<Tensor>>(client, i, produced, schedule);
    run_op<SquareOp<Tensor>>(client, i, produced, schedule);
    run_op<PackOp<Tensor>>(client, i, produced, schedule);
}

inline void run_steps_of(const std::uint32_t tensor, const std::uint32_t client, const std::uint32_t i, const bool produced, Schedule& schedule)
{
    switch (tensor)
    {
        case 0:
            run_steps<T0>(client, i, produced, schedule);
            break;
        case 1:
            run_steps<T1>(client, i, produced, schedule);
            break;
        case 2:
            run_steps<T2>(client, i, produced, schedule);
            break;
        case 3:
            run_steps<T3>(client, i, produced, schedule);
            break;
        default:
            break;
    }
}

inline void destroy_of(const std::uint32_t tensor)
{
    switch (tensor)
    {
        case 0:
            T0::destroy();
            break;
        case 1:
            T1::destroy();
            break;
        case 2:
            T2::destroy();
            break;
        case 3:
            T3::destroy();
            break;
        default:
            break;
    }
}

inline void run_section(const std::uint32_t order_half, const std::uint32_t i, Schedule& schedule)
{
    bool touched[TENSOR_CNT] = {false, false, false, false};
    for (std::uint32_t k = 0; k < STEPS_PER_SECTION; ++k)
    {
        const std::uint32_t code = step_code(order_half, k);
        if ((code & CLIENT_MASK) == NO_STEP)
        {
            break;
        }
        touched[(code >> TENSOR_SHIFT) & 0x3] = true;
    }

    bool produced[TENSOR_CNT] = {false, false, false, false};
    for (std::uint32_t k = 0; k < STEPS_PER_SECTION; ++k)
    {
        const std::uint32_t code   = step_code(order_half, k);
        const std::uint32_t client = code & CLIENT_MASK;
        if (client == NO_STEP)
        {
            break;
        }
        const std::uint32_t tensor = (code >> TENSOR_SHIFT) & 0x3;
        run_steps_of(tensor, client, i, produced[tensor], schedule);
        produced[tensor] = client != code_of(DestClient::PACK);
    }
    for (std::uint32_t t = 0; t < TENSOR_CNT; ++t)
    {
        if (touched[t])
        {
            destroy_of(t);
        }
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

    UNPACK((bfd_a = trisc::bfd_alloc_and_program<trisc::BfdResource::Unp0>(
                tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src));)
    UNPACK(to_dest_format = static_cast<DataFormat>(formats.unpack_A_src);)
    UNPACK(to_src_format = static_cast<DataFormat>(formats.unpack_A_dst);)

    MATH((configure_alu(static_cast<DataFormat>(formats.math))));
    MATH(datacopy_rows = num_faces * TEST_FACE_R_DIM;)

    SFPU((configure_alu(static_cast<DataFormat>(formats.math))));
    SFPU((_llk_math_eltwise_sfpu_init_()));

    PACK(bfd_pack = trisc::bfd_alloc_and_program<trisc::BfdResource::Pack0>(
             tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);)
    PACK((_llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ReluConfig::none())));

    _llk_dest_sync_configure_();
    T0::seed();
    T1::seed();
    T2::seed();
    T3::seed();

    Schedule schedule;
    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
    {
        const std::uint32_t order_half = (ORDERS[i / 2] >> (16 * (i % 2))) & 0xFFFF;
        run_section(order_half, i, schedule);
    }
    _llk_dest_sync_flush_();
}
