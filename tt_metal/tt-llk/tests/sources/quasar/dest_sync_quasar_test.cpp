// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

#ifdef LLK_TRISC_UNPACK
#include "llk_bfd_alloc.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#define UNPACK(...) __VA_ARGS__
#else
#define UNPACK(...)
#endif

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#define MATH(...) __VA_ARGS__
#else
#define MATH(...)
#endif

#ifdef LLK_TRISC_ISOLATE_SFPU
#include "experimental/ckernel_sfpu_fill.h"
#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_srcs.h"
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

constexpr std::uint32_t STEP_BITS         = 3;
constexpr std::uint32_t STEPS_PER_SECTION = 5;
constexpr std::uint32_t NO_STEP           = 0;

constexpr std::uint32_t TENSOR_COUNT = dest_sync == ckernel::DstSync::SyncHalf ? 2 : 1;
constexpr std::uint32_t TENSOR_ROWS  = (ckernel::trisc::DEST_REGISTER_FULL_SIZE / TENSOR_COUNT) >> is_fp32_dest_acc_en;
constexpr ckernel::DestTensor tensor0 {0, TENSOR_ROWS};
constexpr ckernel::DestTensor tensor1 {TENSOR_ROWS * (TENSOR_COUNT - 1), TENSOR_ROWS};
std::uint32_t tensor_index = TENSOR_COUNT - 1;
DestClient previous        = DestClient::PACK;

template <DestClient CLIENT, const ckernel::DestTensor& TENSOR>
inline void take_tensor(const bool fresh)
{
    ckernel::_llk_dest_sync_take_<CLIENT, TENSOR>();
    if (fresh)
    {
        ckernel::_llk_dest_sync_clear_<TENSOR>();
    }
}

template <DestClient CLIENT>
inline void take_tensor()
{
    const bool fresh = previous == DestClient::PACK && CLIENT != DestClient::PACK;
    if (fresh)
    {
        tensor_index ^= TENSOR_COUNT - 1;
    }
    if (tensor_index == 0)
    {
        take_tensor<CLIENT, tensor0>(fresh);
    }
    else
    {
        take_tensor<CLIENT, tensor1>(fresh);
    }
    previous = CLIENT;
}

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

constexpr std::uint32_t client_bit(const DestClient client)
{
    return 1u << ckernel::to_underlying(client);
}

UNPACK(std::uint32_t bfd_a;)
UNPACK(std::uint32_t bfd_b;)
UNPACK(DataFormat to_dest_format;)
UNPACK(DataFormat to_src_format;)
UNPACK(DataFormat src_b_format;)
MATH(std::uint32_t datacopy_rows;)
MATH(DataFormat binary_math_format;)
PACK(std::uint32_t bfd_pack;)

struct UnpackToDestOp
{
    static constexpr DestClient CLIENT = DestClient::UNPACK;

    // Both producers share the unpacker, so each programs its own output format, after the unpacks in flight.
    static void init()
    {
        UNPACK((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::UNPACK0)));
        UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(to_dest_format)));
        UNPACK((_llk_unpack_unary_operand_to_dest_init_(bfd_a)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        // The unpacker writes the L1 format, Float32 when dest is 32-bit, so it addresses dest like every other client.
        take_tensor<CLIENT>();
        UNPACK((_llk_unpack_unary_operand_to_dest_tile_(i, 0 /*dest tile*/)));
    }
};

struct DatacopyOp
{
    static constexpr DestClient CLIENT = DestClient::FPU;

    // The unpacker goes back to unpacking to SrcA.
    static void init()
    {
        UNPACK((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::UNPACK0)));
        UNPACK((_llk_unpack_configure_unary_<p_unpacr::UNP_A>(to_src_format)));
        UNPACK((
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(bfd_a, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/)));
        MATH((_llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(datacopy_rows, 1 /*num_matrices*/)));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, [[maybe_unused]] const bool produced)
    {
        take_tensor<CLIENT>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, ckernel::DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0 /*dest tile*/)));
    }
};

struct SquareOp
{
    static constexpr DestClient CLIENT = DestClient::SFPU;

    static void init()
    {
        SFPU((sfpu::init_square()));
    }

    static void exec([[maybe_unused]] const std::uint32_t i, const bool produced)
    {
        take_tensor<CLIENT>();
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
        take_tensor<CLIENT>();
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
    if ((schedule.inited & client_bit(Op::CLIENT)) == 0 || (USES_UNPACKER && schedule.unpacker_mop != Op::CLIENT))
    {
        Op::init();
        schedule.inited |= client_bit(Op::CLIENT);
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

constexpr ckernel::DestTensor a {0, 64};
constexpr ckernel::DestTensor b {64, 128};

inline void run_tensor_schedule(const std::uint32_t tile_cnt)
{
    using namespace ckernel;
    DatacopyOp::init();
    SquareOp::init();
    PackOp::init();

    for (std::uint32_t i = 0; i < tile_cnt; ++i)
    {
        _llk_dest_sync_take_<DestClient::FPU, a>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0)));

        _llk_dest_sync_take_<DestClient::FPU, b>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(tile_cnt - 1 - i, DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0)));

        // FPU still owns both tensors; taking a again must retarget without handing b off.
        _llk_dest_sync_take_<DestClient::FPU, a>();
        UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_unary_datacopy_(0)));

        _llk_dest_sync_take_<DestClient::SFPU, b>();
        SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), 0, VectorMode::RC)));
        _llk_dest_sync_take_<DestClient::PACK, a>();
        PACK((_llk_pack_(0, 4 * i, DEFAULT_TENSOR_SHAPE)));

        _llk_dest_sync_take_<DestClient::SFPU, a>();
        SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), 0, VectorMode::RC)));
        _llk_dest_sync_take_<DestClient::SFPU, b>();
        SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), 0, VectorMode::RC)));

        _llk_dest_sync_take_<DestClient::PACK, b>();
        PACK((_llk_pack_(0, 4 * i + 1, DEFAULT_TENSOR_SHAPE)));
        _llk_dest_sync_take_<DestClient::PACK, a>();
        PACK((_llk_pack_(0, 4 * i + 2, DEFAULT_TENSOR_SHAPE)));
        _llk_dest_sync_take_<DestClient::PACK, b>();
        PACK((_llk_pack_(0, 4 * i + 3, DEFAULT_TENSOR_SHAPE)));
    }
    dst_sync::drain();
}

// Compute-API-shaped operations: tensor acquisition belongs to the operation, not the schedule.
namespace compute
{
using namespace ckernel;

template <const DestTensor& TENSOR>
inline void copy([[maybe_unused]] const std::uint32_t tile)
{
    static_assert(TENSOR.rows == 64, "This copy operation writes one full tile");
    UNPACK(mop_sync());
    MATH(mop_sync());
    _llk_dest_sync_take_<DestClient::FPU, TENSOR>();
    MATH((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH)));
    DatacopyOp::init();
    UNPACK((_llk_unpack_unary_operand_<p_unpacr::UNP_A>(tile, DEFAULT_TENSOR_SHAPE)));
    MATH((_llk_math_eltwise_unary_datacopy_(0 /*dest tile*/)));
}

template <const DestTensor& TENSOR>
inline void unpack_to_dest([[maybe_unused]] const std::uint32_t tile)
{
    static_assert(TENSOR.rows == 64, "This unpack operation writes one full tile");
    UNPACK(mop_sync());
    _llk_dest_sync_take_<DestClient::UNPACK, TENSOR>();
    UnpackToDestOp::init();
    UNPACK((_llk_unpack_unary_operand_to_dest_tile_(tile, 0 /*dest tile*/)));
}

template <const DestTensor& TENSOR>
inline void accumulate(const std::uint32_t first, const std::uint32_t second)
{
    static_assert(TENSOR.rows == 128, "The two input indices update two output tiles");
    UNPACK(mop_sync());
    MATH(mop_sync());
    _llk_dest_sync_take_<DestClient::FPU, TENSOR>();
    UNPACK((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::UNPACK1, p_stall::UNPACK0)));
    UNPACK((_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(to_src_format, src_b_format)));
    UNPACK((_llk_unpack_binary_operands_init_(bfd_a, bfd_b, DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/)));
    MATH((TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH)));
    MATH((_llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, MathFidelity::LoFi>(
        binary_math_format, binary_math_format, DEFAULT_TENSOR_SHAPE, true /*acc_to_dest*/)));
    for (std::uint32_t tile = 0; tile < 2; ++tile)
    {
        [[maybe_unused]] const auto input = tile == 0 ? first : second;
        UNPACK((_llk_unpack_binary_operands_(input, input, DEFAULT_TENSOR_SHAPE)));
        MATH((_llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD>(tile, DEFAULT_TENSOR_SHAPE)));
    }
}

template <const DestTensor& TENSOR>
inline void square()
{
    _llk_dest_sync_take_<DestClient::SFPU, TENSOR>();
    SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), 0, VectorMode::RC)));
}

template <const DestTensor& TENSOR>
inline void fill([[maybe_unused]] const float value)
{
    _llk_dest_sync_take_<DestClient::SFPU, TENSOR>();
    SFPU((SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, _calculate_fill_, (SFPU_ITERATIONS), 0, VectorMode::RC, value)));
}

template <const DestTensor& TENSOR>
inline void pack([[maybe_unused]] const std::uint32_t output)
{
    _llk_dest_sync_take_<DestClient::PACK, TENSOR>();
    for (std::uint32_t tile = 0; tile < TENSOR.rows / 64; ++tile)
    {
        PACK((_llk_pack_(tile, output + tile, DEFAULT_TENSOR_SHAPE)));
    }
}

#ifdef LLK_TRISC_ISOLATE_SFPU
constexpr auto REGISTER_LAYOUT = is_fp32_dest_acc_en ? sfpi::DataLayout::F32 : sfpi::DataLayout::F16b;
using RegisterFormat           = sfpu::SfpiFormat<REGISTER_LAYOUT, sfpi::vFloat>;
using DestOperand              = sfpu::SfpuOperand<sfpu::SfpuReg::Dest, RegisterFormat>;
using SrcsOperand              = sfpu::SfpuOperand<sfpu::SfpuReg::SrcS, RegisterFormat>;
using SrcsLayout               = sfpu::SrcsLayout<REGISTER_LAYOUT>;

struct Srcs
{
    static constexpr auto SLICES = srcs_dims::slice_count(is_fp32_dest_acc_en);
    static inline std::uint32_t input_bfd;
    static inline std::uint32_t output_bfd;

    static void init(const std::uint32_t input, const std::uint32_t output, const DataFormat input_format, const DataFormat output_format)
    {
        const auto shape = make_tensor_shape(SrcsLayout::rows, srcs_dims::XDIM, 1, 1);
        input_bfd        = trisc::bfd_alloc_and_program<trisc::BfdResource::Unp2_Slice0>(shape, input, to_underlying(input_format));
        output_bfd       = trisc::bfd_alloc_and_program<trisc::BfdResource::Pack1>(shape, output, to_underlying(output_format));
        _llk_unpack_configure_unary_<p_unpacr::UNP_S>(SrcsLayout::format);
        _llk_pack_hw_configure_<p_pacr::PACK1, false>(SrcsLayout::format, ReluConfig::none());
        _llk_unpack_srcs_config_for_tile_<1>(is_fp32_dest_acc_en);
        _llk_pack_srcs_config_for_tile_<1>(is_fp32_dest_acc_en);
    }

    template <bool SQUARE>
    static void to_dest(const std::uint32_t input)
    {
        _llk_math_eltwise_sfpu_start_(0 /*tile_index*/);
        _llk_unpack_srcs_<1>(input_bfd, input * SLICES);
        for (std::uint32_t slice = 0; slice < SLICES; ++slice)
        {
            for (int d = 0; d < SrcsLayout::ops; ++d)
            {
                sfpi::vFloat value = SrcsOperand {SrcsLayout::in0}.load(d);
                if constexpr (SQUARE)
                {
                    value = value * value;
                }
                DestOperand {}.store(slice * SrcsLayout::ops + d, value);
            }
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, false>();
        }
        _llk_math_eltwise_sfpu_done_();
    }

    static void from_dest(const std::uint32_t tile, const std::uint32_t output)
    {
        _llk_math_eltwise_sfpu_start_(tile);
        // Queue PACK1 before producing slices so instruction-buffer backpressure cannot hide the consumer.
        _llk_pack_srcs_<1>(output_bfd, output * SLICES);
        for (std::uint32_t slice = 0; slice < SLICES; ++slice)
        {
            for (int d = 0; d < SrcsLayout::ops; ++d)
            {
                const sfpi::vFloat value = DestOperand {}.load(slice * SrcsLayout::ops + d);
                SrcsOperand {SrcsLayout::out}.store(d, value);
            }
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<false, true>();
        }
        _llk_math_eltwise_sfpu_done_();
    }

    static void square(const std::uint32_t input, const std::uint32_t output)
    {
        _llk_unpack_srcs_<1>(input_bfd, input * SLICES);
        _llk_pack_srcs_<1>(output_bfd, output * SLICES);
        for (std::uint32_t slice = 0; slice < SLICES; ++slice)
        {
            sfpu::SquareSrcs<REGISTER_LAYOUT>::calculate();
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>();
        }
    }
};
#endif

template <const DestTensor& TENSOR, bool SQUARE>
inline void srcs_to_dest([[maybe_unused]] const std::uint32_t input)
{
    static_assert(TENSOR.rows == 64, "The SrcS auto-loop transfers one full tile");
    _llk_dest_sync_take_<DestClient::SFPU, TENSOR>();
    SFPU((Srcs::to_dest<SQUARE>(input)));
}

template <const DestTensor& TENSOR>
inline void dest_to_srcs([[maybe_unused]] const std::uint32_t output)
{
    _llk_dest_sync_take_<DestClient::SFPU, TENSOR>();
#ifdef LLK_TRISC_ISOLATE_SFPU
    for (std::uint32_t tile = 0; tile < TENSOR.rows / 64; ++tile)
    {
        Srcs::from_dest(tile, output + tile);
    }
#endif
}

template <const DestTensor& OUTPUT, const DestTensor& INPUT, bool HALF>
inline void add_scaled()
{
    _llk_dest_sync_take_<DestClient::SFPU, OUTPUT>();
    _llk_dest_sync_take_<DestClient::SFPU, INPUT>();
    constexpr auto& base = OUTPUT.address < INPUT.address ? OUTPUT : INPUT;
    _llk_dest_sync_take_<DestClient::SFPU, base>();
#ifdef LLK_TRISC_ISOLATE_SFPU
    _llk_math_eltwise_sfpu_start_(0 /*tile_index*/);
    const DestOperand input {(INPUT.address - base.address) / 2};
    const DestOperand output {(OUTPUT.address - base.address) / 2};
    for (std::uint32_t d = 0; d < OUTPUT.rows / 2; ++d)
    {
        const sfpi::vFloat previous = output.load(d);
        const sfpi::vFloat update   = input.load(d % 32);
        output.store(d, previous + update * (HALF ? 0.5f : 1.0f));
    }
    _llk_math_eltwise_sfpu_done_();
#endif
}

template <const DestTensor& TENSOR>
inline void scale_half()
{
    _llk_dest_sync_take_<DestClient::SFPU, TENSOR>();
#ifdef LLK_TRISC_ISOLATE_SFPU
    _llk_math_eltwise_sfpu_start_(0 /*tile_index*/);
    for (std::uint32_t d = 0; d < TENSOR.rows / 2; ++d)
    {
        DestOperand {}.store(d, DestOperand {}.load(d) * 0.5f);
    }
    _llk_math_eltwise_sfpu_done_();
#endif
}
} // namespace compute

// SDPA-shaped lifetimes: scores are recycled each chunk, history and output survive until the tail.
constexpr auto HALF_ROWS   = (ckernel::trisc::DEST_REGISTER_FULL_SIZE / 2) >> is_fp32_dest_acc_en;
constexpr auto SCORE_BASE  = dest_swap_halves ? HALF_ROWS : 0;
constexpr auto OUTPUT_BASE = dest_swap_halves ? 0 : HALF_ROWS;
constexpr ckernel::DestTensor scores {SCORE_BASE, 64};
constexpr ckernel::DestTensor history {SCORE_BASE + 64, 64};
constexpr ckernel::DestTensor output {OUTPUT_BASE, 128};
constexpr ckernel::DestTensor guard {OUTPUT_BASE + HALF_ROWS - 64, 64};

template <typename Params>
inline void run_split_pipeline(const Params& params)
{
    using namespace ckernel;
    static_assert(dest_srcs_mode <= 3, "Unknown score producer");
    const auto chunks = params.TILE_CNT;
    UNPACK((bfd_b = trisc::bfd_alloc_and_program<trisc::BfdResource::Unp1>(DEFAULT_TENSOR_SHAPE, L1_ADDRESS(params.buffer_B[0]), params.formats.unpack_B_src)));
    UNPACK(src_b_format = static_cast<DataFormat>(params.formats.unpack_B_dst);)
    MATH(binary_math_format = static_cast<DataFormat>(params.formats.math);)
    SFPU((compute::Srcs::init(
        L1_ADDRESS(params.buffer_S[0]),
        L1_ADDRESS(params.buffer_C[0]),
        static_cast<DataFormat>(params.formats.unpack_S_src),
        static_cast<DataFormat>(params.formats.pack_S_dst))));
    SquareOp::init();
    PackOp::init();
    _llk_dest_sync_take_<DestClient::FPU, output>();
    _llk_dest_sync_clear_<output>();
    compute::fill<history>(0.0f);
    compute::fill<guard>(-0.75f);

    for (std::uint32_t chunk = 0; chunk < chunks; ++chunk)
    {
        if constexpr (dest_srcs_mode == 0)
        {
            compute::unpack_to_dest<scores>(chunk);
        }
        else if constexpr (dest_srcs_mode == 3)
        {
            compute::copy<scores>(chunk);
        }
        else
        {
            compute::srcs_to_dest<scores, dest_srcs_mode == 2>(chunk);
        }
        if constexpr (dest_srcs_mode != 2)
        {
            compute::square<scores>();
        }
        if (chunk != 0)
        {
            compute::add_scaled<output, history, true>();
        }
        compute::accumulate<output>(chunk, chunks - 1 - chunk);
        compute::add_scaled<history, scores, false>();

        SFPU((compute::Srcs::square(chunk, 4 * chunk)));
        compute::dest_to_srcs<scores>(4 * chunk + 1);
        compute::dest_to_srcs<output>(4 * chunk + 2);
        compute::pack<scores>(2 * chunk);
        _llk_dest_sync_clear_<scores>();
        compute::pack<history>(2 * chunk + 1);
    }

    compute::scale_half<output>();
    compute::pack<output>(2 * chunks);
    compute::pack<guard>(2 * chunks + 2);
    SFPU((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::UNPACK2, p_stall::PACK1, p_stall::WAIT_SFPU)));
    dst_sync::drain();
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

    _llk_dest_sync_configure_<dest_sync, is_fp32_dest_acc_en>();
    if constexpr (dest_split_pipeline)
    {
        run_split_pipeline(params);
    }
    else if constexpr (dest_tensor_test)
    {
        run_tensor_schedule(TILE_CNT);
    }
    else
    {
        run_schedule(TILE_CNT, ORDERS);
    }
}
