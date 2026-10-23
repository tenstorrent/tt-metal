// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_typecast.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifndef ARCH_QUASAR
#include "sanitizer/api.h"
#include "sfpu/dispatch.h"
#endif
#endif

namespace ckernel {

#ifdef ARCH_QUASAR
namespace detail {
// MX / block-float typecasts are a pure unpack/pack format conversion (a datacopy) on Quasar:
// the unpacker converts MX -> float into Dest and the packer converts float -> MX on the way out,
// so no SFPU op runs. This mirrors the BH "handled by unpacker/packer" no-op bfp arms.
inline constexpr bool _typecast_is_mx_format_(DataFormat fmt) {
    return fmt == DataFormat::MxFp8R || fmt == DataFormat::MxFp8P || fmt == DataFormat::MxFp6R ||
           fmt == DataFormat::MxFp6P || fmt == DataFormat::MxFp4 || fmt == DataFormat::MxInt8 ||
           fmt == DataFormat::MxInt4 || fmt == DataFormat::MxInt2;
}

// Float16_b <-> Float32 does not need an SFPU op. Dest already holds the value as Float32 (widen by
// setting Dest to 32-bit; narrow by letting the packer emit Float16_b from that Float32 Dest).
inline constexpr bool _typecast_is_sfpu_no_op_(DataFormat src, DataFormat dst) {
    return (src == DataFormat::Float16_b && dst == DataFormat::Float32) ||
           (src == DataFormat::Float32 && dst == DataFormat::Float16_b);
}
}  // namespace detail
#endif

#if defined(TRISC_MATH) && !defined(ARCH_QUASAR)
namespace sfpu {

/**
 * @brief Typecast of a Dest tile from IN to OUT, as an llk::sfpu::UnaryDispatch op.
 *
 * Every pair shares one init (ADDR_MOD_6, then the pair's own init if it has one). Execute walks the
 * tile one face at a time with the pair's SFPU kernel, in place. Pairs the unpacker / packer convert
 * on their own have no kernel (@ref kernel() is None) and are never executed.
 *
 * @tparam IN: Input data format
 * @tparam OUT: Output data format
 * @tparam is_fp32_dest_acc_en: Whether Dest is in 32 bit mode
 */
template <DataFormat IN, DataFormat OUT, bool is_fp32_dest_acc_en>
struct Typecast : llk::sfpu::SfpuUnary {
    /// SFPU kernel a pair runs: the same kernel serves several pairs.
    enum class Kernel {
        None,
        Fp32ToFp16b,
        Fp32ToInt32,
        Fp32ToUint16,
        Fp32ToUint32,
        Fp32ToUint8,
        Int32ToFp16b,
        Int32ToFp32,
        Int32ToUint16,
        Uint16ToFp16b,
        Uint16ToFp32,
        Uint16ToUint32,
        Uint32ToFp16b,
        Uint32ToFp32,
        Uint32ToUint16,
        UintToUint8,
        Int8ToInt32,
        Int8ToFp32,
    };

    static constexpr Kernel kernel() {
        // Float inputs are Float32 or Float16_b in Dest; Float16_b-class outputs are packed from Float16_b.
        constexpr bool from_float = IN == DataFormat::Float32 || IN == DataFormat::Float16_b ||
                                    IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b;
        constexpr bool to_fp16b =
            OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b;
        constexpr bool to_8bit = OUT == DataFormat::UInt8 || OUT == DataFormat::Int8;
        // UInt8 is held zero-extended in Dest, so it converts like UInt32.
        constexpr bool from_uint32 = IN == DataFormat::UInt32 || IN == DataFormat::UInt8;

        if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Float16_b) {
            return Kernel::Fp32ToFp16b;
        } else if constexpr (from_float && OUT == DataFormat::Int32) {
            return Kernel::Fp32ToInt32;
        } else if constexpr (from_float && OUT == DataFormat::UInt16) {
            return Kernel::Fp32ToUint16;
        } else if constexpr (from_float && OUT == DataFormat::UInt32) {
            return Kernel::Fp32ToUint32;
        } else if constexpr (from_float && to_8bit) {
            return Kernel::Fp32ToUint8;
        } else if constexpr (IN == DataFormat::Int32 && to_fp16b) {
            return Kernel::Int32ToFp16b;
        } else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float32) {
            return Kernel::Int32ToFp32;
        } else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::UInt16) {
            return Kernel::Int32ToUint16;
        } else if constexpr (IN == DataFormat::UInt16 && to_fp16b) {
            return Kernel::Uint16ToFp16b;
        } else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float32) {
            return Kernel::Uint16ToFp32;
        } else if constexpr (IN == DataFormat::UInt16 && (OUT == DataFormat::UInt32 || OUT == DataFormat::Int32)) {
            return Kernel::Uint16ToUint32;
        } else if constexpr (from_uint32 && to_fp16b) {
            return Kernel::Uint32ToFp16b;
        } else if constexpr (from_uint32 && OUT == DataFormat::Float32) {
            return Kernel::Uint32ToFp32;
        } else if constexpr (from_uint32 && OUT == DataFormat::UInt16) {
            return Kernel::Uint32ToUint16;
        } else if constexpr (
            (IN == DataFormat::Int32 || IN == DataFormat::UInt32 || IN == DataFormat::UInt16) && to_8bit) {
            return Kernel::UintToUint8;
        } else if constexpr (
            IN == DataFormat::Int8 &&
            (OUT == DataFormat::Int32 || OUT == DataFormat::UInt32 || OUT == DataFormat::UInt16)) {
            return Kernel::Int8ToInt32;
        } else if constexpr (IN == DataFormat::Int8 && (OUT == DataFormat::Float32 || to_fp16b)) {
            return Kernel::Int8ToFp32;
        } else {
            // The unpacker / packer convert the rest (e.g. Bfp8_b <-> Float16_b, Float16_b -> Float32,
            // UInt8 <-> Int8); a pair not listed above is not a supported typecast and does nothing.
            return Kernel::None;
        }
    }

    /// Elementwise: a lane's result depends on that lane alone, so padding lanes may be converted too.
    static constexpr llk::sfpu::PaddingAgnostic padding_agnostic() { return llk::sfpu::PaddingAgnostic::Yes; }

    struct Init : llk::sfpu::UnaryInit<Typecast> {
        void run() const {
            typecast_init();

            constexpr Kernel KERNEL = kernel();
            if constexpr (KERNEL == Kernel::Fp32ToFp16b) {
                init_typecast_fp32_to_fp16b<APPROX>();
            } else if constexpr (KERNEL == Kernel::Fp32ToUint16) {
                init_typecast_fp32_to_uint16<APPROX>();
            } else if constexpr (KERNEL == Kernel::Fp32ToUint8) {
                init_typecast_fp32_to_uint8<APPROX>();
            } else if constexpr (KERNEL == Kernel::Int32ToFp16b) {
                init_typecast_int32_to_fp16b<APPROX>();
            } else if constexpr (KERNEL == Kernel::Int32ToFp32) {
                init_typecast_int32_to_fp32<APPROX>();
            } else if constexpr (KERNEL == Kernel::Int32ToUint16) {
                init_typecast_int32_to_uint16<APPROX>();
            } else if constexpr (KERNEL == Kernel::Uint16ToFp16b) {
                init_typecast_uint16_to_fp16b<APPROX>();
            } else if constexpr (KERNEL == Kernel::Uint16ToFp32) {
                init_typecast_uint16_to_fp32<APPROX>();
            } else if constexpr (KERNEL == Kernel::Uint16ToUint32) {
                init_typecast_uint16_to_uint32<APPROX>();
            } else if constexpr (KERNEL == Kernel::Uint32ToFp16b) {
                init_typecast_uint32_to_fp16b<APPROX>();
            } else if constexpr (KERNEL == Kernel::Uint32ToFp32) {
                init_typecast_uint32_to_fp32<APPROX>();
            } else if constexpr (KERNEL == Kernel::Uint32ToUint16) {
                init_typecast_uint32_to_uint16<APPROX>();
            } else if constexpr (KERNEL == Kernel::UintToUint8) {
                init_typecast_uint_to_uint8<APPROX>();
            } else if constexpr (KERNEL == Kernel::Int8ToInt32) {
                init_typecast_int8_input<APPROX>();
            } else if constexpr (KERNEL == Kernel::Int8ToFp32) {
                init_typecast_int8_to_fp32<APPROX>();
            }
            // Fp32ToInt32, Fp32ToUint32 and None need no init of their own.
        }
    };

    template <const TensorShape& SHAPE, llk::sfpu::PaddingAgnostic PADDING>
    struct Execute : llk::sfpu::UnaryExecute<Typecast> {
        void run() const {
            static_assert(kernel() != Kernel::None, "this typecast has no SFPU kernel; do not execute it");
            // The kernels convert whole 16x16 faces.
            static_assert(
                PADDING == llk::sfpu::PaddingAgnostic::Yes ||
                    (SHAPE.face_r_dim == MAX_FACE_R_DIM && SHAPE.face_c_dim == MAX_FACE_C_DIM),
                "typecast converts whole faces, so a padding-sensitive call needs full faces");
            LLK_ASSERT(this->output == this->input, "typecast runs in place: output must equal input");

            _llk_math_eltwise_sfpu_start_(this->output);
            llk::sfpu::for_each_face<SHAPE>([]() __attribute__((always_inline)) { face(); });
            _llk_math_eltwise_sfpu_done_();
        }
    };

    static void sanitizer_init() {
        SAN_HOOK(llk::san::init<OperationSfpuTypecast>(
            StateVal<OperationSfpuTypecast::InputFormat>(static_cast<std::uint32_t>(IN)),
            StateVal<OperationSfpuTypecast::OutputFormat>(static_cast<std::uint32_t>(OUT))));
    }

    static void sanitizer_execute(const std::uint32_t output, const std::uint32_t input) {
        SAN_HOOK(llk::san::execute<OperationSfpuTypecast>(
            StateVal<OperationSfpuTypecast::InputFormat>(static_cast<std::uint32_t>(IN)),
            StateVal<OperationSfpuTypecast::OutputFormat>(static_cast<std::uint32_t>(OUT)),
            StateDiscard<std::uint32_t>(output),
            StateDiscard<std::uint32_t>(input)));
    }

private:
    static constexpr int ITERATIONS = 8;  // one face

    /// Convert the face at the current Dest address.
    static inline __attribute__((always_inline)) void face() {
        constexpr Kernel KERNEL = kernel();
        if constexpr (KERNEL == Kernel::Fp32ToFp16b) {
            calculate_typecast_fp32_to_fp16b<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Fp32ToInt32) {
            calculate_typecast_fp32_to_int32<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Fp32ToUint16) {
            calculate_typecast_fp32_to_uint16<APPROX, ITERATIONS, is_fp32_dest_acc_en>();
        } else if constexpr (KERNEL == Kernel::Fp32ToUint32) {
            calculate_typecast_fp32_to_uint32<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Fp32ToUint8) {
            calculate_typecast_fp32_to_uint8<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Int32ToFp16b) {
            calculate_typecast_int32_to_fp16b<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Int32ToFp32) {
            calculate_typecast_int32_to_fp32<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Int32ToUint16) {
            calculate_typecast_int32_to_uint16<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Uint16ToFp16b) {
            calculate_typecast_uint16_to_fp16b<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Uint16ToFp32) {
            calculate_typecast_uint16_to_fp32<APPROX, ITERATIONS, is_fp32_dest_acc_en>();
        } else if constexpr (KERNEL == Kernel::Uint16ToUint32) {
            calculate_typecast_uint16_to_uint32<APPROX, ITERATIONS, is_fp32_dest_acc_en>();
        } else if constexpr (KERNEL == Kernel::Uint32ToFp16b) {
            calculate_typecast_uint32_to_fp16b<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Uint32ToFp32) {
            calculate_typecast_uint32_to_fp32<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::Uint32ToUint16) {
            calculate_typecast_uint32_to_uint16<APPROX, ITERATIONS>();
        } else if constexpr (KERNEL == Kernel::UintToUint8) {
            calculate_typecast_uint_to_uint8<APPROX, ITERATIONS, IN == DataFormat::UInt16>();
        } else if constexpr (KERNEL == Kernel::Int8ToInt32) {
            calculate_typecast_int8_to_int32<APPROX, ITERATIONS, OUT == DataFormat::UInt16 /* CLAMP_TO_UINT16 */>();
        } else if constexpr (KERNEL == Kernel::Int8ToFp32) {
            calculate_typecast_int8_to_fp32<APPROX, ITERATIONS>();
        }
    }
};

}  // namespace sfpu
#endif

// clang-format off
/**
 * Performs an elementwise typecast operation on the input.
 * Supports following typecasts:
 *  Float16_b <-> Float32
 *  Float16_b <-> Int32
 *  Float16_b <-> UInt16
 *  Float16_b <-> UInt32
 *  Float16_b <-> UInt8
 *  Float16_b <-> Int8
 *  Float32 <-> Int32
 *  Float32 <-> UInt16
 *  Float32 <-> UInt32
 *  Float32 <-> UInt8
 *  Float32 <-> Int8
 *  Bfp8_b <-> Int32
 *  Bfp8_b <-> UInt16
 *  Bfp8_b <-> UInt32
 *  Bfp8_b <-> UInt8
 *  Bfp8_b <-> Int8
 *  Bfp8_b <-> Float16_b
 *  Bfp8_b <-> Float32
 *  Bfp4_b <-> Int32
 *  Bfp4_b <-> UInt16
 *  Bfp4_b <-> UInt32
 *  Bfp4_b <-> UInt8
 *  Bfp4_b <-> Int8
 *  Bfp4_b <-> Bfp8_b
 *  Bfp4_b <-> Float16_b
 *  Bfp4_b <-> Float32
 *  UInt16 <-> UInt32
 *  UInt16 <-> Int32
 *  UInt16 <-> UInt8
 *  UInt16 <-> Int8
 *  Int32 <-> Int8
 *  UInt32 <-> Int8
 *  UInt8 <-> Int8
 *
 * For input/output to be UInt32, Int32, or Float32, Dest must be in 32 bit mode.
 *
 * For input/output to be Int8, the caller must additionally declare the circular buffers as UInt8 instead
 * of Int8, so the raw 2's complement byte is zero-extended instead of being decoded as sign-magnitude,
 * and must put Dest in 32 bit mode, which is enforced by a static_assert. The kernels below do the sign
 * handling themselves on that raw byte.
 * Int8 is not available on Quasar.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform typecast operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | IN_DTYPE       | Input data format                                                          | uint32_t | Must be valid tt::DataFormat                          | True     |
 * | OUT_DTYPE      | Desired output data format                                                 | uint32_t | Must be valid tt::DataFormat                          | True     |
 */
// clang-format on
template <std::uint32_t IN_DTYPE, std::uint32_t OUT_DTYPE, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void typecast_tile(std::uint32_t idst) {
    constexpr DataFormat in_format = static_cast<DataFormat>(IN_DTYPE);
    constexpr DataFormat out_format = static_cast<DataFormat>(OUT_DTYPE);

#ifdef ARCH_QUASAR
    // The shared Quasar kernel loads and stores Int8 as sign-magnitude, but callers configure Int8 CBs
    // as UInt8. The byte would be decoded wrong with no error. Reject it here.
    static_assert(
        in_format != DataFormat::Int8 && out_format != DataFormat::Int8, "Int8 typecast is not supported on Quasar");

    // An MX endpoint is unpacked to / packed from Float16_b by the format, so at the SFPU level an MX
    // format behaves as Float16_b. Route through that effective format: MX <-> Float16_b (and MX <-> MX)
    // collapse to a pure format no-op, and so does MX <-> Float32, which reaches the Float16_b <-> Float32
    // no-op arm. The rest (MX <-> {Int32, ...}) run the Float16_b <-> X SFPU conversion on top of the
    // format (X -> MX runs X -> Float16_b, then the packer emits MX).
    constexpr DataFormat effective_input_format =
        detail::_typecast_is_mx_format_(in_format) ? DataFormat::Float16_b : in_format;
    constexpr DataFormat effective_output_format =
        detail::_typecast_is_mx_format_(out_format) ? DataFormat::Float16_b : out_format;
    constexpr bool is_sfpu_no_op = detail::_typecast_is_sfpu_no_op_(effective_input_format, effective_output_format);
    if constexpr (effective_input_format != effective_output_format && !is_sfpu_no_op) {
        // Single unified Quasar typecast kernel, templated on the effective source/destination formats.
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_typecast,
            (effective_input_format, effective_output_format, SFPU_ITERATIONS),
            idst,
            VectorMode::RC));
    }
#else
    static_assert(
        is_fp32_dest_acc_en || (in_format != DataFormat::Int8 && out_format != DataFormat::Int8),
        "Int8 typecast requires Dest in 32 bit mode");

#ifdef TRISC_MATH
    using Op = sfpu::Typecast<in_format, out_format, is_fp32_dest_acc_en>;
    if constexpr (Op::kernel() != Op::Kernel::None) {
        llk::sfpu::UnaryDispatch<Op>::template execute<DST_SYNC_MODE, DEFAULT_TENSOR_SHAPE>(idst, idst).run();
    }
#endif
#endif  // ARCH_QUASAR
}

/**
 * Please refer to documentation for any_init.
 */
template <std::uint32_t IN_DTYPE, std::uint32_t OUT_DTYPE>
ALWI void typecast_tile_init() {
    constexpr DataFormat in_format = static_cast<DataFormat>(IN_DTYPE);
    constexpr DataFormat out_format = static_cast<DataFormat>(OUT_DTYPE);

#ifdef ARCH_QUASAR
    // Mirror typecast_tile: an MX endpoint behaves as Float16_b at the SFPU level, so only a pair that
    // dispatches an SFPU op needs the init. The no-op pairs (MX <-> Float16_b, MX <-> MX, and
    // Float16_b/MX <-> Float32) are gated out here too, so init and execute stay in lockstep.
    constexpr DataFormat effective_input_format =
        detail::_typecast_is_mx_format_(in_format) ? DataFormat::Float16_b : in_format;
    constexpr DataFormat effective_output_format =
        detail::_typecast_is_mx_format_(out_format) ? DataFormat::Float16_b : out_format;
    constexpr bool is_sfpu_no_op = detail::_typecast_is_sfpu_no_op_(effective_input_format, effective_output_format);
    if constexpr (effective_input_format != effective_output_format && !is_sfpu_no_op) {
        MATH(SFPU_UNARY_INIT(typecast, sfpu::init_typecast));
    }
#else
    // Every pair runs the init, including those with no SFPU kernel to execute.
    MATH((llk::sfpu::UnaryDispatch<sfpu::Typecast<in_format, out_format, DST_ACCUM_MODE>>::init().run()));
#endif  // ARCH_QUASAR
}

}  // namespace ckernel
