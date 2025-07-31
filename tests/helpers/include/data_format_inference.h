// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include "build.h"
#include "tensix_types.h"

#if defined(ARCH_WORMHOLE) && defined(ARCH_BLACKHOLE)
#error "Only one of ARCH_WORMHOLE or ARCH_BLACKHOLE can be defined"
#elif defined(ARCH_WORMHOLE)
constexpr bool is_blackhole = false;
constexpr bool is_wormhole  = true;
#elif defined(ARCH_BLACKHOLE)
constexpr bool is_blackhole = true;
constexpr bool is_wormhole  = false;
#else
#error "You must define either ARCH_WORMHOLE or ARCH_BLACKHOLE"
#endif

/**
 * @struct FormatConfig
 * @brief Holds data format configurations for each stage of compute pipeline.
 * including unpacking, math operations, and packing.
 *
 * Each member represents the data format used at a specific stage:
 * - unpack_src: unpacker input format found in L1.
 * - unpack_dst: unpacker output format when unpacking from L1 to the register(s).
 * - math: math format used during compute operations and storing in dest register.
 * - pack_src: packer input format, when packing from dest register to L1.
 * - pack_dst: packer output format, desired result format in L1.
 */
struct FormatConfig
{
    const uint32_t unpack_src;
    const uint32_t unpack_dst;
    const uint32_t math;
    const uint32_t pack_src;
    const uint32_t pack_dst;

    constexpr FormatConfig(uint32_t unpack_src_, uint32_t unpack_dst_, uint32_t math_, uint32_t pack_src_, uint32_t pack_dst_) :
        unpack_src(unpack_src_), unpack_dst(unpack_dst_), math(math_), pack_src(pack_src_), pack_dst(pack_dst_)
    {
    }
};

constexpr bool is_exponentB(DataFormat format)
{
    // Return true if format has an exponentB representation i.e 8-bit exponent
    return (format == DataFormat::Float16_b || format == DataFormat::Bfp8_b || format == DataFormat::Tf32);
}

constexpr bool is_32bit_format(DataFormat format)
{
    return format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32;
}

/**
 * Checks if the given input/output format combination is an outlier case
 * that is unsupported by hardware and requires a workaround.
 *
 * This outlier case occurs when converting an 8-bit exponent format datum
 * directly to Float16 without using an intermediate Float32 representation
 * in the dest register.
 *
 * To handle this hardware limitation, the destination register stores 32-bit datums,
 * and the packer input format is converted to Float32.
 *
 * @param input The input data format in L1.
 * @param output The output data format in L1.
 * @param is_fp32_dest_acc_en Flag indicating if 32-bit destination accumulation is enabled (dest_acc).
 *
 * @return true if the format combination is an unsupported hardware outlier; false otherwise.
 */
constexpr bool is_format_combination_outlier(DataFormat input, DataFormat output, bool is_fp32_dest_acc_en)
{
    return (is_exponentB(input) && output == DataFormat::Float16 && !is_fp32_dest_acc_en);
}

/**
 * Returns the output format for the packer based on the input format in L1
 * and whether unpacking targets the source or dest register.
 *
 * @tparam INPUT The input data format.
 * @return The inferred output data format for unpacking.
 */
constexpr FormatConfig get_data_formats(DataFormat unpack_in, DataFormat unpack_out, DataFormat math, DataFormat pack_in, DataFormat pack_out)
{
    return {
        static_cast<std::underlying_type_t<DataFormat>>(unpack_in),
        static_cast<std::underlying_type_t<DataFormat>>(unpack_out),
        static_cast<std::underlying_type_t<DataFormat>>(math),
        static_cast<std::underlying_type_t<DataFormat>>(pack_in),
        static_cast<std::underlying_type_t<DataFormat>>(pack_out)};
}

/**
 * Returns the output format for the unpacker (data format config for registers)
 * based on the input format in L1 and whether unpacking targets the source or destination register.
 *
 * @tparam INPUT The data format currently stored in L1 cache.
 * @return The inferred output data format for unpacking to registers.
 *
 * Uses the global constexpr:
 *   - UNPACKING_TO_DEST: Indicates whether unpacking targets the destination register.
 */
template <DataFormat INPUT, DataFormat OUTPUT, bool FP32_ACC>
constexpr DataFormat infer_unpack_out()
{
    if constexpr (INPUT == DataFormat::Float32 && !UNPACKING_TO_DEST)
    {
        // When input format in L1 is Float32 + unpacking to src registers (instead of directly to dest register)
        // Source registers can store 19-bit values, so we truncate Float32 to Tf32 if we know dest will be 32-bit format
        // which preserves the 8-bit exponent and as much mantissa precision as fits. If our dst regoster is 16-bit we directly truncate to 16-bit format
        if constexpr (FP32_ACC)
        {
            return DataFormat::Tf32;
        }
        else if constexpr (is_exponentB(OUTPUT) || OUTPUT == DataFormat::Float32)
        {
            return DataFormat::Float16_b; // If output Float32 or Float16_b
        }
        return DataFormat::Float16; // Tilize to Float16
    }
    // For all other cases, we can keep the format the same in L1 and src register or dest register
    return INPUT;
}

/**
 * Infers all data formats needed for unpacking, math, and packing stages in a pipeline.
 *
 * @tparam INPUT   Input data format in L1 (unpacker input)
 * @tparam OUTPUT  Final output data format after packing
 * @tparam FP32_ACC  Flag indicating if FP32 accumulation is enabled
 *
 * @return FormatConfig struct containing all formats
 */
template <DataFormat INPUT, DataFormat OUTPUT, DataFormat unpack_out, bool FP32_ACC>
constexpr DataFormat infer_pack_in()
{
    if constexpr (is_wormhole && FP32_ACC && OUTPUT == DataFormat::Float16)
    {
        // On wormhole architecture, data stored as Float32 in dest register,
        // gasket cannot convert Float32 ->Float16_A, so it leaves the data as Float32,
        // allowing the packer to handle the conversion successfully.
        return DataFormat::Float32;
    }
    else if constexpr (INPUT == DataFormat::Float32 && !UNPACKING_TO_DEST)
    {
        // When input is Float32 in L1 and we are unpacking the input tensor to source registers (not directly to dest registers)
        if constexpr (FP32_ACC || is_exponentB(OUTPUT))
        {
            // If FP32 dest accumulation is enabled and the output format has an 8-bit exponent,
            // the packer input format can directly be the output format since packer can convert Float32 to another 8-bit exponent format
            return OUTPUT;
        }
        else
        {
            // Otherwise, we truncate Float32 to Tf32 or 16-bit format
            // because the packer cannot convert Float32 directly to output formats with less than 8-bit exponent (e.g., 5-bit exponent formats).
            return unpack_out;
        }
    }
    else if constexpr (INPUT == DataFormat::Float16 && OUTPUT == DataFormat::Bfp8_b && !FP32_ACC)
    {
        // When storing Float16 input in destination registers without FP32 accumulation,
        // the packer cannot convert Float16_A directly to Block Float format (in this case Bfp8_B).
        // The gasket will convert Float16_A to Bfp8_A before passing it to the packer,
        // which then converts Bfp8_A to Bfp8_B.
        return DataFormat::Bfp8;
    }
    else if constexpr (is_format_combination_outlier(INPUT, OUTPUT, FP32_ACC))
    {
        // Handling a hardware limitation: cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register.
        // In this case, we set dest registers store 32-bit datums (in params.h).
        // For wormhole architecture, gasket cannot perform this conversion and packer takes input Float32 (from dest register) converting to Float16_A.
        // For blackhole architecture, gasket able to convert Float32 to Float16_A before packing (reduces work on packer).
        return is_wormhole ? DataFormat::Float32 : OUTPUT;
    }

    // For all other cases:
    // - If dest register stores 32-bit data (FP32_ACC = true), packer input format can be set to desired output format,
    //   as gasket can convert Float32 to any format (except Float16_A).
    // - If destination registers do not store 32-bit data, gasket cannot convert,
    //   so the packer input format will be same as dest register format.
    return FP32_ACC ? OUTPUT : INPUT;
}

template <DataFormat INPUT, DataFormat OUTPUT, bool FP32_ACC>
constexpr FormatConfig infer_data_formats()
{
    // The following two formats are hard-coded for this test case
    constexpr DataFormat unpack_in = INPUT;  // The input format for Unpcker (data format in L1)
    constexpr DataFormat pack_out  = OUTPUT; // The final desired output format after packing (format in L1 after leaving the pipeline)

    // Determine the intermediate formats
    constexpr DataFormat unpack_out = infer_unpack_out<INPUT, OUTPUT, FP32_ACC>(); // output format for Unpacker, desired format in src register(s)
    constexpr DataFormat math =
        unpack_out; // The data format used for mathematical computations, desired format in dest register (typically matches unpack_out)
    constexpr DataFormat pack_in =
        infer_pack_in<INPUT, OUTPUT, unpack_out, FP32_ACC>(); // input to the packing stage, determines what gasket can convert from dest register
                                                              // potentially different from unpack_out and pack_out depending on FP32 accumulation

    // Return a FormatConfig struct capturing all the inferred formats needed for this stage
    return get_data_formats(unpack_in, unpack_out, math, pack_in, pack_out);
}

/**
 * @brief Helper function to build a constexpr array of FormatConfig objects.
 *
 * This function uses a compile-time index sequence to generate an array of
 * FormatConfig, simulating a loop unrolling over pipeline iterations.
 *
 * @tparam INPUT   Input data format (enum class DataFormat).
 * @tparam OUTPUT  Final desired output format (used only in the last iteration).
 * @tparam FP32_ACC Whether 32-bit datums are enabled for the destination register (dest_acc).
 * @tparam N       Number of L1-to-L1 pipeline iterations (array size).
 * @tparam Is...   Compile-time index sequence used to unroll loop iterations.
 *
 * @return constexpr std::array<FormatConfig, N> Array of FormatConfig for each iteration.
 */
template <size_t N, size_t... Is>
constexpr std::array<FormatConfig, N> build_data_formats(std::index_sequence<Is...>, const FormatConfig& intermediate_config, const FormatConfig& final_config)
{
    return {{(Is < N - 1 ? intermediate_config : final_config)...}};
}

/**
 * @brief Entry point for computing an array of FormatConfig objects.
 *
 * Each FormatConfig object contains all the data formats necessary to execute
 * a specific L1-to-L1 compute run across all 3 cores: unpack, math, and pack.
 * This function abstracts away the index sequence logic from callers.
 *
 * @tparam INPUT    The input data format for all pipeline runs.
 * @tparam OUTPUT   The output data format for the final pipeline run.
 * @tparam FP32_ACC Whether FP32 accumulation is enabled.
 * @tparam N        The number of pipeline runs (iterations), determines array length.
 *
 * @return A constexpr std::array of FormatConfig objects of length N.
 */
template <DataFormat INPUT, DataFormat OUTPUT, bool FP32_ACC, size_t N>
constexpr std::array<FormatConfig, N> data_formats()
{
    constexpr auto intermediate_config = infer_data_formats<INPUT, INPUT, FP32_ACC>();
    constexpr auto final_config        = infer_data_formats<INPUT, OUTPUT, FP32_ACC>();

    return build_data_formats<N>(std::make_index_sequence<N> {}, intermediate_config, final_config);
}
