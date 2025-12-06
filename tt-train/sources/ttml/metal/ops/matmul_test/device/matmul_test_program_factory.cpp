// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_test_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/matmul_test/device/kernels/dataflow/reader_matmul_test.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/matmul_test/device/kernels/dataflow/writer_matmul_test.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/matmul_test/device/kernels/compute/matmul_test_kernel.cpp";

// Reader buffer indices
constexpr uint32_t kInputABufferIdx = 0;
constexpr uint32_t kInputBBufferIdx = 1U;

// Writer buffer indices
constexpr uint32_t kOutputBufferIdx = 0;

// CBs - for reading inputs in BF16
constexpr auto kInputACbIndex = tt::CBIndex::c_0;
constexpr auto kInputBCbIndex = tt::CBIndex::c_1;
// CBs - for optionally copying to FP32
constexpr auto kInputAFP32CbIndex = tt::CBIndex::c_2;
constexpr auto kInputBFP32CbIndex = tt::CBIndex::c_3;
// CB for output
constexpr auto kOutputCbIndex = tt::CBIndex::c_4;

}  // namespace

namespace ttml::metal::ops::matmul_test::device {

MatmulTestProgramFactory::cached_program_t MatmulTestProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes
    // -------------------------------------------------------------------------
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    auto* device = input_a.device();
    tt::tt_metal::Program program{};

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // We're only doing 1 tile x 1 tile = 1 tile (32x32 @ 32x32 = 32x32)
    // Use a single core
    tt::tt_metal::CoreCoord core = {0, 0};
    tt::tt_metal::CoreRange single_core_range(core, core);
    tt::tt_metal::CoreRangeSet core_range({single_core_range});

    // -------------------------------------------------------------------------
    // 2) Determine CB formats based on test case
    // -------------------------------------------------------------------------
    tt::DataFormat cb_a_format = tt::DataFormat::Float16_b;
    tt::DataFormat cb_b_format = tt::DataFormat::Float16_b;
    bool need_copy_a = false;
    bool need_copy_b = false;

    switch (args.test_case) {
        case TestCase::BF16_BF16:
            cb_a_format = tt::DataFormat::Float16_b;
            cb_b_format = tt::DataFormat::Float16_b;
            break;
        case TestCase::FP32_FP32:
            cb_a_format = tt::DataFormat::Float32;
            cb_b_format = tt::DataFormat::Float32;
            need_copy_a = true;
            need_copy_b = true;
            break;
        case TestCase::BF16_FP32:
            cb_a_format = tt::DataFormat::Float16_b;
            cb_b_format = tt::DataFormat::Float32;
            need_copy_b = true;
            break;
        case TestCase::FP32_BF16:
            cb_a_format = tt::DataFormat::Float32;
            cb_b_format = tt::DataFormat::Float16_b;
            need_copy_a = true;
            break;
    }

    // -------------------------------------------------------------------------
    // 3) Create circular buffers
    // -------------------------------------------------------------------------
    // Always read to BF16 CBs
    [[maybe_unused]] auto cb_input_a = create_circular_buffer(
        program, core_range, kInputACbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1);
    [[maybe_unused]] auto cb_input_b = create_circular_buffer(
        program, core_range, kInputBCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1);

    // Create FP32 CBs if needed for copying
    if (need_copy_a) {
        [[maybe_unused]] auto cb_input_a_fp32 = create_circular_buffer(
            program, core_range, kInputAFP32CbIndex, tt::DataFormat::Float32, float32_single_tile_size_bytes, 1);
    }
    if (need_copy_b) {
        [[maybe_unused]] auto cb_input_b_fp32 = create_circular_buffer(
            program, core_range, kInputBFP32CbIndex, tt::DataFormat::Float32, float32_single_tile_size_bytes, 1);
    }

    // Output CB - always BF16
    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program, core_range, kOutputCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1);

    // -------------------------------------------------------------------------
    // 4) Create kernels
    // -------------------------------------------------------------------------
    std::map<std::string, std::string> reader_defines;
    if (need_copy_a) {
        reader_defines["COPY_A_TO_FP32"] = "1";
    }
    if (need_copy_b) {
        reader_defines["COPY_B_TO_FP32"] = "1";
    }

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        kReaderKernelPath,
        core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = reader_defines});

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        kWriterKernelPath,
        core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});

    std::map<std::string, std::string> compute_defines;
    compute_defines["USE_FP32_A"] = need_copy_a ? "1" : "0";
    compute_defines["USE_FP32_B"] = need_copy_b ? "1" : "0";

    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = compute_defines});

    // -------------------------------------------------------------------------
    // 5) Set runtime arguments
    // -------------------------------------------------------------------------
    auto input_a_buffer = input_a.buffer();
    auto input_b_buffer = input_b.buffer();
    auto output_buffer = output.buffer();

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, {input_a_buffer->address(), input_b_buffer->address()});

    tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address()});

    // No runtime args needed for compute kernel

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel,
         .writer_kernel_id = writer_kernel,
         .compute_kernel_id = compute_kernel,
         .core = core}};
}

void MatmulTestProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto input_a_buffer = tensor_args.input_a.buffer();
    auto input_b_buffer = tensor_args.input_b.buffer();
    auto output_buffer = output.buffer();

    tt::tt_metal::SetRuntimeArgs(
        program,
        shared_vars.reader_kernel_id,
        shared_vars.core,
        {input_a_buffer->address(), input_b_buffer->address()});

    tt::tt_metal::SetRuntimeArgs(program, shared_vars.writer_kernel_id, shared_vars.core, {output_buffer->address()});
}

}  // namespace ttml::metal::ops::matmul_test::device
