// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "rmsnorm_fw_device_operation.hpp"
#include "rmsnorm_fw_device_operation_types.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/dataflow/"
    "writer_rmsnorm_fw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/dataflow/"
    "reader_rmsnorm_fw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/compute/rmsnorm_fw_kernel.cpp";

// Kernel push order: reader, writer, then the compute kernel of each work group
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kGammaAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0U;
constexpr uint32_t kRmsOutputAddrIdx = 1U;

constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskCbIndex = tt::CBIndex::c_1;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kEpsCbIndex = tt::CBIndex::c_3;
constexpr auto kGammaCbIndex = tt::CBIndex::c_4;
constexpr auto kRmsBeforeReductionCbIndex = tt::CBIndex::c_5;
constexpr auto kRmsAfterReductionCbIndex = tt::CBIndex::c_6;
constexpr auto kInverseRmsAfterReductionCbIndex = tt::CBIndex::c_7;
constexpr auto kOutputCbIndex = tt::CBIndex::c_8;
constexpr auto kRmsOutputCbIndex = tt::CBIndex::c_9;
constexpr auto kOutputIntermediateCbIndex = tt::CBIndex::c_10;

constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumRmsBeforeReductionTiles = 2U;
constexpr uint32_t kNumRmsAfterReductionTiles = 2U;
constexpr uint32_t kNumInverseRmsAfterReductionTiles = 2U;
constexpr uint32_t kNumRmsOutputTiles = 2U;
constexpr uint32_t kNumEpsTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kReturnRMSDefineKey = "RETURN_RMS";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kEverythingExceptGammaFitsInL1DefineKey = "EVERYTHING_EXCEPT_GAMMA_FITS_IN_L1";

}  // namespace

namespace ttml::metal::ops::rmsnorm_fw::device {

namespace {

// The input's tile rows over the device's compute grid. create_descriptor() pins it against split_work_to_cores, and
// override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& input) {
    const auto& shape = input.padded_shape();
    return compute_work_split(
        input.device()->compute_with_storage_grid_size(),
        shape[0] * shape[1] * (shape[-2] / tt::constants::TILE_HEIGHT));
}

}  // namespace

tt::tt_metal::ProgramDescriptor RMSNormForwardProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;

    auto* device = input.device();

    tt::tt_metal::ProgramDescriptor desc;

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    uint32_t num_inner = input.logical_shape()[-1];

    // compile arguments
    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(1.F / static_cast<float>(num_inner));
    uint32_t packed_eps = pack_two_bfloat16_to_uint32(args.epsilon);
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;
    uint32_t block_size = get_block_size(Wt, 3U);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    uint32_t twice_block_size = 2U * block_size;
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Memory allocation for L1 cache
    const uint64_t weight_memory = 2U * Wt * bfloat16_single_tile_size_bytes;
    const uint64_t mask_scaler_eps_memory =
        (kNumMaskTiles + kNumScalerTiles + kNumEpsTiles) * bfloat16_single_tile_size_bytes;
    const uint64_t rms_memory =
        (kNumRmsBeforeReductionTiles + kNumRmsAfterReductionTiles + kNumInverseRmsAfterReductionTiles) *
        float32_single_tile_size_bytes;
    const uint64_t block_memory = 2 * twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t rms_output_memory = kNumRmsOutputTiles * bfloat16_single_tile_size_bytes;
    // Total L1 memory required
    const uint64_t required_L1_in_bytes =
        weight_memory + mask_scaler_eps_memory + rms_memory + block_memory + rms_output_memory;
    const bool everything_fits_in_l1 = required_L1_in_bytes <= available_L1_in_bytes;

    const uint64_t required_L1_in_bytes_except_gamma = required_L1_in_bytes - Wt * bfloat16_single_tile_size_bytes +
                                                       twice_block_size * bfloat16_single_tile_size_bytes;
    const bool everything_except_gamma_fits_in_l1 = required_L1_in_bytes_except_gamma <= available_L1_in_bytes;

    const uint32_t num_input_tiles =
        (everything_fits_in_l1 | everything_except_gamma_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_gamma_tiles = everything_fits_in_l1 ? Wt : twice_block_size;

    auto data_format = input_data_format;
    auto precise_data_format = tt::DataFormat::Float32;

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kEpsCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumEpsTiles));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_gamma_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kRmsBeforeReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumRmsBeforeReductionTiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kRmsAfterReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumRmsAfterReductionTiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kInverseRmsAfterReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumInverseRmsAfterReductionTiles));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kRmsOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRmsOutputTiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kOutputIntermediateCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* gamma_buffer = gamma.buffer();
    TT_FATAL(
        gamma_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Gamma buffer must be in DRAM. Gamma buffer of type {}",
        enchantum::to_string(gamma_buffer->buffer_type()));

    auto* output_buffer = output.front().buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    // Without return_intermediates the output is the only returned tensor, so this is the output buffer again; the
    // writer ignores its rms address slot then.
    auto* rms_output_buffer = output.back().buffer();
    TT_FATAL(
        rms_output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "RMS output buffer must be in DRAM. RMS output buffer of type {}",
        enchantum::to_string(rms_output_buffer->buffer_type()));

    // configure defines
    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    if (everything_except_gamma_fits_in_l1) {
        defines[kEverythingExceptGammaFitsInL1DefineKey] = "1";
    }

    if (args.return_intermediates) {
        defines[kReturnRMSDefineKey] = "1";
    }

    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    std::vector<uint32_t> reader_compile_time_args{packed_scaler, packed_eps, mask_w, Wt, block_size};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{Wt, block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    if (args.return_intermediates) {
        tt::tt_metal::TensorAccessorArgs(rms_output_buffer).append_to(writer_compile_time_args);
    } else {
        tt::tt_metal::TensorAccessorArgs().append_to(writer_compile_time_args);
    }
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for rmsnorm_fw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };

    desc.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            Wt                          // num_inner / TILE_W
        };

        desc.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // num_rows/start_row follow from the hashed input shape.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        // Reader kernel: (input_addr, gamma_addr, number_of_rows, offset_in_rows)
        reader.emplace_runtime_args(core, {input_buffer, gamma_buffer, num_rows, start_row});
        // Writer kernel: (dst_addr, dst_rms_addr, number_of_rows, offset_in_rows)
        writer.emplace_runtime_args(core, {output_buffer, rms_output_buffer, num_rows, start_row});
    });

    // -------------------------------------------------------------------------
    // 6) Return the program descriptor
    // -------------------------------------------------------------------------
    return desc;
}

void RMSNormForwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t gamma_addr = tensor_args.gamma.buffer()->address();
    const uint32_t output_addr = output.front().buffer()->address();
    const uint32_t rms_output_addr = output.back().buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kGammaAddrIdx] = gamma_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kOutputAddrIdx] = output_addr;
        writer_args[kRmsOutputAddrIdx] = rms_output_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(RMSNormForwardDeviceOperation, RMSNormForwardProgramFactory);

}  // namespace ttml::metal::ops::rmsnorm_fw::device
