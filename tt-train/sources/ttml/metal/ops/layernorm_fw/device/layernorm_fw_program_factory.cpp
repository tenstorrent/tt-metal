// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fw_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "layernorm_fw_device_operation.hpp"
#include "metal/common/program_utils.hpp"
#include "tt-metalium/device.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/dataflow/writer_layernorm_fw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/dataflow/reader_layernorm_fw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/compute/layernorm_fw_kernel.cpp";

// Kernel push order: reader, writer, then the compute kernel of each work group
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kGammaAddrIdx = 1U;
constexpr uint32_t kBetaAddrIdx = 2U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0U;
constexpr uint32_t kMeanAddrIdx = 1U;
constexpr uint32_t kRstdAddrIdx = 2U;

// CBs with input data
constexpr auto kScalerCbIndex = tt::CBIndex::c_0;  // 1/N scaler
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;   // mask for width dimension
constexpr auto kEpsCbIndex = tt::CBIndex::c_2;     // epsilon
constexpr auto kGammaCbIndex = tt::CBIndex::c_3;   // gamma (scale parameter)
constexpr auto kBetaCbIndex = tt::CBIndex::c_4;    // beta (shift parameter)
constexpr auto kInputCbIndex = tt::CBIndex::c_5;   // input tensor

// CBs with output data
constexpr auto kOutputCbIndex = tt::CBIndex::c_6;  // normalized output
constexpr auto kMeanCbIndex = tt::CBIndex::c_7;    // mean (for backward pass)
constexpr auto kRstdCbIndex = tt::CBIndex::c_8;    // rstd (for backward pass)

// CBs with intermediate computations
constexpr auto kSumCbIndex = tt::CBIndex::c_9;                  // sum of inputs
constexpr auto kMeanBcastCbIndex = tt::CBIndex::c_10;           // broadcasted mean
constexpr auto kVarianceSumCbIndex = tt::CBIndex::c_11;         // sum((x - mean)^2)
constexpr auto kRstdBcastCbIndex = tt::CBIndex::c_12;           // broadcasted rstd
constexpr auto kXHatCbIndex = tt::CBIndex::c_13;                // normalized x_hat
constexpr auto kOutputIntermediateCbIndex = tt::CBIndex::c_14;  // intermediate for x_hat * gamma

constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumEpsTiles = 1U;
constexpr uint32_t kNumMeanBcastTiles = 1U;
constexpr uint32_t kNumVarianceReducedTiles = 1U;
constexpr uint32_t kNumRstdBcastTiles = 1U;
constexpr uint32_t kNumSumTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kReturnMeanRstdDefineKey = "RETURN_MEAN_RSTD";

bool fits_in_l1_check(
    const uint32_t Wt,
    const uint32_t closest_to_Wt_multiple_of_block_size,
    const uint32_t block_size,
    const uint32_t bfloat16_single_tile_size_bytes,
    const uint32_t float32_single_tile_size_bytes,
    tt::tt_metal::IDevice* device,
    const bool return_mean_rstd) {
    // Calculate available L1 memory
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    const uint32_t bf16_row_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint32_t bf16_row_memory_padded_to_block_size =
        closest_to_Wt_multiple_of_block_size * bfloat16_single_tile_size_bytes;
    // Memory for input data CBs
    const uint32_t scaler_memory = kNumScalerTiles * bfloat16_single_tile_size_bytes;
    const uint32_t mask_memory = kNumMaskTiles * bfloat16_single_tile_size_bytes;
    const uint32_t eps_memory = kNumEpsTiles * bfloat16_single_tile_size_bytes;
    const uint32_t gamma_memory = bf16_row_memory;
    const uint32_t beta_memory = bf16_row_memory;
    const uint32_t input_memory = bf16_row_memory;

    // Memory for output CBs
    const uint32_t output_memory = 2 * block_size * bfloat16_single_tile_size_bytes;

    // Memory for intermediate computation CBs
    const uint32_t sum_memory = kNumSumTiles * float32_single_tile_size_bytes;
    const uint32_t mean_bcast_memory = kNumMeanBcastTiles * bfloat16_single_tile_size_bytes;
    const uint32_t variance_sum_memory = kNumSumTiles * float32_single_tile_size_bytes;
    const uint32_t variance_reduced_memory = kNumVarianceReducedTiles * bfloat16_single_tile_size_bytes;
    const uint32_t rstd_bcast_memory = kNumRstdBcastTiles * bfloat16_single_tile_size_bytes;
    const uint32_t x_hat_memory = bf16_row_memory_padded_to_block_size;
    const uint32_t output_intermediate_memory = block_size * bfloat16_single_tile_size_bytes;

    // Total L1 memory required
    uint32_t required_L1_in_bytes = scaler_memory + mask_memory + eps_memory + gamma_memory + beta_memory +
                                    input_memory + output_memory + sum_memory + mean_bcast_memory +
                                    variance_sum_memory + variance_reduced_memory + rstd_bcast_memory + x_hat_memory +
                                    output_intermediate_memory;

    if (return_mean_rstd) {
        const uint32_t mean_output_memory = kNumMeanBcastTiles * bfloat16_single_tile_size_bytes;
        const uint32_t rstd_output_memory = kNumRstdBcastTiles * bfloat16_single_tile_size_bytes;
        required_L1_in_bytes += mean_output_memory + rstd_output_memory;
    }
    return required_L1_in_bytes <= available_L1_in_bytes;
}

}  // namespace

namespace ttml::metal::ops::layernorm_fw::device {

namespace {

// The input's tile rows over the device's compute grid. create_descriptor() pins it against split_work_to_cores, and
// override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& input) {
    const auto& shape = input.padded_shape();
    return compute_work_split(
        input.device()->compute_with_storage_grid_size(),
        (shape[-2] * shape[-3] * shape[-4]) / tt::constants::TILE_HEIGHT);
}

}  // namespace

tt::tt_metal::ProgramDescriptor LayerNormForwardProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const float epsilon = operation_attributes.epsilon;
    const bool return_mean_rstd = operation_attributes.return_mean_rstd;

    // Check input shape is [B, N, S, C]
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);

    // Check gamma shape is [1, 1, 1, C]
    const auto& gamma_shape = gamma.logical_shape();
    TT_FATAL(gamma_shape.rank() == 4, "Gamma tensor must be 4D [1, 1, 1, C], got shape {}", gamma_shape);

    // Check beta shape is [1, 1, 1, C]
    const auto& beta_shape = beta.logical_shape();
    TT_FATAL(beta_shape.rank() == 4, "Beta tensor must be 4D [1, 1, 1, C], got shape {}", beta_shape);

    tt::tt_metal::ProgramDescriptor desc;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // -------------------------------------------------------------------------
    // 1) Precompute tensor dimensions and padding
    // -------------------------------------------------------------------------
    const auto& padded_tensor_shape = input.padded_shape();
    uint32_t padded_tensor_volume = padded_tensor_shape.volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t total_rows_to_process =
        (padded_tensor_shape[-2] * padded_tensor_shape[-3] * padded_tensor_shape[-4]) / tt::constants::TILE_HEIGHT;

    // Get the number of inner dimension (assumes divisible by TILE_WIDTH)
    uint32_t num_inner = input.logical_shape()[-1];

    // This parameter is used to determine if we need to mask tiles, i.e. if the operation applied over inner dimension
    // might produce incorrect results due to some random data in the end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Compile arguments
    uint32_t block_size = 3U;  // Need 1 extra registers for layernorm forward

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(1.F / num_inner));
    uint32_t packed_eps = pack_two_bfloat16_to_uint32(epsilon);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const uint32_t closest_to_Wt_multiple_of_block_size = ((Wt + block_size - 1) / block_size) * block_size;

    const bool everything_fits_in_l1 = fits_in_l1_check(
        Wt,
        closest_to_Wt_multiple_of_block_size,
        block_size,
        bfloat16_single_tile_size_bytes,
        float32_single_tile_size_bytes,
        device,
        return_mean_rstd);

    const uint32_t num_input_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_x_hat_tiles = (everything_fits_in_l1) ? closest_to_Wt_multiple_of_block_size : twice_block_size;

    tt::DataFormat default_data_format = tt::DataFormat::Float16_b;
    // Input data CBs
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kScalerCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMaskWCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kEpsCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumEpsTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kGammaCbIndex, default_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kBetaCbIndex, default_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kInputCbIndex, default_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    // Output CBs
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kOutputCbIndex, default_data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    if (return_mean_rstd) {
        desc.cbs.push_back(make_cb_descriptor(
            all_cores, kMeanCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumMeanBcastTiles));
        desc.cbs.push_back(make_cb_descriptor(
            all_cores, kRstdCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumRstdBcastTiles));
    }

    // Intermediate computation CBs
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumCbIndex, default_data_format, float32_single_tile_size_bytes, kNumSumTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMeanBcastCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumMeanBcastTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kVarianceSumCbIndex, default_data_format, float32_single_tile_size_bytes, kNumSumTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kRstdBcastCbIndex, default_data_format, bfloat16_single_tile_size_bytes, kNumRstdBcastTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kXHatCbIndex, default_data_format, bfloat16_single_tile_size_bytes, num_x_hat_tiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kOutputIntermediateCbIndex, default_data_format, bfloat16_single_tile_size_bytes, block_size));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    auto* gamma_buffer = gamma.buffer();
    auto* beta_buffer = beta.buffer();
    auto* output_buffer = output[0].value().buffer();
    tt::tt_metal::Buffer* mean_buffer = nullptr;
    tt::tt_metal::Buffer* rstd_buffer = nullptr;
    if (return_mean_rstd) {
        mean_buffer = output[1].value().buffer();
        rstd_buffer = output[2].value().buffer();
    }

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }
    if (return_mean_rstd) {
        defines[kReturnMeanRstdDefineKey] = "1";
    }

    std::vector<uint32_t> reader_compile_time_args{packed_scaler, packed_eps, block_size, mask_w, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta_buffer).append_to(reader_compile_time_args);
    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    if (return_mean_rstd) {
        tt::tt_metal::TensorAccessorArgs(mean_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(rstd_buffer).append_to(writer_compile_time_args);
    } else {
        // Add dummy TensorAccessorArgs for mean and rstd (they won't be used)
        auto dummy_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            tt::tt_metal::distributed::ReplicatedBufferConfig{.size = 0},
            tt::tt_metal::distributed::DeviceLocalBufferConfig{
                .page_size = 0, .buffer_type = tt::tt_metal::BufferType::DRAM},
            input.device());
        tt::tt_metal::TensorAccessorArgs(dummy_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dummy_buffer).append_to(writer_compile_time_args);
    }
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        mask_w,                     // mask_w
        Wt                          // num_inner / TILE_W
    };

    desc.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            mask_w,                     // mask_w
            Wt                          // num_inner / TILE_W
        };

        desc.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // Absent mean/rstd are null (0); start_row/num_rows follow from the hashed input shape.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        // Reader kernel runtime args
        reader.emplace_runtime_args(core, {input_buffer, gamma_buffer, beta_buffer, start_row, num_rows});
        // Writer kernel runtime args
        writer.emplace_runtime_args(core, {output_buffer, mean_buffer, rstd_buffer, start_row, num_rows});
    });

    // -------------------------------------------------------------------------
    // 6) Return the program descriptor
    // -------------------------------------------------------------------------
    return desc;
}

void LayerNormForwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t gamma_addr = tensor_args.gamma.buffer()->address();
    const uint32_t beta_addr = tensor_args.beta.buffer()->address();
    const uint32_t output_addr = output[0].value().buffer()->address();
    const uint32_t mean_addr = operation_attributes.return_mean_rstd ? output[1].value().buffer()->address() : 0U;
    const uint32_t rstd_addr = operation_attributes.return_mean_rstd ? output[2].value().buffer()->address() : 0U;

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kGammaAddrIdx] = gamma_addr;
        reader_args[kBetaAddrIdx] = beta_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kOutputAddrIdx] = output_addr;
        writer_args[kMeanAddrIdx] = mean_addr;
        writer_args[kRstdAddrIdx] = rstd_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(LayerNormForwardDeviceOperation, LayerNormForwardProgramFactory);

}  // namespace ttml::metal::ops::layernorm_fw::device
