// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "rmsnorm_bw_device_operation.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/dataflow/writer_rmsnorm_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/dataflow/reader_rmsnorm_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/compute/rmsnorm_bw_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, then one compute kernel per core
// group.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kGammaAddrIdx = 1U;
constexpr uint32_t kRmsAddrIdx = 2U;
constexpr uint32_t kDLdoutAddrIdx = 3U;
// writer runtime args
constexpr uint32_t kDLdaAddrIdx = 0U;
constexpr uint32_t kDLdgammaComponentsAddrIdx = 1U;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kGammaCbIndex = tt::CBIndex::c_3;
constexpr auto kRmsACbIndex = tt::CBIndex::c_4;
constexpr auto kDLoutCbIndex = tt::CBIndex::c_5;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;
constexpr auto kZeroCbIndex = tt::CBIndex::c_7;
// CBs with output data
constexpr auto kDLdaCbIndex = tt::CBIndex::c_8;
constexpr auto kDLdgammaComponentsCbIndex = tt::CBIndex::c_9;
// CBs with intermediate computations
constexpr auto kRecipRmsACbIndex = tt::CBIndex::c_10;
constexpr auto kScaleCbIndex = tt::CBIndex::c_11;

// Some of the below constants are set to 2U because we might need to push a new value before popping the old one.
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumRmsATiles = 2U;
constexpr uint32_t kNumMatMulReduceTiles = 1U;
constexpr uint32_t kNumRecipRmsATiles = 1U;
constexpr uint32_t kNumScaleTiles = 2U;
constexpr uint32_t kNumZeroTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw::device {

bool fits_in_l1_check(
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t bfloat16_single_tile_size_bytes,
    const uint32_t float32_single_tile_size_bytes,
    ttnn::IDevice* device) {
    const uint32_t twice_block_size = 2U * block_size;

    // Move the memory check to a separate function. And just return boolean whether it fits in L1 or not.
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Memory of input tensors
    const uint64_t input_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t mask_memory = kNumMaskTiles * bfloat16_single_tile_size_bytes;
    const uint64_t scaler_memory = kNumScalerTiles * bfloat16_single_tile_size_bytes;
    const uint64_t gamma_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t rms_a_memory = kNumRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dout_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t matmul_reduce_memory = kNumMatMulReduceTiles * bfloat16_single_tile_size_bytes;
    const uint64_t zero_memory = kNumZeroTiles * bfloat16_single_tile_size_bytes;
    // Memory for output tensors. Both output CBs are drained block-by-block by the writer, so they are
    // double-buffered (twice_block_size), independent of Wt.
    const uint64_t dL_da_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dgamma_components_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    // Memory for intermediate computations
    const uint64_t recip_rms_a_bcasted_memory = kNumRecipRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t scale_memory = kNumScaleTiles * float32_single_tile_size_bytes;

    // Total L1 memory required
    const uint64_t required_L1_in_bytes = input_memory + mask_memory + scaler_memory + gamma_memory + rms_a_memory +
                                          dL_dout_memory + matmul_reduce_memory + dL_da_memory +
                                          dL_dgamma_components_memory + recip_rms_a_bcasted_memory + scale_memory +
                                          zero_memory;

    return required_L1_in_bytes <= available_L1_in_bytes;
}

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

tt::tt_metal::ProgramDescriptor RMSNormBackwardProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& rms = tensor_args.rms;
    const auto& dLdout = tensor_args.dL_dout;

    // Check input shape is [B, N, S, C]
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);

    // Check gamma shape is [1, 1, 1, C]
    const auto& gamma_shape = gamma.logical_shape();
    TT_FATAL(gamma_shape.rank() == 4, "Gamma tensor must be 4D [1, 1, 1, C], got shape {}", gamma_shape);
    TT_FATAL(
        gamma_shape[0] == 1 && gamma_shape[1] == 1 && gamma_shape[2] == 1,
        "Gamma tensor must have shape [1, 1, 1, C], got shape {}",
        gamma_shape);

    // Check C matches between input and gamma
    TT_FATAL(
        input_shape[3] == gamma_shape[3],
        "Gamma last dim (C) must match input last dim (C): input C={}, gamma C={}",
        input_shape[3],
        gamma_shape[3]);

    auto* device = input.device();
    tt::tt_metal::ProgramDescriptor desc;

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

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

    // Get the number of inner dimension
    uint32_t num_inner = input.logical_shape()[-1];

    // This parameter is used to determine if we need to mask tiles, i.e. if the operation applied over inner dimension
    // might produce incorrect results due to some random data in the end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Compile arguments
    uint32_t block_size = get_block_size(Wt, 2U);  // We need two extra registers during calculation

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(1.F / num_inner));

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const bool everything_fits_in_l1 =
        fits_in_l1_check(Wt, block_size, bfloat16_single_tile_size_bytes, float32_single_tile_size_bytes, device);

    const uint32_t num_input_tiles =
        (everything_fits_in_l1) ? Wt : twice_block_size;  // If everything fits in L1, read Wt tiles, else read 2x block

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b
    auto precise_data_format = tt::DataFormat::Float32;

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRmsATiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMatMulReduceTiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kZeroCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumZeroTiles));
    // Output CBs are produced and drained one block at a time (see writer: "interleave the writes to avoid waiting
    // for the entire Wt tiles at once"), so they only need double-buffering, not a full row. Sizing them to
    // num_input_tiles (= Wt on the fits-in-L1 path) overflows L1 for large hidden dims (e.g. 32B, Wt=160).
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kDLdaCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kDLdgammaComponentsCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kRecipRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRecipRmsATiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kScaleCbIndex, precise_data_format, float32_single_tile_size_bytes, kNumScaleTiles));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* gamma_buffer = gamma.buffer();
    TT_FATAL(
        gamma_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Gamma buffer must be in DRAM. Gamma buffer of type {}",
        enchantum::to_string(gamma_buffer->buffer_type()));

    auto* rms_buffer = rms.buffer();
    TT_FATAL(
        rms_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "RMS buffer must be in DRAM. RMS buffer of type {}",
        enchantum::to_string(rms_buffer->buffer_type()));

    auto* dLdout_buffer = dLdout.buffer();
    TT_FATAL(
        dLdout_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_dout buffer must be in DRAM. dL_dout buffer of type {}",
        enchantum::to_string(dLdout_buffer->buffer_type()));

    auto* dL_da_buffer = output[0].buffer();
    TT_FATAL(
        dL_da_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_da buffer must be in DRAM. dL_da buffer of type {}",
        enchantum::to_string(dL_da_buffer->buffer_type()));

    auto* dL_dgamma_components_buffer = output[1].buffer();
    TT_FATAL(
        dL_dgamma_components_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_dgamma buffer must be in DRAM. dL_dgamma buffer of type {}",
        enchantum::to_string(dL_dgamma_components_buffer->buffer_type()));

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    std::vector<uint32_t> reader_compile_time_args{packed_scaler, block_size, mask_w, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(rms_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dLdout_buffer).append_to(reader_compile_time_args);
    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_da_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dL_dgamma_components_buffer).append_to(writer_compile_time_args);
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for cross_entropy_bw
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
    // 5) Per-core runtime args
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // The row split depends only on the hashed input spec.
    // Taken after the last kernel push: a later push could reallocate desc.kernels.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        // Reader kernel: (input_addr, gamma_addr, rms_addr, dLdout_addr, num_rows, offset)
        reader.emplace_runtime_args(core, {input_buffer, gamma_buffer, rms_buffer, dLdout_buffer, num_rows, start_row});
        // Writer kernel: (da_addr, dgamma_addr, num_rows, offset)
        writer.emplace_runtime_args(core, {dL_da_buffer, dL_dgamma_components_buffer, num_rows, start_row});
    });

    return desc;
}

void RMSNormBackwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t gamma_addr = tensor_args.gamma.buffer()->address();
    const uint32_t rms_addr = tensor_args.rms.buffer()->address();
    const uint32_t dLdout_addr = tensor_args.dL_dout.buffer()->address();
    const uint32_t dL_da_addr = output[0].buffer()->address();
    const uint32_t dL_dgamma_components_addr = output[1].buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kGammaAddrIdx] = gamma_addr;
        reader_args[kRmsAddrIdx] = rms_addr;
        reader_args[kDLdoutAddrIdx] = dLdout_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kDLdaAddrIdx] = dL_da_addr;
        writer_args[kDLdgammaComponentsAddrIdx] = dL_dgamma_components_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(RMSNormBackwardDeviceOperation, RMSNormBackwardProgramFactory);

}  // namespace ttml::metal::ops::rmsnorm_bw::device
