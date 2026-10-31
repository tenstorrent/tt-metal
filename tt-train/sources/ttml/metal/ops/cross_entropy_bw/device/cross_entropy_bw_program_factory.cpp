// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "cross_entropy_bw_device_operation.hpp"
#include "cross_entropy_bw_device_operation_types.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/cross_entropy_bw/device/kernels/dataflow/"
    "reader_cross_entropy_bw_interleaved_start_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/cross_entropy_bw/device/kernels/dataflow/"
    "writer_cross_entropy_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/cross_entropy_bw/device/kernels/compute/cross_entropy_bw_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, compute group 1, then compute
// group 2 when it has cores.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kTargetAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0U;

constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kTargetCbIndex = tt::CBIndex::c_1;
constexpr auto kMaskCbIndex = tt::CBIndex::c_2;
constexpr auto kMaxMaskCbIndex = tt::CBIndex::c_3;
constexpr auto KReductionScalerCbIndex = tt::CBIndex::c_4;  // used to reduction
constexpr auto kMaxValueBeforeReductionCbIndex = tt::CBIndex::c_5;
constexpr auto kMaxValueAfterReductionCbIndex = tt::CBIndex::c_6;
constexpr auto kExpSumBeforeReductionCbIndex = tt::CBIndex::c_7;
constexpr auto KExpSumAfterReductionCbIndex = tt::CBIndex::c_8;
constexpr auto kMatMulCbIndex = tt::CBIndex::c_9;
constexpr auto kOutputCbIndex = tt::CBIndex::c_10;

constexpr uint32_t kNumTargetIndexesTiles = 2U;
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kMaxValueBeforeReductionTiles = 2U;
constexpr uint32_t kNumMaxValueAfterReductionTiles = 2U;
constexpr uint32_t kNumExpSumBeforeReductionTiles = 2U;
constexpr uint32_t kNumExpSumAfterReductionTiles = 2U;
constexpr uint32_t kNumScalerTiles = 1U;  // used it to reduction

constexpr uint32_t kPageElementsNumber = 32U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";

}  // namespace

namespace ttml::metal::ops::cross_entropy_bw::device {

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

tt::tt_metal::ProgramDescriptor CrossEntropyBackwardProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;

    auto* device = input.device();

    tt::tt_metal::ProgramDescriptor program;

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");

    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;  // <- number of tiles in inner dimension
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // get size of target indexes inner dimension
    uint32_t target_indexes_inner_dim_size = target.logical_shape()[-1] * target.element_size();
    // read target indexes by pages(32 indexes in page)
    uint32_t uint32_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * kPageElementsNumber;

    // get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // get the number of inner dimension
    uint32_t num_inner = input.logical_shape()[-1];  // (N, 1, C, H)

    // mask_w - this mask used to avoid calculation of extra data
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;  // width index of first trash value in tile

    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);

    // compile arguments
    uint32_t block_size = get_block_size(Wt, 3U);  // we need one extra register during calculation

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;

    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    const uint64_t masks_memory = 2U * kNumMaskTiles * bfloat16_single_tile_size_bytes;
    const uint64_t scalers_memory = (2U * kNumScalerTiles) * bfloat16_single_tile_size_bytes;  // scaler and matmul
    const uint64_t output_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t max_value_memory =
        (kNumMaxValueAfterReductionTiles + kMaxValueBeforeReductionTiles) * bfloat16_single_tile_size_bytes;
    const uint64_t exp_sum_memory =
        (kNumExpSumBeforeReductionTiles + kNumExpSumAfterReductionTiles) * float32_single_tile_size_bytes;
    const uint64_t input_memory =
        /* input */ (Wt * bfloat16_single_tile_size_bytes) +
        /* target */ (uint32_read_page_size * kNumTargetIndexesTiles);

    // Total L1 memory required
    const uint64_t required_L1_in_bytes =
        input_memory + masks_memory + scalers_memory + max_value_memory + exp_sum_memory + output_memory;
    // Is everything fits in L1
    const bool everything_fits_in_l1 = required_L1_in_bytes <= available_L1_in_bytes;

    const uint32_t num_input_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_output_tiles = twice_block_size;

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b
    auto precise_data_format = tt::DataFormat::Float32;
    auto target_indexes_data_format = tt::DataFormat::UInt32;

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    program.cbs.push_back(make_cb_descriptor(
        all_cores, kTargetCbIndex, target_indexes_data_format, uint32_read_page_size, kNumTargetIndexesTiles));

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles));

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kMaxMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles));

    program.cbs.push_back(make_cb_descriptor(
        all_cores, KReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles));

    program.cbs.push_back(make_cb_descriptor(
        all_cores,
        kMaxValueBeforeReductionCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kMaxValueBeforeReductionTiles));

    program.cbs.push_back(make_cb_descriptor(
        all_cores,
        kMaxValueAfterReductionCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kNumMaxValueAfterReductionTiles));

    program.cbs.push_back(make_cb_descriptor(
        all_cores,
        kExpSumBeforeReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumExpSumBeforeReductionTiles));

    program.cbs.push_back(make_cb_descriptor(
        all_cores,
        KExpSumAfterReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumExpSumAfterReductionTiles));

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kMatMulCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles));

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_output_tiles));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* target_buffer = target.buffer();
    TT_FATAL(
        target_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Target buffer must be in DRAM. Target buffer of type {}",
        enchantum::to_string(target_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    // configure defines
    std::map<std::string, std::string> defines;
    // define whether I need mask or not
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }

    // define whether everything fits in L1 or not
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    {
        std::vector<uint32_t> reader_compile_time_args{
            block_size, Wt, mask_w, target_indexes_inner_dim_size, Ht, uint32_read_page_size};
        tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(target_buffer).append_to(reader_compile_time_args);
        program.kernels.push_back(  // kReaderKernelIdx
            make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));
    }

    {
        std::vector<uint32_t> writer_compile_time_args{block_size, Wt, scaler_bits, mask_w};
        tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
        program.kernels.push_back(  // kWriterKernelIdx
            make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));
    }

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for cross_entropy_bw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt,                         // num_inner / TILE_W
        scaler_bits};

    program.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            Wt,                         // num_inner / TILE_W
            scaler_bits};

        program.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // num_rows/start_row follow from the hashed input shape.
    auto& reader = program.kernels[kReaderKernelIdx];
    auto& writer = program.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        reader.emplace_runtime_args(core, {input_buffer, target_buffer, num_rows, start_row});
        writer.emplace_runtime_args(core, {output_buffer, num_rows, start_row});
    });

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program descriptor
    // -------------------------------------------------------------------------

    return program;
}

void CrossEntropyBackwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t target_addr = tensor_args.target.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kTargetAddrIdx] = target_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kOutputAddrIdx] = output_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(CrossEntropyBackwardDeviceOperation, CrossEntropyBackwardProgramFactory);

}  // namespace ttml::metal::ops::cross_entropy_bw::device
