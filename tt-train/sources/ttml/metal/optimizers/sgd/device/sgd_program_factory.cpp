// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_program_factory.hpp"

#include <common/TracyQueue.hpp>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "sgd_device_operation.hpp"
#include "sgd_device_operation_types.hpp"
#include "tt-metalium/bfloat16.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/optimizers/sgd/device/kernels/dataflow/"
    "reader_sgd_interleaved_start_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/optimizers/sgd/device/kernels/dataflow/"
    "writer_sgd_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath = "tt-train/sources/ttml/metal/optimizers/sgd/device/kernels/compute/sgd_kernel.cpp";

// Kernel push order: reader, writer, then the compute kernel of each work group (group 2 only when the tiles do not
// split evenly over the cores)
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;
constexpr uint32_t kComputeGroup1KernelIdx = 2U;
constexpr uint32_t kComputeGroup2KernelIdx = 3U;

// reader runtime args
constexpr uint32_t kParamAddrIdx = 0;
constexpr uint32_t kGradAddrIdx = 1U;
constexpr uint32_t kMomentumInAddrIdx = 2U;
constexpr uint32_t kBcastLrIdx = 3U;
constexpr uint32_t kBcastMomentumIdx = 4U;
constexpr uint32_t kBcastOneMinusDampeningIdx = 5U;
constexpr uint32_t kBcastWdIdx = 6U;
// compute runtime args
constexpr uint32_t kUseWeightDecayIdx = 0U;
constexpr uint32_t kUseDampeningIdx = 1U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0;
constexpr uint32_t kMomentumDramAddrIdx = 1U;

constexpr auto kParamCbIndex = tt::CBIndex::c_0;
constexpr auto kGradCbIndex = tt::CBIndex::c_1;
constexpr auto kMomentumInCbIndex = tt::CBIndex::c_2;

constexpr auto kParamWdCbIndex = tt::CBIndex::c_3;
constexpr auto kGradWdCbIndex = tt::CBIndex::c_4;

constexpr auto kMomentumScaledCbIndex = tt::CBIndex::c_5;
constexpr auto kMomentumOutCbIndex = tt::CBIndex::c_6;
constexpr auto kMomentumDramCbIndex = tt::CBIndex::c_7;

constexpr auto kGradDampenedCbIndex = tt::CBIndex::c_8;

constexpr auto kNesterovMomentumCbIndex = tt::CBIndex::c_9;
constexpr auto kNesterovUpdateCbIndex = tt::CBIndex::c_10;

constexpr auto kUpdateCbIndex = tt::CBIndex::c_11;

constexpr auto kBcastLrCbIndex = tt::CBIndex::c_12;
constexpr auto kBcastMomentumCbIndex = tt::CBIndex::c_13;
constexpr auto kBcastOneMinusDampeningCbIndex = tt::CBIndex::c_14;
constexpr auto kBcastWdCbIndex = tt::CBIndex::c_15;
constexpr auto kOutputCbIndex = tt::CBIndex::c_16;

}  // namespace

namespace ttml::metal::optimizers::sgd::device {

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SGDDeviceOperation, SGDProgramFactory);

namespace {

// Runtime args derived from the hash-excluded lr, momentum, dampening and weight_decay. create_descriptor() and
// override_runtime_arguments() both take them from here, so a cache hit writes exactly what a miss would.
struct SGDRuntimeScalars {
    uint32_t packed_lr = 0U;
    uint32_t packed_momentum = 0U;
    uint32_t packed_one_minus_dampening = 0U;
    uint32_t packed_wd = 0U;
    bool use_weight_decay = false;
    bool use_dampening = false;
};

// Each value is rounded to bf16 (to nearest even) and packed into both halves of a uint32. ttml's
// pack_two_bfloat16_to_uint32 truncates instead, so it would change the results.
SGDRuntimeScalars compute_runtime_scalars(const operation_attributes_t& operation_attributes) {
    const bfloat16 bfloat_lr = bfloat16(operation_attributes.lr);
    const bfloat16 bfloat_momentum = bfloat16(operation_attributes.momentum);
    const bfloat16 bfloat_one_minus_dampening = bfloat16(1.0F - operation_attributes.dampening);
    const bfloat16 bfloat_wd = bfloat16(operation_attributes.weight_decay);
    return SGDRuntimeScalars{
        .packed_lr = pack_two_bfloat16_into_uint32({bfloat_lr, bfloat_lr}),
        .packed_momentum = pack_two_bfloat16_into_uint32({bfloat_momentum, bfloat_momentum}),
        .packed_one_minus_dampening =
            pack_two_bfloat16_into_uint32({bfloat_one_minus_dampening, bfloat_one_minus_dampening}),
        .packed_wd = pack_two_bfloat16_into_uint32({bfloat_wd, bfloat_wd}),
        .use_weight_decay = !(bfloat_wd == bfloat16(0.0F)),
        .use_dampening = !(bfloat_one_minus_dampening == bfloat16(1.0F)),
    };
}

// The work unit is one tile of the parameter (grad and the momentum buffer have its padded shape). Both functions walk
// the cores with this split; create_descriptor() pins it to split_work_to_cores.
WorkSplit compute_sgd_work_split(const ttnn::Tensor& param) {
    return compute_work_split(
        param.device()->compute_with_storage_grid_size(),
        static_cast<uint32_t>(param.physical_volume() / tt::constants::TILE_HW));
}

}  // namespace

tt::tt_metal::ProgramDescriptor SGDProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& param = tensor_args.param;
    const auto& grad = tensor_args.grad;
    const auto& momentum_wrapped = tensor_args.momentum_buffer;
    const bool nesterov = operation_attributes.nesterov;

    auto* device = param.device();

    tt::tt_metal::ProgramDescriptor desc;

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(param.dtype());
    tt::DataFormat output_data_format = input_data_format;
    [[maybe_unused]] tt::DataFormat intermediate_data_format = tt::DataFormat::Float32;

    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");
    TT_FATAL(output_data_format == tt::DataFormat::Float16_b, "Output data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    [[maybe_unused]] uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    auto padded_tensor_shape = param.padded_shape();
    auto padded_tensor_volume = param.physical_volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded param tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");

    uint32_t total_tiles_to_process = padded_tensor_volume / tt::constants::TILE_HW;

    // get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_tiles_to_process);
    const WorkSplit split = compute_sgd_work_split(param);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);

    uint32_t block_size = std::min(8U, num_tiles_per_core_group_1);
    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;

    const uint32_t num_input_tiles = twice_block_size;
    const uint32_t num_output_tiles = twice_block_size;

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kParamCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kGradCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMomentumInCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kParamWdCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kGradWdCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMomentumScaledCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMomentumOutCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMomentumDramCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_output_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kGradDampenedCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kNesterovMomentumCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kNesterovUpdateCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kUpdateCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kBcastLrCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1U));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kBcastMomentumCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1U));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kBcastOneMinusDampeningCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1U));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kBcastWdCbIndex, tt::DataFormat::Float16_b, bfloat16_single_tile_size_bytes, 1U));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kOutputCbIndex, output_data_format, bfloat16_single_tile_size_bytes, num_output_tiles));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* param_buffer = param.buffer();
    auto* grad_buffer = grad.buffer();
    auto* momentum_buffer = momentum_wrapped.has_value() ? momentum_wrapped.value().buffer() : nullptr;
    auto* output_buffer = output.buffer();

    std::map<std::string, std::string> defines;
    defines["USE_MOMENTUM"] = momentum_buffer != nullptr ? "1" : "0";
    defines["USE_NESTEROV"] = nesterov ? "1" : "0";

    std::vector<uint32_t> reader_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(param_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(grad_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(momentum_buffer).append_to(reader_compile_time_args);

    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(momentum_buffer).append_to(writer_compile_time_args);
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for fused sgd
    // -------------------------------------------------------------------------
    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        block_size};                 // per_core_block_size

    desc.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/false));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_tiles_per_core_group_2, block_size};
        desc.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/false));
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // The slots named by the k*Idx constants are rewritten by override_runtime_arguments() on every cache hit; tile
    // counts and offsets follow from the hashed param shape.
    const SGDRuntimeScalars scalars = compute_runtime_scalars(operation_attributes);
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_tiles, start_tile, in_group_1] = work;
        // Reader: param, grad and momentum (0 without a momentum buffer) addresses, then the packed lr, momentum,
        // 1 - dampening and weight decay (kParamAddrIdx .. kBcastWdIdx), then num_tiles and start_tile
        reader.emplace_runtime_args(
            core,
            {param_buffer,
             grad_buffer,
             momentum_buffer,
             scalars.packed_lr,
             scalars.packed_momentum,
             scalars.packed_one_minus_dampening,
             scalars.packed_wd,
             num_tiles,
             start_tile});
        // Compute: (use_weight_decay, use_dampening)
        desc.kernels[in_group_1 ? kComputeGroup1KernelIdx : kComputeGroup2KernelIdx].emplace_runtime_args(
            core, {static_cast<uint32_t>(scalars.use_weight_decay), static_cast<uint32_t>(scalars.use_dampening)});
        // Writer: (output_addr, momentum_addr, num_tiles, start_tile)
        writer.emplace_runtime_args(core, {output_buffer, momentum_buffer, num_tiles, start_tile});
    });

    // -------------------------------------------------------------------------
    // 6) Return the program descriptor
    // -------------------------------------------------------------------------

    return desc;
}

void SGDProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    // Tile counts and offsets come from the hashed param shape and stay as create_descriptor() set them. The hash also
    // keys whether a momentum buffer is present, so its slots hold an address exactly when the cached program has one.
    const uint32_t param_addr = tensor_args.param.buffer()->address();
    const uint32_t grad_addr = tensor_args.grad.buffer()->address();
    const uint32_t momentum_addr =
        tensor_args.momentum_buffer.has_value() ? tensor_args.momentum_buffer->buffer()->address() : 0U;
    const uint32_t output_addr = tensor_return_value.buffer()->address();
    const SGDRuntimeScalars scalars = compute_runtime_scalars(operation_attributes);
    const WorkSplit split = compute_sgd_work_split(tensor_args.param);

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    auto& compute_group_1_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kComputeGroup1KernelIdx);
    // Without group 2 there is no fourth kernel and every core is in group 1.
    auto& compute_group_2_runtime_args = split.has_group_2()
                                             ? tt::tt_metal::GetRuntimeArgs(program, kComputeGroup2KernelIdx)
                                             : compute_group_1_runtime_args;

    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kParamAddrIdx] = param_addr;
            runtime_args[kGradAddrIdx] = grad_addr;
            runtime_args[kMomentumInAddrIdx] = momentum_addr;
            runtime_args[kBcastLrIdx] = scalars.packed_lr;
            runtime_args[kBcastMomentumIdx] = scalars.packed_momentum;
            runtime_args[kBcastOneMinusDampeningIdx] = scalars.packed_one_minus_dampening;
            runtime_args[kBcastWdIdx] = scalars.packed_wd;
        }
        {
            auto& runtime_args =
                (work.in_group_1 ? compute_group_1_runtime_args : compute_group_2_runtime_args)[core.x][core.y];
            runtime_args[kUseWeightDecayIdx] = static_cast<uint32_t>(scalars.use_weight_decay);
            runtime_args[kUseDampeningIdx] = static_cast<uint32_t>(scalars.use_dampening);
        }
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputAddrIdx] = output_addr;
            runtime_args[kMomentumDramAddrIdx] = momentum_addr;
        }
    });
}

}  // namespace ttml::metal::optimizers::sgd::device
