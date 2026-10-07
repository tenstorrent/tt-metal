// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_k4_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/depthwise_conv1d_k4/device/kernels/dataflow/reader_depthwise_conv1d_k4.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/depthwise_conv1d_k4/device/kernels/dataflow/writer_depthwise_conv1d_k4.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/depthwise_conv1d_k4/device/kernels/compute/depthwise_conv1d_k4.cpp";

constexpr uint32_t kTapCount = 4U;
// Channel tiles handled per work item; bounded so all four taps' weights fit comfortably in L1.
constexpr uint32_t kMaxBlockTiles = 8U;

constexpr auto kActRmCb = tt::CBIndex::c_0;
constexpr auto kActTileCb = tt::CBIndex::c_1;
constexpr auto kWeightsCb = tt::CBIndex::c_2;
constexpr auto kPartialCb = tt::CBIndex::c_3;
constexpr auto kOutputCb = tt::CBIndex::c_4;
constexpr auto kGradCb = tt::CBIndex::c_5;
constexpr auto kConvCb = tt::CBIndex::c_6;
constexpr auto kSigmoidCb = tt::CBIndex::c_7;
constexpr auto kScratchACb = tt::CBIndex::c_8;
constexpr auto kScratchBCb = tt::CBIndex::c_9;

// Reader runtime-arg slots that hold buffer addresses (refreshed on program-cache hits).
constexpr uint32_t kReaderInputIdx = 0U;
constexpr uint32_t kReaderTap0Idx = 1U;
constexpr uint32_t kReaderGradIdx = 5U;
constexpr uint32_t kWriterOutputIdx = 0U;

uint32_t largest_divisor_at_most(uint32_t value, uint32_t bound) {
    for (uint32_t d = bound; d > 1U; --d) {
        if (value % d == 0U) {
            return d;
        }
    }
    return 1U;
}

}  // namespace

namespace ttml::metal::ops::depthwise_conv1d_k4::device {

DepthwiseConv1dK4ProgramFactory::cached_program_t DepthwiseConv1dK4ProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const bool has_grad = tensor_args.silu_grad.has_value();
    auto* device = input.device();
    tt::tt_metal::Program program{};

    const auto& shape = input.logical_shape();
    const uint32_t seq = shape[-2];
    const uint32_t channels = shape[-1];
    const uint32_t Mt = seq / tt::constants::TILE_HEIGHT;
    const uint32_t Ct = channels / tt::constants::TILE_WIDTH;
    const uint32_t block_ct = largest_divisor_at_most(Ct, kMaxBlockTiles);
    const uint32_t num_blocks = Ct / block_ct;
    const uint32_t total_work = Mt * num_blocks;

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, total_work);

    const auto data_format = tt::DataFormat::Float16_b;
    const uint32_t tile_bytes = tt::tile_size(data_format);

    create_circular_buffer(program, all_cores, kActRmCb, data_format, tile_bytes, 2U * block_ct);
    create_circular_buffer(program, all_cores, kActTileCb, data_format, tile_bytes, block_ct);
    create_circular_buffer(program, all_cores, kWeightsCb, data_format, tile_bytes, kTapCount * block_ct);
    create_circular_buffer(program, all_cores, kPartialCb, data_format, tile_bytes, 2U * block_ct);
    create_circular_buffer(program, all_cores, kOutputCb, data_format, tile_bytes, 2U * block_ct);
    if (has_grad) {
        create_circular_buffer(program, all_cores, kGradCb, data_format, tile_bytes, 2U * block_ct);
        create_circular_buffer(program, all_cores, kConvCb, data_format, tile_bytes, block_ct);
        create_circular_buffer(program, all_cores, kSigmoidCb, data_format, tile_bytes, 2U);
        create_circular_buffer(program, all_cores, kScratchACb, data_format, tile_bytes, 2U);
        create_circular_buffer(program, all_cores, kScratchBCb, data_format, tile_bytes, 2U);
    }

    const std::map<std::string, std::string> defines =
        has_grad ? std::map<std::string, std::string>{{"SILU_GRAD", "1"}} : std::map<std::string, std::string>{};

    std::vector<uint32_t> reader_ct_args{block_ct, num_blocks, Mt, Ct, args.anti_causal ? 1U : 0U};
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.tap0.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.tap1.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.tap2.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.tap3.buffer()).append_to(reader_ct_args);
    if (has_grad) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.silu_grad->buffer()).append_to(reader_ct_args);
    }
    auto reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{block_ct, num_blocks, Ct};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        create_compute_kernel(
            program,
            cores,
            {work_count, block_ct, num_blocks},
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/false);
    };
    make_compute(core_group_1, work_per_core_1);
    if (!core_group_2.ranges().empty()) {
        make_compute(core_group_2, work_per_core_2);
    }

    const uint32_t grad_addr = has_grad ? tensor_args.silu_grad->buffer()->address() : 0U;
    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t work_count = core_group_1.contains(core) ? work_per_core_1 : work_per_core_2;
        SetRuntimeArgs(
            program,
            reader,
            core,
            {input.buffer()->address(),
             tensor_args.tap0.buffer()->address(),
             tensor_args.tap1.buffer()->address(),
             tensor_args.tap2.buffer()->address(),
             tensor_args.tap3.buffer()->address(),
             grad_addr,
             work_start,
             work_count});
        SetRuntimeArgs(program, writer, core, {output.buffer()->address(), work_start, work_count});
        cores.push_back(core);
        work_start += work_count;
    }

    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void DepthwiseConv1dK4ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    const std::array<const ttnn::Tensor*, 4> taps = {
        &tensor_args.tap0, &tensor_args.tap1, &tensor_args.tap2, &tensor_args.tap3};

    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderInputIdx] = tensor_args.input.buffer()->address();
        for (uint32_t tap = 0; tap < taps.size(); ++tap) {
            r[kReaderTap0Idx + tap] = taps[tap]->buffer()->address();
        }
        if (tensor_args.silu_grad.has_value()) {
            r[kReaderGradIdx] = tensor_args.silu_grad->buffer()->address();
        }
        writer_args[core.x][core.y][kWriterOutputIdx] = output.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::depthwise_conv1d_k4::device
