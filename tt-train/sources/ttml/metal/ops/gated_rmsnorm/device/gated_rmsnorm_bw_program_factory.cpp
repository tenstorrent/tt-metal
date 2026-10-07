// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_bw_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/gated_rmsnorm_cbs.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/dataflow/reader_gated_rmsnorm.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/dataflow/writer_gated_rmsnorm.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/compute/gated_rmsnorm_bw.cpp";

// Runtime-arg slots holding buffer addresses (refreshed on program-cache hits).
constexpr uint32_t kReaderInputIdx = 0U;
constexpr uint32_t kReaderGateIdx = 1U;
constexpr uint32_t kReaderGammaIdx = 2U;
constexpr uint32_t kReaderDyIdx = 3U;
constexpr uint32_t kWriterDxIdx = 0U;
constexpr uint32_t kWriterDgateIdx = 1U;
constexpr uint32_t kWriterDgammaIdx = 2U;

}  // namespace

namespace ttml::metal::ops::gated_rmsnorm::device {

GatedRmsNormBackwardProgramFactory::cached_program_t GatedRmsNormBackwardProgramFactory::create(
    const bw::operation_attributes_t& args, const bw::tensor_args_t& tensor_args, bw::tensor_return_value_t& outputs) {
    namespace cb = gated_rmsnorm_cb;
    const auto& input = tensor_args.input;
    auto* device = input.device();
    tt::tt_metal::Program program{};

    const auto geo =
        validate_and_get_geometry("gated_rmsnorm_bw", input, tensor_args.gate, tensor_args.gamma, tensor_args.dL_dout);
    const uint32_t Gt = geo.group_tiles;
    const uint32_t total_work = geo.rows_tiles * geo.num_groups;
    const bool compute_dgamma = args.compute_dgamma;
    const auto& dx = outputs[0].value();
    const auto& dgate = outputs[1].value();

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, total_work);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t f32_tile = tt::tile_size(f32);

    create_circular_buffer(program, all_cores, cb::x, bf16, bf16_tile, 2U * Gt);
    create_circular_buffer(program, all_cores, cb::gate, bf16, bf16_tile, 2U * Gt);
    create_circular_buffer(program, all_cores, cb::gamma, bf16, bf16_tile, Gt);
    create_circular_buffer(program, all_cores, cb::gamma_b, bf16, bf16_tile, Gt);
    create_circular_buffer(program, all_cores, cb::ones, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::sq, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::inv, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::out, bf16, bf16_tile, 2U * Gt);
    create_circular_buffer(program, all_cores, cb::dy, bf16, bf16_tile, 2U * Gt);
    create_circular_buffer(program, all_cores, cb::u, f32, f32_tile, Gt);
    create_circular_buffer(program, all_cores, cb::du, f32, f32_tile, Gt);
    create_circular_buffer(program, all_cores, cb::prod, f32, f32_tile, Gt);
    create_circular_buffer(program, all_cores, cb::acc, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::t, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::dgate, bf16, bf16_tile, 2U * Gt);
    if (compute_dgamma) {
        create_circular_buffer(program, all_cores, cb::dgamma, bf16, bf16_tile, 2U * Gt);
    }

    std::map<std::string, std::string> defines{{"BACKWARD", "1"}};
    if (compute_dgamma) {
        defines["COMPUTE_DGAMMA"] = "1";
    }

    std::vector<uint32_t> reader_ct_args{Gt, geo.width_tiles, geo.num_groups};
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.gate.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.gamma.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.dL_dout.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{Gt, geo.width_tiles, geo.num_groups};
    tt::tt_metal::TensorAccessorArgs(dx.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(dgate.buffer()).append_to(writer_ct_args);
    if (compute_dgamma) {
        tt::tt_metal::TensorAccessorArgs(outputs[2]->buffer()).append_to(writer_ct_args);
    }
    auto writer = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    const uint32_t inv_group_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(geo.group));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(args.epsilon);
    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        create_compute_kernel(
            program,
            cores,
            {work_count, Gt, inv_group_bits, eps_bits},
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    };
    make_compute(core_group_1, work_per_core_1);
    if (!core_group_2.ranges().empty()) {
        make_compute(core_group_2, work_per_core_2);
    }

    const uint32_t dgamma_addr = compute_dgamma ? outputs[2]->buffer()->address() : 0U;
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
             tensor_args.gate.buffer()->address(),
             tensor_args.gamma.buffer()->address(),
             tensor_args.dL_dout.buffer()->address(),
             work_start,
             work_count});
        SetRuntimeArgs(
            program,
            writer,
            core,
            {dx.buffer()->address(), dgate.buffer()->address(), dgamma_addr, work_start, work_count});
        cores.push_back(core);
        work_start += work_count;
    }

    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void GatedRmsNormBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const bw::operation_attributes_t& args,
    const bw::tensor_args_t& tensor_args,
    bw::tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderInputIdx] = tensor_args.input.buffer()->address();
        r[kReaderGateIdx] = tensor_args.gate.buffer()->address();
        r[kReaderGammaIdx] = tensor_args.gamma.buffer()->address();
        r[kReaderDyIdx] = tensor_args.dL_dout.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        w[kWriterDxIdx] = outputs[0]->buffer()->address();
        w[kWriterDgateIdx] = outputs[1]->buffer()->address();
        if (args.compute_dgamma) {
            w[kWriterDgammaIdx] = outputs[2]->buffer()->address();
        }
    }
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device
