// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lowrank_update_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/lowrank_update_cbs.hpp"
#include "lowrank_update_device_operation.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/lowrank_update/device/kernels/dataflow/reader_lowrank_update.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/lowrank_update/device/kernels/dataflow/writer_lowrank_update.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/lowrank_update/device/kernels/compute/lowrank_update_kernel.cpp";

constexpr uint32_t kReaderAccIdx = 0U;
constexpr uint32_t kReaderAIdx = 1U;
constexpr uint32_t kReaderWIdx = 2U;
constexpr uint32_t kWriterOutIdx = 0U;

}  // namespace

namespace ttml::metal::ops::lowrank_update::device {

// Work items are the output tiles in column-major order inside each image (n outer, r inner): a core's range
// keeps the w tile of a column for Mt consecutive items and the image's a column (Mt tiles) resident in L1.
LowrankUpdateProgramFactory::cached_program_t LowrankUpdateProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, ttnn::Tensor& output) {
    namespace cb = lowrank_update_cb;
    const auto& acc = tensor_args.acc;
    auto* device = acc.device();
    tt::tt_metal::Program program{};

    const auto geo = lowrank_geometry(acc);
    const uint32_t total_items = geo.images * geo.Mt * geo.Nt;
    const auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, total_items);

    const auto bf16 = tt::DataFormat::Float16_b;
    const uint32_t tile = tt::tile_size(bf16);
    create_circular_buffer(program, all_cores, cb::a, bf16, tile, geo.Mt);
    create_circular_buffer(program, all_cores, cb::w, bf16, tile, 2U);
    create_circular_buffer(program, all_cores, cb::acc, bf16, tile, 2U);
    create_circular_buffer(program, all_cores, cb::out, bf16, tile, 2U);

    std::map<std::string, std::string> defines;
    if (args.transpose_b) {
        defines["TRANSPOSE_B"] = "1";
    }
    const uint32_t scale_bits = std::bit_cast<uint32_t>(args.scale);

    std::vector<uint32_t> reader_ct_args{geo.Mt, geo.Nt};
    tt::tt_metal::TensorAccessorArgs(acc.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.a.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.w.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{geo.Mt, geo.Nt};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program,
            cores,
            {work_count, geo.Mt, geo.Nt, scale_bits},
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    };
    const auto compute_1 = make_compute(core_group_1, work_per_core_1);
    const auto compute_2 = core_group_2.ranges().empty() ? compute_1 : make_compute(core_group_2, work_per_core_2);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / grid.y, i % grid.y};
        const bool in_group_1 = core_group_1.contains(core);
        const uint32_t work_count = in_group_1 ? work_per_core_1 : work_per_core_2;
        SetRuntimeArgs(program, in_group_1 ? compute_1 : compute_2, core, {work_start});
        SetRuntimeArgs(
            program,
            reader,
            core,
            {acc.buffer()->address(),
             tensor_args.a.buffer()->address(),
             tensor_args.w.buffer()->address(),
             work_start,
             work_count});
        SetRuntimeArgs(program, writer, core, {output.buffer()->address(), work_start, work_count});
        cores.push_back(core);
        work_start += work_count;
    }
    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void LowrankUpdateProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    ttnn::Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderAccIdx] = tensor_args.acc.buffer()->address();
        r[kReaderAIdx] = tensor_args.a.buffer()->address();
        r[kReaderWIdx] = tensor_args.w.buffer()->address();
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::lowrank_update::device
