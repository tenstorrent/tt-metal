// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "frobenius_normalize_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "frobenius_normalize_device_operation_types.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/dataflow/"
    "reader_frobenius_normalize.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/dataflow/"
    "writer_frobenius_normalize.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/compute/"
    "frobenius_normalize_compute.cpp";

constexpr auto kCbInput = tt::CBIndex::c_0;
constexpr auto kCbSqAcc = tt::CBIndex::c_1;

constexpr auto kCbRecv = tt::CBIndex::c_3;
constexpr auto kCbNorm = tt::CBIndex::c_4;
constexpr auto kCbOutput = tt::CBIndex::c_5;
constexpr auto kCbSqPartial = tt::CBIndex::c_7;

constexpr uint32_t kReaderInputAddrIdx = 0;

constexpr uint32_t kWriterOutputAddrIdx = 0;

}  // namespace

namespace ttml::metal::ops::frobenius_normalize::device {

FrobeniusNormalizeProgramFactory::cached_program_t FrobeniusNormalizeProgramFactory::create(
    const FrobeniusNormalizeAttributes& args,
    const FrobeniusNormalizeTensorArgs& tensor_args,
    FrobeniusNormalizeTensorReturn& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::Program program{};

    // -------------------------------------------------------------------------
    // 1) Data formats and tile sizes and compute split
    // -------------------------------------------------------------------------
    tt::DataFormat bf16_format = tt::DataFormat::Float16_b;
    tt::DataFormat fp32_format = tt::DataFormat::Float32;

    uint32_t bf16_tile_size = tt::tile_size(bf16_format);
    uint32_t fp32_tile_size = tt::tile_size(fp32_format);

    uint32_t total_tiles = input.physical_volume() / tt::constants::TILE_HW;

    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_tiles);

    uint32_t block_size = std::min(4U, tiles_per_core_g1);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t input_buf_tiles = 2 * block_size;
    const uint32_t output_buf_tiles = 2 * block_size;

    [[maybe_unused]] auto cb_input =
        create_circular_buffer(program, all_cores, kCbInput, bf16_format, bf16_tile_size, input_buf_tiles);
    [[maybe_unused]] auto cb_sq_acc =
        create_circular_buffer(program, all_cores, kCbSqAcc, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_recv = create_circular_buffer(program, all_cores, kCbRecv, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_norm = create_circular_buffer(program, all_cores, kCbNorm, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kCbOutput, bf16_format, bf16_tile_size, output_buf_tiles);
    [[maybe_unused]] auto cb_sq_partial =
        create_circular_buffer(program, all_cores, kCbSqPartial, fp32_format, fp32_tile_size, 1);

    // -------------------------------------------------------------------------
    // 3) Create semaphores
    // -------------------------------------------------------------------------
    // Reduction: non-origin cores increment after writing partial to origin's L1
    // Origin waits for count == num_cores - 1 before reading all partials
    uint32_t reduction_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    // Broadcast signal: origin multicasts 1 after norm is ready
    // All cores wait on this before starting Phase 4 (normalization)
    uint32_t bcast_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Multicast bounding box: physical coords covering all active cores
    auto get_phys = [&](uint32_t lx, uint32_t ly) -> tt::tt_metal::CoreCoord {
        return device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{lx, ly});
    };
    auto bb = all_cores.bounding_box();
    auto mcast_start = get_phys(bb.start_coord.x, bb.start_coord.y);
    auto mcast_end = get_phys(bb.end_coord.x, bb.end_coord.y);
    uint32_t mcast_num_dests = bb.size();

    // -------------------------------------------------------------------------
    // 4) Create kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    auto* output_buffer = output[0].buffer();

    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Got {}",
        enchantum::to_string(input_buffer->buffer_type()));
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Got {}",
        enchantum::to_string(output_buffer->buffer_type()));

    // Origin = core (0,0)
    tt::tt_metal::CoreCoord origin_core{0, 0};
    auto origin_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(origin_core));
    auto non_origin_cores = all_cores.subtract(origin_set);
    std::map<std::string, std::string> defines;
    auto origin_defines = defines;
    origin_defines["IS_ORIGIN"] = "1";

    auto origin_phys = get_phys(0, 0);
    std::vector<uint32_t> reader_ct_args{
        static_cast<uint32_t>(origin_phys.x),
        static_cast<uint32_t>(origin_phys.y),
        num_cores,
        static_cast<uint32_t>(mcast_start.x),
        static_cast<uint32_t>(mcast_start.y),
        static_cast<uint32_t>(mcast_end.x),
        static_cast<uint32_t>(mcast_end.y),
        mcast_num_dests,
        block_size};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);

    auto reader_origin = create_reader_kernel(program, origin_set, reader_ct_args, origin_defines, kReaderKernelPath);
    auto reader_kernel = create_reader_kernel(program, non_origin_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    auto writer_kernel = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest[kCbInput] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbSqAcc] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbRecv] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    auto make_compute_config = [&](const std::vector<uint32_t>& args, const std::map<std::string, std::string>& defs) {
        return tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest,
            .math_approx_mode = false,
            .compile_args = args,
            .defines = defs};
    };

    // Origin compute kernel (IS_ORIGIN=1)
    uint32_t origin_tiles = core_group_1.contains(origin_core) ? tiles_per_core_g1 : tiles_per_core_g2;
    auto compute_origin = tt::tt_metal::CreateKernel(
        program, kComputeKernelPath, origin_set, make_compute_config({origin_tiles, block_size}, origin_defines));

    // Non-origin compute kernels (core_group_1/2 minus origin)
    auto non_origin_g1 = core_group_1.subtract(origin_set);
    auto compute_g1 = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        non_origin_g1,
        make_compute_config({tiles_per_core_g1, block_size}, defines));

    tt::tt_metal::KernelHandle compute_g2{};
    if (!core_group_2.ranges().empty()) {
        auto non_origin_g2 = core_group_2.subtract(origin_set);
        if (!non_origin_g2.ranges().empty()) {
            compute_g2 = tt::tt_metal::CreateKernel(
                program,
                kComputeKernelPath,
                non_origin_g2,
                make_compute_config({tiles_per_core_g2, block_size}, defines));
        }
    }

    // -------------------------------------------------------------------------
    // 5) Set per-core runtime args
    // -------------------------------------------------------------------------
    uint32_t tiles_written = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t gx = i / num_cores_y;
        uint32_t gy = i % num_cores_y;
        tt::tt_metal::CoreCoord logical_core{gx, gy};

        uint32_t tiles_this_core = core_group_1.contains(logical_core) ? tiles_per_core_g1 : tiles_per_core_g2;

        // Reader runtime args
        auto reader_handle = (logical_core == origin_core) ? reader_origin : reader_kernel;
        SetRuntimeArgs(
            program,
            reader_handle,
            logical_core,
            {input_buffer->address(),
             tiles_this_core,
             tiles_written,
             reduction_sem_id,
             static_cast<uint32_t>(i),  // core_index
             bcast_sem_id});

        // Writer runtime args
        SetRuntimeArgs(
            program, writer_kernel, logical_core, {output_buffer->address(), tiles_this_core, tiles_written});

        // Compute runtime args
        auto compute_handle = (logical_core == origin_core)
                                  ? compute_origin
                                  : (core_group_1.contains(logical_core) ? compute_g1 : compute_g2);
        SetRuntimeArgs(program, compute_handle, logical_core, {std::bit_cast<uint32_t>(args.epsilon)});

        tiles_written += tiles_this_core;
    }

    // -------------------------------------------------------------------------
    // 7) Return cached program
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {reader_kernel,
         reader_origin,
         writer_kernel,
         compute_origin,
         compute_g1,
         compute_g2,
         core_group_1,
         core_group_2,
         num_cores,
         num_cores_y}};
}

void FrobeniusNormalizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FrobeniusNormalizeAttributes& operation_attributes,
    const FrobeniusNormalizeTensorArgs& tensor_args,
    FrobeniusNormalizeTensorReturn& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* input_buffer = tensor_args.input.buffer();
    auto* output_buffer = output[0].buffer();

    auto& reader_rt = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& reader_origin_rt = GetRuntimeArgs(program, shared.reader_origin_id);
    auto& writer_rt = GetRuntimeArgs(program, shared.writer_kernel_id);
    auto& compute_origin_rt = GetRuntimeArgs(program, shared.compute_origin_id);
    auto& compute_g1_rt = GetRuntimeArgs(program, shared.compute_group_1_id);
    auto& compute_g2_rt = GetRuntimeArgs(program, shared.compute_group_2_id);

    const uint32_t epsilon_bits = std::bit_cast<uint32_t>(operation_attributes.epsilon);
    const tt::tt_metal::CoreCoord origin_core{0, 0};

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        tt::tt_metal::CoreCoord core{i / shared.num_cores_y, i % shared.num_cores_y};
        auto& rt = (core == origin_core) ? reader_origin_rt : reader_rt;
        rt[core.x][core.y][kReaderInputAddrIdx] = input_buffer->address();
        writer_rt[core.x][core.y][kWriterOutputAddrIdx] = output_buffer->address();

        auto& compute_rt = (core == origin_core) ? compute_origin_rt
                                                 : (shared.core_group_1.contains(core) ? compute_g1_rt : compute_g2_rt);
        compute_rt[core.x][core.y][0] = epsilon_bits;
    }
}

}  // namespace ttml::metal::ops::frobenius_normalize::device
