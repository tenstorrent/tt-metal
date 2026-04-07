// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "frobenius_normalize_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "frobenius_normalize_device_operation_types.hpp"

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

// CB indices — must match kernel code
constexpr auto kCbInput = tt::CBIndex::c_0;
constexpr auto kCbSqAcc = tt::CBIndex::c_1;
constexpr auto kCbScalar = tt::CBIndex::c_2;
constexpr auto kCbRecv = tt::CBIndex::c_3;
constexpr auto kCbNorm = tt::CBIndex::c_4;
constexpr auto kCbOutput = tt::CBIndex::c_5;
// c_6 unused — was for reduce_tile scaler

// Runtime arg indices for reader
constexpr uint32_t kReaderInputAddrIdx = 0;

// Runtime arg indices for writer
constexpr uint32_t kWriterOutputAddrIdx = 0;

}  // namespace

namespace ttml::metal::ops::frobenius_normalize::device {

FrobeniusNormalizeProgramFactory::cached_program_t FrobeniusNormalizeProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::Program program{};

    // -------------------------------------------------------------------------
    // 1) Data formats and tile sizes
    // -------------------------------------------------------------------------
    tt::DataFormat bf16_format = tt::DataFormat::Float16_b;
    tt::DataFormat fp32_format = tt::DataFormat::Float32;

    uint32_t bf16_tile_size = tt::tile_size(bf16_format);
    uint32_t fp32_tile_size = tt::tile_size(fp32_format);

    // -------------------------------------------------------------------------
    // 2) Compute dimensions and core distribution
    // Use split_work_to_cores to distribute tiles. Chain reduction only runs
    // on active cores. The active set forms a column-major prefix.
    // -------------------------------------------------------------------------
    uint32_t total_tiles = input.physical_volume() / tt::constants::TILE_HW;

    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_tiles);

    // Compute the effective grid dimensions from the active cores.
    // Cores are assigned in column-major order: core i is at (i/num_cores_y, i%num_cores_y).
    // The active grid has active_grid_x full columns + possibly a partial last column.
    // Chain topology is computed per-core below based on active core indices

    // -------------------------------------------------------------------------
    // 3) Create circular buffers
    // -------------------------------------------------------------------------
    constexpr uint32_t input_double_buf = 2;
    constexpr uint32_t output_double_buf = 2;

    [[maybe_unused]] auto cb_input =
        create_circular_buffer(program, all_cores, kCbInput, bf16_format, bf16_tile_size, input_double_buf);
    [[maybe_unused]] auto cb_sq_acc =
        create_circular_buffer(program, all_cores, kCbSqAcc, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_scalar =
        create_circular_buffer(program, all_cores, kCbScalar, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_recv =
        create_circular_buffer(program, all_cores, kCbRecv, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_norm =
        create_circular_buffer(program, all_cores, kCbNorm, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kCbOutput, bf16_format, bf16_tile_size, output_double_buf);
    // c_6 (scaler) removed — not needed with sfpu_reduce

    // -------------------------------------------------------------------------
    // 4) Create semaphore (one per core, same L1 address on all cores)
    // -------------------------------------------------------------------------
    uint32_t sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // -------------------------------------------------------------------------
    // 5) Create kernels
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

    std::map<std::string, std::string> defines;

    // Reader compile-time args: [0] = packed_eps, then TensorAccessorArgs for input
    uint32_t packed_eps = pack_two_bfloat16_to_uint32(args.epsilon);
    std::vector<uint32_t> reader_ct_args{packed_eps};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);

    auto reader_kernel = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    // Writer compile-time args: TensorAccessorArgs for output
    std::vector<uint32_t> writer_ct_args;
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    auto writer_kernel = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    // UnpackToDestFp32: bypass srcA/B TF32 truncation for FP32 CBs.
    // cb_sq_acc: required for sfpu_reduce
    // cb_scalar, cb_recv: FP32 chain adds via copy_tile + add_binary_tile
    std::vector<UnpackToDestMode> unpack_to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest[kCbSqAcc] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbScalar] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbRecv] = UnpackToDestMode::UnpackToDestFp32;

    auto make_compute_config = [&](const std::vector<uint32_t>& args) {
        return tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest,
            .math_approx_mode = false,
            .compile_args = args,
            .defines = defines};
    };

    std::vector<uint32_t> compute_g1_args{tiles_per_core_g1};
    auto compute_g1 = tt::tt_metal::CreateKernel(
        program, kComputeKernelPath, core_group_1, make_compute_config(compute_g1_args));

    tt::tt_metal::KernelHandle compute_g2{};
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_g2_args{tiles_per_core_g2};
        compute_g2 = tt::tt_metal::CreateKernel(
            program, kComputeKernelPath, core_group_2, make_compute_config(compute_g2_args));
    }

    // -------------------------------------------------------------------------
    // 6) Set per-core runtime args
    // -------------------------------------------------------------------------
    auto get_phys = [&](uint32_t lx, uint32_t ly) -> tt::tt_metal::CoreCoord {
        return device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{lx, ly});
    };

    uint32_t tiles_written = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t gx = i / num_cores_y;
        uint32_t gy = i % num_cores_y;
        tt::tt_metal::CoreCoord logical_core{gx, gy};

        uint32_t tiles_this_core = core_group_1.contains(logical_core) ? tiles_per_core_g1 : tiles_per_core_g2;

        // Neighbor physical coords (set to 0,0 if not applicable — won't be used)
        uint32_t left_px = 0, left_py = 0;
        uint32_t up_px = 0, up_py = 0;
        uint32_t right_px = 0, right_py = 0;
        uint32_t down_px = 0, down_py = 0;

        if (gx > 0) {
            auto p = get_phys(gx - 1, gy);
            left_px = p.x;
            left_py = p.y;
        }
        if (gy > 0) {
            auto p = get_phys(gx, gy - 1);
            up_px = p.x;
            up_py = p.y;
        }
        // Right/down neighbors: only reference active cores
        uint32_t right_idx = (gx + 1) * num_cores_y + gy;
        if (right_idx < num_cores) {
            auto p = get_phys(gx + 1, gy);
            right_px = p.x;
            right_py = p.y;
        }
        if (gy + 1 < num_cores_y && (gx * num_cores_y + gy + 1) < num_cores) {
            auto p = get_phys(gx, gy + 1);
            down_px = p.x;
            down_py = p.y;
        }

        // Determine chain role based on active grid topology
        // Check if the core at (gx+1, gy) is active (exists in the active set)
        uint32_t right_core_idx = (gx + 1) * num_cores_y + gy;
        bool has_right = (right_core_idx < num_cores);
        bool has_left = (gx > 0);
        // Column chain: core (0, gy+1) must exist
        bool col_has_below = (gx == 0 && (gy + 1) < num_cores_y && (gy + 1) < num_cores);
        bool col_has_above = (gx == 0 && gy > 0);

        uint32_t do_row_receive = has_right ? 1 : 0;
        uint32_t do_row_send = has_left ? 1 : 0;
        uint32_t do_col_receive = col_has_below ? 1 : 0;
        uint32_t do_col_send = col_has_above ? 1 : 0;
        uint32_t is_origin = (gx == 0 && gy == 0) ? 1 : 0;

        // Broadcast forwarding flags
        uint32_t do_bcast_col_fwd = col_has_below ? 1 : 0;
        uint32_t do_bcast_row_fwd = has_right ? 1 : 0;

        // Reader runtime args
        SetRuntimeArgs(
            program,
            reader_kernel,
            logical_core,
            {input_buffer->address(),
             tiles_this_core,
             tiles_written,
             sem_id,
             do_row_receive,
             do_row_send,
             do_col_receive,
             do_col_send,
             is_origin,
             do_bcast_col_fwd,
             do_bcast_row_fwd,
             left_px,
             left_py,
             up_px,
             up_py,
             right_px,
             right_py,
             down_px,
             down_py});

        // Writer runtime args
        SetRuntimeArgs(
            program,
            writer_kernel,
            logical_core,
            {output_buffer->address(), tiles_this_core, tiles_written});

        // Compute runtime args
        auto compute_handle = core_group_1.contains(logical_core) ? compute_g1 : compute_g2;
        SetRuntimeArgs(
            program,
            compute_handle,
            logical_core,
            {do_row_receive, do_row_send, do_col_receive, do_col_send, is_origin});

        tiles_written += tiles_this_core;
    }

    // -------------------------------------------------------------------------
    // 7) Return cached program
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {reader_kernel, writer_kernel, compute_g1, compute_g2, core_group_1, core_group_2, num_cores, num_cores_y}};
}

void FrobeniusNormalizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* input_buffer = tensor_args.input.buffer();
    auto* output_buffer = output[0].buffer();

    auto& reader_rt = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_rt = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        tt::tt_metal::CoreCoord core{i / shared.num_cores_y, i % shared.num_cores_y};
        reader_rt[core.x][core.y][kReaderInputAddrIdx] = input_buffer->address();
        writer_rt[core.x][core.y][kWriterOutputAddrIdx] = output_buffer->address();
    }
}

}  // namespace ttml::metal::ops::frobenius_normalize::device
