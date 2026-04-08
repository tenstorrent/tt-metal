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

    [[maybe_unused]] uint32_t bf16_tile_size = tt::tile_size(bf16_format);
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
        create_circular_buffer(program, all_cores, kCbInput, fp32_format, fp32_tile_size, input_double_buf);
    [[maybe_unused]] auto cb_sq_acc =
        create_circular_buffer(program, all_cores, kCbSqAcc, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_scalar =
        create_circular_buffer(program, all_cores, kCbScalar, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_recv =
        create_circular_buffer(program, all_cores, kCbRecv, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_norm =
        create_circular_buffer(program, all_cores, kCbNorm, fp32_format, fp32_tile_size, 1);
    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kCbOutput, fp32_format, fp32_tile_size, output_double_buf);
    // c_6 (scaler) removed — not needed with sfpu_reduce

    // -------------------------------------------------------------------------
    // 4) Create semaphores and scalar storage
    // -------------------------------------------------------------------------
    uint32_t chain_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    uint32_t bcast_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    // Repurpose a semaphore slot for 4-byte FP32 norm scalar storage (multicast target)
    uint32_t norm_scalar_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // Compute multicast bounding box (physical coords of active grid)
    auto get_phys = [&](uint32_t lx, uint32_t ly) -> tt::tt_metal::CoreCoord {
        return device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{lx, ly});
    };
    // Active grid spans logical (0,0) to (max_gx, max_gy).
    // Cores are assigned column-major: core i is at (i/num_cores_y, i%num_cores_y).
    // The bounding box must cover exactly the active cores.
    uint32_t max_gx = (num_cores - 1) / num_cores_y;
    // Last column may be partial; max_gy is the largest y among active cores
    uint32_t max_gy = (max_gx == 0) ? (num_cores - 1) : (num_cores_y - 1);
    auto mcast_start = get_phys(0, 0);
    auto mcast_end = get_phys(max_gx, max_gy);
    // num_dests for multicast = total cores in the bounding box rectangle
    // For a rectangular grid from (0,0) to (max_gx, max_gy): (max_gx+1) * (max_gy+1) physical cores
    // But the last column may be partial, so we use num_cores directly.
    // HOWEVER: multicast sends to ALL cores in the rectangle, which may include
    // inactive cores if the last column is partial. We must use the full rectangle count.
    uint32_t mcast_num_dests = (max_gx + 1) * (max_gy + 1);

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

    // Reader compile-time args: TensorAccessorArgs for input (eps moved to compute runtime args)
    std::vector<uint32_t> reader_ct_args;
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
    unpack_to_dest[kCbInput] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbSqAcc] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbScalar] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[kCbRecv] = UnpackToDestMode::UnpackToDestFp32;
    // cb_norm: NOT UnpackToDestFp32 — filled by generate_tile_with_uint32_value
    // which writes standard tile layout, not unpack-to-dest format.
    // TF32 truncation on a uniform scalar is negligible.

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
    uint32_t tiles_written = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t gx = i / num_cores_y;
        uint32_t gy = i % num_cores_y;
        tt::tt_metal::CoreCoord logical_core{gx, gy};

        uint32_t tiles_this_core = core_group_1.contains(logical_core) ? tiles_per_core_g1 : tiles_per_core_g2;

        // Neighbor physical coords (set to 0,0 if not applicable — won't be used)
        // Chain flows left→right (row) then top→bottom (column) — with NOC_0 natural direction.
        // Row chain: send to right neighbor. Column chain runs on rightmost column, sends down.
        // Origin = bottom-right active core.
        uint32_t right_px = 0, right_py = 0;
        uint32_t down_px = 0, down_py = 0;

        uint32_t right_core_idx = (gx + 1) * num_cores_y + gy;
        bool has_right = (right_core_idx < num_cores);
        bool has_left = (gx > 0);

        if (has_right) {
            auto p = get_phys(gx + 1, gy);
            right_px = p.x;
            right_py = p.y;
        }

        // Column chain: rightmost column. A core is the row leader (rightmost in its row)
        // if it has no right neighbor.
        bool is_row_end = !has_right;
        // Column neighbor below: the rightmost core in row gy+1
        // For a full-column case (all rows have same width), that's (max_gx, gy+1).
        // For partial last column, row gy+1 might have fewer columns.
        // The rightmost core in row gy+1: find max gx such that gx*num_cores_y + (gy+1) < num_cores.
        bool col_has_below = false;
        if (is_row_end && (gy + 1) < num_cores_y) {
            // Find rightmost core in row gy+1
            for (int gx2 = static_cast<int>(max_gx); gx2 >= 0; --gx2) {
                uint32_t idx = gx2 * num_cores_y + (gy + 1);
                if (idx < num_cores) {
                    col_has_below = true;
                    auto p = get_phys(gx2, gy + 1);
                    down_px = p.x;
                    down_py = p.y;
                    break;
                }
            }
        }
        uint32_t do_row_receive = has_left ? 1 : 0;     // receive from left
        uint32_t do_row_send = has_right ? 1 : 0;       // send to right
        uint32_t do_col_receive = (is_row_end && gy > 0) ? 1 : 0;  // receive from above (row end only)
        uint32_t do_col_send = col_has_below ? 1 : 0;   // send to below (row end only)
        // Origin = bottom-right: row end with no core below
        uint32_t is_origin = (is_row_end && !col_has_below) ? 1 : 0;

        // Reader runtime args
        SetRuntimeArgs(
            program,
            reader_kernel,
            logical_core,
            {input_buffer->address(),
             tiles_this_core,
             tiles_written,
             chain_sem_id,
             do_row_receive,
             do_row_send,
             do_col_receive,
             do_col_send,
             is_origin,
             right_px,
             right_py,
             down_px,
             down_py,
             bcast_sem_id,
             norm_scalar_sem_id,
             static_cast<uint32_t>(mcast_start.x),
             static_cast<uint32_t>(mcast_start.y),
             static_cast<uint32_t>(mcast_end.x),
             static_cast<uint32_t>(mcast_end.y),
             mcast_num_dests});

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
            {do_row_receive, do_row_send, do_col_receive, do_col_send, is_origin,
             std::bit_cast<uint32_t>(args.epsilon)});

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
