// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_rows_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "add_rmsnorm_fw_rows_device_operation.hpp"
#include "kernels/add_rmsnorm_fw_rows_cbs.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/device/kernels/dataflow/reader_add_rmsnorm_fw_rows.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/device/kernels/dataflow/writer_add_rmsnorm_fw_rows.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/device/kernels/compute/add_rmsnorm_fw_rows_kernel.cpp";

constexpr uint32_t kBlockTiles = 4U;  // pass-2 output block
constexpr uint32_t kGroupTiles =
    4U;  // pass-1 group: residual / delta streamed four tiles at a time (fp32 DEST holds 4)
constexpr uint32_t kMaxSlices = 16U;  // iteration 44: 13 slices x 8 rows on the sequence-parallel shard (104 cores)

constexpr uint32_t kReaderResidualIdx = 0U;
constexpr uint32_t kReaderDeltaIdx = 1U;
constexpr uint32_t kReaderGammaIdx = 2U;
constexpr uint32_t kWriterSumIdx = 0U;
constexpr uint32_t kWriterOutIdx = 1U;
constexpr uint32_t kWriterRmsIdx = 2U;

}  // namespace

namespace ttml::metal::ops::add_rmsnorm_fw_rows::device {

// One item (tile-row r, slice s) per core, core index r * S + s. Residual and delta are streamed (groups of two
// tiles, double-buffered), gamma and the bf16 sum are resident; the row's S cores exchange their partial sums of
// squares by writing them into each other's cb::partials slot s and incrementing a semaphore (rmsnorm_bw_rows).
AddRMSNormForwardRowsProgramFactory::cached_program_t AddRMSNormForwardRowsProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    namespace cb = add_rmsnorm_fw_rows_cb;
    const auto& residual = tensor_args.residual;
    auto* device = residual.device();
    tt::tt_metal::Program program{};
    auto& sum = outputs[0];
    auto& out = outputs[1];
    auto& rms = outputs[2];

    const auto padded = residual.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t S = args.num_slices;
    const uint32_t St = args.slice_tiles;
    const uint32_t block = std::min(kBlockTiles, St);
    const uint32_t num_cores = rows * S;
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        num_cores <= grid.x * grid.y,
        "AddRMSNormForwardRows: {} items do not fit {} cores",
        num_cores,
        grid.x * grid.y);
    TT_FATAL(S <= kMaxSlices, "AddRMSNormForwardRows: at most {} slices per row, got {}", kMaxSlices, S);
    const auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid, false);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t f32_tile = tt::tile_size(f32);
    create_circular_buffer(program, all_cores, cb::residual, bf16, bf16_tile, 2U * kGroupTiles);
    create_circular_buffer(program, all_cores, cb::delta, bf16, bf16_tile, 2U * kGroupTiles);
    create_circular_buffer(program, all_cores, cb::gamma, bf16, bf16_tile, St);
    create_circular_buffer(program, all_cores, cb::sum, bf16, bf16_tile, St);
    create_circular_buffer(program, all_cores, cb::zero, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::ones, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::acc, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::partial_out, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::partials, f32, f32_tile, S);
    create_circular_buffer(program, all_cores, cb::row_acc, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::rms_tmp, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::rms_out, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::inv, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::out, bf16, bf16_tile, 2U * block);
    const uint32_t exchange_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    std::map<std::string, std::string> defines;

    if (args.delta_transposed) {
        defines["DELTA_TRANSPOSED"] = "1";  // the reader fetches delta's tile (c, r), the compute transposes it in DEST
        create_circular_buffer(program, all_cores, cb::delta_t, bf16, bf16_tile, 2U * kGroupTiles);
    }
    const uint32_t Ht = padded[2] / tt::constants::TILE_HEIGHT;  // delta^T's tile-row stride
    std::vector<uint32_t> reader_ct_args{Wt, S, St, block, exchange_sem, Ht};
    tt::tt_metal::TensorAccessorArgs(residual.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.delta.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.gamma.buffer()).append_to(reader_ct_args);
    auto reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{Wt, S, St, block, exchange_sem, Wt + args.out_tail_tiles, args.out_tail_tiles};
    tt::tt_metal::TensorAccessorArgs(sum.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(out.buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(rms.buffer()).append_to(writer_ct_args);
    auto writer = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    const uint32_t inv_c_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(residual.logical_shape()[-1]));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(args.epsilon);
    auto compute = create_compute_kernel(
        program,
        all_cores,
        {Wt, S, St, block, inv_c_bits, eps_bits},
        defines,
        kComputeKernelPath,
        /*fp32_dest_acc_en=*/true);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / grid.y, i % grid.y};
        const uint32_t r = i / S;
        const uint32_t s = i - r * S;
        SetRuntimeArgs(program, compute, core, {r, s});
        SetRuntimeArgs(
            program,
            reader,
            core,
            {residual.buffer()->address(),
             tensor_args.delta.buffer()->address(),
             tensor_args.gamma.buffer()->address(),
             r,
             s});
        // Writer: sum / out / rms addresses, (r, s), then the row's S cores' NoC coordinates (x, y) in slice order.
        std::vector<uint32_t> w_args{sum.buffer()->address(), out.buffer()->address(), rms.buffer()->address(), r, s};
        for (uint32_t j = 0; j < S; ++j) {
            const uint32_t k = r * S + j;
            const auto phys = device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{k / grid.y, k % grid.y});
            w_args.push_back(static_cast<uint32_t>(phys.x));
            w_args.push_back(static_cast<uint32_t>(phys.y));
        }
        SetRuntimeArgs(program, writer, core, w_args);
        cores.push_back(core);
    }
    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void AddRMSNormForwardRowsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderResidualIdx] = tensor_args.residual.buffer()->address();
        r[kReaderDeltaIdx] = tensor_args.delta.buffer()->address();
        r[kReaderGammaIdx] = tensor_args.gamma.buffer()->address();
        auto& w = writer_args[core.x][core.y];
        w[kWriterSumIdx] = outputs[0].buffer()->address();
        w[kWriterOutIdx] = outputs[1].buffer()->address();
        w[kWriterRmsIdx] = outputs[2].buffer()->address();
    }
}

}  // namespace ttml::metal::ops::add_rmsnorm_fw_rows::device
