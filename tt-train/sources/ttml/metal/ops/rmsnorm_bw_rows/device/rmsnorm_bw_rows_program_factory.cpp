// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_rows_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/rmsnorm_bw_rows_cbs.hpp"
#include "metal/common/program_utils.hpp"
#include "rmsnorm_bw_rows_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/kernels/dataflow/reader_rmsnorm_bw_rows.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/kernels/dataflow/writer_rmsnorm_bw_rows.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/kernels/compute/rmsnorm_bw_rows_kernel.cpp";

constexpr uint32_t kBlockTiles = 4U;
constexpr uint32_t kMaxSlices = 8U;

constexpr uint32_t kReaderInputIdx = 0U;
constexpr uint32_t kReaderGammaIdx = 1U;
constexpr uint32_t kReaderDyIdx = 2U;
constexpr uint32_t kReaderRmsIdx = 3U;
constexpr uint32_t kReaderDresIdx = 4U;
constexpr uint32_t kWriterOutIdx = 0U;
constexpr uint32_t kWriterOutTIdx = 1U;

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw_rows::device {

// One item (tile-row r, slice s) per core, core index r * S + s; the row's S cores exchange their partial dot
// products by writing them into each other's cb::partials slot s and incrementing a semaphore.
RMSNormBackwardRowsProgramFactory::cached_program_t RMSNormBackwardRowsProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    auto& output = outputs[0];
    namespace cb = rmsnorm_bw_rows_cb;
    const auto& input = tensor_args.input;
    auto* device = input.device();
    tt::tt_metal::Program program{};

    const auto padded = input.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t S = args.num_slices;
    const uint32_t St = args.slice_tiles;
    const uint32_t block = std::min(kBlockTiles, St);
    const uint32_t num_cores = rows * S;
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        num_cores <= grid.x * grid.y, "RMSNormBackwardRows: {} items do not fit {} cores", num_cores, grid.x * grid.y);
    TT_FATAL(S <= kMaxSlices, "RMSNormBackwardRows: at most {} slices per row, got {}", kMaxSlices, S);
    const auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid, false);
    const bool add_residual_grad = tensor_args.residual_grad.has_value();

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t f32_tile = tt::tile_size(f32);
    create_circular_buffer(program, all_cores, cb::a, bf16, bf16_tile, St);
    create_circular_buffer(program, all_cores, cb::gamma, bf16, bf16_tile, St);
    create_circular_buffer(program, all_cores, cb::dy, bf16, bf16_tile, St);
    create_circular_buffer(program, all_cores, cb::zero, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::partial_out, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::rms, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::partials, f32, f32_tile, S);
    create_circular_buffer(program, all_cores, cb::ones, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::ones_row0, bf16, bf16_tile, 1U);
    create_circular_buffer(program, all_cores, cb::inv, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::acc, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::t, f32, f32_tile, 1U);
    create_circular_buffer(program, all_cores, cb::dx, bf16, bf16_tile, 2U * block);
    create_circular_buffer(program, all_cores, cb::acc1, f32, f32_tile, 1U);
    if (add_residual_grad) {
        create_circular_buffer(program, all_cores, cb::dres, bf16, bf16_tile, 2U * block);
    }
    if (args.transposed_grad) {
        create_circular_buffer(program, all_cores, cb::dx_copy, bf16, bf16_tile, 2U * block);
        create_circular_buffer(program, all_cores, cb::dx_t, bf16, bf16_tile, 2U * block);
    }
    const uint32_t exchange_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    std::map<std::string, std::string> defines;
    if (add_residual_grad) {
        defines["ADD_RESIDUAL_GRAD"] = "1";
    }
    if (args.transposed_grad) {
        defines["DX_TRANSPOSED"] = "1";  // the compute transposes each dL_da block through DEST, the writer scatters it
    }
    const uint32_t Ht = padded[2] / tt::constants::TILE_HEIGHT;  // tile rows: dL_da^T's tile-row stride

    std::vector<uint32_t> reader_ct_args{Wt, S, St, block, exchange_sem};
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.gamma.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.dL_dout.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.rms.buffer()).append_to(reader_ct_args);
    if (add_residual_grad) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.residual_grad->buffer()).append_to(reader_ct_args);
    }
    auto reader = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{Wt, S, St, block, exchange_sem, Ht};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    if (args.transposed_grad) {
        tt::tt_metal::TensorAccessorArgs(outputs[1].buffer()).append_to(writer_ct_args);
    }
    auto writer = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);

    // -1/C: pass 2 accumulates a * (-t) onto gamma * dy on the FPU (iteration 29).
    const uint32_t neg_inv_c_bits = std::bit_cast<uint32_t>(-1.0F / static_cast<float>(input.logical_shape()[-1]));
    auto compute = create_compute_kernel(
        program, all_cores, {Wt, S, St, block, neg_inv_c_bits}, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

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
            {input.buffer()->address(),
             tensor_args.gamma.buffer()->address(),
             tensor_args.dL_dout.buffer()->address(),
             tensor_args.rms.buffer()->address(),
             add_residual_grad ? tensor_args.residual_grad->buffer()->address() : 0U,
             r,
             s});
        // Writer: output address, (r, s), then the row's S cores' NoC coordinates (x, y) in slice order.
        std::vector<uint32_t> w_args{
            output.buffer()->address(), args.transposed_grad ? outputs[1].buffer()->address() : 0U, r, s};
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

void RMSNormBackwardRowsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& output = outputs[0];
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderInputIdx] = tensor_args.input.buffer()->address();
        r[kReaderGammaIdx] = tensor_args.gamma.buffer()->address();
        r[kReaderDyIdx] = tensor_args.dL_dout.buffer()->address();
        r[kReaderRmsIdx] = tensor_args.rms.buffer()->address();
        if (tensor_args.residual_grad.has_value()) {
            r[kReaderDresIdx] = tensor_args.residual_grad->buffer()->address();
        }
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
        if (args.transposed_grad) {
            writer_args[core.x][core.y][kWriterOutTIdx] = outputs[1].buffer()->address();
        }
    }
}

}  // namespace ttml::metal::ops::rmsnorm_bw_rows::device
