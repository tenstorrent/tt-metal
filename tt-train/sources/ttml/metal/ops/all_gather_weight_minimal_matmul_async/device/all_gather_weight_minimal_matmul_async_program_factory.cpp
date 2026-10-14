// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_weight_minimal_matmul_async_program_factory.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>
#include <vector>

#include "stripe_layout.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
namespace {
constexpr const char* kKernelDir =
    "tt-train/sources/ttml/metal/ops/all_gather_weight_minimal_matmul_async/device/kernels/";

std::pair<std::vector<ttnn::CoreCoord>, uint32_t> build_core_order_for_axis(
    const ttnn::CoreCoord& core,
    bool transpose_core_grid,
    uint32_t axis_length,
    tt::tt_metal::NOC noc,
    bool axis_is_x_when_not_transposed,
    const ttnn::CoreCoord& initial_endpoint) {
    std::vector<ttnn::CoreCoord> order;
    order.reserve(axis_length);
    order.push_back(initial_endpoint);
    const size_t current_axis_value = transpose_core_grid ? (axis_is_x_when_not_transposed ? core.y : core.x)
                                                          : (axis_is_x_when_not_transposed ? core.x : core.y);
    const bool increasing = (noc == tt::tt_metal::NOC::NOC_0);
    uint32_t index_of_current = 0;
    for (uint32_t worker_idx = 1; worker_idx < axis_length; ++worker_idx) {
        ttnn::CoreCoord worker_core = core;
        size_t& coord_to_modify = transpose_core_grid ? (axis_is_x_when_not_transposed ? worker_core.y : worker_core.x)
                                                      : (axis_is_x_when_not_transposed ? worker_core.x : worker_core.y);
        coord_to_modify = increasing ? worker_idx : (axis_length - worker_idx);
        if (coord_to_modify == current_axis_value) {
            index_of_current = worker_idx;
        }
        order.push_back(worker_core);
    }
    return {order, index_of_current};
}

ttnn::CoreCoord clamped_prev(const std::vector<ttnn::CoreCoord>& order, uint32_t index) {
    return order[index == 0 ? 0 : index - 1];
}

ttnn::CoreCoord clamped_next(const std::vector<ttnn::CoreCoord>& order, uint32_t index) {
    const auto last = static_cast<uint32_t>(order.size() - 1);
    return order[index >= last ? last : index + 1];
}

constexpr uint32_t kRtInputAddrIdx = 0;
constexpr uint32_t kRtOutputAddrIdx = 1;
}  // namespace

AllGatherWeightMinimalMatmulAsyncProgramFactory::cached_program_t
AllGatherWeightMinimalMatmulAsyncProgramFactory::create(
    const AllGatherWeightMinimalMatmulAsyncParams& operation_attributes,
    const AllGatherWeightMinimalMatmulAsyncInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& weight_tensor = tensor_args.weight_tensor;
    const ttnn::Tensor& output_tensor = tensor_return_value;
    const auto& config = operation_attributes.config;
    auto* device = input_tensor.device();

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = ttnn::CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    auto out_tile_size = tt::tile_size(output_data_format);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    const uint32_t K = input_tensor.padded_shape()[-1];
    const uint32_t M = input_tensor.physical_volume() / K;
    const uint32_t N = weight_tensor.padded_shape()[-2];
    const uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    const uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    const uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    uint32_t M_block_tiles = 8;
    uint32_t K_block_tiles = 8;
    uint32_t N_block_tiles = 8;
    uint32_t subblock_h = 2;
    uint32_t subblock_w = 2;
    if (!fp32_dest_acc_en) {
        subblock_h = N >= M ? 2 : 4;
        subblock_w = N >= M ? 4 : 2;
    }
    if (config.has_value()) {
        M_block_tiles = config.value().M_block_size;
        K_block_tiles = config.value().K_block_size;
        N_block_tiles = config.value().N_block_size;
        subblock_h = config.value().subblock_h;
        subblock_w = config.value().subblock_w;
    }

    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;
    const bool transpose_core_grid = M > N;
    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;
    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    const uint32_t padded_M_tiles = tt::round_up(M_tiles, in0_parallel_axis_cores);
    const uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);
    const uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    const uint32_t K_blocks = padded_K_tiles / K_block_tiles;
    const uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);

    const StripeLayout layout = make_stripe_layout(
        N_tiles,
        operation_attributes.ring_size,
        operation_attributes.ring_index,
        in1_parallel_axis_cores,
        N_block_tiles,
        operation_attributes.topology == ttnn::ccl::Topology::Ring);
    const uint32_t N_blocks_per_core = layout.n_blocks_per_core();

    const uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    const uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    constexpr uint32_t double_buffer_factor = 2;
    tt::tt_metal::create_cb(
        tt::CBIndex::c_0,
        program,
        core_grid,
        in0_tile_size,
        in0_block_num_tiles * double_buffer_factor,
        in0_data_format);
    tt::tt_metal::create_cb(
        tt::CBIndex::c_1,
        program,
        core_grid,
        in1_tile_size,
        in1_block_num_tiles * double_buffer_factor,
        in1_data_format);
    tt::tt_metal::create_cb(
        tt::CBIndex::c_2,
        program,
        core_grid,
        out_tile_size,
        out_block_num_tiles * double_buffer_factor,
        output_data_format);
    tt::tt_metal::create_cb(
        tt::CBIndex::c_3, program, core_grid, intermediate_tile_size, out_block_num_tiles, intermediate_data_format);

    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    auto core_0_0 = ttnn::CoreCoord{0, 0};
    auto core_0_1 = ttnn::CoreCoord{0, 1};
    auto core_1_0 = ttnn::CoreCoord{1, 0};
    auto core_endx_0 = ttnn::CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = ttnn::CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = ttnn::CoreCoord{grid_size.x - 1, grid_size.y - 1};
    auto in0_sender_cores = ttnn::CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    auto in0_receiver_cores = ttnn::CoreRange(transpose_core_grid ? core_0_1 : core_1_0, core_endx_endy);
    auto in1_sender_cores = ttnn::CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    auto in1_receiver_cores = ttnn::CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    const bool in0_is_output_writer = !transpose_core_grid;
    const bool in1_is_output_writer = transpose_core_grid;

    auto dm_compile_args = [&](uint32_t input_tile_size,
                               uint32_t sender_sem,
                               uint32_t receiver_sem,
                               uint32_t valid_sem,
                               bool is_output_writer,
                               bool is_injector,
                               const ttnn::Tensor& own_input) {
        std::vector<uint32_t> args = {
            M_tiles,
            padded_M_tiles,
            K_tiles,
            padded_K_tiles,
            N_tiles,
            M_block_tiles,
            K_block_tiles,
            N_block_tiles,
            M_blocks_per_core,
            N_blocks_per_core,
            input_tile_size,
            out_tile_size,
            sender_sem,
            receiver_sem,
            valid_sem,
            static_cast<uint32_t>(is_output_writer),
            static_cast<uint32_t>(is_injector),
            layout.ring_size,
            layout.stripe_tiles,
            layout.piece_tiles,
            subblock_w,
        };
        tt::tt_metal::TensorAccessorArgs(*own_input.buffer()).append_to(args);
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
        return args;
    };

    const std::string in0_source = std::string(kKernelDir) + "dataflow/dm_in0.cpp";
    const std::string in1_source = std::string(kKernelDir) + "dataflow/dm_in1.cpp";

    auto make_dm_kernel = [&](const std::string& source,
                              const ttnn::CoreRange& cores,
                              tt::tt_metal::DataMovementProcessor risc,
                              tt::tt_metal::NOC noc,
                              std::vector<uint32_t> compile_args) {
        return tt::tt_metal::CreateKernel(
            program,
            source,
            cores,
            tt::tt_metal::DataMovementConfig{.processor = risc, .noc = noc, .compile_args = std::move(compile_args)});
    };

    auto in0_sender_kernels_id = make_dm_kernel(
        in0_source,
        in0_sender_cores,
        in0_risc,
        in0_noc,
        dm_compile_args(
            in0_tile_size,
            in0_sender_semaphore_id,
            in0_receiver_semaphore_id,
            in0_valid_semaphore_id,
            in0_is_output_writer,
            true,
            input_tensor));
    auto in0_receiver_kernels_id = make_dm_kernel(
        in0_source,
        in0_receiver_cores,
        in0_risc,
        in0_noc,
        dm_compile_args(
            in0_tile_size,
            in0_sender_semaphore_id,
            in0_receiver_semaphore_id,
            in0_valid_semaphore_id,
            in0_is_output_writer,
            false,
            input_tensor));
    auto in1_sender_kernels_id = make_dm_kernel(
        in1_source,
        in1_sender_cores,
        in1_risc,
        in1_noc,
        dm_compile_args(
            in1_tile_size,
            in1_sender_semaphore_id,
            in1_receiver_semaphore_id,
            in1_valid_semaphore_id,
            in1_is_output_writer,
            true,
            weight_tensor));
    auto in1_receiver_kernels_id = make_dm_kernel(
        in1_source,
        in1_receiver_cores,
        in1_risc,
        in1_noc,
        dm_compile_args(
            in1_tile_size,
            in1_sender_semaphore_id,
            in1_receiver_semaphore_id,
            in1_valid_semaphore_id,
            in1_is_output_writer,
            false,
            weight_tensor));

    std::map<std::string, std::string> compute_defines;
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        compute_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));
    auto compute_kernels_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "compute/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args =
                {K_blocks,
                 M_block_tiles,
                 K_block_tiles,
                 N_block_tiles,
                 M_blocks_per_core,
                 N_blocks_per_core,
                 subblock_h,
                 subblock_w,
                 layout.piece_tiles,
                 layout.ring_size},
            .defines = compute_defines});

    const uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);
    for (const auto& core : cores) {
        const uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        const uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        ttnn::CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        ttnn::CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};
        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in1_parallel_axis_cores,
            in0_noc,
            true,
            (transpose_core_grid ? top_core : left_core));
        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in0_parallel_axis_cores,
            in1_noc,
            false,
            (transpose_core_grid ? left_core : top_core));
        auto in0_prev = device->worker_core_from_logical_core(clamped_prev(in0_core_order, in0_core_order_index));
        auto in0_next = device->worker_core_from_logical_core(clamped_next(in0_core_order, in0_core_order_index));
        auto in1_prev = device->worker_core_from_logical_core(clamped_prev(in1_core_order, in1_core_order_index));
        auto in1_next = device->worker_core_from_logical_core(clamped_next(in1_core_order, in1_core_order_index));

        const uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        const uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);
        const uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);
        const bool is_in0_sink = core == in0_core_order.back();
        const bool is_in1_sink = core == in1_core_order.back();

        const uint32_t chain_offset = layout.chain_offset(in1_idx);
        const uint32_t chain_width = layout.chain_width(in1_idx);

        auto dm_runtime_args =
            [&](uint32_t input_addr, bool is_sink, const ttnn::CoreCoord& next, const ttnn::CoreCoord& prev) {
                std::vector<uint32_t> args = {
                    input_addr,
                    output_tensor.buffer()->address(),
                    static_cast<uint32_t>(is_sink),
                    static_cast<uint32_t>(next.x),
                    static_cast<uint32_t>(next.y),
                    static_cast<uint32_t>(prev.x),
                    static_cast<uint32_t>(prev.y),
                    M_start_tile,
                    M_end_tile,
                    chain_offset,
                    chain_width,
                    defer_write_k_block,
                };
                args.insert(args.end(), layout.stripe_order.begin(), layout.stripe_order.end());
                return args;
            };

        tt::tt_metal::SetRuntimeArgs(
            program,
            in1_idx == 0 ? in0_sender_kernels_id : in0_receiver_kernels_id,
            core,
            dm_runtime_args(input_tensor.buffer()->address(), is_in0_sink, in0_next, in0_prev));
        tt::tt_metal::SetRuntimeArgs(
            program,
            in0_idx == 0 ? in1_sender_kernels_id : in1_receiver_kernels_id,
            core,
            dm_runtime_args(weight_tensor.buffer()->address(), is_in1_sink, in1_next, in1_prev));
        tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {M_start_tile, M_end_tile});
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .cores = cores,
            .in0_sender_kernels_id = in0_sender_kernels_id,
            .in0_receiver_kernels_id = in0_receiver_kernels_id,
            .in1_sender_kernels_id = in1_sender_kernels_id,
            .in1_receiver_kernels_id = in1_receiver_kernels_id,
            .transpose_core_grid = transpose_core_grid}};
}

void AllGatherWeightMinimalMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const AllGatherWeightMinimalMatmulAsyncParams&,
    const AllGatherWeightMinimalMatmulAsyncInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& vars = cached_program.shared_variables;
    auto& in0_sender_args = GetRuntimeArgs(program, vars.in0_sender_kernels_id);
    auto& in0_receiver_args = GetRuntimeArgs(program, vars.in0_receiver_kernels_id);
    auto& in1_sender_args = GetRuntimeArgs(program, vars.in1_sender_kernels_id);
    auto& in1_receiver_args = GetRuntimeArgs(program, vars.in1_receiver_kernels_id);
    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    for (const auto& core : vars.cores) {
        const uint32_t in0_idx = vars.transpose_core_grid ? core.x : core.y;
        const uint32_t in1_idx = vars.transpose_core_grid ? core.y : core.x;
        auto& in0_args = (in1_idx == 0 ? in0_sender_args : in0_receiver_args)[core.x][core.y];
        in0_args[kRtInputAddrIdx] = in0_addr;
        in0_args[kRtOutputAddrIdx] = out_addr;
        auto& in1_args = (in0_idx == 0 ? in1_sender_args : in1_receiver_args)[core.x][core.y];
        in1_args[kRtInputAddrIdx] = in1_addr;
        in1_args[kRtOutputAddrIdx] = out_addr;
    }
}
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
