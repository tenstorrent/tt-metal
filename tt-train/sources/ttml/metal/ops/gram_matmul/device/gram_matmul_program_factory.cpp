// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_matmul_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/cb_utils.hpp"

namespace ttml::metal::ops::gram_matmul::device {

namespace {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> determine_default_block_sizes(bool fp32_dest_acc_en) {
    uint32_t M_block_tiles = 8;
    uint32_t K_block_tiles = 8;
    uint32_t N_block_tiles = 8;

    // Output is always square (M == N), so subblocks are symmetric when not using fp32 dest accumulation.
    uint32_t subblock_h = 2;
    uint32_t subblock_w = fp32_dest_acc_en ? 2 : 4;

    return {M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w};
}

// Build a linear order of cores along one axis for data movement, plus index of the current core.
// axis_is_x: true means the axis runs along the x coordinate, false means along y.
std::pair<std::vector<tt::tt_metal::CoreCoord>, uint32_t> build_core_order_for_axis(
    const tt::tt_metal::CoreCoord& core,
    uint32_t axis_length,
    tt::tt_metal::NOC noc,
    bool axis_is_x,
    const tt::tt_metal::CoreCoord& initial_endpoint) {
    std::vector<tt::tt_metal::CoreCoord> order;
    order.reserve(axis_length);
    order.push_back(initial_endpoint);

    const size_t current_axis_value = axis_is_x ? core.x : core.y;

    // Direction along the axis: increasing for NOC_0, decreasing for NOC_1
    const bool increasing = (noc == tt::tt_metal::NOC::NOC_0);

    uint32_t index_of_current = 0;  // default to 0 if axis_length == 1
    for (uint32_t worker_idx = 1; worker_idx < axis_length; ++worker_idx) {
        tt::tt_metal::CoreCoord worker_core = core;
        size_t& coord_to_modify = axis_is_x ? worker_core.x : worker_core.y;

        coord_to_modify = increasing ? worker_idx : (axis_length - worker_idx);
        if (coord_to_modify == current_axis_value) {
            index_of_current = worker_idx;
        }
        order.push_back(worker_core);
    }
    return {order, index_of_current};
}

tt::tt_metal::CoreCoord clamped_prev(const std::vector<tt::tt_metal::CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

tt::tt_metal::CoreCoord clamped_next(const std::vector<tt::tt_metal::CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

// Append tensor accessors in a consistent order
void append_accessors(std::vector<uint32_t>& args, const ttnn::Tensor& main_tensor, const ttnn::Tensor& output_tensor) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
}

}  // namespace

GramMatmulProgramFactory::cached_program_t GramMatmulProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    Program program = CreateProgram();

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& config = operation_attributes.config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto* device = input_tensor.device();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    /**
     * Determine dataformats, compute kernel config.
     * Both inputs are always BFLOAT16.
     */
    auto in0_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    auto out_tile_size = tt::tile_size(output_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // Intermediate CB dataformat is the same datatype as DST register.
    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    auto in0_tensor_shape = input_tensor.padded_shape();
    auto in1_tensor_shape = weight_tensor.padded_shape();
    // Fold activation (LHS) upper dimensions into rows: M_total = prod(upper dims) * M
    uint32_t K = in0_tensor_shape[-1];
    uint32_t M = input_tensor.physical_volume() / K;
    uint32_t N = in1_tensor_shape[-1];

    // Output is always square: M == N (enforced by validation)

    uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    auto [default_M_block_tiles, default_K_block_tiles, default_N_block_tiles, default_subblock_h, default_subblock_w] =
        determine_default_block_sizes(fp32_dest_acc_en);

    uint32_t subblock_h = config.has_value() ? config.value().subblock_h : default_subblock_h;
    uint32_t subblock_w = config.has_value() ? config.value().subblock_w : default_subblock_w;

    uint32_t M_block_tiles = config.has_value() ? config.value().M_block_size : default_M_block_tiles;
    uint32_t K_block_tiles = config.has_value() ? config.value().K_block_size : default_K_block_tiles;
    uint32_t N_block_tiles = config.has_value() ? config.value().N_block_size : default_N_block_tiles;

    /**
     * Core grid assignment for Gram matmul.
     *
     * Since the output is always square (M == N), we never need to transpose the core grid.
     * in0 (activation) is multicast across rows:   (0, core_y) -> (grid_size.x-1, core_y)
     * in1 (weight)     is multicast across columns: (core_x, 0) -> (core_x, grid_size.y-1)
     *
     * in0 DM runs on RISCV_1 / NOC preferred for DRAM writes (also handles output writes).
     * in1 DM runs on RISCV_0 / NOC preferred for DRAM reads.
     */

    auto in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto in0_risc = DataMovementProcessor::RISCV_1;
    uint32_t in0_parallel_axis_cores = grid_size.y;

    auto in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto in1_risc = DataMovementProcessor::RISCV_0;
    uint32_t in1_parallel_axis_cores = grid_size.x;

    uint32_t padded_M_tiles = tt::round_up(M_tiles, in0_parallel_axis_cores);
    uint32_t padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);
    uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);

    uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    uint32_t N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered

    auto core_0_0 = CoreCoord{0, 0};
    auto core_0_1 = CoreCoord{0, 1};
    auto core_1_0 = CoreCoord{1, 0};
    auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};

    // in0 sender cores: first column (x=0)
    auto in0_sender_cores = CoreRange(core_0_0, core_0_endy);
    // in0 receiver cores: remaining columns (x>=1)
    auto in0_receiver_cores = CoreRange(core_1_0, core_endx_endy);
    // in1 sender cores: first row (y=0)
    auto in1_sender_cores = CoreRange(core_0_0, core_endx_0);
    // in1 receiver cores: remaining rows (y>=1)
    auto in1_receiver_cores = CoreRange(core_0_1, core_endx_endy);

    auto in0_sender_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = CreateSemaphore(program, core_grid, VALID);

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(in0_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::create_cb(in1_cb_id, program, core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format);

    uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, out_tile_size, out_cb_num_tiles, output_data_format);

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    tt::tt_metal::create_cb(
        intermediate_cb_id, program, core_grid, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    std::map<std::string, std::string> defines;

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t out_addr = tensor_return_value.buffer()->address();

    /**
     * Create kernels.
     * in0 DM always writes output (is_output_writer = true).
     * in1 DM never writes output  (is_output_writer = false).
     */

    constexpr bool in0_is_output_writer = true;
    constexpr bool in1_is_output_writer = false;

    std::vector<uint32_t> in0_sender_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        true,  // is_injector_core
    };
    append_accessors(in0_sender_compile_time_args, input_tensor, tensor_return_value);
    auto in0_sender_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/gram_matmul/device/kernels/dm_in0_sender.cpp",
        in0_sender_cores,
        DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_sender_compile_time_args, .defines = defines});

    std::vector<uint32_t> in0_receiver_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        false,  // is_injector_core
    };
    append_accessors(in0_receiver_compile_time_args, input_tensor, tensor_return_value);

    auto in0_receiver_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/gram_matmul/device/kernels/dm_in0_sender.cpp",
        in0_receiver_cores,
        DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_receiver_compile_time_args, .defines = defines});

    std::vector<uint32_t> in1_sender_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        true,  // is_injector_core
    };
    append_accessors(in1_sender_compile_time_args, weight_tensor, tensor_return_value);
    auto in1_sender_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/gram_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_sender_cores,
        DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_sender_compile_time_args, .defines = defines});

    std::vector<uint32_t> in1_receiver_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        false,  // is_injector_core
    };
    append_accessors(in1_receiver_compile_time_args, weight_tensor, tensor_return_value);
    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/gram_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_receiver_cores,
        DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_receiver_compile_time_args, .defines = defines});

    std::vector<uint32_t> compute_compile_time_args = {
        K_blocks,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        subblock_h,
        subblock_w};

    auto compute_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/gram_matmul/device/kernels/compute.cpp",
        core_grid,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    /**
     * The receiver writer cores defer their writes in order to reduce NOC congestion.
     * Further, the amount of K_blocks they defer by depends on their core coordinate.
     */
    uint32_t k_blocks_per_core = tt::div_up(K_blocks, in0_parallel_axis_cores);

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        // in0 is parallelized along y, in1 along x
        uint32_t in0_idx = core.y;
        uint32_t in1_idx = core.x;

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            in1_parallel_axis_cores,
            in0_noc,
            /*axis_is_x=*/true,
            /*initial_endpoint=*/left_core);

        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            in0_parallel_axis_cores,
            in1_noc,
            /*axis_is_x=*/false,
            /*initial_endpoint=*/top_core);

        auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
        auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
        auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
        auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

        uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);
        uint32_t N_start_tile = N_tiles_per_core * in1_idx;
        uint32_t N_end_tile = N_tiles_per_core * (in1_idx + 1);

        // Defer write to K block with same coordinate as core
        uint32_t defer_write_k_block = core.y * k_blocks_per_core;
        defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        // RT args layout: [in_addr, is_sink, noc_coords(4), tile_ranges(4), defer_k, out_addr]
        std::vector<uint32_t> in0_args = {
            in0_addr,
            is_in0_sink,
            (std::uint32_t)in0_next_core_physical.x,
            (std::uint32_t)in0_next_core_physical.y,
            (std::uint32_t)in0_prev_core_physical.x,
            (std::uint32_t)in0_prev_core_physical.y,
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            out_addr,
        };
        if (in1_idx == 0) {
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else {
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
        }

        std::vector<uint32_t> in1_args = {
            in1_addr,
            is_in1_sink,
            (std::uint32_t)in1_next_core_physical.x,
            (std::uint32_t)in1_next_core_physical.y,
            (std::uint32_t)in1_prev_core_physical.x,
            (std::uint32_t)in1_prev_core_physical.y,
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            out_addr,
        };
        if (in0_idx == 0) {
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_args);
        } else {
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_args);
        }

        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    return {
        std::move(program),
        GramMatmulProgramFactory::shared_variables_t{
            num_cores,
            cores,
            in0_sender_kernels_id,
            in0_receiver_kernels_id,
            in1_sender_kernels_id,
            in1_receiver_kernels_id}};
}

void GramMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& override_variables = cached_program.shared_variables;

    auto& in0_sender_runtime_args = GetRuntimeArgs(program, override_variables.in0_sender_kernels_id);
    auto& in0_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in0_receiver_kernels_id);
    auto& in1_sender_runtime_args = GetRuntimeArgs(program, override_variables.in1_sender_kernels_id);
    auto& in1_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in1_receiver_kernels_id);

    auto in0_addr = tensor_args.input_tensor.buffer()->address();
    auto in1_addr = tensor_args.weight_tensor.buffer()->address();
    auto out_addr = tensor_return_value.buffer()->address();

    // RT args layout: [in_addr, is_sink, noc_coords(4), tile_ranges(4), defer_k, out_addr]
    constexpr uint32_t out_addr_idx = 11;

    for (uint32_t i = 0; i < override_variables.num_cores; ++i) {
        tt::tt_metal::CoreCoord core = override_variables.cores.at(i);
        // in0 parallelized along y, in1 along x
        uint32_t in0_idx = core.y;
        uint32_t in1_idx = core.x;

        if (in1_idx == 0) {
            auto& in0_sender_args = in0_sender_runtime_args[core.x][core.y];
            in0_sender_args[0] = in0_addr;
            in0_sender_args[out_addr_idx] = out_addr;
        } else {
            auto& in0_receiver_args = in0_receiver_runtime_args[core.x][core.y];
            in0_receiver_args[out_addr_idx] = out_addr;
        }

        if (in0_idx == 0) {
            auto& in1_sender_args = in1_sender_runtime_args[core.x][core.y];
            in1_sender_args[0] = in1_addr;
            in1_sender_args[out_addr_idx] = out_addr;
        } else {
            auto& in1_receiver_args = in1_receiver_runtime_args[core.x][core.y];
            in1_receiver_args[out_addr_idx] = out_addr;
        }
    }
}

}  // namespace ttml::metal::ops::gram_matmul::device
