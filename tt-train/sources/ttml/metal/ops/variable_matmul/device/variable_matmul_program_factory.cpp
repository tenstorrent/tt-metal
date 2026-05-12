// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variable_matmul_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttml::metal::ops::variable_matmul::device {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using ttnn::Tensor;

namespace {

// Build a linear order of cores along one axis for data movement, plus index of the current core
std::pair<std::vector<CoreCoord>, uint32_t> build_core_order_for_axis(
    const CoreCoord& core,
    bool transpose_core_grid,
    uint32_t axis_length,
    tt::tt_metal::NOC noc,
    bool axis_is_x_when_not_transposed,
    const CoreCoord& initial_endpoint) {
    std::vector<CoreCoord> order;
    order.reserve(axis_length);
    order.push_back(initial_endpoint);

    const size_t current_axis_value = transpose_core_grid ? (axis_is_x_when_not_transposed ? core.y : core.x)
                                                          : (axis_is_x_when_not_transposed ? core.x : core.y);

    const bool increasing = (noc == tt::tt_metal::NOC::NOC_0);

    uint32_t index_of_current = 0;
    for (uint32_t worker_idx = 1; worker_idx < axis_length; ++worker_idx) {
        CoreCoord worker_core = core;
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

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

// Append tensor accessors for input + output (no bias/ternary/AG)
void append_accessors(std::vector<uint32_t>& args, const Tensor& main_tensor, const Tensor& output_tensor) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
}

}  // namespace

VariableMatmulProgramFactory::cached_program_t VariableMatmulProgramFactory::create(
    const VariableMatmulParams& operation_attributes,
    const VariableMatmulInputs& tensor_args,
    ttnn::Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& config = operation_attributes.config;
    auto* device = input_tensor.device();

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    // ----- Data formats -----
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    auto out_tile_size = tt::tile_size(output_data_format);

    // in2 (bias) tile size — unused but kept for compile-time arg layout compatibility
    auto in2_tile_size = in1_tile_size;
    // in3 (AG input) tile size — unused but kept for compile-time arg layout compatibility
    auto in3_tile_size = in1_tile_size;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    auto in0_tensor_shape = input_tensor.padded_shape();
    auto in1_tensor_shape = weight_tensor.padded_shape();
    const bool transpose_a = config.transpose_a;
    const bool transpose_b = config.transpose_b;
    // Matmul-K source depends on which side is the parent buffer:
    //   - in1_k_offset > 0 (or K_w > K_in): weight is the parent, matmul-K = K_in
    //   - else: in0 may be the parent (or both match), matmul-K = K_w
    uint32_t K_w = transpose_b ? in1_tensor_shape[-1] : in1_tensor_shape[-2];
    uint32_t N = transpose_b ? in1_tensor_shape[-2] : in1_tensor_shape[-1];
    uint32_t K_in = transpose_a ? in0_tensor_shape[-2] : in0_tensor_shape[-1];
    uint32_t parent_K_tiles_in0 = K_in / tt::constants::TILE_WIDTH;
    uint32_t parent_K_tiles_in1 = K_w / tt::constants::TILE_WIDTH;
    const bool in1_parent_k_mode =
        operation_attributes.in1_k_offset_tiles > 0 || parent_K_tiles_in1 > parent_K_tiles_in0;
    uint32_t K_tiles = in1_parent_k_mode ? parent_K_tiles_in0 : parent_K_tiles_in1;
    uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    // M extent of the input — independent of K offsets. parent_M is the input's stored M
    // dimension (matmul-M = stored inner when transpose_a, stored outer otherwise).
    uint32_t parent_M = transpose_a ? in0_tensor_shape[-1] : in0_tensor_shape[-2];
    uint32_t parent_M_tiles = parent_M / tt::constants::TILE_HEIGHT;

    // effective_M_tiles overrides for offset-read mode; otherwise process the whole input.
    uint32_t actual_M_tiles =
        (operation_attributes.effective_M_tiles > 0) ? operation_attributes.effective_M_tiles : parent_M_tiles;
    uint32_t actual_M = actual_M_tiles * tt::constants::TILE_HEIGHT;

    // Two-program strategy: transpose_core_grid from PARENT-M vs N (not effective_M) so the
    // grid decision is stable across offset-read calls on the same parent tensor.
    // At most 2 cached programs (one per transpose variant).
    bool transpose_core_grid = parent_M > N;

    uint32_t M_block_tiles = config.M_block_size;
    uint32_t K_block_tiles = config.K_block_size;
    uint32_t N_block_tiles = config.N_block_size;
    uint32_t subblock_h = config.subblock_h;
    uint32_t subblock_w = config.subblock_w;

    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;

    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;

    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    // ----- M tile counts (from actual_M — used for both compile-time and runtime args) -----
    uint32_t actual_padded_M_tiles = tt::round_up(actual_M_tiles, in0_parallel_axis_cores);
    uint32_t padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);
    uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);

    uint32_t actual_M_tiles_per_core = actual_padded_M_tiles / in0_parallel_axis_cores;
    uint32_t N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t actual_M_blocks_per_core = tt::div_up(actual_M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    // ----- CB sizing (depends only on block sizes from config, not on total M) -----
    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered

    log_debug(tt::LogOp, "variable_matmul: actual_M={}, K={}, N={}", actual_M, K_w, N);
    log_debug(
        tt::LogOp,
        "variable_matmul: actual_M_tiles_per_core={}, actual_M_blocks_per_core={}",
        actual_M_tiles_per_core,
        actual_M_blocks_per_core);
    log_debug(tt::LogOp, "variable_matmul: transpose_core_grid={} (from actual_M > N)", transpose_core_grid);

    // ----- Sender/receiver core ranges -----
    auto core_0_0 = CoreCoord{0, 0};
    auto core_0_1 = CoreCoord{0, 1};
    auto core_1_0 = CoreCoord{1, 0};
    auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};

    auto in0_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    auto in0_receiver_cores = CoreRange(transpose_core_grid ? core_0_1 : core_1_0, core_endx_endy);
    auto in1_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    auto in1_receiver_cores = CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    // ----- Semaphores -----
    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    // ----- Circular buffers -----
    uint32_t in0_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(in0_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::create_cb(in1_cb_id, program, core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format);

    uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, out_tile_size, out_cb_num_tiles, output_data_format);

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    tt::tt_metal::create_cb(
        intermediate_cb_id, program, core_grid, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    // CB for the transposed in0 block (only used when transpose_a is true). Same data format
    // and capacity as in0 since the transpose preserves both. Compute kernel transposes
    // c_0 -> c_7 tile-by-tile, then matmul consumes c_7.
    if (transpose_a) {
        constexpr uint32_t in0_transposed_cb_id = tt::CBIndex::c_7;
        tt::tt_metal::create_cb(
            in0_transposed_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);
    }

    // ----- Defines (empty — no FUSE_BIAS, FUSE_TERNARY, FUSE_AG) -----
    std::map<std::string, std::string> defines;

    // ----- Kernel compile-time args -----
    // Layout matches original minimal_matmul exactly (22 args for in0, 21 for in1) so TensorAccessor
    // offsets remain correct. Indices 0, 1, 9 are unused by kernels (kept for arg layout compat).
    // N_chunks=1, N_tiles_per_chunk=N_tiles (dummy values for stripped features).

    bool in0_is_output_writer = !transpose_core_grid;
    bool in1_is_output_writer = transpose_core_grid;

    // in0 sender compile-time args (22 fixed + tensor accessor args)
    std::vector<uint32_t> in0_sender_compile_time_args = {
        actual_M_tiles,                      // 0: M_tiles (unused by kernel, kept for arg layout compat)
        actual_padded_M_tiles,               // 1: padded_M_tiles (max)
        K_tiles,                             // 2: K_tiles
        padded_K_tiles,                      // 3: padded_K_tiles
        N_tiles,                             // 4: N_tiles
        padded_N_tiles,                      // 5: padded_N_tiles
        M_block_tiles,                       // 6: M_block_tiles
        K_block_tiles,                       // 7: K_block_tiles
        N_block_tiles,                       // 8: N_block_tiles
        actual_M_blocks_per_core,            // 9: M_blocks_per_core (unused by kernel, kept for arg layout compat)
        N_blocks_per_core,                   // 10: N_blocks_per_core
        in0_tile_size,                       // 11: in_tile_size
        out_tile_size,                       // 12: out_tile_size
        in2_tile_size,                       // 13: in2_tile_size (dummy, no bias)
        in0_sender_semaphore_id,             // 14: sender_sem_id
        in0_receiver_semaphore_id,           // 15: receiver_sem_id
        in0_valid_semaphore_id,              // 16: valid_sem_id
        in0_is_output_writer,                // 17: is_output_writer
        true,                                // 18: is_injector_core
        1U,                                  // 19: N_chunks (always 1)
        N_tiles,                             // 20: N_tiles_per_chunk (= N_tiles when N_chunks=1)
        in3_tile_size,                       // 21: in3_tile_size (dummy, no AG)
        static_cast<uint32_t>(transpose_a),  // 22: transpose_a
        // 23: use_offset — when false (caller has offset_tiles=0 and effective_M_tiles=0),
        // the kernel skips the per-tile offset add in the address formula. The runtime
        // args still carry offset / parent_M_tiles_stride for hot reload, but they're
        // not read. This avoids paying the per-tile address-compute overhead for the
        // common (no offset) case (e.g. all moe-ffn backward calls).
        static_cast<uint32_t>(
            operation_attributes.in0_row_offset_tiles > 0 || operation_attributes.effective_M_tiles > 0 ||
            operation_attributes.in0_k_offset_tiles > 0 || parent_K_tiles_in0 > K_tiles),
    };
    append_accessors(in0_sender_compile_time_args, input_tensor, output_tensor);

    auto in0_sender_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in0_sender.cpp",
        in0_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_sender_compile_time_args, .defines = defines});

    // in0 receiver compile-time args (same layout, is_injector_core=false)
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        actual_M_tiles,
        actual_padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        actual_M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        false,    // is_injector_core
        1U,       // N_chunks
        N_tiles,  // N_tiles_per_chunk
        in3_tile_size,
        static_cast<uint32_t>(transpose_a),  // 22: transpose_a
        static_cast<uint32_t>(
            operation_attributes.in0_row_offset_tiles > 0 || operation_attributes.effective_M_tiles > 0 ||
            operation_attributes.in0_k_offset_tiles > 0 || parent_K_tiles_in0 > K_tiles),  // 23: use_offset
    };
    append_accessors(in0_receiver_compile_time_args, input_tensor, output_tensor);

    auto in0_receiver_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in0_sender.cpp",
        in0_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_receiver_compile_time_args, .defines = defines});

    // in1 sender compile-time args (22 fixed + tensor accessor args; index 21 = transpose_b)
    std::vector<uint32_t> in1_sender_compile_time_args = {
        actual_M_tiles,
        actual_padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        actual_M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        true,                                // is_injector_core
        1U,                                  // N_chunks
        N_tiles,                             // N_tiles_per_chunk
        static_cast<uint32_t>(transpose_b),  // 21: transpose_b
        // 22: use_offset_in1 — K-offset path on the weight (analogous to in0's use_offset).
        static_cast<uint32_t>(operation_attributes.in1_k_offset_tiles > 0 || parent_K_tiles_in1 > K_tiles),
    };
    append_accessors(in1_sender_compile_time_args, weight_tensor, output_tensor);

    auto in1_sender_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in1_sender_out.cpp",
        in1_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_sender_compile_time_args, .defines = defines});

    // in1 receiver compile-time args
    std::vector<uint32_t> in1_receiver_compile_time_args = {
        actual_M_tiles,
        actual_padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        actual_M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        false,                               // is_injector_core
        1U,                                  // N_chunks
        N_tiles,                             // N_tiles_per_chunk
        static_cast<uint32_t>(transpose_b),  // 21: transpose_b
        static_cast<uint32_t>(
            operation_attributes.in1_k_offset_tiles > 0 || parent_K_tiles_in1 > K_tiles),  // 22: use_offset_in1
    };
    append_accessors(in1_receiver_compile_time_args, weight_tensor, output_tensor);

    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in1_sender_out.cpp",
        in1_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_receiver_compile_time_args, .defines = defines});

    // ----- Compute kernel -----
    std::vector<uint32_t> compute_compile_time_args = {
        K_blocks,                            // 0: K_num_blocks
        M_block_tiles,                       // 1: M_block_tiles
        K_block_tiles,                       // 2: K_block_tiles
        N_block_tiles,                       // 3: N_block_tiles
        actual_M_blocks_per_core,            // 4: M_blocks_per_core (unused, kept for arg layout compat)
        N_blocks_per_core,                   // 5: N_blocks_per_core
        subblock_h,                          // 6: subblock_h
        subblock_w,                          // 7: subblock_w
        static_cast<uint32_t>(transpose_b),  // 8: transpose_b
        static_cast<uint32_t>(transpose_a),  // 9: transpose_a
    };

    std::map<std::string, std::string> compute_defines;
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        compute_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));

    auto compute_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/compute/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    // ----- Per-core runtime args -----
    uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in1_parallel_axis_cores,
            in0_noc,
            /*axis_is_x_when_not_transposed=*/true,
            /*initial_endpoint=*/(transpose_core_grid ? top_core : left_core));

        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in0_parallel_axis_cores,
            in1_noc,
            /*axis_is_x_when_not_transposed=*/false,
            /*initial_endpoint=*/(transpose_core_grid ? left_core : top_core));

        auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
        auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
        auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
        auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

        // M tile ranges use actual_M (runtime), N tile ranges use padded_N (compile-time)
        uint32_t M_start_tile = actual_M_tiles_per_core * in0_idx;
        uint32_t M_end_tile = actual_M_tiles_per_core * (in0_idx + 1);
        uint32_t N_start_tile = N_tiles_per_core * in1_idx;
        uint32_t N_end_tile = N_tiles_per_core * (in1_idx + 1);

        uint32_t defer_write_k_block = core.y * k_blocks_per_core;
        defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        // in0 runtime args layout:
        //  0: in0_addr
        //  1: in2_addr (0, no bias)
        //  2: in3_addr (0, no AG)
        //  3: is_sink_core
        //  4: in0_dest_noc_x
        //  5: in0_dest_noc_y
        //  6: in0_sender_noc_x
        //  7: in0_sender_noc_y
        //  8: M_start_tile     (in effective-M coord, per-core partition)
        //  9: M_end_tile
        // 10: N_start_tile
        // 11: N_end_tile
        // 12: defer_write_k_block
        // 13: out_addr
        // 14: actual_M_tiles            (= effective_M_tiles)
        // 15: actual_padded_M_tiles
        // 16: actual_M_blocks_per_core
        // 17: in0_row_offset_tiles      (offset into parent input's M axis, in tiles)
        // 18: parent_M_tiles_stride     (parent M tile count; used as K-row stride for transpose_a)
        // 19: in0_k_offset_tiles        (offset into parent input's K axis, in tiles)
        // 20: parent_K_tiles_stride     (parent K tile count; used as M-row stride for non-transpose)
        std::vector<uint32_t> in0_args = {
            in0_addr,
            0U,  // in2_addr (no bias)
            0U,  // in3_addr (no AG)
            static_cast<uint32_t>(is_in0_sink),
            static_cast<uint32_t>(in0_next_core_physical.x),
            static_cast<uint32_t>(in0_next_core_physical.y),
            static_cast<uint32_t>(in0_prev_core_physical.x),
            static_cast<uint32_t>(in0_prev_core_physical.y),
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            out_addr,
            actual_M_tiles,
            actual_padded_M_tiles,
            actual_M_blocks_per_core,
            operation_attributes.in0_row_offset_tiles,
            parent_M_tiles,
            operation_attributes.in0_k_offset_tiles,
            parent_K_tiles_in0,
        };

        if (in1_idx == 0) {
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else {
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
        }

        // in1 runtime args layout:
        //  0: in1_addr
        //  1: in2_addr (0, no bias)
        //  2: is_sink_core
        //  3: dest_noc_x
        //  4: dest_noc_y
        //  5: sender_noc_x
        //  6: sender_noc_y
        //  7: M_start_tile
        //  8: M_end_tile
        //  9: N_start_tile
        // 10: N_end_tile
        // 11: defer_write_k_block
        // 12: out_addr
        // 13: actual_M_tiles
        // 14: actual_padded_M_tiles
        // 15: actual_M_blocks_per_core
        // 16: in1_k_offset_tiles    (offset into parent weight's K axis, in tiles)
        // 17: parent_K_tiles_in1    (parent K tile count; row stride for transpose_b)
        std::vector<uint32_t> in1_args = {
            in1_addr,
            0U,  // in2_addr (no bias)
            static_cast<uint32_t>(is_in1_sink),
            static_cast<uint32_t>(in1_next_core_physical.x),
            static_cast<uint32_t>(in1_next_core_physical.y),
            static_cast<uint32_t>(in1_prev_core_physical.x),
            static_cast<uint32_t>(in1_prev_core_physical.y),
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            out_addr,
            actual_M_tiles,
            actual_padded_M_tiles,
            actual_M_blocks_per_core,
            operation_attributes.in1_k_offset_tiles,
            parent_K_tiles_in1,
        };

        if (in0_idx == 0) {
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_args);
        } else {
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_args);
        }

        // Compute runtime args:
        //  0: M_start_tile
        //  1: M_end_tile
        //  2: N_start_tile
        //  3: N_end_tile
        //  4: actual_M_blocks_per_core
        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            actual_M_blocks_per_core,
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    return {
        std::move(program),
        shared_variables_t{
            .num_cores = static_cast<uint32_t>(num_cores),
            .cores = std::move(cores),
            .in0_sender_kernels_id = in0_sender_kernels_id,
            .in0_receiver_kernels_id = in0_receiver_kernels_id,
            .in1_sender_kernels_id = in1_sender_kernels_id,
            .in1_receiver_kernels_id = in1_receiver_kernels_id,
            .compute_kernels_id = compute_kernels_id,
            .transpose_core_grid = transpose_core_grid,
            .in0_parallel_axis_cores = in0_parallel_axis_cores,
            .in1_parallel_axis_cores = in1_parallel_axis_cores,
            .M_block_tiles = M_block_tiles,
            .N_tiles_per_core = N_tiles_per_core,
        }};
}

void VariableMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const VariableMatmulParams& operation_attributes,
    const VariableMatmulInputs& tensor_args,
    ttnn::Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    // Recompute actual M values. Parent M comes from the input tensor; the matmul itself
    // processes effective_M_tiles when set (offset-read mode), else the full parent.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& a_padded = input_tensor.padded_shape();
    const auto& w_padded = tensor_args.weight_tensor.padded_shape();
    const bool transpose_a = operation_attributes.config.transpose_a;
    const bool transpose_b = operation_attributes.config.transpose_b;
    uint32_t parent_M_tiles = (transpose_a ? a_padded[-1] : a_padded[-2]) / tt::constants::TILE_HEIGHT;
    uint32_t parent_K_tiles_in0 = (transpose_a ? a_padded[-2] : a_padded[-1]) / tt::constants::TILE_WIDTH;
    uint32_t parent_K_tiles_in1 = (transpose_b ? w_padded[-1] : w_padded[-2]) / tt::constants::TILE_WIDTH;
    uint32_t actual_M_tiles =
        (operation_attributes.effective_M_tiles > 0) ? operation_attributes.effective_M_tiles : parent_M_tiles;
    uint32_t actual_padded_M_tiles = tt::round_up(actual_M_tiles, sv.in0_parallel_axis_cores);
    uint32_t actual_M_tiles_per_core = actual_padded_M_tiles / sv.in0_parallel_axis_cores;
    uint32_t actual_M_blocks_per_core = tt::div_up(actual_M_tiles_per_core, sv.M_block_tiles);

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();

    auto& in0_sender_rt = GetRuntimeArgs(program, sv.in0_sender_kernels_id);
    auto& in0_receiver_rt = GetRuntimeArgs(program, sv.in0_receiver_kernels_id);
    auto& in1_sender_rt = GetRuntimeArgs(program, sv.in1_sender_kernels_id);
    auto& in1_receiver_rt = GetRuntimeArgs(program, sv.in1_receiver_kernels_id);
    auto& compute_rt = GetRuntimeArgs(program, sv.compute_kernels_id);

    // in0 runtime arg indices
    constexpr uint32_t IN0_ADDR_IDX = 0;
    constexpr uint32_t IN0_M_START_IDX = 8;
    constexpr uint32_t IN0_M_END_IDX = 9;
    constexpr uint32_t IN0_OUT_ADDR_IDX = 13;
    constexpr uint32_t IN0_ACTUAL_M_TILES_IDX = 14;
    constexpr uint32_t IN0_ACTUAL_PADDED_M_TILES_IDX = 15;
    constexpr uint32_t IN0_ACTUAL_M_BLOCKS_IDX = 16;
    constexpr uint32_t IN0_ROW_OFFSET_TILES_IDX = 17;
    constexpr uint32_t IN0_PARENT_M_TILES_STRIDE_IDX = 18;
    constexpr uint32_t IN0_K_OFFSET_TILES_IDX = 19;
    constexpr uint32_t IN0_PARENT_K_TILES_STRIDE_IDX = 20;

    // in1 runtime arg indices
    constexpr uint32_t IN1_ADDR_IDX = 0;
    constexpr uint32_t IN1_M_START_IDX = 7;
    constexpr uint32_t IN1_M_END_IDX = 8;
    constexpr uint32_t IN1_OUT_ADDR_IDX = 12;
    constexpr uint32_t IN1_ACTUAL_M_TILES_IDX = 13;
    constexpr uint32_t IN1_ACTUAL_PADDED_M_TILES_IDX = 14;
    constexpr uint32_t IN1_ACTUAL_M_BLOCKS_IDX = 15;
    constexpr uint32_t IN1_K_OFFSET_TILES_IDX = 16;
    constexpr uint32_t IN1_PARENT_K_TILES_STRIDE_IDX = 17;

    // Compute runtime arg indices
    constexpr uint32_t COMPUTE_M_START_IDX = 0;
    constexpr uint32_t COMPUTE_M_END_IDX = 1;
    constexpr uint32_t COMPUTE_ACTUAL_M_BLOCKS_IDX = 4;

    for (uint32_t i = 0; i < sv.num_cores; ++i) {
        CoreCoord core = sv.cores.at(i);
        uint32_t in0_idx = sv.transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = sv.transpose_core_grid ? core.y : core.x;

        uint32_t M_start_tile = actual_M_tiles_per_core * in0_idx;
        uint32_t M_end_tile = actual_M_tiles_per_core * (in0_idx + 1);

        // Update in0 args
        if (in1_idx == 0) {
            auto& args = in0_sender_rt[core.x][core.y];
            args[IN0_ADDR_IDX] = in0_addr;
            args[IN0_M_START_IDX] = M_start_tile;
            args[IN0_M_END_IDX] = M_end_tile;
            args[IN0_OUT_ADDR_IDX] = out_addr;
            args[IN0_ACTUAL_M_TILES_IDX] = actual_M_tiles;
            args[IN0_ACTUAL_PADDED_M_TILES_IDX] = actual_padded_M_tiles;
            args[IN0_ACTUAL_M_BLOCKS_IDX] = actual_M_blocks_per_core;
            args[IN0_ROW_OFFSET_TILES_IDX] = operation_attributes.in0_row_offset_tiles;
            args[IN0_PARENT_M_TILES_STRIDE_IDX] = parent_M_tiles;
            args[IN0_K_OFFSET_TILES_IDX] = operation_attributes.in0_k_offset_tiles;
            args[IN0_PARENT_K_TILES_STRIDE_IDX] = parent_K_tiles_in0;
        } else {
            auto& args = in0_receiver_rt[core.x][core.y];
            args[IN0_M_START_IDX] = M_start_tile;
            args[IN0_M_END_IDX] = M_end_tile;
            args[IN0_OUT_ADDR_IDX] = out_addr;
            args[IN0_ACTUAL_M_TILES_IDX] = actual_M_tiles;
            args[IN0_ACTUAL_PADDED_M_TILES_IDX] = actual_padded_M_tiles;
            args[IN0_ACTUAL_M_BLOCKS_IDX] = actual_M_blocks_per_core;
            args[IN0_ROW_OFFSET_TILES_IDX] = operation_attributes.in0_row_offset_tiles;
            args[IN0_PARENT_M_TILES_STRIDE_IDX] = parent_M_tiles;
            args[IN0_K_OFFSET_TILES_IDX] = operation_attributes.in0_k_offset_tiles;
            args[IN0_PARENT_K_TILES_STRIDE_IDX] = parent_K_tiles_in0;
        }

        // Update in1 args
        if (in0_idx == 0) {
            auto& args = in1_sender_rt[core.x][core.y];
            args[IN1_ADDR_IDX] = in1_addr;
            args[IN1_M_START_IDX] = M_start_tile;
            args[IN1_M_END_IDX] = M_end_tile;
            args[IN1_OUT_ADDR_IDX] = out_addr;
            args[IN1_ACTUAL_M_TILES_IDX] = actual_M_tiles;
            args[IN1_ACTUAL_PADDED_M_TILES_IDX] = actual_padded_M_tiles;
            args[IN1_ACTUAL_M_BLOCKS_IDX] = actual_M_blocks_per_core;
            args[IN1_K_OFFSET_TILES_IDX] = operation_attributes.in1_k_offset_tiles;
            args[IN1_PARENT_K_TILES_STRIDE_IDX] = parent_K_tiles_in1;
        } else {
            auto& args = in1_receiver_rt[core.x][core.y];
            args[IN1_M_START_IDX] = M_start_tile;
            args[IN1_M_END_IDX] = M_end_tile;
            args[IN1_OUT_ADDR_IDX] = out_addr;
            args[IN1_ACTUAL_M_TILES_IDX] = actual_M_tiles;
            args[IN1_ACTUAL_PADDED_M_TILES_IDX] = actual_padded_M_tiles;
            args[IN1_ACTUAL_M_BLOCKS_IDX] = actual_M_blocks_per_core;
            args[IN1_K_OFFSET_TILES_IDX] = operation_attributes.in1_k_offset_tiles;
            args[IN1_PARENT_K_TILES_STRIDE_IDX] = parent_K_tiles_in1;
        }

        // Update compute args
        auto& cargs = compute_rt[core.x][core.y];
        cargs[COMPUTE_M_START_IDX] = M_start_tile;
        cargs[COMPUTE_M_END_IDX] = M_end_tile;
        cargs[COMPUTE_ACTUAL_M_BLOCKS_IDX] = actual_M_blocks_per_core;
    }
}

}  // namespace ttml::metal::ops::variable_matmul::device
