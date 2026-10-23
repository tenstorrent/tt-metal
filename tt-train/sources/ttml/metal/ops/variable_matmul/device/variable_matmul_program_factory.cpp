// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

// Append tensor accessors for input + output
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

    const auto grid_size = config.compute_with_storage_grid_size;
    const auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const auto num_cores = core_grid.size();

    // ----- Data formats -----
    const auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto in0_tile_size = tt::tile_size(in0_data_format);
    const auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    const auto in1_tile_size = tt::tile_size(in1_data_format);
    const auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const auto out_tile_size = tt::tile_size(output_data_format);

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    const auto in0_tensor_shape = input_tensor.padded_shape();
    const auto in1_tensor_shape = weight_tensor.padded_shape();
    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;
    // Matmul-K source depends on which side is the parent buffer:
    //   - K_w > K_in: weight is the parent, matmul-K = K_in
    //   - else: in0 may be the parent (or both match), matmul-K = K_w
    const uint32_t K_w = transpose_b ? in1_tensor_shape[-1] : in1_tensor_shape[-2];
    const uint32_t N = transpose_b ? in1_tensor_shape[-2] : in1_tensor_shape[-1];
    const uint32_t K_in = transpose_a ? in0_tensor_shape[-2] : in0_tensor_shape[-1];
    const uint32_t parent_K_tiles_in0 = K_in / tt::constants::TILE_WIDTH;
    const uint32_t parent_K_tiles_in1 = K_w / tt::constants::TILE_WIDTH;
    const bool in1_parent_k_mode = parent_K_tiles_in1 > parent_K_tiles_in0;
    const uint32_t K_tiles = in1_parent_k_mode ? parent_K_tiles_in0 : parent_K_tiles_in1;
    const uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    // M extent of the input — independent of K offsets. parent_M is the input's stored M
    // dimension (matmul-M = stored inner when transpose_a, stored outer otherwise).
    const uint32_t parent_M = transpose_a ? in0_tensor_shape[-1] : in0_tensor_shape[-2];
    // Non-tile-aligned logical M is supported (like minimal_matmul / ttnn::matmul): TILE storage
    // already rounds up to TILE_HEIGHT, so we ceil-div to count the partial last tile; the
    // dataflow writer's per-tile bounds check clips writes back to the logical tile count.
    const uint32_t parent_M_tiles = tt::div_up(parent_M, tt::constants::TILE_HEIGHT);

    // expected_M_tiles overrides for offset-read mode; otherwise process the whole input.
    const uint32_t logical_M_tiles =
        (operation_attributes.expected_M_tiles > 0) ? operation_attributes.expected_M_tiles : parent_M_tiles;
    const uint32_t logical_M = logical_M_tiles * tt::constants::TILE_HEIGHT;

    // Pick the grid orientation from the matmul-M extent the caller actually uses: logical_M_tiles
    // is expected_M_tiles when the caller passes it (>0), else parent_M_tiles. For EP shared-tensor
    // callers (moe_ffn) parent_M = T_cap, so using logical_M_tiles gets the right orientation per
    // shape instead of being skewed by T_cap; stable as long as expected_M_tiles is stable.
    const bool transpose_core_grid = logical_M_tiles > N_tiles;

    const uint32_t M_block_tiles = config.M_block_size;
    const uint32_t K_block_tiles = config.K_block_size;
    const uint32_t N_block_tiles = config.N_block_size;
    const uint32_t subblock_h = config.subblock_h;
    const uint32_t subblock_w = config.subblock_w;

    const auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    const auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    const auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;

    const auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    const auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    const uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;

    const auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    const auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    const uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    // ----- M tile counts (from logical_M — used for both compile-time and runtime args) -----
    const uint32_t padded_M_tiles = tt::round_up(logical_M_tiles, in0_parallel_axis_cores);
    const uint32_t padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);

    const uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    const uint32_t N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;

    const uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    const uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    // ----- CB sizing (depends only on block sizes from config, not on total M) -----
    const uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    const uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t double_buffer_factor = 2;
    const uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    const uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    const uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    const uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered

    log_debug(tt::LogOp, "variable_matmul: logical_M={}, K={}, N={}", logical_M, K_w, N);
    log_debug(
        tt::LogOp, "variable_matmul: M_tiles_per_core={}, M_blocks_per_core={}", M_tiles_per_core, M_blocks_per_core);
    log_debug(tt::LogOp, "variable_matmul: transpose_core_grid={} (from logical_M > N)", transpose_core_grid);

    // ----- Sender/receiver core ranges -----
    const auto core_0_0 = CoreCoord{0, 0};
    const auto core_0_1 = CoreCoord{0, 1};
    const auto core_1_0 = CoreCoord{1, 0};
    const auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    const auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    const auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};

    const auto in0_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    const auto in0_receiver_cores = CoreRange(transpose_core_grid ? core_0_1 : core_1_0, core_endx_endy);
    const auto in1_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    const auto in1_receiver_cores = CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    // ----- Semaphores -----
    const auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    const auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    const auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    const auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    const auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    const auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    // ----- Circular buffers -----
    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(in0_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);

    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::create_cb(in1_cb_id, program, core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format);

    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, out_tile_size, out_cb_num_tiles, output_data_format);

    constexpr uint32_t intermediate_cb_id = tt::CBIndex::c_3;
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

    // Control CB used by both roles: a dm kernel publishes (M values for InputAndOutputRow,
    // K_tiles for InputAndWeightK) — derived from on-device offsets — and compute cb_wait_fronts
    // on it before overriding its RT-arg-derived values.
    // 4×uint32 page so both layouts (M values at [0..2], K at [3]) share one CB.
    constexpr uint32_t cb_ctrl_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ctrl_bytes = 16U;
    const bool cb_ctrl_active = operation_attributes.offsets_role == OffsetsRole::InputAndOutputRow ||
                                operation_attributes.offsets_role == OffsetsRole::InputAndWeightK;
    if (cb_ctrl_active) {
        tt::tt_metal::CircularBufferConfig cb_ctrl_cfg =
            tt::tt_metal::CircularBufferConfig(cb_ctrl_bytes, {{cb_ctrl_id, tt::DataFormat::UInt32}})
                .set_page_size(cb_ctrl_id, cb_ctrl_bytes);
        tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_ctrl_cfg);
    }

    // One offset mode per role — kernel branches key off these. OFFSET_ROW_MODE drives the
    // input-row + output-row offsetting and the per-core M re-derivation; OFFSET_K_MODE drives
    // the in0/in1 K-slice.
    const auto role = operation_attributes.offsets_role;
    const bool offset_row_mode = role == OffsetsRole::InputAndOutputRow;
    const bool offset_k_mode = role == OffsetsRole::InputAndWeightK;
    // `use_offset` / `use_offset_in1` — when true, the dm kernel adds the row/K offset to
    // the per-tile address. Computed once and shared across all four dm kernel CTA lists
    // (in0 sender + in0 receiver, in1 sender + in1 receiver). Sender and receiver MUST
    // agree on this flag — a mismatch makes one side read different runtime args than the other.
    // The role drives use_offset_in0: OFFSET_ROW_MODE needs the in0 row offset, OFFSET_K_MODE the
    // in0 K offset. NOT expected_M_tiles (that's only the transpose-orientation hint) — keying on
    // it would drop the row offset whenever a caller left expected_M_tiles at its 0 default.
    // Parent-K mode also needs the parent stride; expected_M_tiles>0 is kept as a legacy signal.
    const bool use_offset_in0 =
        offset_row_mode || offset_k_mode || operation_attributes.expected_M_tiles > 0 || parent_K_tiles_in0 > K_tiles;
    const bool use_offset_in1 = parent_K_tiles_in1 > K_tiles || offset_k_mode;
    // Both dm kernels always read the offsets tensor (each one derives its own slice).
    // Compute reads its override values from cb_ctrl, not directly from offsets.
    constexpr bool in0_needs_offsets = true;
    constexpr bool in1_needs_offsets = true;
    std::map<std::string, std::string> in0_defines;
    std::map<std::string, std::string> in1_defines;
    std::map<std::string, std::string> compute_offsets_defines;  // merged into compute_defines below
    auto set_flag = [](std::map<std::string, std::string>& m, const char* name, bool active) {
        if (active) {
            m[name] = "1";
        }
    };
    for (auto* m : {&in0_defines, &in1_defines, &compute_offsets_defines}) {
        set_flag(*m, "OFFSET_ROW_MODE", offset_row_mode);
        set_flag(*m, "OFFSET_K_MODE", offset_k_mode);
    }
    // CT args used by the row-mode per-core M computation. Compile-time constant so the
    // kernel can specialize.
    const uint32_t in0_axis_cores_ct = transpose_core_grid ? grid_size.x : grid_size.y;
    if (offset_row_mode) {
        in0_defines["IN0_AXIS_CORES"] = std::to_string(in0_axis_cores_ct);
        in1_defines["IN0_AXIS_CORES"] = std::to_string(in0_axis_cores_ct);
        compute_offsets_defines["IN0_AXIS_CORES"] = std::to_string(in0_axis_cores_ct);
    }
    // Y_AXIS_CORES: number of cores along the Y direction. Used by dm kernels to compute
    // `defer_write_k_block = core_y * ceil(K_num_blocks / Y_AXIS_CORES)` from runtime
    // K_num_blocks (which can shrink when a K-axis OffsetsRole overrides K_tiles from
    // on-device offsets). The host can't know runtime K, so it passes core.y instead of a
    // pre-computed defer value; the kernel runs the stagger formula after the K override.
    in0_defines["Y_AXIS_CORES"] = std::to_string(grid_size.y);
    in1_defines["Y_AXIS_CORES"] = std::to_string(grid_size.y);

    // ----- Kernel compile-time args -----
    const bool in0_is_output_writer = !transpose_core_grid;
    const bool in1_is_output_writer = transpose_core_grid;

    // in0 sender compile-time args (16 scalar + tensor accessor args)
    std::vector<uint32_t> in0_sender_compile_time_args = {
        N_tiles,                                                       // 0:  N_tiles
        padded_N_tiles,                                                // 1:  padded_N_tiles
        M_block_tiles,                                                 // 2:  M_block_tiles
        K_block_tiles,                                                 // 3:  K_block_tiles
        N_block_tiles,                                                 // 4:  N_block_tiles
        N_blocks_per_core,                                             // 5:  N_blocks_per_core
        in0_tile_size,                                                 // 6:  in_tile_size
        out_tile_size,                                                 // 7:  out_tile_size
        in0_sender_semaphore_id,                                       // 8:  sender_sem_id
        in0_receiver_semaphore_id,                                     // 9:  receiver_sem_id
        in0_valid_semaphore_id,                                        // 10: valid_sem_id
        in0_is_output_writer,                                          // 11: is_output_writer
        true,                                                          // 12: is_injector_core
        static_cast<uint32_t>(transpose_a),                            // 13: transpose_a
        static_cast<uint32_t>(use_offset_in0),                         // 14: use_offset
        static_cast<uint32_t>(tensor_args.output_tensor.has_value()),  // 15: use_out_offset
    };
    append_accessors(in0_sender_compile_time_args, input_tensor, output_tensor);
    if (in0_needs_offsets) {
        tt::tt_metal::TensorAccessorArgs(*tensor_args.offsets_tensor.buffer()).append_to(in0_sender_compile_time_args);
    }

    const auto in0_sender_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in0_sender.cpp",
        in0_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = in0_defines});

    // in0 receiver compile-time args (same layout, is_injector_core=false)
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        N_tiles,                                                       // 0:  N_tiles
        padded_N_tiles,                                                // 1:  padded_N_tiles
        M_block_tiles,                                                 // 2:  M_block_tiles
        K_block_tiles,                                                 // 3:  K_block_tiles
        N_block_tiles,                                                 // 4:  N_block_tiles
        N_blocks_per_core,                                             // 5:  N_blocks_per_core
        in0_tile_size,                                                 // 6:  in_tile_size
        out_tile_size,                                                 // 7:  out_tile_size
        in0_sender_semaphore_id,                                       // 8:  sender_sem_id
        in0_receiver_semaphore_id,                                     // 9:  receiver_sem_id
        in0_valid_semaphore_id,                                        // 10: valid_sem_id
        in0_is_output_writer,                                          // 11: is_output_writer
        false,                                                         // 12: is_injector_core
        static_cast<uint32_t>(transpose_a),                            // 13: transpose_a
        static_cast<uint32_t>(use_offset_in0),                         // 14: use_offset
        static_cast<uint32_t>(tensor_args.output_tensor.has_value()),  // 15: use_out_offset
    };
    append_accessors(in0_receiver_compile_time_args, input_tensor, output_tensor);
    if (in0_needs_offsets) {
        tt::tt_metal::TensorAccessorArgs(*tensor_args.offsets_tensor.buffer())
            .append_to(in0_receiver_compile_time_args);
    }

    const auto in0_receiver_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in0_sender.cpp",
        in0_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_receiver_compile_time_args,
            .defines = in0_defines});

    // in1 sender compile-time args (16 scalar + tensor accessor args)
    std::vector<uint32_t> in1_sender_compile_time_args = {
        N_tiles,                                                       // 0:  N_tiles
        padded_N_tiles,                                                // 1:  padded_N_tiles
        M_block_tiles,                                                 // 2:  M_block_tiles
        K_block_tiles,                                                 // 3:  K_block_tiles
        N_block_tiles,                                                 // 4:  N_block_tiles
        N_blocks_per_core,                                             // 5:  N_blocks_per_core
        in1_tile_size,                                                 // 6:  in1_tile_size
        out_tile_size,                                                 // 7:  out_tile_size
        in1_sender_semaphore_id,                                       // 8:  sender_sem_id
        in1_receiver_semaphore_id,                                     // 9:  receiver_sem_id
        in1_valid_semaphore_id,                                        // 10: valid_sem_id
        in1_is_output_writer,                                          // 11: is_output_writer
        true,                                                          // 12: is_injector_core
        static_cast<uint32_t>(transpose_b),                            // 13: transpose_b
        static_cast<uint32_t>(use_offset_in1),                         // 14: use_offset_in1
        static_cast<uint32_t>(tensor_args.output_tensor.has_value()),  // 15: use_out_offset
    };
    append_accessors(in1_sender_compile_time_args, weight_tensor, output_tensor);
    if (in1_needs_offsets) {
        tt::tt_metal::TensorAccessorArgs(*tensor_args.offsets_tensor.buffer()).append_to(in1_sender_compile_time_args);
    }

    const auto in1_sender_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in1_sender_out.cpp",
        in1_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .compile_args = in1_sender_compile_time_args,
            .defines = in1_defines});

    // in1 receiver compile-time args
    std::vector<uint32_t> in1_receiver_compile_time_args = {
        N_tiles,                                                       // 0:  N_tiles
        padded_N_tiles,                                                // 1:  padded_N_tiles
        M_block_tiles,                                                 // 2:  M_block_tiles
        K_block_tiles,                                                 // 3:  K_block_tiles
        N_block_tiles,                                                 // 4:  N_block_tiles
        N_blocks_per_core,                                             // 5:  N_blocks_per_core
        in1_tile_size,                                                 // 6:  in1_tile_size
        out_tile_size,                                                 // 7:  out_tile_size
        in1_sender_semaphore_id,                                       // 8:  sender_sem_id
        in1_receiver_semaphore_id,                                     // 9:  receiver_sem_id
        in1_valid_semaphore_id,                                        // 10: valid_sem_id
        in1_is_output_writer,                                          // 11: is_output_writer
        false,                                                         // 12: is_injector_core
        static_cast<uint32_t>(transpose_b),                            // 13: transpose_b
        static_cast<uint32_t>(use_offset_in1),                         // 14: use_offset_in1
        static_cast<uint32_t>(tensor_args.output_tensor.has_value()),  // 15: use_out_offset
    };
    append_accessors(in1_receiver_compile_time_args, weight_tensor, output_tensor);
    if (in1_needs_offsets) {
        tt::tt_metal::TensorAccessorArgs(*tensor_args.offsets_tensor.buffer())
            .append_to(in1_receiver_compile_time_args);
    }

    const auto in1_receiver_kernels_id = CreateKernel(
        program,
        "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in1_sender_out.cpp",
        in1_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .compile_args = in1_receiver_compile_time_args,
            .defines = in1_defines});

    // ----- Compute kernel -----
    std::vector<uint32_t> compute_compile_time_args = {
        M_block_tiles,                       // 0: M_block_tiles
        K_block_tiles,                       // 1: K_block_tiles
        N_block_tiles,                       // 2: N_block_tiles
        N_blocks_per_core,                   // 3: N_blocks_per_core
        subblock_h,                          // 4: subblock_h
        subblock_w,                          // 5: subblock_w
        static_cast<uint32_t>(transpose_b),  // 6: transpose_b
        static_cast<uint32_t>(transpose_a),  // 7: transpose_a
    };

    std::map<std::string, std::string> compute_defines;
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        compute_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));
    for (const auto& kv : compute_offsets_defines) {
        compute_defines[kv.first] = kv.second;
    }

    const auto compute_kernels_id = CreateKernel(
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
    // defer_write_k_block stagger is computed kernel-side from runtime K_num_blocks +
    // Y_AXIS_CORES (CT). Host just hands the kernel core.y here.

    const auto cores = corerange_to_cores(core_grid, num_cores, true);

    const uint32_t in0_addr = input_tensor.buffer()->address();
    const uint32_t in1_addr = weight_tensor.buffer()->address();
    const uint32_t out_addr = output_tensor.buffer()->address();

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        const CoreCoord core = cores.at(core_id);
        const uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        const uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        const CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        const CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

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

        const auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
        const auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
        const auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
        const auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

        const auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        const auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        const auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        const auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

        // M tile ranges use logical_M (runtime), N tile ranges use padded_N (compile-time)
        const uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        const uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);
        const uint32_t N_start_tile = N_tiles_per_core * in1_idx;
        const uint32_t N_end_tile = N_tiles_per_core * (in1_idx + 1);

        // Pass core.y to the kernel; it computes defer_write_k_block from runtime K_num_blocks
        // and Y_AXIS_CORES so the stagger is correct even when on-device offsets shrink K.
        const uint32_t defer_write_k_block = core.y;

        const bool is_in0_sink = core == in0_core_order.back();
        const bool is_in1_sink = core == in1_core_order.back();

        // in0 runtime args layout:
        //  0: in0_addr
        //  1: is_sink_core
        //  2: in0_dest_noc_x
        //  3: in0_dest_noc_y
        //  4: in0_sender_noc_x
        //  5: in0_sender_noc_y
        //  6: M_start_tile     (in effective-M coord, per-core partition)
        //  7: M_end_tile
        //  8: N_start_tile
        //  9: N_end_tile
        // 10: defer_write_k_block
        // 11: out_addr
        // 12: logical_M_tiles            (= expected_M_tiles)
        // 13: padded_M_tiles
        // 14: M_blocks_per_core
        // 15: parent_M_tiles_stride     (parent M tile count; used as K-row stride for transpose_a)
        // 16: parent_K_tiles_stride     (parent K tile count; used as M-row stride for non-transpose)
        // 17: K_tiles                   (variable-K: matmul-K extent in tiles)
        std::vector<uint32_t> in0_args = {
            in0_addr,
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
            logical_M_tiles,
            padded_M_tiles,
            M_blocks_per_core,
            parent_M_tiles,
            parent_K_tiles_in0,
            K_tiles,
        };

        // On-device offsets RT args for dm_in0_sender:
        //   InputAndOutputRow: (offsets_addr, offsets_start_index, in0_idx) — kernel reads the
        //                      M range and derives per-core M from in0_idx.
        //   InputAndWeightK:   (offsets_addr, offsets_start_index)          — K-range only.
        if (in0_needs_offsets) {
            in0_args.push_back(tensor_args.offsets_tensor.buffer()->address());
            in0_args.push_back(operation_attributes.offsets_start_index);
            if (offset_row_mode) {
                in0_args.push_back(in0_idx);
            }
        }

        if (in1_idx == 0) {
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else {
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
        }

        // in1 runtime args layout:
        //  0: in1_addr
        //  1: is_sink_core
        //  2: dest_noc_x
        //  3: dest_noc_y
        //  4: sender_noc_x
        //  5: sender_noc_y
        //  6: M_start_tile
        //  7: M_end_tile
        //  8: N_start_tile
        //  9: N_end_tile
        // 10: defer_write_k_block
        // 11: out_addr
        // 12: logical_M_tiles
        // 13: padded_M_tiles
        // 14: M_blocks_per_core
        // 15: parent_K_tiles_in1    (parent K tile count; row stride for transpose_b)
        // 16: K_tiles               (variable-K: matmul-K extent in tiles)
        std::vector<uint32_t> in1_args = {
            in1_addr,
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
            logical_M_tiles,
            padded_M_tiles,
            M_blocks_per_core,
            parent_K_tiles_in1,
            K_tiles,
        };

        // On-device offsets RT args for dm_in1_sender_out:
        //   InputAndWeightK:   (offsets_addr, offsets_start_index) — K-range only.
        //   InputAndOutputRow: also append in0_idx (kernel derives per-core M).
        if (in1_needs_offsets) {
            in1_args.push_back(tensor_args.offsets_tensor.buffer()->address());
            in1_args.push_back(operation_attributes.offsets_start_index);
            if (offset_row_mode) {
                in1_args.push_back(in0_idx);
            }
        }

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
        //  4: M_blocks_per_core
        //  5: K_tiles (variable-K)
        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            M_blocks_per_core,
            K_tiles,
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
            .K_block_tiles = K_block_tiles,
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
    // processes expected_M_tiles when set (offset-read mode), else the full parent.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& a_padded = input_tensor.padded_shape();
    const auto& w_padded = tensor_args.weight_tensor.padded_shape();
    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;
    const uint32_t parent_M_tiles = (transpose_a ? a_padded[-1] : a_padded[-2]) / tt::constants::TILE_HEIGHT;
    const uint32_t parent_K_tiles_in0 = (transpose_a ? a_padded[-2] : a_padded[-1]) / tt::constants::TILE_WIDTH;
    const uint32_t parent_K_tiles_in1 = (transpose_b ? w_padded[-1] : w_padded[-2]) / tt::constants::TILE_WIDTH;
    const uint32_t logical_M_tiles =
        (operation_attributes.expected_M_tiles > 0) ? operation_attributes.expected_M_tiles : parent_M_tiles;
    const uint32_t padded_M_tiles = tt::round_up(logical_M_tiles, sv.in0_parallel_axis_cores);
    const uint32_t M_tiles_per_core = padded_M_tiles / sv.in0_parallel_axis_cores;
    const uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, sv.M_block_tiles);

    const uint32_t in0_addr = input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = output_tensor.buffer()->address();

    auto& in0_sender_rt = GetRuntimeArgs(program, sv.in0_sender_kernels_id);
    auto& in0_receiver_rt = GetRuntimeArgs(program, sv.in0_receiver_kernels_id);
    auto& in1_sender_rt = GetRuntimeArgs(program, sv.in1_sender_kernels_id);
    auto& in1_receiver_rt = GetRuntimeArgs(program, sv.in1_receiver_kernels_id);
    auto& compute_rt = GetRuntimeArgs(program, sv.compute_kernels_id);

    // in0 runtime arg indices
    constexpr uint32_t IN0_ADDR_IDX = 0;
    constexpr uint32_t IN0_M_START_IDX = 6;
    constexpr uint32_t IN0_M_END_IDX = 7;
    constexpr uint32_t IN0_OUT_ADDR_IDX = 11;
    constexpr uint32_t IN0_LOGICAL_M_TILES_IDX = 12;
    constexpr uint32_t IN0_PADDED_M_TILES_IDX = 13;
    constexpr uint32_t IN0_M_BLOCKS_IDX = 14;
    constexpr uint32_t IN0_PARENT_M_TILES_STRIDE_IDX = 15;
    constexpr uint32_t IN0_PARENT_K_TILES_STRIDE_IDX = 16;
    constexpr uint32_t IN0_K_TILES_IDX = 17;

    // in1 runtime arg indices
    constexpr uint32_t IN1_ADDR_IDX = 0;
    constexpr uint32_t IN1_M_START_IDX = 6;
    constexpr uint32_t IN1_M_END_IDX = 7;
    constexpr uint32_t IN1_OUT_ADDR_IDX = 11;
    constexpr uint32_t IN1_LOGICAL_M_TILES_IDX = 12;
    constexpr uint32_t IN1_PADDED_M_TILES_IDX = 13;
    constexpr uint32_t IN1_M_BLOCKS_IDX = 14;
    constexpr uint32_t IN1_PARENT_K_TILES_STRIDE_IDX = 15;
    constexpr uint32_t IN1_K_TILES_IDX = 16;

    // Compute runtime arg indices
    constexpr uint32_t COMPUTE_M_START_IDX = 0;
    constexpr uint32_t COMPUTE_M_END_IDX = 1;
    constexpr uint32_t COMPUTE_M_BLOCKS_IDX = 4;
    constexpr uint32_t COMPUTE_K_TILES_IDX = 5;

    // Recompute matmul-K. Mirror the create() derivation: in1 is parent when its K extent
    // exceeds in0's; otherwise in0/weight K match (weight provides K_w).
    const bool in1_parent_k = parent_K_tiles_in1 > parent_K_tiles_in0;
    const uint32_t K_tiles_rt = in1_parent_k ? parent_K_tiles_in0 : parent_K_tiles_in1;

    // defer_write_k_block is computed kernel-side from runtime K_num_blocks + Y_AXIS_CORES.
    // Host just passes core.y at that RT-arg slot.
    constexpr uint32_t IN0_DEFER_WRITE_K_BLOCK_IDX = 10;
    constexpr uint32_t IN1_DEFER_WRITE_K_BLOCK_IDX = 10;

    // On-device offsets RT-arg indices (appended after K_tiles on each kernel).
    // Must mirror init-path: both in0 and in1 kernels are compiled with the offset defines for
    // every role, so RT-arg updates must match — otherwise the kernel's `offsets_start_index`
    // arg keeps the value from the first build and subsequent cache-hit invocations read
    // offsets[0] instead of offsets[e].
    // offsets_tensor is always set.
    constexpr bool in0_needs_offsets = true;
    constexpr bool in1_needs_offsets = true;
    const uint32_t offsets_addr = tensor_args.offsets_tensor.buffer()->address();
    constexpr uint32_t IN0_OFFSETS_ADDR_IDX = 18;  // appended after K_tiles (idx 17).
    constexpr uint32_t IN0_OFFSETS_START_IDX_IDX = 19;
    constexpr uint32_t IN1_OFFSETS_ADDR_IDX = 17;
    constexpr uint32_t IN1_OFFSETS_START_IDX_IDX = 18;

    // Sender and receiver kernels share the same RT-arg layout. Only the sender actually
    // reads in0_addr / in1_addr (the receiver gets data via NOC handshake from the sender),
    // but writing the addr to both keeps the slot consistent.
    auto update_in0 = [&](auto& args, uint32_t M_start, uint32_t M_end, uint32_t defer_write_k_block) {
        args[IN0_ADDR_IDX] = in0_addr;
        args[IN0_M_START_IDX] = M_start;
        args[IN0_M_END_IDX] = M_end;
        args[IN0_OUT_ADDR_IDX] = out_addr;
        args[IN0_LOGICAL_M_TILES_IDX] = logical_M_tiles;
        args[IN0_PADDED_M_TILES_IDX] = padded_M_tiles;
        args[IN0_M_BLOCKS_IDX] = M_blocks_per_core;
        args[IN0_PARENT_M_TILES_STRIDE_IDX] = parent_M_tiles;
        args[IN0_PARENT_K_TILES_STRIDE_IDX] = parent_K_tiles_in0;
        args[IN0_K_TILES_IDX] = K_tiles_rt;
        args[IN0_DEFER_WRITE_K_BLOCK_IDX] = defer_write_k_block;
        if (in0_needs_offsets) {
            args[IN0_OFFSETS_ADDR_IDX] = offsets_addr;
            args[IN0_OFFSETS_START_IDX_IDX] = operation_attributes.offsets_start_index;
        }
    };
    auto update_in1 = [&](auto& args, uint32_t M_start, uint32_t M_end, uint32_t defer_write_k_block) {
        args[IN1_ADDR_IDX] = in1_addr;
        args[IN1_M_START_IDX] = M_start;
        args[IN1_M_END_IDX] = M_end;
        args[IN1_OUT_ADDR_IDX] = out_addr;
        args[IN1_LOGICAL_M_TILES_IDX] = logical_M_tiles;
        args[IN1_PADDED_M_TILES_IDX] = padded_M_tiles;
        args[IN1_M_BLOCKS_IDX] = M_blocks_per_core;
        args[IN1_PARENT_K_TILES_STRIDE_IDX] = parent_K_tiles_in1;
        args[IN1_K_TILES_IDX] = K_tiles_rt;
        args[IN1_DEFER_WRITE_K_BLOCK_IDX] = defer_write_k_block;
        if (in1_needs_offsets) {
            args[IN1_OFFSETS_ADDR_IDX] = offsets_addr;
            args[IN1_OFFSETS_START_IDX_IDX] = operation_attributes.offsets_start_index;
        }
    };

    for (uint32_t i = 0; i < sv.num_cores; ++i) {
        const CoreCoord core = sv.cores.at(i);
        const uint32_t in0_idx = sv.transpose_core_grid ? core.x : core.y;
        const uint32_t in1_idx = sv.transpose_core_grid ? core.y : core.x;

        const uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        const uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);

        // Kernel computes defer_write_k_block from runtime K_num_blocks (post K-axis offset
        // override) using Y_AXIS_CORES — just pass core.y.
        const uint32_t defer_write_k_block = core.y;

        update_in0(
            in1_idx == 0 ? in0_sender_rt[core.x][core.y] : in0_receiver_rt[core.x][core.y],
            M_start_tile,
            M_end_tile,
            defer_write_k_block);
        update_in1(
            in0_idx == 0 ? in1_sender_rt[core.x][core.y] : in1_receiver_rt[core.x][core.y],
            M_start_tile,
            M_end_tile,
            defer_write_k_block);

        // Update compute args
        auto& cargs = compute_rt[core.x][core.y];
        cargs[COMPUTE_M_START_IDX] = M_start_tile;
        cargs[COMPUTE_M_END_IDX] = M_end_tile;
        cargs[COMPUTE_M_BLOCKS_IDX] = M_blocks_per_core;
        cargs[COMPUTE_K_TILES_IDX] = K_tiles_rt;
    }
}

}  // namespace ttml::metal::ops::variable_matmul::device
