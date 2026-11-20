// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "variable_matmul_program_factory.hpp"

#include <map>
#include <optional>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "variable_matmul_device_operation.hpp"

namespace ttml::metal::ops::variable_matmul::device {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using ttnn::Tensor;

TTML_ASSERT_DESCRIPTOR_OVERRIDE(VariableMatmulDeviceOperation, VariableMatmulProgramFactory);

namespace {

constexpr auto kIn0KernelPath =
    "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in0_sender.cpp";
constexpr auto kIn1KernelPath =
    "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/dataflow/dm_in1_sender_out.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/variable_matmul/device/kernels/compute/compute.cpp";

// Kernel handles in the built Program: the push order of ProgramDescriptor::kernels. The grid is at least 2x2, so the
// sender line of each input and the receivers behind it all have cores.
constexpr uint32_t kIn0SenderKernel = 0U;
constexpr uint32_t kIn0ReceiverKernel = 1U;
constexpr uint32_t kIn1SenderKernel = 2U;
constexpr uint32_t kIn1ReceiverKernel = 3U;
constexpr uint32_t kComputeKernel = 4U;

// Semaphore ids, in the order legacy CreateSemaphore handed them out; the kernels get them as compile-time args.
constexpr uint32_t kIn0SenderSemId = 0U;
constexpr uint32_t kIn0ReceiverSemId = 1U;
constexpr uint32_t kIn0ValidSemId = 2U;
constexpr uint32_t kIn1SenderSemId = 3U;
constexpr uint32_t kIn1ReceiverSemId = 4U;
constexpr uint32_t kIn1ValidSemId = 5U;

// Runtime arg slots that override_runtime_arguments() rewrites on every cache hit: the buffer addresses,
// offsets_start_index and everything derived from M and matmul-K. The other slots (NoC neighbours, sink flags, N
// ranges, defer_write_k_block, the in0 walk index) follow from the grid, its orientation and N, which the program hash
// keys on.
constexpr uint32_t kIn0AddrIdx = 0U;
constexpr uint32_t kIn0MStartIdx = 6U;
constexpr uint32_t kIn0MEndIdx = 7U;
constexpr uint32_t kIn0OutAddrIdx = 11U;
constexpr uint32_t kIn0LogicalMTilesIdx = 12U;
constexpr uint32_t kIn0PaddedMTilesIdx = 13U;
constexpr uint32_t kIn0MBlocksIdx = 14U;
constexpr uint32_t kIn0ParentMTilesIdx = 15U;
constexpr uint32_t kIn0ParentKTilesIdx = 16U;
constexpr uint32_t kIn0KTilesIdx = 17U;
constexpr uint32_t kIn0OffsetsAddrIdx = 18U;
constexpr uint32_t kIn0OffsetsStartIdx = 19U;

constexpr uint32_t kIn1AddrIdx = 0U;
constexpr uint32_t kIn1MStartIdx = 6U;
constexpr uint32_t kIn1MEndIdx = 7U;
constexpr uint32_t kIn1OutAddrIdx = 11U;
constexpr uint32_t kIn1LogicalMTilesIdx = 12U;
constexpr uint32_t kIn1PaddedMTilesIdx = 13U;
constexpr uint32_t kIn1MBlocksIdx = 14U;
constexpr uint32_t kIn1ParentKTilesIdx = 15U;
constexpr uint32_t kIn1KTilesIdx = 16U;
constexpr uint32_t kIn1OffsetsAddrIdx = 17U;
constexpr uint32_t kIn1OffsetsStartIdx = 18U;

constexpr uint32_t kComputeMStartIdx = 0U;
constexpr uint32_t kComputeMEndIdx = 1U;
constexpr uint32_t kComputeMBlocksIdx = 4U;
constexpr uint32_t kComputeKTilesIdx = 5U;

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

// The matmul's extents in tiles and how they split over the grid. create_descriptor() and override_runtime_arguments()
// both take them from here, so a cache hit cannot drift from the build.
struct MatmulSplit {
    // From the grid, N and the grid orientation, which the program hash keys on: fixed for a cached program.
    bool transpose_core_grid = false;
    uint32_t in0_parallel_axis_cores = 0U;
    uint32_t in1_parallel_axis_cores = 0U;
    uint32_t N_tiles = 0U;
    uint32_t padded_N_tiles = 0U;
    uint32_t N_tiles_per_core = 0U;
    uint32_t N_blocks_per_core = 0U;
    // From M and matmul-K, which the program hash leaves out: re-derived on every launch.
    uint32_t parent_M_tiles = 0U;
    uint32_t parent_K_tiles_in0 = 0U;
    uint32_t parent_K_tiles_in1 = 0U;
    uint32_t K_tiles = 0U;
    uint32_t logical_M_tiles = 0U;
    uint32_t padded_M_tiles = 0U;
    uint32_t M_tiles_per_core = 0U;
    uint32_t M_blocks_per_core = 0U;
};

MatmulSplit compute_matmul_split(
    const VariableMatmulParams& operation_attributes, const VariableMatmulInputs& tensor_args) {
    const auto& config = operation_attributes.config;
    const auto in0_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto in1_tensor_shape = tensor_args.weight_tensor.padded_shape();
    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;

    MatmulSplit split;
    // Matmul-K source depends on which side is the parent buffer:
    //   - K_w > K_in: weight is the parent, matmul-K = K_in
    //   - else: in0 may be the parent (or both match), matmul-K = K_w
    const uint32_t K_w = transpose_b ? in1_tensor_shape[-1] : in1_tensor_shape[-2];
    const uint32_t N = transpose_b ? in1_tensor_shape[-2] : in1_tensor_shape[-1];
    const uint32_t K_in = transpose_a ? in0_tensor_shape[-2] : in0_tensor_shape[-1];
    split.parent_K_tiles_in0 = K_in / tt::constants::TILE_WIDTH;
    split.parent_K_tiles_in1 = K_w / tt::constants::TILE_WIDTH;
    const bool in1_parent_k_mode = split.parent_K_tiles_in1 > split.parent_K_tiles_in0;
    split.K_tiles = in1_parent_k_mode ? split.parent_K_tiles_in0 : split.parent_K_tiles_in1;
    split.N_tiles = N / tt::constants::TILE_WIDTH;

    // M extent of the input — independent of K offsets. parent_M is the input's stored M
    // dimension (matmul-M = stored inner when transpose_a, stored outer otherwise).
    const uint32_t parent_M = transpose_a ? in0_tensor_shape[-1] : in0_tensor_shape[-2];
    // Non-tile-aligned logical M is supported (like minimal_matmul / ttnn::matmul): TILE storage
    // already rounds up to TILE_HEIGHT, so we ceil-div to count the partial last tile; the
    // dataflow writer's per-tile bounds check clips writes back to the logical tile count.
    split.parent_M_tiles = tt::div_up(parent_M, tt::constants::TILE_HEIGHT);
    // expected_M_tiles overrides for offset-read mode; otherwise process the whole input.
    split.logical_M_tiles =
        (operation_attributes.expected_M_tiles > 0) ? operation_attributes.expected_M_tiles : split.parent_M_tiles;

    // Pick the grid orientation from the matmul-M extent the caller actually uses: logical_M_tiles
    // is expected_M_tiles when the caller passes it (>0), else parent_M_tiles. For EP shared-tensor
    // callers (moe_ffn) parent_M = T_cap, so using logical_M_tiles gets the right orientation per
    // shape instead of being skewed by T_cap; stable as long as expected_M_tiles is stable.
    split.transpose_core_grid = split.logical_M_tiles > split.N_tiles;
    const auto grid_size = config.compute_with_storage_grid_size;
    split.in0_parallel_axis_cores = static_cast<uint32_t>(split.transpose_core_grid ? grid_size.x : grid_size.y);
    split.in1_parallel_axis_cores = static_cast<uint32_t>(split.transpose_core_grid ? grid_size.y : grid_size.x);

    split.padded_M_tiles = tt::round_up(split.logical_M_tiles, split.in0_parallel_axis_cores);
    split.padded_N_tiles = tt::round_up(split.N_tiles, split.in1_parallel_axis_cores);
    split.M_tiles_per_core = split.padded_M_tiles / split.in0_parallel_axis_cores;
    split.N_tiles_per_core = split.padded_N_tiles / split.in1_parallel_axis_cores;
    split.M_blocks_per_core = tt::div_up(split.M_tiles_per_core, config.M_block_size);
    split.N_blocks_per_core = tt::div_up(split.N_tiles_per_core, config.N_block_size);
    return split;
}

// A core's slice along each input's parallel axis: in0_idx picks its M tiles, in1_idx its N tiles. A core with
// in1_idx == 0 is on the in0 sender line, one with in0_idx == 0 on the in1 sender line.
struct CoreAxes {
    uint32_t in0_idx = 0U;
    uint32_t in1_idx = 0U;
};

CoreAxes core_axes(const CoreCoord& core, bool transpose_core_grid) {
    return CoreAxes{
        .in0_idx = static_cast<uint32_t>(transpose_core_grid ? core.x : core.y),
        .in1_idx = static_cast<uint32_t>(transpose_core_grid ? core.y : core.x),
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor VariableMatmulProgramFactory::create_descriptor(
    const VariableMatmulParams& operation_attributes,
    const VariableMatmulInputs& tensor_args,
    ttnn::Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& config = operation_attributes.config;
    auto* device = input_tensor.device();

    tt::tt_metal::ProgramDescriptor desc;

    const auto grid_size = config.compute_with_storage_grid_size;
    const auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const tt::tt_metal::CoreRangeSet core_grid_set(core_grid);
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

    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;
    const MatmulSplit split = compute_matmul_split(operation_attributes, tensor_args);
    const bool transpose_core_grid = split.transpose_core_grid;

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
    const auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    const auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;

    // ----- CB sizing (depends only on block sizes from config, not on total M) -----
    const uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    const uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t double_buffer_factor = 2;
    const uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    const uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    const uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    const uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered

    log_debug(
        tt::LogOp,
        "variable_matmul: logical_M_tiles={}, K_tiles={}, N_tiles={}",
        split.logical_M_tiles,
        split.K_tiles,
        split.N_tiles);
    log_debug(
        tt::LogOp,
        "variable_matmul: M_tiles_per_core={}, M_blocks_per_core={}",
        split.M_tiles_per_core,
        split.M_blocks_per_core);
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
    desc.semaphores.push_back(make_semaphore_descriptor(kIn0SenderSemId, core_grid_set, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kIn0ReceiverSemId, core_grid_set, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kIn0ValidSemId, core_grid_set, VALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kIn1SenderSemId, core_grid_set, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kIn1ReceiverSemId, core_grid_set, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kIn1ValidSemId, core_grid_set, VALID));

    // ----- Circular buffers -----
    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    desc.cbs.push_back(make_cb_descriptor(core_grid_set, in0_cb_id, in0_data_format, in0_tile_size, in0_cb_num_tiles));

    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    desc.cbs.push_back(make_cb_descriptor(core_grid_set, in1_cb_id, in1_data_format, in1_tile_size, in1_cb_num_tiles));

    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    desc.cbs.push_back(
        make_cb_descriptor(core_grid_set, out_cb_id, output_data_format, out_tile_size, out_cb_num_tiles));

    constexpr uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    desc.cbs.push_back(make_cb_descriptor(
        core_grid_set, intermediate_cb_id, intermediate_data_format, intermediate_tile_size, interm_cb_num_tiles));

    // CB for the transposed in0 block (only used when transpose_a is true). Same data format
    // and capacity as in0 since the transpose preserves both. Compute kernel transposes
    // c_0 -> c_7 tile-by-tile, then matmul consumes c_7.
    if (transpose_a) {
        constexpr uint32_t in0_transposed_cb_id = tt::CBIndex::c_7;
        desc.cbs.push_back(
            make_cb_descriptor(core_grid_set, in0_transposed_cb_id, in0_data_format, in0_tile_size, in0_cb_num_tiles));
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
        desc.cbs.push_back(make_cb_descriptor_bytes(core_grid_set, cb_ctrl_id, tt::DataFormat::UInt32, cb_ctrl_bytes));
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
    const bool use_offset_in0 = offset_row_mode || offset_k_mode || operation_attributes.expected_M_tiles > 0 ||
                                split.parent_K_tiles_in0 > split.K_tiles;
    const bool use_offset_in1 = split.parent_K_tiles_in1 > split.K_tiles || offset_k_mode;
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
    const uint32_t in0_axis_cores_ct = split.in0_parallel_axis_cores;
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
    const auto use_out_offset = static_cast<uint32_t>(tensor_args.output_tensor.has_value());

    // in0 sender and receiver compile-time args (16 scalar + tensor accessor args); only is_injector_core differs.
    auto in0_compile_time_args = [&](bool is_injector_core) {
        std::vector<uint32_t> args = {
            split.N_tiles,                          // 0:  N_tiles
            split.padded_N_tiles,                   // 1:  padded_N_tiles
            M_block_tiles,                          // 2:  M_block_tiles
            K_block_tiles,                          // 3:  K_block_tiles
            N_block_tiles,                          // 4:  N_block_tiles
            split.N_blocks_per_core,                // 5:  N_blocks_per_core
            in0_tile_size,                          // 6:  in_tile_size
            out_tile_size,                          // 7:  out_tile_size
            kIn0SenderSemId,                        // 8:  sender_sem_id
            kIn0ReceiverSemId,                      // 9:  receiver_sem_id
            kIn0ValidSemId,                         // 10: valid_sem_id
            in0_is_output_writer,                   // 11: is_output_writer
            is_injector_core,                       // 12: is_injector_core
            static_cast<uint32_t>(transpose_a),     // 13: transpose_a
            static_cast<uint32_t>(use_offset_in0),  // 14: use_offset
            use_out_offset,                         // 15: use_out_offset
        };
        append_accessors(args, input_tensor, output_tensor);
        if (in0_needs_offsets) {
            tt::tt_metal::TensorAccessorArgs(*tensor_args.offsets_tensor.buffer()).append_to(args);
        }
        return args;
    };

    // in1 sender and receiver compile-time args (16 scalar + tensor accessor args); only is_injector_core differs.
    auto in1_compile_time_args = [&](bool is_injector_core) {
        std::vector<uint32_t> args = {
            split.N_tiles,                          // 0:  N_tiles
            split.padded_N_tiles,                   // 1:  padded_N_tiles
            M_block_tiles,                          // 2:  M_block_tiles
            K_block_tiles,                          // 3:  K_block_tiles
            N_block_tiles,                          // 4:  N_block_tiles
            split.N_blocks_per_core,                // 5:  N_blocks_per_core
            in1_tile_size,                          // 6:  in1_tile_size
            out_tile_size,                          // 7:  out_tile_size
            kIn1SenderSemId,                        // 8:  sender_sem_id
            kIn1ReceiverSemId,                      // 9:  receiver_sem_id
            kIn1ValidSemId,                         // 10: valid_sem_id
            in1_is_output_writer,                   // 11: is_output_writer
            is_injector_core,                       // 12: is_injector_core
            static_cast<uint32_t>(transpose_b),     // 13: transpose_b
            static_cast<uint32_t>(use_offset_in1),  // 14: use_offset_in1
            use_out_offset,                         // 15: use_out_offset
        };
        append_accessors(args, weight_tensor, output_tensor);
        if (in1_needs_offsets) {
            tt::tt_metal::TensorAccessorArgs(*tensor_args.offsets_tensor.buffer()).append_to(args);
        }
        return args;
    };

    const tt::tt_metal::DataMovementConfigDescriptor in0_config{.processor = in0_risc, .noc = in0_noc};
    const tt::tt_metal::DataMovementConfigDescriptor in1_config{.processor = in1_risc, .noc = in1_noc};

    // ----- Compute kernel -----
    std::vector<uint32_t> compute_compile_time_args = {
        M_block_tiles,                       // 0: M_block_tiles
        K_block_tiles,                       // 1: K_block_tiles
        N_block_tiles,                       // 2: N_block_tiles
        split.N_blocks_per_core,             // 3: N_blocks_per_core
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

    // Kernels in the order of the k*Kernel handles.
    desc.kernels.push_back(make_kernel_descriptor(
        kIn0KernelPath,
        tt::tt_metal::CoreRangeSet(in0_sender_cores),
        in0_compile_time_args(/*is_injector_core=*/true),
        in0_defines,
        in0_config));
    desc.kernels.push_back(make_kernel_descriptor(
        kIn0KernelPath,
        tt::tt_metal::CoreRangeSet(in0_receiver_cores),
        in0_compile_time_args(/*is_injector_core=*/false),
        in0_defines,
        in0_config));
    desc.kernels.push_back(make_kernel_descriptor(
        kIn1KernelPath,
        tt::tt_metal::CoreRangeSet(in1_sender_cores),
        in1_compile_time_args(/*is_injector_core=*/true),
        in1_defines,
        in1_config));
    desc.kernels.push_back(make_kernel_descriptor(
        kIn1KernelPath,
        tt::tt_metal::CoreRangeSet(in1_receiver_cores),
        in1_compile_time_args(/*is_injector_core=*/false),
        in1_defines,
        in1_config));
    desc.kernels.push_back(make_compute_kernel_descriptor(
        core_grid_set,
        compute_compile_time_args,
        compute_defines,
        kComputeKernelPath,
        fp32_dest_acc_en,
        /*unpack_to_dest_mode=*/{},
        math_fidelity,
        math_approx_mode));

    // ----- Per-core runtime args -----
    // On a cache hit override_runtime_arguments() rewrites the k*Idx slots on every core.
    // defer_write_k_block stagger is computed kernel-side from runtime K_num_blocks +
    // Y_AXIS_CORES (CT). Host just hands the kernel core.y here.
    auto* const in0_buffer = input_tensor.buffer();
    auto* const in1_buffer = weight_tensor.buffer();
    auto* const out_buffer = output_tensor.buffer();
    auto* const offsets_buffer = tensor_args.offsets_tensor.buffer();

    const auto cores = tt::tt_metal::corerange_to_cores(core_grid, num_cores, true);
    for (const CoreCoord& core : cores) {
        const auto [in0_idx, in1_idx] = core_axes(core, transpose_core_grid);

        const CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        const CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            split.in1_parallel_axis_cores,
            in0_noc,
            /*axis_is_x_when_not_transposed=*/true,
            /*initial_endpoint=*/(transpose_core_grid ? top_core : left_core));

        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            split.in0_parallel_axis_cores,
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
        const uint32_t M_start_tile = split.M_tiles_per_core * in0_idx;
        const uint32_t M_end_tile = split.M_tiles_per_core * (in0_idx + 1);
        const uint32_t N_start_tile = split.N_tiles_per_core * in1_idx;
        const uint32_t N_end_tile = split.N_tiles_per_core * (in1_idx + 1);

        // Pass core.y to the kernel; it computes defer_write_k_block from runtime K_num_blocks
        // and Y_AXIS_CORES so the stagger is correct even when on-device offsets shrink K.
        const auto defer_write_k_block = static_cast<uint32_t>(core.y);

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
        tt::tt_metal::KernelDescriptor::RTArgList in0_args;
        in0_args.reserve(21U);
        in0_args.push_back(in0_buffer);
        in0_args.push_back(static_cast<uint32_t>(is_in0_sink));
        in0_args.push_back(static_cast<uint32_t>(in0_next_core_physical.x));
        in0_args.push_back(static_cast<uint32_t>(in0_next_core_physical.y));
        in0_args.push_back(static_cast<uint32_t>(in0_prev_core_physical.x));
        in0_args.push_back(static_cast<uint32_t>(in0_prev_core_physical.y));
        in0_args.push_back(M_start_tile);
        in0_args.push_back(M_end_tile);
        in0_args.push_back(N_start_tile);
        in0_args.push_back(N_end_tile);
        in0_args.push_back(defer_write_k_block);
        in0_args.push_back(out_buffer);
        in0_args.push_back(split.logical_M_tiles);
        in0_args.push_back(split.padded_M_tiles);
        in0_args.push_back(split.M_blocks_per_core);
        in0_args.push_back(split.parent_M_tiles);
        in0_args.push_back(split.parent_K_tiles_in0);
        in0_args.push_back(split.K_tiles);

        // On-device offsets RT args for dm_in0_sender:
        //   InputAndOutputRow: (offsets_addr, offsets_start_index, in0_idx) — kernel reads the
        //                      M range and derives per-core M from in0_idx.
        //   InputAndWeightK:   (offsets_addr, offsets_start_index)          — K-range only.
        if (in0_needs_offsets) {
            in0_args.push_back(offsets_buffer);
            in0_args.push_back(operation_attributes.offsets_start_index);
            if (offset_row_mode) {
                in0_args.push_back(in0_idx);
            }
        }
        desc.kernels[in1_idx == 0 ? kIn0SenderKernel : kIn0ReceiverKernel].emplace_runtime_args(core, in0_args);

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
        tt::tt_metal::KernelDescriptor::RTArgList in1_args;
        in1_args.reserve(20U);
        in1_args.push_back(in1_buffer);
        in1_args.push_back(static_cast<uint32_t>(is_in1_sink));
        in1_args.push_back(static_cast<uint32_t>(in1_next_core_physical.x));
        in1_args.push_back(static_cast<uint32_t>(in1_next_core_physical.y));
        in1_args.push_back(static_cast<uint32_t>(in1_prev_core_physical.x));
        in1_args.push_back(static_cast<uint32_t>(in1_prev_core_physical.y));
        in1_args.push_back(M_start_tile);
        in1_args.push_back(M_end_tile);
        in1_args.push_back(N_start_tile);
        in1_args.push_back(N_end_tile);
        in1_args.push_back(defer_write_k_block);
        in1_args.push_back(out_buffer);
        in1_args.push_back(split.logical_M_tiles);
        in1_args.push_back(split.padded_M_tiles);
        in1_args.push_back(split.M_blocks_per_core);
        in1_args.push_back(split.parent_K_tiles_in1);
        in1_args.push_back(split.K_tiles);

        // On-device offsets RT args for dm_in1_sender_out:
        //   InputAndWeightK:   (offsets_addr, offsets_start_index) — K-range only.
        //   InputAndOutputRow: also append in0_idx (kernel derives per-core M).
        if (in1_needs_offsets) {
            in1_args.push_back(offsets_buffer);
            in1_args.push_back(operation_attributes.offsets_start_index);
            if (offset_row_mode) {
                in1_args.push_back(in0_idx);
            }
        }
        desc.kernels[in0_idx == 0 ? kIn1SenderKernel : kIn1ReceiverKernel].emplace_runtime_args(core, in1_args);

        // Compute runtime args:
        //  0: M_start_tile
        //  1: M_end_tile
        //  2: N_start_tile
        //  3: N_end_tile
        //  4: M_blocks_per_core
        //  5: K_tiles (variable-K)
        desc.kernels[kComputeKernel].emplace_runtime_args(
            core, {M_start_tile, M_end_tile, N_start_tile, N_end_tile, split.M_blocks_per_core, split.K_tiles});
    }

    return desc;
}

void VariableMatmulProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VariableMatmulParams& operation_attributes,
    const VariableMatmulInputs& tensor_args,
    ttnn::Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    const MatmulSplit split = compute_matmul_split(operation_attributes, tensor_args);
    const auto grid_size = operation_attributes.config.compute_with_storage_grid_size;

    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = output_tensor.buffer()->address();
    const uint32_t offsets_addr = tensor_args.offsets_tensor.buffer()->address();
    const uint32_t offsets_start_index = operation_attributes.offsets_start_index;

    auto& in0_sender_rt = tt::tt_metal::GetRuntimeArgs(program, kIn0SenderKernel);
    auto& in0_receiver_rt = tt::tt_metal::GetRuntimeArgs(program, kIn0ReceiverKernel);
    auto& in1_sender_rt = tt::tt_metal::GetRuntimeArgs(program, kIn1SenderKernel);
    auto& in1_receiver_rt = tt::tt_metal::GetRuntimeArgs(program, kIn1ReceiverKernel);
    auto& compute_rt = tt::tt_metal::GetRuntimeArgs(program, kComputeKernel);

    for (uint32_t x = 0; x < grid_size.x; ++x) {
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            const auto [in0_idx, in1_idx] = core_axes(CoreCoord{x, y}, split.transpose_core_grid);
            const uint32_t M_start_tile = split.M_tiles_per_core * in0_idx;
            const uint32_t M_end_tile = split.M_tiles_per_core * (in0_idx + 1);

            // Sender and receiver kernels share the same RT-arg layout. Only the sender reads the
            // input address (the receiver gets data via NOC handshake from the sender), but
            // writing it to both keeps the slot consistent.
            auto& in0_args = (in1_idx == 0 ? in0_sender_rt : in0_receiver_rt)[x][y];
            in0_args[kIn0AddrIdx] = in0_addr;
            in0_args[kIn0MStartIdx] = M_start_tile;
            in0_args[kIn0MEndIdx] = M_end_tile;
            in0_args[kIn0OutAddrIdx] = out_addr;
            in0_args[kIn0LogicalMTilesIdx] = split.logical_M_tiles;
            in0_args[kIn0PaddedMTilesIdx] = split.padded_M_tiles;
            in0_args[kIn0MBlocksIdx] = split.M_blocks_per_core;
            in0_args[kIn0ParentMTilesIdx] = split.parent_M_tiles;
            in0_args[kIn0ParentKTilesIdx] = split.parent_K_tiles_in0;
            in0_args[kIn0KTilesIdx] = split.K_tiles;
            in0_args[kIn0OffsetsAddrIdx] = offsets_addr;
            in0_args[kIn0OffsetsStartIdx] = offsets_start_index;

            auto& in1_args = (in0_idx == 0 ? in1_sender_rt : in1_receiver_rt)[x][y];
            in1_args[kIn1AddrIdx] = in1_addr;
            in1_args[kIn1MStartIdx] = M_start_tile;
            in1_args[kIn1MEndIdx] = M_end_tile;
            in1_args[kIn1OutAddrIdx] = out_addr;
            in1_args[kIn1LogicalMTilesIdx] = split.logical_M_tiles;
            in1_args[kIn1PaddedMTilesIdx] = split.padded_M_tiles;
            in1_args[kIn1MBlocksIdx] = split.M_blocks_per_core;
            in1_args[kIn1ParentKTilesIdx] = split.parent_K_tiles_in1;
            in1_args[kIn1KTilesIdx] = split.K_tiles;
            in1_args[kIn1OffsetsAddrIdx] = offsets_addr;
            in1_args[kIn1OffsetsStartIdx] = offsets_start_index;

            auto& compute_args = compute_rt[x][y];
            compute_args[kComputeMStartIdx] = M_start_tile;
            compute_args[kComputeMEndIdx] = M_end_tile;
            compute_args[kComputeMBlocksIdx] = split.M_blocks_per_core;
            compute_args[kComputeKTilesIdx] = split.K_tiles;
        }
    }
}

}  // namespace ttml::metal::ops::variable_matmul::device
