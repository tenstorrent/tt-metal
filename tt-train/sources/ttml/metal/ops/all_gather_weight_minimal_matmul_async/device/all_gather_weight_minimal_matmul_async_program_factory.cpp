// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_weight_minimal_matmul_async_program_factory.hpp"

#include <algorithm>
#include <array>
#include <internal/mesh_device.hpp>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "stripe_layout.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
namespace {
constexpr const char* kKernelDir =
    "tt-train/sources/ttml/metal/ops/all_gather_weight_minimal_matmul_async/device/kernels/";
// Runtime-arg slots that override_runtime_arguments patches on cache hits.
constexpr uint32_t kRtInputAddrIdx = 0;
constexpr uint32_t kRtOutputAddrIdx = 1;
constexpr uint32_t kRtDmFixedArgs = 12;  // data-mover args before stripe_order
constexpr uint32_t kRtSenderShardAddrIdx = 0;
constexpr uint32_t kRtSenderPwbAddrIdx = 1;
constexpr uint32_t kRtSenderSemaphoresIdx = 2;
constexpr uint32_t kMaxSendSlots = 16;
constexpr uint32_t kMinTilesPerPacket = 2;
constexpr uint32_t kMaxTilesPerPacket = 4;
// Each send slot has its packet header in front of the payload (send_on_grid.cpp); the header area is the fabric's
// packet header size rounded up to this alignment, which keeps the payload aligned for DRAM reads.
constexpr uint32_t kSendSlotAlignBytes = 64;

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

// Output tiles per core once M is padded over the M cores and every stripe over the chains.
uint64_t padded_work_per_core(
    uint32_t M_tiles, uint32_t N_tiles, uint32_t ring_size, uint32_t m_cores, uint32_t n_chains) {
    return static_cast<uint64_t>(tt::div_up(M_tiles, m_cores)) * ring_size * tt::div_up(N_tiles / ring_size, n_chains);
}

// A scatter-write packet carries at most kMaxTilesPerPacket tiles (the caller requires at least kMinTilesPerPacket).
uint32_t send_tiles_per_packet(uint32_t tile_size) {
    return std::min<uint32_t>(kMaxTilesPerPacket, tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / tile_size);
}

// On the sink row one data-mover kernel takes both operands, which frees a RISC for the fabric sender.
struct CoreRole {
    bool in0_injector;
    bool in1_injector;
    bool in0_sinks_in1;
    bool in1_sinks_in0;
    bool has_in0() const {
        return !in1_sinks_in0;
    }
    bool has_in1() const {
        return !in0_sinks_in1;
    }
};

CoreRole core_role(const ttnn::CoreCoord& core, bool transpose_core_grid, uint32_t sink_row) {
    const bool on_sink_row = core.y == sink_row;
    return CoreRole{
        .in0_injector = (transpose_core_grid ? core.y : core.x) == 0,
        .in1_injector = (transpose_core_grid ? core.x : core.y) == 0,
        .in0_sinks_in1 = on_sink_row && !transpose_core_grid,
        .in1_sinks_in0 = on_sink_row && transpose_core_grid};
}
}  // namespace

ttnn::device_operation::CachedProgram<AllGatherWeightMinimalMatmulAsyncProgramFactory::shared_variables_t>
AllGatherWeightMinimalMatmulAsyncProgramFactory::create_at(
    const AllGatherWeightMinimalMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherWeightMinimalMatmulAsyncInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& weight_tensor = tensor_args.weight_tensor;
    const ttnn::Tensor& output_tensor = tensor_return_value;
    const auto& config = operation_attributes.config;
    auto* device = input_tensor.device();

    const ttnn::ccl::Topology topology = operation_attributes.topology;
    const auto& axis = operation_attributes.cluster_axis;
    const uint32_t ring_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, axis);
    const std::optional<ttnn::MeshCoordinate> forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, mesh_coordinate, 1, topology, axis);
    const std::optional<ttnn::MeshCoordinate> backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, mesh_coordinate, -1, topology, axis);

    const auto grid_size =
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

    const bool transpose_b = operation_attributes.transpose_b;
    const uint32_t K = input_tensor.padded_shape()[-1];
    const uint32_t M = input_tensor.physical_volume() / K;  // all leading dims of the input fold into M
    const uint32_t N = tensor_args.persistent_weight_buffer.padded_shape()[transpose_b ? -2 : -1];
    const uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    const uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    const uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    uint32_t M_block_tiles = kDefaultBlockTiles;
    uint32_t K_block_tiles = kDefaultBlockTiles;
    uint32_t N_block_tiles = kDefaultBlockTiles;
    uint32_t subblock_h = 4;
    uint32_t subblock_w = 1;
    if (config.has_value()) {
        M_block_tiles = config.value().M_block_size;
        K_block_tiles = config.value().K_block_size;
        N_block_tiles = config.value().N_block_size;
        subblock_h = config.value().subblock_h;
        subblock_w = config.value().subblock_w;
    }

    // As in minimal_matmul: the bigger operand reads on the DRAM-read NOC, the smaller one also writes the output.
    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;
    const uint64_t work_rows_on_y =
        padded_work_per_core(M_tiles, N_tiles, operation_attributes.ring_size, grid_size.y, grid_size.x);
    const uint64_t work_rows_on_x =
        padded_work_per_core(M_tiles, N_tiles, operation_attributes.ring_size, grid_size.x, grid_size.y);
    // Put M on whichever grid axis leaves less padded work per core.
    const bool transpose_core_grid = work_rows_on_x != work_rows_on_y ? work_rows_on_x < work_rows_on_y : M > N;
    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;
    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    // The column chains end on this row.
    const uint32_t sink_row =
        build_core_order_for_axis({0, 0}, false, grid_size.y, large_input_noc, false, {0, 0}).first.back().y;

    const uint32_t padded_M_tiles = tt::round_up(M_tiles, in0_parallel_axis_cores);
    const uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);
    const uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    const uint32_t K_blocks = padded_K_tiles / K_block_tiles;
    const uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);

    const bool ring = topology == ttnn::ccl::Topology::Ring;
    const StripeLayout layout = make_stripe_layout(
        N_tiles, operation_attributes.ring_size, ring_index, in1_parallel_axis_cores, N_block_tiles, ring);
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
    // Partials accumulate here (packer L1 accumulation) before being copied out in the output format.
    tt::tt_metal::create_cb(
        tt::CBIndex::c_3, program, core_grid, intermediate_tile_size, out_block_num_tiles, intermediate_data_format);

    const uint32_t tiles_per_packet = send_tiles_per_packet(in1_tile_size);
    TT_FATAL(
        tiles_per_packet >= kMinTilesPerPacket,
        "all_gather_weight_minimal_matmul_async: the fabric packet payload ({} B) must hold at least {} weight "
        "tiles of {} B",
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
        kMinTilesPerPacket,
        in1_tile_size);
    const uint32_t slot_header_bytes = tt::round_up(
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()), kSendSlotAlignBytes);
    const uint64_t slot_bytes = slot_header_bytes + (static_cast<uint64_t>(tiles_per_packet) * in1_tile_size);
    TT_FATAL(
        slot_bytes % kSendSlotAlignBytes == 0,
        "all_gather_weight_minimal_matmul_async: a send slot ({} B) must be a multiple of {} B to keep its payload "
        "aligned for DRAM reads",
        slot_bytes,
        kSendSlotAlignBytes);
    const uint64_t matmul_cb_bytes = tt::tt_metal::calculate_total_cb_size(program);
    const uint64_t free_l1 = ttml::metal::available_l1_bytes(*device);
    TT_FATAL(
        matmul_cb_bytes + kSendSlotAlignBytes <= free_l1,
        "all_gather_weight_minimal_matmul_async: the matmul's circular buffers take {} B of the {} B of L1 available "
        "per core, which leaves no room for the fabric senders' packet slots on the sink row; use smaller blocks",
        matmul_cb_bytes,
        free_l1);
    const uint64_t slot_room = free_l1 - matmul_cb_bytes - kSendSlotAlignBytes;
    const uint32_t grid_send_slots =
        static_cast<uint32_t>(std::min<uint64_t>(kMaxSendSlots, slot_room / slot_bytes)) & ~1U;
    TT_FATAL(
        grid_send_slots >= 2,
        "all_gather_weight_minimal_matmul_async: {} B of L1 are left for the fabric senders' packet slots after the "
        "matmul's circular buffers, but two slots need {} B; use smaller blocks",
        slot_room,
        2 * slot_bytes);

    // Chain handshake: a core with room bumps its upstream's sender semaphore, the upstream writes the block and
    // flips the core's receiver semaphore to VALID.
    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    const bool in0_is_output_writer = !transpose_core_grid;
    const bool in1_is_output_writer = transpose_core_grid;

    auto dm_compile_args = [&](uint32_t input_tile_size,
                               uint32_t sender_sem,
                               uint32_t receiver_sem,
                               uint32_t valid_sem,
                               bool is_output_writer,
                               bool is_injector,
                               bool sink_other,
                               uint32_t other_sender_sem,
                               uint32_t other_receiver_sem,
                               const ttnn::Tensor& own_input,
                               const ttnn::Tensor* extra_input = nullptr) {
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
            layout.hops_before,
            static_cast<uint32_t>(transpose_b),
            static_cast<uint32_t>(sink_other),  // sink row: also receive the other operand, with its semaphores
            other_sender_sem,
            other_receiver_sem,
        };
        tt::tt_metal::TensorAccessorArgs(*own_input.buffer()).append_to(args);
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
        if (extra_input != nullptr) {
            tt::tt_metal::TensorAccessorArgs(*extra_input->buffer()).append_to(args);
        }
        return args;
    };

    const std::string in0_source = std::string(kKernelDir) + "dataflow/dm_in0.cpp";
    const std::string in1_source = std::string(kKernelDir) + "dataflow/dm_in1.cpp";
    const ttnn::Tensor& pwb = tensor_args.persistent_weight_buffer;

    // Cores with the same (injector, sink row) role share one kernel.
    std::map<std::pair<bool, bool>, std::vector<ttnn::CoreCoord>> in0_groups;
    std::map<std::pair<bool, bool>, std::vector<ttnn::CoreCoord>> in1_groups;
    for (uint32_t y = 0; y < grid_size.y; ++y) {
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            const ttnn::CoreCoord core{x, y};
            const CoreRole role = core_role(core, transpose_core_grid, sink_row);
            if (role.has_in0()) {
                in0_groups[{role.in0_injector, role.in0_sinks_in1}].emplace_back(core);
            }
            if (role.has_in1()) {
                in1_groups[{role.in1_injector, role.in1_sinks_in0}].emplace_back(core);
            }
        }
    }

    std::map<std::pair<bool, bool>, tt::tt_metal::KernelHandle> in0_kernels;
    for (const auto& [key, group_cores] : in0_groups) {
        const auto [injector, sinks_in1] = key;
        in0_kernels[key] = tt::tt_metal::CreateKernel(
            program,
            in0_source,
            ttnn::CoreRangeSet(group_cores),
            tt::tt_metal::DataMovementConfig{
                .processor = in0_risc,
                .noc = in0_noc,
                .compile_args = dm_compile_args(
                    in0_tile_size,
                    in0_sender_semaphore_id,
                    in0_receiver_semaphore_id,
                    in0_valid_semaphore_id,
                    in0_is_output_writer,
                    injector,
                    sinks_in1,
                    in1_sender_semaphore_id,
                    in1_receiver_semaphore_id,
                    input_tensor)});
    }
    std::map<std::pair<bool, bool>, tt::tt_metal::KernelHandle> in1_kernels;
    for (const auto& [key, group_cores] : in1_groups) {
        const auto [injector, sinks_in0] = key;
        in1_kernels[key] = tt::tt_metal::CreateKernel(
            program,
            in1_source,
            ttnn::CoreRangeSet(group_cores),
            tt::tt_metal::DataMovementConfig{
                .processor = in1_risc,
                .noc = in1_noc,
                .compile_args = dm_compile_args(
                    in1_tile_size,
                    in1_sender_semaphore_id,
                    in1_receiver_semaphore_id,
                    in1_valid_semaphore_id,
                    in1_is_output_writer,
                    injector,
                    sinks_in0,
                    in0_sender_semaphore_id,
                    in0_receiver_semaphore_id,
                    weight_tensor,
                    &pwb)});
    }

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
            .dst_full_sync_en = dst_full_sync_en,
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
                 layout.ring_size,
                 static_cast<uint32_t>(transpose_b),
                 layout.hops_before},
            .defines = compute_defines});

    const uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);
    std::vector<shared_variables_t::CoreKernels> core_kernel_list;
    core_kernel_list.reserve(cores.size());
    std::vector<uint32_t> stripe_sem_addrs;
    stripe_sem_addrs.reserve(operation_attributes.semaphores.size());
    for (const auto& sem : operation_attributes.semaphores) {
        stripe_sem_addrs.push_back(static_cast<uint32_t>(sem.address()));
    }
    for (const auto& core : cores) {
        const uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        const uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        // Injectors of this core's row and column chains.
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
        // Rows write their output at different K blocks so the writes don't all hit the NOC at once.
        const uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);
        const bool is_in0_sink = core == in0_core_order.back();
        const bool is_in1_sink = core == in1_core_order.back();

        const uint32_t chain_offset = layout.chain_offset(in1_idx);
        const uint32_t chain_width = layout.chain_width(in1_idx);

        auto dm_runtime_args = [&](uint32_t input_addr,
                                   bool is_sink,
                                   const ttnn::CoreCoord& next,
                                   const ttnn::CoreCoord& prev) {
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
            // override_runtime_arguments finds the persistent buffer and semaphores past these.
            TT_FATAL(
                args.size() == kRtDmFixedArgs && layout.stripe_order.size() == operation_attributes.ring_size,
                "all_gather_weight_minimal_matmul_async: the data-mover runtime args do not match kRtDmFixedArgs "
                "({} fixed args, expected {}) or the ring ({} stripes, ring size {})",
                args.size(),
                kRtDmFixedArgs,
                layout.stripe_order.size(),
                operation_attributes.ring_size);
            args.insert(args.end(), layout.stripe_order.begin(), layout.stripe_order.end());
            return args;
        };

        std::vector<uint32_t> in1_args =
            dm_runtime_args(weight_tensor.buffer()->address(), is_in1_sink, in1_next, in1_prev);
        // in1 also gets the persistent buffer and one semaphore per stripe.
        in1_args.push_back(pwb.buffer()->address());
        in1_args.insert(in1_args.end(), stripe_sem_addrs.begin(), stripe_sem_addrs.end());
        std::vector<uint32_t> in0_args =
            dm_runtime_args(input_tensor.buffer()->address(), is_in0_sink, in0_next, in0_prev);

        const CoreRole role = core_role(core, transpose_core_grid, sink_row);
        shared_variables_t::CoreKernels core_kernels{.core = core};
        // A sink-row kernel also receives the other operand, so it needs that chain's previous core.
        if (role.has_in0()) {
            if (role.in0_sinks_in1) {
                in0_args.insert(in0_args.end(), {static_cast<uint32_t>(in1_prev.x), static_cast<uint32_t>(in1_prev.y)});
            }
            core_kernels.in0 = in0_kernels.at({role.in0_injector, role.in0_sinks_in1});
            tt::tt_metal::SetRuntimeArgs(program, *core_kernels.in0, core, in0_args);
        }
        if (role.has_in1()) {
            if (role.in1_sinks_in0) {
                in1_args.insert(in1_args.end(), {static_cast<uint32_t>(in0_prev.x), static_cast<uint32_t>(in0_prev.y)});
            }
            core_kernels.in1 = in1_kernels.at({role.in1_injector, role.in1_sinks_in0});
            tt::tt_metal::SetRuntimeArgs(program, *core_kernels.in1, core, in1_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {M_start_tile, M_end_tile});
        core_kernel_list.push_back(core_kernels);
    }

    const uint32_t num_links = operation_attributes.num_links;
    std::vector<shared_variables_t::Sender> senders;
    senders.reserve(2 * num_links);
    // parity_unit = num_links: each run of num_links chains (one per link) sends the far stripe the same way, so
    // every link splits it evenly between the two directions.
    const RelayPlan plan = make_relay_plan(layout, ring_index, ring, num_links);
    const auto [forward_route, backward_route] = ttnn::ccl::get_forward_backward_line_unicast_configuration(
        mesh_coordinate, forward_coord, backward_coord, device);

    const uint32_t chunk_tiles = K_block_tiles;  // senders ship K in the chunks the matmul consumes
    const auto src_node_id = device->get_fabric_node_id(mesh_coordinate);

    // Senders run on the sink row's free data movers - column 0 stays free. Each one sits in the column of the worker
    // closest to its link's Ethernet core or in the nearest free column before it: its packets leave on
    // large_input_noc, which goes one way round the row, so every column back costs one hop and a column past it
    // costs almost the whole row.
    auto sender_cores_at = [&](const ttnn::MeshCoordinate& coord) {
        const std::array<std::optional<ttnn::MeshCoordinate>, 2> peers = {
            ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, coord, 1, topology, axis),
            ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, coord, -1, topology, axis)};
        const tt::tt_metal::IDevice* chip = tt::tt_metal::internal::get_device(*device, coord);
        const auto src = device->get_fabric_node_id(coord);
        std::vector<ttnn::CoreCoord> cores(2 * num_links);
        std::set<uint32_t> taken = {0};
        auto take_at_or_before = [&](uint32_t x) {
            x = std::min<uint32_t>(x, grid_size.x - 1);
            while (taken.contains(x)) {
                x = (x == 0) ? grid_size.x - 1 : x - 1;
            }
            taken.insert(x);
            return ttnn::CoreCoord{x, sink_row};
        };
        for (uint32_t link = 0; link < num_links; ++link) {
            for (uint32_t dir = 0; dir < 2; ++dir) {
                if (!peers[dir].has_value()) {
                    continue;  // no sender in this direction, and nobody looks its core up
                }
                const auto dst = device->get_fabric_node_id(peers[dir].value());
                const auto links = tt::tt_fabric::get_forwarding_link_indices(*device, src, dst);
                TT_FATAL(link < links.size(), "AGMM: link {} of {} has no fabric route", link, num_links);
                const ttnn::CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(*device, src, dst, links[link]);
                const auto closest =
                    tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(*chip, eth, large_input_noc);
                cores[(2 * link) + dir] = take_at_or_before(closest.logical_coord.x);
            }
        }
        return cores;
    };
    const std::vector<ttnn::CoreCoord> my_sender_cores = sender_cores_at(mesh_coordinate);

    // Packet slots (header + payload each), plus one alignment unit of slack so the kernel can align the base.
    const auto send_slots_bytes = static_cast<uint32_t>((grid_send_slots * slot_bytes) + kSendSlotAlignBytes);
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < 2; ++dir) {
            const auto& neighbour = dir == 0 ? forward_coord : backward_coord;
            if (!neighbour.has_value()) {
                continue;
            }
            const auto& segments = dir == 0 ? plan.forward : plan.backward;
            // Schedule: per segment {stripe, distance, num_pieces}, then per piece
            // {chain offset, chain width, injector x, injector y, notify}. Chains are dealt round-robin over links.
            uint32_t num_segments = 0;
            std::vector<uint32_t> schedule_body;
            for (const auto& seg : segments) {
                std::vector<uint32_t> piece_args;
                uint32_t num_pieces = 0;
                for (const auto& p : seg.pieces) {
                    if (p.chain % num_links != link) {
                        continue;
                    }
                    const ttnn::CoreCoord injector = device->worker_core_from_logical_core(
                        transpose_core_grid ? ttnn::CoreCoord{0, p.chain} : ttnn::CoreCoord{p.chain, 0});
                    piece_args.insert(
                        piece_args.end(),
                        {layout.chain_offset(p.chain),
                         layout.chain_width(p.chain),
                         static_cast<uint32_t>(injector.x),
                         static_cast<uint32_t>(injector.y),
                         p.notify});
                    ++num_pieces;
                }
                if (num_pieces == 0) {
                    continue;
                }
                schedule_body.insert(schedule_body.end(), {seg.stripe, seg.distance, num_pieces});
                schedule_body.insert(schedule_body.end(), piece_args.begin(), piece_args.end());
                ++num_segments;
            }
            if (num_segments == 0) {
                continue;
            }

            const uint32_t sender_index = (2 * link) + dir;
            const ttnn::CoreCoord send_core = my_sender_cores[sender_index];
            // The sender of this link and direction on the next device, which forwards what we send it.
            const ttnn::CoreCoord peer_virtual_core =
                device->worker_core_from_logical_core(sender_cores_at(neighbour.value())[sender_index]);
            create_circular_buffer_bytes(
                program,
                tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(send_core)),
                tt::CBIndex::c_4,
                tt::DataFormat::UInt32,
                send_slots_bytes);
            std::vector<uint32_t> send_ct = {
                in1_tile_size,
                tiles_per_packet,
                K_tiles,
                chunk_tiles,
                layout.stripe_tiles,
                layout.ring_size,
                static_cast<uint32_t>(transpose_b),
                grid_send_slots,
                slot_header_bytes};
            const auto& route = dir == 0 ? forward_route : backward_route;
            send_ct.insert(send_ct.end(), route.begin(), route.end());
            tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(send_ct);
            tt::tt_metal::TensorAccessorArgs(*pwb.buffer()).append_to(send_ct);
            auto send_id = tt::tt_metal::CreateKernel(
                program,
                std::string(kKernelDir) + "dataflow/send_on_grid.cpp",
                send_core,
                tt::tt_metal::DataMovementConfig{
                    .processor = large_input_risc, .noc = large_input_noc, .compile_args = send_ct});
            std::vector<uint32_t> send_rt = {weight_tensor.buffer()->address(), pwb.buffer()->address()};
            send_rt.insert(send_rt.end(), stripe_sem_addrs.begin(), stripe_sem_addrs.end());
            send_rt.insert(
                send_rt.end(),
                {static_cast<uint32_t>(peer_virtual_core.x), static_cast<uint32_t>(peer_virtual_core.y)});
            // Router connection for this link; the kernel opens it with open_connections.
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                *device,
                src_node_id,
                {device->get_fabric_node_id(neighbour.value())},
                {link},
                program,
                send_id,
                send_core,
                send_rt,
                tt::tt_fabric::FabricApiType::Linear);
            send_rt.push_back(num_segments);
            send_rt.insert(send_rt.end(), schedule_body.begin(), schedule_body.end());
            tt::tt_metal::SetRuntimeArgs(program, send_id, send_core, send_rt);
            senders.push_back({.core = send_core, .kernel = send_id});
        }
    }

    return {
        std::move(program), shared_variables_t{.cores = std::move(core_kernel_list), .senders = std::move(senders)}};
}

void AllGatherWeightMinimalMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherWeightMinimalMatmulAsyncParams& operation_attributes,
    const AllGatherWeightMinimalMatmulAsyncInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    // Only buffer and semaphore addresses change between cache hits.
    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();
    const uint32_t pwb_addr = tensor_args.persistent_weight_buffer.buffer()->address();
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t in1_pwb_idx = kRtDmFixedArgs + ring_size;

    auto patch_semaphores = [&](auto& args, uint32_t first) {
        for (uint32_t s = 0; s < ring_size; ++s) {
            args[first + s] = static_cast<uint32_t>(operation_attributes.semaphores[s].address());
        }
    };
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto vars_it = cached_workload.shared_variables.find(range);
        TT_FATAL(
            vars_it != cached_workload.shared_variables.end(),
            "No cached program state for mesh coordinate range {}: the cached workload does not match its programs.",
            range);
        const auto& vars = vars_it->second;
        std::map<tt::tt_metal::KernelHandle, std::vector<std::vector<tt::tt_metal::RuntimeArgsData>>*> args_by_kernel;
        auto core_args = [&](tt::tt_metal::KernelHandle kernel,
                             const ttnn::CoreCoord& core) -> tt::tt_metal::RuntimeArgsData& {
            auto [it, inserted] = args_by_kernel.try_emplace(kernel, nullptr);
            if (inserted) {
                it->second = &GetRuntimeArgs(program, kernel);
            }
            return (*it->second)[core.x][core.y];
        };
        for (const auto& [core, in0_kernel, in1_kernel] : vars.cores) {
            if (in0_kernel.has_value()) {
                auto& in0_args = core_args(*in0_kernel, core);
                in0_args[kRtInputAddrIdx] = in0_addr;
                in0_args[kRtOutputAddrIdx] = out_addr;
            }
            if (in1_kernel.has_value()) {
                auto& in1_args = core_args(*in1_kernel, core);
                in1_args[kRtInputAddrIdx] = in1_addr;
                in1_args[kRtOutputAddrIdx] = out_addr;
                in1_args[in1_pwb_idx] = pwb_addr;
                patch_semaphores(in1_args, in1_pwb_idx + 1);
            }
        }
        for (const auto& sender : vars.senders) {
            auto& sender_args = core_args(sender.kernel, sender.core);
            sender_args[kRtSenderShardAddrIdx] = in1_addr;
            sender_args[kRtSenderPwbAddrIdx] = pwb_addr;
            patch_semaphores(sender_args, kRtSenderSemaphoresIdx);
        }
    }
}
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
