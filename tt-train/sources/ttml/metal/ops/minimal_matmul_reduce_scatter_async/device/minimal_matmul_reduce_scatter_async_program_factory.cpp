// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_reduce_scatter_async_program_factory.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "slice_layout.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
namespace {
constexpr const char* kKernelDir =
    "tt-train/sources/ttml/metal/ops/minimal_matmul_reduce_scatter_async/device/kernels/";
constexpr uint32_t kRtInputAddrIdx = 0;
constexpr uint32_t kRtOutputAddrIdx = 1;
constexpr uint32_t kRtBufferAddrIdx = 2;

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
}  // namespace

ttnn::device_operation::CachedProgram<MinimalMatmulReduceScatterAsyncProgramFactory::shared_variables_t>
MinimalMatmulReduceScatterAsyncProgramFactory::create_at(
    const MinimalMatmulReduceScatterAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MinimalMatmulReduceScatterAsyncInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& weight_tensor = tensor_args.weight_tensor;
    const ttnn::Tensor& ring_buffer = tensor_args.persistent_buffer;
    const ttnn::Tensor& output_tensor = tensor_return_value;
    const auto& config = operation_attributes.config;
    auto* device = input_tensor.device();

    const auto& axis = operation_attributes.cluster_axis;
    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t ring_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, axis);

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

    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;
    const uint32_t K = input_tensor.padded_shape()[transpose_a ? -2 : -1];
    const uint32_t M = input_tensor.physical_volume() / K;
    const uint32_t N = weight_tensor.padded_shape()[transpose_b ? -2 : -1];
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

    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;
    const uint64_t work_rows_on_y = padded_work_per_core(M_tiles, N_tiles, ring_size, grid_size.y, grid_size.x);
    const uint64_t work_rows_on_x = padded_work_per_core(M_tiles, N_tiles, ring_size, grid_size.x, grid_size.y);
    const bool transpose_core_grid = work_rows_on_x != work_rows_on_y ? work_rows_on_x < work_rows_on_y : M > N;
    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;
    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    const SliceLayout layout = make_slice_layout(
        M_tiles,
        N_tiles,
        ring_size,
        in0_parallel_axis_cores,
        in1_parallel_axis_cores,
        M_block_tiles,
        N_block_tiles,
        transpose_a);
    const uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);
    const uint32_t K_blocks = padded_K_tiles / K_block_tiles;

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

    std::vector<ttnn::CoreCoord> in0_injector_cores;
    for (uint32_t y = 0; y < grid_size.y; ++y) {
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            if ((transpose_core_grid ? y : x) == 0) {
                in0_injector_cores.emplace_back(x, y);
            }
        }
    }
    const ttnn::CoreRangeSet in0_injectors(in0_injector_cores);
    if (transpose_a) {
        const uint64_t matmul_cb_bytes = tt::tt_metal::calculate_total_cb_size(program);
        const uint64_t free_l1 = ttml::metal::available_l1_bytes(*device);
        const uint64_t injector_cb_bytes = static_cast<uint64_t>(in0_block_num_tiles + 1) * in0_tile_size;
        TT_FATAL(
            matmul_cb_bytes + injector_cb_bytes < free_l1,
            "The matmul's circular buffers ({} B) and the in0 injectors' transposed copy ({} B) do not fit in L1 ({} B "
            "free); use smaller blocks",
            matmul_cb_bytes,
            injector_cb_bytes,
            free_l1);
        tt::tt_metal::create_cb(
            tt::CBIndex::c_7, program, in0_injectors, in0_tile_size, in0_block_num_tiles, in0_data_format);
        tt::tt_metal::create_cb(tt::CBIndex::c_8, program, in0_injectors, in0_tile_size, 1, in0_data_format);
    }

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
                               const ttnn::Tensor& own_input) {
        std::vector<uint32_t> args = {
            M_tiles,
            K_tiles,
            padded_K_tiles,
            N_tiles,
            M_block_tiles,
            K_block_tiles,
            N_block_tiles,
            layout.m_piece,
            layout.M_blocks,
            layout.N_blocks,
            input_tile_size,
            out_tile_size,
            sender_sem,
            receiver_sem,
            valid_sem,
            static_cast<uint32_t>(is_output_writer),
            static_cast<uint32_t>(is_injector),
            ring_size,
            layout.slice_tiles,
            static_cast<uint32_t>(transpose_b),
            static_cast<uint32_t>(transpose_a),
        };
        tt::tt_metal::TensorAccessorArgs(*own_input.buffer()).append_to(args);
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
        tt::tt_metal::TensorAccessorArgs(*ring_buffer.buffer()).append_to(args);
        return args;
    };

    const std::string in0_source = std::string(kKernelDir) + "dataflow/dm_in0.cpp";
    const std::string in1_source = std::string(kKernelDir) + "dataflow/dm_in1.cpp";

    std::map<bool, std::vector<ttnn::CoreCoord>> in0_groups;
    std::map<bool, std::vector<ttnn::CoreCoord>> in1_groups;
    for (uint32_t y = 0; y < grid_size.y; ++y) {
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            const ttnn::CoreCoord core{x, y};
            in0_groups[(transpose_core_grid ? y : x) == 0].emplace_back(core);
            in1_groups[(transpose_core_grid ? x : y) == 0].emplace_back(core);
        }
    }

    std::map<bool, tt::tt_metal::KernelHandle> in0_kernels;
    for (const auto& [injector, group_cores] : in0_groups) {
        in0_kernels[injector] = tt::tt_metal::CreateKernel(
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
                    input_tensor)});
    }
    std::map<bool, tt::tt_metal::KernelHandle> in1_kernels;
    for (const auto& [injector, group_cores] : in1_groups) {
        in1_kernels[injector] = tt::tt_metal::CreateKernel(
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
                    weight_tensor)});
    }

    std::map<std::string, std::string> compute_defines;
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        compute_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));
    auto create_compute = [&](const ttnn::CoreRangeSet& cores, uint32_t n_width, bool transposes_in0) {
        tt::tt_metal::CreateKernel(
            program,
            std::string(kKernelDir) + "compute/compute.cpp",
            cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args =
                    {K_blocks,
                     M_block_tiles,
                     K_block_tiles,
                     N_block_tiles,
                     layout.M_blocks,
                     layout.N_blocks,
                     subblock_h,
                     subblock_w,
                     layout.m_piece,
                     n_width,
                     ring_size,
                     static_cast<uint32_t>(transpose_b),
                     static_cast<uint32_t>(transposes_in0)},
                .defines = compute_defines});
    };
    if (transpose_a) {
        create_compute(in0_injectors, layout.n_first, true);
        create_compute(ttnn::CoreRangeSet(core_grid).subtract(in0_injectors), layout.n_piece, false);
    } else {
        create_compute(ttnn::CoreRangeSet(core_grid), layout.n_piece, false);
    }

    const uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);
    std::vector<shared_variables_t::CoreKernels> core_kernel_list;
    core_kernel_list.reserve(cores.size());
    for (uint32_t core_index = 0; core_index < cores.size(); ++core_index) {
        const auto& core = cores[core_index];
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

        const uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);
        const bool is_in0_sink = core == in0_core_order.back();
        const bool is_in1_sink = core == in1_core_order.back();
        const mmrs::Span m_share = layout.m_share(in0_idx);
        const mmrs::Span n_share = layout.n_share(in1_idx);

        auto dm_runtime_args =
            [&](uint32_t input_addr, bool is_sink, const ttnn::CoreCoord& next, const ttnn::CoreCoord& prev) {
                std::vector<uint32_t> args = {
                    input_addr,
                    output_tensor.buffer()->address(),
                    ring_buffer.buffer()->address(),
                    static_cast<uint32_t>(is_sink),
                    static_cast<uint32_t>(next.x),
                    static_cast<uint32_t>(next.y),
                    static_cast<uint32_t>(prev.x),
                    static_cast<uint32_t>(prev.y),
                    m_share.start,
                    m_share.size,
                    n_share.start,
                    n_share.size,
                    defer_write_k_block,
                    ring_index,
                };
                return args;
            };

        std::vector<uint32_t> in1_args =
            dm_runtime_args(weight_tensor.buffer()->address(), is_in1_sink, in1_next, in1_prev);
        std::vector<uint32_t> in0_args =
            dm_runtime_args(input_tensor.buffer()->address(), is_in0_sink, in0_next, in0_prev);

        shared_variables_t::CoreKernels core_kernels{.core = core};
        core_kernels.in0 = in0_kernels.at((transpose_core_grid ? core.y : core.x) == 0);
        tt::tt_metal::SetRuntimeArgs(program, *core_kernels.in0, core, in0_args);
        core_kernels.in1 = in1_kernels.at((transpose_core_grid ? core.x : core.y) == 0);
        tt::tt_metal::SetRuntimeArgs(program, *core_kernels.in1, core, in1_args);
        core_kernel_list.push_back(core_kernels);
    }

    return {std::move(program), shared_variables_t{.cores = std::move(core_kernel_list)}};
}

void MinimalMatmulReduceScatterAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MinimalMatmulReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulReduceScatterAsyncInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();
    const uint32_t buffer_addr = tensor_args.persistent_buffer.buffer()->address();

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
        auto patch_dm = [&](tt::tt_metal::KernelHandle kernel, const ttnn::CoreCoord& core, uint32_t input_addr) {
            auto& args = core_args(kernel, core);
            args[kRtInputAddrIdx] = input_addr;
            args[kRtOutputAddrIdx] = out_addr;
            args[kRtBufferAddrIdx] = buffer_addr;
        };
        for (const auto& [core, in0_kernel, in1_kernel] : vars.cores) {
            if (in0_kernel.has_value()) {
                patch_dm(*in0_kernel, core, in0_addr);
            }
            if (in1_kernel.has_value()) {
                patch_dm(*in1_kernel, core, in1_addr);
            }
        }
    }
}
}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async
