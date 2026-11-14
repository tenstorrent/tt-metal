// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "metal/ttnn_all_includes.hpp"

// Shared by rmsnorm_fw and rmsnorm_bw: the shapes of rms and of the tensors that follow the gate, and the input and
// gamma checks both validates run.
namespace ttml::metal::ops::rmsnorm {

// One rms per row: the input's shape with the last dim 1.
inline ttnn::Shape get_rms_shape(const ttnn::Tensor& input) {
    auto shape = input.logical_shape();
    shape[-1] = 1U;
    return shape;
}

// The forward output, dL_dout and dL_dgate take the gate's head-merged shape in gated mode and the input's otherwise.
inline ttnn::Shape get_output_shape(const ttnn::Tensor& input, const std::optional<ttnn::Tensor>& gate) {
    return gate.has_value() ? gate->logical_shape() : input.logical_shape();
}

// A rank-4 input and a gamma of [1, 1, 1, V] for the input's V, both with only tile padding.
inline void check_input_and_gamma_shapes(const ttnn::Tensor& input, const ttnn::Tensor& gamma, std::string_view op) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4U, "{}: Input must be rank 4, got shape {}", op, input_shape);
    check_tile_padding(input, op, "Input");
    check_shape(gamma, op, "Gamma", ttnn::Shape({1U, 1U, 1U, input_shape[-1]}));
}

// Upper bound on num_slices. Every core of a row-group shares its partial row sum with all the others, so each core
// receives num_slices tiles per row and keeps two rows of them in L1: exchange time and L1 both grow with num_slices.
constexpr uint32_t kMaxRowSlices = 16U;
// A sliced row costs S + kRowSliceExchangeCostTiles extra tiles of work: S to sum the gathered partials, plus the NOC
// writes and semaphore wait of the exchange. 4 is an unmeasured estimate: that round trip takes a few us, against ~1 us
// per tile of compute. To retune, rerun rmsnorm_benchmark on its H=1, T=1024/4096 rows.
constexpr uint32_t kRowSliceExchangeCostTiles = 4U;

// How rmsnorm_fw and rmsnorm_bw split tile-rows over the cores: each row is cut into num_slices column slices of
// slice_tiles tiles, and num_slices = 1 keeps whole rows. Core i of the walk {i / num_cores_y, i % num_cores_y} handles
// slice i % num_slices of row-group i / num_slices. Row-groups [0, num_groups_1) take rows_per_group_1 rows, the rest
// rows_per_group_2. The core ranges come from row_slice_cores().
struct RowSlicePlan {
    uint32_t num_slices = 1U;
    uint32_t slice_tiles = 0U;
    uint32_t num_groups_1 = 0U;
    uint32_t rows_per_group_1 = 0U;
    uint32_t rows_per_group_2 = 0U;
    uint32_t num_cores = 0U;
    uint32_t num_cores_y = 0U;
    // L1 bytes each core needs for its slice_tiles and has, both from make_row_slice_plan, so the factory sizes its
    // CBs from the same budget the planner chose by. plan_row_slices leaves them 0.
    uint64_t l1_bytes_required = 0U;
    uint64_t l1_bytes_available = 0U;

    [[nodiscard]] bool has_group_2() const {
        return num_groups_1 * num_slices < num_cores;
    }
    [[nodiscard]] bool fits_in_l1() const {
        return l1_bytes_required <= l1_bytes_available;
    }
};

struct RowSliceCores {
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
};

// Picks num_slices by the work of the busiest core: rounds of rows times the tiles per row, plus the exchange.
// Only slice counts up to max_slices that divide Wt and whose slice `slice_fits_in_l1(slice_tiles, num_slices)`
// are considered; ties keep the smaller count.
template <typename SliceFitsInL1>
RowSlicePlan plan_row_slices(
    const tt::tt_metal::CoreCoord grid,
    const uint32_t total_rows,
    const uint32_t Wt,
    const uint32_t max_slices,
    SliceFitsInL1&& slice_fits_in_l1) {
    RowSlicePlan plan;
    plan.slice_tiles = Wt;
    plan.num_cores_y = grid.y;
    if (total_rows == 0U) {
        return plan;
    }

    const uint32_t max_cores = grid.x * grid.y;
    const auto num_groups_for = [&](const uint32_t num_slices) { return std::min(total_rows, max_cores / num_slices); };
    const auto cost = [&](const uint32_t num_slices) -> uint64_t {
        const uint32_t num_groups = num_groups_for(num_slices);
        const uint64_t rounds = (total_rows + num_groups - 1U) / num_groups;
        const uint64_t exchange = num_slices > 1U ? num_slices + kRowSliceExchangeCostTiles : 0U;
        return rounds * (Wt / num_slices + exchange);
    };

    uint32_t num_slices = 1U;
    uint64_t best_cost = cost(1U);
    for (uint32_t candidate = 2U; candidate <= std::min({Wt, max_cores, max_slices, kMaxRowSlices}); ++candidate) {
        if (Wt % candidate != 0U || !slice_fits_in_l1(Wt / candidate, candidate)) {
            continue;
        }
        if (const uint64_t candidate_cost = cost(candidate); candidate_cost < best_cost) {
            best_cost = candidate_cost;
            num_slices = candidate;
        }
    }

    plan.num_slices = num_slices;
    plan.slice_tiles = Wt / num_slices;
    const uint32_t num_groups = num_groups_for(num_slices);
    plan.num_cores = num_groups * num_slices;
    const uint32_t extra_rows = total_rows % num_groups;
    plan.rows_per_group_2 = total_rows / num_groups;
    if (extra_rows == 0U) {
        plan.num_groups_1 = num_groups;
        plan.rows_per_group_1 = plan.rows_per_group_2;
    } else {
        plan.num_groups_1 = extra_rows;
        plan.rows_per_group_1 = plan.rows_per_group_2 + 1U;
    }
    return plan;
}

// plan_row_slices over input's tile rows, keeping slices whose required_l1_bytes fit in free L1; gated rows are never
// sliced. The plan carries the chosen split's L1, which for whole rows may not fit (they then stream).
template <typename RequiredL1Bytes>
RowSlicePlan make_row_slice_plan(
    const ttnn::Tensor& input, const bool gated, const uint32_t max_row_slices, RequiredL1Bytes&& required_l1_bytes) {
    const uint64_t l1_bytes_available = available_l1_bytes(*input.device());
    auto plan = plan_row_slices(
        input.device()->compute_with_storage_grid_size(),
        num_tile_rows(input),
        input.padded_shape()[-1] / tt::constants::TILE_WIDTH,
        /* max_slices */ gated ? 1U : max_row_slices,
        [&](const uint32_t slice_tiles, const uint32_t num_slices) {
            return required_l1_bytes(slice_tiles, num_slices) <= l1_bytes_available;
        });
    plan.l1_bytes_required = required_l1_bytes(plan.slice_tiles, plan.num_slices);
    plan.l1_bytes_available = l1_bytes_available;
    return plan;
}

// The cores of a plan made by plan_row_slices on the same grid: all of them, and those of each row-group size.
inline RowSliceCores row_slice_cores(const RowSlicePlan& plan, const tt::tt_metal::CoreCoord grid) {
    RowSliceCores cores;
    const uint32_t num_cores_1 = plan.num_groups_1 * plan.num_slices;
    cores.all_cores = tt::tt_metal::num_cores_to_corerangeset(plan.num_cores, grid, /* row_wise */ false);
    cores.core_group_1 = tt::tt_metal::num_cores_to_corerangeset(num_cores_1, grid, /* row_wise */ false);
    if (plan.has_group_2()) {
        cores.core_group_2 = tt::tt_metal::num_cores_to_corerangeset(
            tt::tt_metal::CoreCoord{num_cores_1 / grid.y, num_cores_1 % grid.y},
            plan.num_cores - num_cores_1,
            grid,
            /* row_wise */ false);
    }
    return cores;
}

// Calls fn(core) for each core a kernel holds runtime args on. On a cache hit these are the cores create_descriptor()
// planned, so override_runtime_arguments() finds them without running the planner again.
template <typename KernelRuntimeArgs, typename Fn>
inline void for_each_core_with_runtime_args(const KernelRuntimeArgs& runtime_args, Fn&& fn) {
    for (size_t x = 0; x < runtime_args.size(); ++x) {
        for (size_t y = 0; y < runtime_args[x].size(); ++y) {
            if (runtime_args[x][y].size() > 0U) {
                fn(tt::tt_metal::CoreCoord{x, y});
            }
        }
    }
}

// One core's share of a RowSlicePlan, as handed out by for_each_row_slice_core.
struct RowSliceWork {
    tt::tt_metal::CoreCoord core;
    uint32_t group;
    uint32_t slice;
    uint32_t num_rows;
    uint32_t start_row;
};

template <typename Fn>
inline void for_each_row_slice_core(const RowSlicePlan& plan, Fn&& fn) {
    for_each_core(plan.num_cores, plan.num_cores_y, [&](const tt::tt_metal::CoreCoord& core, const uint32_t i) {
        const uint32_t group = i / plan.num_slices;
        const bool in_group_1 = group < plan.num_groups_1;
        const uint32_t start_row = in_group_1 ? group * plan.rows_per_group_1
                                              : plan.num_groups_1 * plan.rows_per_group_1 +
                                                    (group - plan.num_groups_1) * plan.rows_per_group_2;
        fn(RowSliceWork{
            .core = core,
            .group = group,
            .slice = i % plan.num_slices,
            .num_rows = in_group_1 ? plan.rows_per_group_1 : plan.rows_per_group_2,
            .start_row = start_row});
    });
}

// Whether the cores of a row-group are all in one column of the logical grid, which they are unless the group wraps
// from the bottom of one column to the top of the next.
inline bool row_group_in_one_column(const RowSlicePlan& plan, const uint32_t group) {
    const uint32_t first = group * plan.num_slices;
    const uint32_t last = first + plan.num_slices - 1U;
    return first / plan.num_cores_y == last / plan.num_cores_y;
}

// The writer's exchange arguments: the core's slice, whether its row-group's cores are one column of consecutive NOC
// coordinates (so the exchange can multicast), then the NOC x and y of each core of the row-group in slice order. Empty
// without slicing.
inline std::vector<uint32_t> row_slice_exchange_args(
    const RowSlicePlan& plan, const RowSliceWork& work, const tt::tt_metal::IDevice& device) {
    if (plan.num_slices == 1U) {
        return {};
    }
    std::vector<uint32_t> args{work.slice, 0U};
    args.reserve(2U + 2U * plan.num_slices);
    bool in_one_column = row_group_in_one_column(plan, work.group);
    tt::tt_metal::CoreCoord first_noc;
    for (uint32_t slice = 0; slice < plan.num_slices; ++slice) {
        const uint32_t peer = work.group * plan.num_slices + slice;
        const auto noc = device.worker_core_from_logical_core({peer / plan.num_cores_y, peer % plan.num_cores_y});
        if (slice == 0U) {
            first_noc = noc;
        }
        // The multicast covers the rectangle from the first core to the last, so it must be exactly these cores.
        in_one_column = in_one_column && noc.x == first_noc.x && noc.y == first_noc.y + slice;
        args.push_back(noc.x);
        args.push_back(noc.y);
    }
    args[1] = in_one_column ? 1U : 0U;
    return args;
}

// The partial exchange's CBs, in fp32: this core's partial, and two rows of gathered partials, so a peer can send the
// next row's partial while this core still sums the current row.
constexpr uint32_t kNumSlicePartialTiles = 1U;

// L1 the partial exchange of num_slices cores per row adds to each core; 0 without slicing.
inline uint64_t row_slice_exchange_l1_bytes(const uint32_t num_slices) {
    if (num_slices == 1U) {
        return 0U;
    }
    return static_cast<uint64_t>(kNumSlicePartialTiles + 2U * num_slices) * tt::tile_size(tt::DataFormat::Float32);
}

// Adds the partial exchange's CBs and its semaphore, which starts at 0, to every core of the plan. Nothing without
// slicing.
inline void add_row_slice_exchange(
    tt::tt_metal::ProgramDescriptor& desc,
    const tt::tt_metal::CoreRangeSet& all_cores,
    const uint32_t cb_partial,
    const uint32_t cb_gathered,
    const uint32_t sem_id,
    const uint32_t num_slices) {
    if (num_slices == 1U) {
        return;
    }
    const uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, cb_partial, tt::DataFormat::Float32, float32_single_tile_size_bytes, kNumSlicePartialTiles));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, cb_gathered, tt::DataFormat::Float32, float32_single_tile_size_bytes, 2U * num_slices));
    desc.semaphores.push_back(make_semaphore_descriptor(sem_id, all_cores, 0U));
}

}  // namespace ttml::metal::ops::rmsnorm
