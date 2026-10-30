// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string_view>
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

// Every core sends its partial to every other core of its row-group, so the exchange grows as num_slices^2.
constexpr uint32_t kMaxRowSlices = 16U;
// Per-row latency of the partial exchange, in tiles of per-tile work. Each core also sums num_slices partials.
constexpr uint32_t kRowSliceExchangeCostTiles = 4U;

/**
 * How rmsnorm_fw and rmsnorm_bw split tile-rows over the cores. When whole rows leave cores idle (e.g. fewer rows than
 * cores), each row is cut into num_slices column slices of slice_tiles tiles, handled by the num_slices cores of one
 * row-group, which exchange their partial row sums. num_slices = 1 is one whole row per work item, the
 * split_work_to_cores split.
 *
 * Core i of the walk {i / num_cores_y, i % num_cores_y} handles slice i % num_slices of the rows of row-group
 * i / num_slices. Row-groups [0, num_groups_1) take rows_per_group_1 rows each, the rest rows_per_group_2;
 * core_group_1 and core_group_2 hold their cores, so each group can keep its own compute kernel.
 */
struct RowSlicePlan {
    uint32_t num_slices = 1U;
    uint32_t slice_tiles = 0U;
    uint32_t num_groups_1 = 0U;
    uint32_t rows_per_group_1 = 0U;
    uint32_t rows_per_group_2 = 0U;
    uint32_t num_cores = 0U;
    uint32_t num_cores_y = 0U;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
};

/**
 * Picks num_slices by the work of the busiest core: rounds of rows times the tiles per row, plus the exchange.
 * Only slice counts up to max_slices that divide Wt and whose slice `slice_fits_in_l1(slice_tiles, num_slices)`
 * are considered; ties keep the smaller count.
 */
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

    const uint32_t num_cores_1 = plan.num_groups_1 * num_slices;
    plan.all_cores = tt::tt_metal::num_cores_to_corerangeset(plan.num_cores, grid, /* row_wise */ false);
    plan.core_group_1 = tt::tt_metal::num_cores_to_corerangeset(num_cores_1, grid, /* row_wise */ false);
    if (num_cores_1 < plan.num_cores) {
        plan.core_group_2 = tt::tt_metal::num_cores_to_corerangeset(
            tt::tt_metal::CoreCoord{num_cores_1 / grid.y, num_cores_1 % grid.y},
            plan.num_cores - num_cores_1,
            grid,
            /* row_wise */ false);
    }
    return plan;
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

// The writer's exchange arguments: the core's slice, then the NOC x and y of each core of its row-group in slice
// order. Empty without slicing.
inline std::vector<uint32_t> row_slice_exchange_args(
    const RowSlicePlan& plan, const RowSliceWork& work, const tt::tt_metal::IDevice& device) {
    if (plan.num_slices == 1U) {
        return {};
    }
    std::vector<uint32_t> args{work.slice};
    args.reserve(1U + 2U * plan.num_slices);
    for (uint32_t slice = 0; slice < plan.num_slices; ++slice) {
        const uint32_t peer = work.group * plan.num_slices + slice;
        const auto noc = device.worker_core_from_logical_core({peer / plan.num_cores_y, peer % plan.num_cores_y});
        args.push_back(noc.x);
        args.push_back(noc.y);
    }
    return args;
}

}  // namespace ttml::metal::ops::rmsnorm
