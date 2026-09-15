// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_cyclic_sdpa_bw_program_factory.hpp"

#include <vector>

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/cyclic_sdpa_bw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_bw {

namespace {

namespace cyclic = ttml::metal::ops::cyclic_sdpa_bw::device;

// What this chip computes at this ring step, and with which schedule.
//
// The decision is the same one ring_sdpa_bw makes, from the same helper, so
// the two implementations skip exactly the same (chip, step) pairs and a
// comparison between them is not quietly comparing different work. What
// differs is only what an executing chip runs: the causal schedule on the
// diagonal chunk pair, the dense one on an earlier chunk.
struct StepPlan {
    bool execute{};
    AttentionMaskType mask_type{AttentionMaskType::Causal};
    // Zigzag only: the chunk pairs this chip runs, as slices of one program.
    uint32_t sequence_chunks{1U};
    std::vector<uint32_t> row_chunks{};
    std::vector<uint32_t> col_chunks{};
};

StepPlan plan_for(const operation_attributes_t& args, uint32_t device_ring_id) {
    if (args.layout == ops::RingLayout::Zigzag) {
        // Two chunks per chip, two live pairs a step, no chip idle. The
        // launch's mask type says which pairs: the triangles or the blocks.
        auto sub = ops::zigzag_sub_problems(
            device_ring_id, args.step, args.ring_size, args.mask_type, args.ring_direction);
        if (args.zigzag_pair != operation_attributes_t::kAllPairs) {
            if (args.zigzag_pair >= sub.row_chunks.size()) {
                return {false, args.mask_type, 2U, {}, {}};
            }
            return {
                true,
                args.mask_type,
                2U,
                {sub.row_chunks[args.zigzag_pair]},
                {sub.col_chunks[args.zigzag_pair]}};
        }
        return {sub.execute, args.mask_type, 2U, std::move(sub.row_chunks), std::move(sub.col_chunks)};
    }
    const auto [should_execute, effective_mask_type] = ops::get_device_execution_info(
        device_ring_id, args.step, args.ring_size, args.mask_type, args.ring_direction);
    return {should_execute, effective_mask_type, 1U, {}, {}};
}

cyclic::operation_attributes_t cyclic_attrs(const operation_attributes_t& args, const StepPlan& plan) {
    return cyclic::operation_attributes_t{
        .rows_per_block_tiles = args.rows_per_block_tiles,
        .mask_type = plan.mask_type,
        .use_barrier = args.use_barrier,
        .accumulate_into_outputs = args.accumulate_into_outputs,
        .sequence_chunks = plan.sequence_chunks,
        .row_chunks = plan.row_chunks,
        .col_chunks = plan.col_chunks};
}

cyclic::tensor_args_t cyclic_tensors(const tensor_args_t& t, tensor_return_value_t& out) {
    return cyclic::tensor_args_t{
        .query = t.query,
        .key = t.key,
        .value = t.value,
        .grad_output = t.grad_output,
        .log_sum_exp = t.log_sum_exp,
        .row_scalar = t.row_scalar,
        .preallocated_grad_query = out[0],
        .preallocated_grad_key = out[1],
        .preallocated_grad_value = out[2]};
}

}  // namespace

RingCyclicSDPABackwardProgramFactory::cached_mesh_workload_t
RingCyclicSDPABackwardProgramFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* mesh_device = tensor_args.query.device();
    const auto mesh_shape = mesh_device->shape();

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& mesh_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const uint32_t device_ring_id = mesh_coord[args.ring_axis];
        const auto plan = plan_for(args, device_ring_id);
        if (!plan.execute) {
            // No program at all, not an empty one: a chip that receives
            // kernels but no runtime arguments waits forever on semaphores
            // nobody posts to.
            continue;
        }

        auto attrs = cyclic_attrs(args, plan);
        auto tensors = cyclic_tensors(tensor_args, tensor_return_value);
        auto cached_program =
            cyclic::CyclicSDPABackwardProgramFactory::create(attrs, tensors, tensor_return_value);

        ttnn::MeshCoordinateRange single_coord_range{mesh_coord};
        shared_vars[single_coord_range] = std::move(cached_program.shared_variables);
        mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
    }

    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingCyclicSDPABackwardProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& shared = cached_workload.shared_variables.at(coord_range);
        const uint32_t device_ring_id = coord_range.start_coord()[args.ring_axis];
        const auto plan = plan_for(args, device_ring_id);

        auto attrs = cyclic_attrs(args, plan);
        auto tensors = cyclic_tensors(tensor_args, tensor_return_value);
        auto proxy = cyclic::CyclicSDPABackwardProgramFactory::cached_program_t::proxy(program, shared);
        cyclic::CyclicSDPABackwardProgramFactory::override_runtime_arguments(
            proxy, attrs, tensors, tensor_return_value);
    }
}

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_bw
