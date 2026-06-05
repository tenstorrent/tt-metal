// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_env_mock_ccl_test_helper.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/shape.hpp>

#include <ttnn/graph/graph_query_op_constraints.hpp>
#include <ttnn/operations/ccl/all_gather/all_gather.hpp>
#include <ttnn/tensor/layout/page_config.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/types.hpp>

#include <optional>

namespace tt::tt_metal::test_support {
namespace {

ttnn::TensorSpec make_all_gather_input_spec() {
    return ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
}

}  // namespace

OpConstraintQueryResult run_all_gather_constraint_query(distributed::MeshDevice* device) {
    auto sharded_topology = TensorTopology::create_sharded_tensor_topology(device->shape(), /*shard_dim=*/0);
    ttnn::graph::DistributedTensorSpec dist_input{make_all_gather_input_spec(), sharded_topology};

    auto response = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::all_gather(std::forward<decltype(args)>(args)...); },
        device,
        dist_input,
        /*dim=*/3,
        /*cluster_axis=*/std::optional<uint32_t>(1),
        /*subdevice_id=*/std::optional<SubDeviceId>{},
        /*memory_config=*/std::optional<MemoryConfig>{},
        /*optional_output_tensor=*/std::optional<::ttnn::Tensor>{},
        /*num_links=*/std::optional<uint32_t>(1),
        /*topology=*/std::optional<tt_fabric::Topology>(tt_fabric::Topology::Linear));

    return OpConstraintQueryResult{
        .success = response.status == ttnn::graph::ExecutionStatus::Success,
        .error_message = response.error_message,
    };
}

}  // namespace tt::tt_metal::test_support
