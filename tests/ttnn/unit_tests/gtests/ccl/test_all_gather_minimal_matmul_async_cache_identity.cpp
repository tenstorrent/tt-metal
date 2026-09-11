// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <tt_stl/reflection.hpp>
#include <tt-metalium/global_semaphore.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation_types.hpp"

namespace {

using ttnn::experimental::prim::AllGatherMinimalMatmulAsyncParams;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

// Hash the reflected program-cache identity of an AGMM op whose tensor specs and every other listed
// attribute are fixed, varying only the three compile-affecting fields under test. Only the fields
// in attribute_values() participate, so the semaphores and the (unused) barrier reference never
// enter the hash.
ttsl::hash::hash_t hash_agmm_identity(
    std::optional<UnaryWithParam> fused_activation,
    std::optional<tt::tt_metal::DataType> output_dtype,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    // barrier_semaphore is a const-ref member; keep a named optional alive for the params' lifetime.
    const std::optional<tt::tt_metal::GlobalSemaphore> no_barrier;
    const AllGatherMinimalMatmulAsyncParams params(
        /*config=*/std::nullopt,
        /*fused_activation=*/std::move(fused_activation),
        /*output_mem_config=*/std::nullopt,
        /*output_dtype=*/output_dtype,
        /*compute_kernel_config=*/compute_kernel_config,
        /*num_links=*/1,
        /*ring_size=*/8,
        /*topology=*/ttnn::ccl::Topology::Ring,
        /*semaphore=*/{},
        /*cluster_axis=*/std::optional<uint32_t>{1},
        /*barrier_semaphore=*/no_barrier,
        /*using_persistent_buffers=*/false,
        /*force_transpose=*/false,
        /*num_workers_per_link=*/4,
        /*num_buffers_per_channel=*/1,
        /*fused_ternary_scalar=*/std::nullopt,
        /*chunks=*/1,
        /*dim=*/-1,
        /*fsdp_cluster_axis=*/std::nullopt,
        /*fsdp_ring_size=*/1,
        /*fsdp_semaphore=*/{},
        /*using_persistent_weight_buffer=*/false,
        /*fsdp_topology=*/ttnn::ccl::Topology::Linear,
        /*fuse_swiglu=*/false);
    return ttsl::hash::hash_objects_with_default_seed(params);
}

}  // namespace

// fused_activation, output_dtype, and compute_kernel_config each change the compiled program but not
// the tensor args, so a cache hit cannot repair them. They must contribute to program-cache identity.
TEST(AllGatherMinimalMatmulAsync, CompileAffectingAttributesHaveDistinctProgramCacheIdentity) {
    const auto baseline = hash_agmm_identity(std::nullopt, std::nullopt, ttnn::DeviceComputeKernelConfig{});

    EXPECT_NE(
        hash_agmm_identity(UnaryWithParam{UnaryOpType::EXP}, std::nullopt, ttnn::DeviceComputeKernelConfig{}),
        baseline);

    EXPECT_NE(
        hash_agmm_identity(std::nullopt, tt::tt_metal::DataType::BFLOAT16, ttnn::DeviceComputeKernelConfig{}),
        baseline);

    auto changed_compute_config = ttnn::DeviceComputeKernelConfig{};
    changed_compute_config.math_approx_mode = !changed_compute_config.math_approx_mode;
    EXPECT_NE(hash_agmm_identity(std::nullopt, std::nullopt, changed_compute_config), baseline);
}
