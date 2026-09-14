// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include <tt_stl/reflection.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation_types.hpp"

namespace {

std::size_t hash_attributes(
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<tt::tt_metal::DataType> output_dtype,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    ttnn::ccl::Topology fsdp_topology = ttnn::ccl::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<uint32_t> fsdp_cluster_axis = std::nullopt) {
    const std::optional<ttnn::GlobalSemaphore> barrier_semaphore = std::nullopt;
    const ttnn::experimental::prim::AllGatherMinimalMatmulAsyncParams attributes{
        /*config=*/std::nullopt,
        std::move(fused_activation),
        /*output_mem_config=*/std::nullopt,
        output_dtype,
        compute_kernel_config,
        /*num_links=*/1,
        /*ring_size=*/2,
        ttnn::ccl::Topology::Ring,
        /*semaphore=*/{},
        cluster_axis,
        barrier_semaphore,
        /*using_persistent_buffers=*/false,
        /*force_transpose=*/false,
        /*num_workers_per_link=*/1,
        /*num_buffers_per_channel=*/1,
        fused_ternary_scalar,
        /*chunks=*/1,
        /*dim=*/-1,
        fsdp_cluster_axis,
        /*fsdp_ring_size=*/1,
        /*fsdp_semaphore=*/{},
        /*using_persistent_weight_buffer=*/false,
        fsdp_topology};
    return ttsl::hash::hash_objects_with_default_seed(attributes);
}

TEST(AllGatherMinimalMatmulAsync, CompileAffectingAttributesHaveDistinctProgramCacheIdentity) {
    const ttnn::DeviceComputeKernelConfig default_compute_config{};
    const auto baseline_hash = hash_attributes(std::nullopt, std::nullopt, default_compute_config);

    EXPECT_NE(
        hash_attributes(
            ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::EXP},
            std::nullopt,
            default_compute_config),
        baseline_hash);
    EXPECT_NE(hash_attributes(std::nullopt, tt::tt_metal::DataType::BFLOAT16, default_compute_config), baseline_hash);

    const auto bfloat16_hash = hash_attributes(std::nullopt, tt::tt_metal::DataType::BFLOAT16, default_compute_config);
    EXPECT_NE(hash_attributes(std::nullopt, tt::tt_metal::DataType::FLOAT32, default_compute_config), bfloat16_hash);

    auto changed_compute_config = default_compute_config;
    changed_compute_config.math_approx_mode = !changed_compute_config.math_approx_mode;
    EXPECT_NE(hash_attributes(std::nullopt, std::nullopt, changed_compute_config), baseline_hash);

    const auto positive_zero_hash = hash_attributes(std::nullopt, std::nullopt, default_compute_config, +0.0F);
    EXPECT_NE(positive_zero_hash, baseline_hash);
    EXPECT_NE(hash_attributes(std::nullopt, std::nullopt, default_compute_config, -0.0F), positive_zero_hash);
    EXPECT_NE(hash_attributes(std::nullopt, std::nullopt, default_compute_config, 1.0F), positive_zero_hash);

    EXPECT_NE(
        hash_attributes(std::nullopt, std::nullopt, default_compute_config, std::nullopt, ttnn::ccl::Topology::Linear),
        baseline_hash);
    EXPECT_NE(
        hash_attributes(
            std::nullopt,
            std::nullopt,
            default_compute_config,
            std::nullopt,
            ttnn::ccl::Topology::Ring,
            /*cluster_axis=*/0),
        baseline_hash);
    EXPECT_NE(
        hash_attributes(
            std::nullopt,
            std::nullopt,
            default_compute_config,
            std::nullopt,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            /*fsdp_cluster_axis=*/0),
        baseline_hash);
}

}  // namespace
