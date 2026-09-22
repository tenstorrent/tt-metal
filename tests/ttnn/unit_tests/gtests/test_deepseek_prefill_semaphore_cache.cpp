// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <gtest/gtest.h>

#include "ttnn_test_fixtures.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/combine_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_device_operation.hpp"

namespace ttnn::test::deepseek_prefill_semaphore_cache {

class DeepseekPrefillSemaphoreCache : public TTNNFixtureWithDevice, public ::testing::WithParamInterface<BufferType> {
public:
    // The default fixture reserves no L1_SMALL space. Exercise both memory pools.
    DeepseekPrefillSemaphoreCache() : TTNNFixtureWithDevice(DEFAULT_TRACE_REGION_SIZE, 1024) {}
};

template <typename Operation>
void check_semaphore_cache_keys(
    tt::tt_metal::distributed::MeshDevice* device,
    BufferType buffer_type,
    typename Operation::operation_attributes_t attributes,
    const typename Operation::tensor_args_t& inputs) {
    const CoreRangeSet cores{CoreRange{CoreCoord{0, 0}}};
    // Keep both allocations alive: freeing A could let B reuse the same address.
    const auto semaphore_a = global_semaphore::create_global_semaphore(device, cores, 0, buffer_type);
    const auto semaphore_b = global_semaphore::create_global_semaphore(device, cores, 0, buffer_type);
    ASSERT_NE(semaphore_a.address(), semaphore_b.address());

    const auto key = [&](const auto& semaphore) {
        attributes.global_semaphore = semaphore;
        // Exercise the actual operation cache-key path, not GlobalSemaphore's generic
        // hash: other operations can legitimately refresh semaphore addresses on hits.
        return device_operation::detail::compute_program_hash<Operation>(attributes, inputs);
    };
    const auto no_semaphore = key(std::nullopt);
    const auto key_a = key(semaphore_a);
    const auto key_b = key(semaphore_b);
    EXPECT_NE(key_a, key_b) << "Different scalar semaphore addresses must not reuse a cached program";
    EXPECT_NE(no_semaphore, key_a);
    EXPECT_NE(no_semaphore, key_b);

    // Copies represent the same allocation. Counter changes must not cause recompilation.
    const auto copy_a = semaphore_a;
    semaphore_a.reset_semaphore_value(17);
    semaphore_b.reset_semaphore_value(31);
    for (const auto* semaphore : std::array{&copy_a, &semaphore_b, &semaphore_a, &semaphore_b}) {
        const auto expected = semaphore->address() == semaphore_a.address() ? key_a : key_b;
        EXPECT_EQ(key(*semaphore), expected);
    }
    EXPECT_EQ(key(std::nullopt), no_semaphore);
}

TEST_P(DeepseekPrefillSemaphoreCache, CombineDistinguishesLiveAllocations) {
    using Operation = operations::experimental::deepseek_prefill::combine::CombineDeviceOperation;
    // The tensors are fixed hash inputs; no compute kernel or model weights are needed.
    const auto tensor = ttnn::create_device_tensor(
        TensorSpec(Shape{1, 1, 32, 32}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig())),
        device_);
    Operation::operation_attributes_t attributes{};
    Operation::tensor_args_t inputs{tensor, tensor, tensor, tensor};
    check_semaphore_cache_keys<Operation>(device_, GetParam(), attributes, inputs);
}

TEST_P(DeepseekPrefillSemaphoreCache, UnifiedExpertDistinguishesLiveAllocations) {
    using Operation =
        operations::experimental::deepseek_prefill::unified_routed_expert_ffn::UnifiedRoutedExpertFfnDeviceOperation;
    const auto tensor = ttnn::create_device_tensor(
        TensorSpec(Shape{1, 1, 32, 32}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig())),
        device_);
    Operation::operation_attributes_t attributes{};
    Operation::tensor_args_t inputs{
        .x = tensor,
        .gate_proj = tensor,
        .up_proj = tensor,
        .down_proj = tensor,
        .counts = tensor,
        .global_expert_idx_table = tensor};
    check_semaphore_cache_keys<Operation>(device_, GetParam(), attributes, inputs);
}

INSTANTIATE_TEST_SUITE_P(
    MemoryPools, DeepseekPrefillSemaphoreCache, ::testing::Values(BufferType::L1, BufferType::L1_SMALL));

}  // namespace ttnn::test::deepseek_prefill_semaphore_cache
