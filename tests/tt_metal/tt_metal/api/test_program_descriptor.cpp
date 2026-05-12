// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

namespace tt::tt_metal {
namespace {

// ============================================================================
// emplace_runtime_args — binding invariants
// ============================================================================

// uint32_t-only args must never produce buffer_bindings entries.
TEST(ProgramDescriptor, EmplaceRuntimeArgs_Uint32Only_NoBufferBindings) {
    KernelDescriptor kd;
    kd.emplace_runtime_args({0, 0}, {1u, 2u, 3u});
    kd.emplace_runtime_args({1, 0}, {4u, 5u});

    EXPECT_TRUE(kd.buffer_bindings.empty());
    EXPECT_TRUE(kd.common_buffer_bindings.empty());
}

// An empty arg list must still record a slot for that core.
TEST(ProgramDescriptor, EmplaceRuntimeArgs_EmptyList_StillRecordsSlot) {
    KernelDescriptor kd;
    kd.emplace_runtime_args({0, 0}, {});

    ASSERT_EQ(kd.runtime_args.size(), 1u);
    EXPECT_EQ(kd.runtime_args[0].first, CoreCoord(0, 0));
    EXPECT_TRUE(kd.runtime_args[0].second.empty());
}

// RTArgList::append must concatenate, not nest.
TEST(ProgramDescriptor, RTArgList_Append_ConcatenatesInPlace) {
    KernelDescriptor kd;
    KernelDescriptor::RTArgList args;
    args.push_back(1u);
    args.append({2u, 3u, 4u});
    args.append({});  // appending empty must be a no-op
    kd.emplace_runtime_args({0, 0}, args);

    EXPECT_EQ(kd.runtime_args[0].second, (std::vector<uint32_t>{1u, 2u, 3u, 4u}));
}

// ============================================================================
// merge_program_descriptors
// ============================================================================

TEST(ProgramDescriptor, MergeDescriptors_AccumulatesAllSections) {
    ProgramDescriptor a, b;

    KernelDescriptor ka;
    ka.kernel_source = "reader.cpp";
    ka.core_ranges = CoreRangeSet{CoreRange{{0, 0}, {0, 0}}};
    a.kernels.push_back(ka);
    a.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = CoreRangeSet{CoreRange{{0, 0}}}});

    KernelDescriptor kb;
    kb.kernel_source = "writer.cpp";
    kb.core_ranges = CoreRangeSet{CoreRange{{1, 0}, {1, 0}}};
    b.kernels.push_back(kb);
    b.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = CoreRangeSet{CoreRange{{1, 0}}}});

    ProgramDescriptor result = merge_program_descriptors({a, b});

    ASSERT_EQ(result.kernels.size(), 2u);
    EXPECT_EQ(result.kernels[0].kernel_source, "reader.cpp");
    EXPECT_EQ(result.kernels[1].kernel_source, "writer.cpp");
    EXPECT_EQ(result.semaphores.size(), 2u);
}

TEST(ProgramDescriptor, MergeDescriptors_OverlappingCoreRanges_Throws) {
    ProgramDescriptor a, b;

    KernelDescriptor ka;
    ka.core_ranges = CoreRangeSet{CoreRange{{0, 0}, {1, 1}}};
    a.kernels.push_back(ka);

    KernelDescriptor kb;
    kb.core_ranges = CoreRangeSet{CoreRange{{1, 1}, {2, 2}}};  // overlaps at {1,1}
    b.kernels.push_back(kb);

    EXPECT_ANY_THROW(merge_program_descriptors({a, b}));
}

TEST(ProgramDescriptor, MergeDescriptors_InvalidatesCustomHash) {
    ProgramDescriptor a, b;
    a.custom_program_hash = 0xAAAAAAAAULL;

    KernelDescriptor ka;
    ka.core_ranges = CoreRangeSet{CoreRange{{0, 0}}};
    a.kernels.push_back(ka);

    KernelDescriptor kb;
    kb.core_ranges = CoreRangeSet{CoreRange{{1, 0}}};
    b.kernels.push_back(kb);

    ProgramDescriptor result = merge_program_descriptors({a, b});
    EXPECT_FALSE(result.custom_program_hash.has_value());
}

// ============================================================================
// find_available_semaphore_id
// ============================================================================

TEST(ProgramDescriptor, FindSemaphoreId_RespectsUsedIds) {
    ProgramDescriptor desc;
    // Use IDs 0, 1, and 3 — expect 2 to be returned as the first gap.
    for (uint32_t id : {0u, 1u, 3u}) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id,
            .core_type = CoreType::WORKER,
            .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        });
    }

    auto result = desc.find_available_semaphore_id({0, 0}, CoreType::WORKER);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 2u);
}

TEST(ProgramDescriptor, FindSemaphoreId_IgnoresSemaphoresOnOtherCores) {
    ProgramDescriptor desc;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0, .core_type = CoreType::WORKER, .core_ranges = CoreRangeSet{CoreRange{{1, 0}}},  // different core
    });

    auto result = desc.find_available_semaphore_id({0, 0}, CoreType::WORKER);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
}

TEST(ProgramDescriptor, FindSemaphoreId_IgnoresSemaphoresOfDifferentCoreType) {
    ProgramDescriptor desc;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = CoreType::ETH,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
    });

    auto result = desc.find_available_semaphore_id({0, 0}, CoreType::WORKER);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
}

// ============================================================================
// ProgramDescriptor hash
// ============================================================================

// Hash must be sensitive to kernel content.
TEST(ProgramDescriptor, Hash_DifferentKernelSources_ProduceDifferentHashes) {
    std::hash<ProgramDescriptor> hasher;

    ProgramDescriptor a, b;
    KernelDescriptor ka, kb;
    ka.kernel_source = "reader.cpp";
    ka.core_ranges = CoreRangeSet{CoreRange{{0, 0}}};
    kb.kernel_source = "writer.cpp";
    kb.core_ranges = CoreRangeSet{CoreRange{{0, 0}}};
    a.kernels.push_back(ka);
    b.kernels.push_back(kb);

    EXPECT_NE(hasher(a), hasher(b));
}

// Hash must be deterministic for equal descriptors.
TEST(ProgramDescriptor, Hash_SameContents_ProduceSameHash) {
    std::hash<ProgramDescriptor> hasher;

    ProgramDescriptor a, b;
    KernelDescriptor kd;
    kd.kernel_source = "compute.cpp";
    kd.core_ranges = CoreRangeSet{CoreRange{{0, 0}}};
    kd.compile_time_args = {1u, 2u, 3u};
    a.kernels.push_back(kd);
    b.kernels.push_back(kd);

    EXPECT_EQ(hasher(a), hasher(b));
}

// custom_program_hash must override the computed value regardless of content.
TEST(ProgramDescriptor, Hash_CustomHashAlwaysOverrides) {
    std::hash<ProgramDescriptor> hasher;

    ProgramDescriptor desc;
    KernelDescriptor kd;
    kd.kernel_source = "reader.cpp";
    desc.kernels.push_back(kd);
    const auto computed = hasher(desc);

    desc.custom_program_hash = computed + 1;
    EXPECT_EQ(hasher(desc), computed + 1);
}

// The std::hash specialization must satisfy unordered_map requirements.
TEST(ProgramDescriptor, Hash_UsableAsUnorderedMapKey) {
    std::unordered_map<ProgramDescriptor, int> cache;

    ProgramDescriptor desc;
    KernelDescriptor kd;
    kd.kernel_source = "reader.cpp";
    kd.core_ranges = CoreRangeSet{CoreRange{{0, 0}}};
    desc.kernels.push_back(kd);

    cache[desc] = 42;
    EXPECT_EQ(cache.at(desc), 42);
}

}  // namespace
}  // namespace tt::tt_metal
