// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <type_traits>

#include <internal/disaggregation/kv_chunk_address_table.hpp>
#include <internal/disaggregation/kv_chunk_table_cache.hpp>
#include <tt-metalium/experimental/disaggregation/kv_chunk_address_table.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace tt::tt_metal::internal::disaggregation {
namespace {

static_assert(std::is_same_v<KvChunkAddressTable, CacheAddressTable>);
static_assert(std::is_same_v<KvChunkAddressTableConfig, CacheAddressTableConfig>);
static_assert(std::is_same_v<KvCacheLocation, CacheLocation>);
static_assert(std::is_same_v<KvChunkTensorType, CacheTensorType>);

static_assert(std::is_same_v<experimental::disaggregation::KvChunkAddressTable, CacheAddressTable>);
static_assert(std::is_same_v<experimental::disaggregation::KvChunkAddressTableConfig, CacheAddressTableConfig>);
static_assert(std::is_same_v<experimental::disaggregation::KvCacheLocation, CacheLocation>);

TEST(CacheAddressTableDeprecatedAliases, CPU_OldCacheFunctionsForwardToTheNewOnes) {
    EXPECT_EQ(kv_chunk_table_cache_path("/c", "seed", "key"), cache_address_table_cache_path("/c", "seed", "key"));
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation

#pragma GCC diagnostic pop
