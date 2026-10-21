// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED: the table no longer holds only KV chunks, so its types dropped the Kv prefix:
//
//   #include <internal/disaggregation/cache_address_table.hpp>
//   KvChunkAddressTable -> CacheAddressTable, KvCacheLocation -> CacheLocation, ...
//
// This header keeps the old names as deprecated aliases so existing consumers keep compiling
// while they migrate. It will be removed once tt-llm-engine, tt-blaze and tt-d-gen are scrubbed.

#include <internal/disaggregation/cache_address_table.hpp>

namespace tt::tt_metal::internal::disaggregation {

using KvCacheLocation [[deprecated("renamed to CacheLocation; include <internal/disaggregation/cache_address_table.hpp>")]] =
    CacheLocation;

using KvChunkTensorType
    [[deprecated("renamed to CacheTensorType; include <internal/disaggregation/cache_address_table.hpp>")]] =
        CacheTensorType;

using KvChunkAddressTableConfig
    [[deprecated("renamed to CacheAddressTableConfig; include <internal/disaggregation/cache_address_table.hpp>")]] =
        CacheAddressTableConfig;

using KvChunkAddressTable
    [[deprecated("renamed to CacheAddressTable; include <internal/disaggregation/cache_address_table.hpp>")]] =
        CacheAddressTable;

}  // namespace tt::tt_metal::internal::disaggregation
