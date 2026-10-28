// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DEPRECATED: renamed to <internal/disaggregation/cache_address_table_cache.hpp>. These forwarders keep
// the old names compiling while consumers migrate.

#include <functional>
#include <string>

#include <internal/disaggregation/cache_address_table_cache.hpp>

namespace tt::tt_metal::internal::disaggregation {

[[deprecated("renamed to default_cache_address_table_cache_dir")]] inline std::string
default_kv_chunk_table_cache_dir() {
    return default_cache_address_table_cache_dir();
}

[[deprecated("renamed to cache_address_table_cache_path")]] inline std::string kv_chunk_table_cache_path(
    const std::string& cache_dir, const std::string& seed, const std::string& key) {
    return cache_address_table_cache_path(cache_dir, seed, key);
}

[[deprecated("renamed to get_or_build_cache_address_table")]] inline bool get_or_build_kv_chunk_table(
    const std::string& cache_dir,
    const std::string& seed,
    const std::string& key,
    const std::function<void(const std::string& path)>& write_table,
    const std::string& out_path) {
    return get_or_build_cache_address_table(cache_dir, seed, key, write_table, out_path);
}

}  // namespace tt::tt_metal::internal::disaggregation
