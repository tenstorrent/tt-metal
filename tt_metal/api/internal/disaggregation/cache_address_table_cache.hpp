// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>

namespace tt::tt_metal::internal::disaggregation {

// <tt-metal cache>/kv-chunk-tables, resolved from the live MetalContext's RunTimeOptions, or from the
// environment a new context would read when none exists.
std::string default_cache_address_table_cache_dir();

// Cached KV chunk tables live at <cache_dir>/<hash(seed, key)>.pb. `seed` is the commit of the repo that
// owns the table layout; `key` must cover every table input, including DRAM bases and fabric node ids.
std::string cache_address_table_cache_path(const std::string& cache_dir, const std::string& seed, const std::string& key);

// Writes the table for (seed, key) to `out_path`: copies the cached file on a hit, otherwise calls
// `write_table` and caches the result. An empty seed skips the cache. Returns true on a hit.
bool get_or_build_cache_address_table(
    const std::string& cache_dir,
    const std::string& seed,
    const std::string& key,
    const std::function<void(const std::string& path)>& write_table,
    const std::string& out_path);

}  // namespace tt::tt_metal::internal::disaggregation
