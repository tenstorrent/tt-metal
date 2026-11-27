// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/kv_chunk_table_cache.hpp>

#include <filesystem>
#include <random>
#include <system_error>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "common/stable_hash.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/build.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::internal::disaggregation {

namespace {

namespace fs = std::filesystem;

// Files are written under a unique temporary name and renamed, so a reader never sees a partial table.
fs::path temp_sibling(const fs::path& path) {
    std::random_device rd;
    return fmt::format("{}.tmp.{:08x}{:08x}", path.string(), rd(), rd());
}

bool copy_atomically(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    const fs::path tmp = temp_sibling(to);
    fs::copy_file(from, tmp, fs::copy_options::overwrite_existing, ec);
    if (!ec) {
        fs::rename(tmp, to, ec);
    }
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

void write_atomically(const std::function<void(const std::string&)>& write_table, const fs::path& out) {
    const fs::path tmp = temp_sibling(out);
    try {
        write_table(tmp.string());
        fs::rename(tmp, out);
    } catch (...) {
        std::error_code ignored;
        fs::remove(tmp, ignored);
        throw;
    }
}

}  // namespace

std::string default_kv_chunk_table_cache_dir() {
    const std::string root = MetalContext::instance_exists() ? get_cache_root(MetalContext::instance().rtoptions())
                                                             : get_cache_root(llrt::RunTimeOptions{});
    return (fs::path(root) / "kv-chunk-tables").string();
}

std::string kv_chunk_table_cache_path(const std::string& cache_dir, const std::string& seed, const std::string& key) {
    tt::StableHasher hasher;
    for (const std::string& part : {seed, key}) {
        hasher.update(static_cast<uint64_t>(part.size()));
        hasher.update(part);
    }
    return (fs::path(cache_dir) / fmt::format("{:016x}.pb", hasher.digest())).string();
}

bool get_or_build_kv_chunk_table(
    const std::string& cache_dir,
    const std::string& seed,
    const std::string& key,
    const std::function<void(const std::string& path)>& write_table,
    const std::string& out_path) {
    if (seed.empty()) {
        write_atomically(write_table, out_path);
        return false;
    }
    const fs::path cached = kv_chunk_table_cache_path(cache_dir, seed, key);
    std::error_code ec;
    if (fs::is_regular_file(cached, ec) && copy_atomically(cached, out_path)) {
        log_info(tt::LogMetal, "KV chunk table cache hit: {}", cached.string());
        return true;
    }
    write_atomically(write_table, out_path);
    fs::create_directories(cached.parent_path(), ec);
    if (ec || !copy_atomically(out_path, cached)) {
        log_warning(tt::LogMetal, "Could not store KV chunk table in cache {}", cached.string());
    }
    return false;
}

}  // namespace tt::tt_metal::internal::disaggregation
