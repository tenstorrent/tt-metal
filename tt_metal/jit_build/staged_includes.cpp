// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "staged_includes.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "common/stable_hash.hpp"
#include "jit_build_utils.hpp"

namespace tt::jit_build {

namespace fs = std::filesystem;

std::string stage_observability_api(const fs::path& root, const fs::path& cache_root) {
    static const std::array<std::string, 2> extensions = {".h", ".hpp"};
    const fs::path staged_subdir = "observability/api";

    // The host's headers, then the device's, so a device header replaces the host header of the same name.
    std::map<std::string, std::vector<std::uint8_t>> headers;
    for (const std::string_view dir : {OBSERVABILITY_API_HOST_DIR, OBSERVABILITY_API_DEVICE_DIR}) {
        for (auto& file : utils::read_directory_files((root / dir).string(), extensions)) {
            headers[file.name] = std::move(file.content);
        }
    }
    TT_FATAL(
        !headers.empty(), "No observability/api headers to stage in {}", (root / OBSERVABILITY_API_HOST_DIR).string());

    // Keyed by content, so a staged folder never changes once published: processes sharing the cache stage the same
    // bytes, and an edited header gets a new folder, whose path changes the compile's include flags and with them the
    // build state that every cached kernel and firmware is checked against.
    tt::StableHasher hasher;
    for (const auto& [name, content] : headers) {
        hasher.update(name);
        hasher.update(content.size());
        hasher.update(content.data(), content.size());
    }
    const fs::path dir = fs::absolute(cache_root / "include" / fmt::format("{:016x}", hasher.digest()));

    // Staged at most once per directory per process; a folder another process published is complete, since it
    // appears only by the rename below.
    static std::mutex mutex;
    static std::unordered_set<std::string> staged;
    std::lock_guard lock(mutex);
    if (staged.contains(dir.string()) || fs::is_directory(dir)) {
        staged.insert(dir.string());
        return dir.string();
    }

    // Written under a private name and renamed into place, so no compile sees a partly written folder.
    const fs::path temp = utils::FileRenamer::generate_temp_path(dir);
    fs::create_directories(temp / staged_subdir);
    for (const auto& [name, content] : headers) {
        const fs::path path = temp / staged_subdir / name;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(content.data()), static_cast<std::streamsize>(content.size()));
        TT_FATAL(out.good(), "Cannot write {}", path.string());
    }
    std::error_code ec;
    fs::rename(temp, dir, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove_all(temp, ignored);
        // Another process published the same folder first.
        TT_FATAL(fs::is_directory(dir), "Cannot publish {}: {}", dir.string(), ec.message());
    }
    staged.insert(dir.string());
    return dir.string();
}

}  // namespace tt::jit_build
