// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/path_roots.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <system_error>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "jit_build/build.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::jit_build {

namespace {

// True when `path` is `prefix` or lies under it.
bool is_under(std::string_view path, std::string_view prefix) {
    return path.starts_with(prefix) && (path.size() == prefix.size() || path[prefix.size()] == '/');
}

// Whether the cache in `cache_dir` is relocatable, marking it first when the caller asks for that.
bool cache_is_relocatable(const tt::llrt::RunTimeOptions& rtoptions, const std::filesystem::path& cache_dir) {
    const std::optional<bool> requested = rtoptions.get_jit_relocatable_cache();
    if (requested == false) {
        return false;
    }
    const std::filesystem::path marker = cache_dir / RELOCATABLE_CACHE_MARKER;
    std::error_code ec;
    if (std::filesystem::exists(marker, ec)) {
        return true;
    }
    if (requested != true) {
        return false;
    }
    std::filesystem::create_directories(cache_dir, ec);
    std::ofstream(marker) << "Paths in this cache are recorded relative to named roots.\n";
    if (!std::filesystem::exists(marker, ec)) {
        log_warning(
            tt::LogBuildKernels,
            "Cannot mark {} as a relocatable JIT cache; using absolute paths.",
            cache_dir.string());
        return false;
    }
    return true;
}

}  // namespace

void PathRoots::add(std::string_view name, std::string_view path) {
    while (path.size() > 1 && path.ends_with('/')) {
        path.remove_suffix(1);
    }
    if (name.empty() || !path.starts_with('/') || path == "/") {
        log_warning(
            tt::LogBuildKernels, "Ignoring JIT path root '{}={}': need a name and an absolute path.", name, path);
        return;
    }
    Root root{.token = fmt::format("${{{}}}", name), .path = std::string(path)};
    const bool defined = std::ranges::any_of(
        roots_, [&](const Root& other) { return other.token == root.token || other.path == root.path; });
    if (defined) {
        return;
    }
    const auto longer = [&](const Root& other) { return other.path.size() >= root.path.size(); };
    roots_.insert(std::ranges::find_if_not(roots_, longer), std::move(root));
}

std::string PathRoots::relocatable(std::string_view path) const {
    for (const Root& root : roots_) {
        if (is_under(path, root.path)) {
            return root.token + std::string(path.substr(root.path.size()));
        }
    }
    return std::string(path);
}

std::string PathRoots::resolve(std::string_view path) const {
    for (const Root& root : roots_) {
        if (is_under(path, root.token)) {
            return root.path + std::string(path.substr(root.token.size()));
        }
    }
    return std::string(path);
}

std::string PathRoots::relocatable_text(std::string_view text) const {
    if (roots_.empty()) {
        return std::string(text);
    }
    std::string out;
    out.reserve(text.size());
    while (!text.empty()) {
        const size_t end = std::min(text.find_first_of(" \t\n"), text.size());
        const std::string_view token = text.substr(0, end);
        // A token's path starts at its first '/': "-I/a/b", "-Wl,--just-symbols=/a/b.elf", "/a/b".
        const size_t slash = std::min(token.find('/'), token.size());
        out.append(token.substr(0, slash));
        out.append(relocatable(token.substr(slash)));
        out.append(text.substr(end, 1));
        text.remove_prefix(std::min(end + 1, text.size()));
    }
    return out;
}

PathRoots PathRoots::from_rtoptions(const tt::llrt::RunTimeOptions& rtoptions) {
    PathRoots roots;
    // The cache root is a string prefix ("<dir>/tt-metal-cache" + "<build_key>/..."), so the
    // directory that holds a whole cache is its parent.
    const std::filesystem::path cache_dir =
        std::filesystem::absolute(std::filesystem::path(tt::tt_metal::get_cache_root(rtoptions)).parent_path());
    if (!cache_is_relocatable(rtoptions, cache_dir)) {
        return roots;
    }
    roots.add("CACHE", cache_dir.string());
    roots.add("ROOT", rtoptions.get_root_dir());
    // TT_METAL_JIT_PATH_ROOTS: "NAME=/abs/dir[:NAME2=/abs/dir2...]".
    std::string_view user_roots = rtoptions.get_jit_path_roots();
    while (!user_roots.empty()) {
        const std::string_view entry = user_roots.substr(0, user_roots.find(':'));
        user_roots.remove_prefix(std::min(entry.size() + 1, user_roots.size()));
        const size_t eq = std::min(entry.find('='), entry.size());
        roots.add(entry.substr(0, eq), entry.substr(std::min(eq + 1, entry.size())));
    }
    return roots;
}

}  // namespace tt::jit_build
