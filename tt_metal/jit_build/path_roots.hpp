// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::jit_build {

// Marks a JIT cache directory as relocatable (see PathRoots::from_rtoptions).
inline constexpr std::string_view RELOCATABLE_CACHE_MARKER = ".tt-metal-jit-relocatable";

// Named absolute directories ("roots") that keep JIT cache keys and dependency records independent
// of where a tree is checked out, like ccache's base_dir.
//
// A path under a root is written as "${NAME}/rest" wherever it enters a cache key (the build key,
// Kernel::compute_hash, the ".build_state" recipe stamp) or a dependency record (".dephash",
// ".fulldephash"), and is resolved against the reader's own roots. A cache written with the roots in
// one place then validates with them in another, as long as the file contents match. A path under
// no root stays absolute. Paths are compared as strings, so a file reached through another spelling
// of its root (a symlink) is not under it.
//
// An instance with no roots leaves every path as it is, which is the default.
class PathRoots {
public:
    // The roots of a relocatable cache: CACHE (the cache directory), ROOT (the tt-metal runtime
    // root) and those named in TT_METAL_JIT_PATH_ROOTS. JitBuildEnv adds SFPI (the toolchain).
    // Empty unless the cache directory is relocatable, which TT_METAL_JIT_RELOCATABLE_CACHE=1 makes
    // it by writing RELOCATABLE_CACHE_MARKER there. A marked cache is relocatable for every later
    // process unless that process sets TT_METAL_JIT_RELOCATABLE_CACHE=0.
    static PathRoots from_rtoptions(const tt::llrt::RunTimeOptions& rtoptions);

    // The first root added with a given name or path wins.
    void add(std::string_view name, std::string_view path);
    bool empty() const { return roots_.empty(); }

    // "/root/a/b" -> "${NAME}/a/b". The longest root that holds the path wins.
    std::string relocatable(std::string_view path) const;
    // Inverse of relocatable().
    std::string resolve(std::string_view path) const;
    // relocatable() applied to the path in each whitespace-separated token of a flag string:
    // "-I/root/a -T/root/b.ld" -> "-I${NAME}/a -T${NAME}/b.ld".
    std::string relocatable_text(std::string_view text) const;

private:
    struct Root {
        std::string token;  // "${NAME}"
        std::string path;   // absolute, no trailing slash
    };
    std::vector<Root> roots_;  // longest path first
};

}  // namespace tt::jit_build
