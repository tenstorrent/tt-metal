// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Off-device kernel prewarm driver.
//
// Compiles every kernel recipe recorded in the prewarm manifest into the JIT cache without opening a
// device. Run this before reserving a device (e.g. before a broker/mcp job) so the reservation holds
// the device only for device work -- host-side kernel compilation happens here instead of inside the
// device-held window.
//
// The manifest is self-contained and build_key-tagged, so no device or config query is needed. A cold
// cache with no manifest is a no-op (the first on-device run captures it); every run after -- including
// cache-invalidating kernel/header edits -- warms here.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/program/kernel_prewarm.hpp"

namespace fs = std::filesystem;

namespace {

// Mirror JitBuildEnv::out_root_ resolution (jit_build/build.cpp:109 with rtoptions normalize_path and
// get_default_root_path): TT_METAL_CACHE -> path(value)/"tt-metal-cache" (lexically normalized, no
// trailing slash); else $HOME/.cache/tt-metal-cache/ (trailing slash); else /tmp. Kept in sync by hand
// -- a mismatch warms the wrong tree and leaves the device-held run cold.
std::string resolve_out_root(const char* explicit_cache_dir) {
    const char* c = (explicit_cache_dir != nullptr && explicit_cache_dir[0] != '\0') ? explicit_cache_dir
                                                                                     : std::getenv("TT_METAL_CACHE");
    if (c != nullptr && c[0] != '\0') {
        return (fs::path(c) / "tt-metal-cache").lexically_normal().string();
    }
    if (const char* h = std::getenv("HOME"); h != nullptr && h[0] != '\0' && fs::exists(h)) {
        return std::string(h) + "/.cache/tt-metal-cache/";
    }
    return "/tmp/tt-metal-cache/";
}

// Mirror RunTimeOptions root_dir (used to find the pre-compiled firmware bundle): TT_METAL_RUNTIME_ROOT,
// else TT_METAL_HOME (set by run envs), else CWD when it looks like a metal checkout. Trailing slash so
// find_precompiled_dir's "<root>tt_metal/pre-compiled/<bk>/" concatenation matches. Empty when unknown
// -- prewarm_manifest_offline then uses the jit firmware subtree.
std::string resolve_root_dir() {
    auto with_slash = [](std::string s) {
        if (!s.empty() && s.back() != '/') {
            s.push_back('/');
        }
        return s;
    };
    if (const char* r = std::getenv("TT_METAL_RUNTIME_ROOT"); r != nullptr && r[0] != '\0') {
        return with_slash(r);
    }
    if (const char* h = std::getenv("TT_METAL_HOME"); h != nullptr && h[0] != '\0') {
        return with_slash(h);
    }
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec && fs::is_directory(cwd / "tt_metal", ec)) {
        return with_slash(cwd.string());
    }
    return "";
}

}  // namespace

int main(int argc, char** argv) {
    // Optional arg: the TT_METAL_CACHE value (user-facing cache dir). Defaults to $TT_METAL_CACHE, then
    // the default cache root -- matching how the run this warms for resolves its cache.
    const char* arg = argc > 1 ? argv[1] : nullptr;
    const std::string out_root = resolve_out_root(arg);
    const std::string root_dir = resolve_root_dir();

    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t built = tt::tt_metal::kernel_prewarm::prewarm_manifest_offline(out_root, root_dir);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    log_info(tt::LogMetal, "kernel prewarm (offline): {} targets built in {}ms (cache {})", built, ms, out_root);
    return 0;
}
