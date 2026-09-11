// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pch.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include "common/stable_hash.hpp"
#include "jit_build_utils.hpp"

namespace tt::jit_build {

namespace fs = std::filesystem;

namespace {

// Source of the umbrella, relative to the tt-metal root. Installed alongside the other
// jit_build inputs, see tt_metal/hw/firmware/CMakeLists.txt.
constexpr std::string_view UMBRELLA_REL_PATH = "tt_metal/hw/firmware/src/pch.h";

}  // namespace

std::string ensure_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::string& root,
    const std::string& pch_root) {
    const fs::path umbrella = fs::path(root) / UMBRELLA_REL_PATH;

    // The umbrella's own text is part of the key, so editing it produces a new artifact
    // instead of silently reusing the one built from the previous contents. Re-reading a
    // ~1 KB file per compile costs nothing next to spawning a compiler.
    std::ifstream in(umbrella, std::ios::binary);
    const std::string umbrella_text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};

    tt::StableHasher hasher;
    hasher.update(gpp);
    hasher.update(opt_level);
    hasher.update(cflags);
    hasher.update(includes);
    hasher.update(umbrella_text);
    const std::string key = fmt::format("{:016x}", hasher.digest());

    // Built at most once per key per process. The map also caches failure (an empty string),
    // so a flag set whose PCH cannot be built is not retried on every compile.
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> staged;
    std::lock_guard lock(mutex);
    if (auto it = staged.find(key); it != staged.end()) {
        return it->second;
    }
    std::string& result = staged[key];

    if (umbrella_text.empty()) {
        log_warning(tt::LogBuildKernels, "Skipping the shared PCH: cannot read {}.", umbrella.string());
        return result;
    }

    const fs::path dir = fs::path(pch_root) / key;
    const fs::path header = dir / umbrella.filename();
    const fs::path gch = header.string() + ".gch";

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        log_warning(tt::LogBuildKernels, "Skipping the shared PCH: cannot create {}: {}", dir.string(), ec.message());
        return result;
    }

    // GCC looks for <header>.gch beside the header it is told to include, so the umbrella is
    // staged into the cache next to the artifact rather than compiled in the source tree.
    //
    // Copied to a private name and renamed into place. Overwriting the staged header directly
    // would truncate it, and a concurrent process could then build its .gch from the truncated
    // text: a PCH that GCC accepts but that holds nothing, so every later compile reparses the
    // standard headers while the cache looks populated.
    const fs::path header_temp = fmt::format("{}.{}.tmp", header.string(), ::getpid());
    fs::copy_file(umbrella, header_temp, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        log_warning(tt::LogBuildKernels, "Skipping the shared PCH: cannot stage {}: {}", header.string(), ec.message());
        return result;
    }
    fs::rename(header_temp, header, ec);
    if (ec) {
        fs::remove(header_temp, ec);
        log_warning(
            tt::LogBuildKernels, "Skipping the shared PCH: cannot publish {}: {}", header.string(), ec.message());
        return result;
    }

    if (fs::exists(gch)) {  // built by an earlier run against this same cache
        result = header.string();
        return result;
    }

    std::vector<std::string> args = utils::tokenize_flags(gpp);
    args.emplace_back("-x");
    args.emplace_back("c++-header");
    args.push_back("-" + opt_level);
    for (std::string& tok : utils::tokenize_flags(cflags)) {
        args.push_back(std::move(tok));
    }
    for (std::string& tok : utils::tokenize_flags(includes)) {
        args.push_back(std::move(tok));
    }
    // -MMD would drop a stray depfile beside the artifact. The umbrella's only dependencies are
    // toolchain headers, and its own text is already part of the key.
    std::erase(args, "-MMD");
    args.push_back(header.string());
    args.emplace_back("-o");

    // Compile to a private name and rename into place, so concurrent processes sharing one
    // cache cannot observe a half-written .gch.
    const fs::path temp = fmt::format("{}.{}.tmp", gch.string(), ::getpid());
    args.push_back(temp.string());

    const std::string log_path = (dir / "build.log").string();
    if (!utils::exec_command(args, dir.string(), log_path)) {
        log_warning(
            tt::LogBuildKernels,
            "Skipping the shared PCH: {} failed, see {}. Compiles will parse the standard headers "
            "as before.",
            fmt::join(args, " "),
            log_path);
        fs::remove(temp, ec);
        return result;
    }
    fs::rename(temp, gch, ec);
    if (ec) {
        fs::remove(temp, ec);
        log_warning(tt::LogBuildKernels, "Skipping the shared PCH: cannot publish {}: {}", gch.string(), ec.message());
        return result;
    }

    result = header.string();
    return result;
}

}  // namespace tt::jit_build
