// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pch.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include "common/stable_hash.hpp"
#include "depend.hpp"
#include "jit_build_utils.hpp"

namespace tt::jit_build {

namespace fs = std::filesystem;

std::string ensure_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const fs::path& umbrella,
    const fs::path& pch_root) {
    // The umbrella's own text is part of the key, so editing it produces a new artifact
    // instead of silently reusing the one built from the previous contents. Re-reading a
    // ~1 KB file per compile costs nothing next to spawning a compiler.
    const auto umbrella_bytes = utils::read_file_bytes(umbrella.string());

    tt::StableHasher hasher;
    hasher.update(gpp);
    hasher.update(utils::compiler_version(gpp));
    hasher.update(opt_level);
    hasher.update(cflags);
    hasher.update(umbrella_bytes.data(), umbrella_bytes.size());
    const std::string key = fmt::format("{:016x}", hasher.digest());
    const fs::path dir = fs::absolute(pch_root / key);

    // A server handles multiple cache roots in one process; keep their staged paths
    // separate even when the compiler flags and umbrella text match.
    // Built at most once per directory per process. The map also caches failure (an empty string),
    // so a flag set whose PCH cannot be built is not retried on every compile.
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> staged;
    std::lock_guard lock(mutex);
    if (auto it = staged.find(dir.string()); it != staged.end()) {
        return it->second;
    }
    std::string& result = staged[dir.string()];

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
    const fs::path header_temp = utils::FileRenamer::generate_temp_path(header);
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
    // -MMD would drop a stray depfile beside the artifact. The umbrella's only dependencies are
    // toolchain headers, and its own text is already part of the key.
    std::erase(args, "-MMD");
    args.push_back(header.string());
    args.emplace_back("-o");

    // Compile to a private name and rename into place, so concurrent processes sharing one
    // cache cannot observe a half-written .gch.
    const fs::path temp = utils::FileRenamer::generate_temp_path(gch);
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

namespace {

// Writes to a private name and renames into place, so a concurrent process never reads a partial file.
bool publish(const fs::path& path, const std::string& contents) {
    const fs::path temp = utils::FileRenamer::generate_temp_path(path);
    std::error_code ec;
    {
        std::ofstream file(temp);
        file << contents;
        if (!file.flush()) {
            ec = std::make_error_code(std::errc::io_error);
        }
    }
    if (!ec) {
        fs::rename(temp, path, ec);
    }
    if (ec) {
        fs::remove(temp, ec);
        return false;
    }
    return true;
}

// One entry per kernel PCH directory, checked and built at most once per process; a failed build is
// cached as an empty header.
struct KernelPchEntry {
    std::mutex mutex;
    bool requested = false;
    bool building = false;
    std::atomic<bool> settled = false;
    std::string header;  // Written once, before `settled`.
};

struct KernelPchBuild {
    std::shared_ptr<KernelPchEntry> entry;
    std::jthread thread;
};

std::mutex& kernel_pch_builds_mutex() {
    static std::mutex mutex;
    return mutex;
}

// Kernel PCH builds not yet joined, joined at exit at the latest. A build hashes the PCH's headers, so the
// hash cache is constructed first and is destroyed after the builds are joined.
std::vector<KernelPchBuild>& kernel_pch_builds() {
    ensure_file_hash_cache();
    static std::vector<KernelPchBuild> builds;
    return builds;
}

}  // namespace

std::string ensure_kernel_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::vector<std::string>& defines,
    const std::vector<fs::path>& headers,
    const std::map<std::string, fs::path>& generated,
    const fs::path& pch_root) {
    std::string header_text;
    for (const auto& h : headers) {
        header_text += fmt::format("#include \"{}\"\n", h.string());
    }
    std::map<std::string, std::string> staged_files;
    for (const auto& [name, source] : generated) {
        const auto bytes = utils::read_file_bytes(source.string());
        staged_files.emplace(name, std::string(bytes.begin(), bytes.end()));
    }

    // Everything that can change the PCH's contents, except the text of the headers it includes:
    // the dependency hash below rebuilds the PCH in place when one of those changes.
    tt::StableHasher hasher;
    hasher.update(gpp);
    hasher.update(utils::compiler_version(gpp));
    hasher.update(opt_level);
    hasher.update(cflags);
    hasher.update(includes);
    for (const auto& define : defines) {
        hasher.update(define);
        hasher.update("\n");
    }
    hasher.update(header_text);
    for (const auto& [name, contents] : staged_files) {
        hasher.update(name);
        hasher.update(contents);
    }
    const fs::path dir = fs::absolute(pch_root / fmt::format("kernel-{:016x}", hasher.digest()));

    // The second request starts the build on a thread of its own and parses the headers, as does every
    // compile until the PCH is ready: no compile waits for a PCH, so a PCH that few compiles share never
    // lengthens a build.
    static std::mutex entries_mutex;
    static std::unordered_map<std::string, std::shared_ptr<KernelPchEntry>> entries;
    std::shared_ptr<KernelPchEntry> entry;
    {
        std::lock_guard lock(entries_mutex);
        auto& slot = entries[dir.string()];
        if (!slot) {
            slot = std::make_shared<KernelPchEntry>();
        }
        entry = slot;
    }
    if (entry->settled.load(std::memory_order_acquire)) {
        return entry->header;
    }
    std::lock_guard lock(entry->mutex);
    if (entry->settled.load(std::memory_order_acquire)) {
        return entry->header;
    }
    if (entry->building) {
        return {};
    }
    const auto settle = [entry](std::string built) {
        entry->header = std::move(built);
        entry->settled.store(true, std::memory_order_release);
        return entry->header;
    };

    const std::string gch_name = "kernel_pch.h.gch";
    const fs::path header = dir / "kernel_pch.h";
    const fs::path gch = dir / gch_name;
    if (fs::exists(gch) && dependencies_up_to_date(dir.string(), gch_name)) {
        return settle(header.string());
    }
    // Building a PCH costs a few compiles, so build one only when a second kernel needs it. A kernel
    // whose flags and defines no other kernel shares parses the headers as before.
    if (!entry->requested) {
        entry->requested = true;
        return {};
    }

    // The prefix includes the generated headers by name; "-I<dir>" goes first so it finds the staged copies.
    std::error_code ec;
    fs::create_directories(dir, ec);
    bool staged_all = !ec && publish(header, header_text);
    for (const auto& [name, contents] : staged_files) {
        staged_all = staged_all && publish(dir / name, contents);
    }
    if (!staged_all) {
        log_warning(tt::LogBuildKernels, "Skipping the kernel PCH: cannot stage {}", dir.string());
        return settle({});
    }
    const fs::path temp = utils::FileRenamer::generate_temp_path(gch);
    const fs::path deps = temp.string() + ".d";
    auto args = utils::build_gpp_argv(
        gpp,
        opt_level,
        cflags + " -x c++-header -MT " + gch_name,
        "-I" + dir.string() + " " + includes,
        defines,
        header.string(),
        utils::GppAction::Compile,
        temp.string(),
        deps.string());
    entry->building = true;
    auto build = [=, args = std::move(args)] {
        try {
            std::error_code error;
            const std::string log_path = (dir / "build.log").string();
            if (!utils::exec_command(args, dir.string(), log_path)) {
                log_warning(
                    tt::LogBuildKernels,
                    "Skipping the kernel PCH: {} failed, see {}. Kernels will parse the headers as before.",
                    fmt::join(args, " "),
                    log_path);
                fs::remove(temp, error);
                fs::remove(deps, error);
                settle({});
                return;
            }

            // Publish the PCH, then the hashes of the headers it was built from. A process that reads the old
            // hashes beside the new PCH only rebuilds it again.
            std::ostringstream hashes;
            {
                std::ifstream deps_file(deps);
                write_dependency_hashes(parse_dependency_file(deps_file), dir.string(), gch_name, hashes);
            }
            fs::remove(deps, error);
            fs::rename(temp, gch, error);
            if (error || !hashes || !publish(dir / (gch_name + ".dephash"), hashes.str())) {
                fs::remove(temp, error);
                log_warning(tt::LogBuildKernels, "Skipping the kernel PCH: cannot publish {}", gch.string());
                settle({});
                return;
            }
            settle(header.string());
        } catch (const std::exception& e) {
            log_warning(tt::LogBuildKernels, "Skipping the kernel PCH in {}: {}", dir.string(), e.what());
            settle({});
        }
    };
    std::lock_guard builds_lock(kernel_pch_builds_mutex());
    auto& builds = kernel_pch_builds();
    // Join the builds that have finished: settling is a build's last step.
    std::erase_if(builds, [](const KernelPchBuild& b) { return b.entry->settled.load(std::memory_order_acquire); });
    builds.push_back({entry, std::jthread(std::move(build))});
    return {};
}

void wait_for_kernel_pch_builds() {
    std::vector<KernelPchBuild> builds;
    {
        std::lock_guard lock(kernel_pch_builds_mutex());
        builds.swap(kernel_pch_builds());
    }
}

PchUse select_pch(
    const std::string& gpp, const TargetRecipe& recipe, const fs::path& kernel_dir, const fs::path& pch_root) {
    PchUse use;
    std::string pch;
    // The generated headers the kernel prefix includes, staged into the PCH directory under these names.
    std::map<std::string, fs::path> generated = {
        {"chlkc_descriptors.h", kernel_dir / utils::CB_TABLE_DECLARATIONS_HEADER}};
    if (fs::exists(kernel_dir / "defines_generated.h")) {
        generated.emplace("defines_generated.h", kernel_dir / "defines_generated.h");
    }
    const auto readable = [](const auto& path) { return fs::exists(path); };
    if (!recipe.kernel_pch_headers.empty() && std::ranges::all_of(recipe.kernel_pch_headers, readable) &&
        fs::exists(generated.at("chlkc_descriptors.h"))) {
        // FULL_KERNEL_NAME differs for every kernel, and a precompiled header must not read it.
        std::vector<std::string> defines = recipe.defines;
        std::erase_if(defines, [](const std::string& d) { return d.starts_with("-DFULL_KERNEL_NAME="); });
        std::vector<fs::path> headers = {recipe.pch_umbrella};
        headers.insert(headers.end(), recipe.kernel_pch_headers.begin(), recipe.kernel_pch_headers.end());
        pch = ensure_kernel_pch(
            gpp, recipe.compiler_opt_level, recipe.cflags, recipe.includes, defines, headers, generated, pch_root);
        if (!pch.empty()) {
            // The PCH declares the CB tables; the kernel defines them (found through "-I..").
            use.includes = {"-include", pch, "-include", std::string(utils::CB_TABLE_VALUES_HEADER)};
            // List the PCH's headers in the kernel's dependency file, so editing one rebuilds the kernel.
            use.cflags = " -fpch-deps";
        }
    }
    if (pch.empty()) {
        pch = ensure_pch(gpp, recipe.compiler_opt_level, recipe.cflags, recipe.pch_umbrella, pch_root);
        if (!pch.empty()) {
            use.includes = {"-include", pch};
        }
    }
    if (!pch.empty()) {
        // Warn if GCC rejects the PCH, while allowing textual fallback.
        use.cflags += " -Winvalid-pch -Wno-error=invalid-pch";
    }
    return use;
}

}  // namespace tt::jit_build
