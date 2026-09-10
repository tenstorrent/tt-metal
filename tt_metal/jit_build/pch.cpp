// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pch.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include "common/stable_hash.hpp"
#include "depend.hpp"
#include "jit_build_utils.hpp"

namespace tt::jit_build {
namespace fs = std::filesystem;
namespace {

// One entry per staged header path, held by shared_ptr so a build already in flight
// keeps its entry alive across a pch_cache_clear().
struct PchEntry {
    std::mutex mutex;
    bool ok = false;
    bool failed = false;
};

std::mutex pch_map_mutex;
std::unordered_map<std::string, std::shared_ptr<PchEntry>> pch_entries;

// Keyed by the staged header path rather than by the recipe key alone: that path also
// embeds out_root, so two build environments pointing at different cache roots get
// separate entries instead of the first one reporting success for a .gch the second
// never built.
std::shared_ptr<PchEntry> pch_entry_for(const std::string& header) {
    std::lock_guard lock(pch_map_mutex);
    std::shared_ptr<PchEntry>& entry = pch_entries[header];
    if (!entry) {
        entry = std::make_shared<PchEntry>();
    }
    return entry;
}

// Publishes the dependency-hash sidecar that later calls in this process use to decide
// whether a built .gch still matches its closure. Every rule in the .d is flattened
// under one label because the target GCC records there follows the output name it was
// given, which is a per-process temp path.
bool write_pch_dephash(
    const std::string& dep_path, const std::string& dir, const std::string& gch, const std::string& hash_path) {
    jit_build::ParsedDependencies parsed;
    {
        std::ifstream dep_file(dep_path);
        if (!dep_file.is_open()) {
            return false;
        }
        parsed = jit_build::parse_dependency_file(dep_file);
    }
    if (parsed.empty()) {
        return false;
    }
    jit_build::ParsedDependencies flattened;
    std::vector<std::string>& deps = flattened[gch];
    for (const auto& [target, dep_list] : parsed) {
        deps.insert(deps.end(), dep_list.begin(), dep_list.end());
    }

    const std::string tmp = tt::jit_build::utils::FileRenamer::generate_temp_path(hash_path);
    {
        std::ofstream hash_file(tmp);
        if (!hash_file.is_open()) {
            return false;
        }
        jit_build::write_dependency_hashes(flattened, dir, gch, hash_file);
        hash_file.close();
        if (hash_file.fail()) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, hash_path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

}  // namespace

// ClearKernelCache() also clears the file-hash cache, which is what makes a mid-run
// header edit visible to need_compile(); an entry that survived it would keep handing
// out a .gch built from the pre-edit contents.
void pch_cache_clear() {
    std::lock_guard lock(pch_map_mutex);
    pch_entries.clear();
}

// Returns the header path to force-include for this compile recipe, building the
// PCH on first use. Returns empty if a PCH is unavailable, in which case the
// caller simply compiles without one.
//
// `defines` must already have any -include pairs stripped, and must otherwise
// match the compile exactly, or GCC will reject the result.
std::string ensure_pch(
    const std::string& gpp,
    const std::string& root,
    const std::string& out_root,
    const std::string& target_name,
    std::string_view umbrella_rel,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::vector<std::string>& defines) {
    // The relative include roots ("-I.", "-I..") resolve against a kernel's own build
    // directory and are how a compile reaches per-kernel generated headers. A PCH must
    // not consume anything per-kernel, so they are dropped from the PCH build: if an
    // umbrella ever grows a (transitive) dependency on a generated header, its PCH
    // build fails loudly and every compile falls back to plain parsing, instead of one
    // kernel's generated state -- or a negative __has_include probe -- being silently
    // baked into all the other kernels sharing the key.
    std::vector<std::string> include_args;
    for (auto& tok : tt::jit_build::utils::tokenize_flags(includes)) {
        if (tok == "-I." || tok == "-I..") {
            continue;
        }
        include_args.push_back(std::move(tok));
    }

    // One PCH per distinct recipe. Blaze passes compile-time args through generated
    // headers, so this stays at one key per target type there; kernels that carry
    // per-kernel -D defines or include paths (common in ttnn) fan out to one PCH
    // build per distinct set, which is part of why the feature is opt-in.
    tt::StableHasher hasher;
    hasher.update(target_name);
    hasher.update(std::string(umbrella_rel));
    hasher.update(opt_level);
    hasher.update(cflags);
    for (const auto& inc : include_args) {
        hasher.update(inc);
    }
    for (const auto& define : defines) {
        hasher.update(define);
    }
    const uint64_t key = hasher.digest();

    const fs::path umbrella = fs::path(std::string(umbrella_rel));
    const fs::path dir = fs::path(out_root) / "pch" / target_name / fmt::format("{:016x}", key);
    const fs::path header = dir / umbrella.filename();
    const std::string gch = header.string() + ".gch";
    const std::string dep = header.string() + ".d";
    const std::string dephash = gch + ".dephash";

    // The map lock is held only for the entry lookup; the build runs under the entry's
    // own mutex, so distinct keys build in parallel and compiles whose key is already
    // built and still valid never wait. A fresh process rebuilds rather than trusting a
    // .gch found on disk: the key does not cover header contents, so an existing file
    // could predate a source edit. Every shared-path write goes through a unique temp
    // file and an atomic rename -- concurrent processes rebuilding the same key never
    // expose a partial file, and a compile holding the old .gch open keeps reading it
    // after the rename.
    const std::shared_ptr<PchEntry> entry = pch_entry_for(header.string());
    std::lock_guard entry_lock(entry->mutex);

    // A failed build is not retried: giving up on the PCH for the rest of the process
    // costs one wasted build, retrying per compile would cost thousands.
    if (entry->failed) {
        return {};
    }

    // A successful build is not trusted for the rest of the process either. The key
    // covers flags and defines but not header contents, and ClearKernelCache() drops
    // the file-hash cache that makes a mid-run header edit visible to need_compile().
    // An entry that outlived such an edit would keep force-including a .gch holding
    // the pre-edit contents while every consumer wrote a dephash sidecar computed from
    // the current ones -- marking objects compiled against stale headers as up to date
    // on disk, for this run and every later one.
    if (entry->ok && jit_build::dependencies_up_to_date_file(dephash)) {
        return header.string();
    }

    const bool built = [&] {
        std::error_code ec;
        fs::create_directories(dir, ec);
        const std::string header_tmp = tt::jit_build::utils::FileRenamer::generate_temp_path(header);
        fs::copy_file(fs::path(root) / umbrella, header_tmp, fs::copy_options::overwrite_existing, ec);
        if (!ec) {
            fs::rename(header_tmp, header, ec);
        }
        if (ec) {
            log_warning(
                tt::LogBuildKernels,
                "PCH: could not stage {}: {}; compiling without it",
                header.string(),
                ec.message());
            return false;
        }

        const std::string gch_tmp = tt::jit_build::utils::FileRenamer::generate_temp_path(gch);
        const std::string dep_tmp = tt::jit_build::utils::FileRenamer::generate_temp_path(dep);

        // Built through the same argv builder as the kernel compiles: any drift between
        // the two flag sets makes GCC reject the PCH. The .d (via the -MMD already in
        // cflags) records the PCH's dependency closure; a kernel compile that consumes
        // the PCH emits a truncated .d of its own (-MMD stops at the PCH boundary), so
        // compile_one merges this file back in to keep the dephash cache honest.
        const std::vector<std::string> args = tt::jit_build::utils::build_gpp_argv(
            gpp,
            opt_level,
            cflags,
            fmt::format("{}", fmt::join(include_args, " ")),
            defines,
            header.string(),
            tt::jit_build::utils::GppAction::PrecompileHeader,
            gch_tmp,
            dep_tmp);

        // The log is per-temp (and so per-process): concurrent rebuilds of one key must
        // not interleave writes. Removed on success, kept for the warning on failure.
        const std::string log = gch_tmp + ".log";
        bool ok = tt::jit_build::utils::exec_command(args, dir.string(), log);
        if (ok) {
            // -MMD lists the staged copy the compiler was handed, never the umbrella in
            // the source tree it was copied from. That path is recorded explicitly so it
            // reaches every consumer's dephash through merge_pch_deps_into_kernel_d.
            // need_compile() runs before this function, so without it an edited umbrella
            // leaves every object built against the old one reported as up to date, and
            // the staged copy is never refreshed.
            std::ofstream dep_out(dep_tmp, std::ios::app);
            dep_out << gch << ": " << (fs::path(root) / umbrella).string() << '\n';
            dep_out.flush();
            ok = dep_out.good();
            if (!ok) {
                log_warning(tt::LogBuildKernels, "PCH: could not record {} in {}", umbrella_rel, dep_tmp);
            }
        }
        if (ok) {
            fs::rename(dep_tmp, dep, ec);
            if (!ec) {
                fs::rename(gch_tmp, gch, ec);
            }
            if (ec) {
                log_warning(tt::LogBuildKernels, "PCH: could not publish {}: {}", gch, ec.message());
                ok = false;
            }
        }
        if (ok && !write_pch_dephash(dep, dir.string(), gch, dephash)) {
            log_warning(tt::LogBuildKernels, "PCH: could not record the closure of {} in {}", gch, dephash);
            ok = false;
        }
        if (ok) {
            fs::remove(log, ec);
            log_info(tt::LogBuildKernels, "PCH: built {} for {}", gch, target_name);
        } else {
            fs::remove(gch_tmp, ec);
            fs::remove(dep_tmp, ec);
            log_warning(
                tt::LogBuildKernels, "PCH: build failed for {} (log: {}); compiling without it", target_name, log);
        }
        return ok;
    }();
    entry->ok = built;
    entry->failed = !built;
    return built ? header.string() : std::string{};
}

// A compile that consumed a PCH emits a truncated .d: -MMD lists only what was parsed
// after the PCH boundary, which would blind the dephash sidecar to edits of any header
// inside the umbrella. Append the PCH's own dependency closure (recorded at PCH build
// time, see ensure_pch) under the kernel object's target -- parse_dependency_file
// accumulates rules for one target across lines. On any failure the kernel .d is
// removed, which drops the sidecar and forces a recompile next run: slower, never stale.
void merge_pch_deps_into_kernel_d(
    const std::string& kernel_d_path, const std::string& kernel_obj, const std::string& pch_d_path) {
    std::ifstream pch_d(pch_d_path);
    std::ofstream out(kernel_d_path, std::ios::app);
    bool ok = pch_d.is_open() && out.is_open();
    if (ok) {
        const jit_build::ParsedDependencies deps = jit_build::parse_dependency_file(pch_d);
        ok = !deps.empty();
        for (const auto& [target, dep_list] : deps) {
            for (const auto& dep : dep_list) {
                out << kernel_obj << ": " << dep << '\n';
            }
        }
        out.flush();
        ok = ok && out.good();
    }
    if (!ok) {
        out.close();
        std::error_code ec;
        fs::remove(kernel_d_path, ec);
        log_warning(
            tt::LogBuildKernels,
            "PCH: could not merge {} into {}; dropping the .d so this object is rebuilt next run",
            pch_d_path,
            kernel_d_path);
    }
}

}  // namespace tt::jit_build
