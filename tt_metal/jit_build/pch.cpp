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
#include <tt_stl/assert.hpp>

#include "common/stable_hash.hpp"
#include "depend.hpp"
#include "jit_build_utils.hpp"

namespace tt::jit_build {
namespace fs = std::filesystem;
namespace {

// Shared ownership keeps in-flight entries alive across cache clearing.
struct PchEntry {
    std::mutex mutex;
    bool ok = false;
    bool failed = false;
};

std::mutex pch_map_mutex;
std::unordered_map<std::string, std::shared_ptr<PchEntry>> pch_entries;

// The full header path separates output roots as well as recipes.
std::shared_ptr<PchEntry> pch_entry_for(const std::string& header) {
    std::lock_guard lock(pch_map_mutex);
    std::shared_ptr<PchEntry>& entry = pch_entries[header];
    if (!entry) {
        entry = std::make_shared<PchEntry>();
    }
    return entry;
}

// Flatten rules under the final target; GCC records the temporary output name.
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

void require_pch_consumed(const std::string& log_path, const std::string& header) {
    TT_FATAL(!header.empty(), "Strict PCH mode: no header requested (log: {})", log_path);
    std::ifstream log(log_path);
    TT_FATAL(log.is_open(), "Strict PCH mode: cannot read {}", log_path);
    const std::string expected = "! " + header + ".gch";
    for (std::string line; std::getline(log, line);) {
        if (line == expected) {
            return;
        }
    }
    TT_THROW("Strict PCH mode: compiler did not consume {}.gch (log: {})", header, log_path);
}

// Drop completed and failed entries; existing users retain their shared ownership.
void pch_cache_clear() {
    std::lock_guard lock(pch_map_mutex);
    pch_entries.clear();
}

// Returns the staged header to force-include, or empty to request ordinary compilation.
// The caller must remove per-kernel -include pairs from defines.
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
    // These roots expose per-kernel generated headers and cannot be shared.
    // Required generated includes then fail; umbrellas must also exclude headers
    // whose __has_include probes would silently capture the wrong state.
    std::vector<std::string> include_args;
    for (auto& tok : tt::jit_build::utils::tokenize_flags(includes)) {
        if (tok == "-I." || tok == "-I..") {
            continue;
        }
        include_args.push_back(std::move(tok));
    }

    // Distinct flags, defines or include paths require separate PCH builds.
    // Recipes with few consumers may cost more than ordinary compilation.
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

    // Serialize validation/builds per entry, allowing different recipes to proceed.
    // Fresh processes rebuild. Atomic renames protect each file from partial reads;
    // publication of the header, .d, .gch and sidecar is not one transaction.
    const std::shared_ptr<PchEntry> entry = pch_entry_for(header.string());
    std::lock_guard entry_lock(entry->mutex);

    // Retry failures only after cache clearing, not once per consuming kernel.
    if (entry->failed) {
        return {};
    }

    // Recipe keys exclude contents, so validate the dependency hashes before reuse.
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

        // Match consumer flags and record the closure that kernel -MMD output omits.
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

        // Keep separate logs for concurrent processes and retain failures for diagnosis.
        const std::string log = gch_tmp + ".log";
        bool ok = tt::jit_build::utils::exec_command(args, dir.string(), log);
        if (ok) {
            // GCC sees only the staged copy. Track the source umbrella too so edits
            // invalidate consumers before ensure_pch is called again.
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

// Add the PCH dependency closure omitted by the consumer's -MMD output.
// If merging fails, drop the .d to prevent reuse with incomplete dependency hashes.
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
