// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/jit_server/jit_compile_server_controller.hpp"
#include "impl/jit_server/types.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"

// Remote JIT compile server
//
// WARNING: Experimental feature with security implications. Do not use in production.
// WARNING: The server can execute arbitrary code.  Make sure not to expose the RPC endpoint to untrusted clients.
//
// TODO 1:
// The compile/link logic below currently duplicates the core of
// `jit_build/build.cpp` (compile_one, link_one, dependency checks). Two reasons:
// 1. Compile server cannot have dependency on MetalContext, which would take a lock on the device.
// 2. At this point, we are not sure what the server will need to do that jit_build does not.
// We can unify then once the server's requirements stabilize.
//
// TODO 2:
// Local builds (`JitBuildState` in build.cpp) compute `build_state_hash_` over effective
// compile/link inputs (gpp, cflags, defines, includes, lflags, linker script, extra link
// objs, src list, opt levels), persist it in each output dir as `.build_state`, and treat a
// mismatch as "state changed" to force recompile/relink even when individual `.o` dependency
// hashes might still look fresh.
//
// This server does not implement `.build_state` / `build_state_hash_`. It partitions the
// on-disk cache by `build_key` and kernel path, and decides compile/link via
// `dependencies_up_to_date` + `.dephash` (same helper family as local JIT). That matches
// common cases when the client uses a fresh cache subtree (e.g. kernel path includes a
// compile hash) or when sources and listed link inputs change on disk.
//
// Limitation: without an explicit recipe/state fingerprint in the cache dir, reusing the
// same `build_key` + kernel path while changing only flags or other recipe fields that are
// not reflected in dependency tracking could theoretically reuse stale `.o`/`.elf` in edge
// cases where local build.cpp would invalidate via `build_state_hash_`.
//

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_keep_running{true};
std::atomic<int> g_outstanding_compiles{0};

constexpr const char* kEndpointEnv = "TT_METAL_JIT_SERVER_ENDPOINT";
// SECURITY: binds to all interfaces by default. Any host on the network can reach this
// server and trigger arbitrary compilation (i.e., arbitrary code execution). Restrict to
// localhost or use a firewall in shared environments.
constexpr const char* kDefaultEndpoint = "0.0.0.0:9876";
constexpr const char* kServerCacheRootEnv = "TT_METAL_JIT_SERVER_CACHE_ROOT";
constexpr const char* kDefaultServerCacheRoot = "/tmp/tt-metal-cache/";
std::string g_server_cache_root = kDefaultServerCacheRoot;

std::string normalize_cache_root(std::string cache_root) {
    if (cache_root.empty()) {
        return std::string(kDefaultServerCacheRoot);
    }
    if (cache_root.back() != '/') {
        cache_root.push_back('/');
    }
    return cache_root;
}

// SECURITY: kernel_name is client-supplied and used verbatim in path construction.
// A malicious client could use "../" segments to escape the cache subtree.
// Currently acceptable because this is a trusted internal tool; if the server is ever
// exposed to untrusted clients, validate that kernel_name is a safe relative path.
std::string kernel_cache_dir(std::uint64_t build_key, const std::string& kernel_name) {
    return g_server_cache_root + std::to_string(build_key) + "/kernels/" + kernel_name;
}

std::string target_cache_dir(std::uint64_t build_key, const std::string& kernel_name, const std::string& target_name) {
    return kernel_cache_dir(build_key, kernel_name) + target_name + "/";
}

void handle_signal(int /*signal*/) { g_keep_running.store(false); }

std::string firmware_cache_dir(std::uint64_t build_key, const std::string& target_name) {
    return g_server_cache_root + std::to_string(build_key) + "/firmware/" + target_name + "/";
}

// Resolve firmware path from the server cache populated by uploadFirmware RPC.
// PoC limitation: uploaded firmware is assumed durable in the configured server cache root.
// If the server cache is cleared after upload, compile will fail until the client re-uploads.
fs::path resolve_uploaded_firmware_path(std::uint64_t build_key, const tt::tt_metal::jit_server::TargetRecipe& target) {
    if (target.weakened_firmware_name.empty()) {
        throw std::runtime_error(
            fmt::format("Internal error: expected non-empty weakened_firmware_name for target {}", target.target_name));
    }
    const std::string firmware_file = fs::path(target.weakened_firmware_name).filename().string();
    const std::string fw_dir = firmware_cache_dir(build_key, target.target_name);
    fs::path candidate = fs::path(fw_dir) / firmware_file;
    if (fs::exists(candidate)) {
        return candidate;
    }

    throw std::runtime_error(fmt::format(
        "Firmware artifact not found for build_key {} target {} file {}. "
        "Expected at {}. Ensure the client uploads firmware via uploadFirmware RPC before compiling.",
        build_key,
        target.target_name,
        firmware_file,
        candidate.string()));
}

void build_failure(
    const std::string& target, const std::string& op, const std::string& cmd, const std::string& log_file) {
    std::ifstream file{log_file};
    if (file.is_open()) {
        std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        throw std::runtime_error(fmt::format("{} {} failure -- cmd: {}\nLog: {}", target, op, cmd, log_contents));
    }
    throw std::runtime_error(
        fmt::format("{} {} failure -- cmd: {} (log file {} not found)", target, op, cmd, log_file));
}

bool need_compile(const std::string& out_dir, const std::string& obj) {
    return !fs::exists(out_dir + obj) || !tt::jit_build::dependencies_up_to_date(out_dir, obj);
}

bool need_link(const std::string& out_dir, const std::string& target_name) {
    std::string elf_path = out_dir + target_name + ".elf";
    return !fs::exists(elf_path) || !tt::jit_build::dependencies_up_to_date(out_dir, elf_path);
}

// SECURITY: constructs a shell command from client-supplied strings (gpp, cflags, defines,
// includes, srcs, objs). A malicious client can inject arbitrary shell commands via any of
// these fields. This is equivalent to remote code execution by design — the server's
// purpose is to run the compiler on behalf of the client.
void compile_one(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const std::string& out_dir,
    size_t src_index,
    const std::string& temp_obj) {
    std::string cmd = fmt::format("cd {} && {} ", out_dir, gpp);
    cmd += fmt::format("-{} ", target.compiler_opt_level);
    cmd += target.cflags;
    cmd += target.includes;

    std::string obj_path = out_dir + target.objs[src_index];
    std::string obj_temp_path = out_dir + temp_obj;
    std::string temp_d_path = fs::path(obj_temp_path).replace_extension("d").string();
    cmd += fmt::format("-c -o {} {} -MF {} ", obj_temp_path, target.srcs[src_index], temp_d_path);
    cmd += target.defines;

    tt::jit_build::utils::FileRenamer log_file(obj_path + ".log");
    fs::remove(log_file.path());
    if (!tt::jit_build::utils::run_command(cmd, log_file.path(), false)) {
        build_failure(target.target_name, "compile", cmd, log_file.path());
    }
    tt::jit_build::write_dependency_hashes(out_dir, obj_temp_path, obj_temp_path + ".dephash");
    fs::remove(temp_d_path);
}

// SECURITY: same shell injection exposure as compile_one — lflags, extra_link_objs,
// linker_script, and weakened_firmware_name are all client-supplied and interpolated
// into the shell command without escaping.
void link_one(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const std::string& out_dir,
    const std::string& link_objs_str) {
    std::string cmd = fmt::format("cd {} && {} ", out_dir, gpp);
    cmd += fmt::format("-{} ", target.linker_opt_level);

    std::vector<std::string> link_deps = {target.linker_script};
    if (!target.weakened_firmware_name.empty()) {
        link_deps.push_back(target.weakened_firmware_name);
        if (!target.firmware_is_kernel_object) {
            cmd += "-Wl,--just-symbols=";
        }
        cmd += target.weakened_firmware_name + " ";
    }

    cmd += target.lflags;
    cmd += target.extra_link_objs;
    cmd += link_objs_str;
    std::string elf_name = out_dir + target.target_name + ".elf";
    tt::jit_build::utils::FileRenamer elf_file(elf_name);
    cmd += "-o " + elf_file.path();

    tt::jit_build::utils::FileRenamer log_file(elf_name + ".log");
    fs::remove(log_file.path());
    if (!tt::jit_build::utils::run_command(cmd, log_file.path(), false)) {
        build_failure(target.target_name, "link", cmd, log_file.path());
    }

    tt::jit_build::utils::FileRenamer dephash_file(elf_name + ".dephash");
    std::ofstream hash_file(dephash_file.path());
    tt::jit_build::write_dependency_hashes({{elf_name, std::move(link_deps)}}, out_dir, elf_name, hash_file);
    hash_file.close();
    if (hash_file.fail()) {
        fs::remove(dephash_file.path());
    }
}

std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot read ELF file: " + path);
    }
    std::streampos pos = file.tellg();
    if (pos == std::streampos(-1)) {
        throw std::runtime_error("Cannot determine size of ELF file: " + path);
    }
    auto byte_count = static_cast<std::streamsize>(pos);
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<size_t>(byte_count));
    file.read(reinterpret_cast<char*>(data.data()), byte_count);
    return data;
}

void build_target(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const std::string& out_dir,
    tt::tt_metal::jit_server::CompileResponse& response) {
    if (target.srcs.size() != target.objs.size()) {
        throw std::runtime_error("srcs and objs must have the same size for target " + target.target_name);
    }

    fs::create_directories(out_dir);

    const size_t num_objs = target.objs.size();
    std::vector<std::string> temp_objs;
    temp_objs.reserve(num_objs);
    for (const auto& obj : target.objs) {
        temp_objs.push_back(tt::jit_build::utils::FileRenamer::generate_temp_path(obj));
    }

    std::vector<bool> compiled(num_objs, false);
    for (size_t i = 0; i < num_objs; ++i) {
        if (need_compile(out_dir, target.objs[i])) {
            compiled[i] = true;
        }
    }

    const size_t recompiled = std::count(compiled.begin(), compiled.end(), true);
    const size_t cache_hit = num_objs - recompiled;
    bool needs_link = need_link(out_dir, target.target_name);

    log_info(
        tt::LogMetal,
        "  target {}: obj={} hit={} miss={} link={}",
        target.target_name,
        num_objs,
        cache_hit,
        recompiled,
        needs_link || recompiled > 0 ? "yes" : "no");

    for (size_t i = 0; i < num_objs; ++i) {
        if (compiled[i]) {
            compile_one(gpp, target, out_dir, i, temp_objs[i]);
        }
    }

    if (recompiled > 0 || needs_link) {
        std::string link_objs_str;
        for (size_t i = 0; i < num_objs; ++i) {
            std::string temp_path = out_dir + temp_objs[i];
            if (!compiled[i]) {
                std::error_code ec;
                fs::create_hard_link(out_dir + target.objs[i], temp_path, ec);
                if (ec) {
                    fs::copy_file(out_dir + target.objs[i], temp_path, fs::copy_options::overwrite_existing);
                }
            }
            link_objs_str += temp_path + " ";
        }
        link_one(gpp, target, out_dir, link_objs_str);
    }

    for (size_t i = 0; i < num_objs; ++i) {
        fs::path src_path = out_dir + temp_objs[i];
        fs::path dst_path = out_dir + target.objs[i];
        if (compiled[i]) {
            fs::rename(src_path, dst_path);
            fs::rename(fs::path(src_path).concat(".dephash"), fs::path(dst_path).concat(".dephash"));
        } else if (fs::exists(src_path)) {
            fs::remove(src_path);
        }
    }

    std::string elf_path = out_dir + target.target_name + ".elf";
    tt::tt_metal::jit_server::ElfBlob blob;
    blob.name = target.target_name;
    blob.data = read_file_bytes(elf_path);
    response.elf_blobs.push_back(std::move(blob));
}

tt::tt_metal::jit_server::CompileResponse compile_callback(const tt::tt_metal::jit_server::CompileRequest& request) {
    tt::tt_metal::jit_server::CompileResponse response;
    auto request_start = std::chrono::steady_clock::now();
    int outstanding = g_outstanding_compiles.fetch_add(1) + 1;

    try {
        log_info(
            tt::LogMetal,
            "compile {}: targets={} genfiles={} outstanding={}",
            request.kernel_name,
            request.targets.size(),
            request.generated_files.size(),
            outstanding);

        // SECURITY: file.name is client-supplied. While fs::path join handles separators,
        // a name like "../../foo" could still write outside the cache. Acceptable for a
        // trusted client; validate if the server is ever exposed to untrusted input.
        if (!request.generated_files.empty()) {
            const fs::path genfiles_dir = kernel_cache_dir(request.build_key, request.kernel_name);
            fs::create_directories(genfiles_dir);
            for (const auto& file : request.generated_files) {
                std::string target_path = (genfiles_dir / file.name).string();
                tt::jit_build::utils::FileRenamer tmp(target_path);
                std::ofstream out(tmp.path(), std::ios::binary);
                if (!out.is_open()) {
                    throw std::runtime_error("Cannot create file: " + target_path);
                }
                out.write(
                    reinterpret_cast<const char*>(file.content.data()),
                    static_cast<std::streamsize>(file.content.size()));
                if (!out) {
                    throw std::runtime_error("Failed to write file: " + target_path);
                }
            }
        }

        for (const auto& target : request.targets) {
            tt::tt_metal::jit_server::TargetRecipe resolved_target = target;
            if (!resolved_target.weakened_firmware_name.empty()) {
                resolved_target.weakened_firmware_name =
                    resolve_uploaded_firmware_path(request.build_key, resolved_target).string();
            }
            std::string out_dir = target_cache_dir(request.build_key, request.kernel_name, target.target_name);
            build_target(request.gpp, resolved_target, out_dir, response);
        }

        response.success = true;

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - request_start)
                .count();
        log_info(tt::LogMetal, "done {}: {}ms, elfs={}", request.kernel_name, elapsed_ms, response.elf_blobs.size());
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = e.what();
        log_warning(tt::LogMetal, "FAIL {}: {}", request.kernel_name, response.error_message);
    }
    g_outstanding_compiles.fetch_sub(1);
    return response;
}

tt::tt_metal::jit_server::UploadFirmwareResponse upload_firmware_callback(
    const tt::tt_metal::jit_server::UploadFirmwareRequest& request) {
    tt::tt_metal::jit_server::UploadFirmwareResponse response;
    try {
        log_info(
            tt::LogMetal, "uploadFirmware build_key={}: artifacts={}", request.build_key, request.artifacts.size());

        for (const auto& artifact : request.artifacts) {
            std::string safe_target = fs::path(artifact.target_name).filename().string();
            std::string safe_file = fs::path(artifact.file_name).filename().string();
            if (safe_target.empty() || safe_file.empty() || safe_target.find("..") != std::string::npos ||
                safe_file.find("..") != std::string::npos) {
                throw std::runtime_error(fmt::format(
                    "Invalid firmware artifact names: target='{}' file='{}'",
                    artifact.target_name,
                    artifact.file_name));
            }

            std::string fw_dir = firmware_cache_dir(request.build_key, safe_target);
            fs::create_directories(fw_dir);
            std::string target_path = fw_dir + safe_file;

            tt::jit_build::utils::FileRenamer tmp(target_path);
            std::ofstream out(tmp.path(), std::ios::binary);
            if (!out.is_open()) {
                throw std::runtime_error("Cannot create firmware file: " + target_path);
            }
            out.write(
                reinterpret_cast<const char*>(artifact.data.data()),
                static_cast<std::streamsize>(artifact.data.size()));
            if (!out) {
                throw std::runtime_error("Failed to write firmware file: " + target_path);
            }

            log_info(
                tt::LogMetal,
                "  persisted firmware artifact: build_key={} target={} file={} size={}",
                request.build_key,
                safe_target,
                safe_file,
                artifact.data.size());
        }

        response.success = true;
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = e.what();
        log_warning(tt::LogMetal, "uploadFirmware FAIL: {}", response.error_message);
    }
    return response;
}

}  // namespace

int main() {
    const char* endpoint_env = std::getenv(kEndpointEnv);
    const std::string endpoint = endpoint_env != nullptr ? endpoint_env : kDefaultEndpoint;
    const char* cache_root_env = std::getenv(kServerCacheRootEnv);
    g_server_cache_root = normalize_cache_root(cache_root_env != nullptr ? cache_root_env : kDefaultServerCacheRoot);

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    tt::tt_metal::jit_server::JitCompileServerController server(compile_callback, upload_firmware_callback);
    server.start(endpoint);
    log_info(tt::LogMetal, "JIT compile server listening on {}", endpoint);
    log_info(tt::LogMetal, "JIT compile server cache root: {}", g_server_cache_root);

    while (g_keep_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    server.stop();
    return 0;
}
