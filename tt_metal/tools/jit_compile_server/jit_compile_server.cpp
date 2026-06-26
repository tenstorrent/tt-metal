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
// WARNING: Experimental feature — do not expose to untrusted networks or clients.
//
// Security posture (consistent with Metal's MPI-based transport model):
//  - Authentication: none.  The server trusts any client that can reach the endpoint.
//    Default bind is localhost; set TT_METAL_JIT_SERVER_ENDPOINT to widen.
//  - Encryption: none (Cap'n Proto over plain TCP).  Use SSH tunneling or a VPN if the
//    link crosses an untrusted network.
//  - Command execution: the server's purpose is to run the compiler the client specifies,
//    so the client inherently has code-execution capability.  Shell injection is mitigated
//    by using posix_spawn with explicit argument vectors instead of system().
//  - Path traversal: client-supplied path components (kernel_name, target_name, obj names,
//    generated file names) are validated to reject absolute paths and ".." components.
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
// Binds to localhost by default — only local processes can reach the server.
// Set TT_METAL_JIT_SERVER_ENDPOINT=0.0.0.0:9876 (or a specific interface) to accept
// remote connections.  In that case, restrict access with a firewall or SSH tunnel.
constexpr const char* kDefaultEndpoint = "localhost:9876";
constexpr const char* kServerCacheRootEnv = "TT_METAL_JIT_SERVER_CACHE_ROOT";
constexpr const char* kDefaultServerCacheRoot = "/tmp/tt-metal-cache/";
std::string g_server_cache_root = kDefaultServerCacheRoot;

// Reject paths that could escape the cache root: absolute paths, ".." components, or
// empty strings.  Throws on violation.
void validate_safe_relative_path(const std::string& path, const char* field_name) {
    if (path.empty()) {
        throw std::runtime_error(fmt::format("Empty {} is not allowed", field_name));
    }
    if (path[0] == '/') {
        throw std::runtime_error(fmt::format("Absolute {} is not allowed: {}", field_name, path));
    }
    for (const auto& component : fs::path(path)) {
        if (component == "..") {
            throw std::runtime_error(fmt::format("{} must not contain '..' components: {}", field_name, path));
        }
    }
}

std::string normalize_cache_root(std::string cache_root) {
    if (cache_root.empty()) {
        return std::string(kDefaultServerCacheRoot);
    }
    if (cache_root.back() != '/') {
        cache_root.push_back('/');
    }
    return cache_root;
}

// This server's own build root (TT_METAL_HOME, trailing-slash), set at startup. Used to re-root
// client-shipped toolchain/link paths onto THIS server's tree. Empty => re-root disabled.
std::string g_server_root;

// Replace every occurrence of `client_root` with `g_server_root` in `s`. The client ships absolute
// paths rooted at its TT_METAL_HOME (sfpi g++, linker script, hw link objects); when the server's
// tree lives at a different path, those paths don't exist here, so we swap the root prefix to
// resolve them against the server's own tree. No-op when the roots match, either is empty, or the
// path is not under the client root (e.g. a machine-global toolchain present on the server too).
void reroot_to_server(std::string& s, const std::string& client_root) {
    if (client_root.empty() || g_server_root.empty() || client_root == g_server_root || s.empty()) {
        return;
    }
    std::string::size_type pos = 0;
    while ((pos = s.find(client_root, pos)) != std::string::npos) {
        s.replace(pos, client_root.size(), g_server_root);
        pos += g_server_root.size();
    }
}

// All client-supplied path components (kernel_name, target_name, obj names, generated file
// names) are validated by validate_safe_relative_path() before reaching these helpers.
std::string kernel_cache_dir(std::uint64_t build_key, const std::string& kernel_name) {
    return (fs::path(g_server_cache_root) / std::to_string(build_key) / "kernels" / kernel_name).string() + "/";
}

std::string target_cache_dir(std::uint64_t build_key, const std::string& kernel_name, const std::string& target_name) {
    return (fs::path(kernel_cache_dir(build_key, kernel_name)) / target_name).string() + "/";
}

void handle_signal(int /*signal*/) { g_keep_running.store(false); }

std::string firmware_cache_dir(std::uint64_t build_key, const std::string& target_name) {
    return (fs::path(g_server_cache_root) / std::to_string(build_key) / "firmware" / target_name).string() + "/";
}

// Resolve firmware path from the server cache populated by uploadFirmware RPC.
// Note: uploaded firmware is assumed durable in the configured server cache root; if the cache is
// cleared after upload, compile fails until the client re-uploads.
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

std::string format_args(const std::vector<std::string>& args) {
    std::string result;
    for (const auto& a : args) {
        if (!result.empty()) {
            result += ' ';
        }
        result += a;
    }
    return result;
}

void append_tokenized(std::vector<std::string>& args, const std::string& flags) {
    auto tokens = tt::jit_build::utils::tokenize_flags(flags);
    args.insert(args.end(), std::make_move_iterator(tokens.begin()), std::make_move_iterator(tokens.end()));
}

// Uses posix_spawn with an explicit argument vector — no shell interpretation of
// client-supplied fields.
void compile_one(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const std::string& out_dir,
    size_t src_index,
    const std::string& temp_obj) {
    std::string obj_path = out_dir + target.objs[src_index];
    std::string obj_temp_path = out_dir + temp_obj;
    std::string temp_d_path = fs::path(obj_temp_path).replace_extension("d").string();
    // Build the g++ argv via the shared jit_build_utils builder and run it directly (posix_spawn,
    // no shell), the same path used by the local compile.
    std::vector<std::string> defines(target.defines.begin(), target.defines.end());
    std::vector<std::string> args = tt::jit_build::utils::build_gpp_argv(
        gpp,
        target.compiler_opt_level,
        target.cflags,
        target.includes,
        defines,
        target.srcs[src_index],
        tt::jit_build::utils::GppAction::Compile,
        obj_temp_path,
        temp_d_path);

    tt::jit_build::utils::FileRenamer log_file(obj_path + ".log");
    fs::remove(log_file.path());
    if (!tt::jit_build::utils::exec_command(args, out_dir, log_file.path())) {
        build_failure(target.target_name, "compile", format_args(args), log_file.path());
    }
    const std::string dephash_path = obj_temp_path + ".dephash";
    tt::jit_build::write_dependency_hashes(out_dir, obj_temp_path, dephash_path);
    fs::remove(temp_d_path);

    // A preprocessed .ii has no #includes, so -MMD emits no usable .d and the dephash above isn't
    // written -> the object looks stale on every request and the server recompiles it every time. The
    // .ii is self-contained (headers expanded inline), so record it as the sole dependency: an
    // identical re-shipped .ii then validates the hash and the server reuses the cached .o.
    const std::string& src = target.srcs[src_index];
    const bool is_preprocessed = src.size() >= 3 && src.compare(src.size() - 3, 3, ".ii") == 0;
    if (is_preprocessed && !fs::exists(dephash_path)) {
        std::ofstream hash_file(dephash_path);
        tt::jit_build::ParsedDependencies deps{{obj_temp_path, {src}}};
        tt::jit_build::write_dependency_hashes(deps, out_dir, obj_temp_path, hash_file);
        hash_file.close();
        if (hash_file.fail()) {
            fs::remove(dephash_path);
        }
    }
}

void link_one(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const std::string& out_dir,
    const std::vector<std::string>& link_obj_paths) {
    std::vector<std::string> args;
    append_tokenized(args, gpp);
    args.push_back("-" + target.linker_opt_level);

    std::vector<std::string> link_deps = {target.linker_script};
    if (!target.weakened_firmware_name.empty()) {
        link_deps.push_back(target.weakened_firmware_name);
        if (!target.firmware_is_kernel_object) {
            args.push_back("-Wl,--just-symbols=" + target.weakened_firmware_name);
        } else {
            args.push_back(target.weakened_firmware_name);
        }
    }

    append_tokenized(args, target.lflags);
    append_tokenized(args, target.extra_link_objs);
    args.insert(args.end(), link_obj_paths.begin(), link_obj_paths.end());
    std::string elf_name = out_dir + target.target_name + ".elf";
    tt::jit_build::utils::FileRenamer elf_file(elf_name);
    args.push_back("-o");
    args.push_back(elf_file.path());

    tt::jit_build::utils::FileRenamer log_file(elf_name + ".log");
    fs::remove(log_file.path());
    if (!tt::jit_build::utils::exec_command(args, out_dir, log_file.path())) {
        build_failure(target.target_name, "link", format_args(args), log_file.path());
    }

    tt::jit_build::utils::FileRenamer dephash_file(elf_name + ".dephash");
    std::ofstream hash_file(dephash_file.path());
    tt::jit_build::write_dependency_hashes({{elf_name, std::move(link_deps)}}, out_dir, elf_name, hash_file);
    hash_file.close();
    if (hash_file.fail()) {
        fs::remove(dephash_file.path());
    }
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
        std::vector<std::string> link_obj_paths;
        link_obj_paths.reserve(num_objs);
        for (size_t i = 0; i < num_objs; ++i) {
            std::string temp_path = out_dir + temp_objs[i];
            if (!compiled[i]) {
                std::error_code ec;
                fs::create_hard_link(out_dir + target.objs[i], temp_path, ec);
                if (ec) {
                    fs::copy_file(out_dir + target.objs[i], temp_path, fs::copy_options::overwrite_existing);
                }
            }
            link_obj_paths.push_back(std::move(temp_path));
        }
        link_one(gpp, target, out_dir, link_obj_paths);
    }

    for (size_t i = 0; i < num_objs; ++i) {
        fs::path src_path = out_dir + temp_objs[i];
        fs::path dst_path = out_dir + target.objs[i];
        if (compiled[i]) {
            fs::rename(src_path, dst_path);
            // A preprocessed (.ii) input has no #includes, so -MMD yields an empty .d and no
            // .dephash is written. Tolerate its absence: a missing dephash conservatively forces
            // a recompile next time, which is correct (and the common fresh-cache farm case).
            std::error_code dephash_ec;
            fs::rename(fs::path(src_path).concat(".dephash"), fs::path(dst_path).concat(".dephash"), dephash_ec);
        } else if (fs::exists(src_path)) {
            fs::remove(src_path);
        }
    }

    std::string elf_path = out_dir + target.target_name + ".elf";
    tt::tt_metal::jit_server::ElfBlob blob;
    blob.name = target.target_name;
    blob.data = tt::jit_build::utils::read_file_bytes(elf_path);
    response.elf_blobs.push_back(std::move(blob));
}

tt::tt_metal::jit_server::CompileResponse compile_callback(const tt::tt_metal::jit_server::CompileRequest& request) {
    tt::tt_metal::jit_server::CompileResponse response;
    auto request_start = std::chrono::steady_clock::now();
    int outstanding = g_outstanding_compiles.fetch_add(1) + 1;

    try {
        validate_safe_relative_path(request.kernel_name, "kernel_name");
        for (const auto& target : request.targets) {
            validate_safe_relative_path(target.target_name, "target_name");
            for (const auto& obj : target.objs) {
                validate_safe_relative_path(obj, "obj");
            }
        }
        for (const auto& file : request.generated_files) {
            validate_safe_relative_path(file.name, "generated_file name");
        }

        log_info(
            tt::LogMetal,
            "compile {}: targets={} genfiles={} outstanding={}",
            request.kernel_name,
            request.targets.size(),
            request.generated_files.size(),
            outstanding);

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

        // Re-root the client's toolchain path onto this server's tree (sfpi g++ lives under the
        // client's TT_METAL_HOME, which is absent here when the client's tree is at a different
        // path). No-op if the roots match or the toolchain is a machine-global /opt sfpi.
        std::string gpp = request.gpp;
        reroot_to_server(gpp, request.client_root);

        for (const auto& target : request.targets) {
            tt::tt_metal::jit_server::TargetRecipe resolved_target = target;
            if (!resolved_target.weakened_firmware_name.empty()) {
                resolved_target.weakened_firmware_name =
                    resolve_uploaded_firmware_path(request.build_key, resolved_target).string();
            }
            // Re-root the remaining client-absolute paths: the -T linker script (in lflags), the
            // linker script (dephash), the hw link objects, and any -I includes (cleared under
            // preprocess-and-ship, harmless otherwise).
            reroot_to_server(resolved_target.includes, request.client_root);
            reroot_to_server(resolved_target.lflags, request.client_root);
            reroot_to_server(resolved_target.linker_script, request.client_root);
            reroot_to_server(resolved_target.extra_link_objs, request.client_root);
            std::string out_dir = target_cache_dir(request.build_key, request.kernel_name, target.target_name);
            build_target(gpp, resolved_target, out_dir, response);
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

    // Server's own build root — the re-root target for client-shipped toolchain/link paths.
    const char* home_env = std::getenv("TT_METAL_HOME");
    if (home_env != nullptr && *home_env != '\0') {
        g_server_root = home_env;
        if (g_server_root.back() != '/') {
            g_server_root.push_back('/');
        }
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    tt::tt_metal::jit_server::JitCompileServerController server(compile_callback, upload_firmware_callback);
    server.start(endpoint);
    log_info(tt::LogMetal, "JIT compile server listening on {}", endpoint);
    log_info(tt::LogMetal, "JIT compile server cache root: {}", g_server_cache_root);
    log_info(
        tt::LogMetal,
        "JIT compile server build root (re-root target): {}",
        g_server_root.empty() ? "<TT_METAL_HOME unset; re-root disabled>" : g_server_root);

    while (g_keep_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    server.stop();
    return 0;
}
