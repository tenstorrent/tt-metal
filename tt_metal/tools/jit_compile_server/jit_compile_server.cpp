// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "common/filesystem_utils.hpp"
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
fs::path g_server_cache_root = kDefaultServerCacheRoot;

bool safe_exists_value_or_false(const fs::path& path) { return tt::filesystem::safe_exists(path).value_or(false); }

void ensure_safe_create_directories(const fs::path& path) {
    if (!tt::filesystem::safe_create_directories(path)) {
        throw std::runtime_error(fmt::format("Failed to create directory: {}", path.string()));
    }
}

void ensure_safe_hard_link_or_copy(const fs::path& target, const fs::path& link) {
    if (!tt::filesystem::safe_hard_link_or_copy(target, link)) {
        throw std::runtime_error(
            fmt::format("Failed to create hard link or copy from {} to {}", target.string(), link.string()));
    }
}

void ensure_safe_rename(const fs::path& src, const fs::path& dst) {
    if (!tt::filesystem::safe_rename(src, dst)) {
        throw std::runtime_error(fmt::format("Failed to rename {} to {}", src.string(), dst.string()));
    }
}

std::error_code stream_open_error_code() {
    if (errno != 0) {
        return std::error_code(errno, std::system_category());
    }
    return std::make_error_code(std::errc::io_error);
}

template <typename Stream>
void reset_stream_before_open(Stream& stream) {
    if (stream.is_open()) {
        stream.close();
    }
    stream.clear();
}

bool open_ifstream_with_retry(
    std::ifstream& stream, const fs::path& path, std::error_code& ec, std::ios_base::openmode mode = std::ios::in) {
    return tt::filesystem::retry_on_estale_ec(
        [&](std::error_code& open_ec) {
            reset_stream_before_open(stream);
            errno = 0;
            stream.open(path, mode | std::ios::in);
            if (!stream.is_open()) {
                open_ec = stream_open_error_code();
                return false;
            }
            return true;
        },
        ec);
}

bool open_ofstream_with_retry(
    std::ofstream& stream, const fs::path& path, std::error_code& ec, std::ios_base::openmode mode = std::ios::out) {
    return tt::filesystem::retry_on_estale_ec(
        [&](std::error_code& open_ec) {
            reset_stream_before_open(stream);
            errno = 0;
            stream.open(path, mode | std::ios::out);
            if (!stream.is_open()) {
                open_ec = stream_open_error_code();
                return false;
            }
            return true;
        },
        ec);
}

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

fs::path normalize_cache_root(const std::string& cache_root) {
    if (cache_root.empty()) {
        return fs::path(kDefaultServerCacheRoot);
    }
    return fs::path(cache_root);
}

// All client-supplied path components (kernel_name, target_name, obj names, generated file
// names) are validated by validate_safe_relative_path() before reaching these helpers.
fs::path kernel_cache_dir(std::uint64_t build_key, const std::string& kernel_name) {
    return g_server_cache_root / std::to_string(build_key) / "kernels" / kernel_name;
}

fs::path target_cache_dir(std::uint64_t build_key, const std::string& kernel_name, const std::string& target_name) {
    return kernel_cache_dir(build_key, kernel_name) / target_name;
}

void handle_signal(int /*signal*/) { g_keep_running.store(false); }

fs::path firmware_cache_dir(std::uint64_t build_key, const std::string& target_name) {
    return g_server_cache_root / std::to_string(build_key) / "firmware" / target_name;
}

// Resolve firmware path from the server cache populated by uploadFirmware RPC.
// PoC limitation: uploaded firmware is assumed durable in the configured server cache root.
// If the server cache is cleared after upload, compile will fail until the client re-uploads.
fs::path resolve_uploaded_firmware_path(std::uint64_t build_key, const tt::tt_metal::jit_server::TargetRecipe& target) {
    if (target.weakened_firmware_name.empty()) {
        throw std::runtime_error(
            fmt::format("Internal error: expected non-empty weakened_firmware_name for target {}", target.target_name));
    }
    const fs::path firmware_file = fs::path(target.weakened_firmware_name).filename();
    const fs::path fw_dir = firmware_cache_dir(build_key, target.target_name);
    fs::path candidate = fw_dir / firmware_file;
    auto candidate_exists = tt::filesystem::safe_exists(candidate);
    if (candidate_exists.value_or(false)) {
        return candidate;
    }
    if (!candidate_exists.has_value()) {
        throw std::runtime_error(fmt::format(
            "Failed to check firmware artifact for target {} at {}", target.target_name, candidate.string()));
    }

    throw std::runtime_error(fmt::format(
        "Firmware artifact not found for build_key {} target {} file {}. "
        "Expected at {}. Ensure the client uploads firmware via uploadFirmware RPC before compiling.",
        build_key,
        target.target_name,
        firmware_file.string(),
        candidate.string()));
}

void build_failure(const std::string& target, const std::string& op, const std::string& cmd, const fs::path& log_file) {
    std::ifstream file;
    std::error_code open_ec;
    if (open_ifstream_with_retry(file, log_file, open_ec)) {
        std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        throw std::runtime_error(fmt::format("{} {} failure -- cmd: {}\nLog: {}", target, op, cmd, log_contents));
    }
    throw std::runtime_error(fmt::format(
        "{} {} failure -- cmd: {} (log file {} could not be opened: {})",
        target,
        op,
        cmd,
        log_file.string(),
        open_ec.message()));
}

bool need_compile(const fs::path& out_dir, const fs::path& obj) {
    return !safe_exists_value_or_false(out_dir / obj) || !tt::jit_build::dependencies_up_to_date(out_dir, obj);
}

bool need_link(const fs::path& out_dir, const std::string& target_name) {
    fs::path elf_path = out_dir / (target_name + ".elf");
    return !safe_exists_value_or_false(elf_path) || !tt::jit_build::dependencies_up_to_date(out_dir, elf_path);
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
    const fs::path& out_dir,
    size_t src_index,
    const fs::path& temp_obj) {
    std::vector<std::string> args;
    append_tokenized(args, gpp);
    args.push_back("-" + target.compiler_opt_level);
    append_tokenized(args, target.cflags);
    append_tokenized(args, target.includes);

    fs::path obj_path = out_dir / target.objs[src_index];
    fs::path obj_temp_path = out_dir / temp_obj;
    fs::path temp_d_path = fs::path(obj_temp_path).replace_extension("d");
    args.push_back("-c");
    args.push_back("-o");
    args.push_back(obj_temp_path.string());
    args.push_back(target.srcs[src_index]);
    args.push_back("-MF");
    args.push_back(temp_d_path.string());
    append_tokenized(args, target.defines);

    fs::path log_path = obj_path;
    log_path += ".log";
    tt::jit_build::utils::FileRenamer log_file(log_path);
    tt::filesystem::safe_remove(log_file.path());
    if (!tt::jit_build::utils::exec_command(args, out_dir, log_file.path())) {
        build_failure(target.target_name, "compile", format_args(args), log_file.path());
    }
    fs::path dephash_path = obj_temp_path;
    dephash_path += ".dephash";
    tt::jit_build::write_dependency_hashes(out_dir, obj_temp_path, dephash_path);
    tt::filesystem::safe_remove(temp_d_path);
}

void link_one(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const fs::path& out_dir,
    const std::vector<fs::path>& link_obj_paths) {
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
    for (const auto& link_obj_path : link_obj_paths) {
        args.push_back(link_obj_path.string());
    }
    fs::path elf_path = out_dir / (target.target_name + ".elf");
    tt::jit_build::utils::FileRenamer elf_file(elf_path);
    args.push_back("-o");
    args.push_back(elf_file.path().string());

    fs::path log_path = elf_path;
    log_path += ".log";
    tt::jit_build::utils::FileRenamer log_file(log_path);
    tt::filesystem::safe_remove(log_file.path());
    if (!tt::jit_build::utils::exec_command(args, out_dir, log_file.path())) {
        build_failure(target.target_name, "link", format_args(args), log_file.path());
    }

    fs::path dephash_path = elf_path;
    dephash_path += ".dephash";
    tt::jit_build::utils::FileRenamer dephash_file(dephash_path);
    std::ofstream hash_file;
    [[maybe_unused]] std::error_code hash_open_ec;
    const bool hash_file_opened = open_ofstream_with_retry(hash_file, dephash_file.path(), hash_open_ec);
    if (hash_file_opened) {
        tt::jit_build::write_dependency_hashes(
            {{elf_path.string(), std::move(link_deps)}}, out_dir, elf_path, hash_file);
        hash_file.close();
        if (hash_file.fail()) {
            tt::filesystem::safe_remove(dephash_file.path());
        }
    } else {
        tt::filesystem::safe_remove(dephash_file.path());
    }
}

std::vector<std::uint8_t> read_file_bytes(const fs::path& path) {
    std::ifstream file;
    std::error_code open_ec;
    if (!open_ifstream_with_retry(file, path, open_ec, std::ios::binary | std::ios::ate)) {
        throw std::runtime_error(fmt::format("Cannot read ELF file {}: {}", path.string(), open_ec.message()));
    }
    std::streampos pos = file.tellg();
    if (pos == std::streampos(-1)) {
        throw std::runtime_error("Cannot determine size of ELF file: " + path.string());
    }
    auto byte_count = static_cast<std::streamsize>(pos);
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<size_t>(byte_count));
    file.read(reinterpret_cast<char*>(data.data()), byte_count);
    if (file.gcount() != byte_count || (!file && !file.eof())) {
        throw std::runtime_error(fmt::format(
            "Failed to read ELF file '{}' fully (expected {} bytes, got {})",
            path.string(),
            byte_count,
            file.gcount()));
    }
    return data;
}

void build_target(
    const std::string& gpp,
    const tt::tt_metal::jit_server::TargetRecipe& target,
    const fs::path& out_dir,
    tt::tt_metal::jit_server::CompileResponse& response) {
    if (target.srcs.size() != target.objs.size()) {
        throw std::runtime_error("srcs and objs must have the same size for target " + target.target_name);
    }

    ensure_safe_create_directories(out_dir);

    const size_t num_objs = target.objs.size();
    std::vector<fs::path> temp_objs;
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
        std::vector<fs::path> link_obj_paths;
        link_obj_paths.reserve(num_objs);
        for (size_t i = 0; i < num_objs; ++i) {
            fs::path temp_path = out_dir / temp_objs[i];
            if (!compiled[i]) {
                ensure_safe_hard_link_or_copy(out_dir / target.objs[i], temp_path);
            }
            link_obj_paths.push_back(std::move(temp_path));
        }
        link_one(gpp, target, out_dir, link_obj_paths);
    }

    for (size_t i = 0; i < num_objs; ++i) {
        fs::path src_path = out_dir / temp_objs[i];
        fs::path dst_path = out_dir / target.objs[i];
        if (compiled[i]) {
            ensure_safe_rename(src_path, dst_path);
            fs::path src_dephash = src_path;
            src_dephash += ".dephash";
            fs::path dst_dephash = dst_path;
            dst_dephash += ".dephash";
            ensure_safe_rename(src_dephash, dst_dephash);
        } else if (safe_exists_value_or_false(src_path)) {
            tt::filesystem::safe_remove(src_path);
        }
    }

    fs::path elf_path = out_dir / (target.target_name + ".elf");
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
            ensure_safe_create_directories(genfiles_dir);
            for (const auto& file : request.generated_files) {
                fs::path target_path = genfiles_dir / file.name;
                fs::path temp_path = target_path;
                temp_path += ".tmp";

                std::error_code cleanup_ec;
                fs::remove(temp_path, cleanup_ec);

                std::ofstream out;
                std::error_code open_ec;
                if (!open_ofstream_with_retry(out, temp_path, open_ec, std::ios::binary)) {
                    throw std::runtime_error(
                        fmt::format("Cannot create file {}: {}", target_path.string(), open_ec.message()));
                }

                try {
                    out.write(
                        reinterpret_cast<const char*>(file.content.data()),
                        static_cast<std::streamsize>(file.content.size()));
                    if (!out) {
                        throw std::runtime_error("Failed to write file: " + target_path.string());
                    }

                    out.close();
                    if (!out) {
                        throw std::runtime_error("Failed to finalize file: " + target_path.string());
                    }
                } catch (...) {
                    std::error_code remove_ec;
                    fs::remove(temp_path, remove_ec);
                    throw;
                }

                std::error_code rename_ec;
                fs::rename(temp_path, target_path, rename_ec);
                if (rename_ec) {
                    std::error_code remove_ec;
                    fs::remove(temp_path, remove_ec);
                    throw std::runtime_error(
                        fmt::format(
                            "Failed to publish generated file {}: {}", target_path.string(), rename_ec.message()));
                }
            }
        }

        for (const auto& target : request.targets) {
            tt::tt_metal::jit_server::TargetRecipe resolved_target = target;
            if (!resolved_target.weakened_firmware_name.empty()) {
                resolved_target.weakened_firmware_name =
                    resolve_uploaded_firmware_path(request.build_key, resolved_target).string();
            }
            fs::path out_dir = target_cache_dir(request.build_key, request.kernel_name, target.target_name);
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

            fs::path fw_dir = firmware_cache_dir(request.build_key, safe_target);
            ensure_safe_create_directories(fw_dir);
            fs::path target_path = fw_dir / safe_file;

            tt::jit_build::utils::FileRenamer tmp(target_path);
            std::ofstream out;
            std::error_code open_ec;
            if (!open_ofstream_with_retry(out, tmp.path(), open_ec, std::ios::binary)) {
                throw std::runtime_error(
                    fmt::format("Cannot create firmware file {}: {}", target_path.string(), open_ec.message()));
            }
            out.write(
                reinterpret_cast<const char*>(artifact.data.data()),
                static_cast<std::streamsize>(artifact.data.size()));
            if (!out) {
                throw std::runtime_error("Failed to write firmware file: " + target_path.string());
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
    log_info(tt::LogMetal, "JIT compile server cache root: {}", g_server_cache_root.string());

    while (g_keep_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    server.stop();
    return 0;
}
