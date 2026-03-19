// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/jit_server/jit_compile_server_controller.hpp"
#include "impl/jit_server/types.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_keep_running{true};
std::atomic<int> g_outstanding_compiles{0};

constexpr const char* kEndpointEnv = "TT_METAL_JIT_SERVER_ENDPOINT";
constexpr const char* kDefaultEndpoint = "0.0.0.0:9876";
constexpr const char* kServerCacheRoot = "/tmp/tt-metal-cache/";

std::string kernel_cache_dir(std::uint64_t build_key, const std::string& kernel_name) {
    return std::string(kServerCacheRoot) + std::to_string(build_key) + "/kernels/" + kernel_name + "/";
}

std::string target_cache_dir(std::uint64_t build_key, const std::string& kernel_name, const std::string& target_name) {
    return kernel_cache_dir(build_key, kernel_name) + target_name + "/";
}

void handle_signal(int /*signal*/) { g_keep_running.store(false); }

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

void compile_one(
    const tt::tt_metal::jit_server::CompileRequest& request,
    const std::string& out_dir,
    size_t src_index,
    const std::string& temp_obj) {
    std::string cmd = fmt::format("cd {} && {}", out_dir, request.gpp);
    cmd += fmt::format("-{} ", request.compiler_opt_level);
    cmd += request.cflags;
    cmd += request.includes;

    std::string obj_path = out_dir + request.objs[src_index];
    std::string obj_temp_path = out_dir + temp_obj;
    std::string temp_d_path = fs::path(obj_temp_path).replace_extension("d").string();
    cmd += fmt::format("-c -o {} {} -MF {} ", obj_temp_path, request.srcs[src_index], temp_d_path);
    cmd += request.defines;

    tt::jit_build::utils::FileRenamer log_file(obj_path + ".log");
    fs::remove(log_file.path());
    if (!tt::jit_build::utils::run_command(cmd, log_file.path(), false)) {
        build_failure(request.target_name, "compile", cmd, log_file.path());
    }
    tt::jit_build::write_dependency_hashes(out_dir, obj_temp_path, obj_temp_path + ".dephash");
    fs::remove(temp_d_path);
}

void link_one(
    const tt::tt_metal::jit_server::CompileRequest& request,
    const std::string& out_dir,
    const std::string& link_objs_str) {
    std::string cmd = fmt::format("cd {} && {}", out_dir, request.gpp);
    cmd += fmt::format("-{} ", request.linker_opt_level);

    std::vector<std::string> link_deps = {request.linker_script};
    if (!request.weakened_firmware_name.empty()) {
        link_deps.push_back(request.weakened_firmware_name);
        if (!request.firmware_is_kernel_object) {
            cmd += "-Wl,--just-symbols=";
        }
        cmd += request.weakened_firmware_name + " ";
    }

    cmd += request.lflags;
    cmd += request.extra_link_objs;
    cmd += link_objs_str;
    std::string elf_name = out_dir + request.target_name + ".elf";
    tt::jit_build::utils::FileRenamer elf_file(elf_name);
    cmd += "-o " + elf_file.path();

    tt::jit_build::utils::FileRenamer log_file(elf_name + ".log");
    fs::remove(log_file.path());
    if (!tt::jit_build::utils::run_command(cmd, log_file.path(), false)) {
        build_failure(request.target_name, "link", cmd, log_file.path());
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
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

tt::tt_metal::jit_server::CompileResponse compile_callback(const tt::tt_metal::jit_server::CompileRequest& request) {
    tt::tt_metal::jit_server::CompileResponse response;
    auto request_start = std::chrono::steady_clock::now();
    int outstanding = g_outstanding_compiles.fetch_add(1) + 1;

    try {
        if (request.srcs.size() != request.objs.size()) {
            throw std::runtime_error("srcs and objs must have the same size");
        }

        std::string out_dir = target_cache_dir(request.build_key, request.kernel_name, request.target_name);
        fs::create_directories(out_dir);

        const size_t num_objs = request.objs.size();
        std::vector<std::string> temp_objs;
        temp_objs.reserve(num_objs);
        for (const auto& obj : request.objs) {
            temp_objs.push_back(tt::jit_build::utils::FileRenamer::generate_temp_path(obj));
        }

        std::vector<bool> compiled(num_objs, false);
        for (size_t i = 0; i < num_objs; ++i) {
            if (need_compile(out_dir, request.objs[i])) {
                compiled[i] = true;
            }
        }

        const size_t recompiled = std::count(compiled.begin(), compiled.end(), true);
        const size_t cache_hit = num_objs - recompiled;
        bool needs_link = need_link(out_dir, request.target_name);

        log_info(
            tt::LogMetal,
            "compile {}{}: obj={} hit={} miss={} link={} outstanding={}",
            request.kernel_name,
            request.target_name,
            num_objs,
            cache_hit,
            recompiled,
            needs_link || recompiled > 0 ? "yes" : "no",
            outstanding);

        for (size_t i = 0; i < num_objs; ++i) {
            if (compiled[i]) {
                compile_one(request, out_dir, i, temp_objs[i]);
            }
        }

        if (recompiled > 0 || needs_link) {
            std::string link_objs_str;
            for (size_t i = 0; i < num_objs; ++i) {
                std::string temp_path = out_dir + temp_objs[i];
                if (!compiled[i]) {
                    std::error_code ec;
                    fs::create_hard_link(out_dir + request.objs[i], temp_path, ec);
                    if (ec) {
                        fs::copy_file(out_dir + request.objs[i], temp_path, fs::copy_options::overwrite_existing);
                    }
                }
                link_objs_str += temp_path + " ";
            }

            link_one(request, out_dir, link_objs_str);
        }

        for (size_t i = 0; i < num_objs; ++i) {
            fs::path src_path = out_dir + temp_objs[i];
            fs::path dst_path = out_dir + request.objs[i];
            if (compiled[i]) {
                fs::rename(src_path, dst_path);
                fs::rename(fs::path(src_path).concat(".dephash"), fs::path(dst_path).concat(".dephash"));
            } else if (fs::exists(src_path)) {
                fs::remove(src_path);
            }
        }

        {
            static std::mutex stats_mutex;
            std::lock_guard lock(stats_mutex);
            std::ofstream stats("/tmp/tt-jit-compile-server-stats.log", std::ios::app);
            if (stats) {
                auto now = std::chrono::system_clock::now();
                std::time_t t = std::chrono::system_clock::to_time_t(now);
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
                stats << buf << "  " << request.kernel_name << request.target_name << "  recompiled=" << recompiled
                      << "  cache_hit=" << cache_hit << "\n";
            }
        }

        std::string elf_path = out_dir + request.target_name + ".elf";
        tt::tt_metal::jit_server::ElfBlob blob;
        blob.name = request.target_name;
        blob.data = read_file_bytes(elf_path);
        response.elf_blobs.push_back(std::move(blob));
        response.success = true;

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - request_start)
                .count();
        log_info(
            tt::LogMetal,
            "done {}{}: {}ms, elf={}B",
            request.kernel_name,
            request.target_name,
            elapsed_ms,
            response.elf_blobs.back().data.size());
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = e.what();
        log_warning(tt::LogMetal, "FAIL {}{}: {}", request.kernel_name, request.target_name, response.error_message);
    }
    g_outstanding_compiles.fetch_sub(1);
    return response;
}

tt::tt_metal::jit_server::PrepareGenfilesResponse prepare_genfiles_callback(
    const tt::tt_metal::jit_server::PrepareGenfilesRequest& request) {
    tt::tt_metal::jit_server::PrepareGenfilesResponse response;

    try {
        const std::string out_dir = kernel_cache_dir(request.build_key, request.kernel_name);
        fs::create_directories(out_dir);
        for (const auto& file : request.files) {
            std::string target_path = out_dir + file.name;
            tt::jit_build::utils::FileRenamer tmp(target_path);
            std::ofstream out(tmp.path(), std::ios::binary);
            if (!out.is_open()) {
                throw std::runtime_error("Cannot create file: " + target_path);
            }
            out.write(
                reinterpret_cast<const char*>(file.content.data()), static_cast<std::streamsize>(file.content.size()));
            if (!out) {
                throw std::runtime_error("Failed to write file: " + target_path);
            }
        }
        response.success = true;
        log_info(tt::LogMetal, "genfiles {}: {} files", request.kernel_name, request.files.size());
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = e.what();
        log_warning(tt::LogMetal, "genfiles FAIL {}: {}", request.kernel_name, response.error_message);
    }
    return response;
}

}  // namespace

int main() {
    const char* endpoint_env = std::getenv(kEndpointEnv);
    const std::string endpoint = endpoint_env != nullptr ? endpoint_env : kDefaultEndpoint;

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    tt::tt_metal::jit_server::JitCompileServerController server(compile_callback, prepare_genfiles_callback);
    server.start(endpoint);
    log_info(tt::LogMetal, "JIT compile server listening on {}", endpoint);

    while (g_keep_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    server.stop();
    return 0;
}
