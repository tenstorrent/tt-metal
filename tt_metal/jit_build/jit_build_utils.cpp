// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <mutex>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tt-logger/tt-logger.hpp>

#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose) {
    TTZoneScopedD(JIT);
    TTZoneTextD(JIT, cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;

    if (verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
        ret = system(cmd.c_str());
    } else {
        std::string redirected_cmd = cmd + " >> " + log_file + " 2>&1";
        ret = system(redirected_cmd.c_str());
    }

    return (ret == 0);
}

std::vector<std::string> tokenize_flags(const std::string& flags) {
    std::vector<std::string> tokens;
    std::size_t i = 0;
    while (i < flags.size()) {
        while (i < flags.size() && std::isspace(static_cast<unsigned char>(flags[i]))) {
            ++i;
        }
        if (i >= flags.size()) {
            break;
        }
        std::size_t start = i;
        while (i < flags.size() && !std::isspace(static_cast<unsigned char>(flags[i]))) {
            ++i;
        }
        tokens.emplace_back(flags, start, i - start);
    }
    return tokens;
}

std::vector<std::string> build_gpp_argv(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::vector<std::string>& defines,
    const std::string& src,
    GppAction action,
    const std::string& out_path,
    const std::string& dep_path) {
    std::vector<std::string> args = tokenize_flags(gpp);
    args.push_back("-" + opt_level);
    auto append = [&args](const std::string& flags) {
        auto toks = tokenize_flags(flags);
        args.insert(args.end(), std::make_move_iterator(toks.begin()), std::make_move_iterator(toks.end()));
    };
    append(cflags);
    append(includes);
    // Each define is one argv element, passed verbatim (no shell) — this is what makes
    // map-valued defines like -DKERNEL_COMPILE_TIME_ARG_MAP={"cb_in0",1},... survive.
    args.insert(args.end(), defines.begin(), defines.end());
    switch (action) {
        case GppAction::Compile:
            args.push_back("-c");
            args.push_back("-o");
            args.push_back(out_path);
            args.push_back(src);
            args.push_back("-MF");
            args.push_back(dep_path);
            break;
        case GppAction::Preprocess:
            // Keep line markers: the .ii is later compiled with -fpreprocessed, which uses them to
            // keep -Werror suppressed inside system headers and fatal only on kernel code.
            args.push_back("-E");
            args.push_back("-o");
            args.push_back(out_path);
            args.push_back(src);
            break;
    }
    return args;
}

bool exec_command(const std::vector<std::string>& args, const std::string& working_dir, const std::string& log_file) {
    if (args.empty()) {
        return false;
    }

    // Linux rejects an exec when either the complete argv+environment exceeds
    // ARG_MAX or one argument exceeds MAX_ARG_STRLEN. Named kernel CT-arg maps
    // can hit both limits because the whole map is intentionally one -D argv. A
    // response file gets the driver past execve(), but GCC expands it and passes
    // the same giant -D to cc1plus, which then fails at the next execve(). Move
    // oversized macro definitions into a forced-include header first. Keep that
    // header beside the compile log: GCC records it in the .d file, so it must
    // remain readable for dependency-hash validation after the compiler exits.
    constexpr std::size_t max_inline_arg_bytes = 64 * 1024;
    constexpr std::size_t max_inline_argv_bytes = 512 * 1024;
    std::size_t argv_bytes = 0;
    for (const auto& arg : args) {
        argv_bytes += arg.size() + 1;
    }

    std::vector<bool> materialize_define(args.size(), false);
    std::size_t rewritten_argv_bytes = argv_bytes;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i].starts_with("-D") && args[i].size() > max_inline_arg_bytes) {
            materialize_define[i] = true;
            rewritten_argv_bytes -= args[i].size() + 1;
        }
    }
    if (rewritten_argv_bytes > max_inline_argv_bytes) {
        std::vector<std::size_t> define_indices;
        for (std::size_t i = 1; i < args.size(); ++i) {
            if (!materialize_define[i] && args[i].starts_with("-D") && args[i].size() > 2) {
                define_indices.push_back(i);
            }
        }
        std::sort(define_indices.begin(), define_indices.end(), [&args](std::size_t lhs, std::size_t rhs) {
            return args[lhs].size() > args[rhs].size();
        });
        for (const std::size_t i : define_indices) {
            materialize_define[i] = true;
            rewritten_argv_bytes -= args[i].size() + 1;
            if (rewritten_argv_bytes <= max_inline_argv_bytes / 2) {
                break;
            }
        }
    }

    const bool has_materialized_defines =
        std::find(materialize_define.begin(), materialize_define.end(), true) != materialize_define.end();
    std::vector<std::string> rewritten_args;
    std::filesystem::path defines_path;
    if (has_materialized_defines) {
        const std::filesystem::path defines_target =
            log_file.empty() ? (working_dir.empty() ? std::filesystem::temp_directory_path() / "tt_jit_large_defines.h"
                                                    : std::filesystem::path(working_dir) / "tt_jit_large_defines.h")
                             : std::filesystem::path(log_file + ".defines.h");
        defines_path = FileRenamer::generate_temp_path(defines_target);
        std::ofstream defines_file(defines_path, std::ios::out | std::ios::trunc);
        if (!defines_file.is_open()) {
            log_error(tt::LogBuildKernels, "Failed to create JIT defines file '{}'", defines_path.string());
            return false;
        }
        for (std::size_t i = 1; i < args.size(); ++i) {
            if (!materialize_define[i]) {
                continue;
            }
            const std::string_view definition(args[i].data() + 2, args[i].size() - 2);
            const std::size_t equals = definition.find('=');
            if (equals == std::string_view::npos) {
                defines_file << "#define " << definition << " 1\n";
            } else {
                defines_file << "#define " << definition.substr(0, equals) << ' ' << definition.substr(equals + 1)
                             << '\n';
            }
        }
        defines_file.close();
        if (defines_file.fail()) {
            std::error_code error;
            std::filesystem::remove(defines_path, error);
            log_error(tt::LogBuildKernels, "Failed to write JIT defines file '{}'", defines_path.string());
            return false;
        }

        rewritten_args.reserve(args.size() + 2);
        bool inserted_include = false;
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (materialize_define[i]) {
                if (!inserted_include) {
                    rewritten_args.push_back("-include");
                    rewritten_args.push_back(defines_path.string());
                    inserted_include = true;
                }
            } else {
                rewritten_args.push_back(args[i]);
            }
        }
    }
    const auto& command_args = has_materialized_defines ? rewritten_args : args;

    argv_bytes = 0;
    bool use_response_file = false;
    for (const auto& arg : command_args) {
        argv_bytes += arg.size() + 1;
        use_response_file |= arg.size() > max_inline_arg_bytes;
    }
    use_response_file |= argv_bytes > max_inline_argv_bytes;

    std::vector<std::string> spawn_args;
    std::filesystem::path response_path;
    const auto remove_response_file = [&response_path]() {
        if (!response_path.empty()) {
            std::error_code error;
            std::filesystem::remove(response_path, error);
        }
    };
    if (use_response_file) {
        const std::filesystem::path response_target =
            log_file.empty() ? (working_dir.empty() ? std::filesystem::temp_directory_path() / "tt_jit_args.rsp"
                                                    : std::filesystem::path(working_dir) / "tt_jit_args.rsp")
                             : std::filesystem::path(log_file + ".rsp");
        response_path = FileRenamer::generate_temp_path(response_target);
        std::ofstream response(response_path, std::ios::out | std::ios::trunc);
        if (!response.is_open()) {
            log_error(tt::LogBuildKernels, "Failed to create JIT response file '{}'", response_path.string());
            return false;
        }
        for (std::size_t i = 1; i < command_args.size(); ++i) {
            response << '"';
            for (const char ch : command_args[i]) {
                if (ch == '\\' || ch == '"') {
                    response << '\\';
                }
                response << ch;
            }
            response << "\"\n";
        }
        response.close();
        if (response.fail()) {
            remove_response_file();
            log_error(tt::LogBuildKernels, "Failed to write JIT response file '{}'", response_path.string());
            return false;
        }
        spawn_args = {command_args.front(), "@" + response_path.string()};
    }
    const auto& effective_args = use_response_file ? spawn_args : command_args;

    // Build a null-terminated argv array for posix_spawn.
    std::vector<const char*> argv;
    argv.reserve(effective_args.size() + 1);
    for (const auto& arg : effective_args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);

    int log_fd = -1;
    if (!log_file.empty()) {
        log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (log_fd < 0) {
            remove_response_file();
            posix_spawn_file_actions_destroy(&file_actions);
            return false;
        }
        posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDERR_FILENO);
    }

    if (!working_dir.empty()) {
        posix_spawn_file_actions_addchdir_np(&file_actions, working_dir.c_str());
    }

    pid_t pid = 0;
    int spawn_ret =
        posix_spawnp(&pid, argv[0], &file_actions, nullptr, const_cast<char* const*>(argv.data()), ::environ);

    if (log_fd >= 0) {
        close(log_fd);
    }
    posix_spawn_file_actions_destroy(&file_actions);

    if (spawn_ret != 0) {
        remove_response_file();
        log_error(tt::LogBuildKernels, "posix_spawnp failed for '{}': {}", argv[0], std::strerror(spawn_ret));
        return false;
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            remove_response_file();
            log_error(tt::LogBuildKernels, "waitpid failed for '{}': {}", argv[0], std::strerror(errno));
            return false;
        }
    }
    remove_response_file();

    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot read file: " + path);
    }
    std::streampos pos = file.tellg();
    if (pos == std::streampos(-1)) {
        throw std::runtime_error("Cannot determine size of file: " + path);
    }
    auto byte_count = static_cast<std::streamsize>(pos);
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(byte_count));
    file.read(reinterpret_cast<char*>(data.data()), byte_count);
    if (file.gcount() != byte_count || (!file && !file.eof())) {
        throw std::runtime_error(
            fmt::format("Failed to read file '{}' fully (expected {} bytes, got {})", path, byte_count, file.gcount()));
    }
    return data;
}

std::vector<tt::jit_build::GeneratedFile> read_directory_files(
    const std::string& dir, std::span<const std::string> extensions) {
    namespace fs = std::filesystem;
    std::vector<tt::jit_build::GeneratedFile> files;
    if (!fs::is_directory(dir)) {
        return files;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!extensions.empty() &&
            std::find(extensions.begin(), extensions.end(), entry.path().extension().string()) == extensions.end()) {
            continue;
        }
        // Tolerate concurrent writers: another process compiling the same kernel into this
        // shared cache dir may rename a FileRenamer temp file away between enumeration and read.
        // Skip files that vanish or fail to read rather than aborting the whole upload.
        try {
            files.push_back({entry.path().filename().string(), read_file_bytes(entry.path().string())});
        } catch (const std::runtime_error& e) {
            log_debug(
                tt::LogBuildKernels,
                "Skipping file that could not be read during directory scan of {}: {}",
                dir,
                e.what());
        }
    }
    return files;
}

void create_file(const std::string& file_path_str) {
    namespace fs = std::filesystem;

    fs::path file_path(file_path_str);
    fs::create_directories(file_path.parent_path());

    std::ofstream ofs(file_path);
    ofs.close();
}

uint64_t FileRenamer::unique_id_ = []() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> distr;
    return distr(rd);
}();

std::string FileRenamer::generate_temp_path(const std::filesystem::path& target_path) {
    // unique_id_ is initialized once per process, so a fork()ed child inherits the
    // parent's value and would otherwise generate byte-identical temp paths. Mix in
    // the live pid so forked siblings -- e.g. pytest --forked test processes sharing
    // one kernel cache -- never collide on the same temp file.
    //
    // Formatted in one call rather than through an intermediate tag string: this runs
    // once per source file during JIT setup, and the extra allocation measured more
    // expensive than the getpid() syscall it accompanies.
    std::filesystem::path path(target_path);
    if (path.has_extension()) {
        path.replace_extension(fmt::format("{}_{}{}", unique_id_, ::getpid(), path.extension().string()));
        return path.string();
    }
    return fmt::format("{}.{}_{}", target_path.string(), unique_id_, ::getpid());
}

FileRenamer::FileRenamer(const std::string& target_path) :
    temp_path_(generate_temp_path(target_path)), target_path_(target_path) {}

FileRenamer::~FileRenamer() {
    std::error_code ec;
    if (target_path_.empty()) {
        return;
    }
    std::filesystem::rename(temp_path_, target_path_, ec);
    if (ec) {
        log_error(
            tt::LogBuildKernels,
            "Failed to rename temporary file {} to target file {}: {}",
            temp_path_,
            target_path_,
            ec.message());
    }
}

}  // namespace tt::jit_build::utils
