// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "jit_build/types.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);

// Like run_command but bypasses the shell entirely by using posix_spawn with an explicit
// argument vector.  Immune to shell metacharacter injection.
// |working_dir| is passed as the cwd for the child process (empty = inherit parent cwd).
bool exec_command(const std::vector<std::string>& args, const std::string& working_dir, const std::string& log_file);

// Split a whitespace-delimited string into tokens (no shell quoting support).
std::vector<std::string> tokenize_flags(const std::string& flags);

// What a gpp invocation should do with its source.
enum class GppAction {
    Compile,     // -c -o <out_obj> <src> -MF <dep>   (produces an object + .d depfile)
    Preprocess,  // -E -o <out>        <src>          (self-contained .ii; keeps line markers)
};
// Build the argv for a single gpp invocation from recipe fields. The argv is for
// exec_command() (posix_spawn, NO shell), so defines/flags containing shell metacharacters
// (e.g. -DFULL_KERNEL_NAME="<name>") are passed verbatim — each define is
// its own argv element and is never re-quoted or shell-split. `opt_level` is the bare level
// without the leading dash (e.g. "O3"). `dep_path` is used only for Compile.
// One argv builder shared by the local compile, the remote compile server, and preprocess-and-ship.
std::vector<std::string> build_gpp_argv(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::vector<std::string>& defines,
    const std::string& src,
    GppAction action,
    const std::string& out_path,
    const std::string& dep_path = "");

// Filename, within a kernel's generated-files directory, of the header carrying the named
// compile-time-arg map (KERNEL_COMPILE_TIME_ARG_MAP, consumed by hw/inc/api/compile_time_args.h).
//
// The map travels as a force-included (-include) header instead of a
// -DKERNEL_COMPILE_TIME_ARG_MAP=... define because one define is one argv element, and the kernel
// caps a single element at MAX_ARG_STRLEN (32 * PAGE_SIZE = 128 KB) no matter how the compiler is
// spawned -- posix_spawn removes the whole-command ceiling, not the per-element one. Kernels
// emitting thousands of long named args cross it on the strength of this one define alone, and the
// only symptom is "posix_spawnp failed for .../riscv-tt-elf-g++: Argument list too long", which
// names neither the define nor a size limit. File contents are not subject to that limit.
//
// A bare filename (not a path) is deliberate: it resolves through the "-I.." carried by every
// compile, since a kernel's generated-files directory is the parent of its per-target out_dir on
// both the local and the JIT-compile-server layouts. That keeps one recipe valid on both sides of
// the RPC, where an absolute client-side path would not exist on the server.
inline constexpr std::string_view NAMED_CT_ARG_MAP_HEADER = "named_ct_arg_map_generated.h";

// Render |named_args| as the initializer body of KERNEL_COMPILE_TIME_ARG_MAP:
//   {"a",1},{"b",2}
// Empty when |named_args| is empty. Entries are emitted in ascending key order so the text is
// byte-identical for a given arg set: std::unordered_map iteration order is not reproducible, and
// this text feeds both the on-disk generated header (per-object dephash cache) and, for emule, the
// wrapper source.
std::string format_named_ct_arg_map(const std::unordered_map<std::string, std::uint32_t>& named_args);

// Full contents of NAMED_CT_ARG_MAP_HEADER for |named_args|.
std::string format_named_ct_arg_map_header(const std::unordered_map<std::string, std::uint32_t>& named_args);

void create_file(const std::string& file_path_str);

// Read the entire contents of a binary file into a byte vector.
// Throws std::runtime_error if the file cannot be read or if the read is incomplete.
std::vector<std::uint8_t> read_file_bytes(const std::string& path);

// Read regular files in |dir| and return them as (filename, content) entries.
// When |extensions| is non-empty, only files whose extension matches one of the
// entries (e.g. ".h", ".cpp") are included.
// Returns an empty vector if |dir| does not exist or is not a directory.
std::vector<tt::jit_build::GeneratedFile> read_directory_files(
    const std::string& dir, std::span<const std::string> extensions = {});

// An RAII wrapper that generates a temporary filename and renames the file on destruction.
// This is to allow multiple processes to write to the same target file without clobbering each other.
class FileRenamer {
public:
    FileRenamer(const std::string& target_path);
    FileRenamer(const FileRenamer&) = delete;
    FileRenamer& operator=(const FileRenamer&) = delete;
    FileRenamer(FileRenamer&&) = default;
    FileRenamer& operator=(FileRenamer&&) = default;
    ~FileRenamer();

    const std::string& path() const { return temp_path_; }
    static std::string generate_temp_path(const std::filesystem::path& target_path);

private:
    std::string temp_path_;
    std::string target_path_;
    static uint64_t unique_id_;
};

}  // namespace tt::jit_build::utils
