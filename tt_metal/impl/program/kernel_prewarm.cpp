// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/program/kernel_prewarm.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/jit_server/rpc.capnp.h"
#include "jit_build/build.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/types.hpp"

namespace tt::tt_metal::kernel_prewarm {

namespace fs = std::filesystem;

namespace {

const char* env_or_null(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

// ------------------------------------------------------------------------------------------------
// Build one target from a portable recipe.
//
// This mirrors build_target()/compile_one()/link_one() in
// tools/jit_compile_server/jit_compile_server.cpp. It is copied (not shared) on purpose: the
// prewarm must run in-process with no RPC and no MetalContext, exactly like the server's copy which
// "cannot have dependency on MetalContext". Once both stabilize they should be unified (the server
// already carries that TODO). The one behavioral difference: no CompileResponse blob is produced,
// and .build_state is intentionally left untouched -- on a warm build_key the existing .build_state
// still matches (source/header *content* is not part of build_state_hash_), so refreshing
// .o/.elf/.dephash is enough to turn the later op-by-op build into a cache hit.
// ------------------------------------------------------------------------------------------------

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
    return !fs::exists(out_dir + obj) || !jit_build::dependencies_up_to_date(out_dir, obj);
}

bool need_link(const std::string& out_dir, const std::string& target_name) {
    std::string elf_path = out_dir + target_name + ".elf";
    return !fs::exists(elf_path) || !jit_build::dependencies_up_to_date(out_dir, elf_path);
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
    auto tokens = jit_build::utils::tokenize_flags(flags);
    args.insert(args.end(), std::make_move_iterator(tokens.begin()), std::make_move_iterator(tokens.end()));
}

void compile_one(
    const std::string& gpp,
    const jit_build::TargetRecipe& target,
    const std::string& out_dir,
    size_t src_index,
    const std::string& temp_obj) {
    std::vector<std::string> args;
    append_tokenized(args, gpp);
    args.push_back("-" + target.compiler_opt_level);
    append_tokenized(args, target.cflags);
    append_tokenized(args, target.includes);

    std::string obj_path = out_dir + target.objs[src_index];
    std::string obj_temp_path = out_dir + temp_obj;
    std::string temp_d_path = fs::path(obj_temp_path).replace_extension("d").string();
    args.push_back("-c");
    args.push_back("-o");
    args.push_back(obj_temp_path);
    args.push_back(target.srcs[src_index]);
    args.push_back("-MF");
    args.push_back(temp_d_path);
    args.insert(args.end(), target.defines.begin(), target.defines.end());

    jit_build::utils::FileRenamer log_file(obj_path + ".log");
    fs::remove(log_file.path());
    if (!jit_build::utils::exec_command(args, out_dir, log_file.path())) {
        build_failure(target.target_name, "compile", format_args(args), log_file.path());
    }
    jit_build::write_dependency_hashes(out_dir, obj_temp_path, obj_temp_path + ".dephash");
    fs::remove(temp_d_path);
}

void link_one(
    const std::string& gpp,
    const jit_build::TargetRecipe& target,
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
    jit_build::utils::FileRenamer elf_file(elf_name);
    args.push_back("-o");
    args.push_back(elf_file.path());

    jit_build::utils::FileRenamer log_file(elf_name + ".log");
    fs::remove(log_file.path());
    if (!jit_build::utils::exec_command(args, out_dir, log_file.path())) {
        build_failure(target.target_name, "link", format_args(args), log_file.path());
    }

    jit_build::utils::FileRenamer dephash_file(elf_name + ".dephash");
    std::ofstream hash_file(dephash_file.path());
    jit_build::write_dependency_hashes({{elf_name, std::move(link_deps)}}, out_dir, elf_name, hash_file);
    hash_file.close();
    if (hash_file.fail()) {
        fs::remove(dephash_file.path());
    }
}

void build_target(const std::string& gpp, const jit_build::TargetRecipe& target, const std::string& out_dir) {
    if (target.srcs.size() != target.objs.size()) {
        throw std::runtime_error("srcs and objs must have the same size for target " + target.target_name);
    }

    fs::create_directories(out_dir);

    const size_t num_objs = target.objs.size();
    std::vector<std::string> temp_objs;
    temp_objs.reserve(num_objs);
    for (const auto& obj : target.objs) {
        temp_objs.push_back(jit_build::utils::FileRenamer::generate_temp_path(obj));
    }

    std::vector<bool> compiled(num_objs, false);
    for (size_t i = 0; i < num_objs; ++i) {
        if (need_compile(out_dir, target.objs[i])) {
            compiled[i] = true;
        }
    }

    const size_t recompiled = std::count(compiled.begin(), compiled.end(), true);
    bool needs_link = need_link(out_dir, target.target_name);

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
            fs::rename(fs::path(src_path).concat(".dephash"), fs::path(dst_path).concat(".dephash"));
        } else if (fs::exists(src_path)) {
            fs::remove(src_path);
        }
    }

    // Publish the .build_state gate (mirrors JitBuildState::write_build_state_hash). Without this, a
    // cold build_key (e.g. the profiler build_key, whose subtree has no prior .build_state) makes the
    // op-by-op JitBuildState force a full recompile even though .o/.elf/.dephash are already warm.
    if (target.build_state_hash != 0) {
        jit_build::utils::FileRenamer bs_file(out_dir + ".build_state");
        std::ofstream f(bs_file.path());
        f << target.build_state_hash;
    }
}

void write_generated_files(const std::string& kernel_dir, const std::vector<jit_build::GeneratedFile>& files) {
    if (files.empty()) {
        return;
    }
    fs::create_directories(kernel_dir);
    for (const auto& file : files) {
        std::string target_path = kernel_dir + file.name;
        jit_build::utils::FileRenamer tmp(target_path);
        std::ofstream out(tmp.path(), std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot create generated file: " + target_path);
        }
        out.write(
            reinterpret_cast<const char*>(file.content.data()), static_cast<std::streamsize>(file.content.size()));
        if (!out) {
            throw std::runtime_error("Failed to write generated file: " + target_path);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Manifest serialization -- a stream of capnp CompileRequest messages (existing rpc.capnp schema).
// ------------------------------------------------------------------------------------------------

std::mutex g_manifest_mutex;
int g_manifest_fd = -1;

void fill_target_recipe(jit_server::rpc::TargetRecipe::Builder builder, const jit_build::TargetRecipe& t) {
    builder.setTargetName(t.target_name);
    builder.setCflags(t.cflags);
    auto defines = builder.initDefines(t.defines.size());
    for (std::size_t i = 0; i < t.defines.size(); ++i) {
        defines.set(i, t.defines[i]);
    }
    builder.setIncludes(t.includes);
    builder.setCompilerOptLevel(t.compiler_opt_level);
    auto srcs = builder.initSrcs(t.srcs.size());
    for (std::size_t i = 0; i < t.srcs.size(); ++i) {
        srcs.set(i, t.srcs[i]);
    }
    auto objs = builder.initObjs(t.objs.size());
    for (std::size_t i = 0; i < t.objs.size(); ++i) {
        objs.set(i, t.objs[i]);
    }
    builder.setLflags(t.lflags);
    builder.setExtraLinkObjs(t.extra_link_objs);
    builder.setLinkerScript(t.linker_script);
    builder.setWeakenedFirmwareName(t.weakened_firmware_name);
    builder.setFirmwareIsKernelObject(t.firmware_is_kernel_object);
    builder.setLinkerOptLevel(t.linker_opt_level);
    builder.setBuildStateHash(t.build_state_hash);
}

jit_build::TargetRecipe read_target_recipe(jit_server::rpc::TargetRecipe::Reader r) {
    jit_build::TargetRecipe t;
    t.target_name = r.getTargetName().cStr();
    t.cflags = r.getCflags().cStr();
    for (auto d : r.getDefines()) {
        t.defines.push_back(d.cStr());
    }
    t.includes = r.getIncludes().cStr();
    t.compiler_opt_level = r.getCompilerOptLevel().cStr();
    for (auto s : r.getSrcs()) {
        t.srcs.push_back(s.cStr());
    }
    for (auto o : r.getObjs()) {
        t.objs.push_back(o.cStr());
    }
    t.lflags = r.getLflags().cStr();
    t.extra_link_objs = r.getExtraLinkObjs().cStr();
    t.linker_script = r.getLinkerScript().cStr();
    t.weakened_firmware_name = r.getWeakenedFirmwareName().cStr();
    t.firmware_is_kernel_object = r.getFirmwareIsKernelObject();
    t.linker_opt_level = r.getLinkerOptLevel().cStr();
    t.build_state_hash = r.getBuildStateHash();
    return t;
}

std::vector<jit_server::CompileRequest> read_manifest(const std::string& path) {
    std::vector<jit_server::CompileRequest> out;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return out;
    }
    kj::AutoCloseFd afd(fd);
    kj::FdInputStream fd_in(afd.get());
    kj::BufferedInputStreamWrapper buffered(fd_in);
    while (buffered.tryGetReadBuffer().size() > 0) {
        capnp::InputStreamMessageReader message(buffered);
        auto r = message.getRoot<jit_server::rpc::CompileRequest>();
        jit_server::CompileRequest req;
        req.build_key = r.getBuildKey();
        req.kernel_name = r.getKernelName().cStr();
        req.gpp = r.getGpp().cStr();
        for (auto t : r.getTargets()) {
            req.targets.push_back(read_target_recipe(t));
        }
        for (auto f : r.getGeneratedFiles()) {
            jit_build::GeneratedFile gf;
            gf.name = f.getName().cStr();
            auto c = f.getContent();
            gf.content.assign(c.begin(), c.end());
            req.generated_files.push_back(std::move(gf));
        }
        out.push_back(std::move(req));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
// Prewarm orchestration.
// ------------------------------------------------------------------------------------------------

// The recipe stores weakened_firmware_name as a filename only (see export_target_recipe); resolve it
// to the local firmware ELF path. Firmware ELFs are few (one per processor target), so scan once.
std::unordered_map<std::string, std::string> build_firmware_filename_map(const std::string& firmware_root) {
    std::unordered_map<std::string, std::string> m;
    std::error_code ec;
    if (!fs::exists(firmware_root, ec)) {
        return m;
    }
    for (fs::recursive_directory_iterator it(firmware_root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            m[it->path().filename().string()] = it->path().string();
        }
    }
    return m;
}

std::once_flag g_launch_once;
std::thread g_prewarm_thread;

void run_prewarm(const std::string& out_kernel_root, const std::string& firmware_root, std::uint64_t build_key) {
    const char* path = env_or_null("TT_METAL_KERNEL_PREWARM_MANIFEST");
    if (path == nullptr) {
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<jit_server::CompileRequest> requests = read_manifest(path);
    if (requests.empty()) {
        log_warning(tt::LogMetal, "kernel prewarm: manifest {} empty or unreadable; skipping", path);
        return;
    }

    const auto fw_map = build_firmware_filename_map(firmware_root);

    std::vector<std::shared_future<void>> events;
    size_t launched = 0;
    size_t skipped_build_key = 0;
    size_t skipped_dup = 0;
    // A kernel is captured once per program instance, so shared kernels (dispatch, fabric, reused
    // ops) appear many times with the SAME out_dir. FileRenamer temp names are per-process, so two
    // concurrent builds of one out_dir collide on the temp .o and destructively corrupt the final
    // .elf (leaving "no ELF magic" for the op-by-op reader). Build each unique out_dir exactly once.
    std::unordered_set<std::string> built_out_dirs;
    std::unordered_set<std::string> written_kernel_dirs;
    for (const auto& req : requests) {
        if (req.build_key != build_key) {
            ++skipped_build_key;
            continue;
        }
        // Generated files live in the kernel dir (parent of the per-target dirs); target srcs
        // reference them via "../". Write once per kernel dir (stable wrappers, identical content).
        const std::string kernel_dir = out_kernel_root + req.kernel_name + "/";
        if (written_kernel_dirs.insert(kernel_dir).second) {
            try {
                write_generated_files(kernel_dir, req.generated_files);
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "kernel prewarm: genfiles for {} failed: {}", req.kernel_name, e.what());
                continue;
            }
        }
        for (const auto& target : req.targets) {
            const std::string out_dir = kernel_dir + target.target_name + "/";
            if (!built_out_dirs.insert(out_dir).second) {
                ++skipped_dup;
                continue;
            }
            jit_build::TargetRecipe resolved = target;
            if (!resolved.weakened_firmware_name.empty()) {
                auto it = fw_map.find(fs::path(resolved.weakened_firmware_name).filename().string());
                if (it != fw_map.end()) {
                    resolved.weakened_firmware_name = it->second;
                }
            }
            const std::string gpp = req.gpp;
            launch_build_step(
                [gpp, resolved, out_dir]() {
                    try {
                        build_target(gpp, resolved, out_dir);
                    } catch (const std::exception& e) {
                        // A single kernel's failure must not abort the batch. Wipe this target dir:
                        // a failed link can still leave a truncated .elf behind, which the op-by-op
                        // path would treat as a cache HIT and then crash on ("no ELF magic"). Removing
                        // it forces a clean op-by-op rebuild of just this target.
                        std::error_code ec;
                        fs::remove_all(out_dir, ec);
                        log_warning(tt::LogMetal, "kernel prewarm: build {} failed: {}", out_dir, e.what());
                    }
                },
                events);
            ++launched;
        }
    }
    sync_build_steps(events);
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    log_info(
        tt::LogMetal,
        "kernel prewarm: built {} unique targets in {}ms ({} dup targets skipped, {} entries skipped for "
        "build_key mismatch)",
        launched,
        elapsed_ms,
        skipped_dup,
        skipped_build_key);
}

}  // namespace

const char* manifest_write_path() {
    static const char* path = env_or_null("TT_METAL_KERNEL_MANIFEST_WRITE");
    return path;
}

void append_manifest_entry(const jit_server::CompileRequest& request) {
    const char* path = manifest_write_path();
    if (path == nullptr) {
        return;
    }

    capnp::MallocMessageBuilder mb;
    auto root = mb.initRoot<jit_server::rpc::CompileRequest>();
    root.setBuildKey(request.build_key);
    root.setKernelName(request.kernel_name);
    root.setGpp(request.gpp);
    auto targets = root.initTargets(request.targets.size());
    for (std::size_t i = 0; i < request.targets.size(); ++i) {
        fill_target_recipe(targets[i], request.targets[i]);
    }
    if (!request.generated_files.empty()) {
        auto files = root.initGeneratedFiles(request.generated_files.size());
        for (std::size_t i = 0; i < request.generated_files.size(); ++i) {
            files[i].setName(request.generated_files[i].name);
            files[i].setContent(
                kj::arrayPtr(request.generated_files[i].content.data(), request.generated_files[i].content.size()));
        }
    }

    // capnp writeMessageToFd emits several write() calls; hold the lock across the whole message so
    // concurrent pool threads never interleave framed messages in the append stream.
    std::lock_guard<std::mutex> lk(g_manifest_mutex);
    if (g_manifest_fd < 0) {
        g_manifest_fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (g_manifest_fd < 0) {
            log_warning(tt::LogMetal, "kernel prewarm: cannot open manifest {} for write", path);
            return;
        }
    }
    capnp::writeMessageToFd(g_manifest_fd, mb);
}

void maybe_launch_prewarm(
    const std::string& out_kernel_root, const std::string& firmware_root, std::uint64_t build_key) {
    if (env_or_null("TT_METAL_KERNEL_PREWARM_MANIFEST") == nullptr) {
        return;
    }
    std::call_once(g_launch_once, [&]() {
        g_prewarm_thread = std::thread([out_kernel_root, firmware_root, build_key]() {
            try {
                run_prewarm(out_kernel_root, firmware_root, build_key);
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "kernel prewarm aborted: {}", e.what());
            }
        });
    });
}

void wait_for_prewarm() {
    // ProgramImpl::compile may run on more than one thread; serialize the join so only one thread
    // joins the prewarm thread (joining a std::thread from two threads is UB).
    static std::mutex join_mutex;
    std::lock_guard<std::mutex> lk(join_mutex);
    if (g_prewarm_thread.joinable()) {
        g_prewarm_thread.join();
    }
}

}  // namespace tt::tt_metal::kernel_prewarm
