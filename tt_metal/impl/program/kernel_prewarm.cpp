// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/program/kernel_prewarm.hpp"

#include <fcntl.h>
#include <sys/file.h>
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
#include <string_view>
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

// Prewarm is on by default so any checkout is fast to iterate with no manual setup.
// TT_METAL_KERNEL_PREWARM=0 (or false/off) restores the pre-prewarm behavior byte-for-byte.
bool prewarm_globally_disabled() {
    const char* v = std::getenv("TT_METAL_KERNEL_PREWARM");
    if (v == nullptr) {
        return false;
    }
    std::string_view s{v};
    return s == "0" || s == "false" || s == "off";
}

// Runtime-settable so one process can flip capture-only ON for a manifest-capture warmup pass and OFF
// for the batch compile + real run (the in-process cold-start). Initial value is the env flag.
std::atomic<bool> g_capture_only{env_or_null("TT_METAL_KERNEL_CAPTURE_ONLY") != nullptr};

// out_kernel_root is "<cache_root>/<build_key>/kernels/"; the manifest lives at the cache root so a
// single file serves every build_key (arch + compile config) on this machine -- entries are
// build_key-tagged and filtered on read. The cache root already moves with TT_METAL_CACHE, so this is
// correct on any box without configuration.
std::string default_manifest_path(const std::string& out_kernel_root) {
    fs::path p(out_kernel_root);
    if (p.filename().empty()) {
        p = p.parent_path();
    }
    return (p.parent_path().parent_path() / "kernel_prewarm.manifest").string();
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

// Resolved once at device init (maybe_launch_prewarm): the manifest path this process appends to, or
// empty when capture is disabled.
std::string g_capture_path;

// Guards g_capture_path/g_captured_keys writes. Kept distinct from g_manifest_mutex (which serializes
// the append fd) so a capture dedup lookup never blocks on an in-flight file write.
std::mutex g_capture_mutex;

// (build_key + "/" + kernel_name) already recorded, so capture appends each unique kernel build once.
// Seeded from the on-disk manifest at init so warm runs re-append only genuinely new kernels.
std::unordered_set<std::string> g_captured_keys;

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
        try {
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
        } catch (const std::exception& e) {
            // A partial trailing frame (a concurrent writer mid-append, or an interrupted run) must
            // degrade to "prewarm what parsed cleanly", never abort. The dropped kernels just compile
            // op-by-op and get re-captured.
            log_warning(tt::LogMetal, "kernel prewarm: manifest truncated after {} entries: {}", out.size(), e.what());
            break;
        }
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

// True once a prewarm batch has been spawned for this process's build_key. Gates the op-by-op
// barrier: on a cold cache (no batch) the compile path skips the barrier entirely.
std::atomic<bool> g_batch_launched{false};

// Base kernel names (Kernel::name(), i.e. the out_dir prefix before the per-kernel hash) that
// the launched prewarm batch is (re)building. Written exactly once under g_launch_once, before
// g_prewarm_thread is created, then only read; g_prewarm_names_ready publishes it with
// release/acquire ordering so any thread that can observe an in-flight batch also observes this
// set. Empty/false when prewarm is disabled.
std::unordered_set<std::string> g_prewarm_kernel_names;
std::atomic<bool> g_prewarm_names_ready{false};

// Captured recipes store kernel_name as "<base>/<hash>" (see build_kernel_descriptor), whereas the
// op-by-op gate compares against Kernel::name(), which is just "<base>". Strip the hash so the two
// use the same key.
std::string base_kernel_name(const std::string& kernel_name) {
    const auto pos = kernel_name.find('/');
    return pos == std::string::npos ? kernel_name : kernel_name.substr(0, pos);
}

// Command-queue dispatch (cq_*) and fabric (*fabric*) kernels are compiled during device
// initialization -- before the host reaches any idle (weight-load / warmup) window -- and form a
// set DISJOINT from model kernels: their kernel names are distinct, so their cache out_dirs
// (<root>/<name>/<hash>/<target>/) can never coincide with a model kernel's. Excluding them from
// the prewarm batch lets device-init compile them concurrently with the batch (no shared out_dir,
// so no FileRenamer temp collision), instead of forcing the first ProgramImpl::compile (a
// device-init dispatch/fabric program) to block on the entire batch. Misclassifying only costs
// overlap, never correctness: a kept device-init kernel would still gate its program via
// prewarm_warms_kernel(); an excluded model kernel would just compile cold.
bool is_device_init_kernel(const std::string& base_name) {
    return base_name.rfind("cq_", 0) == 0 || base_name.find("fabric") != std::string::npos;
}

std::size_t run_prewarm(
    const std::vector<jit_server::CompileRequest>& requests,
    const std::string& out_kernel_root,
    const std::string& firmware_root,
    std::uint64_t build_key,
    const std::string& root_dir,
    bool skip_device_init) {
    if (requests.empty()) {
        log_warning(tt::LogMetal, "kernel prewarm: manifest empty or unreadable; skipping");
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();

    const auto fw_map = build_firmware_filename_map(firmware_root);

    std::vector<std::shared_future<void>> events;
    size_t launched = 0;
    size_t skipped_build_key = 0;
    size_t skipped_dup = 0;
    size_t skipped_device_init = 0;
    size_t skipped_foreign_tree = 0;
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
        // build_key covers arch + compile config but NOT the source tree, so sibling trees with the
        // same config collide on one build_key and, under a shared cache root, on this one manifest.
        // A recipe's gpp and -I flags are absolute paths into the tree that captured it: replaying a
        // sibling's recipe compiles that tree's headers, which mismatch this tree's binaries. gpp
        // lives under root_dir (<root>/runtime/sfpi/...), optionally behind a ccache prefix, so it
        // identifies the owning tree. Skip foreign recipes: a shared root then costs recompiles
        // rather than miscompiles. An empty root_dir means the caller cannot identify the tree, so
        // the filter is inert.
        if (!root_dir.empty() && req.gpp.find(root_dir) == std::string::npos) {
            ++skipped_foreign_tree;
            continue;
        }
        // Device-init dispatch/fabric kernels are compiled concurrently by device init (disjoint
        // out_dirs); skip them here so the batch overlaps device init instead of racing it. The
        // offline pass (no device to overlap) clears skip_device_init so the reservation later does no
        // compilation at all.
        if (skip_device_init && is_device_init_kernel(base_kernel_name(req.kernel_name))) {
            ++skipped_device_init;
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
        "kernel prewarm: built {} unique targets in {}ms ({} dup targets skipped, {} device-init entries "
        "skipped, {} entries skipped for build_key mismatch, {} entries skipped as foreign-tree)",
        launched,
        elapsed_ms,
        skipped_dup,
        skipped_device_init,
        skipped_build_key,
        skipped_foreign_tree);
    if (skipped_foreign_tree > 0) {
        log_warning(
            tt::LogMetal,
            "kernel prewarm: {} manifest entries for this build_key were captured by a different source "
            "tree and were skipped; this cache root is shared with another tree ({}). Give each tree its "
            "own TT_METAL_CACHE to keep the prewarm effective.",
            skipped_foreign_tree,
            root_dir);
    }
    return launched;
}

}  // namespace

const char* manifest_write_path() {
    // Resolved at device init (maybe_launch_prewarm): explicit override or the cache-root default.
    return g_capture_path.empty() ? nullptr : g_capture_path.c_str();
}

bool capture_needed(std::uint64_t build_key, const std::string& kernel_name) {
    if (g_capture_path.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lk(g_capture_mutex);
    return g_captured_keys.insert(std::to_string(build_key) + "/" + kernel_name).second;
}

bool capture_only() { return g_capture_only.load(std::memory_order_acquire); }

void set_capture_only(bool enabled) { g_capture_only.store(enabled, std::memory_order_release); }

bool capture_only_skip_gcc(const std::string& kernel_base_name) {
    // Device-init (cq_/fabric) kernels are compiled + loaded during CreateDevice/fabric bring-up via
    // the slow-dispatch configure() path, before any pipeline op; they MUST have real binaries or the
    // device never comes up. Only model kernels (compiled later, op-by-op) are skipped and left to the
    // off-device prewarm.
    return capture_only() && !is_device_init_kernel(kernel_base_name);
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
    // A shared cache dir can be written by concurrent runs (e.g. CI). flock serializes the multi-write
    // capnp frame across processes; O_APPEND keeps each write at EOF so frames stay contiguous.
    ::flock(g_manifest_fd, LOCK_EX);
    capnp::writeMessageToFd(g_manifest_fd, mb);
    ::flock(g_manifest_fd, LOCK_UN);
}

void maybe_launch_prewarm(
    const std::string& out_kernel_root,
    const std::string& firmware_root,
    std::uint64_t build_key,
    const std::string& root_dir) {
    if (prewarm_globally_disabled()) {
        return;
    }
    std::call_once(g_launch_once, [&]() {
        // Enable capture unconditionally (explicit override else cache-root default). On a cold cache
        // this run's op-by-op compiles populate the manifest; that reuse is what makes the *next* run
        // fast with no manual setup.
        const char* write_override = env_or_null("TT_METAL_KERNEL_MANIFEST_WRITE");
        g_capture_path =
            write_override != nullptr ? std::string(write_override) : default_manifest_path(out_kernel_root);

        const char* read_override = env_or_null("TT_METAL_KERNEL_PREWARM_MANIFEST");
        const std::string read_path =
            read_override != nullptr ? std::string(read_override) : default_manifest_path(out_kernel_root);

        std::error_code ec;
        if (!fs::exists(read_path, ec) || fs::file_size(read_path, ec) == 0) {
            // Cold cache: nothing to prewarm yet. Capture above records this run so the next is warm.
            return;
        }

        // Read + parse the manifest on this (device-init) thread so the batch's kernel-name set is
        // PUBLISHED before g_prewarm_thread is created. That guarantees any later ProgramImpl::compile
        // that can observe the in-flight batch also observes the set (via prewarm_warms_kernel), so the
        // precise barrier is race-free without a condition variable.
        std::vector<jit_server::CompileRequest> requests = read_manifest(read_path);
        bool have_this_build_key = false;
        {
            // Seed the capture dedup set with everything already on disk so warm runs append only
            // genuinely new kernels, keeping the manifest at the working-set size.
            std::lock_guard<std::mutex> lk(g_capture_mutex);
            for (const auto& req : requests) {
                g_captured_keys.insert(std::to_string(req.build_key) + "/" + req.kernel_name);
                if (req.build_key != build_key) {
                    continue;
                }
                have_this_build_key = true;
                std::string base = base_kernel_name(req.kernel_name);
                if (is_device_init_kernel(base)) {
                    continue;
                }
                g_prewarm_kernel_names.insert(std::move(base));
            }
        }
        g_prewarm_names_ready.store(true, std::memory_order_release);

        if (!have_this_build_key) {
            // Manifest exists but has no entries for this build_key (e.g. first profiler run against a
            // normal-only manifest). Nothing to build; capture will record this build_key's kernels.
            return;
        }

        g_batch_launched.store(true, std::memory_order_release);
        g_prewarm_thread =
            std::thread([reqs = std::move(requests), out_kernel_root, firmware_root, build_key, root_dir]() mutable {
                try {
                    run_prewarm(reqs, out_kernel_root, firmware_root, build_key, root_dir, /*skip_device_init=*/true);
                } catch (const std::exception& e) {
                    log_warning(tt::LogMetal, "kernel prewarm aborted: {}", e.what());
                }
            });
    });
}

std::size_t prewarm_manifest_offline(const std::string& out_root, const std::string& root_dir) {
    // JitBuildEnv concatenates: out_kernel_root is "<out_root><build_key>/kernels/", and the manifest
    // sits two levels above that (default_manifest_path). A single parent_path() of out_root lands
    // there for either spelling, because the spelling decides the layout: "<X>/tt-metal-cache/" makes
    // the build_key a subdir ("<X>/tt-metal-cache/<bk>/") whose manifest dir is "<X>/tt-metal-cache",
    // while "<X>/tt-metal-cache" makes it a suffix ("<X>/tt-metal-cache<bk>/") whose manifest dir is
    // "<X>". Do not "normalize" the trailing slash away: that breaks the former.
    const std::string manifest_path = (fs::path(out_root).parent_path() / "kernel_prewarm.manifest").string();
    std::vector<jit_server::CompileRequest> requests = read_manifest(manifest_path);
    if (requests.empty()) {
        log_warning(tt::LogMetal, "kernel prewarm (offline): no entries in {} (cold cache or empty)", manifest_path);
        return 0;
    }

    std::unordered_set<std::uint64_t> build_keys;
    for (const auto& req : requests) {
        build_keys.insert(req.build_key);
    }

    std::size_t total = 0;
    for (std::uint64_t bk : build_keys) {
        // String concat (not fs::path /) to match JitBuildEnv's fmt "{}{}/kernels/", which places the
        // build_key directly after out_root -- e.g. ".../tt-metal-cache" + "1234" = ".../tt-metal-cache1234".
        const std::string out_kernel_root = out_root + std::to_string(bk) + "/kernels/";
        // Firmware: mirror BuildEnvManager::add_build_env_locked -- prefer the pre-compiled bundle at
        // <root_dir>tt_metal/pre-compiled/<bk>/ (the runtime default) that the link's --just-symbols
        // needs, else the jit firmware subtree. Both hold byte-identical firmware for a build_key, so
        // the choice only affects whether a later op-by-op relink is needed, never the kernel binary.
        std::string firmware_root = out_root + std::to_string(bk) + "/firmware/";
        if (!root_dir.empty()) {
            const std::string precompiled = root_dir + "tt_metal/pre-compiled/" + std::to_string(bk) + "/";
            std::error_code ec;
            if (fs::is_directory(precompiled, ec)) {
                firmware_root = precompiled;
            }
        }
        total += run_prewarm(requests, out_kernel_root, firmware_root, bk, root_dir, /*skip_device_init=*/false);
    }
    log_info(
        tt::LogMetal,
        "kernel prewarm (offline): built {} targets across {} build_key(s) from {}",
        total,
        build_keys.size(),
        manifest_path);
    return total;
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

bool prewarm_enabled() { return g_batch_launched.load(std::memory_order_acquire); }

bool cold_start_needed() {
    // manifest_write_path() != nullptr means capture is armed (maybe_launch_prewarm ran and prewarm is
    // not globally disabled). !prewarm_enabled() means no batch was launched for this build_key -- cold
    // cache or a build_key the manifest does not cover -- so the real run would compile op-by-op.
    // !capture_only() excludes an externally forced capture pass (which captures the whole process).
    return manifest_write_path() != nullptr && !prewarm_enabled() && !capture_only();
}

bool prewarm_warms_kernel(const std::string& kernel_name) {
    if (!g_prewarm_names_ready.load(std::memory_order_acquire)) {
        return false;
    }
    return g_prewarm_kernel_names.find(kernel_name) != g_prewarm_kernel_names.end();
}

}  // namespace tt::tt_metal::kernel_prewarm
