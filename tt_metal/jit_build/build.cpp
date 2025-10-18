// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "build.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <taskflow/core/async.hpp>

#include <tt_stl/assert.hpp>
#include "common/executor.hpp"
#include "env_lib.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/kernel_args.hpp"
#include "jit_build_settings.hpp"
#include "jit_build_utils.hpp"
#include <tt-logger/tt-logger.hpp>
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include "tt_stl/reflection.hpp"
#include <umd/device/types/arch.hpp>
#include <unordered_map>

namespace fs = std::filesystem;

using namespace std;
using namespace tt;

namespace {

void sync_events(auto& events) {
    for (auto& f : events) {
        f.get();
    }
}

}  // namespace

namespace tt::tt_metal {

namespace {

void build_failure(const string& target_name, const string& op, const string& cmd, const string& log_file) {
    log_error(tt::LogBuildKernels, "{} {} failure -- cmd: {}", target_name, op, cmd);
    std::ifstream file{log_file};
    if (file.is_open()) {
        std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        TT_THROW("{} build failed. Log: {}", target_name, log_contents);
    } else {
        TT_THROW("Failed to open {} failure log file {}", op, log_file);
    }
}

void write_successful_jit_build_marker(const JitBuildState& build, const JitBuildSettings* settings) {
    const string out_dir = (settings == nullptr) ? build.get_out_path() + "/"
                                                 : build.get_out_path() + settings->get_full_kernel_name() + "/";
    std::ofstream file(out_dir + SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME);
}

void check_built_dir(const std::filesystem::path& dir_path, const std::filesystem::path& git_hash_path) {
    if (dir_path.compare(git_hash_path) != 0) {
        std::filesystem::remove_all(dir_path);
    }
}

}  // namespace

std::string get_default_root_path() {
    const std::string emptyString("");
    const std::string home_path = parse_env<std::string>("HOME", emptyString);
    if (!home_path.empty() && std::filesystem::exists(home_path)) {
        return home_path + "/.cache/tt-metal-cache/";
    } else {
        return "/tmp/tt-metal-cache/";
    }
}

JitBuildEnv::JitBuildEnv() = default;

void JitBuildEnv::init(
    uint32_t build_key,
    size_t fw_compile_hash,
    tt::ARCH arch,
    const std::map<std::string, std::string>& device_kernel_defines) {
    // Paths
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    this->root_ = rtoptions.get_root_dir();
    this->out_root_ = rtoptions.is_cache_dir_specified() ? rtoptions.get_cache_dir() : get_default_root_path();

    this->arch_ = arch;

#ifndef GIT_COMMIT_HASH
    log_info(tt::LogBuildKernels, "GIT_COMMIT_HASH not found");
#else
    std::string git_hash(GIT_COMMIT_HASH);

    std::filesystem::path git_hash_path(this->out_root_ + git_hash);
    std::filesystem::path root_path(this->out_root_);
    if ((not rtoptions.get_skip_deleting_built_cache()) && std::filesystem::exists(root_path)) {
        std::ranges::for_each(
            std::filesystem::directory_iterator{root_path},
            [&git_hash_path](const auto& dir_entry) { check_built_dir(dir_entry.path(), git_hash_path); });
    } else {
        log_info(tt::LogBuildKernels, "Skipping deleting built cache");
    }

    this->out_root_ = this->out_root_  + git_hash + "/";
#endif
    // Firmware build path is a combination of build_key and fw_compile_hash
    // If either change, the firmware build path will change and FW will be rebuilt
    // if it's not already in MetalContext::firmware_built_keys_
    const std::filesystem::path firmware_path = std::filesystem::path(this->out_root_) / std::to_string(build_key) /
                                                std::to_string(fw_compile_hash) / "firmware/";
    this->out_firmware_root_ = firmware_path.string();
    const std::filesystem::path kernel_path =
        std::filesystem::path(this->out_root_) / std::to_string(build_key) / "kernels/";
    this->out_kernel_root_ = kernel_path.string();

    // Tools
    const static bool use_ccache = std::getenv("TT_METAL_CCACHE_KERNEL_SUPPORT") != nullptr;
    if (use_ccache) {
        this->gpp_ = "ccache ";
    } else {
        this->gpp_ = "";
    }

    // Use local sfpi for development
    // Use system sfpi for production to avoid packaging it
    // Ordered by precedence
    const std::array<std::string, 2> sfpi_roots = {
        this->root_ + "runtime/sfpi",
        "/opt/tenstorrent/sfpi"
    };

    bool sfpi_found = false;
    for (unsigned i = 0; i < 2; ++i) {
        auto gxx = sfpi_roots[i] + "/compiler/bin/riscv-tt-elf-g++";
        if (std::filesystem::exists(gxx)) {
            this->gpp_ += gxx + " ";
            this->gpp_include_dir_ = sfpi_roots[i] + "/include";
            log_debug(tt::LogBuildKernels, "Using {} sfpi at {}", i ? "system" : "local", sfpi_roots[i]);
            sfpi_found = true;
            break;
        }
    }
    if (!sfpi_found) {
        TT_THROW("sfpi not found at {} or {}", sfpi_roots[0], sfpi_roots[1]);
    }

    // Flags
    string common_flags = "-std=c++17 -flto=auto -ffast-math -fno-exceptions ";

    if (rtoptions.get_riscv_debug_info_enabled()) {
        common_flags += "-g ";
    }

    this->cflags_ = common_flags;
    this->cflags_ +=
        "-MMD "
        "-fno-use-cxa-atexit "
        "-Wall -Werror -Wno-unknown-pragmas "
        "-Wno-deprecated-declarations "
        "-Wno-error=multistatement-macros -Wno-error=parentheses "
        "-Wno-error=unused-but-set-variable -Wno-unused-variable "
        "-Wno-unused-function ";

    // Defines
    this->defines_ = "";
    for (auto it = device_kernel_defines.begin(); it != device_kernel_defines.end(); ++it) {
        this->defines_ += "-D" + it->first + "=" + it->second + " ";
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    if (tt::tt_metal::getDeviceProfilerState()) {
        uint32_t profiler_options = 1;
        if (rtoptions.get_profiler_do_dispatch_cores()) {
            profiler_options |= PROFILER_OPT_DO_DISPATCH_CORES;
        }
        if (rtoptions.get_profiler_trace_only()) {
            profiler_options |= PROFILER_OPT_DO_TRACE_ONLY;
        }
        this->defines_ += "-DPROFILE_KERNEL=" + std::to_string(profiler_options) + " ";
    }
    if (rtoptions.get_profiler_noc_events_enabled()) {
        // force profiler on if noc events are being profiled
        if (not tt::tt_metal::getDeviceProfilerState()) {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
        this->defines_ += "-DPROFILE_NOC_EVENTS=1 ";
    }

    if (rtoptions.get_watcher_enabled()) {
        this->defines_ += "-DWATCHER_ENABLED ";
    }
    if (rtoptions.get_watcher_noinline()) {
        this->defines_ += "-DWATCHER_NOINLINE ";
    }
    if (rtoptions.get_watcher_noc_sanitize_linked_transaction()) {
        this->defines_ += "-DWATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION ";
    }
    for (auto& feature : rtoptions.get_watcher_disabled_features()) {
        this->defines_ += "-DWATCHER_DISABLE_" + feature + " ";
    }

    if (rtoptions.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        this->defines_ += "-DDEBUG_PRINT_ENABLED ";
    }

    if (rtoptions.get_record_noc_transfers()) {
        this->defines_ += "-DNOC_LOGGING_ENABLED ";
    }

    if (rtoptions.get_kernels_nullified()) {
        this->defines_ += "-DDEBUG_NULL_KERNELS ";
    }

    if (rtoptions.get_kernels_early_return()) {
        this->defines_ += "-DDEBUG_EARLY_RETURN_KERNELS ";
    }

    if (rtoptions.get_watcher_debug_delay()) {
        this->defines_ += "-DWATCHER_DEBUG_DELAY=" + to_string(rtoptions.get_watcher_debug_delay()) + " ";
    }

    if (rtoptions.get_hw_cache_invalidation_enabled()) {
        this->defines_ += "-DENABLE_HW_CACHE_INVALIDATION ";
    }

    if (rtoptions.get_relaxed_memory_ordering_disabled()) {
        this->defines_ += "-DDISABLE_RELAXED_MEMORY_ORDERING ";
    }

    if (rtoptions.get_gathering_enabled()) {
        this->defines_ += "-DENABLE_GATHERING ";
    }

    if (tt::tt_metal::MetalContext::instance().get_cluster().is_base_routing_fw_enabled()) {
        this->defines_ += "-DROUTING_FW_ENABLED ";
    }

    // Includes
    // TODO(pgk) this list is insane
    std::vector<std::string> includeDirs = {
        ".",
        "..",
        root_,
        root_ + "ttnn",
        root_ + "ttnn/cpp",
        root_ + "tt_metal",
        root_ + "tt_metal/include",
        root_ + "tt_metal/hw/inc",
        root_ + "tt_metal/hostdevcommon/api",
        root_ + "tt_metal/hw/inc/debug",
        root_ + "tt_metal/api/",
        root_ + "tt_metal/api/tt-metalium/"};

    std::ostringstream oss;
    for (size_t i = 0; i < includeDirs.size(); ++i) {
        oss << "-I" << includeDirs[i] << " ";
    }
    this->includes_ = oss.str();

    this->lflags_ = common_flags;
    this->lflags_ += "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
}

JitBuildState::JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    env_(env),
    core_id_(build_config.processor_id),
    is_fw_(build_config.is_fw),
    dispatch_message_addr_(build_config.dispatch_message_addr),
    out_path_(build_config.is_fw ? env_.out_firmware_root_ : env_.out_kernel_root_),
    cflags_(env.cflags_),
    defines_(env.defines_),
    includes_(env.includes_),
    lflags_(env.lflags_),
    default_compile_opt_level_("Os"),
    default_linker_opt_level_("Os"),
    process_defines_at_compile_(true) {
    // Anything that is arch-specific should be added to HalJitBuildQueryInterface instead of here.
    if (build_config.core_type == HalProgrammableCoreType::TENSIX &&
        build_config.processor_class == HalProcessorClassType::COMPUTE) {
        this->default_compile_opt_level_ = "O3";
        this->default_linker_opt_level_ = "O3";
        this->includes_ += "-I" + env_.gpp_include_dir_ + " ";
        this->process_defines_at_compile_ = false;
    } else if (build_config.core_type == HalProgrammableCoreType::ACTIVE_ETH && build_config.is_cooperative) {
        // Only cooperative active ethernet needs "-L <root>/tt_metal/hw/toolchain",
        // because its linker script depends on some files in that directory.
        // Maybe we should move the dependencies to runtime/hw/toolchain/<arch>/?
        fmt::format_to(std::back_inserter(this->lflags_), "-L{}/tt_metal/hw/toolchain/ ", env_.root_);
    }

    HalJitBuildQueryInterface::Params params{
        this->is_fw_, build_config.core_type, build_config.processor_class, this->core_id_};
    const auto& jit_build_query = tt_metal::MetalContext::instance().hal().get_jit_build_query();

    this->target_name_ = jit_build_query.target_name(params);
    // Includes
    {
        auto it = std::back_inserter(this->includes_);
        for (const auto& include : jit_build_query.includes(params)) {
            fmt::format_to(it, "-I{}{} ", env_.root_, include);
        }
    }
    // Defines
    {
        auto it = std::back_inserter(this->defines_);
        for (const auto& define : jit_build_query.defines(params)) {
            fmt::format_to(it, "-D{} ", define);
        }
        fmt::format_to(it, "-DDISPATCH_MESSAGE_ADDR={} ", this->dispatch_message_addr_);
    }
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }
    // Flags
    {
        auto common_flags = jit_build_query.common_flags(params);
        this->cflags_ += common_flags;
        this->lflags_ += common_flags;
    }
    this->lflags_ += fmt::format("-T{}{} ", env_.root_, jit_build_query.linker_script(params));
    // Source files
    {
        auto srcs = jit_build_query.srcs(params);
        this->srcs_.insert(this->srcs_.end(), std::move_iterator(srcs.begin()), std::move_iterator(srcs.end()));
    }

    // Create the objs from the srcs
    for (const string& src : srcs_) {
        // Lop off the right side from the last "."
        string stub = src.substr(0, src.find_last_of("."));
        // Lop off the leading path
        stub = stub.substr(stub.find_last_of("/") + 1, stub.length());
        this->objs_.push_back(stub + ".o");
    }

    // Prepend root path to srcs, but not to outputs (objs) due to device dependency
    for (string& src : this->srcs_) {
        src = env_.root_ + src;
    }

    // Create list of object files for link
    for (const string& obj : this->objs_) {
        this->link_objs_ += obj + " ";
    }

    // Append hw build objects compiled offline
    {
        auto it = std::back_inserter(this->link_objs_);
        for (const auto& obj : jit_build_query.link_objs(params)) {
            fmt::format_to(it, "{}{} ", env_.root_, obj);
        }
    }

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".elf";

    if (not this->is_fw_) {
        // Emit relocations, so we can relocate the resulting binary
        this->lflags_ += "-Wl,--emit-relocs ";
    }
}

namespace {

uint64_t hash_file_contents(std::string_view out_dir, const std::string& file_path) {
    std::filesystem::path path{file_path};
    std::hash<std::string> hasher;
    if (!path.is_absolute()) {
        path = std::filesystem::path{out_dir} / path;
    }
    std::ifstream file{path, std::ios::binary};
    if (!file.is_open()) {
        TT_THROW("file {} not found", path);
    }
    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return hasher(contents);
}

std::unordered_map<std::string, uint64_t> dependency_hash(std::string_view out_dir, std::string_view depend_file) {
    std::ifstream file{std::string(depend_file)};
    if (!file.is_open()) {
        log_info(tt::LogBuildKernels, "depend file {} not found", depend_file);
        return {};
    }
    enum class ParseState { FileName, Dependency };
    ParseState parse_state = ParseState::FileName;
    uint64_t* current_file_hash = nullptr;
    std::unordered_map<std::string, uint64_t> file_to_hash;
    std::string line;
    while (std::getline(file, line)) {
        // std::cerr << "line: " << line << std::endl;
    again:
        switch (parse_state) {
            case ParseState::FileName: {
                auto colon_pos = line.find(':');
                if (colon_pos == std::string::npos) {
                    TT_THROW("couldn't find colon after filename");
                }
                std::string file = line.substr(0, colon_pos);
                uint64_t hash = 0;
                current_file_hash = &file_to_hash[std::move(file)];
                *current_file_hash = hash;
                line = line.substr(colon_pos + 1);
                parse_state = ParseState::Dependency;
                goto again;
            }
            case ParseState::Dependency:
                // Extract the dependency
                for (;;) {
                    line.erase(0, line.find_first_not_of(" \t"));
                    if (line.empty()) {
                        // end of dependencies
                        parse_state = ParseState::FileName;
                        break;
                    }
                    if (line == "\\") {
                        break;
                    }
                    auto end_of_depend_file = line.find_first_of(" \t\\");
                    auto depend_file = line.substr(0, end_of_depend_file);
                    // Hash the dependency file
                    uint64_t depend_file_hash = hash_file_contents(out_dir, depend_file);
                    ttsl::hash::hash_combine(*current_file_hash, depend_file_hash);
                    if (end_of_depend_file == std::string::npos) {
                        // end of dependencies
                        parse_state = ParseState::FileName;
                        break;
                    }
                    line = line.substr(end_of_depend_file);
                }
                break;
        }
    }
    return file_to_hash;
}

bool check_compile_needed(std::string_view out_dir, std::string_view obj) {
    auto base_name = obj.substr(0, obj.find_last_of('.'));
    std::string depend_file = fmt::format("{}{}.d", out_dir, base_name);
    std::string hash_file = fmt::format("{}{}.hash", out_dir, base_name);
    auto dependency_hashes = dependency_hash(out_dir, depend_file);

    std::ifstream file{hash_file};
    uint64_t hash{};
    if (file.is_open()) {
        file >> std::hex >> hash;
        auto iter = dependency_hashes.find(std::string(obj));
        if (iter == dependency_hashes.end()) {
            log_debug(tt::LogBuildKernels, "object {} not found in depend file {}", obj, depend_file);
            return true;
        }
        if (iter->second == hash) {
            log_debug(tt::LogBuildKernels, "object {} is up to date", obj);
            return false;
        } else {
            log_debug(tt::LogBuildKernels, "object {} is out of date", obj);
            return true;
        }
    } else {
        log_debug(tt::LogBuildKernels, "hash file {} not found", hash_file);
        return true;
    }
}

}  // namespace

void JitBuildState::compile_one(
    const string& log_file,
    const string& out_dir,
    const JitBuildSettings* settings,
    const string& src,
    const string& obj) const {
    // ZoneScoped;
    fs::create_directories(out_dir);

    if (!check_compile_needed(out_dir, obj)) {
        return;
    }
    string cmd{"cd " + out_dir + " && " + env_.gpp_};
    string defines = this->defines_;

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_build_map_enabled()) {
        cmd += "-save-temps=obj -fdump-tree-all -fdump-rtl-all ";
    }

    if (settings) {
        // Append user args
        if (process_defines_at_compile_) {
            settings->process_defines([&defines](const string& define, const string& value) {
                defines += fmt::format("-D{}='{}' ", define, value);
            });
        }

        settings->process_compile_time_args([&defines](const std::vector<uint32_t>& values) {
            if (values.empty()) {
                return;
            }
            defines += fmt::format("-DKERNEL_COMPILE_TIME_ARGS={} ", fmt::join(values, ","));
        });

        // This creates a command-line define for named compile time args
        // Ex. for named_args like {"buffer_size": 1024, "num_tiles": 64}
        // This generates:
        // -DKERNEL_COMPILE_TIME_ARG_MAP="{{\"buffer_size\",1024}, {\"num_tiles\",64}} "
        // The macro expansion is defined in tt_metal/hw/inc/compile_time_args.h
        settings->process_named_compile_time_args(
            [&defines](const std::unordered_map<std::string, uint32_t>& named_args) {
                if (named_args.empty()) {
                    return;
                }
                std::ostringstream ss;
                ss << "-DKERNEL_COMPILE_TIME_ARG_MAP=\"";
                for (const auto& [name, value] : named_args) {
                    ss << "{\\\"" << name << "\\\"," << value << "}, ";
                }
                ss << "\"";
                defines += ss.str() + " ";
            });

        cmd += fmt::format("-{} ", settings->get_compiler_opt_level());
    } else {
        cmd += fmt::format("-{} ", this->default_compile_opt_level_);
    }

    // Append common args provided by the build state
    cmd += this->cflags_;
    cmd += this->includes_;
    cmd += fmt::format("-c -o {} {} ", obj, src);
    cmd += defines;

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_log_kernels_compilation_commands()) {
        log_info(tt::LogBuildKernels, "    g++ compile cmd: {}", cmd);
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled() && settings) {
        log_kernel_defines_and_args(out_dir, settings->get_full_kernel_name(), defines);
    }

    if (!tt::jit_build::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "compile", cmd, log_file);
    } else {
        // On successful compilation, write out the dependency hash file
        auto base_name = obj.substr(0, obj.find_last_of('.'));
        std::string depend_file = fmt::format("{}{}.d", out_dir, base_name);
        std::string hash_file = fmt::format("{}{}.hash", out_dir, base_name);
        auto dependency_hashes = dependency_hash(out_dir, depend_file);
        auto iter = dependency_hashes.find(obj);
        if (iter == dependency_hashes.end()) {
            log_critical(tt::LogBuildKernels, "object {} not found in depend file {}", obj, depend_file);
        } else {
            std::ofstream file{hash_file};
            file << std::hex << iter->second;
        }
    }
}

void JitBuildState::compile(const string& log_file, const string& out_dir, const JitBuildSettings* settings) const {
    // ZoneScoped;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        launch_build_step(
            [this, &log_file, &out_dir, settings, i] {
                this->compile_one(log_file, out_dir, settings, this->srcs_[i], this->objs_[i]);
            },
            events);
    }

    sync_events(events);

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
}

void JitBuildState::link(const string& log_file, const string& out_dir, const JitBuildSettings* settings) const {
    // ZoneScoped;
    string cmd{"cd " + out_dir + " && " + env_.gpp_};
    string lflags = this->lflags_;
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_build_map_enabled()) {
        lflags += "-Wl,-Map=" + out_dir + this->target_name_ + ".map ";
        lflags += "-save-temps=obj -fdump-tree-all -fdump-rtl-all ";
    }

    // Append user args
    cmd += fmt::format("-{} ", settings ? settings->get_linker_opt_level() : this->default_linker_opt_level_);

    if (!this->is_fw_) {
        string weakened_elf_name =
            env_.out_firmware_root_ + this->target_name_ + "/" + this->target_name_ + "_weakened.elf";
        cmd += "-Wl,--just-symbols=" + weakened_elf_name + " ";
    }

    // Append common args provided by the build state
    cmd += lflags;
    cmd += this->link_objs_;
    cmd += "-o " + out_dir + this->target_name_ + ".elf";
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_log_kernels_compilation_commands()) {
        log_info(tt::LogBuildKernels, "    g++ link cmd: {}", cmd);
    }
    if (!tt::jit_build::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "link", cmd, log_file);
    }
}

// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B. B imports A's weakened symbols, B's symbols of the
// same name don't result in duplicate symbols but B can reference A's symbols. Force the fw_export symbols to remain
// strong so to propogate link addresses
void JitBuildState::weaken(const string& /*log_file*/, const string& out_dir) const {
    // ZoneScoped;

    std::string pathname_in = out_dir + target_name_ + ".elf";
    std::string pathname_out = out_dir + target_name_ + "_weakened.elf";

    ll_api::ElfFile elf;
    elf.ReadImage(pathname_in);
    static std::string_view const strong_names[] = {"__fw_export_*", "__global_pointer$"};
    elf.WeakenDataSymbols(strong_names);
    elf.WriteImage(pathname_out);
}

void JitBuildState::extract_zone_src_locations(const string& log_file) const {
    // ZoneScoped;
    static std::atomic<bool> new_log = true;
    if (tt::tt_metal::getDeviceProfilerState()) {
        if (new_log.exchange(false) && std::filesystem::exists(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            std::remove(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG.c_str());
        }

        if (!std::filesystem::exists(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            tt::jit_build::utils::create_file(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG);
        }

        // Only interested in log entries with KERNEL_PROFILER inside them as device code
        // tags source location info with it using pragma messages
        string cmd = "cat " + log_file + " | grep KERNEL_PROFILER";
        tt::jit_build::utils::run_command(cmd, tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG, false);
    }
}

void JitBuildState::build(const JitBuildSettings* settings) const {
    // ZoneScoped;
    string out_dir = (settings == nullptr)
                         ? this->out_path_ + this->target_name_ + "/"
                         : this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    string log_file = out_dir + "build.log";
    if (fs::exists(log_file)) {
        std::remove(log_file.c_str());
    }
    compile(log_file, out_dir, settings);
    link(log_file, out_dir, settings);
    if (this->is_fw_) {
        weaken(log_file, out_dir);
    }

    extract_zone_src_locations(log_file);
}

void jit_build(const JitBuildState& build, const JitBuildSettings* settings) {
    // ZoneScoped;

    build.build(settings);
    write_successful_jit_build_marker(build, settings);
}

void jit_build_subset(JitBuildStateSubset build_subset, const JitBuildSettings* settings) {
    std::vector<std::shared_future<void>> events;
    for (auto& build : build_subset) {
        // Capture the necessary objects by reference
        launch_build_step([&build, settings] { build.build(settings); }, events);
    }

    sync_events(events);
    for (auto& build : build_subset) {
        write_successful_jit_build_marker(build, settings);
    }
}

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events) {
    events.emplace_back(detail::async(build_func));
}

void sync_build_steps(std::vector<std::shared_future<void>>& events) {
    for (auto& event : events) {
        event.wait();
    }
}

}  // namespace tt::tt_metal
