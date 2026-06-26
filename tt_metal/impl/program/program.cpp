// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <circular_buffer.hpp>
#include <circular_buffer_config.hpp>
#include <device.hpp>
#include <graph_tracking.hpp>
#include <enchantum/enchantum.hpp>
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "impl/buffers/semaphore.hpp"
#include <ranges>
#include <tt_align.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "hal_types.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include "tt-metalium/mesh_device.hpp"
#include <unistd.h>
#include "jit_build/build.hpp"
#include "jit_build/depend.hpp"
#include "profiler_paths.hpp"
#include <tt_stl/enum.hpp>
#include "jit_build/jit_build_options.hpp"
#include "kernel_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include <tt-logger/tt-logger.hpp>
#include "program_command_sequence.hpp"
#include "program_device_map.hpp"
#include "program_impl.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/overloaded.hpp>
#include "sub_device_types.hpp"
#include "tt_memory.h"
#include "tt_metal/impl/debug/inspector/inspector.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_common.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include "tt_metal/jit_build/jit_build_utils.hpp"
#include "impl/jit_server/remote_compile_coordinator.hpp"
#include "kernel_compile_utils.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "host_api.hpp"
#include "tt_metal.hpp"  // WriteRuntimeArgsToDevice
#include "kernels/kernel.hpp"
#include <tt_stl/reflection.hpp>
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/allocator/allocator.hpp"
#include <internal/service/service_core_manager.hpp>
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt {
enum CBIndex : std::uint8_t;
namespace tt_metal::experimental {
class GlobalCircularBuffer;
}  // namespace tt_metal::experimental
}  // namespace tt

namespace {

using namespace tt::tt_metal;

size_t get_ringbuffer_size(IDevice* device, HalProgrammableCoreType programmable_core_type) {
    if (programmable_core_type == HalProgrammableCoreType::TENSIX) {
        return device->allocator_impl()->get_config().l1_unreserved_base -
               MetalContext::instance().hal().get_dev_addr(
                   HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
    }
    return MetalContext::instance().hal().get_dev_size(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);
}

void validate_kernel_placement(bool force_slow_dispatch, std::shared_ptr<Kernel> kernel, tt::ChipId device_id) {
    // Placement rules:
    //  Fast dispatch (tensix):
    //      - tensix kernels cannot be on dispatch cores
    //  Fast dispatch (ethernet):
    //      - eth kernels cannot be on idle eth cores
    bool slow_dispatch = !(MetalContext::instance().rtoptions().get_fast_dispatch());

    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    tt::CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);

    // Kernels used to implement fast dispatch can be placed on dispatch cores
    if (not slow_dispatch and not force_slow_dispatch) {
        const std::vector<CoreCoord>& dispatch_cores =
            MetalContext::instance().get_dispatch_query_manager().get_logical_dispatch_cores_on_user_chips();
        const auto& service_claims = MetalContext::instance().get_service_core_manager().impl();
        bool on_dispatch_core = std::any_of(
            dispatch_cores.begin(),
            dispatch_cores.end(),
            [&kernel, &dispatch_core_type, &service_claims, device_id](const CoreCoord& dispatch_core) {
                if (kernel->get_kernel_core_type() != dispatch_core_type) {
                    return false;
                }
                // Claimed service cores are permitted to run user kernels in FD mode.
                if (service_claims.is_service_core(device_id, dispatch_core)) {
                    return false;
                }
                return kernel->is_on_logical_core(dispatch_core);
            });

        TT_FATAL(
            not on_dispatch_core,
            "Illegal kernel placement for {}, Kernels cannot be placed on dispatch cores!",
            kernel->name());
    }
};

}  // namespace

namespace tt::tt_metal {

using detail::ProgramImpl;

namespace {

// Similar to Kernel::generate_binaries(), but does not run the compiler.  Used by remote compilation.
void generate_kernel_source_files(
    IDevice* device, const JitBuildOptions& build_options, const std::shared_ptr<Kernel>& kernel) {
    const auto& env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id()).build_env;
    jit_build_genfiles_descriptors(env, build_options);
    if (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
        jit_build_genfiles_triscs_src(env, *kernel, kernel->kernel_source());
    } else {
        jit_build_genfiles_kernel_include(env, *kernel, kernel->kernel_source());
    }
}

// Returns true if every expected ELF for this kernel is already present locally with unchanged
// source+header dependencies. The client then skips the remote round-trip (preprocess + RPC + ELF
// transfer) entirely and read_binaries() loads the cached ELF. The validating ".dephash" sidecar was
// written during a prior compile's preprocess step, from the real-source .d the compiler emits. A
// source change moves the kernel-hash path and a header change invalidates the dephash, so both
// still force a recompile.
bool remote_kernel_cached(IDevice* device, const std::shared_ptr<Kernel>& kernel) {
    uint32_t core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(kernel->get_kernel_programmable_core_type());
    uint32_t proc_class = enchantum::to_underlying(kernel->get_kernel_processor_class());
    int num_binaries = kernel->expected_num_binaries();
    if (num_binaries <= 0) {
        return false;
    }
    for (int i = 0; i < num_binaries; ++i) {
        const JitBuildState& bs = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), core_type, proc_class, kernel->get_kernel_processor_type(i));
        const std::string elf_path = bs.get_target_out_path(kernel->get_full_kernel_name());
        std::filesystem::path p(elf_path);
        const std::string out_dir = p.parent_path().string() + "/";
        const std::string elf_name = p.filename().string();
        if (!std::filesystem::exists(elf_path) || !tt::jit_build::dependencies_up_to_date(out_dir, elf_name)) {
            return false;
        }
    }
    return true;
}

// Harvest profiler zone-source locations from a preprocessed (.ii) translation unit and append them
// to the profiler's zone-source-location log.
//
// The device profiler stores only a 16-bit hash per zone; the host rebuilds the
// hash -> (zone_name, file, line) table by grepping the compiler's `#pragma message(...,KERNEL_PROFILER)`
// notes out of *.o.log (see jit_build/build.cpp). That harvest runs only on the LOCAL compile path, so
// kernels compiled on the JIT server never contribute their zones and `DEVICE KERNEL DURATION [ns]`
// reads 0 for every test.
//
// In preprocess-and-ship the `_Pragma(message(...))` survives `-E` as a literal
// `#pragma message("zone" "," "file" "," "line" ",KERNEL_PROFILER")` directive in the shipped .ii.
// Reconstruct the same string the device hashed (mirroring C++ string-literal concatenation) and append
// it, in the exact shape profiler.cpp::populateZoneSrcLocations parses, to the zone-source log. The host
// dedupes by zone string, so re-appending an already-seen zone is harmless. Best-effort: profiler
// bookkeeping must never block kernel compilation.
void harvest_zone_src_locations_from_ii(const std::vector<std::uint8_t>& ii_content) {
    const auto concat_pragma_literals = [](const std::string& line, std::size_t open_paren) {
        std::string zone;
        bool in_str = false;
        for (std::size_t i = open_paren + 1; i < line.size(); ++i) {
            const char c = line[i];
            if (in_str) {
                if (c == '\\' && i + 1 < line.size()) {
                    zone.push_back(line[++i]);  // keep the escaped character verbatim
                } else if (c == '"') {
                    in_str = false;
                } else {
                    zone.push_back(c);
                }
            } else if (c == '"') {
                in_str = true;
            } else if (c == ')') {
                break;  // end of message(...) argument list
            }
        }
        return zone;
    };

    std::ofstream log_file;
    const char* data = reinterpret_cast<const char*>(ii_content.data());
    const std::size_t n = ii_content.size();
    std::size_t pos = 0;
    while (pos < n) {
        std::size_t eol = pos;
        while (eol < n && data[eol] != '\n') {
            ++eol;
        }
        const std::string line(data + pos, eol - pos);
        pos = eol + 1;

        if (line.find("#pragma message(") == std::string::npos || line.find("KERNEL_PROFILER") == std::string::npos) {
            continue;
        }
        const std::size_t open_paren = line.find('(');
        if (open_paren == std::string::npos) {
            continue;
        }
        const std::string zone = concat_pragma_literals(line, open_paren);
        if (zone.empty()) {
            continue;
        }
        if (!log_file.is_open()) {
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path(NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG).parent_path(), ec);
            log_file.open(NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG, std::ios::app);
            if (!log_file.is_open()) {
                return;  // best-effort: never block compilation on profiler bookkeeping
            }
        }
        // populateZoneSrcLocations() locates the delimiter "'#pragma message: " and strips the final
        // character of the line, so wrap the zone string in a trailing sentinel quote to match.
        log_file << "'#pragma message: " << zone << "'\n";
    }
}

// Build a KernelCompileDescriptor to be submitted to RemoteCompileCoordinator.
KernelCompileDescriptor build_kernel_descriptor(
    IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const JitBuildOptions& build_options,
    std::size_t kernel_hash) {
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id());

    uint32_t core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(kernel->get_kernel_programmable_core_type());
    uint32_t proc_class = enchantum::to_underlying(kernel->get_kernel_processor_class());

    KernelCompileDescriptor desc;
    desc.kernel_hash = kernel_hash;
    desc.request.build_key = build_env.build_key();
    desc.request.kernel_name = kernel->name() + "/" + std::to_string(kernel_hash);
    desc.request.gpp = build_env.build_env.get_gpp();
    // Ship our build root so the server can re-root the sfpi toolchain / linker script / hw link
    // objects to its own tree — lets a client whose tree is at a different path compile remotely
    // without the server needing the client's filesystem layout.
    desc.request.client_root = build_env.build_env.get_root_path();
    static const std::vector<std::string> extensions = {".h", ".hpp", ".cpp"};
    desc.request.generated_files = tt::jit_build::utils::read_directory_files(build_options.path, extensions);

    int num_binaries = kernel->expected_num_binaries();
    for (int i = 0; i < num_binaries; ++i) {
        const JitBuildState& bs =
            BuildEnvManager::get_instance(extract_context_id(device))
                .get_kernel_build_state(
                    device->build_id(), core_type, proc_class, kernel->get_kernel_processor_type(i));
        desc.request.targets.push_back(bs.export_target_recipe(kernel.get()));
        desc.expected_elf_paths.push_back(bs.get_target_out_path(kernel->get_full_kernel_name()));
    }

    // Preprocess-and-ship (TT_METAL_JIT_PREPROCESS=1): run each source through the
    // preprocessor (-E) on the client and ship the self-contained .ii as content. The
    // server then compiles the .ii with no include tree, no defines, and no source file
    // on its filesystem -- only the toolchain. Reuses the generated_files content channel
    // (written into the per-kernel cache dir on the server), so no RPC/server change is
    // required: the .ii is referenced as a sibling of the target output dir ("../<name>").
    static const bool preprocess_and_ship = std::getenv("TT_METAL_JIT_PREPROCESS") != nullptr;
    if (preprocess_and_ship) {
        // When the device profiler is on, the remotely-compiled kernels' zone-source locations are
        // never harvested locally (the compiler ran on the server), so recover them from each .ii below.
        const bool profiler_enabled = MetalContext::instance().rtoptions().get_profiler_enabled();
        for (std::size_t t = 0; t < desc.request.targets.size(); ++t) {
            auto& target = desc.request.targets[t];
            const std::string client_out_dir = std::filesystem::path(desc.expected_elf_paths[t]).parent_path().string();
            std::filesystem::create_directories(client_out_dir);
            for (std::size_t i = 0; i < target.srcs.size(); ++i) {
                const std::string ii_name =
                    target.target_name + "__" + std::filesystem::path(target.objs[i]).filename().string() + ".ii";
                const std::string ii_path = client_out_dir + "/" + ii_name;
                // Preprocess with the EXACT compile flags via the shared argv builder + exec_command
                // (posix_spawn, NO shell). A shell command string would mangle map-valued defines
                // like -DKERNEL_COMPILE_TIME_ARG_MAP={"cb_in0",1},... (braces/quotes/commas/spaces)
                // and drop named compile-time args. cwd = client_out_dir so -I. / -I.. resolve to
                // the target + generated-files dirs, identical to the real compile env.
                const auto args = tt::jit_build::utils::build_gpp_argv(
                    desc.request.gpp,
                    target.compiler_opt_level,
                    target.cflags,
                    target.includes,
                    target.defines,
                    target.srcs[i],
                    tt::jit_build::utils::GppAction::Preprocess,
                    ii_path);
                if (!tt::jit_build::utils::exec_command(args, client_out_dir, ii_path + ".log")) {
                    TT_THROW("preprocess-and-ship: -E failed for {} (log: {})", target.srcs[i], ii_path + ".log");
                }
                const auto bytes = tt::jit_build::utils::read_file_bytes(ii_path);
                if (profiler_enabled) {
                    harvest_zone_src_locations_from_ii(bytes);
                }
                tt::jit_build::GeneratedFile gf;
                gf.name = ii_name;
                gf.content.assign(bytes.begin(), bytes.end());
                desc.request.generated_files.push_back(std::move(gf));
                // Server compiles this self-contained unit instead of the original source path.
                target.srcs[i] = "../" + ii_name;
                // For a single-source target, record a .dephash next to the expected ELF from the
                // real-source .d the preprocessor just emitted (-MMD lists the kernel source + every
                // header). A later run validates it in remote_kernel_cached() and skips the round-trip.
                // The .d's filename derives from the .ii while its internal target key is the bare obj,
                // so parse it and write via the explicit-deps overload using the .d's own key.
                if (target.srcs.size() == 1) {
                    const std::filesystem::path d_path =
                        std::filesystem::path(client_out_dir) / std::filesystem::path(ii_name).replace_extension(".d");
                    std::ifstream d_file(d_path);
                    if (d_file.is_open()) {
                        tt::jit_build::ParsedDependencies deps = tt::jit_build::parse_dependency_file(d_file);
                        if (!deps.empty()) {
                            const std::string& dep_key = deps.begin()->first;
                            const std::string dephash_path = desc.expected_elf_paths[t] + ".dephash";
                            std::ofstream hash_file(dephash_path);
                            tt::jit_build::write_dependency_hashes(deps, client_out_dir + "/", dep_key, hash_file);
                            hash_file.close();
                            if (hash_file.fail()) {
                                std::filesystem::remove(dephash_path);
                            }
                        }
                    }
                }
            }
            // The .ii has includes + defines baked in; the server must not need the tree.
            target.includes.clear();
            target.defines.clear();
            // -P drops the system-header markers that normally suppress warnings inside
            // libstdc++; without them -Werror trips on standard-library internals. Downgrade
            // warnings-to-errors for the preprocessed compile (codegen is unaffected, and the
            // original source compiled clean, so this only tolerates now-unmasked system warnings).
            target.cflags += " -Wno-error";
        }
    }

    return desc;
}

std::string ensure_kernel_binaries(
    const std::shared_ptr<Kernel>& kernel,
    IDevice* device,
    JitBuildOptions& build_options,
    const DeviceBuildEnv& build_env,
    size_t kernel_hash) {
    if (const auto& precompiled_config = kernel->precompiled_config(); precompiled_config.has_value()) {
        if (kernel->binaries_exist_on_disk(device, precompiled_config->precompiled_dir)) {
            log_debug(
                tt::LogBuildKernels,
                "Using precompiled kernel binaries. kernel_name={}, compile_hash={}, precompiled_dir={}",
                kernel->name(),
                kernel_hash,
                precompiled_config->precompiled_dir);
            return precompiled_config->precompiled_dir;
        }

        if (precompiled_config->fallback_policy == experimental::PrecompiledKernelConfig::FallbackPolicy::Error) {
            throw experimental::PrecompiledKernelNotFoundError(
                kernel->name(), kernel_hash, precompiled_config->precompiled_dir, precompiled_config->fallback_policy);
        }
    }

    jit_build_once(kernel_hash, [&] {
        try {
            jit_build_genfiles_descriptors(build_env.build_env, build_options);
            kernel->generate_binaries(device, build_options);
        } catch (std::runtime_error& ex) {
            TT_THROW("Failed to generate binaries for {} {}", kernel->name(), ex.what());
        }
    });
    return build_env.build_env.get_out_kernel_root_path();
}
}  // namespace

namespace experimental {

void ClearKernelCache() { jit_build_cache_clear(); }

}  // namespace experimental

std::atomic<uint64_t> detail::ProgramImpl::program_counter = 0;

detail::ProgramImpl::ProgramImpl() :

    cached_device_hash_(std::nullopt),
    programmable_core_count_(MetalContext::instance().hal().get_programmable_core_type_count()),
    max_cbs_(MetalContext::instance().hal().get_arch_num_circular_buffers()),
    id(program_counter++) {
    for (uint32_t i = 0; i < programmable_core_count_; i++) {
        kernels_.push_back({});
        grid_extent_.push_back({});
        kernel_groups_.push_back({});
        core_to_kernel_group_index_table_.push_back({});
    }

    TT_ASSERT(
        cb_mask_width_ >= max_cbs_,
        "CB mask width ({}) is insufficient for architecture's {} CBs",
        cb_mask_width_,
        max_cbs_);

    program_configs_.resize(programmable_core_count_);
    program_config_sizes_.resize(programmable_core_count_ + 2);

    Inspector::program_created(this);
}

detail::ProgramImpl::~ProgramImpl() noexcept {
    // Deallocate circular buffers and unregister from devices
    deallocate_circular_buffers();
    Inspector::program_destroyed(this);
}

Program::Program() : internal_(std::make_shared<detail::ProgramImpl>()) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureProgramConstructor, *this);
}

Program::Program(std::shared_ptr<detail::ProgramImpl> impl) : internal_(std::move(impl)) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureProgramConstructor, *this);
}

Program::Program(const ProgramDescriptor& descriptor) : internal_(std::make_shared<detail::ProgramImpl>()) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureProgramConstructor, *this);

    for (const auto& cb_descriptor : descriptor.cbs) {
        internal_->add_circular_buffer_(std::make_shared<CircularBufferImpl>(cb_descriptor));
    }

    for (const auto& semaphore_descriptor : descriptor.semaphores) {
        internal_->add_semaphore(
            semaphore_descriptor.core_ranges,
            semaphore_descriptor.id,
            semaphore_descriptor.initial_value,
            semaphore_descriptor.core_type);
    }

    for (const auto& kernel_descriptor : descriptor.kernels) {
        bool is_file = kernel_descriptor.source_type == KernelDescriptor::SourceType::FILE_PATH;
        std::vector<uint32_t> compile_args(
            kernel_descriptor.compile_time_args.begin(), kernel_descriptor.compile_time_args.end());
        std::map<std::string, std::string> defines(kernel_descriptor.defines.begin(), kernel_descriptor.defines.end());
        std::unordered_map<std::string, uint32_t> named_compile_args(
            kernel_descriptor.named_compile_time_args.begin(), kernel_descriptor.named_compile_time_args.end());

        std::vector<std::filesystem::path> compiler_include_paths(
            kernel_descriptor.compiler_include_paths.begin(), kernel_descriptor.compiler_include_paths.end());

        auto config = std::visit(
            ttsl::overloaded{
                [&](const ReaderConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig> {
                    return ReaderDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
                        std::move(named_compile_args),
                        kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2),
                        std::move(compiler_include_paths)};
                },
                [&](const WriterConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig> {
                    return WriterDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
                        std::move(named_compile_args),
                        kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2),
                        std::move(compiler_include_paths)};
                },
                [&](const DataMovementConfigDescriptor& dm_descriptor)
                    -> std::variant<DataMovementConfig, ComputeConfig> {
                    return DataMovementConfig{
                        .processor = dm_descriptor.processor,
                        .noc = dm_descriptor.noc,
                        .noc_mode = dm_descriptor.noc_mode,
                        .compile_args = std::move(compile_args),
                        .defines = std::move(defines),
                        .named_compile_args = std::move(named_compile_args),
                        .opt_level = kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2),
                        .compiler_include_paths = std::move(compiler_include_paths),
                    };
                },
                [&](const ComputeConfigDescriptor& compute_descriptor)
                    -> std::variant<DataMovementConfig, ComputeConfig> {
                    return ComputeConfig{
                        .math_fidelity = compute_descriptor.math_fidelity,
                        .fp32_dest_acc_en = compute_descriptor.fp32_dest_acc_en,
                        .dst_full_sync_en = compute_descriptor.dst_full_sync_en,
                        .unpack_to_dest_mode = compute_descriptor.unpack_to_dest_mode,
                        .bfp8_pack_precise = compute_descriptor.bfp8_pack_precise,
                        .math_approx_mode = compute_descriptor.math_approx_mode,
                        .compile_args = std::move(compile_args),
                        .defines = std::move(defines),
                        .named_compile_args = std::move(named_compile_args),
                        .opt_level = kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O3),
                        .compiler_include_paths = std::move(compiler_include_paths),
                    };
                },
            },
            kernel_descriptor.config);

        auto kernel_handle =
            is_file
                ? CreateKernel(*this, kernel_descriptor.kernel_source, kernel_descriptor.core_ranges, config)
                : CreateKernelFromString(*this, kernel_descriptor.kernel_source, kernel_descriptor.core_ranges, config);

        for (const auto& [core_coord, core_runtime_args] : kernel_descriptor.runtime_args) {
            SetRuntimeArgs(*this, kernel_handle, core_coord, core_runtime_args);
        }
        SetCommonRuntimeArgs(*this, kernel_handle, kernel_descriptor.common_runtime_args);
    }
}

namespace {

std::bitset<MAX_PROCESSOR_TYPES_COUNT> get_kernel_processor_set(const Kernel& kernel) {
    std::bitset<MAX_PROCESSOR_TYPES_COUNT> set;
    for (int i = 0; i < kernel.expected_num_binaries(); i++) {
        int processor_id = kernel.get_kernel_processor_type(i);
        TT_ASSERT(0 <= processor_id && processor_id < MAX_PROCESSOR_TYPES_COUNT);
        set.set(processor_id);
    }
    return set;
}

}  // namespace

KernelHandle detail::ProgramImpl::add_kernel(
    const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& programmable_core_type) {
    TT_FATAL(this->compiled_.empty(), "Cannot add kernel to an already compiled program {}", this->id);
    // Id is unique across all kernels on all core types
    KernelHandle id = this->num_kernels();
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);

    auto new_kernel_core_type = kernel->get_kernel_programmable_core_type();
    auto new_kernel_processor_class = kernel->get_kernel_processor_class();

    std::set<CoreCoord> kernel_logical_cores = kernel->logical_cores();
    auto new_kernel_processor_set = get_kernel_processor_set(*kernel);
    for (size_t i = 0; i < this->num_kernels(); i++) {
        // Note, looks like id is program specific, and increments naturally as kernels are added.
        //  add_kernel -> id = num_kernels -> kernel is inserted -> next num_kernels() increments.
        std::shared_ptr<Kernel> check_kernel = this->get_kernel(i);
        auto check_kernel_core_type = check_kernel->get_kernel_programmable_core_type();
        auto check_kernel_processor_class = check_kernel->get_kernel_processor_class();
        if (check_kernel_core_type == new_kernel_core_type &&
            check_kernel_processor_class == new_kernel_processor_class &&
            (new_kernel_processor_set & get_kernel_processor_set(*check_kernel)).any()) {
            // Two kernels are using the same processor, need to check core ranges.
            std::set<CoreCoord> check_kernel_logical_cores = check_kernel->logical_cores();
            for (CoreCoord coreCoord : kernel_logical_cores) {
                TT_FATAL(
                    !check_kernel_logical_cores.contains(coreCoord),
                    "Core Overlap Between (\"{}\") and new kernel (\"{}\") at {}",
                    check_kernel->name(),
                    kernel->name(),
                    coreCoord.str());
            }
        }
    }

    kernels_[index].insert({id, kernel});
    kernel_groups_[index].resize(0);
    core_to_kernel_group_index_table_[index].clear();
    return id;
}

std::shared_ptr<Kernel> detail::ProgramImpl::get_kernel(KernelHandle kernel_id) const {
    // TT_ASSERT(kernel_id < this->kernels_.size(), "Expected Kernel with ID {} to be in Program {}", kernel_id,
    // this->id);
    //  find coretype based on kernel_id
    for (const auto& kernels : this->kernels_) {
        if (kernels.contains(kernel_id)) {
            return kernels.at(kernel_id);
        }
    }

    TT_ASSERT(false, "Did not find kernel id across all core types!");
    return nullptr;
}

// ============================================================================
// Metal 2.0 Name Registry Methods
// ============================================================================

void ProgramImpl::register_kernel_spec_name(const std::string& name, KernelHandle handle) {
    if (!metal2_registry_) {
        metal2_registry_ = Metal2NameRegistry{};
    }
    auto [it, inserted] = metal2_registry_->kernel_handles.try_emplace(name, handle);
    TT_FATAL(inserted, "Duplicate kernel spec name: {}", name);
}

void ProgramImpl::set_dfb_alias(uint32_t primary_id, uint32_t secondary_id) {
    TT_FATAL(
        primary_id < dataflow_buffers_.size(),
        "set_dfb_alias: primary DFB id {} has not been created yet (only {} DFBs exist). "
        "Both DFBs must be created via add_dataflow_buffer before aliasing.",
        primary_id,
        dataflow_buffers_.size());
    TT_FATAL(
        secondary_id < dataflow_buffers_.size(),
        "set_dfb_alias: secondary DFB id {} has not been created yet (only {} DFBs exist). "
        "Both DFBs must be created via add_dataflow_buffer before aliasing.",
        secondary_id,
        dataflow_buffers_.size());
    TT_FATAL(
        primary_id != secondary_id,
        "set_dfb_alias: cannot alias a DFB with itself. Primary and secondary DFB IDs must be different");

    auto& primary_dfb = dataflow_buffers_[primary_id];
    auto& secondary_dfb = dataflow_buffers_[secondary_id];

    TT_FATAL(
        !primary_dfb->alias_primary_id.has_value(),
        "set_dfb_alias: primary DFB id {} is already a secondary of DFB id {}. Alias chains are not allowed.",
        primary_id,
        primary_dfb->alias_primary_id.value());
    TT_FATAL(
        !secondary_dfb->alias_primary_id.has_value(),
        "set_dfb_alias: secondary DFB id {} is already aliased to primary DFB id {}.",
        secondary_id,
        secondary_dfb->alias_primary_id.value());

    dataflow_buffers_[primary_id]->alias_secondary_ids.push_back(secondary_id);
    dataflow_buffers_[secondary_id]->alias_primary_id = primary_id;
}

void ProgramImpl::register_dfb_spec_name(const std::string& name, uint32_t dfb_id) {
    if (!metal2_registry_) {
        metal2_registry_ = Metal2NameRegistry{};
    }
    auto [it, inserted] = metal2_registry_->dfb_handles.try_emplace(name, dfb_id);
    TT_FATAL(inserted, "Duplicate DFB spec name: {}", name);
}

void ProgramImpl::register_semaphore_spec_name(const std::string& name, uint32_t sem_id) {
    if (!metal2_registry_) {
        metal2_registry_ = Metal2NameRegistry{};
    }
    auto [it, inserted] = metal2_registry_->semaphore_handles.try_emplace(name, sem_id);
    TT_FATAL(inserted, "Duplicate semaphore spec name: {}", name);
}

KernelHandle ProgramImpl::get_kernel_handle(const std::string& name) const {
    TT_FATAL(metal2_registry_, "Metal 2.0 registry not initialized (program was not created from ProgramSpec)");
    auto it = metal2_registry_->kernel_handles.find(name);
    TT_FATAL(it != metal2_registry_->kernel_handles.end(), "Unknown kernel spec name: {}", name);
    return it->second;
}

uint32_t ProgramImpl::get_dfb_handle(const std::string& name) const {
    TT_FATAL(metal2_registry_, "Metal 2.0 registry not initialized (program was not created from ProgramSpec)");
    auto it = metal2_registry_->dfb_handles.find(name);
    TT_FATAL(it != metal2_registry_->dfb_handles.end(), "Unknown DFB spec name: {}", name);
    return it->second;
}

uint32_t ProgramImpl::get_semaphore_handle(const std::string& name) const {
    TT_FATAL(metal2_registry_, "Metal 2.0 registry not initialized (program was not created from ProgramSpec)");
    auto it = metal2_registry_->semaphore_handles.find(name);
    TT_FATAL(it != metal2_registry_->semaphore_handles.end(), "Unknown semaphore spec name: {}", name);
    return it->second;
}

void ProgramImpl::register_kernel_rta_schema(const std::string& name, const KernelRTASchema& schema) {
    if (!metal2_registry_) {
        metal2_registry_ = Metal2NameRegistry{};
    }
    auto [it, inserted] = metal2_registry_->kernel_rta_schemas.try_emplace(name, schema);
    TT_FATAL(inserted, "Duplicate kernel RTA schema for: {}", name);
}

const ProgramImpl::KernelRTASchema* ProgramImpl::get_kernel_rta_schema(const std::string& name) const {
    if (!metal2_registry_) {
        return nullptr;
    }
    auto it = metal2_registry_->kernel_rta_schemas.find(name);
    if (it == metal2_registry_->kernel_rta_schemas.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<std::string> ProgramImpl::get_registered_kernel_names() const {
    std::vector<std::string> names;
    if (metal2_registry_) {
        names.reserve(metal2_registry_->kernel_handles.size());
        for (const auto& [name, handle] : metal2_registry_->kernel_handles) {
            names.push_back(name);
        }
    }
    return names;
}

void ProgramImpl::register_tensor_parameter(
    const std::string& name,
    const TensorSpec& spec,
    bool dynamic_tensor_shape,
    bool match_padded_shape_only,
    bool enqueue_invariant) {
    if (!metal2_registry_) {
        metal2_registry_ = Metal2NameRegistry{};
    }
    auto [it, inserted] = metal2_registry_->tensor_parameter_layouts.try_emplace(
        name,
        Metal2NameRegistry::RegisteredTensorParameter{
            spec, dynamic_tensor_shape, match_padded_shape_only, enqueue_invariant});
    TT_FATAL(inserted, "Duplicate tensor parameter name: {}", name);
}

const TensorSpec* ProgramImpl::get_tensor_parameter_layout(const std::string& name) const {
    if (!metal2_registry_) {
        return nullptr;
    }
    auto it = metal2_registry_->tensor_parameter_layouts.find(name);
    if (it == metal2_registry_->tensor_parameter_layouts.end()) {
        return nullptr;
    }
    return &it->second.spec;
}

bool ProgramImpl::get_tensor_parameter_dynamic_tensor_shape(const std::string& name) const {
    if (!metal2_registry_) {
        return false;
    }
    auto it = metal2_registry_->tensor_parameter_layouts.find(name);
    if (it == metal2_registry_->tensor_parameter_layouts.end()) {
        return false;
    }
    return it->second.dynamic_tensor_shape;
}

bool ProgramImpl::get_tensor_parameter_match_padded_shape_only(const std::string& name) const {
    if (!metal2_registry_) {
        return false;
    }
    auto it = metal2_registry_->tensor_parameter_layouts.find(name);
    if (it == metal2_registry_->tensor_parameter_layouts.end()) {
        return false;
    }
    return it->second.match_padded_shape_only;
}

bool ProgramImpl::get_tensor_parameter_enqueue_invariant(const std::string& name) const {
    if (!metal2_registry_) {
        return false;
    }
    auto it = metal2_registry_->tensor_parameter_layouts.find(name);
    if (it == metal2_registry_->tensor_parameter_layouts.end()) {
        return false;
    }
    return it->second.enqueue_invariant;
}

std::vector<std::string> ProgramImpl::get_registered_tensor_parameter_names() const {
    std::vector<std::string> names;
    if (metal2_registry_) {
        names.reserve(metal2_registry_->tensor_parameter_layouts.size());
        for (const auto& [name, entry] : metal2_registry_->tensor_parameter_layouts) {
            names.push_back(name);
        }
    }
    return names;
}

void ProgramImpl::register_dfb_borrowed_binding(uint32_t dfb_id, const std::string& tensor_parameter_name) {
    if (!metal2_registry_) {
        metal2_registry_ = Metal2NameRegistry{};
    }
    metal2_registry_->dfb_borrowed_bindings.emplace_back(dfb_id, tensor_parameter_name);
}

const std::vector<std::pair<uint32_t, std::string>>& ProgramImpl::get_dfb_borrowed_bindings() const {
    static const std::vector<std::pair<uint32_t, std::string>> empty;
    if (!metal2_registry_) {
        return empty;
    }
    return metal2_registry_->dfb_borrowed_bindings;
}
// ============================================================================

std::vector<detail::KernelMeta> detail::collect_kernel_meta(const Program& program, IDevice* device) {
    return program.impl().collect_kernel_meta(device);
}

std::vector<detail::KernelMeta> ProgramImpl::collect_kernel_meta(IDevice* device) const {
    std::vector<detail::KernelMeta> result;
    result.reserve(this->num_kernels());
    for (const auto& m : this->kernels_) {
        for (const auto& [id, kernel] : m) {
            result.push_back(kernel->meta(device));
        }
    }
    return result;
}

KernelGroup::KernelGroup(
    const detail::ProgramImpl& program,
    uint32_t programmable_core_type_index,
    std::vector<KernelHandle> kernel_ids,
    uint64_t local_cb_mask,
    uint32_t min_remote_cb_start_index,
    const CoreRangeSet& new_ranges,
    const dev_msgs::Factory& dev_msgs_factory) :
    programmable_core_type_index(programmable_core_type_index),

    kernel_ids(std::move(kernel_ids)),
    launch_msg(dev_msgs_factory.create<dev_msgs::launch_msg_t>()),
    go_msg(dev_msgs_factory.create<dev_msgs::go_msg_t>()) {
    this->core_ranges = this->core_ranges.merge(new_ranges);

    auto kernel_config = this->launch_msg.view().kernel_config();
    kernel_config.brisc_noc_mode() = NOC_MODE::DM_DEDICATED_NOC;

    // Slow dispatch uses fixed addresses for the kernel config, configured here statically
    // Fast dispatch kernel config management happens under the CQ and will re-program the base
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        kernel_config.kernel_config_base()[index] =
            hal.get_dev_addr(hal.get_programmable_core_type(index), HalL1MemAddrType::KERNEL_CONFIG);
    }

    std::set<NOC_MODE> noc_modes;
    for (auto kernel_id : this->kernel_ids) {
        const auto kernel = program.get_kernel(kernel_id);
        auto num_binaries = kernel->expected_num_binaries();
        for (uint32_t i = 0; i < num_binaries; i++) {
            std::vector<uint32_t> processor_indices = kernel->get_processor_indices_for_binary(i);
            for (uint32_t processor_index : processor_indices) {
                kernel_config.watcher_kernel_ids()[processor_index] = kernel->get_watcher_kernel_id();
                kernel_config.enables() |= 1u << processor_index;
            }
        }

        // Dynamic NOC assignment is only supported on certain core types
        const bool is_tensix_core =
            hal.get_programmable_core_type(programmable_core_type_index) == HalProgrammableCoreType::TENSIX;
        const bool is_supported_eth_core =
            hal.get_programmable_core_type(programmable_core_type_index) == HalProgrammableCoreType::ACTIVE_ETH &&
            !hal.get_eth_fw_is_cooperative();
        if (is_tensix_core || is_supported_eth_core) {
            std::visit(
                [&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, DataMovementConfig> || std::is_same_v<T, EthernetConfig>) {
                        // The code below sets the brisc_noc_id for use by the device firmware
                        // Use 0 if neither brisc nor ncrisc specify a noc
                        if (kernel->get_kernel_processor_type(0) ==
                            ttsl::as_underlying_type<DataMovementProcessor>(DataMovementProcessor::RISCV_0)) {
                            noc_modes.insert(arg.noc_mode);
                            // Use brisc's noc if brisc specifies a noc
                            kernel_config.brisc_noc_id() = arg.noc;
                            // if noc mode is already set to DM_DYNAMIC_NOC then we can't change back to
                            // DM_DEDICATED_NOC
                            if (arg.noc_mode == NOC_MODE::DM_DYNAMIC_NOC) {
                                kernel_config.brisc_noc_mode() = NOC_MODE::DM_DYNAMIC_NOC;
                            }
                        } else if (
                            kernel->get_kernel_processor_type(0) ==
                            ttsl::as_underlying_type<DataMovementProcessor>(DataMovementProcessor::RISCV_1)) {
                            noc_modes.insert(arg.noc_mode);
                            // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
                            // If both brisc and ncrisc set the noc, then this is safe due to prior correctness
                            // validation
                            kernel_config.brisc_noc_id() = 1 - arg.noc;
                            // if noc mode is already set to DM_DYNAMIC_NOC then we can't change back to
                            // DM_DEDICATED_NOC
                            if (arg.noc_mode == NOC_MODE::DM_DYNAMIC_NOC) {
                                kernel_config.brisc_noc_mode() = NOC_MODE::DM_DYNAMIC_NOC;
                            }
                        }
                    }
                },
                kernel->config());
        }

        // Quasar: set per-processor num_sw_threads and kernel_thread_id for dm/runtime access
        if (auto* qk = dynamic_cast<experimental::quasar::QuasarDataMovementKernel*>(kernel.get())) {
            auto config = std::get<experimental::quasar::QuasarDataMovementConfig>(qk->config());
            const auto& dm_cores = qk->get_dm_processors();
            for (uint32_t thread_idx = 0; thread_idx < dm_cores.size(); thread_idx++) {
                uint32_t processor_index = hal.get_processor_index(
                    hal.get_programmable_core_type(programmable_core_type_index),
                    HalProcessorClassType::DM,
                    qk->get_kernel_processor_type(static_cast<int>(thread_idx)));
                kernel_config.num_sw_threads()[processor_index] = config.num_threads_per_cluster;
                kernel_config.kernel_thread_id()[processor_index] = thread_idx;
            }
        }
        // Quasar: set per-processor num_sw_threads and kernel_thread_id for trisc/runtime access
        if (auto* qk = dynamic_cast<experimental::quasar::QuasarComputeKernel*>(kernel.get())) {
            auto config = std::get<experimental::quasar::QuasarComputeConfig>(qk->config());
            const auto& compute_cores = qk->get_compute_processors();
            // Track which NEOs have been used to ensure we don't use the same NEO multiple times

            // Every trisc core in a single NEO/Tensix engine shares the same num_sw_threads and kernel_thread_id
            std::set<uint32_t> neo_indices_set;
            for (auto compute_core : compute_cores) {
                switch (compute_core) {
                    case experimental::quasar::QuasarComputeProcessor::NEO_0_COMPUTE_0:
                    case experimental::quasar::QuasarComputeProcessor::NEO_0_COMPUTE_1:
                    case experimental::quasar::QuasarComputeProcessor::NEO_0_COMPUTE_2:
                    case experimental::quasar::QuasarComputeProcessor::NEO_0_COMPUTE_3:
                        neo_indices_set.insert(0);
                        break;
                    case experimental::quasar::QuasarComputeProcessor::NEO_1_COMPUTE_0:
                    case experimental::quasar::QuasarComputeProcessor::NEO_1_COMPUTE_1:
                    case experimental::quasar::QuasarComputeProcessor::NEO_1_COMPUTE_2:
                    case experimental::quasar::QuasarComputeProcessor::NEO_1_COMPUTE_3:
                        neo_indices_set.insert(1);
                        break;
                    case experimental::quasar::QuasarComputeProcessor::NEO_2_COMPUTE_0:
                    case experimental::quasar::QuasarComputeProcessor::NEO_2_COMPUTE_1:
                    case experimental::quasar::QuasarComputeProcessor::NEO_2_COMPUTE_2:
                    case experimental::quasar::QuasarComputeProcessor::NEO_2_COMPUTE_3:
                        neo_indices_set.insert(2);
                        break;
                    case experimental::quasar::QuasarComputeProcessor::NEO_3_COMPUTE_0:
                    case experimental::quasar::QuasarComputeProcessor::NEO_3_COMPUTE_1:
                    case experimental::quasar::QuasarComputeProcessor::NEO_3_COMPUTE_2:
                    case experimental::quasar::QuasarComputeProcessor::NEO_3_COMPUTE_3:
                        neo_indices_set.insert(3);
                        break;
                }
            }
            std::vector<uint32_t> neo_indices_used(neo_indices_set.begin(), neo_indices_set.end());
            TT_ASSERT(
                neo_indices_used.size() == config.num_threads_per_cluster,
                "Number of NEOs used must match number of threads per cluster");
            // Now that we know which NEOs have been used, we can set the num_sw_threads and kernel_thread_id for each
            // Tensix engine
            for (uint32_t thread_idx = 0; thread_idx < config.num_threads_per_cluster; thread_idx++) {
                uint32_t neo_id = neo_indices_used[thread_idx];
                // First set of indices are used for DM cores, second set are used for Tensix engines
                uint32_t config_index = experimental::quasar::QUASAR_NUM_DM_CORES_PER_CLUSTER + neo_id;
                kernel_config.num_sw_threads()[config_index] = config.num_threads_per_cluster;
                kernel_config.kernel_thread_id()[config_index] = thread_idx;
            }
        }
    }
    TT_FATAL(noc_modes.size() <= 1, "KernelGroup must have the same noc mode for all kernels");

    kernel_config.exit_erisc_kernel() = false;
    kernel_config.local_cb_mask() = local_cb_mask;
    kernel_config.min_remote_cb_start_index() = min_remote_cb_start_index;
    this->go_msg.view().signal() = dev_msgs::RUN_MSG_GO;
}

CoreType KernelGroup::get_core_type() const {
    return MetalContext::instance().hal().get_core_type(this->programmable_core_type_index);
};

std::vector<std::shared_ptr<KernelGroup>>& detail::ProgramImpl::get_kernel_groups(
    uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    return kernel_groups_[programmable_core_type_index];
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& detail::ProgramImpl::get_kernels(
    uint32_t programmable_core_type_index) {
    return this->kernels_.at(programmable_core_type_index);
}

KernelGroup* detail::ProgramImpl::kernels_on_core(const CoreCoord& core, uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    if (core.x >= grid_extent_[programmable_core_type_index].x ||
        core.y >= grid_extent_[programmable_core_type_index].y) {
        return nullptr;
    }
    uint8_t index = core_to_kernel_group_index_table_[programmable_core_type_index].at(
        (core.y * grid_extent_[programmable_core_type_index].x) + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr
                                                         : kernel_groups_[programmable_core_type_index].at(index).get();
}

void detail::ProgramImpl::update_kernel_groups(uint32_t programmable_core_type_index) {
    if (core_to_kernel_group_index_table_[programmable_core_type_index].empty()) {
        // Get the extent of the kernels in x, y
        CoreCoord base = {std::numeric_limits<decltype(base.x)>::max(), std::numeric_limits<decltype(base.y)>::max()};
        grid_extent_[programmable_core_type_index] = {0, 0};
        const auto& handle_to_kernel = kernels_[programmable_core_type_index];
        for (const auto& [id, kernel] : handle_to_kernel) {
            for (auto core : kernel->logical_cores()) {
                grid_extent_[programmable_core_type_index].x =
                    std::max(core.x, grid_extent_[programmable_core_type_index].x);
                grid_extent_[programmable_core_type_index].y =
                    std::max(core.y, grid_extent_[programmable_core_type_index].y);
                base.x = std::min(core.x, base.x);
                base.y = std::min(core.y, base.y);
            }
        }
        grid_extent_[programmable_core_type_index].x++;
        grid_extent_[programmable_core_type_index].y++;

        // grid maps cores to sets-of-kernels running on that core
        size_t grid_size = grid_extent_[programmable_core_type_index].x * grid_extent_[programmable_core_type_index].y;
        std::vector<bool> valid(grid_size, false);
        std::vector<std::set<KernelHandle>> grid(grid_size);
        for (const auto& [id, kernel] : handle_to_kernel) {
            for (auto core : kernel->logical_cores()) {
                int core_index = (core.y * grid_extent_[programmable_core_type_index].x) + core.x;
                valid[core_index] = true;
                grid[core_index].insert(id);
            }
        }

        // Flip the mapping to get sets-of-kernels to cores
        std::map<std::set<KernelHandle>, std::set<CoreRange>> map;
        for (auto y = base.y; y < grid_extent_[programmable_core_type_index].y; y++) {
            for (auto x = base.x; x < grid_extent_[programmable_core_type_index].x; x++) {
                int index = (y * grid_extent_[programmable_core_type_index].x) + x;
                if (valid[index]) {
                    // grid is not used any more. Avoid copy construction by moving.
                    auto [it, inserted] = map.try_emplace(std::move(grid[index]));
                    it->second.insert(CoreRange({x, y}, {x, y}));
                }
            }
        }

        // Build the list of KernelGroups with merged core range sets from the
        // mapping of sets-of-kernels to cores
        TT_ASSERT(map.size() < core_to_kernel_group_invalid_index);
        kernel_groups_.reserve(map.size());
        int index = 0;
        core_to_kernel_group_index_table_[programmable_core_type_index].resize(
            grid_extent_[programmable_core_type_index].x * grid_extent_[programmable_core_type_index].y,
            core_to_kernel_group_invalid_index);
        const auto& hal = MetalContext::instance().hal();
        for (auto& [kernels, cores] : map) {
            // Start inclusive, max exclusive
            uint32_t max_local_cb_end_index = 0;
            uint32_t min_remote_cb_start_index = max_cbs_;
            uint64_t local_cb_mask = 0;
            uint32_t num_dfbs = 0;

            // Map from core X,Y back to the unique KernelGroup
            for (CoreRange range : cores) {
                bool logged_noncontiguous = false;
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                        core_to_kernel_group_index_table_[programmable_core_type_index]
                                                         [(y * grid_extent_[programmable_core_type_index].x) + x] =
                                                             index;

                        if (not hal.get_supports_cbs(programmable_core_type_index)) {
                            continue;
                        }
                        auto core = CoreCoord({x, y});
                        auto local_val = per_core_local_cb_indices_.find(core);
                        if (local_val != per_core_local_cb_indices_.end() && local_val->second.any()) {
                            uint64_t used_cbs = local_val->second.to_ullong();
                            local_cb_mask |= used_cbs;
                            uint32_t calculated_index =
                                cb_mask_width_ - static_cast<uint32_t>(__builtin_clzll(used_cbs));
                            max_local_cb_end_index = std::max(max_local_cb_end_index, calculated_index);
                            if (!logged_noncontiguous) {
                                // Zeroes out the contiguous run of set bits starting at zero. Anything remaining is
                                // above a zero bit.
                                uint64_t non_contiguous_cbs = used_cbs & (used_cbs + 1);
                                if (non_contiguous_cbs) {
                                    // ~used_cbs is always nonzero, because otherwise all CBs are in use and therefore
                                    // contiguous.
                                    uint32_t first_unused_index = static_cast<uint32_t>(__builtin_ctzll(~used_cbs));
                                    std::string kernels_str;
                                    for (auto id : kernels) {
                                        std::shared_ptr<Kernel> kernel = handle_to_kernel.at(id);
                                        if (!kernels_str.empty()) {
                                            kernels_str += ", ";
                                        }
                                        kernels_str += kernel->kernel_source().name();
                                    }

                                    static std::mutex m;
                                    std::lock_guard lock(m);
                                    // Keep track of which programs have been logged to avoid spamming the log. This is
                                    // particularly important for mesh devices.
                                    static std::set<std::tuple<uint64_t, uint32_t, std::string>> logged;
                                    auto cb_tuple =
                                        std::make_tuple(non_contiguous_cbs, first_unused_index, kernels_str);

                                    if (!logged.contains(cb_tuple)) {
                                        logged.insert(cb_tuple);
                                        // This code should be modified to log the core type index if it isn't obvious.
                                        TT_ASSERT(
                                            programmable_core_type_index ==
                                            MetalContext::instance().hal().get_programmable_core_type_index(
                                                HalProgrammableCoreType::TENSIX));

                                        std::string cb_ids;
                                        for (uint32_t i = 0; i < max_cbs_; i++) {
                                            if (non_contiguous_cbs & (1ULL << i)) {
                                                if (!cb_ids.empty()) {
                                                    cb_ids += ",";
                                                }
                                                cb_ids += std::to_string(i);
                                            }
                                        }
                                        log_debug(
                                            tt::LogMetal,
                                            "Circular buffer indices are not contiguous starting at 0. This will hurt "
                                            "dispatch performance. Non-contiguous indices: {}. "
                                            "First unused index: {}. Kernels: {}",
                                            cb_ids,
                                            first_unused_index,
                                            kernels_str);
                                    }
                                    logged_noncontiguous = true;
                                }
                            }
                        }
                        auto remote_val = per_core_remote_cb_indices_.find(core);
                        if (remote_val != per_core_remote_cb_indices_.end() && remote_val->second.any()) {
                            min_remote_cb_start_index = std::min(
                                min_remote_cb_start_index,
                                static_cast<uint32_t>(__builtin_ctzll(remote_val->second.to_ullong())));
                        }

                        if (not hal.get_supports_dfbs(programmable_core_type_index)) {
                            continue;
                        }
                        // Hacky here but dfbs and cbs aren't used interchangeably so use the local_cb_mask on
                        // KernelGroup to hold value of number of dfbs
                        auto local_dfb_val = per_core_num_dfbs_.find(core);
                        if (local_dfb_val != per_core_num_dfbs_.end()) {
                            num_dfbs = local_dfb_val->second;
                        }
                    }
                }
            }
            TT_FATAL(
                max_local_cb_end_index <= min_remote_cb_start_index,
                "Circular buffer indices overlap for KernelGroup {} on programmable core type {}. Local end index {}, "
                "Remote start index {}",
                index,
                programmable_core_type_index,
                max_local_cb_end_index,
                min_remote_cb_start_index);
            TT_FATAL(
                !(local_cb_mask != 0 && num_dfbs != 0),
                "Cannot use both circular buffers and dataflow buffers on the same core. "
                "local_cb_mask: {}, num_dfbs: {}",
                local_cb_mask,
                num_dfbs);
            local_cb_mask = (num_dfbs > 0 && local_cb_mask == 0) ? num_dfbs : local_cb_mask;
            std::vector<KernelHandle> kernel_ids(kernels.begin(), kernels.end());
            // Sort kernel ids by processor index, so loops over this array will be in processor order
            std::sort(kernel_ids.begin(), kernel_ids.end(), [&](KernelHandle a, KernelHandle b) {
                auto ka = handle_to_kernel.at(a);
                auto kb = handle_to_kernel.at(b);
                auto idx_a = hal.get_processor_index(
                    hal.get_programmable_core_type(programmable_core_type_index),
                    ka->get_kernel_processor_class(),
                    ka->get_kernel_processor_type(0));
                auto idx_b = hal.get_processor_index(
                    hal.get_programmable_core_type(programmable_core_type_index),
                    kb->get_kernel_processor_class(),
                    kb->get_kernel_processor_type(0));
                return idx_a < idx_b;
            });
            kernel_groups_[programmable_core_type_index].push_back(std::make_shared<KernelGroup>(
                *this,
                programmable_core_type_index,
                std::move(kernel_ids),
                local_cb_mask,
                min_remote_cb_start_index,
                cores,
                hal.get_dev_msgs_factory(hal.get_programmable_core_type(programmable_core_type_index))));
            index++;
        }
        for (const auto& kg : kernel_groups_[programmable_core_type_index]) {
            RecordKernelGroup(*this, hal.get_programmable_core_type(programmable_core_type_index), *kg);
        }
    }
}

void detail::ProgramImpl::CircularBufferAllocator::mark_address(
    uint64_t address, uint64_t size, uint64_t base_address) {
    if (this->l1_regions.empty()) {
        this->l1_regions.emplace_back(base_address, base_address);
    }
    auto& last_region = this->l1_regions.back();
    if (address < last_region.second) {
        TT_THROW(
            "Local buffer address {} has to append to last L1 region [{}, {}) or be at a higher address",
            address,
            last_region.first,
            last_region.second);
    }
    if (address == last_region.second) {
        last_region.second += size;
    } else {
        this->l1_regions.emplace_back(address, address + size);
    }
}

CBHandle detail::ProgramImpl::add_circular_buffer_(const std::shared_ptr<CircularBufferImpl>& circular_buffer) {
    // Globally allocated circular buffer do not invalidate allocation because their addresses are tracked by memory
    // allocator
    if (not circular_buffer->globally_allocated()) {
        this->invalidate_circular_buffer_allocation();
    } else {
        circular_buffer->assign_global_address();
    }

    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                std::bitset<NUM_CIRCULAR_BUFFERS>& cb_indices = this->per_core_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& local_cb_indices = this->per_core_local_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& remote_cb_indices = this->per_core_remote_cb_indices_[logical_core];
                uint32_t max_cbs = max_cbs_;
                auto add_buffer_indices = [&cb_indices, max_cbs](
                                              const std::unordered_set<uint8_t>& buffer_indices,
                                              std::bitset<NUM_CIRCULAR_BUFFERS>& target_cb_indices) {
                    for (uint32_t buffer_index : buffer_indices) {
                        // TT_ASSERT since we validate when constructing the config that it's within range
                        TT_ASSERT(
                            buffer_index < max_cbs,
                            "Invalid circular buffer index: {} should be between 0 and {}",
                            buffer_index,
                            max_cbs);
                        if (cb_indices[buffer_index]) {
                            TT_THROW(
                                "Invalid circular buffer index: Cannot add circular buffer at index {}, another "
                                "circular "
                                "buffer already exists",
                                buffer_index);
                        }
                        cb_indices[buffer_index] = true;
                        target_cb_indices[buffer_index] = true;
                    }
                };
                add_buffer_indices(circular_buffer->config().local_buffer_indices(), local_cb_indices);
                add_buffer_indices(circular_buffer->config().remote_buffer_indices(), remote_cb_indices);
            }
        }

        // There is one CircularBufferAllocator per unique core range, create one if it does not already exist for
        // current core range
        auto val = std::find_if(
            cb_allocators_.begin(), cb_allocators_.end(), [&core_range](const CircularBufferAllocator& cb_allocator) {
                return cb_allocator.core_range == core_range;
            });
        if (val == cb_allocators_.end()) {
            this->cb_allocators_.emplace_back(core_range);
        }
    }

    this->circular_buffers_.push_back(circular_buffer);
    this->circular_buffer_by_id_.insert({circular_buffer->id(), circular_buffer});
    return circular_buffer->id();
}

CBHandle detail::ProgramImpl::add_circular_buffer(
    const CoreRangeSet& core_range_set, const CircularBufferConfig& config) {
    TT_FATAL(this->compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", this->id);
    TT_FATAL(
        this->dataflow_buffers_.empty(), "Cannot add circular buffer to a program that already has dataflow buffers");
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBufferImpl> circular_buffer =
        std::make_shared<CircularBufferImpl>(core_range_set.merge_ranges(), config);
    return add_circular_buffer_(circular_buffer);
}

CBHandle detail::ProgramImpl::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const experimental::GlobalCircularBuffer& global_circular_buffer) {
    TT_FATAL(this->compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", this->id);
    TT_FATAL(
        this->dataflow_buffers_.empty(), "Cannot add circular buffer to a program that already has dataflow buffers");
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBufferImpl> circular_buffer =
        std::make_shared<CircularBufferImpl>(core_range_set.merge_ranges(), config, global_circular_buffer);
    return add_circular_buffer_(circular_buffer);
}

std::shared_ptr<CircularBufferImpl> detail::ProgramImpl::get_circular_buffer(CBHandle cb_id) const {
    if (!this->circular_buffer_by_id_.contains(cb_id)) {
        TT_THROW("No circular buffer with id {} exists in Program {}", cb_id, this->id);
    }
    return this->circular_buffer_by_id_.at(cb_id);
}

std::vector<std::shared_ptr<CircularBufferImpl>> detail::ProgramImpl::circular_buffers_on_core(
    const CoreCoord& core) const {
    std::vector<std::shared_ptr<CircularBufferImpl>> cbs_on_core;
    for (const auto& circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<std::shared_ptr<CircularBufferImpl>> detail::ProgramImpl::circular_buffers_on_corerange(
    const CoreRange& cr) const {
    std::vector<std::shared_ptr<CircularBufferImpl>> cbs_on_core;
    for (const auto& circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_corerange(cr)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<CoreRange> detail::ProgramImpl::circular_buffers_unique_coreranges() const {
    std::vector<CoreRange> core_ranges;
    for (const auto& circular_buffer : circular_buffers_) {
        for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
            if (std::find(core_ranges.begin(), core_ranges.end(), core_range) == core_ranges.end()) {
                core_ranges.push_back(core_range);
            }
        }
    }

    // Fast path: if no ranges overlap, return as-is.
    bool has_overlap = false;
    for (size_t i = 0; i < core_ranges.size() && !has_overlap; ++i) {
        for (size_t j = i + 1; j < core_ranges.size(); ++j) {
            if (core_ranges[i].intersects(core_ranges[j])) {
                has_overlap = true;
                break;
            }
        }
    }
    if (!has_overlap) {
        return core_ranges;
    }

    // Make ranges non-overlapping so that each core is targeted by exactly one
    // multicast during CB config dispatch.
    //
    // During dispatch, each CoreRange's payload is built from all CBs whose
    // core_ranges intersect it. If a range intersects two CBs with the same
    // buffer index but different configs (valid when their core_ranges don't
    // overlap), the second CB overwrites the first in the payload and the
    // multicast sends the wrong config to cores that need the other.
    //
    // Split every overlapping pair into three non-overlapping pieces (A\B, A∩B,
    // B\A) and repeat until no overlaps remain. Each resulting piece's boundaries
    // align with all original CB CoreRange edges, so it cannot straddle two
    // different configs for the same buffer index.
    std::vector<CoreRange> result = std::move(core_ranges);
    size_t i = 0;
    while (i < result.size()) {
        // Find the first range that overlaps with result[i].
        size_t j = i + 1;
        for (; j < result.size(); ++j) {
            if (result[i].intersects(result[j])) {
                break;
            }
        }
        if (j == result.size()) {
            ++i;  // No overlap, advance.
            continue;
        }
        // Split the overlapping pair into non-overlapping pieces: A\B, A∩B, B\A.
        CoreRangeSet a_set(result[i]), b_set(result[j]);
        result.erase(result.begin() + j);
        result.erase(result.begin() + i);
        auto a_only = a_set.subtract(b_set);
        auto b_only = b_set.subtract(a_set);
        auto common = a_set.intersection(b_set);
        result.insert(result.end(), a_only.ranges().begin(), a_only.ranges().end());
        result.insert(result.end(), b_only.ranges().begin(), b_only.ranges().end());
        result.insert(result.end(), common.ranges().begin(), common.ranges().end());
        i = 0;  // Restart — new pieces may overlap with earlier ranges.
    }
    return result;
}

void detail::ProgramImpl::invalidate_circular_buffer_allocation() {
    if (this->local_circular_buffer_allocation_needed_) {
        return;
    }
    for (CircularBufferAllocator& cb_allocator : this->cb_allocators_) {
        cb_allocator.reset_available_addresses();
    }
    this->local_circular_buffer_allocation_needed_ = true;
}

// Scratchpad is a Metal 2.0-only construct.
void detail::ProgramImpl::allocate_scratchpads(const IDevice* device) {
    if (this->scratchpads_allocated_) {
        return;
    }

    const uint64_t base_l1_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t alignment = device->allocator()->get_alignment(BufferType::DRAM);

    for (auto& kernels_of_core_type : this->kernels_) {
        for (auto& [kernel_handle, kernel] : kernels_of_core_type) {
            auto& scratchpad_handles = kernel->scratchpad_binding_handles();
            if (scratchpad_handles.empty()) {
                continue;
            }
            const CoreRangeSet& kernel_cores = kernel->core_range_set();

            for (auto& handle : scratchpad_handles) {
                // A scratchpad bumps onto the program-scope L1 region, stacking on top of any DFBs.
                // (DFBs and CBs are mutually exclusive, so dfb_allocators_ own the whole region.)
                // Ensure a CircularBufferAllocator exists for each of the kernel's core ranges:
                // a scratchpad-bearing kernel may have no DFBs, so the allocators may not exist yet.
                for (const CoreRange& core_range : kernel_cores.ranges()) {
                    bool exists = false;
                    for (const CircularBufferAllocator& a : this->dfb_allocators_) {
                        if (a.core_range == core_range) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) {
                        this->dfb_allocators_.emplace_back(core_range);
                    }
                }

                // Uniform per-node base address: the scratchpad address is delivered as a CRTA.
                // It must sit at the same L1 offset everywhere that it exists.
                // Take the max region-end over EVERY allocator that intersects the kernel's cores
                // (not just exact-range matches), so the scratchpad cannot overlap a DFB on
                // an overlapping-but-different core range. Mark each such allocator exactly once.
                std::vector<CircularBufferAllocator*> touched;
                for (CircularBufferAllocator& a : this->dfb_allocators_) {
                    for (const CoreRange& core_range : kernel_cores.ranges()) {
                        if (a.core_range.intersects(core_range)) {
                            touched.push_back(&a);
                            break;
                        }
                    }
                }
                uint64_t addr = base_l1_address;
                for (const CircularBufferAllocator* a : touched) {
                    addr = std::max<uint64_t>(addr, a->get_cb_region_end());
                }
                addr = align(addr, alignment);
                for (CircularBufferAllocator* a : touched) {
                    a->mark_address(addr, handle.size_bytes, base_l1_address);
                }

                handle.allocated_address = static_cast<uint32_t>(addr);

                // Patch the allocated address into the kernel's CRTA buffer. This runs at Program-compile
                // time, upstream of where dispatch delivers runtime args to the device:
                //  - FD: the fast/mesh path snapshots the CRTA buffer into the command stream
                //  - SD: the slow-dispatch path writes it via WriteRuntimeArgsToDevice
                //
                // An implicit CRTA slot to hold the scratchpad address is reserved at Program creation.
                // (The actual CRTA buffer itself is allocated when SetProgramRunArgs runs.)
                // Now, we populate the scratchpad address.
                //
                TT_FATAL(
                    !kernel->common_runtime_args().empty(),
                    "CRTA buffer is not allocated; cannot populate scratchpad addresses for kernel {}. "
                    "Ensure that SetProgramRunArgs is called before attempting to enqueue a Program.",
                    kernel->name());
                TT_FATAL(
                    handle.allocated_address != 0,
                    "Internal error: scratchpad '{}' on kernel '{}' "
                    "has a 0 allocated address (allocation failed or was skipped).",
                    handle.accessor_name,
                    kernel->name());

                RuntimeArgsData& crta = kernel->common_runtime_args_data();
                crta.data()[handle.addr_crta_word] = handle.allocated_address;
            }
        }
    }

    this->scratchpads_allocated_ = true;
}

void detail::ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    TTZoneScopedD(PROGRAM);

    // If device is a MeshDevice, we need to track all its sub-devices
    std::vector<const IDevice*> devices_to_track;
    const tt::tt_metal::distributed::MeshDevice* mesh_device =
        dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device);
    if (mesh_device != nullptr) {
        // Mesh device: track all sub-devices
        for (IDevice* sub_device : mesh_device->get_devices()) {
            devices_to_track.push_back(sub_device);
        }
    } else {
        // Single device
        devices_to_track.push_back(device);
    }

    // Track which devices are NEW (not already tracked)
    std::vector<const IDevice*> new_devices;
    for (const IDevice* dev : devices_to_track) {
        auto [iter, inserted] = this->cb_devices_.insert(dev);
        if (inserted) {
            new_devices.push_back(dev);
        }
    }

    // If CB layout already calculated, skip allocation but report for new devices
    if (not this->local_circular_buffer_allocation_needed_) {
        // Report CB allocations for any NEW devices (using cached addresses)
        if (!new_devices.empty() && !this->circular_buffers_.empty()) {
            for (const IDevice* dev : new_devices) {
                for (const auto& circular_buffer : this->circular_buffers_) {
                    if (!circular_buffer->globally_allocated()) {
                        tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                            circular_buffer->core_ranges(),
                            circular_buffer->address(),
                            circular_buffer->size(),
                            circular_buffer->globally_allocated(),
                            dev);
                    }
                }

                // Also register program with the NEW device
                auto* device_obj = dynamic_cast<Device*>(const_cast<IDevice*>(dev));
                if (device_obj) {
                    device_obj->register_program(this);
                    if (device_obj->get_shm_stats_provider()) {
                        device_obj->get_shm_stats_provider()->update_from_allocator(device_obj, getpid());
                    }
                }
            }
        }
        return;
    }

    uint64_t base_cb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (const auto& circular_buffer : this->circular_buffers_) {
        if (circular_buffer->globally_allocated()) {
            // Track globally allocated CBs too (they use L1 memory allocated via the allocator)
            for (const IDevice* dev : devices_to_track) {
                tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                    circular_buffer->core_ranges(),
                    circular_buffer->address(),
                    circular_buffer->size(),
                    circular_buffer->globally_allocated(),
                    dev);
            }
            continue;
        }

        uint64_t computed_addr = base_cb_address;
        for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
            // Need the max available address across all cores circular buffer is placed on
            for (const CircularBufferAllocator& cb_allocator : this->cb_allocators_) {
                if (cb_allocator.core_range == core_range) {
                    computed_addr = std::max(computed_addr, cb_allocator.get_cb_region_end());
                    break;
                }
            }
        }
        computed_addr = align(computed_addr, device->allocator()->get_alignment(BufferType::DRAM));
        for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
            for (CircularBufferAllocator& cb_allocator : this->cb_allocators_) {
                if (cb_allocator.core_range.intersects(core_range)) {
                    if (cb_allocator.core_range != core_range and computed_addr < cb_allocator.get_cb_region_end()) {
                        // Intersecting core range has already been marked to have allocation at this address. This
                        // could have been marked by a circular buffer on a core range disjoint from current
                        // `core_range` but also intersecting `cb_allocator.core_range`
                        continue;
                    }
                    cb_allocator.mark_address(computed_addr, circular_buffer->size(), base_cb_address);
                }
            }
        }
        // Report CB allocation for ALL devices being tracked
        for (const IDevice* dev : devices_to_track) {
            tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                circular_buffer->core_ranges(),
                computed_addr,
                circular_buffer->size(),
                circular_buffer->globally_allocated(),
                dev);
        }
        circular_buffer->set_locally_allocated_address(computed_addr);
    }

    // Register program ONLY with NEW devices (prevents duplicate registration)
    for (const IDevice* dev : new_devices) {
        auto* device_obj = dynamic_cast<Device*>(const_cast<IDevice*>(dev));
        if (device_obj) {
            device_obj->register_program(this);
            // Update locally-allocated CB stats via query (accurate even for cached programs)
            if (device_obj->get_shm_stats_provider()) {
                device_obj->get_shm_stats_provider()->update_from_allocator(device_obj, getpid());
            }
        }
    }
    this->local_circular_buffer_allocation_needed_ = false;
}

std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> detail::ProgramImpl::get_cb_l1_regions_per_core(
    int device_id, size_t num_devices) const {
    (void)device_id;    // TODO: Use device_id once per-device or heterogeneous mesh CB layouts are supported
    (void)num_devices;  // TODO: Use num_devices for multi-device filtering or layout partitioning when implemented

    std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> regions_per_core;

    // For each allocator, iterate through all cores in its CoreRange
    for (const auto& cb_allocator : cb_allocators_) {
        const auto& l1_regions = cb_allocator.l1_regions;

        // Add these regions to every core in the CoreRange
        for (uint32_t x = cb_allocator.core_range.start_coord.x; x <= cb_allocator.core_range.end_coord.x; x++) {
            for (uint32_t y = cb_allocator.core_range.start_coord.y; y <= cb_allocator.core_range.end_coord.y; y++) {
                CoreCoord core(x, y);
                auto& core_regions = regions_per_core[core];
                core_regions.insert(core_regions.end(), l1_regions.begin(), l1_regions.end());
            }
        }
    }

    return regions_per_core;
}

void detail::ProgramImpl::deallocate_circular_buffers() {
    // Deallocate all circular buffers for this program on ALL devices
    // This notifies the GraphTracker to report deallocations
    if (!this->cb_devices_.empty() && !this->circular_buffers_.empty()) {
        for (const IDevice* idevice : this->cb_devices_) {
            tt::tt_metal::GraphTracker::instance().track_deallocate_cb(idevice);
        }

        // Unregister program from ALL devices (matches registration)
        for (const IDevice* idevice : this->cb_devices_) {
            auto* device = dynamic_cast<Device*>(const_cast<IDevice*>(idevice));
            if (device) {
                device->unregister_program(this);
                // Update locally-allocated CB stats via query (accurate after deallocation)
                if (device->get_shm_stats_provider()) {
                    device->get_shm_stats_provider()->update_from_allocator(device, getpid());
                }
            }
        }

        this->cb_devices_.clear();  // Clear device set after deallocation
    }
}

void detail::ProgramImpl::validate_circular_buffer_region(const IDevice* device) {
    TTZoneScopedD(PROGRAM);

    // TODO: Circular buffer allocation and validation could be better optimized by determining usage per sub-device
    std::optional<DeviceAddr> lowest_address =
        device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();
    const auto& allocator = device->allocator_impl();
    const bool hybrid_mode = allocator->get_config().allocator_mode == AllocatorMode::HYBRID;

    // In HYBRID mode, per-core allocations live on each per-device allocator (not the mesh
    // allocator). Collect the physical allocators we must consult so we can see per-core state.
    std::vector<AllocatorImpl*> physical_allocators;
    if (hybrid_mode) {
        if (const auto* mesh = dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device)) {
            for (IDevice* dev : mesh->get_devices()) {
                physical_allocators.push_back(dev->allocator_impl().get());
            }
        } else {
            physical_allocators.push_back(allocator.get());
        }
    }

    // Flatten MeshDevice into constituent physical devices so ServiceCoreManager (keyed by ChipId) can be queried per
    // core
    const auto& svc = tt::tt_metal::MetalContext::instance().get_service_core_manager().impl();
    std::vector<const IDevice*> devices_for_svc_check;
    if (svc.has_any_claims()) {
        if (const auto* mesh = dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device)) {
            for (IDevice* dev : mesh->get_devices()) {
                devices_for_svc_check.push_back(dev);
            }
        } else {
            devices_for_svc_check.push_back(device);
        }
    }

    for (const CircularBufferAllocator& cb_allocator : this->cb_allocators_) {
        if (cb_allocator.l1_regions.empty()) {
            continue;
        }
        uint64_t cb_region_end = cb_allocator.l1_regions.back().second;  // cb_allocator.get_cb_region_end();
        if (cb_region_end > max_l1_size) {
            TT_THROW(
                "Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} "
                "B",
                cb_allocator.core_range.str(),
                cb_region_end,
                max_l1_size);
        }

        // Service cores allocate L1 independently per core (not lock-step like workers), growing down
        // from L1_END. CBs grow up from DEFAULT_UNRESERVED. Min frontier across the CB range catches
        // the most constrained core - collision if any frontier sits below the CB region end
        const bool on_service_core =
            std::any_of(devices_for_svc_check.begin(), devices_for_svc_check.end(), [&](const IDevice* dev) {
                return svc.is_service_core(dev->id(), cb_allocator.core_range.start_coord);
            });

        if (on_service_core) {
            std::optional<DeviceAddr> svc_lowest;
            for (const IDevice* dev : devices_for_svc_check) {
                for (const auto& core : cb_allocator.core_range) {
                    auto a = svc.lowest_allocated_address(dev->id(), core);
                    if (a.has_value()) {
                        svc_lowest = svc_lowest.has_value() ? std::make_optional(std::min(*svc_lowest, *a)) : a;
                    }
                }
            }
            if (svc_lowest.has_value() && svc_lowest.value() < cb_region_end) {
                TT_THROW(
                    "Circular buffers on service-core range {} in program {} clash with ServiceCoreManager-allocated "
                    "L1 (lowest service allocation at {}, CB region ends at {})",
                    cb_allocator.core_range.str(),
                    this->id,
                    svc_lowest.value(),
                    cb_region_end);
            }
            continue;  // Worker-grid checks below are irrelevant for service cores.
        }

        if (hybrid_mode) {
            // Per-core allocations (experimental_set_per_core_allocation) can land at different
            // addresses per core, so query only the banks this CB covers on each physical allocator.
            // Prevents per-core tensors on unrelated cores from spuriously tightening this CB's
            // budget, and lets us see per-core state that lives on per-device (not mesh) allocators.
            lowest_address = std::nullopt;
            for (const auto& core : cb_allocator.core_range) {
                for (auto* phys_alloc : physical_allocators) {
                    auto bank_id = phys_alloc->get_bank_ids_from_logical_core(BufferType::L1, core).front();
                    auto addr = phys_alloc->get_lowest_occupied_l1_address(bank_id);
                    if (addr.has_value()) {
                        lowest_address =
                            lowest_address.has_value() ? std::make_optional(std::min(*lowest_address, *addr)) : addr;
                    }
                }
            }
        }
        if (lowest_address.has_value() and lowest_address.value() < cb_region_end) {
            TT_THROW(
                "Statically allocated circular buffers in program {} clash with L1 buffers on core range {}. L1 buffer "
                "allocated at {} and static circular buffer region ends at {}",
                this->id,
                cb_allocator.core_range.str(),
                lowest_address.value(),
                cb_region_end);
        }
    }
}

void detail::ProgramImpl::validate_circular_buffer_core_ranges(const IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    // Flatten MeshDevice into constituent physical devices so ServiceCoreManager (keyed by ChipId) can be queried per
    // core. Mirrors validate_circular_buffer_region.
    const auto& svc = tt::tt_metal::MetalContext::instance().get_service_core_manager().impl();
    std::unordered_set<CoreCoord> claimed;
    if (svc.has_any_claims()) {
        if (const auto* mesh = dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device)) {
            for (IDevice* dev : mesh->get_devices()) {
                auto chip_claimed = svc.claimed_cores(dev->id());
                claimed.insert(chip_claimed.begin(), chip_claimed.end());
            }
        } else {
            claimed = svc.claimed_cores(device->id());
        }
    }
    auto entirely_on_service_cores = [&](const CoreRange& cr) {
        if (claimed.empty()) {
            return false;
        }
        for (uint32_t x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
            for (uint32_t y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                if (!claimed.contains(CoreCoord{x, y})) {
                    return false;
                }
            }
        }
        return true;
    };
    for (const auto& cb : circular_buffers_) {
        for (const auto& cr : cb->core_ranges().ranges()) {
            const bool in_worker_grid = cr.end_coord.x < grid_size.x && cr.end_coord.y < grid_size.y;
            TT_FATAL(
                in_worker_grid || entirely_on_service_cores(cr),
                "Circular buffer core range {} in program {} exceeds device compute grid ({}x{}) and is "
                "not entirely on cores claimed via ServiceCoreManager",
                cr.str(),
                this->id,
                grid_size.x,
                grid_size.y);
        }
    }
}

void detail::ProgramImpl::init_semaphores(
    const IDevice& device, const CoreCoord& logical_core, uint32_t programmable_core_type_index) const {
    const auto& hal = MetalContext::instance().hal();
    HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(programmable_core_type_index);
    uint64_t kernel_config_base = hal.get_dev_noc_addr(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);
    uint64_t addr = kernel_config_base + this->program_configs_[programmable_core_type_index].sem_offset;
    CoreType core_type = MetalContext::instance().hal().get_core_type(programmable_core_type_index);
    auto semaphores_on_core = this->semaphores_on_core(logical_core, core_type);
    for (auto semaphore : semaphores_on_core) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            device.id(),
            device.virtual_core_from_logical_core(logical_core, core_type),
            std::vector{semaphore.get().initial_value()},
            addr + semaphore.get().offset());
    }
}

void detail::ProgramImpl::validate_semaphore_id(
    const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const {
    TT_FATAL(semaphore_id < NUM_SEMAPHORES, "Semaphore id {} exceeds max value {}", semaphore_id, NUM_SEMAPHORES - 1);

    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing_semaphores = semaphores_on_core(logical_core, core_type);
                for (const auto& semaphore : existing_semaphores) {
                    TT_FATAL(
                        semaphore.get().id() != semaphore_id,
                        "Semaphore id {} already in use on core {}",
                        semaphore_id,
                        logical_core.str());
                }
            }
        }
    }
}

void detail::ProgramImpl::add_semaphore(
    const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    TT_FATAL(this->compiled_.empty(), "Cannot add semaphore to an already compiled program {}", this->id);
    validate_semaphore_id(crs, semaphore_id, core_type);
    semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, core_type));
}

uint32_t detail::ProgramImpl::create_semaphore(const CoreRangeSet& crs, uint32_t initial_value, CoreType core_type) {
    TT_FATAL(!crs.ranges().empty(), "Expecting a non-empty CoreRangeSet!");
    TT_FATAL(
        MetalContext::instance().is_coord_in_range(crs.ranges().back().end_coord, core_type),
        "Coordinates out of range");

    std::optional<uint32_t> semaphore_id;
    std::bitset<NUM_SEMAPHORES> used_semaphore_ids;
    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing = this->semaphores_on_core(logical_core, core_type);
                if (existing.size() == NUM_SEMAPHORES) {
                    TT_THROW(
                        "Cannot add semaphore on core {}. Max number of semaphores ({}) reached!",
                        logical_core.str(),
                        NUM_SEMAPHORES);
                }
                for (const auto& semaphore : existing) {
                    used_semaphore_ids.set(semaphore.get().id());
                }
            }
        }
    }
    for (uint32_t sem_id = 0; sem_id < NUM_SEMAPHORES; sem_id++) {
        if (!used_semaphore_ids.test(sem_id)) {
            semaphore_id = sem_id;
            break;
        }
    }
    TT_FATAL(
        semaphore_id.has_value(),
        "Unable to initialize semaphore on CoreRangeSet {}: all {} IDs are in use",
        crs.str(),
        NUM_SEMAPHORES);

    this->add_semaphore(crs, *semaphore_id, initial_value, core_type);
    return *semaphore_id;
}

std::vector<std::vector<CoreCoord>> detail::ProgramImpl::logical_cores() const {
    std::vector<std::vector<CoreCoord>> cores_in_program;
    std::vector<std::set<CoreCoord>> unique_cores;
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < kernels_.size();
         programmable_core_type_index++) {
        const auto& kernels = this->kernels_[programmable_core_type_index];
        cores_in_program.push_back({});
        unique_cores.push_back({});
        for (const auto& [id, kernel] : kernels) {
            for (auto core : kernel->logical_cores()) {
                if (unique_cores[programmable_core_type_index].contains(core)) {
                    continue;
                }
                unique_cores[programmable_core_type_index].insert(core);
                cores_in_program[programmable_core_type_index].push_back(core);
            }
        }
    }
    return cores_in_program;
}

void detail::ProgramImpl::set_remote_circular_buffer_init(const std::shared_ptr<Kernel>& kernel) const {
    const auto& kernel_defines = kernel->defines();
    const std::string reserved_defines[] = {"ALIGN_LOCAL_CBS_TO_REMOTE_CBS"};
    for (const auto& str : reserved_defines) {
        TT_FATAL(!kernel_defines.contains(str), "{} is a reserved define and can't be manually set", str);
    }
    std::string align_code;
    std::unordered_set<CBHandle> initialized_cbs;
    std::unordered_set<uint8_t> remote_cb_indices;
    for (auto logical_cr : kernel->logical_coreranges()) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            if (circular_buffer->remote_buffer_indices().empty() || initialized_cbs.contains(circular_buffer->id())) {
                continue;
            }
            initialized_cbs.insert(circular_buffer->id());
            auto remote_cb_index = *circular_buffer->remote_buffer_indices().begin();
            remote_cb_indices.insert(remote_cb_index);

            // We only need the first remote buffer index
            if (!circular_buffer->local_buffer_indices().empty()) {
                align_code += fmt::format(
                    "experimental::align_local_cbs_to_remote_cb<{}>({},{{",
                    circular_buffer->local_buffer_indices().size(),
                    remote_cb_index);
                for (auto buffer_index : circular_buffer->local_buffer_indices()) {
                    align_code += fmt::format("{},", buffer_index);
                }
                align_code.back() = '}';
                align_code.append(");");
            }
        }
    }
    if (!remote_cb_indices.empty()) {
        std::map<std::string, std::string> defines;
        if (!align_code.empty()) {
            defines["ALIGN_LOCAL_CBS_TO_REMOTE_CBS"] = align_code;
        }
        kernel->add_defines(defines);
    }
}

void detail::ProgramImpl::set_cb_data_fmt_and_tile(
    const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const {
    TTZoneScopedD(PROGRAM);
    for (const auto& logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                const CBIndex cb_index = static_cast<CBIndex>(buffer_index);
                const DataFormat data_format = circular_buffer->data_format(buffer_index);
                const auto& tile_opt = circular_buffer->tile(buffer_index);
                const auto& unpack_geom = circular_buffer->unpack_face_geometry(buffer_index);
                build_options.set_cb_data_fmt_tile_and_face_geometry(cb_index, data_format, tile_opt, unpack_geom);
            }
        }
    }
}

void detail::ProgramImpl::populate_dispatch_data(IDevice* device) {
    // Mock/emulated devices don't dispatch to hardware, skip dispatch data population
    if (tt::tt_metal::MetalContext::instance(extract_context_id(device)).get_cluster().is_mock_or_emulated()) {
        return;
    }

    auto extract_dst_noc_unicast_info =
        [&device](
            const auto& ranges, const CoreType core_type) -> std::vector<std::pair<transfer_info_cores, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info;
        for (const CoreRange& core_range : ranges) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord virtual_coord = device->virtual_core_from_logical_core(CoreCoord({x, y}), core_type);
                    dst_noc_unicast_info.push_back(std::make_pair(virtual_coord, /*num_mcast_dests=*/0));
                }
            }
        }
        return dst_noc_unicast_info;
    };

    // Circular Buffer Configs handled in progrm_dispatch::update_program_dispatch_commands

    // Assume here and in command queue that kg_buffers is populated with multicast buffers first then unicast buffers
    // Program Binaries and Go Signals
    // TODO: cleanup put the WORKERS and ETH logic together..

    // All program binaries will be packed into a single buffer in memory
    std::vector<uint32_t> binaries_data;
    // Map is used to look up transfer info by kernel id when we populate data ordered by core groups
    std::unordered_map<KernelHandle, kernel_bins_transfer_info> kernel_transfer_info;
    // This is generic for workers and eth cores
    for (const auto& kernels : this->kernels_) {
        for (const auto& [kernel_id, kernel] : kernels) {
            const auto& binaries = kernel->binaries(BuildEnvManager::get_instance(extract_context_id(device))
                                                        .get_device_build_env(device->build_id())
                                                        .build_key());
            std::vector<uint32_t> dst_base_addrs;
            std::vector<uint32_t> page_offsets;
            std::vector<uint32_t> lengths;
            std::vector<uint32_t> processor_ids;
            uint32_t transfer_info_index = 0;

            for (size_t sub_kernel_index = 0; sub_kernel_index < binaries.size(); ++sub_kernel_index) {
                const ll_api::memory& kernel_bin = *binaries[sub_kernel_index];

                // TODO: Pack erisc spans too, and then everything is
                // one span
                uint32_t num_spans = kernel_bin.num_spans();
                dst_base_addrs.resize(dst_base_addrs.size() + num_spans);
                page_offsets.resize(page_offsets.size() + num_spans);
                lengths.resize(lengths.size() + num_spans);
                processor_ids.resize(processor_ids.size() + num_spans);

                kernel_bin.process_spans(
                    [&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                        // Set dst for eth kernels until they move to ring buffer
                        dst_base_addrs[transfer_info_index] = dst;
                        page_offsets[transfer_info_index] =
                            binaries_data.size() * sizeof(uint32_t) / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                        lengths[transfer_info_index] = len * sizeof(uint32_t);
                        processor_ids[transfer_info_index] = kernel->get_kernel_processor_type(sub_kernel_index);

                        binaries_data.insert(binaries_data.end(), mem_ptr, mem_ptr + len);
                        binaries_data.resize(
                            tt::align(binaries_data.size(), HostMemDeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t)),
                            0);
                        transfer_info_index++;
                    });
            }

            kernel_transfer_info.emplace(
                kernel_id,
                kernel_bins_transfer_info{
                    .core_type = kernel->get_kernel_programmable_core_type(),
                    .processor_class = kernel->get_kernel_processor_class(),
                    .dst_base_addrs = std::move(dst_base_addrs),
                    .page_offsets = std::move(page_offsets),
                    .lengths = std::move(lengths),
                    .processor_ids = std::move(processor_ids),
                });
        }
    }

    if (!binaries_data.empty()) {
        this->program_transfer_info.binary_data = binaries_data;
    }

    std::uint32_t num_active_cores = 0;
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        for (const auto& kernel_group : this->get_kernel_groups(index)) {
            if (hal.get_supports_receiving_multicasts(index)) {
                // Below assumes core has a kernel config buffer
                std::vector<multicast_transfer_info> dst_noc_multicast_info =
                    extract_dst_noc_multicast_info(device, kernel_group->core_ranges.ranges(), core_type);
                std::vector<KernelHandle> kernel_ids;
                for (auto kernel_id : kernel_group->kernel_ids) {
                    KernelHandle device_local_kernel_id = program_dispatch::get_device_local_kernel_handle(kernel_id);
                    kernel_ids.push_back(device_local_kernel_id);
                    auto kernel = this->get_kernel(device_local_kernel_id);
                    auto processor_class = kernel->get_kernel_processor_class();
                    auto& transfer_info = kernel_transfer_info.at(device_local_kernel_id);
                    for (uint32_t span_idx = 0; span_idx < transfer_info.dst_base_addrs.size(); span_idx++) {
                        auto processor_type = transfer_info.processor_ids[span_idx];
                        auto processor_index = hal.get_processor_index(
                            hal.get_programmable_core_type(index), processor_class, processor_type);
                        transfer_info.dst_base_addrs[span_idx] = kernel_group->kernel_text_offsets[processor_index];
                    }
                }

                for (const auto& transfer_info : dst_noc_multicast_info) {
                    for (const auto& kernel_id : kernel_ids) {
                        this->program_transfer_info.kernel_bins.emplace_back(
                            transfer_info.cores, transfer_info.num_dests, kernel_transfer_info.at(kernel_id));
                    }
                }
            } else {
                // Below assumes ethernet dispatch class
                TT_ASSERT(core_type == CoreType::ETH);
                std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                    extract_dst_noc_unicast_info(kernel_group->core_ranges.ranges(), core_type);

                // No checks for max dispatch class
                // Validated during CreateKernel if the requested processor is supported
                std::vector<KernelHandle> kernel_ids;
                for (auto kernel_id : kernel_group->kernel_ids) {
                    KernelHandle device_local_kernel_id = program_dispatch::get_device_local_kernel_handle(kernel_id);
                    auto kernel = this->get_kernel(device_local_kernel_id);
                    kernel_ids.push_back(device_local_kernel_id);

                    // Update destination address by kernel config offset
                    if (hal.get_core_kernel_stored_in_config_buffer(hal.get_programmable_core_type(index))) {
                        auto processor_class = kernel->get_kernel_processor_class();
                        auto& transfer_info = kernel_transfer_info.at(device_local_kernel_id);
                        for (uint32_t span_idx = 0; span_idx < transfer_info.dst_base_addrs.size(); span_idx++) {
                            auto processor_type = transfer_info.processor_ids[span_idx];
                            auto processor_index = hal.get_processor_index(
                                hal.get_programmable_core_type(index), processor_class, processor_type);
                            transfer_info.dst_base_addrs[span_idx] = kernel_group->kernel_text_offsets[processor_index];
                        }
                    }
                }

                for (const auto& [cores, num_mcast_dsts] : dst_noc_unicast_info) {
                    for (const auto& kernel_id : kernel_ids) {
                        this->program_transfer_info.kernel_bins.emplace_back(
                            cores, num_mcast_dsts, kernel_transfer_info.at(kernel_id));
                    }
                }
            }
        }
        num_active_cores += this->logical_cores()[index].size();
    }

    this->program_transfer_info.num_active_cores = num_active_cores;
}

ProgramConfig& detail::ProgramImpl::get_program_config(uint32_t programmable_core_type_index) {
    return this->program_configs_[programmable_core_type_index];
}

const ProgramConfig& detail::ProgramImpl::get_program_config(uint32_t programmable_core_type_index) const {
    return this->program_configs_[programmable_core_type_index];
}

void detail::ProgramImpl::set_launch_msg_sem_offsets() {
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t kg_type_index = 0; kg_type_index < hal.get_programmable_core_type_count(); kg_type_index++) {
        for (auto& kg : this->get_kernel_groups(kg_type_index)) {
            auto sem_offset = kg->launch_msg.view().kernel_config().sem_offset();
            for (uint32_t sem_type_index = 0; sem_type_index < hal.get_programmable_core_type_count();
                 sem_type_index++) {
                sem_offset[sem_type_index] = this->program_configs_[sem_type_index].sem_offset;
            }
        }
    }
}

uint32_t& detail::ProgramImpl::get_program_config_size(uint32_t programmable_core_type_index) {
    return this->program_config_sizes_[programmable_core_type_index];
}

const std::vector<SubDeviceId>& detail::ProgramImpl::determine_sub_device_ids(const IDevice* device) {
    // We need to calculate the sub_device_id when we haven't compiled the program yet, or this is the first time we
    // are getting the sub_device_ids after compilation
    auto sub_device_manager_id = device->get_active_sub_device_manager_id();
    auto& sub_device_ids_map = this->sub_device_ids_[device->id()];
    auto sub_device_ids = sub_device_ids_map.find(sub_device_manager_id);
    if (this->compiled_.empty() || sub_device_ids == sub_device_ids_map.end()) {
        if (!MetalContext::instance().rtoptions().get_fast_dispatch() ||
            sub_device_manager_id == device->get_default_sub_device_manager_id()) {
            // No sub device manager, nothing to validate
            auto [sub_device_ids, _] =
                sub_device_ids_map.insert_or_assign(sub_device_manager_id, std::vector<SubDeviceId>{SubDeviceId{0}});
            return sub_device_ids->second;
        }
        std::unordered_set<SubDeviceId> used_sub_device_ids;
        auto find_sub_device_ids = [&](HalProgrammableCoreType core_type) {
            auto core_type_index = MetalContext::instance().hal().get_programmable_core_type_index(core_type);
            if (core_type_index == -1) {
                return;
            }
            const auto& program_kgs =
                this->get_kernel_groups(MetalContext::instance().hal().get_programmable_core_type_index(core_type));
            uint32_t num_intersections = 0;
            uint32_t num_cores = 0;
            for (const auto& kg : program_kgs) {
                for (size_t i = 0; i < device->num_sub_devices(); ++i) {
                    const auto& sub_device_cores =
                        device->worker_cores(core_type, SubDeviceId{static_cast<unsigned char>(i)});
                    auto intersection = sub_device_cores.intersection(kg->core_ranges);
                    if (!intersection.empty()) {
                        used_sub_device_ids.insert(SubDeviceId{static_cast<unsigned char>(i)});
                        num_intersections += intersection.num_cores();
                    }
                }
                num_cores += kg->core_ranges.num_cores();
            }
            TT_FATAL(
                num_intersections == num_cores,
                "Kernel group cores do not match sub device cores for programmable core type {}",
                enchantum::to_string(core_type));
        };
        find_sub_device_ids(HalProgrammableCoreType::TENSIX);
        find_sub_device_ids(HalProgrammableCoreType::ACTIVE_ETH);
        auto [sub_device_ids, _] = sub_device_ids_map.insert_or_assign(
            sub_device_manager_id, std::vector<SubDeviceId>(used_sub_device_ids.begin(), used_sub_device_ids.end()));
        return sub_device_ids->second;
    }
    return sub_device_ids->second;
}

void detail::ProgramImpl::allocate_kernel_bin_buf_on_device(IDevice* device) {
    // Allocate the DRAM kernel binary buffer for this program on the specified device, if not previously allocated.
    // We allocate program binaries top down to minimize fragmentation with other buffers in DRAM, which are typically
    // allocated bottom up
    std::size_t binary_data_size_bytes = this->program_transfer_info.binary_data.size() * sizeof(uint32_t);
    if (!this->kernels_buffer_.contains(device->id()) and binary_data_size_bytes) {
        std::shared_ptr<Buffer> kernel_bin_buf = Buffer::create(
            device,
            binary_data_size_bytes,
            HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
            BufferType::DRAM,
            std::nullopt,
            false);
        this->kernels_buffer_[device->id()] = kernel_bin_buf;
    }
}

void ProgramImpl::generate_dispatch_commands(distributed::MeshDevice* mesh_device, bool use_prefetcher_cache) {
    uint64_t command_hash = *mesh_device->get_active_sub_device_manager_id();

    uint64_t device_hash =
        BuildEnvManager::get_instance(extract_context_id(mesh_device))
            .get_device_build_env(mesh_device->build_id())
            .build_key();
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        ttsl::hash::hash_combine(device_hash, mesh_device->id());
    }
    if (!is_cached()) {
        set_cached(device_hash);
    } else {
        TT_FATAL(
            *get_cached() == device_hash,
            "Enqueueing a Program across devices with different cores harvested is not supported, unless coordinate "
            "virtualization is enabled (only enabled on Wormhole and above).");
    }
    auto& cached_program_command_sequences = this->get_cached_program_command_sequences();
    if (!cached_program_command_sequences.contains(command_hash)) {
        // Programs currently only support spanning a single sub-device
        auto sub_device_id = this->determine_sub_device_ids(mesh_device).at(0);
        ProgramCommandSequence program_command_sequence;
        program_dispatch::insert_empty_program_dispatch_preamble_cmd(program_command_sequence);
        program_dispatch::insert_stall_cmds(program_command_sequence, sub_device_id);
        program_dispatch::assemble_device_commands(
            program_command_sequence, *this, mesh_device, sub_device_id, use_prefetcher_cache);

        program_command_sequence.kernel_bins_sizeB = this->kernel_bins_sizeB;
        program_command_sequence.prefetcher_cache_used = use_prefetcher_cache;

        cached_program_command_sequences.insert({command_hash, std::move(program_command_sequence)});
    } else {
        TT_ASSERT(
            cached_program_command_sequences.at(command_hash).prefetcher_cache_used == use_prefetcher_cache,
            "Prefetcher cache used mismatch for program {} on device {}",
            this->get_id(),
            mesh_device->id());
    }
}

void ProgramImpl::generate_trace_dispatch_commands(distributed::MeshDevice* mesh_device, bool use_prefetcher_cache) {
    uint64_t command_hash = *mesh_device->get_active_sub_device_manager_id();

    uint64_t device_hash =
        BuildEnvManager::get_instance(extract_context_id(mesh_device))
            .get_device_build_env(mesh_device->build_id())
            .build_key();
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        device_hash = (device_hash << 32) | (mesh_device->id());
    }
    if (!is_cached()) {
        set_cached(device_hash);
    } else {
        TT_FATAL(
            *get_cached() == device_hash,
            "Enqueueing a Program across devices with different cores harvested is not supported, unless coordinate "
            "virtualization is enabled (only enabled on Wormhole and above).");
    }
    auto& trace_cached_program_command_sequences = get_trace_cached_program_command_sequences();
    if (!trace_cached_program_command_sequences.contains(command_hash)) {
        // Programs currently only support spanning a single sub-device
        auto sub_device_id = this->determine_sub_device_ids(mesh_device).at(0);
        ProgramCommandSequence program_command_sequence;
        program_dispatch::insert_empty_program_dispatch_preamble_cmd(program_command_sequence);
        program_dispatch::insert_stall_cmds(program_command_sequence, sub_device_id);
        program_dispatch::assemble_device_commands(
            program_command_sequence, *this, mesh_device, sub_device_id, use_prefetcher_cache);
        program_command_sequence.prefetcher_cache_used = use_prefetcher_cache;
        program_command_sequence.kernel_bins_sizeB = this->kernel_bins_sizeB;
        // TODO: We currently do not have a mechanism of removing entries in the cache when a manager is removed
        // This means programs will contain stale entries in the cache until the program is deleted
        trace_cached_program_command_sequences.insert({command_hash, std::move(program_command_sequence)});
    } else {
        TT_ASSERT(
            trace_cached_program_command_sequences.at(command_hash).prefetcher_cache_used == use_prefetcher_cache,
            "Prefetcher cache used mismatch for program {} on device {}",
            this->get_id(),
            mesh_device->id());
    }
}

void detail::ProgramImpl::compile(IDevice* device, bool force_slow_dispatch) {
    TTZoneScopedD(PROGRAM);
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id());

    if (compiled_.contains(build_env.build_key())) {
        Inspector::program_compile_already_exists(this, device, build_env.build_key());
        return;
    }
    // Clear the determined sub_device_ids when we compile the program for the first time
    // This way, determine_sub_device_ids is forced to recalculate with the finalized information on the used cores
    if (compiled_.empty()) {
        this->sub_device_ids_[device->id()].erase(device->get_active_sub_device_manager_id());
    }

    Inspector::program_compile_started(this, device, build_env.build_key());

    TT_FATAL(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is "
        "dependent on information that is set during device initialization.",
        this->get_id());

    bool remote_enabled = jit_server::JitCompileRpcClient::enabled();
    std::vector<std::shared_future<void>> events;

    auto prep_kernel = [&](const std::shared_ptr<Kernel>& kernel) {
        JitBuildOptions build_options(build_env.build_env);
        kernel->set_build_options(build_options);
        if (this->compiled_.empty()) {
            this->set_remote_circular_buffer_init(kernel);
        }
        this->set_cb_data_fmt_and_tile(kernel->logical_coreranges(), build_options);
        this->set_dfb_data_fmt_and_tile(kernel->logical_coreranges(), build_options);

        // Blackhole-only: Fp8_e4m3 / Lf8 dataformats require fp32_dest_acc_en=true in the associated compute
        // kernel. This is due to FP8/LF8 being considered "A" exp width formats, instead of "B" exp width
        // formats that are supported mostly in tt-metal. This conservative check fires whenever a compute
        // kernel shares a core with any FP8 CB — the old Program API has no way to know which CB
        // a given kernel actually reads, so we err on the side of catching the misconfiguration.
        if (build_options.build_env.get_arch() == tt::ARCH::BLACKHOLE &&
            kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE &&
            std::any_of(
                build_options.hlk_desc.buf_dataformat_arr.begin(),
                build_options.hlk_desc.buf_dataformat_arr.end(),
                is_fp8_format)) {
            TT_FATAL(
                build_options.fp32_dest_acc_en,
                "Blackhole: Fp8_e4m3 / Lf8 require fp32_dest_acc_en=true in ComputeConfig. The DEST "
                "register must be in 32-bit (family-agnostic) mode when any CB on the same core uses "
                "an 8-bit float format. Kernel: {}",
                kernel->name());
        }

        auto kernel_hash = detail::KernelCompileHash(kernel, build_options, build_env.build_key());

        const std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash) + "/";
        kernel->set_full_name(kernel_path_suffix);
        build_options.set_name(kernel_path_suffix);

        return std::pair{std::move(build_options), kernel_hash};
    };

    if (remote_enabled) {
        // Remote path: prep and submit are sequential.  Parallelism is on compilation which happens on the remote
        // server.
        // TODO: precompiled kernel is not supported in remote mode
        auto endpoints = jit_server::JitCompileRpcClient::endpoints_from_env();
        TT_FATAL(
            !endpoints.empty(),
            "TT_METAL_JIT_SERVER_ENABLE is set but no compile-server endpoints are configured. "
            "Set TT_METAL_JIT_SERVER_ENDPOINTS or TT_METAL_JIT_SERVER_ENDPOINT.");
        RemoteCompileCoordinator coordinator(
            std::move(endpoints), extract_context_id(device), device->build_id(), build_env.build_key());

        std::vector<std::pair<std::shared_ptr<Kernel>, JitBuildOptions>> submitted_kernels;

        for (auto& kernels : kernels_) {
            for (auto& [id, kernel] : kernels) {
                validate_kernel_placement(force_slow_dispatch, kernel, device->id());
                auto [build_options, kernel_hash] = prep_kernel(kernel);
                // Skip the remote round-trip when the ELF is already validly cached locally.
                if (!remote_kernel_cached(device, kernel)) {
                    coordinator.submit(kernel_hash, [&]() {
                        generate_kernel_source_files(device, build_options, kernel);
                        return build_kernel_descriptor(device, kernel, build_options, kernel_hash);
                    });
                }
                // Always recorded: cached kernels still need read_binaries() to load the on-disk ELF.
                submitted_kernels.emplace_back(kernel, std::move(build_options));
            }
        }

        bool remote_ok = true;
        try {
            coordinator.finish();
        } catch (const jit_server::RemoteCompileTransportError& e) {
            // The compile server became unavailable mid-batch (a response wedged / the
            // connection went half-open under load). This is infrastructure failure, NOT a
            // real compile error, so fall back to a LOCAL compile of this program instead of
            // failing it. The kernels are already prepped (submitted_kernels), so we reuse
            // them directly — re-running prep_kernel would re-add reserved defines and assert.
            // ensure_kernel_binaries is cache-aware: kernels the server did finish are read
            // from disk; only the undelivered ones actually recompile.
            log_warning(
                tt::LogBuildKernels,
                "Remote JIT compile unavailable ({}); falling back to local compile for program {}.",
                e.what(),
                this->get_id());
            remote_ok = false;
        }

        if (remote_ok) {
            const std::string binary_root = build_env.build_env.get_out_kernel_root_path();
            for (const auto& [kernel, build_options] : submitted_kernels) {
                kernel->read_binaries(device, binary_root);
                kernel->register_kernel_elf_paths_with_watcher(*device, binary_root);
                Inspector::program_kernel_compile_finished(this, device, kernel, build_options, binary_root);
            }
        } else {
            for (auto& [kernel, build_options] : submitted_kernels) {
                launch_build_step(
                    [&, kernel] {
                        auto kernel_hash = KernelCompileHash(kernel, build_options, build_env.build_key());
                        const std::string binary_root =
                            ensure_kernel_binaries(kernel, device, build_options, build_env, kernel_hash);
                        kernel->read_binaries(device, binary_root);
                        kernel->register_kernel_elf_paths_with_watcher(*device, binary_root);
                        Inspector::program_kernel_compile_finished(this, device, kernel, build_options, binary_root);
                    },
                    events);
            }
            sync_build_steps(events);
        }
    } else {
        // Local path: parallel build via thread pool.
        for (auto& kernels : kernels_) {
            for (auto& [id, kernel] : kernels) {
                validate_kernel_placement(force_slow_dispatch, kernel, device->id());
                launch_build_step(
                    [&, kernel] {
                        auto [build_options, kernel_hash] = prep_kernel(kernel);
                        const std::string binary_root =
                            ensure_kernel_binaries(kernel, device, build_options, build_env, kernel_hash);
                        kernel->read_binaries(device, binary_root);
                        kernel->register_kernel_elf_paths_with_watcher(*device, binary_root);
                        Inspector::program_kernel_compile_finished(this, device, kernel, build_options, binary_root);
                    },
                    events);
            }
        }
        sync_build_steps(events);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(get_id(), device);
    }

    compiled_.insert(build_env.build_key());

    Inspector::program_compile_finished(this, device, build_env.build_key());
}

void detail::ProgramImpl::set_runtime_id(ProgramId id) { this->runtime_id = id; }

void Program::set_runtime_id(ProgramId id) { internal_->set_runtime_id(id); }

uint32_t detail::ProgramImpl::get_sem_base_addr(IDevice* device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type = tt::tt_metal::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, device, programmable_core_type);
    return base_addr + this->get_program_config(
                               MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type))
                           .sem_offset;
}

uint32_t detail::ProgramImpl::get_cb_base_addr(IDevice* device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type = tt::tt_metal::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, device, programmable_core_type);
    return base_addr + this->get_program_config(
                               MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type))
                           .cb_offset;
}

void detail::ProgramImpl::set_last_used_command_queue_for_testing(HWCommandQueue* queue) {
    this->last_used_command_queue_for_testing = queue;
}

HWCommandQueue* detail::ProgramImpl::get_last_used_command_queue() const {
    return this->last_used_command_queue_for_testing;
}

uint32_t detail::ProgramImpl::get_sem_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const {
    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].sem_size;
}

uint32_t detail::ProgramImpl::get_cb_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const {
    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].cb_size;
}

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool detail::ProgramImpl::runs_on_noc_unicast_only_cores() {
    return (
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
        not this->get_kernel_groups(MetalContext::instance().hal().get_programmable_core_type_index(
                                        HalProgrammableCoreType::ACTIVE_ETH))
                .empty());
}

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool detail::ProgramImpl::runs_on_noc_multicast_only_cores() {
    return (
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX) != -1 and
        not this->get_kernel_groups(
                    MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX))
                .empty());
}

Program::Program(Program&& other) noexcept = default;

Program& Program::operator=(Program&& other) noexcept = default;

Program::~Program() noexcept = default;

ProgramId detail::ProgramImpl::get_id() const { return this->id; }

ProgramId detail::ProgramImpl::get_runtime_id() const { return this->runtime_id; }

ProgramId Program::get_runtime_id() const { return internal_->get_runtime_id(); }

size_t detail::ProgramImpl::num_kernels() const {
    size_t count = 0;
    for (const auto& kernels : kernels_) {
        count += kernels.size();
    }
    return count;
}

std::span<const std::shared_ptr<CircularBufferImpl>> detail::ProgramImpl::circular_buffers() const {
    return circular_buffers_;
}

std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers() const {
    std::ranges::transform_view res_view(impl().circular_buffers(), [](const auto& impl_ptr) {
        return std::make_shared<CircularBuffer>(impl_ptr.get());
    });
    return {res_view.begin(), res_view.end()};
}

const std::vector<Semaphore>& detail::ProgramImpl::semaphores() const { return semaphores_; }

void detail::ProgramImpl::add_buffer(std::shared_ptr<Buffer> buf) { owned_buffer_pool.push_back(std::move(buf)); }

void detail::ProgramImpl::release_buffers() { owned_buffer_pool = {}; }

std::vector<std::reference_wrapper<const Semaphore>> detail::ProgramImpl::semaphores_on_core(
    const CoreCoord& core, CoreType core_type) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    for (const Semaphore& s : this->semaphores_) {
        if (s.initialized_on_logical_core(core) && s.core_type() == core_type) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

bool detail::ProgramImpl::is_finalized() const { return this->finalized_; }
void detail::ProgramImpl::set_finalized() { this->finalized_ = true; }

void detail::ProgramImpl::set_program_binary_status(ChipId device_id, ProgramBinaryStatus status) {
    Inspector::program_set_binary_status(this, device_id, status);
    this->binaries_on_device_[device_id] = status;
}

const ProgramTransferInfo& detail::ProgramImpl::get_program_transfer_info() const noexcept {
    return program_transfer_info;
}

std::shared_ptr<Buffer> ProgramImpl::get_kernels_buffer(IDevice* device) const noexcept {
    if (auto it = kernels_buffer_.find(device->id()); it != kernels_buffer_.end()) {
        return it->second;
    }
    return nullptr;
}

void detail::ProgramImpl::set_kernels_bin_buffer(const std::shared_ptr<Buffer>& buffer) {
    kernels_buffer_.insert({buffer->device()->id(), buffer});
}

std::unordered_map<uint64_t, ProgramCommandSequence>&
detail::ProgramImpl::get_cached_program_command_sequences() noexcept {
    return cached_program_command_sequences_;
}

void detail::ProgramImpl::set_program_offsets_and_sizes(uint32_t index, const ProgramOffsetsState& state) {
    auto& program_config = get_program_config(index);
    program_config.rta_offset = state.rta_offset;
    program_config.sem_offset = state.sem_offset;
    program_config.sem_size = state.sem_size;
    program_config.cb_offset = state.cb_offset;
    program_config.cb_size = state.cb_size;
    program_config.local_cb_size = state.local_cb_size;
    program_config.dfb_offset = state.dfb_offset;
    program_config.dfb_size = state.dfb_size;
    program_config.kernel_text_offset = state.kernel_text_offset;
    program_config.kernel_text_size = state.kernel_text_size;
    program_config_sizes_[index] = state.offset;
}

void detail::ProgramImpl::set_program_attrs_across_core_types(IDevice* device) {
    program_config_sizes_[programmable_core_count_] = runs_on_noc_multicast_only_cores();
    program_config_sizes_[programmable_core_count_ + 1] = runs_on_noc_unicast_only_cores();
    set_launch_msg_sem_offsets();
    // TODO: This check is wrong - it populates dispatch data for dispatch kernels
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        populate_dispatch_data(device);  // TODO: maybe rename
    }
}

void detail::ProgramImpl::finalize_offsets(IDevice* device) {
    if (is_finalized()) {
        return;
    }

    // Create proper function objects that capture 'this'
    detail::KernelsGetter kernels_getter =
        [this](uint32_t index) -> std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& {
        return this->get_kernels(index);
    };

    detail::KernelGroupsGetter kernel_groups_getter =
        [this](uint32_t index) -> std::vector<std::shared_ptr<KernelGroup>>& { return this->get_kernel_groups(index); };

    detail::SemaphoresGetter semaphores_getter = [this]() -> const std::vector<Semaphore>& {
        return this->semaphores();
    };

    // Create a span with just this program
    std::array<ProgramImpl*, 1> programs_array = {this};
    ttsl::Span<ProgramImpl*> programs(programs_array);

    (void)ProgramImpl::finalize_program_offsets(
        extract_context_id(device), device, kernels_getter, kernel_groups_getter, semaphores_getter, programs);

    set_finalized();
}

// Compute relative offsets (wrt the start of the kernel config ring buffer) and sizes of all
// program data structures in L1. Will be used when assembling dispatch commands for this program
uint32_t detail::ProgramImpl::finalize_program_offsets(
    ContextId context_id,
    IDevice* device,
    const KernelsGetter& kernels_getter,
    const KernelGroupsGetter& kernel_groups_getter,
    const SemaphoresGetter& semaphores_getter,
    ttsl::Span<ProgramImpl*> programs) {
    ProgramOffsetsState state;

    const auto& hal = MetalContext::instance(context_id).hal();

    // Collect dataflow buffers from all programs
    std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>> dataflow_buffers;
    for (ProgramImpl* program : programs) {
        for (const auto& dfb : program->dataflow_buffers()) {
            dataflow_buffers.push_back(dfb);
        }
    }

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(index);
        state.offset = program_dispatch::finalize_rt_args(
            kernels_getter(index), kernel_groups_getter(index), state.config_base_offset, index, state.rta_offset);

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        state.offset =
            program_dispatch::finalize_sems(index, state.offset, semaphores_getter(), state.sem_offset, state.sem_size);

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        state.offset = program_dispatch::finalize_cbs(
            index, kernel_groups_getter(index), state.offset, state.cb_offset, state.cb_size, state.local_cb_size);

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        state.offset = tt::tt_metal::experimental::dfb::detail::finalize_dfbs(
            index, kernel_groups_getter(index), dataflow_buffers, state.offset, state.dfb_offset, state.dfb_size);

        // On WH/BH, DFBs reuse the CB firmware init path; set local_cb_mask to a proper DFB
        // slot bitmask so setup_local_cb_read_write_interfaces initialises every DFB slot.
        if (!hal.has_tile_counter_registers() && !dataflow_buffers.empty()) {
            program_dispatch::finalize_dfb_masks(kernel_groups_getter(index), dataflow_buffers);
        }

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        state.offset = program_dispatch::finalize_kernel_bins(
            device,
            index,
            kernels_getter(index),
            kernel_groups_getter(index),
            state.offset,
            state.kernel_text_offset,
            state.kernel_text_size);

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        size_t max_size = get_ringbuffer_size(device, programmable_core_type);

        TT_FATAL(
            state.offset <= max_size,
            "Program size ({}) too large for kernel config buffer ({}) on {}",
            state.offset,
            max_size,
            enchantum::to_string(programmable_core_type));

        for (auto& program : programs) {
            program->set_program_offsets_and_sizes(index, state);
        }
    }

    // The sem offsets cross programmable_core_types so must be set after the loop above
    for (auto& program : programs) {
        program->set_program_attrs_across_core_types(device);
    }

    // Determine the DRAM kernel binary size per program and the max across all programs.
    // populate_dispatch_data (called above via set_program_attrs_across_core_types) packs all
    // kernel binaries from every core type into program_transfer_info.binary_data as a single,
    // page-aligned payload, so its size reflects the full padded binary data the prefetcher must cache.
    uint32_t max_program_sizeB = 0;
    for (auto& program : programs) {
        uint32_t binary_sizeB =
            static_cast<uint32_t>(program->get_program_transfer_info().binary_data.size() * sizeof(uint32_t));
        program->kernel_bins_sizeB = binary_sizeB;
        max_program_sizeB = std::max(max_program_sizeB, binary_sizeB);
    }
    return max_program_sizeB;
}

std::unordered_map<uint64_t, ProgramCommandSequence>&
ProgramImpl::get_trace_cached_program_command_sequences() noexcept {
    return trace_cached_program_command_sequences_;
}

detail::ProgramCompileGroup::~ProgramCompileGroup() { program_device_map_.clear(); }

void detail::ProgramCompileGroup::add_program(
    tt::tt_metal::IDevice* device, std::unique_ptr<tt::tt_metal::Program> program) {
    std::lock_guard lock(mutex_);
    TT_FATAL(!program_device_map_.contains(device), "Program already exists in the compile group.");
    program_device_map_[device] = std::move(program);
}

void detail::ProgramCompileGroup::compile_all(bool force_slow_dispatch) {
    std::lock_guard lock(mutex_);
    std::vector<std::shared_future<void>> events;
    for (auto& [device, program] : program_device_map_) {
        auto* pgm = program.get();
        launch_build_step(
            [device, pgm, force_slow_dispatch]() { pgm->impl().compile(device, force_slow_dispatch); }, events);
    }
    sync_build_steps(events);
}

void detail::ProgramCompileGroup::finalize_offsets() {
    std::lock_guard lock(mutex_);
    for (auto& [device, program] : program_device_map_) {
        if (!program->impl().is_finalized()) {
            program->impl().finalize_offsets(device);
        }
    }
}

void detail::ProgramCompileGroup::write_runtime_args(bool force_slow_dispatch) {
    std::lock_guard lock(mutex_);
    for (auto& [device, program] : program_device_map_) {
        detail::WriteRuntimeArgsToDevice(device, *program, force_slow_dispatch);
    }
}

std::unique_ptr<Program> detail::ProgramCompileGroup::remove_program(tt::tt_metal::IDevice* device) {
    std::lock_guard lock(mutex_);
    TT_FATAL(program_device_map_.contains(device), "Program not found in the compile group.");
    std::unique_ptr<Program> program = std::move(program_device_map_[device]);
    program_device_map_.erase(device);
    return program;
}

void detail::ProgramCompileGroup::clear() {
    std::lock_guard lock(mutex_);
    program_device_map_.clear();
}

bool detail::ProgramCompileGroup::contains(tt::tt_metal::IDevice* device) {
    std::lock_guard lock(mutex_);
    return program_device_map_.contains(device);
}

}  // namespace tt::tt_metal
