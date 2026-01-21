// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "data_types.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "jit_build/build.hpp"
#include <tt_stl/enum.hpp>
#include "jit_build/jit_build_options.hpp"
#include "kernel_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include <tt-logger/tt-logger.hpp>
#include "profiler_state.hpp"
#include "program_command_sequence.hpp"
#include "program_device_map.hpp"
#include "program_impl.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/overloaded.hpp>
#include "sub_device_types.hpp"
#include "tile.hpp"
#include "tt_memory.h"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/impl/debug/inspector/inspector.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_common.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "host_api.hpp"
#include "kernels/kernel.hpp"
#include "tt_stl/reflection.hpp"
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/allocator/allocator.hpp"

namespace tt {
class tt_hlk_desc;
enum CBIndex : std::uint8_t;
namespace tt_metal {
class CommandQueue;
class EnqueueProgramCommand;
namespace experimental {
class GlobalCircularBuffer;
}  // namespace experimental
}  // namespace tt_metal
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

void validate_kernel_placement(bool force_slow_dispatch, std::shared_ptr<Kernel> kernel) {
    // Placement rules:
    //  Fast dispatch (tensix):
    //      - tensix kernels cannot be on dispatch cores
    //  Fast dispatch (ethernet):
    //      - eth kernels cannot be on idle eth cores
    bool slow_dispatch = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    tt::CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);

    // Kernels used to implement fast dispatch can be placed on dispatch cores
    if (not slow_dispatch and not force_slow_dispatch) {
        const std::vector<CoreCoord>& dispatch_cores =
            MetalContext::instance().get_dispatch_query_manager().get_logical_dispatch_cores_on_user_chips();
        bool on_dispatch_core = std::any_of(
            dispatch_cores.begin(),
            dispatch_cores.end(),
            [&kernel, &dispatch_core_type](const CoreCoord& dispatch_core) {
                if (kernel->get_kernel_core_type() != dispatch_core_type) {
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

void GenerateBinaries(IDevice* device, JitBuildOptions& build_options, const std::shared_ptr<Kernel>& kernel) {
    // ZoneScoped;
    // const std::string tracyPrefix = "GenerateBinaries_";
    // ZoneName((tracyPrefix + build_options.name).c_str(), build_options.name.length() + tracyPrefix.length());
    try {
        jit_build_genfiles_descriptors(
            BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env, build_options);
        kernel->generate_binaries(device, build_options);
    } catch (std::runtime_error& ex) {
        TT_THROW("Failed to generate binaries for {} {}", kernel->name(), ex.what());
    }
}

#ifdef GENERATE_HASH_LOG
#include <fstream>
#endif

size_t KernelCompileHash(const std::shared_ptr<Kernel>& kernel, JitBuildOptions& build_options, uint64_t build_key) {
    // Store the build key into the KernelCompile hash. This will be unique per command queue
    // configuration (necessary for dispatch kernels).
    // Also account for watcher/dprint enabled in hash because they enable additional code to
    // be compiled into the kernel.
    std::string compile_hash_str = fmt::format(
        "{}_{}_{}_{}",
        build_key,
        std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc)),
        kernel->compute_hash(),
        tt::tt_metal::MetalContext::instance().rtoptions().get_compile_hash_string());
    size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

#ifdef GENERATE_HASH_LOG
    static std::ofstream f("/tmp/hashlog.txt");
    static std::mutex mutex_;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        f << kernel->name() << " :: " << build_key << "::" << std::hash<tt_hlk_desc>{}(build_options.hlk_desc)
          << " :: " << kernel->compute_hash() << " :: " << compile_hash_str << " " << compile_hash << std::endl
          << std::flush;
    }
#endif
    return compile_hash;
}
}  // namespace

namespace experimental {

void ClearKernelCache() { detail::HashLookup::inst().clear(); }

}  // namespace experimental

std::atomic<uint64_t> detail::ProgramImpl::program_counter = 0;

detail::ProgramImpl::ProgramImpl() :

    cached_device_hash_(std::nullopt),
    programmable_core_count_(MetalContext::instance().hal().get_programmable_core_type_count()),
    id(program_counter++) {
    for (uint32_t i = 0; i < programmable_core_count_; i++) {
        kernels_.push_back({});
        grid_extent_.push_back({});
        kernel_groups_.push_back({});
        core_to_kernel_group_index_table_.push_back({});
    }

    program_configs_.resize(programmable_core_count_);
    program_config_sizes_.resize(programmable_core_count_ + 2);

    Inspector::program_created(this);
}

detail::ProgramImpl::~ProgramImpl() noexcept { Inspector::program_destroyed(this); }

Program::Program() : internal_(std::make_shared<detail::ProgramImpl>()) {
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

        auto config = std::visit(
            tt::stl::overloaded{
                [&](const ReaderConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
                    return ReaderDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
                        std::move(named_compile_args),
                        kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2)};
                },
                [&](const WriterConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
                    return WriterDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
                        std::move(named_compile_args),
                        kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2)};
                },
                [&](const DataMovementConfigDescriptor& dm_descriptor)
                    -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
                    return DataMovementConfig{
                        .processor = dm_descriptor.processor,
                        .noc = dm_descriptor.noc,
                        .noc_mode = dm_descriptor.noc_mode,
                        .compile_args = std::move(compile_args),
                        .defines = std::move(defines),
                        .named_compile_args = std::move(named_compile_args),
                        .opt_level = kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2),
                    };
                },
                [&](const ComputeConfigDescriptor& compute_descriptor)
                    -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
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
                    };
                },
                [&](const EthernetConfigDescriptor& ethernet_descriptor)
                    -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
                    return EthernetConfig{
                        .eth_mode = ethernet_descriptor.eth_mode,
                        .noc = ethernet_descriptor.noc,
                        .processor = ethernet_descriptor.processor,
                        .compile_args = std::move(compile_args),
                        .defines = std::move(defines),
                        .named_compile_args = std::move(named_compile_args),
                        .opt_level = kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::Os),
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
    uint32_t local_cb_mask,
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
    // Fast dispatch kernel config mangement happens under the CQ and will re-program the base
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        kernel_config.kernel_config_base()[index] =
            hal.get_dev_addr(hal.get_programmable_core_type(index), HalL1MemAddrType::KERNEL_CONFIG);
    }

    std::set<NOC_MODE> noc_modes;
    for (auto kernel_id : this->kernel_ids) {
        const auto kernel = program.get_kernel(kernel_id);
        auto processor_class = kernel->get_kernel_processor_class();
        auto num_binaries = kernel->expected_num_binaries();
        for (uint32_t i = 0; i < num_binaries; i++) {
            auto processor_type = kernel->get_kernel_processor_type(i);
            auto processor_index = hal.get_processor_index(
                hal.get_programmable_core_type(programmable_core_type_index), processor_class, processor_type);
            kernel_config.watcher_kernel_ids()[processor_index] = kernel->get_watcher_kernel_id();
            kernel_config.enables() |= 1u << processor_index;
        }
        auto class_id = kernel->dispatch_class();

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
                        if (class_id ==
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
                            class_id ==
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
            uint32_t min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS;
            uint32_t local_cb_mask = 0;

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
                            uint32_t used_cbs = local_val->second.to_ulong();
                            local_cb_mask |= used_cbs;
                            max_local_cb_end_index = std::max(
                                max_local_cb_end_index, NUM_CIRCULAR_BUFFERS - (uint32_t)__builtin_clz(used_cbs));
                            if (!logged_noncontiguous) {
                                // Zeroes out the contiguous run of set bits starting at zero. Anything remaining is
                                // above a zero bit.
                                uint32_t non_contiguous_cbs = used_cbs & (used_cbs + 1);
                                if (non_contiguous_cbs) {
                                    // ~used_cbs is always nonzero, because otherwise all CBs are in use and therefore
                                    // contiguous.
                                    uint32_t first_unused_index = (uint32_t)__builtin_ctz(~used_cbs);
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
                                    static std::set<std::tuple<uint32_t, uint32_t, std::string>> logged;
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
                                        for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
                                            if (non_contiguous_cbs & (1 << i)) {
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
                                min_remote_cb_start_index, (uint32_t)__builtin_ctz(remote_val->second.to_ulong()));
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
            std::vector<KernelHandle> kernel_ids(kernels.begin(), kernels.end());
            // Sort kernel ids by dispatch class, so loops over this array will be in dispatch class order
            std::sort(kernel_ids.begin(), kernel_ids.end(), [&handle_to_kernel](KernelHandle a, KernelHandle b) {
                return handle_to_kernel.at(a)->dispatch_class() < handle_to_kernel.at(b)->dispatch_class();
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
                auto add_buffer_indices = [&cb_indices](
                                              const std::unordered_set<uint8_t>& buffer_indices,
                                              std::bitset<NUM_CIRCULAR_BUFFERS>& target_cb_indices) {
                    for (uint32_t buffer_index : buffer_indices) {
                        // TT_ASSERT since we validate when constructing the config that it's within range
                        TT_ASSERT(
                            buffer_index < NUM_CIRCULAR_BUFFERS,
                            "Invalid circular buffer index: {} should be between 0 and {}",
                            buffer_index,
                            NUM_CIRCULAR_BUFFERS);
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
    return core_ranges;
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

void detail::ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    // ZoneScoped;
    if (not this->local_circular_buffer_allocation_needed_) {
        return;
    }

    uint64_t base_cb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (const auto& circular_buffer : this->circular_buffers_) {
        if (circular_buffer->globally_allocated()) {
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
        tt::tt_metal::GraphTracker::instance().track_allocate_cb(
            circular_buffer->core_ranges(),
            computed_addr,
            circular_buffer->size(),
            circular_buffer->globally_allocated(),
            device);
        circular_buffer->set_locally_allocated_address(computed_addr);
    }
    this->local_circular_buffer_allocation_needed_ = false;
}

void detail::ProgramImpl::validate_circular_buffer_region(const IDevice* device) {
    // ZoneScoped;

    // TODO: Circular buffer allocation and validation could be better optimized by determining usage per sub-device
    std::optional<DeviceAddr> lowest_address =
        device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();

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

void detail::ProgramImpl::init_semaphores(
    const IDevice& device, const CoreCoord& logical_core, uint32_t programmable_core_type_index) const {
    const auto& hal = MetalContext::instance().hal();
    uint64_t kernel_config_base =
        hal.get_dev_addr(hal.get_programmable_core_type(programmable_core_type_index), HalL1MemAddrType::KERNEL_CONFIG);
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

void detail::ProgramImpl::set_cb_data_fmt(const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const {
    // ZoneScoped;
    for (const auto& logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(
                    static_cast<CBIndex>(buffer_index), circular_buffer->data_format(buffer_index));
            }
        }
    }
}

void detail::ProgramImpl::set_cb_tile_dims(const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const {
    // ZoneScoped;
    for (const auto& logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                auto tile = circular_buffer->tile(buffer_index);
                if (tile.has_value()) {
                    build_options.set_cb_tile_dims_all_cores(
                        static_cast<CBIndex>(buffer_index),
                        tile->get_num_faces(),
                        tile->get_partial_face(),
                        tile->get_face_shape()[0],
                        tile->get_narrow_tile(),
                        tile->get_tile_shape()[0],
                        tile->get_tile_shape()[1]);
                    build_options.set_cb_tile_size_all_cores(
                        static_cast<CBIndex>(buffer_index),
                        tile->get_tile_size(circular_buffer->data_format(buffer_index)));
                } else {
                    Tile t;
                    build_options.set_cb_tile_size_all_cores(
                        static_cast<CBIndex>(buffer_index),
                        t.get_tile_size(circular_buffer->data_format(buffer_index)));
                }
            }
        }
    }
}

void detail::ProgramImpl::populate_dispatch_data(IDevice* device) {
    // Mock devices don't dispatch to hardware, skip dispatch data population
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
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
            const auto& binaries =
                kernel->binaries(BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key());
            std::vector<uint32_t> dst_base_addrs;
            std::vector<uint32_t> page_offsets;
            std::vector<uint32_t> lengths;
            std::vector<uint32_t> processor_ids;
            uint32_t transfer_info_index = 0;

            for (size_t sub_kernel_index = 0; sub_kernel_index < binaries.size(); ++sub_kernel_index) {
                const ll_api::memory& kernel_bin = *binaries[sub_kernel_index];

                // TODO: Pack erisc spans too, and then everthing is
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
                    auto dispatch_class = kernel->dispatch_class();
                    int proc_sub_class = 0;
                    for (uint32_t& dst_addr : kernel_transfer_info.at(device_local_kernel_id).dst_base_addrs) {
                        // TODO: ditch this w/ linear writes based on program config kernel_text_offset and size
                        dst_addr = kernel_group->kernel_text_offsets[dispatch_class + proc_sub_class];
                        proc_sub_class++;
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
                    auto dispatch_class = kernel->dispatch_class();
                    kernel_ids.push_back(device_local_kernel_id);

                    // Update destination address by kernel config offset
                    if (hal.get_core_kernel_stored_in_config_buffer(hal.get_programmable_core_type(index))) {
                        int proc_sub_class = 0;
                        for (uint32_t& dst_addr : kernel_transfer_info.at(device_local_kernel_id).dst_base_addrs) {
                            dst_addr = kernel_group->kernel_text_offsets[dispatch_class + proc_sub_class];
                            proc_sub_class++;
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
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr ||
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

void ProgramImpl::generate_dispatch_commands(IDevice* device, bool use_prefetcher_cache) {
    uint64_t command_hash = *device->get_active_sub_device_manager_id();

    uint64_t device_hash = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key();
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        // When coordinate virtualization is not enabled, explicitly encode the device
        // id into the device hash, to always assert on programs being reused across devices.
        ttsl::hash::hash_combine(device_hash, device->id());
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
        auto sub_device_id = this->determine_sub_device_ids(device).at(0);
        ProgramCommandSequence program_command_sequence;
        program_dispatch::insert_empty_program_dispatch_preamble_cmd(program_command_sequence);
        program_dispatch::insert_stall_cmds(program_command_sequence, sub_device_id, device);
        program_dispatch::assemble_device_commands(
            program_command_sequence, *this, device, sub_device_id, use_prefetcher_cache);

        program_command_sequence.kernel_bins_sizeB = this->kernel_bins_sizeB;
        program_command_sequence.prefetcher_cache_used = use_prefetcher_cache;

        // TODO: We currently do not have a mechanism of removing entries in the cache when a manager is removed
        // This means programs will contain stale entries in the cache until the program is deleted
        cached_program_command_sequences.insert({command_hash, std::move(program_command_sequence)});
    } else {
        TT_ASSERT(
            cached_program_command_sequences.at(command_hash).prefetcher_cache_used == use_prefetcher_cache,
            "Prefetcher cache used mismatch for program {} on device {}",
            this->get_id(),
            device->id());
    }
}

void ProgramImpl::generate_trace_dispatch_commands(IDevice* device, bool use_prefetcher_cache) {
    uint64_t command_hash = *device->get_active_sub_device_manager_id();

    uint64_t device_hash = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key();
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        // When coordinate virtualization is not enabled, explicitly encode the device
        // id into the device hash, to always assert on programs being reused across devices.
        device_hash = (device_hash << 32) | (device->id());
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
        auto sub_device_id = this->determine_sub_device_ids(device).at(0);
        ProgramCommandSequence program_command_sequence;
        program_dispatch::insert_empty_program_dispatch_preamble_cmd(program_command_sequence);
        program_dispatch::insert_stall_cmds(program_command_sequence, sub_device_id, device);
        program_dispatch::assemble_device_commands(
            program_command_sequence, *this, device, sub_device_id, use_prefetcher_cache);
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
            device->id());
    }
}

void detail::ProgramImpl::compile(IDevice* device, bool force_slow_dispatch) {
    // ZoneScoped;
    const auto& build_env = BuildEnvManager::get_instance().get_device_build_env(device->build_id());

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

    std::vector<std::shared_future<void>> events;

    for (auto& kernels : kernels_) {
        for (auto& [id, kernel] : kernels) {
            validate_kernel_placement(force_slow_dispatch, kernel);
            launch_build_step(
                [kernel, device, this, &build_env] {
                    JitBuildOptions build_options(build_env.build_env);
                    kernel->set_build_options(build_options);
                    if (this->compiled_.empty()) {
                        this->set_remote_circular_buffer_init(kernel);
                    }
                    this->set_cb_data_fmt(kernel->logical_coreranges(), build_options);
                    this->set_cb_tile_dims(kernel->logical_coreranges(), build_options);

                    auto kernel_hash = KernelCompileHash(kernel, build_options, build_env.build_key());

                    const std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash) + "/";
                    kernel->set_full_name(kernel_path_suffix);
                    build_options.set_name(kernel_path_suffix);

                    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() !=
                        tt::TargetDevice::Mock) {
                        kernel->register_kernel_elf_paths_with_watcher(*device);
                    }

                    bool is_mock = tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() ==
                                   tt::TargetDevice::Mock;

                    if (detail::HashLookup::inst().add(kernel_hash)) {
                        if (!is_mock) {
                            GenerateBinaries(device, build_options, kernel);
                        } else {
                            // Create empty stub binaries for mock devices
                            std::vector<const ll_api::memory*> empty_binaries(kernel->expected_num_binaries(), nullptr);
                            kernel->set_binaries(build_env.build_key(), std::move(empty_binaries));
                        }
                        detail::HashLookup::inst().add_generated_bin(kernel_hash);
                    }
                    detail::HashLookup::inst().wait_for_bin_generated(kernel_hash);

                    Inspector::program_kernel_compile_finished(this, device, kernel, build_options);
                },
                events);
        }
    }
    sync_build_steps(events);

    // Mock devices don't have binaries to read
    bool is_mock =
        tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
    if (!is_mock) {
        for (const auto& kernels : kernels_) {
            for (const auto& pair : kernels) {
                const auto& kernel = pair.second;
                launch_build_step([kernel, device] { kernel->read_binaries(device); }, events);
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

void detail::ProgramImpl::set_last_used_command_queue_for_testing(CommandQueue* queue) {
    this->last_used_command_queue_for_testing = queue;
}

CommandQueue* detail::ProgramImpl::get_last_used_command_queue() const {
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

bool detail::ProgramImpl::kernel_binary_always_stored_in_ringbuffer() {
    // Active ethernet cores use a fixed address for the kernel binary, because they don't have enough memory to have
    // that big of a ringbuffer.
    return !(
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
        not this->get_kernel_groups(MetalContext::instance().hal().get_programmable_core_type_index(
                                        HalProgrammableCoreType::ACTIVE_ETH))
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
    program_config.crta_offsets = state.crta_offsets;
    program_config.crta_sizes = state.crta_sizes;
    program_config.sem_offset = state.sem_offset;
    program_config.sem_size = state.sem_size;
    program_config.cb_offset = state.cb_offset;
    program_config.cb_size = state.cb_size;
    program_config.local_cb_size = state.local_cb_size;
    program_config.kernel_text_offset = state.kernel_text_offset;
    program_config.kernel_text_size = state.kernel_text_size;
    program_config_sizes_[index] = state.offset;
}

void detail::ProgramImpl::set_program_attrs_across_core_types(IDevice* device) {
    program_config_sizes_[programmable_core_count_] = runs_on_noc_multicast_only_cores();
    program_config_sizes_[programmable_core_count_ + 1] = runs_on_noc_unicast_only_cores();
    set_launch_msg_sem_offsets();
    // TODO: This check is wrong - it populates dispatch data for dispatch kernels
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        populate_dispatch_data(device);  // TODO: maybe rename
    }
}

using KernelsGetter = std::function<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>&(uint32_t index)>;
using KernelGroupsGetter = std::function<std::vector<std::shared_ptr<KernelGroup>>&(uint32_t index)>;
using SemaphoresGetter = std::function<const std::vector<Semaphore>&()>;

void detail::ProgramImpl::finalize_offsets(IDevice* device) {
    if (is_finalized()) {
        return;
    }

    // Create proper function objects that capture 'this'
    KernelsGetter kernels_getter =
        [this](uint32_t index) -> std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& {
        return this->get_kernels(index);
    };

    KernelGroupsGetter kernel_groups_getter = [this](uint32_t index) -> std::vector<std::shared_ptr<KernelGroup>>& {
        return this->get_kernel_groups(index);
    };

    SemaphoresGetter semaphores_getter = [this]() -> const std::vector<Semaphore>& { return this->semaphores(); };

    // Create a span with just this program
    std::array<ProgramImpl*, 1> programs_array = {this};
    tt::stl::Span<ProgramImpl*> programs(programs_array);

    (void)ProgramImpl::finalize_program_offsets(
        device, kernels_getter, kernel_groups_getter, semaphores_getter, programs);

    set_finalized();
}

// Compute relative offsets (wrt the start of the kernel config ring buffer) and sizes of all
// program data structures in L1. Will be used when assembling dispatch commands for this program
uint32_t detail::ProgramImpl::finalize_program_offsets(
    IDevice* device,
    const KernelsGetter& kernels_getter,
    const KernelGroupsGetter& kernel_groups_getter,
    const SemaphoresGetter& semaphores_getter,
    tt::stl::Span<ProgramImpl*> programs) {
    ProgramOffsetsState state;

    const auto& hal = MetalContext::instance().hal();

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(index);
        state.offset = program_dispatch::finalize_rt_args(
            kernels_getter(index),
            kernel_groups_getter(index),
            state.config_base_offset,
            index,
            state.rta_offset,
            state.crta_offsets,
            state.crta_sizes);

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        state.offset =
            program_dispatch::finalize_sems(index, state.offset, semaphores_getter(), state.sem_offset, state.sem_size);

        TT_ASSERT(state.offset == tt::align(state.offset, hal.get_alignment(HalMemType::L1)));

        state.offset = program_dispatch::finalize_cbs(
            index, kernel_groups_getter(index), state.offset, state.cb_offset, state.cb_size, state.local_cb_size);

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
            state.offset < max_size,
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

    // determine max program size across all programs
    uint32_t max_program_sizeB = 0;
    for (auto& program : programs) {
        program->kernel_bins_sizeB = state.kernel_text_size;
        max_program_sizeB = std::max(max_program_sizeB, state.kernel_text_size);
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
