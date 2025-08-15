// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <circular_buffer.hpp>
#include <circular_buffer_config.hpp>
#include <device.hpp>
#include <graph_tracking.hpp>
#include <enchantum/enchantum.hpp>
#include <memory_reporter.hpp>
#include <persistent_kernel_cache.hpp>
#include <semaphore.hpp>
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

#include "assert.hpp"
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "dev_msgs.h"
#include "impl/context/metal_context.hpp"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "jit_build/build.hpp"
#include "jit_build/jit_build_options.hpp"
#include "kernel.hpp"
#include "kernel_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include "llrt.hpp"
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
#include "tt_backend_api_types.hpp"
#include "tt_memory.h"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/impl/debug/inspector.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/xy_pair.h>
#include "util.hpp"
#include "utils.hpp"
#include "host_api.hpp"
#include "kernels/kernel_impl.hpp"

namespace tt {
class tt_hlk_desc;
enum CBIndex : std::uint8_t;
namespace tt_metal {
class CommandQueue;
class EnqueueProgramCommand;
namespace detail {
class Internal_;
}  // namespace detail
namespace experimental {
class GlobalCircularBuffer;
}  // namespace experimental
}  // namespace tt_metal
}  // namespace tt

namespace {

using namespace tt::tt_metal;

size_t get_ringbuffer_size(IDevice* device, HalProgrammableCoreType programmable_core_type) {
    if (programmable_core_type == HalProgrammableCoreType::TENSIX) {
        return device->allocator()->get_config().l1_unreserved_base -
               MetalContext::instance().hal().get_dev_addr(
                   HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
    } else {
        return MetalContext::instance().hal().get_dev_size(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);
    }
}

void validate_kernel_placement(IDevice* device, bool force_slow_dispatch, std::shared_ptr<Kernel> kernel) {
    // Placement rules:
    //  Slow dispatch:
    //      - kernels cannot be on storage only cores
    //  Fast dispatch (tensix):
    //      - kernels cannot be on storage only cores an
    //      - tensix kernels cannot be on dispatch cores
    //  Fast dispatch (ethernet):
    //      - kernels cannot be on storage only cores
    //      - eth kernels cannot be on idle eth cores
    bool slow_dispatch = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    const std::vector<CoreCoord>& storage_cores =
        MetalContext::instance().get_dispatch_query_manager().get_logical_storage_cores_on_user_chips();
    bool on_storage_only_core =
        std::any_of(storage_cores.begin(), storage_cores.end(), [&kernel](const CoreCoord& storage_core) {
            return kernel->is_on_logical_core(storage_core);
        });
    TT_FATAL(
        not on_storage_only_core,
        "Illegal kernel placement for {}. Kernels cannot be placed on storage only cores!",
        kernel->name());

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
std::atomic<bool> enable_persistent_kernel_cache = false;

void GenerateBinaries(IDevice* device, JitBuildOptions &build_options, const std::shared_ptr<Kernel>& kernel) {
    //ZoneScoped;
    //const std::string tracyPrefix = "GenerateBinaries_";
    //ZoneName((tracyPrefix + build_options.name).c_str(), build_options.name.length() + tracyPrefix.length());
    try {
        jit_build_genfiles_descriptors(
            BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env, build_options);
        KernelImpl::from(*kernel).generate_binaries(device, build_options);
    } catch (std::runtime_error &ex) {
        TT_THROW("Failed to generate binaries for {} {}", kernel->name(), ex.what());
    }
}

#ifdef GENERATE_HASH_LOG
#include <fstream>
#endif

size_t KernelCompileHash(const std::shared_ptr<Kernel>& kernel, JitBuildOptions& build_options, uint32_t build_key) {
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
namespace detail {

KernelHandle AddKernel (Program &program, const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType core_type) {
    return program.internal_->add_kernel(std::move(kernel), core_type);
}

std::shared_ptr<Kernel> GetKernel(const Program &program, KernelHandle kernel_id) {
    return program.get_kernel(kernel_id);
}

std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id) {
    return program.internal_->get_circular_buffer(id);
}

// Checks that circular buffers do not grow into L1 buffer space
void ValidateCircularBufferRegion(const Program &program, const IDevice* device) {
    program.internal_->validate_circular_buffer_region(device);
}

void EnablePersistentKernelCache() { enable_persistent_kernel_cache = true; }

void DisablePersistentKernelCache() { enable_persistent_kernel_cache = false; }

class Internal_ {
   public:
       using map_type = decltype(detail::ProgramImpl::circular_buffer_by_id_);

       static const map_type& get_circular_buffers_by_id(const Program& program) noexcept {
           return program.internal_->circular_buffer_by_id_;
       }
};

}  // namespace detail

std::atomic<uint64_t> detail::ProgramImpl::program_counter = 0;

detail::ProgramImpl::ProgramImpl() :
    id(program_counter++),
    runtime_id(0),
    local_circular_buffer_allocation_needed_(false),
    finalized_(false),
    cached_device_hash_(std::nullopt) {
    programmable_core_count_ = MetalContext::instance().hal().get_programmable_core_type_count();
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

detail::ProgramImpl::~ProgramImpl() noexcept {
    Inspector::program_destroyed(this);
}

Program::Program() : internal_(std::make_shared<detail::ProgramImpl>()) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureProgramConstructor, *this);
}

Program::Program(const ProgramDescriptor& descriptor) : internal_(std::make_shared<detail::ProgramImpl>()) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureProgramConstructor, *this);

    for (auto& cb_descriptor : descriptor.cbs) {
        internal_->add_circular_buffer_(std::make_shared<CircularBuffer>(cb_descriptor));
    }

    for (size_t i = 0; i < descriptor.semaphores.size(); i++) {
        auto& semaphore_descriptor = descriptor.semaphores[i];
        add_semaphore(
            semaphore_descriptor.core_ranges, i, semaphore_descriptor.initial_value, semaphore_descriptor.core_type);
    }

    for (auto& kernel_descriptor : descriptor.kernels) {
        bool is_file = kernel_descriptor.source_type == KernelDescriptor::SourceType::FILE_PATH;
        std::vector<uint32_t> compile_args(
            kernel_descriptor.compile_time_args.begin(), kernel_descriptor.compile_time_args.end());
        std::map<std::string, std::string> defines(kernel_descriptor.defines.begin(), kernel_descriptor.defines.end());

        auto config = std::visit(
            tt::stl::overloaded{
                [&](const ReaderConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
                    return ReaderDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
                        kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2)};
                },
                [&](const WriterConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> {
                    return WriterDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
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
                        .opt_level = kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::Os),
                    };
                },
            },
            kernel_descriptor.config);

        auto kernel_handle =
            is_file
                ? CreateKernel(*this, kernel_descriptor.kernel_source, kernel_descriptor.core_ranges, config)
                : CreateKernelFromString(*this, kernel_descriptor.kernel_source, kernel_descriptor.core_ranges, config);

        for (size_t i = 0; i < kernel_descriptor.runtime_args.size(); i++) {
            for (size_t j = 0; j < kernel_descriptor.runtime_args[i].size(); j++) {
                SetRuntimeArgs(*this, kernel_handle, CoreCoord(i, j), kernel_descriptor.runtime_args[i][j]);
            }
        }
        SetCommonRuntimeArgs(*this, kernel_handle, kernel_descriptor.common_runtime_args);
    }
}

KernelHandle detail::ProgramImpl::add_kernel(
    const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& programmable_core_type) {
    TT_FATAL(this->compiled_.empty(), "Cannot add kernel to an already compiled program {}", this->id);
    // Id is unique across all kernels on all core types
    KernelHandle id = this->num_kernels();
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);

    RISCV new_kernel_type = kernel->processor();
    std::set<CoreCoord> kernel_logical_cores = kernel->logical_cores();
    for (size_t i = 0; i < this->num_kernels(); i++) {
        // Note, looks like id is program specific, and increments naturally as kernels are added.
        //  add_kernel -> id = num_kernels -> kernel is inserted -> next num_kernels() increments.
        std::shared_ptr<Kernel> check_kernel = this->get_kernel(i);
        RISCV check_kernel_type = check_kernel->processor();
        std::set<CoreCoord> check_kernel_logical_cores = check_kernel->logical_cores();
        for (CoreCoord coreCoord : kernel_logical_cores) {
            TT_FATAL(
                !(check_kernel_logical_cores.find(coreCoord) != check_kernel_logical_cores.end() &&
                  new_kernel_type == check_kernel_type),
                "Core Overlap Between (\"{}\") and new kernel (\"{}\") at {}",
                check_kernel->name(),
                kernel->name(),
                coreCoord.str());
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
    for (const auto &kernels : this->kernels_) {
        if (kernels.find(kernel_id) != kernels.end()) {
            return kernels.at(kernel_id);
        }
    }

    TT_ASSERT(false, "Did not find kernel id across all core types!");
    return nullptr;
}

std::shared_ptr<Kernel> Program::get_kernel(KernelHandle kernel_id) const { return internal_->get_kernel(kernel_id); }

KernelGroup::KernelGroup() : core_ranges(CoreRangeSet()) {}

KernelGroup::KernelGroup(
    const detail::ProgramImpl& program,
    uint32_t programmable_core_type_index,
    kernel_id_array_t kernel_ids,
    bool /*erisc_is_idle*/,
    uint32_t local_cb_mask,
    uint32_t min_remote_cb_start_index,
    const CoreRangeSet& new_ranges) :
    core_ranges(CoreRangeSet()) {
    this->programmable_core_type_index = programmable_core_type_index;
    this->core_ranges = this->core_ranges.merge(new_ranges);
    this->kernel_ids = kernel_ids;
    this->launch_msg.kernel_config.brisc_noc_mode = NOC_MODE::DM_DEDICATED_NOC;

    std::memset(&this->launch_msg, 0, sizeof(launch_msg_t));

    // Slow dispatch uses fixed addresses for the kernel config, configured here statically
    // Fast dispatch kernel config mangement happens under the CQ and will re-program the base
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        this->launch_msg.kernel_config.kernel_config_base[index] =
            hal.get_dev_addr(index, HalL1MemAddrType::KERNEL_CONFIG);
    }

    uint32_t processor_classes = hal.get_processor_classes_count(programmable_core_type_index);
    std::set<NOC_MODE> noc_modes;
    for (int class_id = 0; class_id < processor_classes; class_id++) {
        auto& optional_id = kernel_ids[class_id];
        if (optional_id) {
            const auto kernel = program.get_kernel(optional_id.value());
            this->launch_msg.kernel_config.watcher_kernel_ids[class_id] = kernel->get_watcher_kernel_id();
            this->launch_msg.kernel_config.enables |= 1 << class_id;

            if (programmable_core_type_index == hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)) {
                // The code below sets the brisc_noc_id for use by the device firmware
                // Use 0 if neither brisc nor ncrisc specify a noc
                if (class_id == utils::underlying_type<DataMovementProcessor>(DataMovementProcessor::RISCV_0)) {
                    noc_modes.insert(std::get<DataMovementConfig>(kernel->config()).noc_mode);
                    // Use brisc's noc if brisc specifies a noc
                    this->launch_msg.kernel_config.brisc_noc_id = std::get<DataMovementConfig>(kernel->config()).noc;
                    // if noc mode is already set to DM_DYNAMIC_NOC then we can't change back to DM_DEDICATED_NOC
                    if (std::get<DataMovementConfig>(kernel->config()).noc_mode == NOC_MODE::DM_DYNAMIC_NOC) {
                        this->launch_msg.kernel_config.brisc_noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
                    }
                } else if (class_id == utils::underlying_type<DataMovementProcessor>(DataMovementProcessor::RISCV_1)) {
                    noc_modes.insert(std::get<DataMovementConfig>(kernel->config()).noc_mode);
                    // Use 1-ncrisc's noc (the other noc) if ncrisc specifies a noc
                    // If both brisc and ncrisc set the noc, then this is safe due to prior correctness validation
                    this->launch_msg.kernel_config.brisc_noc_id = 1 - std::get<DataMovementConfig>(kernel->config()).noc;
                    // if noc mode is already set to DM_DYNAMIC_NOC then we can't change back to DM_DEDICATED_NOC
                    if (std::get<DataMovementConfig>(kernel->config()).noc_mode == NOC_MODE::DM_DYNAMIC_NOC) {
                        this->launch_msg.kernel_config.brisc_noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
                    }
                }
            }
        }
    }
    TT_FATAL(noc_modes.size() <= 1, "KernelGroup must have the same noc mode for all kernels");

    for (uint32_t index = 0; index < NUM_PROCESSORS_PER_CORE_TYPE; index ++) {
        this->kernel_bin_sizes[index] = 0;
        this->kernel_text_offsets[index] = 0;
        this->launch_msg.kernel_config.kernel_text_offset[index] = 0;
    }
    this->launch_msg.kernel_config.ncrisc_kernel_size16 = 0;

    this->launch_msg.kernel_config.exit_erisc_kernel = false;
    this->launch_msg.kernel_config.local_cb_mask = local_cb_mask;
    this->launch_msg.kernel_config.min_remote_cb_start_index = min_remote_cb_start_index;
    this->go_msg.signal = RUN_MSG_GO;
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

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& Program::get_kernels(uint32_t programmable_core_type_index) {
    return internal_->get_kernels(programmable_core_type_index);
}

KernelGroup* detail::ProgramImpl::kernels_on_core(const CoreCoord& core, uint32_t programmable_core_type_index) {
    update_kernel_groups(programmable_core_type_index);
    if (core.x >= grid_extent_[programmable_core_type_index].x || core.y >= grid_extent_[programmable_core_type_index].y)
        return nullptr;
    uint8_t index = core_to_kernel_group_index_table_[programmable_core_type_index].at(core.y * grid_extent_[programmable_core_type_index].x + core.x);
    return (index == core_to_kernel_group_invalid_index) ? nullptr : kernel_groups_[programmable_core_type_index].at(index).get();
}

struct KernelGroupInt {
    bool valid;
    kernel_id_array_t kernel_ids;

    bool operator==(const KernelGroupInt &b) const;
    // fix this
    void update(dispatch_core_processor_classes proc_class, size_t kernel_idx) {
        this->kernel_ids[proc_class] = static_cast<KernelHandle>(kernel_idx);
    }
};

bool KernelGroupInt::operator==(const KernelGroupInt &b) const {
    for (int class_id = 0; class_id < DISPATCH_CLASS_MAX; class_id++) {
        if (this->kernel_ids[class_id] != b.kernel_ids[class_id]) {
            return false;
        }
    }

    return true;
}

struct KernelGroupIntHasher {
    std::size_t operator()(const KernelGroupInt &x) const {
        return
            static_cast<size_t>(x.kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value_or(0)) << 0 |
            static_cast<size_t>(x.kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value_or(0)) << 16 |
            static_cast<size_t>(x.kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].value_or(0)) << 32;
    }
};

void detail::ProgramImpl::update_kernel_groups(uint32_t programmable_core_type_index) {
    if (core_to_kernel_group_index_table_[programmable_core_type_index].size() == 0) {
        bool erisc_is_idle = false;

        // Get the extent of the kernels in x, y
        CoreCoord base = {std::numeric_limits<decltype(base.x)>::max(), std::numeric_limits<decltype(base.y)>::max()};
        grid_extent_[programmable_core_type_index] = {0, 0};
        for (auto [id, kernel] : kernels_[programmable_core_type_index]) {
            for (auto core : kernel->logical_cores()) {
                if (core.x > grid_extent_[programmable_core_type_index].x)
                    grid_extent_[programmable_core_type_index].x = core.x;
                if (core.y > grid_extent_[programmable_core_type_index].y)
                    grid_extent_[programmable_core_type_index].y = core.y;
                if (core.x < base.x)
                    base.x = core.x;
                if (core.y < base.y)
                    base.y = core.y;
            }
            erisc_is_idle = kernel->is_idle_eth();
        }
        grid_extent_[programmable_core_type_index].x++;
        grid_extent_[programmable_core_type_index].y++;

        // grid maps cores to sets-of-kernels running on that core
        std::vector<KernelGroupInt> grid;
        grid.resize(grid_extent_[programmable_core_type_index].x * grid_extent_[programmable_core_type_index].y);
        for (auto [id, kernel] : kernels_[programmable_core_type_index]) {
            for (auto core : kernel->logical_cores()) {
                int core_index = core.y * grid_extent_[programmable_core_type_index].x + core.x;
                grid[core_index].valid = true;
                grid[core_index].update(enchantum::cast<dispatch_core_processor_classes>(kernel->dispatch_class()).value(), id);
            }
        }

        // Flip the mapping to get sets-of-kernels to cores
        std::unordered_map<KernelGroupInt, std::set<CoreRange>, KernelGroupIntHasher> map;
        for (auto y = base.y; y < grid_extent_[programmable_core_type_index].y; y++) {
            for (auto x = base.x; x < grid_extent_[programmable_core_type_index].x; x++) {
                int index = y * grid_extent_[programmable_core_type_index].x + x;
                if (grid[index].valid) {
                    std::set<CoreRange> &set = map[grid[index]];
                    set.insert(CoreRange({x, y}, {x, y}));
                }
            }
        }

        // Build the list of KernelGroups with merged core range sets from the
        // mapping of sets-of-kernels to cores
        TT_ASSERT(map.size() < core_to_kernel_group_invalid_index);
        kernel_groups_.reserve(map.size());
        int index = 0;
        core_to_kernel_group_index_table_[programmable_core_type_index].resize(
            grid_extent_[programmable_core_type_index].x * grid_extent_[programmable_core_type_index].y, core_to_kernel_group_invalid_index);
        const auto& hal = MetalContext::instance().hal();
        for (auto &kg_to_cores : map) {
            // Start inclusive, max exclusive
            uint32_t max_local_cb_end_index = 0;
            uint32_t min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS;
            uint32_t local_cb_mask = 0;

            // Map from core X,Y back to the unique KernelGroup
            for (CoreRange range : kg_to_cores.second) {
                bool logged_noncontiguous = false;
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                        core_to_kernel_group_index_table_[programmable_core_type_index][y * grid_extent_[programmable_core_type_index].x + x] = index;

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
                                    std::string kernels;
                                    for (auto id : kg_to_cores.first.kernel_ids) {
                                        if (id.has_value()) {
                                            std::shared_ptr<Kernel> kernel = get_kernel(*id);
                                            if (!kernels.empty()) {
                                                kernels += ", ";
                                            }
                                            kernels += kernel->kernel_source().name();
                                        }
                                    }

                                    static std::mutex m;
                                    std::lock_guard lock(m);
                                    // Keep track of which programs have been logged to avoid spamming the log. This is
                                    // particularly important for mesh devices.
                                    static std::set<std::tuple<uint32_t, uint32_t, std::string>> logged;
                                    auto cb_tuple = std::make_tuple(non_contiguous_cbs, first_unused_index, kernels);

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
                                            kernels);
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
            kernel_groups_[programmable_core_type_index].push_back(std::make_shared<KernelGroup>(
                *this,
                programmable_core_type_index,
                kg_to_cores.first.kernel_ids,
                erisc_is_idle,
                local_cb_mask,
                min_remote_cb_start_index,
                kg_to_cores.second));
            index++;
        }
    }
}

void detail::ProgramImpl::CircularBufferAllocator::mark_address(
    uint64_t address, uint64_t size, uint64_t base_address) {
    if (this->l1_regions.empty()) {
        this->l1_regions.emplace_back(base_address, base_address);
    }
    auto &last_region = this->l1_regions.back();
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

CBHandle detail::ProgramImpl::add_circular_buffer_(const std::shared_ptr<CircularBuffer>& circular_buffer) {
    // Globally allocated circular buffer do not invalidate allocation because their addresses are tracked by memory
    // allocator
    if (not circular_buffer->globally_allocated()) {
        this->invalidate_circular_buffer_allocation();
    } else {
        circular_buffer->assign_global_address();
    }

    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                std::bitset<NUM_CIRCULAR_BUFFERS> &cb_indices = this->per_core_cb_indices_[logical_core];
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
            cb_allocators_.begin(), cb_allocators_.end(), [&core_range](const CircularBufferAllocator &cb_allocator) {
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
    std::shared_ptr<CircularBuffer> circular_buffer = std::make_shared<CircularBuffer>(core_range_set.merge_ranges(), config);
    return add_circular_buffer_(circular_buffer);
}

CBHandle detail::ProgramImpl::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const experimental::GlobalCircularBuffer& global_circular_buffer) {
    TT_FATAL(this->compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", this->id);
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBuffer> circular_buffer =
        std::make_shared<CircularBuffer>(core_range_set.merge_ranges(), config, global_circular_buffer);
    return add_circular_buffer_(circular_buffer);
}

CBHandle Program::add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config) {
    return internal_->add_circular_buffer(core_range_set, config);
}

CBHandle Program::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const experimental::GlobalCircularBuffer& global_circular_buffer) {
    return internal_->add_circular_buffer(core_range_set, config, global_circular_buffer);
}

std::shared_ptr<CircularBuffer> detail::ProgramImpl::get_circular_buffer(CBHandle cb_id) const {
    if (this->circular_buffer_by_id_.find(cb_id) == this->circular_buffer_by_id_.end()) {
        TT_THROW("No circular buffer with id {} exists in Program {}", cb_id, this->id);
    }
    return this->circular_buffer_by_id_.at(cb_id);
}

std::vector<std::shared_ptr<CircularBuffer>> detail::ProgramImpl::circular_buffers_on_core(
    const CoreCoord& core) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (const auto& circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<std::shared_ptr<CircularBuffer>> detail::ProgramImpl::circular_buffers_on_corerange(
    const CoreRange& cr) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
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
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
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
    for (CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        cb_allocator.reset_available_addresses();
    }
    this->local_circular_buffer_allocation_needed_ = true;
}

void Program::invalidate_circular_buffer_allocation() { internal_->invalidate_circular_buffer_allocation(); }

uint32_t Program::get_cb_memory_size() const { return internal_->get_cb_memory_size(); }
uint32_t detail::ProgramImpl::get_cb_memory_size() const {
    uint32_t total_cb_size = 0;
    for (const auto& circular_buffer : this->circular_buffers_) {
        if (circular_buffer->globally_allocated()) {
            continue;
        }
        total_cb_size += circular_buffer->size();
    }
    return total_cb_size;
}
void detail::ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    //ZoneScoped;
    if (not this->local_circular_buffer_allocation_needed_) {
        return;
    }

    uint64_t base_cb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (const auto& circular_buffer : this->circular_buffers_) {
        if (circular_buffer->globally_allocated()) {
            continue;
        }

        uint64_t computed_addr = base_cb_address;
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            // Need the max available address across all cores circular buffer is placed on
            for (const CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
                if (cb_allocator.core_range == core_range) {
                    computed_addr = std::max(computed_addr, cb_allocator.get_cb_region_end());
                    break;
                }
            }
        }
        computed_addr = align(computed_addr, device->allocator()->get_alignment(BufferType::DRAM));
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            for (CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
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
        tt::tt_metal::GraphTracker::instance().track_allocate_cb(circular_buffer->core_ranges(), computed_addr, circular_buffer->size(), circular_buffer->globally_allocated(), device);
        circular_buffer->set_locally_allocated_address(computed_addr);
    }
    this->local_circular_buffer_allocation_needed_ = false;
}

void Program::allocate_circular_buffers(const IDevice* device) { internal_->allocate_circular_buffers(device); }

void detail::ProgramImpl::validate_circular_buffer_region(const IDevice* device) {
    //ZoneScoped;

    // TODO: Circular buffer allocation and validation could be better optimized by determining usage per sub-device
    std::optional<DeviceAddr> lowest_address = device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();

    for (const CircularBufferAllocator &cb_allocator : this->cb_allocators_) {
        if (cb_allocator.l1_regions.empty()) {
            continue;
        }
        uint64_t cb_region_end = cb_allocator.l1_regions.back().second; //cb_allocator.get_cb_region_end();
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

size_t detail::ProgramImpl::num_semaphores() const { return semaphores_.size(); }

size_t Program::num_semaphores() const { return internal_->num_semaphores(); }

void detail::ProgramImpl::init_semaphores(
    const IDevice& device, const CoreCoord& logical_core, uint32_t programmable_core_type_index) const {
    uint64_t kernel_config_base =
        MetalContext::instance().hal().get_dev_addr(programmable_core_type_index, HalL1MemAddrType::KERNEL_CONFIG);
    uint64_t addr = kernel_config_base + this->program_configs_[programmable_core_type_index].sem_offset;
    CoreType core_type = MetalContext::instance().hal().get_core_type(programmable_core_type_index);
    auto semaphores_on_core = this->semaphores_on_core(logical_core, core_type);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(
            device.id(),
            device.virtual_core_from_logical_core(logical_core, core_type),
            std::vector{semaphore.get().initial_value()},
            addr + semaphore.get().offset());
    }
}

void Program::init_semaphores(const IDevice &device, const CoreCoord &logical_core, uint32_t programmable_core_type_index) const {
    internal_->init_semaphores(device, logical_core, programmable_core_type_index);
}

void detail::ProgramImpl::add_semaphore(
    const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    TT_FATAL(this->compiled_.empty(), "Cannot add semaphore to an already compiled program {}", this->id);
    semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, core_type));
}

void Program::add_semaphore(const CoreRangeSet &crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    internal_->add_semaphore(crs, semaphore_id, init_value, core_type);
}

std::vector<std::vector<CoreCoord>> detail::ProgramImpl::logical_cores() const {
    std::vector<std::vector<CoreCoord>> cores_in_program;
    std::vector<std::set<CoreCoord>> unique_cores;
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < kernels_.size(); programmable_core_type_index++) {
        auto &kernels = this->kernels_[programmable_core_type_index];
        cores_in_program.push_back({});
        unique_cores.push_back({});
        for (auto [id, kernel] : kernels) {
            for (auto core : kernel->logical_cores()) {
                if (unique_cores[programmable_core_type_index].find(core) != unique_cores[programmable_core_type_index].end()) {
                    continue;
                }
                unique_cores[programmable_core_type_index].insert(core);
                cores_in_program[programmable_core_type_index].push_back(core);
            }
        }
    }
    return cores_in_program;
}

std::vector<std::vector<CoreCoord>> Program::logical_cores() const { return internal_->logical_cores(); }

void detail::ProgramImpl::set_remote_circular_buffer_init(const std::shared_ptr<Kernel>& kernel) const {
    const auto& kernel_defines = kernel->defines();
    const std::string reserved_defines[] = {"ALIGN_LOCAL_CBS_TO_REMOTE_CBS"};
    for (const auto& str : reserved_defines) {
        TT_FATAL(
            kernel_defines.find(str) == kernel_defines.end(), "{} is a reserved define and can't be manually set", str);
    }
    std::string align_code = "";
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
    //ZoneScoped;
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
    //ZoneScoped;
    for (const auto &logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto &circular_buffer : cbs_on_core) {
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
    auto extract_dst_noc_unicast_info =
        [&device](const auto &ranges, const CoreType core_type) -> std::vector<std::pair<transfer_info_cores, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info;
        for (const CoreRange &core_range : ranges) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord virtual_coord = device->virtual_core_from_logical_core(CoreCoord({x, y}), core_type);
                    dst_noc_unicast_info.push_back(std::make_pair(virtual_coord, /*num_mcast_dests=*/0));
                }
            }
        }
        return dst_noc_unicast_info;
    };

    // Circular Buffer Configs handled in EnqueueProgram

    // Assume here and in command queue that kg_buffers is populated with multicast buffers first then unicast buffers
    // Program Binaries and Go Signals
    // TODO: cleanup put the WORKERS and ETH logic together..

    // All program binaries will be packed into a single buffer in memory
    std::vector<uint32_t> binaries_data;
    // Map is used to look up transfer info by kernel id when we populate data ordered by core groups
    std::unordered_map<KernelHandle, kernel_bins_transfer_info> kernel_transfer_info;
    // This is generic for workers and eth cores
    for (const auto &kernels : this->kernels_) {
        for (const auto &[kernel_id, kernel] : kernels) {
            std::vector<RISCV> sub_kernels;
            if (kernel->processor() == RISCV::COMPUTE) {
                sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
            } else {
                sub_kernels = {kernel->processor()};
            }
            const auto& binaries = KernelImpl::from(*kernel).binaries(
                BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key);
            std::vector<uint32_t> dst_base_addrs;
            std::vector<uint32_t> page_offsets;
            std::vector<uint32_t> lengths;
            std::vector<RISCV> riscvs;
            uint32_t transfer_info_index = 0;

            for (size_t sub_kernel_index = 0; sub_kernel_index < binaries.size(); ++sub_kernel_index) {
                const ll_api::memory& kernel_bin = *binaries[sub_kernel_index];

                // TODO: Pack erisc spans too, and then everthing is
                // one span
                uint32_t num_spans = kernel_bin.num_spans();
                dst_base_addrs.resize(dst_base_addrs.size() + num_spans);
                page_offsets.resize(page_offsets.size() + num_spans);
                lengths.resize(lengths.size() + num_spans);
                riscvs.resize(riscvs.size() + num_spans);

                kernel_bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr,
                                             uint64_t dst,
                                             uint32_t len) {
                    // Set dst for eth kernels until they move to ring buffer
                    dst_base_addrs[transfer_info_index] = dst;
                    page_offsets[transfer_info_index] =
                        binaries_data.size() * sizeof(uint32_t) / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                    lengths[transfer_info_index] = len * sizeof(uint32_t);
                    riscvs[transfer_info_index] = sub_kernels[sub_kernel_index];

                    binaries_data.insert(binaries_data.end(), mem_ptr, mem_ptr + len);
                    binaries_data.resize(
                        tt::align(binaries_data.size(), HostMemDeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t)), 0);
                    transfer_info_index++;
                });
            }

            kernel_bins_transfer_info kb_transfer_info = {
                .dst_base_addrs = dst_base_addrs, .page_offsets = page_offsets, .lengths = lengths, .riscvs = riscvs};
            kernel_transfer_info.insert({kernel_id, kb_transfer_info});
        }
    }

    if (binaries_data.size() > 0) {
        this->program_transfer_info.binary_data = binaries_data;
    }

    std::uint32_t num_active_cores = 0;
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        for (const auto& kernel_group : this->get_kernel_groups(index)) {
            // TODO: add a bit in the hal that says if this core type is unicast/multicast
            if (core_type == CoreType::WORKER) {
                std::vector<multicast_transfer_info> dst_noc_multicast_info =
                    extract_dst_noc_multicast_info(device, kernel_group->core_ranges.ranges(), core_type);
                std::vector<KernelHandle> kernel_ids;
                for (int dispatch_class = 0; dispatch_class < kernel_group->kernel_ids.size(); dispatch_class++) {
                    auto &optional_id = kernel_group->kernel_ids[dispatch_class];
                    if (optional_id) {
                        KernelHandle device_local_kernel_id = program_dispatch::get_device_local_kernel_handle(optional_id.value());
                        kernel_ids.push_back(device_local_kernel_id);
                        int proc_sub_class = 0;
                        for (uint32_t& dst_addr : kernel_transfer_info.at(device_local_kernel_id).dst_base_addrs) {
                            // TODO: ditch this w/ linear writes based on program config kernel_text_offset and size
                            dst_addr = kernel_group->kernel_text_offsets[dispatch_class + proc_sub_class];
                            proc_sub_class++;
                        }
                    }
                }

                for (const auto& transfer_info : dst_noc_multicast_info) {
                    for (const auto &kernel_id : kernel_ids) {
                        this->program_transfer_info.kernel_bins.emplace_back(
                            transfer_info.cores, transfer_info.num_dests, kernel_transfer_info.at(kernel_id));
                    }
                }
            } else {
                TT_ASSERT(core_type == CoreType::ETH);
                std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                    extract_dst_noc_unicast_info(kernel_group->core_ranges.ranges(), core_type);

                std::vector<KernelHandle> kernel_ids;
                if (kernel_group->kernel_ids[DISPATCH_CLASS_ETH_DM0]) {
                    KernelHandle device_local_kernel_id = program_dispatch::get_device_local_kernel_handle(kernel_group->kernel_ids[DISPATCH_CLASS_ETH_DM0].value());
                    kernel_ids.push_back(device_local_kernel_id);
                }

                for (const auto &[cores, num_mcast_dsts] : dst_noc_unicast_info) {
                    for (const auto &kernel_id : kernel_ids) {
                        this->program_transfer_info.kernel_bins.emplace_back(
                            cores, num_mcast_dsts, kernel_transfer_info.at(kernel_id));
                    }
                }
            }
        }
        num_active_cores += this->logical_cores()[index].size();
    }

    this->program_transfer_info.num_active_cores = num_active_cores;

    return;
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
            for (uint32_t sem_type_index = 0; sem_type_index < hal.get_programmable_core_type_count();
                 sem_type_index++) {
                kg->launch_msg.kernel_config.sem_offset[sem_type_index] =
                    this->program_configs_[sem_type_index].sem_offset;
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
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr || sub_device_manager_id == device->get_default_sub_device_manager_id()) {
            // No sub device manager, nothing to validate
            auto [sub_device_ids, _] =
                sub_device_ids_map.insert_or_assign(sub_device_manager_id, std::vector<SubDeviceId>{SubDeviceId{0}});
            return sub_device_ids->second;
        } else {
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
                        const auto& sub_device_cores = device->worker_cores(core_type, SubDeviceId{i});
                        auto intersection = sub_device_cores.intersection(kg->core_ranges);
                        if (intersection.size() > 0) {
                            used_sub_device_ids.insert(SubDeviceId{i});
                            num_intersections += intersection.num_cores();
                        }
                    }
                    num_cores += kg->core_ranges.num_cores();
                }
                TT_FATAL(num_intersections == num_cores,
                         "Kernel group cores do not match sub device cores for programmable core type {}",
                         enchantum::to_string(core_type));
            };
            find_sub_device_ids(HalProgrammableCoreType::TENSIX);
            find_sub_device_ids(HalProgrammableCoreType::ACTIVE_ETH);
            auto [sub_device_ids, _] = sub_device_ids_map.insert_or_assign(
                sub_device_manager_id,
                std::vector<SubDeviceId>(used_sub_device_ids.begin(), used_sub_device_ids.end()));
            return sub_device_ids->second;
        }
    }
    return sub_device_ids->second;
}

void detail::ProgramImpl::allocate_kernel_bin_buf_on_device(IDevice* device) {
    // Allocate the DRAM kernel binary buffer for this program on the specified device, if not previously allocated.
    // We allocate program binaries top down to minimize fragmentation with other buffers in DRAM, which are typically allocated bottom up
    std::size_t binary_data_size_bytes = this->program_transfer_info.binary_data.size() * sizeof(uint32_t);
    if (this->kernels_buffer_.find(device->id()) == this->kernels_buffer_.end() and binary_data_size_bytes) {
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

void Program::generate_dispatch_commands(IDevice* device, bool use_prefetcher_cache) {
    uint64_t command_hash = *device->get_active_sub_device_manager_id();

    uint64_t device_hash = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key;
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        // When coordinate virtualization is not enabled, explicitly encode the device
        // id into the device hash, to always assert on programs being reused across devices.
        device_hash = (device_hash << 32) | (device->id());
    }
    if (!internal_->is_cached()) {
        internal_->set_cached(device_hash);
    } else {
        TT_FATAL(
            *internal_->get_cached() == device_hash,
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
            program_command_sequence, impl(), device, sub_device_id, use_prefetcher_cache);

        program_command_sequence.kernel_bins_sizeB = this->impl().kernel_bins_sizeB;
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

    uint64_t device_hash = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key;
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

void Program::allocate_kernel_bin_buf_on_device(IDevice* device) {
    internal_->allocate_kernel_bin_buf_on_device(device);
}

void detail::ProgramImpl::compile(IDevice* device, bool force_slow_dispatch) {
    //ZoneScoped;
    auto& build_env = BuildEnvManager::get_instance().get_device_build_env(device->build_id());

    if (compiled_.contains(build_env.build_key)) {
        Inspector::program_compile_already_exists(this, device, build_env.build_key);
        return;
    }
    // Clear the determined sub_device_ids when we compile the program for the first time
    // This way, determine_sub_device_ids is forced to recalculate with the finalized information on the used cores
    if (compiled_.empty()) {
        this->sub_device_ids_[device->id()].erase(device->get_active_sub_device_manager_id());
    }

    Inspector::program_compile_started(this, device, build_env.build_key);

    TT_FATAL(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is "
        "dependent on information that is set during device initialization.",
        this->get_id());

    std::vector<std::shared_future<void>> events;

    for (auto & kernels : kernels_) {
        for (auto &[id, kernel] : kernels) {
            validate_kernel_placement(device, force_slow_dispatch, kernel);
            launch_build_step(
                [kernel, device, this, &build_env] {
                    JitBuildOptions build_options(
                        build_env.build_env);
                    KernelImpl::from(*kernel).set_build_options(build_options);
                    if (this->compiled_.empty()) {
                        this->set_remote_circular_buffer_init(kernel);
                    }
                    this->set_cb_data_fmt(kernel->logical_coreranges(), build_options);
                    this->set_cb_tile_dims(kernel->logical_coreranges(), build_options);

                    auto kernel_hash = KernelCompileHash(
                        kernel,
                        build_options,
                        build_env.build_key);

                    const std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash) + "/";
                    kernel->set_full_name(kernel_path_suffix);
                    build_options.set_name(kernel_path_suffix);

                    KernelImpl::from(*kernel).register_kernel_elf_paths_with_watcher(*device);

                    if (enable_persistent_kernel_cache && KernelImpl::from(*kernel).binaries_exist_on_disk(device)) {
                        if (not detail::HashLookup::inst().exists(kernel_hash)) {
                            detail::HashLookup::inst().add(kernel_hash);
                            detail::HashLookup::inst().add_generated_bin(kernel_hash);
                        }
                    } else if (detail::HashLookup::inst().add(kernel_hash)) {
                        GenerateBinaries(device, build_options, kernel);
                        detail::HashLookup::inst().add_generated_bin(kernel_hash);
                    }
                    detail::HashLookup::inst().wait_for_bin_generated(kernel_hash);

                    Inspector::program_kernel_compile_finished(this, device, kernel, build_options);
                },
                events);
        }
    }
    sync_build_steps(events);

    for (auto &kernels : kernels_) {
        for (auto &[id, kernel] : kernels) {
            launch_build_step([kernel, device] { KernelImpl::from(*kernel).read_binaries(device); }, events);
        }
    }
    sync_build_steps(events);
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(get_id(), device);
    }

    compiled_.insert(build_env.build_key);

    Inspector::program_compile_finished(this, device, build_env.build_key);
}

void Program::compile(IDevice* device, bool force_slow_dispatch) { internal_->compile(device, force_slow_dispatch); }

void detail::ProgramImpl::set_runtime_id(uint64_t id) { this->runtime_id = id; }

void Program::set_runtime_id(uint64_t id) { internal_->set_runtime_id(id); }

uint32_t Program::get_sem_base_addr(IDevice* device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type = ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(impl(), device, programmable_core_type);
    return base_addr + impl()
                           .get_program_config(
                               MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type))
                           .sem_offset;
}

uint32_t Program::get_cb_base_addr(IDevice* device, CoreCoord /*logical_core*/, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type = ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(impl(), device, programmable_core_type);
    return base_addr + impl()
                           .get_program_config(
                               MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type))
                           .cb_offset;
}

void detail::ProgramImpl::set_last_used_command_queue_for_testing(CommandQueue* queue) {
    this->last_used_command_queue_for_testing = queue;
}

CommandQueue* detail::ProgramImpl::get_last_used_command_queue() const {
    return this->last_used_command_queue_for_testing;
}

void Program::set_last_used_command_queue_for_testing(CommandQueue* queue) {
    internal_->set_last_used_command_queue_for_testing(queue);
}

CommandQueue* Program::get_last_used_command_queue() const { return internal_->get_last_used_command_queue(); }

uint32_t detail::ProgramImpl::get_sem_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const {
    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].sem_size;
}

uint32_t Program::get_sem_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const {
    return internal_->get_sem_size(device, logical_core, core_type);
}

uint32_t detail::ProgramImpl::get_cb_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const {
    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);

    return this->program_configs_[index].cb_size;
}

uint32_t Program::get_cb_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const {
    return internal_->get_cb_size(device, logical_core, core_type);
}

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool detail::ProgramImpl::runs_on_noc_unicast_only_cores() {
    return (
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
        not this->get_kernel_groups(MetalContext::instance().hal().get_programmable_core_type_index(
                                        HalProgrammableCoreType::ACTIVE_ETH))
                .empty());
}

bool Program::runs_on_noc_unicast_only_cores() { return internal_->runs_on_noc_unicast_only_cores(); }

// TODO: Too low level for program.cpp. Move this to HAL, once we have support.
bool detail::ProgramImpl::runs_on_noc_multicast_only_cores() {
    return (
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX) != -1 and
        not this->get_kernel_groups(
                    MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX))
                .empty());
}

bool Program::runs_on_noc_multicast_only_cores() { return internal_->runs_on_noc_multicast_only_cores(); }

bool detail::ProgramImpl::kernel_binary_always_stored_in_ringbuffer() {
    // Active ethernet cores use a fixed address for the kernel binary, because they don't have enough memory to have
    // that big of a ringbuffer.
    return !(
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1 and
        not this->get_kernel_groups(MetalContext::instance().hal().get_programmable_core_type_index(
                                        HalProgrammableCoreType::ACTIVE_ETH))
                .empty());
}

bool Program::kernel_binary_always_stored_in_ringbuffer() {
    return internal_->kernel_binary_always_stored_in_ringbuffer();
}

Program::Program(Program &&other) noexcept = default;

Program& Program::operator=(Program &&other) noexcept = default;

Program::~Program() noexcept = default;

uint64_t detail::ProgramImpl::get_id() const { return this->id; }

uint64_t Program::get_id() const { return internal_->get_id(); }

uint64_t detail::ProgramImpl::get_runtime_id() const { return this->runtime_id; }

uint64_t Program::get_runtime_id() const { return internal_->get_runtime_id(); }

size_t detail::ProgramImpl::num_kernels() const {
    size_t count = 0;
    for (const auto& kernels : kernels_) {
        count += kernels.size();
    }
    return count;
}

size_t Program::num_kernels() const { return internal_->num_kernels(); }

const std::vector<std::shared_ptr<CircularBuffer>>& detail::ProgramImpl::circular_buffers() const {
    return circular_buffers_;
}

const std::vector<std::shared_ptr<CircularBuffer>>& Program::circular_buffers() const {
    return internal_->circular_buffers();
}

const std::vector<Semaphore>& detail::ProgramImpl::semaphores() const { return semaphores_; }

const std::vector<Semaphore>& Program::semaphores() const { return internal_->semaphores(); }

void detail::ProgramImpl::add_buffer(std::shared_ptr<Buffer> buf) { owned_buffer_pool.push_back(std::move(buf)); }

void Program::add_buffer(std::shared_ptr<Buffer> buf) { internal_->add_buffer(std::move(buf)); }

void detail::ProgramImpl::release_buffers() { owned_buffer_pool = {}; }

void Program::release_buffers() { internal_->release_buffers(); }

std::vector<std::reference_wrapper<const Semaphore>> detail::ProgramImpl::semaphores_on_core(
    const CoreCoord& core, CoreType core_type) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    for (const Semaphore &s : this->semaphores_) {
        if (s.initialized_on_logical_core(core) && s.core_type() == core_type) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

bool detail::ProgramImpl::is_finalized() const { return this->finalized_; }
void detail::ProgramImpl::set_finalized() { this->finalized_ = true; }

bool Program::is_finalized() const { return internal_->is_finalized(); }

ProgramBinaryStatus Program::get_program_binary_status(std::size_t device_id) const {
    return internal_->get_program_binary_status(device_id);
}
void Program::set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status) {
    internal_->set_program_binary_status(device_id, status);
}
void detail::ProgramImpl::set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status) {
    Inspector::program_set_binary_status(this, device_id, status);
    this->binaries_on_device_[device_id] = status;
}

const std::vector<SubDeviceId>& Program::determine_sub_device_ids(const IDevice* device) {
    return internal_->determine_sub_device_ids(device);
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

void Program::set_kernels_bin_buffer(const std::shared_ptr<Buffer>& buffer) {
    internal_->kernels_buffer_.insert({buffer->device()->id(), buffer});
}

std::unordered_map<uint64_t, ProgramCommandSequence> &Program::get_cached_program_command_sequences() noexcept {
    return internal_->cached_program_command_sequences_;
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

void Program::finalize_offsets(IDevice* device) { internal_->finalize_offsets(device); }

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
    TT_FATAL(!program_device_map_.contains(device), "Program already exists in the compile group.");
    program_device_map_[device] = std::move(program);
}

void detail::ProgramCompileGroup::compile_all(bool force_slow_dispatch) {
    std::vector<std::shared_future<void>> events;
    for (auto& [device, program] : program_device_map_) {
        auto pgm = program.get();
        launch_build_step([device, pgm, force_slow_dispatch]() { pgm->compile(device, force_slow_dispatch); }, events);
    }
    sync_build_steps(events);
}

void detail::ProgramCompileGroup::write_runtime_args(bool force_slow_dispatch) {
    for (auto& [device, program] : program_device_map_) {
        detail::WriteRuntimeArgsToDevice(device, *program, force_slow_dispatch);
    }
}

std::unique_ptr<Program> detail::ProgramCompileGroup::remove_program(tt::tt_metal::IDevice* device) {
    TT_FATAL(program_device_map_.contains(device), "Program not found in the compile group.");
    std::unique_ptr<Program> program = std::move(program_device_map_[device]);
    program_device_map_.erase(device);
    return program;
}

void detail::ProgramCompileGroup::clear() { program_device_map_.clear(); }

bool detail::ProgramCompileGroup::contains(tt::tt_metal::IDevice* device) {
    return program_device_map_.contains(device);
}

}  // namespace tt::tt_metal
