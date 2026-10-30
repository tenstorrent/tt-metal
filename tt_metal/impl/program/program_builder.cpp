// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_builder.hpp"

#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_allocation_support.hpp"
#include "impl/program/program_impl.hpp"
#include <enchantum/enchantum.hpp>
#include <experimental/fabric/control_plane.hpp>
#include <global_circular_buffer.hpp>
#include <global_semaphore.hpp>
#include "impl/buffers/global_semaphore_impl.hpp"
#include "device/device_manager.hpp"
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/enum.hpp>

namespace tt::tt_metal::detail {

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

struct DataMovementConfigStatus {
    bool riscv0_in_use;
    bool riscv1_in_use;
    bool noc0_in_use;
    bool noc1_in_use;
};

DataMovementConfigStatus CheckDataMovementConfig(
    const HalProgrammableCoreType& programmable_core, ProgramImpl& program, const CoreRangeSet& core_ranges) {
    DataMovementConfigStatus data_movement_config_status{
        .riscv0_in_use = false, .riscv1_in_use = false, .noc0_in_use = false, .noc1_in_use = false};

    auto set_global_and_local_noc_usage =
        [&](const std::shared_ptr<Kernel>& kernel, bool& local_noc0_usage, bool& local_noc1_usage) {
            int noc_value;
            switch (programmable_core) {
                case HalProgrammableCoreType::TENSIX:
                    noc_value = enchantum::to_underlying(std::get<DataMovementConfig>(kernel->config()).noc);
                    break;
                case HalProgrammableCoreType::ACTIVE_ETH:
                case HalProgrammableCoreType::IDLE_ETH:
                    noc_value = enchantum::to_underlying(std::get<EthernetConfig>(kernel->config()).noc);
                    break;
                default:
                    TT_THROW(
                        "Checking NoC and DataMovementProcessor is unsupported for programmable core {}",
                        enchantum::to_string(programmable_core));
            }
            local_noc0_usage = noc_value == 0;
            local_noc1_usage = noc_value == 1;
            data_movement_config_status.noc0_in_use = local_noc0_usage;
            data_movement_config_status.noc1_in_use = local_noc1_usage;
        };

    const auto& hal = MetalContext::instance(program.get_context_id()).hal();
    for (const auto& core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const KernelGroup* kernel_group = program.kernels_on_core(
                    CoreCoord(x, y), hal.get_programmable_core_type_index(programmable_core));
                if (kernel_group != nullptr) {
                    bool local_noc0_in_use = false;
                    bool local_noc1_in_use = false;
                    bool has_dm0 = false;
                    bool has_dm1 = false;
                    for (auto kernel_id : kernel_group->kernel_ids) {
                        const auto kernel = program.get_kernel(kernel_id);
                        if (kernel->get_kernel_processor_class() == HalProcessorClassType::DM) {
                            switch (kernel->get_kernel_processor_type(0)) {
                                case 0:
                                    has_dm0 = true;
                                    data_movement_config_status.riscv0_in_use = true;
                                    set_global_and_local_noc_usage(kernel, local_noc0_in_use, local_noc1_in_use);
                                    break;
                                case 1:
                                    has_dm1 = true;
                                    data_movement_config_status.riscv1_in_use = true;
                                    set_global_and_local_noc_usage(kernel, local_noc0_in_use, local_noc1_in_use);
                                    break;
                                default: TT_THROW("Unknown DataMovementProcessor type"); break;
                            }
                        }
                    }
                    if (has_dm0 and has_dm1) {
                        TT_FATAL(
                            local_noc0_in_use and local_noc1_in_use,
                            "Illegal NOC usage: data movement kernels on logical core {} cannot use the same NOC, "
                            "doing so results in hangs!",
                            CoreCoord(x, y).str());
                    }
                }
            }
        }
    }

    return data_movement_config_status;
}

}  // namespace

Metal1ProgramBuilder::Metal1ProgramBuilder(
    ProgramImpl& owner, uint32_t programmable_core_count, uint32_t max_circular_buffers) :
    owner_(owner), max_circular_buffers_(max_circular_buffers), kernels_(programmable_core_count) {}

KernelHandle Metal1ProgramBuilder::insert_kernel(
    const std::shared_ptr<Kernel>& kernel, uint32_t programmable_core_type_index) {
    // Id is unique across all kernels on all core types
    KernelHandle id = num_kernels();

    auto new_kernel_core_type = kernel->get_kernel_programmable_core_type();
    auto new_kernel_processor_class = kernel->get_kernel_processor_class();

    std::set<CoreCoord> kernel_logical_cores = kernel->logical_cores();
    auto new_kernel_processor_set = get_kernel_processor_set(*kernel);
    for (size_t i = 0; i < num_kernels(); i++) {
        // Note, looks like id is program specific, and increments naturally as kernels are added.
        //  add_kernel -> id = num_kernels -> kernel is inserted -> next num_kernels() increments.
        std::shared_ptr<Kernel> check_kernel = get_kernel(i);
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

    kernels_[programmable_core_type_index].insert({id, kernel});
    return id;
}

std::shared_ptr<Kernel> Metal1ProgramBuilder::get_kernel(KernelHandle kernel_id) const {
    // find coretype based on kernel_id
    for (const auto& kernels : kernels_) {
        if (kernels.contains(kernel_id)) {
            return kernels.at(kernel_id);
        }
    }

    TT_ASSERT(false, "Did not find kernel id across all core types!");
    return nullptr;
}

size_t Metal1ProgramBuilder::num_kernels() const {
    size_t count = 0;
    for (const auto& kernels : kernels_) {
        count += kernels.size();
    }
    return count;
}

void Metal1ProgramBuilder::validate_circular_buffer_indices(const CircularBufferImpl& circular_buffer) const {
    // The deserialization constructor sets these sets independently, so local and remote are not
    // guaranteed to be subsets of buffer_indices; all three index the per-core bitsets.
    const CircularBufferConfig& config = circular_buffer.config();
    for (const auto* indices :
         {&config.buffer_indices(), &config.local_buffer_indices(), &config.remote_buffer_indices()}) {
        for (uint32_t buffer_index : *indices) {
            if (buffer_index >= max_circular_buffers_) {
                TT_THROW(
                    "Buffer index ({}) exceeds max number of circular buffers per core ({})",
                    buffer_index,
                    max_circular_buffers_);
            }
        }
    }
}

CBHandle Metal1ProgramBuilder::register_circular_buffer(const std::shared_ptr<CircularBufferImpl>& circular_buffer) {
    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const CoreRange& core_range : circular_buffer->core_ranges().ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                std::bitset<NUM_CIRCULAR_BUFFERS>& cb_indices = per_core_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& local_cb_indices = per_core_local_cb_indices_[logical_core];
                std::bitset<NUM_CIRCULAR_BUFFERS>& remote_cb_indices = per_core_remote_cb_indices_[logical_core];
                auto add_buffer_indices = [&cb_indices](
                                              const std::unordered_set<uint8_t>& buffer_indices,
                                              std::bitset<NUM_CIRCULAR_BUFFERS>& target_cb_indices) {
                    for (uint32_t buffer_index : buffer_indices) {
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
    }

    circular_buffers_.push_back(circular_buffer);
    circular_buffer_by_id_.insert({circular_buffer->id(), circular_buffer});
    return circular_buffer->id();
}

void Metal1ProgramBuilder::validate_semaphore_id(
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

void Metal1ProgramBuilder::insert_semaphore(
    const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, uint32_t l1_alignment, CoreType core_type) {
    validate_semaphore_id(crs, semaphore_id, core_type);
    semaphores_.emplace_back(Semaphore(crs, semaphore_id, init_value, l1_alignment, core_type));
}

uint32_t Metal1ProgramBuilder::allocate_semaphore_id(const CoreRangeSet& crs, CoreType core_type) const {
    std::optional<uint32_t> semaphore_id;
    std::bitset<NUM_SEMAPHORES> used_semaphore_ids;
    for (const auto& core_range : crs.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                auto existing = semaphores_on_core(logical_core, core_type);
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
    return *semaphore_id;
}

std::vector<std::reference_wrapper<const Semaphore>> Metal1ProgramBuilder::semaphores_on_core(
    const CoreCoord& core, CoreType core_type) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    semaphores.reserve(semaphores_.size());
    for (const Semaphore& s : semaphores_) {
        if (s.initialized_on_logical_core(core) && s.core_type() == core_type) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

KernelHandle Metal1ProgramBuilder::add_kernel(
    const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& programmable_core_type) {
    TT_FATAL(owner_.compiled_.empty(), "Cannot add kernel to an already compiled program {}", owner_.id);

    // Metal 2.0 kernels (with named bindings, e.g. dfb::/tensor::/args::) are only legal on Metal 2.0 Programs
    if (kernel->is_metal2_kernel()) {
        TT_FATAL(
            owner_.created_from_spec_,
            "Internal error: Metal 2.0 named bindings (dfb::/sem::/tensor::/args::) are only valid in a "
            "Metal 2.0 Program (created from a ProgramSpec).");
    }

    uint32_t index =
        MetalContext::instance(owner_.context_id_).hal().get_programmable_core_type_index(programmable_core_type);
    KernelHandle kernel_handle = insert_kernel(kernel, index);

    if (!kernel->scratchpad_binding_handles().empty()) {
        register_ephemeral_allocation(owner_.dfb_allocation_metadata_, kernel->core_range_set());
    }

    owner_.kernel_groups_[index].resize(0);
    owner_.core_to_kernel_group_index_table_[index].clear();
    return kernel_handle;
}


CBHandle Metal1ProgramBuilder::add_circular_buffer_(const std::shared_ptr<CircularBufferImpl>& circular_buffer) {
    // Metal 2.0 programs use DataflowBuffers, never legacy circular buffers.
    TT_FATAL(
        !owner_.created_from_spec_,
        "Cannot add a legacy circular buffer to a Metal 2.0 Program; "
        "Metal 2.0 Programs use DataflowBuffers, and cannot be modified after construction.");

    validate_circular_buffer_indices(*circular_buffer);

    // Globally allocated circular buffer do not invalidate allocation because their addresses are tracked by memory
    // allocator
    if (not circular_buffer->globally_allocated()) {
        owner_.invalidate_circular_buffer_allocation();
    } else {
        circular_buffer->assign_global_address();
        // invalidate_circular_buffer_allocation() would have done this; adding a buffer still means
        // the program has to be laid out again.
        owner_.compile_and_allocate_needed_ = true;
    }

    CBHandle handle = register_circular_buffer(circular_buffer);

    // A globally allocated CB does need l1 allocation
    if (!circular_buffer->globally_allocated()) {
        register_ephemeral_allocation(owner_.cb_allocation_metadata_, circular_buffer->core_ranges());
    }
    return handle;
}

CBHandle Metal1ProgramBuilder::add_circular_buffer(
    const CoreRangeSet& core_range_set, const CircularBufferConfig& config) {
    TT_FATAL(owner_.compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", owner_.id);
    TT_FATAL(
        owner_.dataflow_buffers_.empty(), "Cannot add circular buffer to a program that already has dataflow buffers");
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBufferImpl> circular_buffer =
        std::make_shared<CircularBufferImpl>(core_range_set.merge_ranges(), config);
    return add_circular_buffer_(circular_buffer);
}

CBHandle Metal1ProgramBuilder::add_circular_buffer(
    const CoreRangeSet& core_range_set,
    const CircularBufferConfig& config,
    const experimental::GlobalCircularBuffer& global_circular_buffer) {
    TT_FATAL(owner_.compiled_.empty(), "Cannot add circular buffer to an already compiled program {}", owner_.id);
    TT_FATAL(
        owner_.dataflow_buffers_.empty(), "Cannot add circular buffer to a program that already has dataflow buffers");
    TT_FATAL(
        owner_.per_core_cross_node_dfbs_.empty(),
        "Cannot add a GlobalCircularBuffer to a program that already has CrossNodeDFB participants. "
        "GlobalCircularBuffer and CrossNodeDFB are mutually exclusive within a program.");
    TT_FATAL(
        owner_.per_core_prefetcher_pipes_.empty(),
        "Cannot add a GlobalCircularBuffer to a program that already has PrefetcherPipe attachments. "
        "GlobalCircularBuffer and PrefetcherPipe are mutually exclusive within a program.");
    // Merge ranges to reduce the number of multicasts needed to initialize CBs.
    std::shared_ptr<CircularBufferImpl> circular_buffer =
        std::make_shared<CircularBufferImpl>(core_range_set.merge_ranges(), config, global_circular_buffer);
    return add_circular_buffer_(circular_buffer);
}


std::shared_ptr<CircularBufferImpl> Metal1ProgramBuilder::get_circular_buffer(CBHandle cb_id) const {
    if (!circular_buffer_by_id().contains(cb_id)) {
        TT_THROW("No circular buffer with id {} exists in Program {}", cb_id, owner_.id);
    }
    return circular_buffer_by_id().at(cb_id);
}


void Metal1ProgramBuilder::add_semaphore(
    const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type) {
    TT_FATAL(owner_.compiled_.empty(), "Cannot add semaphore to an already compiled program {}", owner_.id);
    const uint32_t l1_alignment = MetalContext::instance(owner_.context_id_).hal().get_alignment(HalMemType::L1);
    insert_semaphore(crs, semaphore_id, init_value, l1_alignment, core_type);
}

uint32_t Metal1ProgramBuilder::create_semaphore(const CoreRangeSet& crs, uint32_t initial_value, CoreType core_type) {
    TT_FATAL(!crs.ranges().empty(), "Expecting a non-empty CoreRangeSet!");
    TT_FATAL(
        MetalContext::instance(owner_.context_id_).is_coord_in_range(crs.ranges().back().end_coord, core_type),
        "Coordinates out of range");

    uint32_t semaphore_id = allocate_semaphore_id(crs, core_type);
    add_semaphore(crs, semaphore_id, initial_value, core_type);
    return semaphore_id;
}

KernelHandle Metal1ProgramBuilder::create_data_movement_kernel(
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const DataMovementConfig& config) {
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(HalProgrammableCoreType::TENSIX, owner_, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    std::string kernel_name;
    if (kernel_src.source_type_ == KernelSource::FILE_PATH) {
        kernel_name = kernel_src.source_;
    } else {
        TT_FATAL(kernel_src.source_type_ == KernelSource::SOURCE_CODE, "Unsupported kernel source type!");
        kernel_name = "kernel";
    }

    TT_FATAL(
        !(are_both_riscv_in_use),
        "DataMovementKernel creation failure: Cannot create data movement kernel for {} across specified "
        "cores because both data movement processors are in use!",
        kernel_name);
    TT_FATAL(
        !(are_both_noc_in_use),
        "DataMovementKernel creation failure: Cannot create data movement kernels for {} across specified "
        "cores because both NOCs are in use!",
        kernel_name);

    TT_FATAL(
        config.processor == DataMovementProcessor::RISCV_0 || config.processor == DataMovementProcessor::RISCV_1,
        "DataMovementKernel creation failure: Data movement kernels can only be created on DM0 or DM1 processors.");

    const ContextId context_id = owner_.get_context_id();
    std::shared_ptr<Kernel> kernel =
        std::make_shared<DataMovementKernel>(context_id, kernel_src, core_range_set, config);

    // Inject all fabric-related defines (routing mode, UDM mode, dynamic header sizes, etc.)
    const auto fabric_defines = MetalContext::instance(context_id).get_control_plane().get_fabric_kernel_defines();
    if (!fabric_defines.empty()) {
        kernel->add_defines(fabric_defines);
    }

    return add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle Metal1ProgramBuilder::create_compute_kernel(
    const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const ComputeConfig& config) {
    std::shared_ptr<Kernel> kernel =
        std::make_shared<ComputeKernel>(owner_.get_context_id(), kernel_src, core_range_set, config);
    return add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle Metal1ProgramBuilder::create_ethernet_kernel(
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const EthernetConfig& config) {
    HalProgrammableCoreType eth_core_type =
        config.eth_mode == Eth::IDLE ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(eth_core_type, owner_, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    TT_FATAL(
        config.processor == DataMovementProcessor::RISCV_0 || config.processor == DataMovementProcessor::RISCV_1,
        "EthernetKernel creation failure: Ethernet kernels can only be created on DM0 or DM1 processors.");

    const ContextId context_id = owner_.get_context_id();
    auto& metal_context = MetalContext::instance(context_id);
    std::shared_ptr<Kernel> kernel = std::make_shared<EthernetKernel>(context_id, kernel_src, core_range_set, config);

    // Inject all fabric-related defines (routing mode, UDM mode, dynamic header sizes, etc.)
    const auto fabric_defines = metal_context.get_control_plane().get_fabric_kernel_defines();
    if (!fabric_defines.empty()) {
        kernel->add_defines(fabric_defines);
    }

    // Disable watcher on ethernet cores if requested
    const auto& rt_options = metal_context.rtoptions();
    if (rt_options.watcher_eth_disabled()) {
        kernel->add_defines({{"FORCE_WATCHER_OFF", "1"}});
    }

    TT_FATAL(
        ttsl::as_underlying_type<DataMovementProcessor>(config.processor) <
            metal_context.hal().get_num_risc_processors(eth_core_type),
        "EthernetKernel creation failure: {} kernel cannot target processor {} because Ethernet core only has {} "
        "processors. "
        "Update DataMovementProcessor in the config.",
        kernel->name(),
        enchantum::to_string(config.processor),
        metal_context.hal().get_num_risc_processors(eth_core_type));
    TT_FATAL(
        !(are_both_riscv_in_use),
        "EthernetKernel creation failure: Cannot create data movement kernel for {} across specified "
        "cores because both data movement processors are in use!",
        kernel->name());
    TT_FATAL(
        !(are_both_noc_in_use),
        "EthernetKernel creation failure: Cannot create data movement kernels for {} across specified "
        "cores because both NOCs are in use!",
        kernel->name());

    //
    // Valid configurations for Blackhole ERISC
    //
    // |                | Valid NOC Configuration     |                             |
    // |----------------|-----------------------------|-----------------------------|
    // | **ERISC Mode** | **Physical ERISC0**         | **Physical ERISC1**         |
    // | Single         | Not enabled for dispatch    | Dedicated NOC1              |
    // | Dual           | Dedicated NOC0, Dynamic NOC | Dedicated NOC1, Dynamic NOC |
    //
    if (!metal_context.hal().get_eth_fw_is_cooperative() && config.eth_mode != Eth::IDLE &&
        config.noc_mode != NOC_MODE::DM_DYNAMIC_NOC) {
        bool is_dual_erisc_mode = metal_context.rtoptions().get_enable_2_erisc_mode();
        bool is_erisc0 = (config.processor == DataMovementProcessor::RISCV_0);
        bool is_erisc1 = (config.processor == DataMovementProcessor::RISCV_1);

        if (is_dual_erisc_mode) {
            // Dual ERISC mode: ERISC0 uses NOC0, ERISC1 uses NOC1 (when in dedicated mode)
            if (is_erisc0) {
                TT_FATAL(
                    config.noc == NOC::NOC_0,
                    "EthernetKernel creation failure: In dual ERISC mode, ERISC0 in dedicated mode must use NOC0. "
                    "Kernel: {}, Current NOC: {}, Required NOC: NOC_0. Use Dynamic NOC mode for flexible routing.",
                    kernel->name(),
                    config.noc);
            } else if (is_erisc1) {
                TT_FATAL(
                    config.noc == NOC::NOC_1,
                    "EthernetKernel creation failure: In dual ERISC mode, ERISC1 in dedicated mode must use NOC1. "
                    "Kernel: {}, Current NOC: {}, Required NOC: NOC_1. Use Dynamic NOC mode for flexible routing.",
                    kernel->name(),
                    config.noc);
            }
        } else {
            // ERISC1 must use NOC1 in dedicated mode
            TT_FATAL(
                config.noc == NOC::NOC_1,
                "EthernetKernel creation failure: In single ERISC mode, ERISC0 must use NOC1. "
                "Kernel: {}, Current NOC: {}, Required NOC: NOC_1.",
                kernel->name(),
                config.noc);
        }
    }

    // Dynamic noc is not supported on single erisc mode
    if (!metal_context.hal().get_eth_fw_is_cooperative() && !metal_context.rtoptions().get_enable_2_erisc_mode()) {
        TT_FATAL(
            config.noc_mode == NOC_MODE::DM_DEDICATED_NOC,
            "EthernetKernel creation failure: Dynamic NOC is not supported on single ERISC mode. "
            "Kernel: {}, Current NOC Mode: {}, Required NOC Mode: DM_DEDICATED_NOC.",
            kernel->name(),
            config.noc_mode);
    }

    if (metal_context.hal().get_eth_fw_is_cooperative()) {
        // Dynamic NOC is not supported with this configuration
        TT_FATAL(
            config.noc_mode != NOC_MODE::DM_DYNAMIC_NOC,
            "EthernetKernel creation failure: Cannot create data movement kernels for {} across specified "
            "cores because NOC Mode {} is not supported on this platform",
            kernel->name(),
            config.noc_mode);
    }
    return add_kernel(kernel, eth_core_type);
}

KernelHandle Metal1ProgramBuilder::create_dram_kernel(
    const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const DramConfig& config) {
    const ContextId context_id = owner_.get_context_id();
    auto& metal_context = MetalContext::instance(context_id);
    TT_FATAL(metal_context.get_cluster().arch() == ARCH::BLACKHOLE, "DramKernel is only supported on Blackhole.");
    TT_FATAL(
        metal_context.hal().has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM programmable cores are not enabled; they auto-enable on Blackhole with firmware >= 19.12.0.0.");
    std::shared_ptr<Kernel> kernel = std::make_shared<DramKernel>(context_id, kernel_src, core_range_set, config);
    return add_kernel(kernel, HalProgrammableCoreType::DRAM);
}

void Metal1ProgramBuilder::update_circular_buffer_total_size(CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBufferImpl> circular_buffer = get_circular_buffer(cb_handle);
    if (not circular_buffer->globally_allocated()) {
        owner_.invalidate_circular_buffer_allocation();
    }
    circular_buffer->set_total_size(total_size);
}

void Metal1ProgramBuilder::update_circular_buffer_page_size(
    CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    get_circular_buffer(cb_handle)->set_page_size(buffer_index, page_size);
}

void Metal1ProgramBuilder::update_dynamic_circular_buffer_address(CBHandle cb_handle, const Buffer& buffer) {
    auto circular_buffer = get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->set_global_buffer(buffer, circular_buffer->size(), circular_buffer->config().address_offset());
}

void Metal1ProgramBuilder::update_dynamic_circular_buffer_address(
    CBHandle cb_handle, const Buffer& buffer, uint32_t address_offset) {
    auto circular_buffer = get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->set_global_buffer(buffer, circular_buffer->size(), address_offset);
}

void Metal1ProgramBuilder::update_dynamic_circular_buffer_address(CBHandle cb_handle, const MeshTensor& tensor) {
    auto circular_buffer = get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->set_global_buffer(
        *tensor.mesh_buffer().get_reference_buffer(),
        circular_buffer->size(),
        circular_buffer->config().address_offset());
}

void Metal1ProgramBuilder::update_dynamic_circular_buffer_address_and_total_size(
    CBHandle cb_handle, const Buffer& buffer, uint32_t total_size) {
    auto circular_buffer = get_circular_buffer(cb_handle);
    circular_buffer->set_global_buffer(buffer, total_size, circular_buffer->config().address_offset());
}

void Metal1ProgramBuilder::update_dynamic_circular_buffer_address_and_total_size(
    CBHandle cb_handle, const MeshTensor& tensor, uint32_t total_size) {
    auto circular_buffer = get_circular_buffer(cb_handle);
    circular_buffer->set_global_buffer(
        *tensor.mesh_buffer().get_reference_buffer(), total_size, circular_buffer->config().address_offset());
}

void Metal1ProgramBuilder::update_dynamic_circular_buffer_address(
    CBHandle cb_handle, const experimental::GlobalCircularBuffer& global_circular_buffer) {
    auto circular_buffer = get_circular_buffer(cb_handle);
    TT_FATAL(circular_buffer->is_global_circular_buffer(), "CircularBuffer must be linked to a GlobalCircularBuffer!");
    circular_buffer->set_global_circular_buffer(global_circular_buffer);
}

uint32_t Metal1ProgramBuilder::create_semaphore(
    const std::variant<CoreRange, CoreRangeSet>& core_spec, uint32_t initial_value, CoreType core_type) {
    CoreRangeSet crs = std::visit(
        ttsl::overloaded{
            [](const CoreRange& c) { return CoreRangeSet(c); },
            [](const CoreRangeSet& c) {
                // Merge ranges to reduce the number of multicasts needed to initialize semaphores.
                return c.merge_ranges();
            },
        },
        core_spec);
    return create_semaphore(crs, initial_value, core_type);
}

GlobalSemaphore Metal1ProgramBuilder::create_global_semaphore(
    distributed::MeshDevice& device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type) {
    return GlobalSemaphore(GlobalSemaphoreImpl(device, std::move(cores), initial_value, buffer_type, /*cq_id=*/0));
}

void Metal1ProgramBuilder::assign_global_buffer(const std::shared_ptr<Buffer>& buffer) {
    MetalContext& metal_ctx = MetalContext::instance(owner_.get_context_id());
    metal_ctx.device_manager()->check_dispatch_mode(metal_ctx.rtoptions().get_fast_dispatch());
    owner_.add_buffer(buffer);
}

}  // namespace tt::tt_metal::detail
