// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_dispatch_utils.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"

namespace tt::tt_metal {
namespace program_utils {

enum DispatchWriteOffsets {
    DISPATCH_WRITE_OFFSET_ZERO = 0,
    DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE = 1,
    DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE = 2,
};

uint32_t configure_rta_offsets_for_kernel_groups(
    uint32_t programmable_core_type_index,
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset) {
    uint32_t processor_classes = hal.get_processor_classes_count(programmable_core_type_index);
    std::vector<uint32_t> max_rtas(processor_classes);
    uint32_t max_unique_rta_size = 0;
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    for (auto& kg : kernel_groups) {
        for (std::size_t dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
            max_rtas[dispatch_class] = 0;
            auto& optional_id = kg->kernel_ids[dispatch_class];
            if (optional_id) {
                auto kernel = kernels.at(optional_id.value());
                for (const CoreRange& core_range : kg->core_ranges.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            CoreCoord core_coord(x, y);
                            max_rtas[dispatch_class] =
                                std::max(max_rtas[dispatch_class], (uint32_t)kernel->runtime_args(core_coord).size());
                        }
                    }
                }
            }
        }
        uint32_t offset = 0;
        for (std::size_t dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
            auto& optional_id = kg->kernel_ids[dispatch_class];
            kg->rta_sizes[dispatch_class] = max_rtas[dispatch_class] * sizeof(uint32_t);
            if (optional_id) {
                auto kernel = kernels.at(optional_id.value());
                kernel->set_runtime_args_count(kg->core_ranges, max_rtas[dispatch_class]);
                kg->launch_msg.kernel_config.rta_offset[dispatch_class].rta_offset = base_offset + offset;
                offset += max_rtas[dispatch_class] * sizeof(uint32_t);
            } else {
                kg->launch_msg.kernel_config.rta_offset[dispatch_class].rta_offset = 0;
            }
        }
        kg->total_rta_size = offset;
        offset = align(offset, l1_alignment);
        max_unique_rta_size = std::max(offset, max_unique_rta_size);
    }
    return max_unique_rta_size;
}

uint32_t configure_crta_offsets_for_kernel_groups(
    uint32_t programmable_core_type_index,
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t crta_base_offset,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_offsets,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_sizes) {
    uint32_t processor_classes = hal.get_processor_classes_count(programmable_core_type_index);
    std::vector<uint32_t> max_crtas(processor_classes);

    for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
        max_crtas[dispatch_class] = 0;
    }
    // Find the max # common RTAs across all kernels for each dispatch class
    for (auto& kernel_info : kernels) {
        auto kernel = kernel_info.second;
        uint32_t dispatch_class = kernel->dispatch_class();
        max_crtas[dispatch_class] = std::max(max_crtas[dispatch_class], (uint32_t)kernel->common_runtime_args().size());
    }

    // Derive crta offsets and sizes per dispatch class
    uint32_t offset = 0;
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
        uint32_t size = max_crtas[dispatch_class] * sizeof(uint32_t);
        crta_offsets[dispatch_class] = crta_base_offset + offset;
        crta_sizes[dispatch_class] = size;
        offset += size;
        offset = align(offset, l1_alignment);
    }
    uint32_t total_crta_size = offset;

    // Set the runtime_args_data sizing info based on the shared max
    for (auto& kernel_info : kernels) {
        auto kernel = kernel_info.second;
        uint32_t dispatch_class = kernel->dispatch_class();
        kernel->set_common_runtime_args_count(max_crtas[dispatch_class]);
    }
    // Set the kernel group common runtime arg offsets use in the launch message
    for (auto& kg : kernel_groups) {
        for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
            kg->launch_msg.kernel_config.rta_offset[dispatch_class].crta_offset = crta_offsets[dispatch_class];
        }
    }
    return total_crta_size;
}

uint32_t finalize_rt_args(
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t programmable_core_type_index,
    uint32_t& rta_offset,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_offsets,
    std::array<uint32_t, DISPATCH_CLASS_MAX>& crta_sizes) {
    CoreType core_type = hal.get_core_type(programmable_core_type_index);
    HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(programmable_core_type_index);

    uint32_t max_unique_rta_size = program_utils::configure_rta_offsets_for_kernel_groups(
        programmable_core_type_index, kernels, kernel_groups, base_offset);
    uint32_t crta_base_offset = base_offset + max_unique_rta_size;
    uint32_t total_crta_size = program_utils::configure_crta_offsets_for_kernel_groups(
        programmable_core_type_index, kernels, kernel_groups, crta_base_offset, crta_offsets, crta_sizes);

    uint32_t offset = max_unique_rta_size + total_crta_size;
    // TODO: this is asserted here as the leveling above can break the limits enforced by the API
    // Once we use a ring buffer, memory space will be dynamic and this assert won't matter
    std::uint32_t l1_kernel_config_size = tt::tt_metal::hal.get_dev_size(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
    TT_FATAL(offset <= l1_kernel_config_size, "offset {} cannot exceed config size {}", offset, l1_kernel_config_size);

    rta_offset = base_offset;
    return offset;
}

uint32_t finalize_sems(
    uint32_t programmable_core_type_index,
    uint32_t sem_base_offset,
    const std::vector<Semaphore>& semaphores,
    uint32_t& semaphore_offset,
    uint32_t& semaphore_size) {
    int max_id = -1;
    CoreType core_type = hal.get_core_type(programmable_core_type_index);
    for (const auto& sem : semaphores) {
        if (sem.core_type() == core_type && (int)sem.id() > max_id) {
            max_id = sem.id();
        }
    }
    uint32_t sem_size = (max_id + 1) * hal.get_alignment(HalMemType::L1);
    semaphore_offset = sem_base_offset;
    semaphore_size = sem_size;
    return sem_base_offset + sem_size;
}

uint32_t finalize_cbs(
    uint32_t programmable_core_type_index,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& cb_offset,
    uint32_t& cb_size,
    uint32_t& local_cb_size) {
    uint32_t max_local_end_index = 0;
    uint32_t min_remote_start_index = NUM_CIRCULAR_BUFFERS;

    for (auto& kg : kernel_groups) {
        max_local_end_index =
            std::max(max_local_end_index, (uint32_t)kg->launch_msg.kernel_config.max_local_cb_end_index);
        min_remote_start_index =
            std::min(min_remote_start_index, (uint32_t)kg->launch_msg.kernel_config.min_remote_cb_start_index);
    }

    local_cb_size = max_local_end_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    uint32_t remote_cb_offset = base_offset + local_cb_size;
    for (auto& kg : kernel_groups) {
        kg->launch_msg.kernel_config.local_cb_offset = base_offset;
        kg->launch_msg.kernel_config.remote_cb_offset = remote_cb_offset;
    }

    uint32_t remote_cb_size = (NUM_CIRCULAR_BUFFERS - min_remote_start_index) *
                              UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    uint32_t total_cb_size = local_cb_size + remote_cb_size;
    cb_offset = base_offset;
    cb_size = total_cb_size;

    return align(base_offset + total_cb_size, hal.get_alignment(HalMemType::L1));
}

uint32_t finalize_kernel_bins(
    Device* device,
    uint32_t programmable_core_type_index,
    const std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& kernel_text_offset,
    uint32_t& kernel_text_size) {
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t max_offset = 0;
    for (auto& kg : kernel_groups) {
        uint32_t offset = base_offset;

        for (int class_id = 0; class_id < DISPATCH_CLASS_MAX; class_id++) {
            auto& optional_id = kg->kernel_ids[class_id];
            if (optional_id) {
                const auto kernel = kernels.at(optional_id.value());
                const std::vector<const ll_api::memory*>& binaries = kernel->binaries(device->build_key());
                // TODO: this is really ugly, save me future-HAL!
                if (programmable_core_type_index ==
                    hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)) {
                    uint32_t binary_packed_size = kernel->get_binary_packed_size(device, 0);

                    if (class_id == DISPATCH_CLASS_TENSIX_DM0) {
                        kg->kernel_bin_sizes[0] = binary_packed_size;
                        kg->kernel_text_offsets[0] = offset;
                        kg->launch_msg.kernel_config.kernel_text_offset[0] = offset;
                        offset += binary_packed_size;
                        offset = align(offset, l1_alignment);
                    } else if (class_id == DISPATCH_CLASS_TENSIX_DM1) {
                        kg->kernel_bin_sizes[1] = binary_packed_size;
                        kg->kernel_text_offsets[1] = offset;
                        kg->launch_msg.kernel_config.kernel_text_offset[1] = offset;
                        offset += binary_packed_size;
                        offset = align(offset, l1_alignment);

                        uint32_t binary_text_size = kernel->get_binary_text_size(device, 0);
                        TT_ASSERT(binary_text_size >> 4 <= std::numeric_limits<uint16_t>::max());
                        kg->launch_msg.kernel_config.ncrisc_kernel_size16 = (binary_text_size + 15) >> 4;
                    } else {
                        constexpr uint32_t max_math_processors_count = 3;
                        for (uint32_t proc_type_index = 0; proc_type_index < max_math_processors_count;
                             proc_type_index++) {
                            uint32_t binary_packed_size = kernel->get_binary_packed_size(device, proc_type_index);
                            kg->kernel_bin_sizes[2 + proc_type_index] = binary_packed_size;
                            kg->kernel_text_offsets[2 + proc_type_index] = offset;
                            kg->launch_msg.kernel_config.kernel_text_offset[2 + proc_type_index] = offset;
                            offset += binary_packed_size;
                            offset = align(offset, l1_alignment);
                        }
                    }
                } else {
                    uint32_t binary_packed_size = kernel->get_binary_packed_size(device, 0);
                    kg->kernel_bin_sizes[class_id] = binary_packed_size;

                    // No kernel config buffer on active eth yet
                    if (hal.get_programmable_core_type(kg->programmable_core_type_index) ==
                        HalProgrammableCoreType::IDLE_ETH) {
                        kg->kernel_text_offsets[class_id] = offset;
                        kg->launch_msg.kernel_config.kernel_text_offset[class_id] = offset;
                        offset += binary_packed_size;
                        offset = align(offset, l1_alignment);
                    } else {
                        kg->kernel_text_offsets[class_id] = binaries[0]->get_text_addr();
                        kg->launch_msg.kernel_config.kernel_text_offset[class_id] = binaries[0]->get_text_addr();
                    }
                }
            }
        }

        max_offset = std::max(offset, max_offset);
    }
    kernel_text_offset = base_offset;
    kernel_text_size = max_offset - base_offset;
    return max_offset;
}

uint32_t get_packed_write_max_unicast_sub_cmds(Device* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}

void insert_empty_program_dispatch_preamble_cmd(ProgramCommandSequence& program_command_sequence) {
    // Initialize an empty preamble command in the Program Dispatch Cmd Sequence, which will be
    // updated with the correct write offsets when the program is enqueued
    uint32_t preamble_cmd_sizeB = hal.get_alignment(HalMemType::HOST);
    program_command_sequence.preamble_command_sequence = HostMemDeviceCommand(preamble_cmd_sizeB);
    program_command_sequence.preamble_command_sequence.add_dispatch_set_write_offsets(0, 0, 0);
}

void insert_stall_cmds(ProgramCommandSequence& program_command_sequence, SubDeviceId sub_device_id, Device* device) {
    // Initialize stall command sequences for this program.
    auto dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    uint32_t dispatch_message_addr =
        dispatch_constants::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE) +
        dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(sub_device_id.to_index());
    uint32_t uncached_stall_cmd_sizeB = hal.get_alignment(HalMemType::HOST) + hal.get_alignment(HalMemType::HOST);
    uint32_t cached_stall_cmd_seqB = hal.get_alignment(HalMemType::HOST);

    program_command_sequence.stall_command_sequences[UncachedStallSequenceIdx] =
        HostMemDeviceCommand(uncached_stall_cmd_sizeB);
    // Empty wait command initialized here. Will get updated when program is enqueued.
    program_command_sequence.stall_command_sequences[UncachedStallSequenceIdx].add_dispatch_wait_with_prefetch_stall(
        true, dispatch_message_addr, 0);
    // Empty wait command initialized here. Will get updated when program is enqueued.
    program_command_sequence.stall_command_sequences[CachedStallSequenceIdx] =
        HostMemDeviceCommand(cached_stall_cmd_seqB);
    program_command_sequence.stall_command_sequences[CachedStallSequenceIdx].add_dispatch_wait(
        false, dispatch_message_addr, 0);
}

template <typename PackedSubCmd>
uint32_t get_max_write_packed_sub_cmds(
    uint32_t data_size, uint32_t max_prefetch_cmd_size, uint32_t packed_write_max_unicast_sub_cmds, bool no_stride) {
    static_assert(
        std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value or
        std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);
    constexpr bool is_unicast = std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value;
    uint32_t sub_cmd_sizeB =
        is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
    // Approximate calculation due to alignment
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    uint32_t max_prefetch_size = max_prefetch_cmd_size - sizeof(CQPrefetchCmd) - hal.get_alignment(HalMemType::HOST) -
                                 sizeof(CQDispatchCmd) - l1_alignment;
    uint32_t max_prefetch_num_packed_cmds =
        no_stride ? (max_prefetch_size - align(data_size * sizeof(uint32_t), l1_alignment)) / sub_cmd_sizeB
                  : max_prefetch_size / (align(data_size * sizeof(uint32_t), l1_alignment) + sub_cmd_sizeB);

    uint32_t packed_write_max_multicast_sub_cmds =
        get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);
    return std::min(
        max_prefetch_num_packed_cmds,
        is_unicast ? packed_write_max_unicast_sub_cmds : packed_write_max_multicast_sub_cmds);
};

template <typename PackedSubCmd>
void generate_runtime_args_cmds(
    std::vector<HostMemDeviceCommand>& runtime_args_command_sequences,
    const uint32_t& l1_arg_base_addr,
    const std::vector<PackedSubCmd>& sub_cmds,
    const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>& rt_data_and_sizes,
    const uint32_t& max_runtime_args_len,
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>>& rt_args_data,
    const uint32_t max_prefetch_command_size,
    const uint32_t packed_write_max_unicast_sub_cmds,
    bool no_stride,
    enum DispatchWriteOffsets write_offset_index) {
    static_assert(
        std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value or
        std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);

    thread_local static auto get_runtime_payload_sizeB =
        [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast, bool no_stride) {
            uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
            uint32_t sub_cmd_sizeB =
                is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB = sizeof(CQDispatchCmd) + align(num_packed_cmds * sub_cmd_sizeB, l1_alignment);
            uint32_t aligned_runtime_data_sizeB =
                (no_stride ? 1 : num_packed_cmds) * align(runtime_args_len * sizeof(uint32_t), l1_alignment);
            return dispatch_cmd_sizeB + aligned_runtime_data_sizeB;
        };
    thread_local static auto get_runtime_args_data_offset =
        [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast) {
            uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
            uint32_t sub_cmd_sizeB =
                is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB = sizeof(CQDispatchCmd) + align(num_packed_cmds * sub_cmd_sizeB, l1_alignment);
            return sizeof(CQPrefetchCmd) + dispatch_cmd_sizeB;
        };

    constexpr bool unicast = std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value;

    uint32_t num_packed_cmds_in_seq = sub_cmds.size();
    uint32_t max_packed_cmds = get_max_write_packed_sub_cmds<PackedSubCmd>(
        max_runtime_args_len, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, no_stride);
    uint32_t offset_idx = 0;
    if (no_stride) {
        TT_FATAL(
            max_packed_cmds >= num_packed_cmds_in_seq,
            "num_packed_cmds_in_seq {} cannot exceed max_packed_cmds {} when no_stride is true",
            num_packed_cmds_in_seq,
            max_packed_cmds);
    }
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    while (num_packed_cmds_in_seq != 0) {
        // Generate the device command
        uint32_t num_packed_cmds = std::min(num_packed_cmds_in_seq, max_packed_cmds);
        uint32_t rt_payload_sizeB =
            get_runtime_payload_sizeB(num_packed_cmds, max_runtime_args_len, unicast, no_stride);
        uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + rt_payload_sizeB, pcie_alignment);
        runtime_args_command_sequences.emplace_back(cmd_sequence_sizeB);
        runtime_args_command_sequences.back().add_dispatch_write_packed<PackedSubCmd>(
            num_packed_cmds,
            l1_arg_base_addr,
            max_runtime_args_len * sizeof(uint32_t),
            rt_payload_sizeB,
            sub_cmds,
            rt_data_and_sizes,
            packed_write_max_unicast_sub_cmds,
            offset_idx,
            no_stride,
            write_offset_index);

        // Update kernel RTA pointers to point into the generated command
        // Future RTA updates through the API will update the command sequence directly
        uint32_t data_offset = (uint32_t)get_runtime_args_data_offset(num_packed_cmds, max_runtime_args_len, unicast);
        const uint32_t data_inc = align(max_runtime_args_len * sizeof(uint32_t), l1_alignment);
        uint32_t num_data_copies = no_stride ? 1 : num_packed_cmds;
        for (uint32_t i = offset_idx; i < offset_idx + num_data_copies; ++i) {
            uint32_t offset = 0;
            for (auto& data : rt_args_data[i]) {
                data.get().rt_args_data =
                    (uint32_t*)((char*)runtime_args_command_sequences.back().data() + data_offset + offset);
                offset += data.get().rt_args_count * sizeof(uint32_t);
            }
            data_offset += data_inc;
        }
        num_packed_cmds_in_seq -= num_packed_cmds;
        offset_idx += num_packed_cmds;
    }
}

void assemble_runtime_args_commands(
    ProgramCommandSequence& program_command_sequence, Program& program, Device* device) {
    static const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(device);
    NOC noc_index = dispatch_downstream_noc;
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();

    // Dispatch Commands to Unicast Unique Runtime Args to Workers
    std::vector<CQDispatchWritePackedUnicastSubCmd> unique_sub_cmds;
    std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>> unique_rt_data_and_sizes;
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>> unique_rt_args_data;
    // Dispatch Commands to Multicast Common Runtime Args to Workers
    std::variant<std::vector<CQDispatchWritePackedMulticastSubCmd>, std::vector<CQDispatchWritePackedUnicastSubCmd>>
        common_sub_cmds;
    std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>> common_rt_data_and_sizes;
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>> common_rt_args_data;

    program_command_sequence.runtime_args_command_sequences = {};
    uint32_t command_count = 0;

    // Unique RTAs
    for (uint32_t programmable_core_type_index = 0;
         programmable_core_type_index < hal.get_programmable_core_type_count();
         programmable_core_type_index++) {
        if (hal.get_programmable_core_type(programmable_core_type_index) == HalProgrammableCoreType::IDLE_ETH) {
            // Fast dispatch not supported on IDLE_ETH yet
            continue;
        }
        for (auto& kg : program.get_kernel_groups(programmable_core_type_index)) {
            if (kg->total_rta_size != 0) {
                uint32_t num_sub_cmds = kg->core_ranges.num_cores();
                uint32_t max_runtime_args_len = kg->total_rta_size / sizeof(uint32_t);
                uint32_t max_packed_cmds = get_max_write_packed_sub_cmds<decltype(unique_sub_cmds)::value_type>(
                    max_runtime_args_len, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, false);
                command_count += div_up(num_sub_cmds, max_packed_cmds);
            }
        }
    }
    // Common RTAs
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        auto programmable_core_type = kernel->get_kernel_programmable_core_type();
        if (programmable_core_type == HalProgrammableCoreType::IDLE_ETH) {
            // Fast dispatch not supported on IDLE_ETH yet
            continue;
        }
        uint32_t programmable_core_type_index = hal.get_programmable_core_type_index(programmable_core_type);
        uint32_t common_size =
            program.get_program_config(programmable_core_type_index).crta_sizes[kernel->dispatch_class()];
        if (common_size != 0) {
            uint32_t max_runtime_args_len = common_size / sizeof(uint32_t);
            const auto& common_rt_args = kernel->common_runtime_args();
            if (common_rt_args.size() > 0) {
                CoreType core_type = hal.get_core_type(programmable_core_type_index);
                if (core_type == CoreType::ETH) {
                    uint32_t num_sub_cmds = kernel->logical_cores().size();
                    uint32_t max_packed_cmds = get_max_write_packed_sub_cmds<CQDispatchWritePackedUnicastSubCmd>(
                        max_runtime_args_len, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, true);
                    command_count += div_up(num_sub_cmds, max_packed_cmds);
                } else {
                    uint32_t num_sub_cmds = kernel->logical_coreranges().size();
                    uint32_t max_packed_cmds = get_max_write_packed_sub_cmds<CQDispatchWritePackedMulticastSubCmd>(
                        max_runtime_args_len, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, true);
                    command_count += div_up(num_sub_cmds, max_packed_cmds);
                }
            }
        }
    }

    program_command_sequence.runtime_args_command_sequences.reserve(command_count);
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        if (hal.get_programmable_core_type(index) == HalProgrammableCoreType::IDLE_ETH) {
            // Fast dispatch not supported on IDLE_ETH yet
            // TODO: can't just loop here as code below confuses ACTIVE/IDLE
            continue;
        }
        CoreType core_type = hal.get_core_type(index);
        uint32_t processor_classes = hal.get_processor_classes_count(index);

        for (auto& kg : program.get_kernel_groups(index)) {
            if (kg->total_rta_size != 0) {
                for (const CoreRange& core_range : kg->core_ranges.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            CoreCoord core_coord(x, y);

                            unique_rt_args_data.resize(unique_rt_args_data.size() + 1);
                            unique_rt_data_and_sizes.resize(unique_rt_data_and_sizes.size() + 1);
                            for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
                                auto& optional_id = kg->kernel_ids[dispatch_class];
                                if (optional_id) {
                                    auto device_local_kernel_handle =
                                        get_device_local_kernel_handle(optional_id.value());
                                    auto kernel = detail::GetKernel(program, device_local_kernel_handle);
                                    if (!kernel->cores_with_runtime_args().empty()) {
                                        const auto& runtime_args_data = kernel->runtime_args(core_coord);
                                        unique_rt_args_data.back().emplace_back(kernel->runtime_args_data(core_coord));
                                        TT_ASSERT(
                                            runtime_args_data.size() * sizeof(uint32_t) <=
                                            kg->rta_sizes[dispatch_class]);
                                        unique_rt_data_and_sizes.back().emplace_back(
                                            runtime_args_data.data(),
                                            runtime_args_data.size() * sizeof(uint32_t),
                                            kg->rta_sizes[dispatch_class]);
                                    }
                                }
                            }
                            CoreCoord virtual_core = device->virtual_core_from_logical_core(core_coord, core_type);
                            unique_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr = device->get_noc_unicast_encoding(noc_index, virtual_core)});
                        }
                    }
                }
                uint32_t rta_offset = program.get_program_config(index).rta_offset;
                generate_runtime_args_cmds(
                    program_command_sequence.runtime_args_command_sequences,
                    rta_offset,
                    unique_sub_cmds,
                    unique_rt_data_and_sizes,
                    kg->total_rta_size / sizeof(uint32_t),
                    unique_rt_args_data,
                    max_prefetch_command_size,
                    packed_write_max_unicast_sub_cmds,
                    false,
                    core_type == CoreType::WORKER ? DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE
                                                  : DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE);
                for (auto& data_per_kernel : unique_rt_data_and_sizes) {
                    for (auto& data_and_sizes : data_per_kernel) {
                        RecordDispatchData(program, DISPATCH_DATA_RTARGS, std::get<1>(data_and_sizes));
                    }
                }
                unique_sub_cmds.clear();
                unique_rt_data_and_sizes.clear();
                unique_rt_args_data.clear();
            }
        }

        for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
            uint32_t common_size = program.get_program_config(index).crta_sizes[dispatch_class];
            if (common_size == 0) {
                continue;
            }
            for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
                auto kernel = detail::GetKernel(program, kernel_id);
                if (kernel->get_kernel_core_type() != core_type) {
                    continue;  // TODO: fixme, need list of kernels by core_typexdispatch_class
                }
                if (kernel->dispatch_class() != dispatch_class) {
                    continue;  // TODO: fixme, need list of kernels by core_typexdispatch_class
                }

                const auto& common_rt_args = kernel->common_runtime_args();
                if (common_rt_args.size() > 0) {
                    common_rt_args_data.resize(common_rt_args_data.size() + 1);
                    common_rt_data_and_sizes.resize(common_rt_data_and_sizes.size() + 1);

                    TT_ASSERT(kernel->common_runtime_args_data().size() * sizeof(uint32_t) == common_size);
                    TT_ASSERT(common_rt_args.size() * sizeof(uint32_t) <= common_size);
                    common_rt_data_and_sizes.back().emplace_back(
                        common_rt_args.data(), common_rt_args.size() * sizeof(uint32_t), common_size);
                    common_rt_args_data.back().emplace_back(kernel->common_runtime_args_data());

                    if (core_type == CoreType::ETH) {
                        common_sub_cmds.emplace<std::vector<CQDispatchWritePackedUnicastSubCmd>>(
                            std::vector<CQDispatchWritePackedUnicastSubCmd>());
                        auto& unicast_sub_cmd =
                            std::get<std::vector<CQDispatchWritePackedUnicastSubCmd>>(common_sub_cmds);
                        unicast_sub_cmd.reserve(kernel->logical_cores().size());
                        for (auto& core_coord : kernel->logical_cores()) {
                            // can make a vector of unicast encodings here
                            CoreCoord virtual_core_coords =
                                device->virtual_core_from_logical_core(core_coord, CoreType::ETH);
                            unicast_sub_cmd.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr = device->get_noc_unicast_encoding(noc_index, virtual_core_coords)});
                        }
                    } else {
                        std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                            device->extract_dst_noc_multicast_info(
                                kernel->logical_coreranges(), core_type);
                        common_sub_cmds.emplace<std::vector<CQDispatchWritePackedMulticastSubCmd>>(
                            std::vector<CQDispatchWritePackedMulticastSubCmd>());
                        auto& multicast_sub_cmd =
                            std::get<std::vector<CQDispatchWritePackedMulticastSubCmd>>(common_sub_cmds);
                        multicast_sub_cmd.reserve(dst_noc_multicast_info.size());
                        for (const auto& mcast_dests : dst_noc_multicast_info) {
                            multicast_sub_cmd.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                                .noc_xy_addr = device->get_noc_multicast_encoding(
                                    noc_index, std::get<CoreRange>(mcast_dests.first)),
                                .num_mcast_dests = mcast_dests.second});
                        }
                    }
                }
            }

            uint32_t crta_offset = program.get_program_config(index).crta_offsets[dispatch_class];

            // Common rtas are always expected to fit in one prefetch cmd
            // TODO: use a linear write instead of a packed-write
            std::visit(
                [&](auto&& sub_cmds) {
                    generate_runtime_args_cmds(
                        program_command_sequence.runtime_args_command_sequences,
                        crta_offset,
                        sub_cmds,
                        common_rt_data_and_sizes,
                        common_size / sizeof(uint32_t),
                        common_rt_args_data,
                        max_prefetch_command_size,
                        packed_write_max_unicast_sub_cmds,
                        true,
                        core_type == CoreType::WORKER ? DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE
                                                      : DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE);
                    sub_cmds.clear();
                },
                common_sub_cmds);

            for (auto& data_per_kernel : common_rt_data_and_sizes) {
                for (auto& data_and_sizes : data_per_kernel) {
                    RecordDispatchData(program, DISPATCH_DATA_RTARGS, std::get<1>(data_and_sizes));
                }
            }
            common_rt_data_and_sizes.clear();
            common_rt_args_data.clear();
        }
    }
    TT_ASSERT(
        command_count >= program_command_sequence.runtime_args_command_sequences.size(),
        "Incorrect number of commands reserved {}, final size {}. Vector reallocation causes cached addresses to be "
        "incorrect.",
        command_count,
        program_command_sequence.runtime_args_command_sequences.size());

    uint32_t runtime_args_fetch_size_bytes = 0;
    for (const auto& cmds : program_command_sequence.runtime_args_command_sequences) {
        // BRISC, NCRISC, TRISC...
        runtime_args_fetch_size_bytes += cmds.size_bytes();
    }
    program_command_sequence.runtime_args_fetch_size_bytes = runtime_args_fetch_size_bytes;
}

template <typename PackedSubCmd>
uint32_t insert_write_packed_payloads(
    const uint32_t num_sub_cmds,
    const uint32_t sub_cmd_sizeB,
    const uint32_t max_prefetch_command_size,
    const uint32_t packed_write_max_unicast_sub_cmds,
    std::vector<std::pair<uint32_t, uint32_t>>& packed_cmd_payloads) {
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t aligned_sub_cmd_sizeB = align(sub_cmd_sizeB, l1_alignment);
    const uint32_t max_packed_sub_cmds_per_cmd = get_max_write_packed_sub_cmds<PackedSubCmd>(
        aligned_sub_cmd_sizeB, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, false);
    uint32_t rem_num_sub_cmds = num_sub_cmds;
    uint32_t cmd_payload_sizeB = 0;
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    while (rem_num_sub_cmds != 0) {
        const uint32_t num_sub_cmds_in_cmd = std::min(max_packed_sub_cmds_per_cmd, rem_num_sub_cmds);
        const uint32_t aligned_data_sizeB = aligned_sub_cmd_sizeB * num_sub_cmds_in_cmd;
        const uint32_t dispatch_cmd_sizeB =
            align(sizeof(CQDispatchCmd) + num_sub_cmds_in_cmd * sizeof(PackedSubCmd), l1_alignment);
        packed_cmd_payloads.emplace_back(num_sub_cmds_in_cmd, dispatch_cmd_sizeB + aligned_data_sizeB);
        cmd_payload_sizeB += align(sizeof(CQPrefetchCmd) + packed_cmd_payloads.back().second, pcie_alignment);
        rem_num_sub_cmds -= num_sub_cmds_in_cmd;
    }
    return cmd_payload_sizeB;
}

void assemble_device_commands(
    ProgramCommandSequence& program_command_sequence, Program& program, Device* device, SubDeviceId sub_device_id) {
    uint32_t cmd_sequence_sizeB = 0;
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    NOC noc_index = dispatch_downstream_noc;
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    static const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(device);
    const auto& program_transfer_info = program.get_program_transfer_info();
    // Multicast Semaphore Cmd
    uint32_t num_multicast_semaphores = program_transfer_info.multicast_semaphores.size();
    std::vector<std::vector<CQDispatchWritePackedMulticastSubCmd>> multicast_sem_sub_cmds(num_multicast_semaphores);
    std::vector<std::vector<std::pair<const void*, uint32_t>>> multicast_sem_data(num_multicast_semaphores);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> multicast_sem_payload(num_multicast_semaphores);
    std::vector<std::pair<uint32_t, uint32_t>> multicast_sem_dst_size;
    multicast_sem_dst_size.reserve(num_multicast_semaphores);

    if (num_multicast_semaphores > 0) {
        uint32_t i = 0;
        for (const auto& [dst, transfer_info_vec] : program_transfer_info.multicast_semaphores) {
            // TODO: loop over things inside transfer_info[i]
            uint32_t write_packed_len = transfer_info_vec[0].data.size();
            multicast_sem_dst_size.emplace_back(std::make_pair(dst, write_packed_len * sizeof(uint32_t)));

            for (const auto& transfer_info : transfer_info_vec) {
                for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                    TT_ASSERT(
                        transfer_info.data.size() == write_packed_len,
                        "Not all data std::vectors in write packed semaphore cmd equal in len");
                    multicast_sem_sub_cmds[i].emplace_back(CQDispatchWritePackedMulticastSubCmd{
                        .noc_xy_addr =
                            device->get_noc_multicast_encoding(noc_index, std::get<CoreRange>(dst_noc_info.first)),
                        .num_mcast_dests = dst_noc_info.second});
                    multicast_sem_data[i].emplace_back(
                        transfer_info.data.data(), transfer_info.data.size() * sizeof(uint32_t));
                }
            }
            cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
                multicast_sem_sub_cmds[i].size(),
                multicast_sem_dst_size.back().second,
                max_prefetch_command_size,
                packed_write_max_unicast_sub_cmds,
                multicast_sem_payload[i]);
            i++;
        }
    }

    // Unicast Semaphore Cmd
    uint32_t num_unicast_semaphores = program_transfer_info.unicast_semaphores.size();
    std::vector<std::vector<CQDispatchWritePackedUnicastSubCmd>> unicast_sem_sub_cmds(num_unicast_semaphores);
    std::vector<std::vector<std::pair<const void*, uint32_t>>> unicast_sem_data(num_unicast_semaphores);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> unicast_sem_payload(num_unicast_semaphores);
    std::vector<std::pair<uint32_t, uint32_t>> unicast_sem_dst_size;
    unicast_sem_dst_size.reserve(num_unicast_semaphores);
    if (num_unicast_semaphores > 0) {
        uint32_t i = 0;
        for (const auto& [dst, transfer_info_vec] : program_transfer_info.unicast_semaphores) {
            // TODO: loop over things inside transfer_info[i]
            uint32_t write_packed_len = transfer_info_vec[0].data.size();
            unicast_sem_dst_size.emplace_back(std::make_pair(dst, write_packed_len * sizeof(uint32_t)));

            for (const auto& transfer_info : transfer_info_vec) {
                for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                    TT_ASSERT(
                        transfer_info.data.size() == write_packed_len,
                        "Not all data std::vectors in write packed semaphore cmd equal in len");
                    unicast_sem_sub_cmds[i].emplace_back(CQDispatchWritePackedUnicastSubCmd{
                        .noc_xy_addr =
                            device->get_noc_unicast_encoding(noc_index, std::get<CoreCoord>(dst_noc_info.first))});
                    unicast_sem_data[i].emplace_back(
                        transfer_info.data.data(), transfer_info.data.size() * sizeof(uint32_t));
                }
            }
            cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedUnicastSubCmd>(
                unicast_sem_sub_cmds[i].size(),
                unicast_sem_dst_size.back().second,
                max_prefetch_command_size,
                packed_write_max_unicast_sub_cmds,
                unicast_sem_payload[i]);
            i++;
        }
    }

    uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);

    const auto& circular_buffers_unique_coreranges = program.circular_buffers_unique_coreranges();
    const uint16_t num_multicast_cb_sub_cmds = circular_buffers_unique_coreranges.size();
    std::vector<std::pair<uint32_t, uint32_t>> mcast_cb_payload;
    uint16_t cb_config_size_bytes = 0;
    uint32_t aligned_cb_config_size_bytes = 0;
    std::vector<std::vector<uint32_t>> cb_config_payloads(
        num_multicast_cb_sub_cmds,
        std::vector<uint32_t>(program.get_program_config(index).cb_size / sizeof(uint32_t), 0));
    std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_cb_config_sub_cmds;
    std::vector<std::pair<const void*, uint32_t>> multicast_cb_config_data;
    if (num_multicast_cb_sub_cmds > 0) {
        multicast_cb_config_sub_cmds.reserve(num_multicast_cb_sub_cmds);
        multicast_cb_config_data.reserve(num_multicast_cb_sub_cmds);
        program_command_sequence.circular_buffers_on_core_ranges.resize(num_multicast_cb_sub_cmds);
        uint32_t i = 0;
        uint32_t max_overall_index = 0;
        uint32_t remote_offset_index = program.get_program_config(index).local_cb_size / sizeof(uint32_t);
        for (const CoreRange& core_range : circular_buffers_unique_coreranges) {
            const CoreCoord virtual_start =
                device->virtual_core_from_logical_core(core_range.start_coord, CoreType::WORKER);
            const CoreCoord virtual_end =
                device->virtual_core_from_logical_core(core_range.end_coord, CoreType::WORKER);

            const uint32_t num_receivers = core_range.size();
            auto& cb_config_payload = cb_config_payloads[i];
            uint32_t max_index = 0;
            const auto& circular_buffers_on_corerange = program.circular_buffers_on_corerange(core_range);
            program_command_sequence.circular_buffers_on_core_ranges[i].reserve(circular_buffers_on_corerange.size());
            for (const std::shared_ptr<CircularBuffer>& cb : circular_buffers_on_corerange) {
                program_command_sequence.circular_buffers_on_core_ranges[i].emplace_back(cb);
                const uint32_t cb_address = cb->address();
                const uint32_t cb_size = cb->size();
                for (const auto& buffer_index : cb->local_buffer_indices()) {
                    // 1 cmd for all 32 buffer indices, populate with real data for specified indices
                    // cb config payload
                    const uint32_t base_index = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * buffer_index;
                    cb_config_payload[base_index] = cb_address;
                    cb_config_payload[base_index + 1] = cb_size;
                    cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                    cb_config_payload[base_index + 3] = cb->page_size(buffer_index);
                    max_index = std::max(max_index, base_index + UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG);
                }
                for (const auto& buffer_index : cb->remote_buffer_indices()) {
                    const uint32_t base_index =
                        remote_offset_index +
                        (NUM_CIRCULAR_BUFFERS - 1 - buffer_index) * UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG;
                    cb_config_payload[base_index] = cb->config_address();
                    cb_config_payload[base_index + 1] = cb->page_size(buffer_index);
                    max_index = std::max(max_index, base_index + UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG);
                }
            }
            multicast_cb_config_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                .noc_xy_addr = device->get_noc_multicast_encoding(noc_index, CoreRange(virtual_start, virtual_end)),
                .num_mcast_dests = (uint32_t)core_range.size()});
            multicast_cb_config_data.emplace_back(cb_config_payload.data(), max_index * sizeof(uint32_t));
            max_overall_index = std::max(max_overall_index, max_index);
            i++;
        }
        uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
        cb_config_size_bytes = max_overall_index * sizeof(uint32_t);
        aligned_cb_config_size_bytes = align(cb_config_size_bytes, l1_alignment);
        cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
            num_multicast_cb_sub_cmds,
            cb_config_size_bytes,
            max_prefetch_command_size,
            packed_write_max_unicast_sub_cmds,
            mcast_cb_payload);
    }

    // Program Binaries and Go Signals
    // Get launch msg data while getting size of cmds
    std::vector<std::vector<CQPrefetchRelayPagedPackedSubCmd>> kernel_bins_prefetch_subcmds;
    std::vector<std::vector<CQDispatchWritePackedLargeSubCmd>> kernel_bins_dispatch_subcmds;
    std::vector<uint32_t> kernel_bins_write_packed_large_data_aligned_sizeB;
    std::vector<HostMemDeviceCommand> kernel_bins_unicast_cmds;
    const uint32_t max_length_per_sub_cmd = dispatch_constants::get(dispatch_core_type).scratch_db_size() / 2;
    const uint32_t max_paged_length_per_sub_cmd =
        max_length_per_sub_cmd / HostMemDeviceCommand::PROGRAM_PAGE_SIZE * HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
    if (program_transfer_info.kernel_bins.size()) {
        TT_FATAL(
            program.get_kernels_buffer(device).get(), "Expected Kernel Binary Buffer to be allocated for program.");
    }
    const auto kernels_buffer = program.get_kernels_buffer(device);
    for (const auto& [cores, num_mcast_dests, kg_transfer_info] : program_transfer_info.kernel_bins) {
        bool write_linear;
        uint32_t noc_encoding;
        std::visit(
            [&](auto&& cores) {
                using T = std::decay_t<decltype(cores)>;
                if constexpr (std::is_same_v<T, CoreRange>) {
                    noc_encoding = device->get_noc_multicast_encoding(noc_index, cores);
                    write_linear = false;
                } else {
                    noc_encoding = device->get_noc_unicast_encoding(noc_index, cores);
                    write_linear = true;
                }
            },
            cores);
        for (uint32_t kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
            if (write_linear) {
                kernel_bins_unicast_cmds.emplace_back(2 * hal.get_alignment(HalMemType::HOST));
                cmd_sequence_sizeB += 2 * hal.get_alignment(HalMemType::HOST);
                constexpr bool flush_prefetch = false;
                kernel_bins_unicast_cmds.back().add_dispatch_write_linear<flush_prefetch>(
                    num_mcast_dests,  // num_mcast_dests
                    noc_encoding,     // noc_xy_addr
                    kg_transfer_info.dst_base_addrs[kernel_idx],
                    kg_transfer_info.lengths[kernel_idx]);
                RecordDispatchData(
                    program,
                    DISPATCH_DATA_BINARY,
                    kg_transfer_info.lengths[kernel_idx],
                    kg_transfer_info.riscvs[kernel_idx]);
                // Difference between prefetch total relayed pages and dispatch write linear
                uint32_t relayed_bytes =
                    align(kg_transfer_info.lengths[kernel_idx], HostMemDeviceCommand::PROGRAM_PAGE_SIZE);
                uint16_t length_adjust = uint16_t(relayed_bytes - kg_transfer_info.lengths[kernel_idx]);

                uint32_t base_address, page_offset;
                if (kg_transfer_info.page_offsets[kernel_idx] > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
                    const uint32_t num_banks = device->num_banks(kernels_buffer->buffer_type());
                    page_offset = kg_transfer_info.page_offsets[kernel_idx] % num_banks;
                    uint32_t num_full_pages_written_per_bank = kg_transfer_info.page_offsets[kernel_idx] / num_banks;
                    base_address =
                        kernels_buffer->address() + num_full_pages_written_per_bank * kernels_buffer->page_size();
                } else {
                    base_address = kernels_buffer->address();
                    page_offset = kg_transfer_info.page_offsets[kernel_idx];
                }

                kernel_bins_unicast_cmds.back().add_prefetch_relay_paged(
                    true,  // is_dram
                    page_offset,
                    base_address,
                    kernels_buffer->page_size(),
                    relayed_bytes / kernels_buffer->page_size(),
                    length_adjust);
            } else {
                uint32_t base_address = kernels_buffer->address();
                uint32_t page_offset = kg_transfer_info.page_offsets[kernel_idx];

                // TODO: pack all these writes into 1 linear write
                uint32_t kernel_config_buffer_offset = kg_transfer_info.dst_base_addrs[kernel_idx];
                uint32_t aligned_length =
                    align(kg_transfer_info.lengths[kernel_idx], hal.get_alignment(HalMemType::DRAM));
                uint32_t padding = aligned_length - kg_transfer_info.lengths[kernel_idx];
                while (aligned_length != 0) {
                    if (kernel_bins_dispatch_subcmds.empty() ||
                        kernel_bins_dispatch_subcmds.back().size() == CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS) {
                        kernel_bins_dispatch_subcmds.push_back({});
                        kernel_bins_prefetch_subcmds.push_back({});
                        kernel_bins_write_packed_large_data_aligned_sizeB.push_back(0);
                    }
                    uint32_t write_length, read_length;
                    if (aligned_length <= max_length_per_sub_cmd) {
                        read_length = aligned_length;
                        write_length = read_length - padding;
                    } else {
                        read_length = max_paged_length_per_sub_cmd;
                        write_length = read_length;
                    }
                    if (!kernel_bins_dispatch_subcmds.back().empty()) {
                        auto& back = kernel_bins_dispatch_subcmds.back().back();
                        if (back.noc_xy_addr != noc_encoding) {
                            back.flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
                        }
                    }
                    kernel_bins_dispatch_subcmds.back().emplace_back(CQDispatchWritePackedLargeSubCmd{
                        .noc_xy_addr = noc_encoding,
                        .addr = kernel_config_buffer_offset,
                        .length = (uint16_t)write_length,
                        .num_mcast_dests = (uint8_t)num_mcast_dests,
                        .flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_NONE});
                    RecordDispatchData(
                        program, DISPATCH_DATA_BINARY, write_length, kg_transfer_info.riscvs[kernel_idx]);
                    kernel_config_buffer_offset += write_length;

                    kernel_bins_prefetch_subcmds.back().emplace_back(CQPrefetchRelayPagedPackedSubCmd{
                        .start_page = (uint16_t)page_offset,
                        .log_page_size = (uint16_t)HostMemDeviceCommand::LOG2_PROGRAM_PAGE_SIZE,
                        .base_addr = base_address,
                        .length = read_length});
                    page_offset += read_length / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                    aligned_length -= read_length;
                    kernel_bins_write_packed_large_data_aligned_sizeB.back() += read_length;
                }
            }
        }
    }
    // Unlink the last subcmd of every dispatch, to ensure we don't hold the
    // path reservation for an incredible long time. This also prevents a hang
    // if the next mcast is to a different destination.
    for (auto& subcmd_list : kernel_bins_dispatch_subcmds) {
        if (!subcmd_list.empty()) {
            subcmd_list.back().flags |= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
        }
    }
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    for (uint32_t i = 0; i < kernel_bins_dispatch_subcmds.size(); ++i) {
        cmd_sequence_sizeB += align(
            ((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd))) +
                kernel_bins_dispatch_subcmds[i].size() * sizeof(CQDispatchWritePackedLargeSubCmd),
            pcie_alignment);
        cmd_sequence_sizeB += align(
            kernel_bins_prefetch_subcmds[i].size() * sizeof(CQPrefetchRelayPagedPackedSubCmd) + sizeof(CQPrefetchCmd),
            pcie_alignment);
    }
    std::vector<std::pair<const void*, uint32_t>> multicast_go_signal_data;
    std::vector<std::pair<const void*, uint32_t>> unicast_go_signal_data;
    std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_go_signal_sub_cmds;
    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_go_signal_sub_cmds;
    std::vector<std::pair<uint32_t, uint32_t>> multicast_go_signals_payload;
    std::vector<std::pair<uint32_t, uint32_t>> unicast_go_signals_payload;
    constexpr uint32_t go_signal_sizeB = sizeof(launch_msg_t);
    uint32_t aligned_go_signal_sizeB = align(go_signal_sizeB, hal.get_alignment(HalMemType::L1));
    uint32_t go_signal_size_words = aligned_go_signal_sizeB / sizeof(uint32_t);

    // TODO: eventually the code below could be structured to loop over programmable_indices
    // and check for mcast/unicast
    uint32_t programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    for (auto& kernel_group : program.get_kernel_groups(programmable_core_index)) {
        kernel_group->launch_msg.kernel_config.mode = DISPATCH_MODE_DEV;
        for (uint32_t i = 0; i < NUM_PROGRAMMABLE_CORE_TYPES; i++) {
            kernel_group->launch_msg.kernel_config.kernel_config_base[i] = 0;
        }
        kernel_group->launch_msg.kernel_config.host_assigned_id = program.get_runtime_id();
        const void* launch_message_data = (const void*)(&(kernel_group->launch_msg));
        for (const CoreRange& core_range : kernel_group->core_ranges.ranges()) {
            CoreCoord virtual_start =
                device->virtual_core_from_logical_core(core_range.start_coord, kernel_group->get_core_type());
            CoreCoord virtual_end =
                device->virtual_core_from_logical_core(core_range.end_coord, kernel_group->get_core_type());

            multicast_go_signal_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                .noc_xy_addr = device->get_noc_multicast_encoding(noc_index, CoreRange(virtual_start, virtual_end)),
                .num_mcast_dests = (uint32_t)core_range.size()});
            multicast_go_signal_data.emplace_back(launch_message_data, go_signal_sizeB);
        }
    }
    if (multicast_go_signal_sub_cmds.size() > 0) {
        cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
            multicast_go_signal_sub_cmds.size(),
            go_signal_sizeB,
            max_prefetch_command_size,
            packed_write_max_unicast_sub_cmds,
            multicast_go_signals_payload);
    }

    programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    // TODO: ugly, can be fixed by looping over indices w/ some work
    if (programmable_core_index != -1) {
        for (auto& kernel_group : program.get_kernel_groups(programmable_core_index)) {
            kernel_group->launch_msg.kernel_config.mode = DISPATCH_MODE_DEV;
            // Set the kernel_config_base addrs to 0 when generating the dispatch commands for the program
            // Will be resolved at runtime
            for (uint32_t i = 0; i < NUM_PROGRAMMABLE_CORE_TYPES; i++) {
                kernel_group->launch_msg.kernel_config.kernel_config_base[i] = 0;
            }
            kernel_group->launch_msg.kernel_config.host_assigned_id = program.get_runtime_id();
            const void* launch_message_data = (const launch_msg_t*)(&(kernel_group->launch_msg));
            for (const CoreRange& core_range : kernel_group->core_ranges.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord virtual_coord =
                            device->virtual_core_from_logical_core(CoreCoord({x, y}), kernel_group->get_core_type());
                        unicast_go_signal_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                            .noc_xy_addr = device->get_noc_unicast_encoding(noc_index, virtual_coord)});
                        unicast_go_signal_data.emplace_back(launch_message_data, go_signal_sizeB);
                    }
                }
            }
        }
    }

    if (unicast_go_signal_sub_cmds.size() > 0) {
        cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedUnicastSubCmd>(
            unicast_go_signal_sub_cmds.size(),
            go_signal_sizeB,
            max_prefetch_command_size,
            packed_write_max_unicast_sub_cmds,
            unicast_go_signals_payload);
    }

    // if dispatch_s is enabled have dispatch_d send a semaphore update to dispatch_s (this will include a write barrier
    // on dispatch_d if program is active) if not,  check if the program is active on workers. If active, have
    // dispatch_d issue a write barrier
    cmd_sequence_sizeB += (device->dispatch_s_enabled() || program_transfer_info.num_active_cores > 0) *
                          hal.get_alignment(HalMemType::HOST);

    // either dispatch_s or dispatch_d will send the go signal (go_signal_mcast command)
    const auto& noc_data_start_idx = device->noc_data_start_index(
        sub_device_id, multicast_go_signal_sub_cmds.size() > 0, unicast_go_signal_sub_cmds.size() > 0);
    const auto& num_noc_mcast_txns =
        multicast_go_signal_sub_cmds.size() > 0 ? device->num_noc_mcast_txns(sub_device_id) : 0;
    const auto& num_noc_unicast_txns =
        unicast_go_signal_sub_cmds.size() > 0 ? device->num_noc_unicast_txns(sub_device_id) : 0;
    cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment);

    program_command_sequence.device_command_sequence = HostMemDeviceCommand(cmd_sequence_sizeB);

    auto& device_command_sequence = program_command_sequence.device_command_sequence;

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    // Semaphores
    // Multicast Semaphore Cmd
    index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    for (uint32_t i = 0; i < num_multicast_semaphores; ++i) {
        uint32_t curr_sub_cmd_idx = 0;
        for (const auto& [num_sub_cmds_in_cmd, multicast_sem_payload_sizeB] : multicast_sem_payload[i]) {
            device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_sub_cmds_in_cmd,
                multicast_sem_dst_size[i].first + program.get_program_config(index).sem_offset,
                multicast_sem_dst_size[i].second,
                multicast_sem_payload_sizeB,
                multicast_sem_sub_cmds[i],
                multicast_sem_data[i],
                packed_write_max_unicast_sub_cmds,
                curr_sub_cmd_idx,
                false,
                DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE);
            curr_sub_cmd_idx += num_sub_cmds_in_cmd;
            for (auto& data_and_size : multicast_sem_data[i]) {
                RecordDispatchData(program, DISPATCH_DATA_SEMAPHORE, data_and_size.second);
            }
        }
    }

    // Unicast Semaphore Cmd
    index = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    for (uint32_t i = 0; i < num_unicast_semaphores; ++i) {
        uint32_t curr_sub_cmd_idx = 0;
        for (const auto& [num_sub_cmds_in_cmd, unicast_sem_payload_sizeB] : unicast_sem_payload[i]) {
            device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_sub_cmds_in_cmd,
                unicast_sem_dst_size[i].first + program.get_program_config(index).sem_offset,
                unicast_sem_dst_size[i].second,
                unicast_sem_payload_sizeB,
                unicast_sem_sub_cmds[i],
                unicast_sem_data[i],
                packed_write_max_unicast_sub_cmds,
                curr_sub_cmd_idx,
                false,
                DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE);
            curr_sub_cmd_idx += num_sub_cmds_in_cmd;
            for (auto& data_and_size : unicast_sem_data[i]) {
                RecordDispatchData(program, DISPATCH_DATA_SEMAPHORE, data_and_size.second);
            }
        }
    }

    // CB Configs commands
    index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    if (num_multicast_cb_sub_cmds > 0) {
        uint32_t curr_sub_cmd_idx = 0;
        program_command_sequence.cb_configs_payloads.reserve(num_multicast_cb_sub_cmds);
        const uint32_t cb_config_size_words = aligned_cb_config_size_bytes / sizeof(uint32_t);
        for (const auto& [num_sub_cmds_in_cmd, mcast_cb_payload_sizeB] : mcast_cb_payload) {
            uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
            device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_sub_cmds_in_cmd,
                program.get_program_config(index).cb_offset,
                cb_config_size_bytes,
                mcast_cb_payload_sizeB,
                multicast_cb_config_sub_cmds,
                multicast_cb_config_data,
                packed_write_max_unicast_sub_cmds,
                curr_sub_cmd_idx,
                false,
                DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE);
            for (auto& data_and_size : multicast_cb_config_data) {
                RecordDispatchData(program, DISPATCH_DATA_CB_CONFIG, data_and_size.second);
            }
            curr_sub_cmd_idx += num_sub_cmds_in_cmd;
            RecordDispatchData(program, DISPATCH_DATA_CB_CONFIG, mcast_cb_payload_sizeB);
            uint32_t curr_sub_cmd_data_offset_words =
                (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                 align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedMulticastSubCmd), l1_alignment)) /
                sizeof(uint32_t);
            for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                program_command_sequence.cb_configs_payloads.push_back(
                    (uint32_t*)device_command_sequence.data() + curr_sub_cmd_data_offset_words);
                curr_sub_cmd_data_offset_words += cb_config_size_words;
            }
        }
    }
    // All Previous Cmds Up to This Point Go Into the Kernel Config Buffer
    program_command_sequence.program_config_buffer_data_size_bytes = device_command_sequence.write_offset_bytes();

    // Program Binaries
    for (const auto& kernel_bins_unicast_cmd : kernel_bins_unicast_cmds) {
        device_command_sequence.add_data(
            kernel_bins_unicast_cmd.data(), kernel_bins_unicast_cmd.size_bytes(), kernel_bins_unicast_cmd.size_bytes());
    }
    uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);
    for (uint32_t i = 0; i < kernel_bins_dispatch_subcmds.size(); ++i) {
        device_command_sequence.add_dispatch_write_packed_large(
            dram_alignment,
            kernel_bins_dispatch_subcmds[i].size(),
            kernel_bins_dispatch_subcmds[i],
            0,
            DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE);
        device_command_sequence.add_prefetch_relay_paged_packed(
            kernel_bins_write_packed_large_data_aligned_sizeB[i],
            kernel_bins_prefetch_subcmds[i],
            kernel_bins_prefetch_subcmds[i].size());
    }

    // Go Signals
    program_command_sequence.go_signals.reserve(
        multicast_go_signal_sub_cmds.size() + unicast_go_signal_sub_cmds.size());

    // Launch Message address is resolved when the program is enqueued
    uint32_t multicast_launch_msg_addr = 0;

    if (multicast_go_signal_sub_cmds.size() > 0) {
        uint32_t curr_sub_cmd_idx = 0;
        for (const auto& [num_sub_cmds_in_cmd, multicast_go_signal_payload_sizeB] : multicast_go_signals_payload) {
            uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
            device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_sub_cmds_in_cmd,
                multicast_launch_msg_addr,
                go_signal_sizeB,
                multicast_go_signal_payload_sizeB,
                multicast_go_signal_sub_cmds,
                multicast_go_signal_data,
                packed_write_max_unicast_sub_cmds,
                curr_sub_cmd_idx);
            curr_sub_cmd_idx += num_sub_cmds_in_cmd;
            program_command_sequence.launch_msg_write_packed_cmd_ptrs.push_back(
                &((CQDispatchCmd*)((uint32_t*)device_command_sequence.data() +
                                   (write_offset_bytes + sizeof(CQPrefetchCmd)) / sizeof(uint32_t)))
                     ->write_packed);
            uint32_t curr_sub_cmd_data_offset_words =
                (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                 align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedMulticastSubCmd), l1_alignment)) /
                sizeof(uint32_t);
            for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                program_command_sequence.go_signals.push_back(
                    (launch_msg_t*)((uint32_t*)device_command_sequence.data() + curr_sub_cmd_data_offset_words));
                curr_sub_cmd_data_offset_words += go_signal_size_words;
            }
        }
    }

    if (unicast_go_signal_sub_cmds.size() > 0) {
        // Launch Message address is resolved when the program is enqueued
        uint32_t unicast_launch_msg_addr = 0;
        uint32_t curr_sub_cmd_idx = 0;
        for (const auto& [num_sub_cmds_in_cmd, unicast_go_signal_payload_sizeB] : unicast_go_signals_payload) {
            uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
            device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_sub_cmds_in_cmd,
                unicast_launch_msg_addr,
                go_signal_sizeB,
                unicast_go_signal_payload_sizeB,
                unicast_go_signal_sub_cmds,
                unicast_go_signal_data,
                packed_write_max_unicast_sub_cmds,
                curr_sub_cmd_idx);
            curr_sub_cmd_idx += num_sub_cmds_in_cmd;
            program_command_sequence.unicast_launch_msg_write_packed_cmd_ptrs.push_back(
                &((CQDispatchCmd*)((uint32_t*)device_command_sequence.data() +
                                   (write_offset_bytes + sizeof(CQPrefetchCmd)) / sizeof(uint32_t)))
                     ->write_packed);
            uint32_t curr_sub_cmd_data_offset_words =
                (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                 align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment)) /
                sizeof(uint32_t);
            for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                program_command_sequence.go_signals.push_back(
                    (launch_msg_t*)((uint32_t*)device_command_sequence.data() + curr_sub_cmd_data_offset_words));
                curr_sub_cmd_data_offset_words += go_signal_size_words;
            }
        }
    }

    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    auto sub_device_index = sub_device_id.to_index();
    uint32_t dispatch_message_addr =
        dispatch_constants::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE) +
        dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(sub_device_index);
    if (device->dispatch_s_enabled()) {
        // dispatch_d signals dispatch_s to send the go signal, use a barrier if there are cores active
        uint16_t index_bitmask = 0;
        index_bitmask |= 1 << sub_device_index;
        device_command_sequence.add_notify_dispatch_s_go_signal_cmd(
            program_transfer_info.num_active_cores > 0, index_bitmask);
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
    } else {
        // Wait Noc Write Barrier, wait for binaries/configs and launch_msg to be written to worker cores
        if (program_transfer_info.num_active_cores > 0) {
            device_command_sequence.add_dispatch_wait(true, dispatch_message_addr, 0, 0, false, false);
        }
    }
    go_msg_t run_program_go_signal;
    run_program_go_signal.signal = RUN_MSG_GO;
    // Dispatch X/Y resolved when the program is enqueued
    run_program_go_signal.master_x = 0;
    run_program_go_signal.master_y = 0;
    run_program_go_signal.dispatch_message_offset =
        (uint8_t)dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(sub_device_index);
    uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
    // Num Workers Resolved when the program is enqueued
    device_command_sequence.add_dispatch_go_signal_mcast(
        0,
        *reinterpret_cast<uint32_t*>(&run_program_go_signal),
        dispatch_message_addr,
        num_noc_mcast_txns,
        num_noc_unicast_txns,
        noc_data_start_idx,
        dispatcher_for_go_signal);
    program_command_sequence.mcast_go_signal_cmd_ptr =
        &((CQDispatchCmd*)((uint32_t*)device_command_sequence.data() +
                           (write_offset_bytes + sizeof(CQPrefetchCmd)) / sizeof(uint32_t)))
             ->mcast;
}

void reserve_space_in_kernel_config_buffer(
    WorkerConfigBufferMgr& config_buffer_mgr,
    const std::vector<uint32_t>& program_config_sizes,
    ProgramBinaryStatus program_binary_status,
    uint32_t num_program_workers,
    uint32_t expected_num_workers_completed,
    ProgramDispatchMetadata& dispatch_md) {
    // Reserve space in kernel config ring buffer for the current program
    std::pair<ConfigBufferSync, std::vector<ConfigBufferEntry>&> reservation =
        config_buffer_mgr.reserve(program_config_sizes);
    // Determine where a sync (dispatch wait on workers) must be inserted in the program sequence and the number
    // of workers to wait on
    dispatch_md.sync_count = 0;
    dispatch_md.stall_first = reservation.first.need_sync;
    dispatch_md.stall_before_program = false;

    if (reservation.first.need_sync) {
        // TODO: attempt to send RTA only without stalling.
        dispatch_md.sync_count = reservation.first.sync_count;
        // Check if the launch message is the only thing preventing us from
        // sending the program. If so, we can at least send the RTAs. Ideally we
        // would also send the kernel binaries in this case, but the rest of the
        // code isn't set up for that.
        auto config_sizes = program_config_sizes;
        config_sizes[config_sizes.size() - 2] = 0;
        config_sizes[config_sizes.size() - 1] = 0;
        const std::pair<ConfigBufferSync, std::vector<ConfigBufferEntry>&> memory_reservation =
            config_buffer_mgr.reserve(config_sizes);
        if (!memory_reservation.first.need_sync) {
            dispatch_md.stall_first = false;
            dispatch_md.stall_before_program = true;
        }
        reservation = config_buffer_mgr.reserve(program_config_sizes);
    }

    if (program_binary_status == ProgramBinaryStatus::InFlight) {
        // Program binary not commited to DRAM. Sync on all workers before dispatching kernel
        // binaries for this program. This requires freeing the entire kernel config buffer.
        config_buffer_mgr.free(expected_num_workers_completed);
    } else {
        if (dispatch_md.stall_first || dispatch_md.stall_before_program) {
            config_buffer_mgr.free(dispatch_md.sync_count);
        }
    }
    config_buffer_mgr.alloc(expected_num_workers_completed + num_program_workers);

    if (program_binary_status != ProgramBinaryStatus::Committed) {
        // Insert a stall before writing any program configs when binaries are in flight
        dispatch_md.stall_first = true;
        dispatch_md.stall_before_program = false;
        // Wait on all previous workers before writing kernel binaries to workers
        dispatch_md.sync_count = expected_num_workers_completed;
    }

    dispatch_md.kernel_config_addrs = reservation.second;
}

void update_program_dispatch_commands(
    Program& program,
    ProgramCommandSequence& cached_program_command_sequence,
    const tt::stl::Span<ConfigBufferEntry> kernel_config_addrs,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    CoreType dispatch_core_type,
    SubDeviceId sub_device_id,
    const ProgramDispatchMetadata& dispatch_md,
    ProgramBinaryStatus program_binary_status,
    int num_unicast_txns) {
    uint32_t i = 0;
    ZoneScopedN("program_loaded_on_device");

    static constexpr uint32_t wait_count_offset = (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, wait.count));
    static constexpr uint32_t tensix_l1_write_offset_offset =
        (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.offset1));
    static constexpr uint32_t eth_l1_write_offset_offset =
        (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.offset2));
    // Update Stall Command Sequence
    if (program_binary_status != ProgramBinaryStatus::Committed) {
        // Program binary is in flight. Issue a Prefetch Stall
        cached_program_command_sequence.current_stall_seq_idx = UncachedStallSequenceIdx;
    } else {
        // Program Binary is in DRAM. Prefetcher does not need to stall before reading
        // binary
        cached_program_command_sequence.current_stall_seq_idx = CachedStallSequenceIdx;
    }

    auto& curr_stall_seq_idx = cached_program_command_sequence.current_stall_seq_idx;
    cached_program_command_sequence.stall_command_sequences[curr_stall_seq_idx].update_cmd_sequence(
        wait_count_offset, &(dispatch_md.sync_count), sizeof(uint32_t));

    // Update preamble based on kernel config ring buffer slot
    cached_program_command_sequence.preamble_command_sequence.update_cmd_sequence(
        tensix_l1_write_offset_offset,
        &kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)],
        sizeof(uint32_t));
    if (hal.get_programmable_core_type_count() >= 2) {
        cached_program_command_sequence.preamble_command_sequence.update_cmd_sequence(
            eth_l1_write_offset_offset,
            &kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)],
            sizeof(uint32_t));
    }

    // Update CB Configs
    uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    uint32_t remote_offset_index = program.get_program_config(index).local_cb_size / sizeof(uint32_t);
    for (const auto& cbs_on_core_range : cached_program_command_sequence.circular_buffers_on_core_ranges) {
        uint32_t* cb_config_payload = cached_program_command_sequence.cb_configs_payloads[i];
        for (const std::shared_ptr<CircularBuffer>& cb : cbs_on_core_range) {
            const uint32_t cb_address = cb->address();
            const uint32_t cb_size = cb->size();
            for (const auto& buffer_index : cb->local_buffer_indices()) {
                // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                // cb config payload
                uint32_t base_index = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * buffer_index;
                cb_config_payload[base_index] = cb_address;
                cb_config_payload[base_index + 1] = cb_size;
                cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                cb_config_payload[base_index + 3] = cb->page_size(buffer_index);
            }
            for (const auto& buffer_index : cb->remote_buffer_indices()) {
                const uint32_t base_index = remote_offset_index + (NUM_CIRCULAR_BUFFERS - 1 - buffer_index) *
                                                                      UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG;
                cb_config_payload[base_index] = cb->config_address();
                cb_config_payload[base_index + 1] = cb->page_size(buffer_index);
            }
        }
        i++;
    }
    // Update launch messages
    for (auto& go_signal : cached_program_command_sequence.go_signals) {
        for (uint32_t i = 0; i < kernel_config_addrs.size(); i++) {
            go_signal->kernel_config.kernel_config_base[i] = kernel_config_addrs[i].addr;
        }
        go_signal->kernel_config.host_assigned_id = program.get_runtime_id();
    }
    // Update launch message addresses to reflect new launch_msg slot in ring buffer
    uint32_t multicast_cores_launch_msg_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LAUNCH) +
        multicast_cores_launch_message_wptr * sizeof(launch_msg_t);
    for (auto launch_msg_cmd_ptr : cached_program_command_sequence.launch_msg_write_packed_cmd_ptrs) {
        launch_msg_cmd_ptr->addr = multicast_cores_launch_msg_addr;
    }
    if (cached_program_command_sequence.unicast_launch_msg_write_packed_cmd_ptrs.size()) {
        uint32_t unicast_cores_launch_message_addr =
            hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::LAUNCH) +
            unicast_cores_launch_message_wptr * sizeof(launch_msg_t);
        for (auto launch_msg_cmd_ptr : cached_program_command_sequence.unicast_launch_msg_write_packed_cmd_ptrs) {
            launch_msg_cmd_ptr->addr = unicast_cores_launch_message_addr;
        }
    }
    // Update go signal to reflect potentially modified dispatch core and new wait count
    go_msg_t run_program_go_signal;
    run_program_go_signal.signal = RUN_MSG_GO;
    run_program_go_signal.master_x = (uint8_t)dispatch_core.x;
    run_program_go_signal.master_y = (uint8_t)dispatch_core.y;
    run_program_go_signal.dispatch_message_offset =
        (uint8_t)dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(sub_device_id.to_index());
    cached_program_command_sequence.mcast_go_signal_cmd_ptr->go_signal =
        *reinterpret_cast<uint32_t*>(&run_program_go_signal);
    cached_program_command_sequence.mcast_go_signal_cmd_ptr->wait_count = expected_num_workers_completed;
    // Update the number of unicast txns based on user provided parameter
    // This is required when a MeshWorkload users ethernet cores on a set of devices
    // where the number of active eth cores is heterogenous across devices.
    // Update the number of unicast txns to eth cores to match the minimum number of cores
    // across devices (specified by user)
    if (num_unicast_txns >= 0 && cached_program_command_sequence.mcast_go_signal_cmd_ptr->num_unicast_txns) {
        cached_program_command_sequence.mcast_go_signal_cmd_ptr->num_unicast_txns = num_unicast_txns;
    }
}

KernelHandle get_device_local_kernel_handle(KernelHandle kernel_handle) {
    // Device local Kernel Handle/Kernel Ids are 16 bit. The top 16 bits of
    // the Kernel Handle may encode device coordinates when MeshWorkloads are
    // being dispatched.
    return kernel_handle & 0xffff;
}

}  // namespace program_utils

}  // namespace tt::tt_metal
