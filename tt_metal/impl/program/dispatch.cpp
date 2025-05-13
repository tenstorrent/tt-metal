// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program/dispatch.hpp"

#include <magic_enum/magic_enum.hpp>
#include <mesh_workload.hpp>
#include <stddef.h>
#include <string.h>
#include <sub_device_types.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt_align.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include "assert.hpp"
#include "buffer.hpp"
#include "circular_buffer.hpp"
#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "dev_msgs.h"
#include "device.hpp"
#include "dispatch/device_command.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "program_device_map.hpp"
#include "tt-metalium/program.hpp"
#include "runtime_args_data.hpp"
#include "semaphore.hpp"
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "tt_memory.h"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/xy_pair.h>
#include "util.hpp"
#include "vector_aligned.hpp"
#include "worker_config_buffer.hpp"

namespace tt {
namespace tt_metal {
enum NOC : uint8_t;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {
using detail::ProgramImpl;

namespace detail {
std::shared_ptr<Kernel> GetKernel(const ProgramImpl& program, KernelHandle kernel_id) {
    return program.get_kernel(kernel_id);
}

}  // namespace detail

namespace program_dispatch {

namespace {
CoreCoord get_sub_device_worker_origin(
    const tt::tt_metal::IDevice* device,
    tt::tt_metal::SubDeviceId sub_device_id,
    tt::tt_metal::HalProgrammableCoreType core_type) {
    const auto grid = device->worker_cores(core_type, sub_device_id);
    if (grid.empty()) {
        return {0, 0};
    }
    return grid.bounding_box().start_coord;
}

struct CommandConstants {
    CoreType dispatch_core_type;
    NOC noc_index;
    uint32_t max_prefetch_command_size;
    uint32_t packed_write_max_unicast_sub_cmds;
};
};  // namespace

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
    uint32_t processor_classes =
        MetalContext::instance().hal().get_processor_classes_count(programmable_core_type_index);
    std::vector<uint32_t> max_rtas(processor_classes);
    uint32_t max_unique_rta_size = 0;
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);

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
        offset = tt::align(offset, l1_alignment);
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
    uint32_t processor_classes =
        MetalContext::instance().hal().get_processor_classes_count(programmable_core_type_index);
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
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
        uint32_t size = max_crtas[dispatch_class] * sizeof(uint32_t);
        crta_offsets[dispatch_class] = crta_base_offset + offset;
        crta_sizes[dispatch_class] = size;
        offset += size;
        offset = tt::align(offset, l1_alignment);
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
    CoreType core_type = MetalContext::instance().hal().get_core_type(programmable_core_type_index);
    HalProgrammableCoreType programmable_core_type =
        MetalContext::instance().hal().get_programmable_core_type(programmable_core_type_index);

    uint32_t max_unique_rta_size = program_dispatch::configure_rta_offsets_for_kernel_groups(
        programmable_core_type_index, kernels, kernel_groups, base_offset);
    uint32_t crta_base_offset = base_offset + max_unique_rta_size;
    uint32_t total_crta_size = program_dispatch::configure_crta_offsets_for_kernel_groups(
        programmable_core_type_index, kernels, kernel_groups, crta_base_offset, crta_offsets, crta_sizes);

    uint32_t offset = max_unique_rta_size + total_crta_size;

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
    CoreType core_type = MetalContext::instance().hal().get_core_type(programmable_core_type_index);
    for (const auto& sem : semaphores) {
        if (sem.core_type() == core_type && (int)sem.id() > max_id) {
            max_id = sem.id();
        }
    }
    uint32_t sem_size = (max_id + 1) * MetalContext::instance().hal().get_alignment(HalMemType::L1);
    semaphore_offset = sem_base_offset;
    semaphore_size = sem_size;
    return sem_base_offset + sem_size;
}

uint32_t finalize_cbs(
    uint32_t /*programmable_core_type_index*/,
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

    return tt::align(base_offset + total_cb_size, MetalContext::instance().hal().get_alignment(HalMemType::L1));
}

uint32_t finalize_kernel_bins(
    IDevice* device,
    uint32_t programmable_core_type_index,
    const std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& kernels,
    std::vector<std::shared_ptr<KernelGroup>>& kernel_groups,
    uint32_t base_offset,
    uint32_t& kernel_text_offset,
    uint32_t& kernel_text_size) {
    const auto& hal = MetalContext::instance().hal();
    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t max_offset = 0;
    for (auto& kg : kernel_groups) {
        uint32_t offset = base_offset;

        for (int class_id = 0; class_id < DISPATCH_CLASS_MAX; class_id++) {
            auto& optional_id = kg->kernel_ids[class_id];
            if (optional_id) {
                const auto kernel = kernels.at(optional_id.value());
                const std::vector<const ll_api::memory*>& binaries = kernel->binaries(
                    BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key);
                // TODO: this is really ugly, save me future-HAL!
                if (programmable_core_type_index ==
                    hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)) {
                    uint32_t binary_packed_size = kernel->get_binary_packed_size(device, 0);

                    if (class_id == DISPATCH_CLASS_TENSIX_DM0) {
                        kg->kernel_bin_sizes[0] = binary_packed_size;
                        kg->kernel_text_offsets[0] = offset;
                        kg->launch_msg.kernel_config.kernel_text_offset[0] = offset;
                        offset += binary_packed_size;
                        offset = tt::align(offset, l1_alignment);
                    } else if (class_id == DISPATCH_CLASS_TENSIX_DM1) {
                        kg->kernel_bin_sizes[1] = binary_packed_size;
                        kg->kernel_text_offsets[1] = offset;
                        kg->launch_msg.kernel_config.kernel_text_offset[1] = offset;
                        offset += binary_packed_size;
                        offset = tt::align(offset, l1_alignment);

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
                            offset = tt::align(offset, l1_alignment);
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
                        offset = tt::align(offset, l1_alignment);
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

uint32_t get_packed_write_max_unicast_sub_cmds(IDevice* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}

void insert_empty_program_dispatch_preamble_cmd(ProgramCommandSequence& program_command_sequence) {
    // Initialize an empty preamble command in the Program Dispatch Cmd Sequence, which will be
    // updated with the correct write offsets when the program is enqueued
    tt::tt_metal::DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_write_offsets();
    const uint32_t preamble_cmd_sizeB = calculator.write_offset_bytes();
    program_command_sequence.preamble_command_sequence = HostMemDeviceCommand(preamble_cmd_sizeB);
    program_command_sequence.preamble_command_sequence.add_dispatch_set_write_offsets(0, 0, 0);
}

void insert_stall_cmds(
    ProgramCommandSequence& program_command_sequence, SubDeviceId sub_device_id, IDevice* /*device*/) {
    // Initialize stall command sequences for this program.
    tt::tt_metal::DeviceCommandCalculator calculator;
    calculator.add_dispatch_wait_with_prefetch_stall();
    const uint32_t uncached_stall_cmd_sizeB = calculator.write_offset_bytes();
    calculator.clear();

    calculator.add_dispatch_wait();
    const uint32_t cached_stall_cmd_seqB = calculator.write_offset_bytes();

    program_command_sequence.stall_command_sequences[UncachedStallSequenceIdx] =
        HostMemDeviceCommand(uncached_stall_cmd_sizeB);
    // Empty wait command initialized here. Will get updated when program is enqueued.
    program_command_sequence.stall_command_sequences[UncachedStallSequenceIdx].add_dispatch_wait_with_prefetch_stall(
        CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER | CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
        0,
        MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(*sub_device_id),
        0);
    // Empty wait command initialized here. Will get updated when program is enqueued.
    program_command_sequence.stall_command_sequences[CachedStallSequenceIdx] =
        HostMemDeviceCommand(cached_stall_cmd_seqB);
    program_command_sequence.stall_command_sequences[CachedStallSequenceIdx].add_dispatch_wait(
        CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
        0,
        MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(*sub_device_id),
        0);
}

template <typename PackedSubCmd>
void generate_runtime_args_cmds(
    std::vector<HostMemDeviceCommand>& runtime_args_command_sequences,
    std::vector<ProgramCommandSequence::RtaUpdate>& rta_updates,
    const uint32_t& l1_arg_base_addr,
    const std::vector<PackedSubCmd>& sub_cmds,
    const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>& rt_data_and_sizes,
    const uint32_t& max_runtime_args_len,
    std::vector<std::vector<
        std::pair<std::reference_wrapper<RuntimeArgsData>, std::reference_wrapper<const std::vector<uint32_t>>>>>&
        rt_args_data,
    const CommandConstants& constants,
    bool no_stride,
    enum DispatchWriteOffsets write_offset_index) {
    static_assert(
        std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value or
        std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);

    thread_local static auto get_runtime_payload_sizeB =
        [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast, bool no_stride) {
            uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
            uint32_t sub_cmd_sizeB =
                is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB =
                sizeof(CQDispatchCmd) + tt::align(num_packed_cmds * sub_cmd_sizeB, l1_alignment);
            uint32_t aligned_runtime_data_sizeB =
                (no_stride ? 1 : num_packed_cmds) * tt::align(runtime_args_len * sizeof(uint32_t), l1_alignment);
            return dispatch_cmd_sizeB + aligned_runtime_data_sizeB;
        };
    thread_local static auto get_runtime_args_data_offset =
        [](uint32_t num_packed_cmds, uint32_t /*runtime_args_len*/, bool is_unicast) {
            uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
            uint32_t sub_cmd_sizeB =
                is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB =
                sizeof(CQDispatchCmd) + tt::align(num_packed_cmds * sub_cmd_sizeB, l1_alignment);
            return sizeof(CQPrefetchCmd) + dispatch_cmd_sizeB;
        };

    constexpr bool unicast = std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value;

    uint32_t num_packed_cmds_in_seq = sub_cmds.size();
    DeviceCommandCalculator calculator;
    uint32_t max_packed_cmds = calculator.get_max_write_packed_sub_cmds<PackedSubCmd>(
        max_runtime_args_len,
        constants.max_prefetch_command_size,
        constants.packed_write_max_unicast_sub_cmds,
        no_stride);
    uint32_t offset_idx = 0;
    if (no_stride) {
        TT_FATAL(
            max_packed_cmds >= num_packed_cmds_in_seq,
            "num_packed_cmds_in_seq {} cannot exceed max_packed_cmds {} when no_stride is true",
            num_packed_cmds_in_seq,
            max_packed_cmds);
    }
    uint32_t pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    while (num_packed_cmds_in_seq != 0) {
        // Generate the device command
        uint32_t num_packed_cmds = std::min(num_packed_cmds_in_seq, max_packed_cmds);
        uint32_t rt_payload_sizeB =
            get_runtime_payload_sizeB(num_packed_cmds, max_runtime_args_len, unicast, no_stride);
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed<PackedSubCmd>(
            num_packed_cmds,
            max_runtime_args_len * sizeof(uint32_t),
            constants.packed_write_max_unicast_sub_cmds,
            no_stride);
        runtime_args_command_sequences.emplace_back(calculator.write_offset_bytes());
        runtime_args_command_sequences.back().add_dispatch_write_packed<PackedSubCmd>(
            CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_RTA,
            num_packed_cmds,
            l1_arg_base_addr,
            max_runtime_args_len * sizeof(uint32_t),
            rt_payload_sizeB,
            sub_cmds,
            rt_data_and_sizes,
            constants.packed_write_max_unicast_sub_cmds,
            offset_idx,
            no_stride,
            write_offset_index);
        TT_ASSERT(
            runtime_args_command_sequences.back().size_bytes() ==
            runtime_args_command_sequences.back().write_offset_bytes());

        uint32_t data_offset = (uint32_t)get_runtime_args_data_offset(num_packed_cmds, max_runtime_args_len, unicast);
        const uint32_t data_inc = tt::align(max_runtime_args_len * sizeof(uint32_t), l1_alignment);
        uint32_t num_data_copies = no_stride ? 1 : num_packed_cmds;
        for (uint32_t i = offset_idx; i < offset_idx + num_data_copies; ++i) {
            uint32_t offset = 0;
            for (uint32_t j = 0; j < rt_args_data[i].size(); ++j) {
                auto& data = rt_args_data[i][j];
                uint32_t* data_in_sequence =
                    (uint32_t*)((char*)runtime_args_command_sequences.back().data() + data_offset + offset);
                if (data.first.get().rt_args_data == data.second.get().data()) {
                    // Update the pointer to point into the command sequence. Future RTA updates will modify the command
                    // sequence directly.
                    data.first.get().rt_args_data = data_in_sequence;
                } else {
                    TT_ASSERT(data.first.get().rt_args_data == std::get<0>(rt_data_and_sizes[i][j]));
                    // Pointer already points into another command sequence. Schedule a copy from there.
                    rta_updates.emplace_back(
                        data.first.get().rt_args_data,
                        data_in_sequence,
                        data.first.get().rt_args_count * sizeof(uint32_t));
                }
                offset += data.first.get().rt_args_count * sizeof(uint32_t);
            }
            data_offset += data_inc;
        }
        num_packed_cmds_in_seq -= num_packed_cmds;
        offset_idx += num_packed_cmds;
    }
}

struct Transfer {
    uint32_t start;
    tt::stl::Span<const uint8_t> data;
    // Keep track of what CBs contributed to this transfer, so we can update the data in
    // update_program_dispatch_commands.
    std::vector<std::shared_ptr<CircularBuffer>> cbs;
    // RTAs must be updated from data every time update_program_dispatch_commmands is called.
    RuntimeArgsData* rta_data = nullptr;
    size_t end() const { return start + data.size(); }
};
struct PairHash {
    std::size_t operator()(const std::pair<uint32_t, uint32_t> pair) const {
        return std::hash<uint32_t>()(pair.first) ^ std::hash<uint32_t>()(pair.second);
    }
};
// Map each corerange to the set of transfer-vectors that need to be sent to it. Each transfer-vector will be sent
// as a single CQDispatchWritePackedLargeSubCmd.
using BatchedTransfers = std::unordered_map<
    std::pair</*noc_xy_addr*/ uint32_t, /*num_mcast_dests*/ uint32_t>,
    std::map</*start_addr*/ uint32_t, std::vector<Transfer>>,
    PairHash>;

BatchedTransfers assemble_runtime_args_commands(
    ProgramCommandSequence& program_command_sequence,
    ProgramImpl& program,
    IDevice* device,
    const CommandConstants& constants) {
    BatchedTransfers transfers = {};
    using RtaDataPair =
        std::pair<std::reference_wrapper<RuntimeArgsData>, std::reference_wrapper<const std::vector<uint32_t>>>;
    // Dispatch Commands to Unicast Unique Runtime Args to Workers
    std::vector<CQDispatchWritePackedUnicastSubCmd> unique_sub_cmds;
    std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>> unique_rt_data_and_sizes;
    std::vector<std::vector<RtaDataPair>> unique_rt_args_data;
    // Dispatch Commands to Multicast Common Runtime Args to Workers
    std::variant<std::vector<CQDispatchWritePackedMulticastSubCmd>, std::vector<CQDispatchWritePackedUnicastSubCmd>>
        common_sub_cmds;
    std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>> common_rt_data_and_sizes;
    std::vector<std::vector<RtaDataPair>> common_rt_args_data;  // Data per kernel group

    program_command_sequence.runtime_args_command_sequences = {};
    uint32_t command_count = 0;
    const DeviceCommandCalculator calculator;

    // Unique RTAs
    const auto& hal = MetalContext::instance().hal();
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
                uint32_t max_packed_cmds =
                    calculator.get_max_write_packed_sub_cmds<decltype(unique_sub_cmds)::value_type>(
                        max_runtime_args_len,
                        constants.max_prefetch_command_size,
                        constants.packed_write_max_unicast_sub_cmds,
                        false);
                command_count += div_up(num_sub_cmds, max_packed_cmds);
            }
        }
    }

    // Calculate the best way to multicast common RTAs.

    // Per-kernel is best when there are a lot of kernel groups and few kernels (which should be rare).
    uint32_t per_kernel_crta_multicast_count = 0;
    for (size_t kernel_index = 0; kernel_index < program.num_kernels(); kernel_index++) {
        auto kernel_id = get_device_local_kernel_handle(kernel_index);
        auto kernel = detail::GetKernel(program, kernel_id);
        if (kernel->get_kernel_core_type() != CoreType::WORKER) {
            continue;  // TODO: fixme, need list of kernels by core_typexdispatch_class
        }

        const auto& common_rt_args = kernel->common_runtime_args();
        if (common_rt_args.empty()) {
            continue;
        }
        per_kernel_crta_multicast_count += kernel->logical_coreranges().size();
    }

    // kernel_group multicast is best when multiple kernels on the same kernel group have common RTAs. It may also merge
    // CRTA writes with CB and semaphore writes.
    uint32_t kernel_group_crta_multicast_count = 0;
    {
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        for (auto& kg : program.get_kernel_groups(index)) {
            bool has_crtas = false;
            for (auto kernel_id : kg->kernel_ids) {
                if (kernel_id) {
                    auto device_local_kernel_handle = get_device_local_kernel_handle(kernel_id.value());
                    auto kernel = detail::GetKernel(program, device_local_kernel_handle);
                    if (!kernel->common_runtime_args().empty()) {
                        has_crtas = true;
                        break;
                    }
                }
            }
            if (has_crtas) {
                kernel_group_crta_multicast_count += kg->core_ranges.size();
            }
        }
    }

    // kernel group multicast can merge with CB multicast, so prefer it in general.
    bool use_kernel_group_crta_multicast = kernel_group_crta_multicast_count <= per_kernel_crta_multicast_count;

    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        auto programmable_core_type = kernel->get_kernel_programmable_core_type();
        if (programmable_core_type == HalProgrammableCoreType::IDLE_ETH) {
            // Fast dispatch not supported on IDLE_ETH yet
            continue;
        }
        if ((programmable_core_type == HalProgrammableCoreType::TENSIX) && use_kernel_group_crta_multicast) {
            continue;
        }
        uint32_t programmable_core_type_index = hal.get_programmable_core_type_index(programmable_core_type);
        uint32_t common_size =
            program.get_program_config(programmable_core_type_index).crta_sizes[kernel->dispatch_class()];
        if (common_size != 0) {
            uint32_t max_runtime_args_len = common_size / sizeof(uint32_t);
            const auto& common_rt_args = kernel->common_runtime_args();

            if (common_rt_args.size() > 0) {
                if (!tt::tt_metal::MetalContext::instance().hal().get_supports_receiving_multicasts(
                        programmable_core_type_index)) {
                    uint32_t num_sub_cmds = kernel->logical_cores().size();
                    uint32_t max_packed_cmds =
                        calculator.get_max_write_packed_sub_cmds<CQDispatchWritePackedUnicastSubCmd>(
                            max_runtime_args_len,
                            constants.max_prefetch_command_size,
                            constants.packed_write_max_unicast_sub_cmds,
                            true);
                    command_count += div_up(num_sub_cmds, max_packed_cmds);
                } else {
                    uint32_t num_sub_cmds = kernel->logical_coreranges().size();
                    uint32_t max_packed_cmds =
                        calculator.get_max_write_packed_sub_cmds<CQDispatchWritePackedMulticastSubCmd>(
                            max_runtime_args_len,
                            constants.max_prefetch_command_size,
                            constants.packed_write_max_unicast_sub_cmds,
                            true);
                    command_count += div_up(num_sub_cmds, max_packed_cmds);
                }
            }
        }
    }

    program_command_sequence.runtime_args_command_sequences.reserve(command_count);

    if (use_kernel_group_crta_multicast) {
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        for (auto& kg : program.get_kernel_groups(index)) {
            for (auto kernel_id : kg->kernel_ids) {
                if (kernel_id) {
                    auto device_local_kernel_handle = get_device_local_kernel_handle(kernel_id.value());
                    auto kernel = detail::GetKernel(program, device_local_kernel_handle);
                    if (kernel->common_runtime_args().empty()) {
                        continue;
                    }
                    uint32_t dispatch_class = kernel->dispatch_class();
                    const uint32_t crta_offset = program.get_program_config(index).crta_offsets[dispatch_class];
                    for (auto& transfer_info :
                         extract_dst_noc_multicast_info(device, kg->core_ranges.ranges(), CoreType::WORKER)) {
                        auto noc_xy_addr = device->get_noc_multicast_encoding(
                            constants.noc_index, std::get<CoreRange>(transfer_info.cores));
                        size_t size =
                            kernel->common_runtime_args().size() * sizeof(*kernel->common_runtime_args().data());
                        RecordDispatchData(program.get_id(), DISPATCH_DATA_RTARGS, size);
                        transfers[std::make_pair(noc_xy_addr, transfer_info.num_dests)][crta_offset] =
                            std::vector<Transfer>{Transfer{
                                .start = crta_offset,
                                .data = tt::stl::Span(
                                    reinterpret_cast<uint8_t*>(kernel->common_runtime_args().data()), size),
                                .cbs = {},
                                .rta_data = &kernel->common_runtime_args_data()}};
                    }
                }
            }
        }
    }

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        if (hal.get_programmable_core_type(index) == HalProgrammableCoreType::IDLE_ETH) {
            // Fast dispatch not supported on IDLE_ETH yet
            // TODO: can't just loop here as code below confuses ACTIVE/IDLE
            continue;
        }
        CoreType core_type = hal.get_core_type(index);
        uint32_t processor_classes = hal.get_processor_classes_count(index);

        // Unique RTAs - Unicast
        // Set by the user based on the kernel and core coord
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
                                        unique_rt_args_data.back().emplace_back(
                                            RtaDataPair(kernel->runtime_args_data(core_coord), runtime_args_data));
                                        TT_ASSERT(
                                            runtime_args_data.size() * sizeof(uint32_t) <=
                                            kg->rta_sizes[dispatch_class]);
                                        unique_rt_data_and_sizes.back().emplace_back(
                                            kernel->runtime_args_data(core_coord).rt_args_data,
                                            runtime_args_data.size() * sizeof(uint32_t),
                                            kg->rta_sizes[dispatch_class]);
                                    }
                                }
                            }
                            CoreCoord virtual_core = device->virtual_core_from_logical_core(core_coord, core_type);
                            unique_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr = device->get_noc_unicast_encoding(constants.noc_index, virtual_core)});
                        }
                    }
                }
                uint32_t rta_offset = program.get_program_config(index).rta_offset;
                generate_runtime_args_cmds(
                    program_command_sequence.runtime_args_command_sequences,
                    program_command_sequence.rta_updates,
                    rta_offset,
                    unique_sub_cmds,
                    unique_rt_data_and_sizes,
                    kg->total_rta_size / sizeof(uint32_t),
                    unique_rt_args_data,
                    constants,
                    false,
                    core_type == CoreType::WORKER ? DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE
                                                  : DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE);
                for (auto& data_per_kernel : unique_rt_data_and_sizes) {
                    for (auto& data_and_sizes : data_per_kernel) {
                        RecordDispatchData(program.get_id(), DISPATCH_DATA_RTARGS, std::get<1>(data_and_sizes));
                    }
                }
                unique_sub_cmds.clear();
                unique_rt_data_and_sizes.clear();
                unique_rt_args_data.clear();
            }
        }

        // Common RTAs
        // Set by the user based on the kernel ID. All cores running that kernel ID will get these RTAs
        // On ETH use unicast
        if (!use_kernel_group_crta_multicast ||
            !tt::tt_metal::MetalContext::instance().hal().get_supports_receiving_multicasts(index)) {
            for (int dispatch_class = 0; dispatch_class < processor_classes; dispatch_class++) {
                const uint32_t crta_offset = program.get_program_config(index).crta_offsets[dispatch_class];
                uint32_t common_size = program.get_program_config(index).crta_sizes[dispatch_class];
                if (common_size == 0) {
                    continue;
                }

                for (size_t kernel_index = 0; kernel_index < program.num_kernels(); kernel_index++) {
                    auto kernel_id = get_device_local_kernel_handle(kernel_index);
                    auto kernel = detail::GetKernel(program, kernel_id);
                    if (kernel->get_kernel_core_type() != core_type) {
                        continue;  // TODO: fixme, need list of kernels by core_typexdispatch_class
                    }
                    if (kernel->dispatch_class() != dispatch_class) {
                        continue;  // TODO: fixme, need list of kernels by core_typexdispatch_class
                    }

                    const auto& common_rt_args = kernel->common_runtime_args();
                    if (common_rt_args.empty()) {
                        continue;
                    }

                    common_rt_args_data.resize(common_rt_args_data.size() + 1);
                    common_rt_data_and_sizes.resize(common_rt_data_and_sizes.size() + 1);

                    TT_ASSERT(kernel->common_runtime_args_data().size() * sizeof(uint32_t) == common_size);
                    TT_ASSERT(common_rt_args.size() * sizeof(uint32_t) <= common_size);
                    common_rt_data_and_sizes.back().emplace_back(
                        kernel->common_runtime_args_data().data(),
                        common_rt_args.size() * sizeof(uint32_t),
                        common_size);
                    common_rt_args_data.back().emplace_back(
                        RtaDataPair(kernel->common_runtime_args_data(), common_rt_args));

                    // Target core cannot receive multicast commands -> send unicast
                    if (!tt::tt_metal::MetalContext::instance().hal().get_supports_receiving_multicasts(index)) {
                        common_sub_cmds.emplace<std::vector<CQDispatchWritePackedUnicastSubCmd>>(
                            std::vector<CQDispatchWritePackedUnicastSubCmd>());
                        auto& unicast_sub_cmd =
                            std::get<std::vector<CQDispatchWritePackedUnicastSubCmd>>(common_sub_cmds);
                        unicast_sub_cmd.reserve(kernel->logical_cores().size());
                        for (auto& core_coord : kernel->logical_cores()) {
                            // can make a vector of unicast encodings here
                            CoreCoord virtual_core_coords =
                                device->virtual_core_from_logical_core(core_coord, core_type);
                            unicast_sub_cmd.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr =
                                    device->get_noc_unicast_encoding(constants.noc_index, virtual_core_coords)});
                        }
                    } else {
                        std::vector<multicast_transfer_info> dst_noc_multicast_info =
                            extract_dst_noc_multicast_info(device, kernel->logical_coreranges(), core_type);
                        common_sub_cmds.emplace<std::vector<CQDispatchWritePackedMulticastSubCmd>>(
                            std::vector<CQDispatchWritePackedMulticastSubCmd>());
                        auto& multicast_sub_cmd =
                            std::get<std::vector<CQDispatchWritePackedMulticastSubCmd>>(common_sub_cmds);
                        multicast_sub_cmd.reserve(dst_noc_multicast_info.size());
                        for (const auto& mcast_dests : dst_noc_multicast_info) {
                            multicast_sub_cmd.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                                .noc_xy_addr = device->get_noc_multicast_encoding(
                                    constants.noc_index, std::get<CoreRange>(mcast_dests.cores)),
                                .num_mcast_dests = mcast_dests.num_dests});
                        }
                    }

                    // Fill out the command for this kernel group and then reset the vectors for the next group
                    // NOTE: Common rtas are always expected to fit in one prefetch cmd
                    // TODO: use a linear write instead of a packed-write
                    std::visit(
                        [&](auto&& sub_cmds) {
                            generate_runtime_args_cmds(
                                program_command_sequence.runtime_args_command_sequences,
                                program_command_sequence.rta_updates,
                                crta_offset,
                                sub_cmds,
                                common_rt_data_and_sizes,
                                common_size / sizeof(uint32_t),
                                common_rt_args_data,
                                constants,
                                true,
                                core_type == CoreType::WORKER ? DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE
                                                              : DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE);
                            sub_cmds.clear();
                        },
                        common_sub_cmds);
                    common_rt_data_and_sizes.clear();
                    common_rt_args_data.clear();
                }

                for (auto& data_per_kernel : common_rt_data_and_sizes) {
                    for (auto& data_and_sizes : data_per_kernel) {
                        RecordDispatchData(program.get_id(), DISPATCH_DATA_RTARGS, std::get<1>(data_and_sizes));
                    }
                }
            }
        }
    }

    TT_ASSERT(
        command_count >= program_command_sequence.runtime_args_command_sequences.size(),
        "Incorrect number of commands reserved {}, final size {}. Vector reallocation causes cached addresses to be "
        "incorrect.",
        command_count,
        program_command_sequence.runtime_args_command_sequences.size());

    return transfers;
}

class SemphoreCommandGenerator {
public:
    // Generate batched_transfers (for multicast) and unicast_semaphore_cmds for the semaphores in the program.
    void size_commands(
        ProgramImpl& program,
        IDevice* device,
        DeviceCommandCalculator& calculator,
        const CommandConstants& constants,
        BatchedTransfers& batched_transfers) {
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
        // Prevent reallocation of semaphore_data to ensure pointers remain valid.
        semaphore_data.reserve(program.semaphores().size());

        // Unicast/Multicast Semaphores
        const auto& hal = MetalContext::instance().hal();
        for (const Semaphore& semaphore : program.semaphores()) {
            semaphore_data.push_back(semaphore.initial_value());

            if (semaphore.core_type() == CoreType::WORKER) {
                uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
                std::vector<multicast_transfer_info> dst_noc_multicast_info =
                    extract_dst_noc_multicast_info(device, semaphore.core_range_set().ranges(), CoreType::WORKER);
                for (const auto& dst_noc_info : dst_noc_multicast_info) {
                    auto noc_xy_addr = device->get_noc_multicast_encoding(
                        constants.noc_index, std::get<CoreRange>(dst_noc_info.cores));
                    uint32_t start_addr = semaphore.offset() + program.get_program_config(index).sem_offset;
                    RecordDispatchData(program.get_id(), DISPATCH_DATA_SEMAPHORE, sizeof(uint32_t));
                    batched_transfers[std::make_pair(noc_xy_addr, dst_noc_info.num_dests)][start_addr] =
                        std::vector<Transfer>{
                            {{.start = start_addr,
                              .data = tt::stl::Span(
                                  reinterpret_cast<const uint8_t*>(&semaphore_data.back()), sizeof(uint32_t))}}};
                }
            } else if (semaphore.core_type() == CoreType::ETH) {
                unicast_semaphore_cmds.push_back({.dst = semaphore.offset(), .size = sizeof(uint32_t)});
                auto& unicast_cmds = unicast_semaphore_cmds.back();
                // TODO: we only fast dispatch to active eth...
                uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
                std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_unicast_info =
                    extract_dst_noc_unicast_info(semaphore.core_range_set().ranges(), CoreType::ETH);
                for (const auto& dst_noc_info : dst_noc_unicast_info) {
                    unicast_cmds.sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                        .noc_xy_addr = device->get_noc_unicast_encoding(
                            constants.noc_index, std::get<CoreCoord>(dst_noc_info.first))});
                    unicast_cmds.data.emplace_back(&semaphore_data.back(), sizeof(uint32_t));
                }
                calculator.insert_write_packed_payloads<CQDispatchWritePackedUnicastSubCmd>(
                    unicast_cmds.sub_cmds.size(),
                    unicast_cmds.size,
                    constants.max_prefetch_command_size,
                    constants.packed_write_max_unicast_sub_cmds,
                    unicast_cmds.payload);
            }
        }
    }

    // Write unicast semaphore commands to the device command sequence.
    void assemble_unicast_commands(
        HostMemDeviceCommand& device_command_sequence, ProgramImpl& program, const CommandConstants& constants) const {
        // Unicast Semaphore Cmd
        uint32_t index =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
        for (auto& cmds : unicast_semaphore_cmds) {
            uint32_t curr_sub_cmd_idx = 0;
            for (const auto& [num_sub_cmds_in_cmd, unicast_sem_payload_sizeB] : cmds.payload) {
                device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_SEMS,
                    num_sub_cmds_in_cmd,
                    cmds.dst + program.get_program_config(index).sem_offset,
                    cmds.size,
                    unicast_sem_payload_sizeB,
                    cmds.sub_cmds,
                    cmds.data,
                    constants.packed_write_max_unicast_sub_cmds,
                    curr_sub_cmd_idx,
                    false,
                    DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE);
                curr_sub_cmd_idx += num_sub_cmds_in_cmd;
                for (auto& data_and_size : cmds.data) {
                    RecordDispatchData(program.get_id(), DISPATCH_DATA_SEMAPHORE, data_and_size.second);
                }
            }
        }
    }

private:
    struct UnicastSemaphoreData {
        std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds;
        // 1 per sub_cmd.
        std::vector<std::pair<const void*, uint32_t>> data;
        // Regions of sub_cmds that each fit in a single command.
        std::vector<std::pair<uint32_t, uint32_t>> payload;
        uint32_t dst;
        uint32_t size;
    };

    std::vector<uint32_t> semaphore_data;
    std::vector<UnicastSemaphoreData> unicast_semaphore_cmds;
};

class CircularBufferCommandGenerator {
public:
    // Construct the circular buffer commands for the program into batched_transfers. This class must stay alive until
    // batched_transfers is used, because batched_transfers contains pointers to the circular buffer data in this class.
    void construct_commands(
        IDevice* device, const CommandConstants& constants, ProgramImpl& program, BatchedTransfers& batched_transfers) {
        const auto& hal = MetalContext::instance().hal();
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);

        const auto& circular_buffers_unique_coreranges = program.circular_buffers_unique_coreranges();
        const uint16_t num_multicast_cb_sub_cmds = circular_buffers_unique_coreranges.size();
        cb_config_payloads = std::vector<std::vector<uint32_t>>(
            num_multicast_cb_sub_cmds,
            std::vector<uint32_t>(program.get_program_config(index).cb_size / sizeof(uint32_t), 0));
        if (num_multicast_cb_sub_cmds > 0) {
            uint32_t i = 0;
            uint32_t max_overall_index = 0;
            uint32_t remote_offset_index = program.get_program_config(index).local_cb_size / sizeof(uint32_t);
            auto index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            for (const CoreRange& core_range : circular_buffers_unique_coreranges) {
                const CoreCoord& virtual_start =
                    device->virtual_core_from_logical_core(core_range.start_coord, CoreType::WORKER);
                const CoreCoord& virtual_end =
                    device->virtual_core_from_logical_core(core_range.end_coord, CoreType::WORKER);

                const uint32_t num_receivers = core_range.size();
                auto& cb_config_payload = cb_config_payloads[i];
                uint32_t max_index = 0;
                const auto& circular_buffers_on_corerange = program.circular_buffers_on_corerange(core_range);
                for (const std::shared_ptr<CircularBuffer>& cb : circular_buffers_on_corerange) {
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
                auto noc_xy_addr =
                    device->get_noc_multicast_encoding(constants.noc_index, CoreRange(virtual_start, virtual_end));
                uint32_t start_addr = program.get_program_config(index).cb_offset;
                RecordDispatchData(program.get_id(), DISPATCH_DATA_CB_CONFIG, max_index * sizeof(uint32_t));

                batched_transfers[std::make_pair(noc_xy_addr, core_range.size())][start_addr] = std::vector<Transfer>{
                    {.start = start_addr,
                     .data = tt::stl::Span(
                         reinterpret_cast<const uint8_t*>(cb_config_payload.data()), max_index * sizeof(uint32_t)),
                     .cbs = circular_buffers_on_corerange}};
                i++;
            }
        }
    }

private:
    std::vector<std::vector<uint32_t>> cb_config_payloads;
};

class ProgramBinaryCommandGenerator {
public:
    // Generate kernel_bins_cmds (for multicast) and kernel_bins_unicast_cmds (for unicast) for the binaries in the
    // program.
    void size_commands(
        IDevice* device,
        ProgramImpl& program,
        const ProgramTransferInfo& program_transfer_info,
        const std::shared_ptr<Buffer>& kernels_buffer,
        const CommandConstants& constants,
        DeviceCommandCalculator& calculator) {
        const auto& hal = MetalContext::instance().hal();
        const uint32_t max_length_per_sub_cmd =
            MetalContext::instance().dispatch_mem_map(constants.dispatch_core_type).scratch_db_size() / 2;
        const uint32_t max_paged_length_per_sub_cmd =
            max_length_per_sub_cmd / HostMemDeviceCommand::PROGRAM_PAGE_SIZE * HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
        for (const auto& [cores, num_mcast_dests, kg_transfer_info] : program_transfer_info.kernel_bins) {
            bool write_linear;
            uint32_t noc_encoding;
            std::visit(
                [&](auto&& cores) {
                    using T = std::decay_t<decltype(cores)>;
                    if constexpr (std::is_same_v<T, CoreRange>) {
                        noc_encoding = device->get_noc_multicast_encoding(constants.noc_index, cores);
                        write_linear = false;
                    } else {
                        noc_encoding = device->get_noc_unicast_encoding(constants.noc_index, cores);
                        write_linear = true;
                    }
                },
                cores);
            for (uint32_t kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
                if (write_linear) {
                    kernel_bins_unicast_cmds.emplace_back(2 * hal.get_alignment(HalMemType::HOST));
                    constexpr bool flush_prefetch = false;
                    calculator.add_dispatch_write_linear<flush_prefetch>(0);
                    kernel_bins_unicast_cmds.back().add_dispatch_write_linear<flush_prefetch>(
                        num_mcast_dests,  // num_mcast_dests
                        noc_encoding,     // noc_xy_addr
                        kg_transfer_info.dst_base_addrs[kernel_idx],
                        kg_transfer_info.lengths[kernel_idx]);
                    RecordDispatchData(
                        program.get_id(),
                        DISPATCH_DATA_BINARY,
                        kg_transfer_info.lengths[kernel_idx],
                        kg_transfer_info.riscvs[kernel_idx]);
                    // Difference between prefetch total relayed pages and dispatch write linear
                    uint32_t relayed_bytes =
                        tt::align(kg_transfer_info.lengths[kernel_idx], HostMemDeviceCommand::PROGRAM_PAGE_SIZE);
                    uint16_t length_adjust = uint16_t(relayed_bytes - kg_transfer_info.lengths[kernel_idx]);

                    uint32_t base_address, page_offset;
                    if (kg_transfer_info.page_offsets[kernel_idx] > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
                        const uint32_t num_banks = device->allocator()->get_num_banks(kernels_buffer->buffer_type());
                        page_offset = kg_transfer_info.page_offsets[kernel_idx] % num_banks;
                        uint32_t num_full_pages_written_per_bank =
                            kg_transfer_info.page_offsets[kernel_idx] / num_banks;
                        base_address =
                            kernels_buffer->address() + num_full_pages_written_per_bank * kernels_buffer->page_size();
                    } else {
                        base_address = kernels_buffer->address();
                        page_offset = kg_transfer_info.page_offsets[kernel_idx];
                    }

                    calculator.add_prefetch_relay_paged();
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
                        tt::align(kg_transfer_info.lengths[kernel_idx], hal.get_alignment(HalMemType::DRAM));
                    uint32_t padding = aligned_length - kg_transfer_info.lengths[kernel_idx];
                    while (aligned_length != 0) {
                        if (kernel_bins_cmds.empty() || kernel_bins_cmds.back().dispatch_subcmds.size() ==
                                                            CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS) {
                            kernel_bins_cmds.push_back({});
                        }
                        auto& kernel_bins_cmd = kernel_bins_cmds.back();
                        uint32_t write_length, read_length;
                        if (aligned_length <= max_length_per_sub_cmd) {
                            read_length = aligned_length;
                            write_length = read_length - padding;
                        } else {
                            read_length = max_paged_length_per_sub_cmd;
                            write_length = read_length;
                        }
                        if (!kernel_bins_cmd.dispatch_subcmds.empty()) {
                            auto& back = kernel_bins_cmd.dispatch_subcmds.back();
                            if (back.noc_xy_addr == noc_encoding) {
                                back.flags &= ~CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
                            }
                        }
                        kernel_bins_cmd.dispatch_subcmds.emplace_back(CQDispatchWritePackedLargeSubCmd{
                            .noc_xy_addr = noc_encoding,
                            .addr = kernel_config_buffer_offset,
                            .length = (uint16_t)write_length,
                            .num_mcast_dests = (uint8_t)num_mcast_dests,
                            .flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK});
                        RecordDispatchData(
                            program.get_id(), DISPATCH_DATA_BINARY, write_length, kg_transfer_info.riscvs[kernel_idx]);
                        kernel_config_buffer_offset += write_length;

                        kernel_bins_cmd.prefetch_subcmds.emplace_back(CQPrefetchRelayPagedPackedSubCmd{
                            .start_page = (uint16_t)page_offset,
                            .log_page_size = (uint16_t)HostMemDeviceCommand::LOG2_PROGRAM_PAGE_SIZE,
                            .base_addr = base_address,
                            .length = read_length});
                        page_offset += read_length / HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
                        aligned_length -= read_length;
                        kernel_bins_cmd.data_aligned_sizeB += read_length;
                    }
                }
            }
        }
        for (auto& kernel_bins_cmd : kernel_bins_cmds) {
            calculator.add_dispatch_write_packed_large(kernel_bins_cmd.dispatch_subcmds.size());
            calculator.add_prefetch_relay_paged_packed(kernel_bins_cmd.prefetch_subcmds.size());
        }
    }

    // Assemble the program binary commands into the device command sequence.
    void assemble_commands(HostMemDeviceCommand& device_command_sequence) const {
        const auto& hal = MetalContext::instance().hal();
        for (const auto& kernel_bins_unicast_cmd : kernel_bins_unicast_cmds) {
            device_command_sequence.add_data(
                kernel_bins_unicast_cmd.data(),
                kernel_bins_unicast_cmd.size_bytes(),
                kernel_bins_unicast_cmd.size_bytes());
        }
        uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);
        for (const KernelBinsCmds& kernel_bins_cmd : kernel_bins_cmds) {
            device_command_sequence.add_dispatch_write_packed_large(
                CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_PROGRAM_BINARIES,
                dram_alignment,
                kernel_bins_cmd.dispatch_subcmds.size(),
                kernel_bins_cmd.dispatch_subcmds,
                0,
                DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE);
            device_command_sequence.add_prefetch_relay_paged_packed(
                kernel_bins_cmd.data_aligned_sizeB,
                kernel_bins_cmd.prefetch_subcmds,
                kernel_bins_cmd.prefetch_subcmds.size());
        }
    }

private:
    struct KernelBinsCmds {
        std::vector<CQPrefetchRelayPagedPackedSubCmd> prefetch_subcmds;
        std::vector<CQDispatchWritePackedLargeSubCmd> dispatch_subcmds;
        uint32_t data_aligned_sizeB{0};
    };
    std::vector<KernelBinsCmds> kernel_bins_cmds;
    std::vector<HostMemDeviceCommand> kernel_bins_unicast_cmds;
};

class BatchedTransferGenerator {
public:
    // Construct and optimal set of CQDispatchWritePackedLargeSubCmds from the
    // batched transfers.  This is done by combining adjacent or nearly adjacent
    // transfers into a single command and linking transfers to the same CoreRanges.
    void construct_commands(BatchedTransfers& batched_transfers, DeviceCommandCalculator& calculator) {
        const auto& hal = MetalContext::instance().hal();
        uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
        // Optimize transfers by combining adjacent or nearly adjacent transfers.
        for (auto& transfer_set : batched_transfers) {
            for (auto it = transfer_set.second.begin(); it != transfer_set.second.end();) {
                auto next_it = std::next(it);
                if (next_it == transfer_set.second.end()) {
                    break;
                }
                TT_ASSERT(next_it->second.size() == 1);
                TT_ASSERT(it->second.back().end() <= next_it->first);
                if (it->second.back().end() + l1_alignment >= next_it->first) {
                    it->second.push_back(std::move(next_it->second.front()));
                    transfer_set.second.erase(next_it);
                } else {
                    it = next_it;
                }
            }
        }

        // Generate WritePackedLargeSubCmds from the transfers.
        for (auto& transfer_set : batched_transfers) {
            for (auto& [start, transfer_vector] : transfer_set.second) {
                if (batched_dispatch_subcmds.empty() ||
                    batched_dispatch_subcmds.back().size() >= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS) {
                    batched_dispatch_subcmds.emplace_back();
                    batched_cmd_data.emplace_back();
                }
                if (batched_dispatch_subcmds.back().size() > 0) {
                    auto& last_transfer = batched_dispatch_subcmds.back().back();
                    if (last_transfer.noc_xy_addr != transfer_set.first.first) {
                        last_transfer.flags |= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
                    }
                }
                TT_ASSERT(start == transfer_vector.front().start);
                size_t size = transfer_vector.back().end() - start;
                batched_dispatch_subcmds.back().emplace_back(CQDispatchWritePackedLargeSubCmd{
                    .noc_xy_addr = transfer_set.first.first,
                    .addr = start,
                    .length = (uint16_t)size,
                    .num_mcast_dests = transfer_set.first.second,
                    .flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_NONE});

                // Modify the start addresses to be relative to the dispatch buffer.
                uint32_t new_start =
                    !batched_cmd_data.back().empty()
                        ? tt::align(static_cast<uint32_t>(batched_cmd_data.back().back().end()), l1_alignment)
                        : 0;
                uint32_t start_offset = transfer_vector.front().start - new_start;
                for (Transfer& sub_transfer : transfer_vector) {
                    batched_cmd_data.back().push_back(std::move(sub_transfer));
                    batched_cmd_data.back().back().start -= start_offset;
                }
            }
        }
        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        for (size_t i = 0; i < batched_dispatch_subcmds.size(); i++) {
            calculator.add_dispatch_write_packed_large(
                batched_dispatch_subcmds[i].size(),
                batched_cmd_data[i].back().end() - batched_cmd_data[i].front().start);

            batched_dispatch_subcmds[i].back().flags |= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
        }
    }

    // Assemble the batched transfer commands into the device command sequence.
    void assemble_commands(
        ProgramCommandSequence& program_command_sequence, HostMemDeviceCommand& device_command_sequence) {
        const auto& hal = MetalContext::instance().hal();
        uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
        const std::vector<uint8_t> fill_data(l1_alignment, 0);

        // Write out batched semaphore + CB multicast transfers.
        for (uint32_t i = 0; i < batched_dispatch_subcmds.size(); ++i) {
            auto& cmd_data = batched_cmd_data[i];
            size_t last_end = cmd_data.front().start;
            std::vector<tt::stl::Span<const uint8_t>> batched_data;
            for (const Transfer& transfer : cmd_data) {
                if (last_end != transfer.start) {
                    TT_ASSERT(transfer.start - last_end <= fill_data.size());
                    TT_ASSERT(last_end < transfer.start);
                    batched_data.emplace_back(fill_data.data(), transfer.start - last_end);
                }
                batched_data.emplace_back(transfer.data);
                last_end = transfer.end();
            }
            std::vector<uint8_t*> data_collection_location;
            device_command_sequence.add_dispatch_write_packed_large(
                CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_CBS_SEMS_CRTAS,
                l1_alignment,
                batched_dispatch_subcmds[i].size(),
                batched_dispatch_subcmds[i],
                batched_data,
                &data_collection_location,
                0,
                DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE);

            last_end = cmd_data.front().start;
            size_t j = 0;
            for (Transfer& transfer : cmd_data) {
                if (last_end != transfer.start) {
                    j++;
                }
                if (transfer.cbs.size() > 0) {
                    program_command_sequence.circular_buffers_on_core_ranges.push_back(std::move(transfer.cbs));
                    program_command_sequence.cb_configs_payloads.push_back(
                        reinterpret_cast<uint32_t*>(data_collection_location[j]));
                }
                if (transfer.rta_data) {
                    if (reinterpret_cast<uint8_t*>(transfer.rta_data->rt_args_data) == transfer.data.data()) {
                        // rt_args_data points to the original vector. Update it so later modifications directly modify
                        // the command stream.
                        transfer.rta_data->rt_args_data = reinterpret_cast<uint32_t*>(data_collection_location[j]);
                    } else {
                        // rt_args_data points into the command stream. Setup a copy from that other location.
                        program_command_sequence.rta_updates.push_back(ProgramCommandSequence::RtaUpdate{
                            transfer.rta_data->rt_args_data, data_collection_location[j], transfer.data.size()});
                    }
                }
                j++;
                last_end = transfer.end();
            }
        }
    }

private:
    std::vector<std::vector<Transfer>> batched_cmd_data;
    std::vector<std::vector<CQDispatchWritePackedLargeSubCmd>> batched_dispatch_subcmds;
};

class LaunchMessageGenerator {
public:
    // Construct the launch message commands for the program.
    // This includes the launch message for the TENSIX and ETH cores.
    void construct_commands(
        IDevice* device,
        ProgramImpl& program,
        DeviceCommandCalculator& calculator,
        const CommandConstants& constants,
        SubDeviceId sub_device_id) {
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        for (uint32_t programmable_core_type_index = 0;
             programmable_core_type_index < tt::tt_metal::NumHalProgrammableCoreTypes;
             ++programmable_core_type_index) {
            for (auto& kernel_group : program.get_kernel_groups(programmable_core_type_index)) {
                kernel_group->launch_msg.kernel_config.mode = DISPATCH_MODE_DEV;
                kernel_group->launch_msg.kernel_config.preload = DISPATCH_ENABLE_FLAG_PRELOAD;

                for (uint32_t i = 0; i < NUM_PROGRAMMABLE_CORE_TYPES; i++) {
                    kernel_group->launch_msg.kernel_config.kernel_config_base[i] = 0;
                }
                kernel_group->launch_msg.kernel_config.host_assigned_id = program.get_runtime_id();

                // Setup values for dataflow kernel APIs for getting logical and relative coordinates
                const auto& origin = get_sub_device_worker_origin(
                    device,
                    sub_device_id,
                    static_cast<tt::tt_metal::HalProgrammableCoreType>(programmable_core_type_index));
                kernel_group->launch_msg.kernel_config.sub_device_origin_x = origin.x;
                kernel_group->launch_msg.kernel_config.sub_device_origin_y = origin.y;

                const void* launch_message_data = (const void*)(&(kernel_group->launch_msg));
                if (hal.get_supports_receiving_multicasts(programmable_core_type_index)) {
                    for (const CoreRange& core_range : kernel_group->core_ranges.ranges()) {
                        CoreCoord virtual_start = device->virtual_core_from_logical_core(
                            core_range.start_coord, kernel_group->get_core_type());
                        CoreCoord virtual_end =
                            device->virtual_core_from_logical_core(core_range.end_coord, kernel_group->get_core_type());

                        multicast_cmds.sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                            .noc_xy_addr = device->get_noc_multicast_encoding(
                                constants.noc_index, CoreRange(virtual_start, virtual_end)),
                            .num_mcast_dests = (uint32_t)core_range.size()});
                        multicast_cmds.data.emplace_back(launch_message_data, launch_msg_sizeB);
                    }
                } else {
                    // Need to unicast to each core in the kernel group
                    for (const CoreRange& core_range : kernel_group->core_ranges.ranges()) {
                        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                                CoreCoord virtual_coord = device->virtual_core_from_logical_core(
                                    CoreCoord({x, y}), kernel_group->get_core_type());
                                unicast_cmds.sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                    .noc_xy_addr =
                                        device->get_noc_unicast_encoding(constants.noc_index, virtual_coord)});
                                unicast_cmds.data.emplace_back(launch_message_data, launch_msg_sizeB);
                            }
                        }
                    }
                }
            }
        }

        if (!multicast_cmds.sub_cmds.empty()) {
            calculator.insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
                multicast_cmds.sub_cmds.size(),
                launch_msg_sizeB,
                constants.max_prefetch_command_size,
                constants.packed_write_max_unicast_sub_cmds,
                multicast_cmds.payload);
        }

        if (!unicast_cmds.sub_cmds.empty()) {
            calculator.insert_write_packed_payloads<CQDispatchWritePackedUnicastSubCmd>(
                unicast_cmds.sub_cmds.size(),
                launch_msg_sizeB,
                constants.max_prefetch_command_size,
                constants.packed_write_max_unicast_sub_cmds,
                unicast_cmds.payload);
        }
    }

    // Assemble the launch message commands into the device command sequence.
    void assemble_commands(
        ProgramCommandSequence& program_command_sequence,
        HostMemDeviceCommand& device_command_sequence,
        const CommandConstants& constants) const {
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
        uint32_t aligned_launch_msg_sizeB = tt::align(launch_msg_sizeB, l1_alignment);
        uint32_t launch_msg_size_words = aligned_launch_msg_sizeB / sizeof(uint32_t);

        program_command_sequence.launch_messages.reserve(multicast_cmds.sub_cmds.size() + unicast_cmds.sub_cmds.size());

        // Launch Message address is resolved when the program is enqueued
        constexpr uint32_t unresolved_launch_msg_addr = 0;

        if (multicast_cmds.sub_cmds.size() > 0) {
            uint32_t curr_sub_cmd_idx = 0;
            for (const auto& [num_sub_cmds_in_cmd, multicast_launch_msg_payload_sizeB] : multicast_cmds.payload) {
                uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
                device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_LAUNCH,
                    num_sub_cmds_in_cmd,
                    unresolved_launch_msg_addr,
                    aligned_launch_msg_sizeB,
                    multicast_launch_msg_payload_sizeB,
                    multicast_cmds.sub_cmds,
                    multicast_cmds.data,
                    constants.packed_write_max_unicast_sub_cmds,
                    curr_sub_cmd_idx);
                curr_sub_cmd_idx += num_sub_cmds_in_cmd;
                program_command_sequence.launch_msg_write_packed_cmd_ptrs.push_back(
                    &((CQDispatchCmd*)((uint32_t*)device_command_sequence.data() +
                                       (write_offset_bytes + sizeof(CQPrefetchCmd)) / sizeof(uint32_t)))
                         ->write_packed);
                uint32_t curr_sub_cmd_data_offset_words =
                    (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                     tt::align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedMulticastSubCmd), l1_alignment)) /
                    sizeof(uint32_t);
                for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                    program_command_sequence.launch_messages.push_back(
                        (launch_msg_t*)((uint32_t*)device_command_sequence.data() + curr_sub_cmd_data_offset_words));
                    curr_sub_cmd_data_offset_words += launch_msg_size_words;
                }
            }
        }

        if (unicast_cmds.sub_cmds.size() > 0) {
            uint32_t curr_sub_cmd_idx = 0;
            for (const auto& [num_sub_cmds_in_cmd, unicast_launch_msg_payload_sizeB] : unicast_cmds.payload) {
                uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
                device_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_LAUNCH,
                    num_sub_cmds_in_cmd,
                    unresolved_launch_msg_addr,
                    aligned_launch_msg_sizeB,
                    unicast_launch_msg_payload_sizeB,
                    unicast_cmds.sub_cmds,
                    unicast_cmds.data,
                    constants.packed_write_max_unicast_sub_cmds,
                    curr_sub_cmd_idx);
                curr_sub_cmd_idx += num_sub_cmds_in_cmd;
                program_command_sequence.unicast_launch_msg_write_packed_cmd_ptrs.push_back(
                    &((CQDispatchCmd*)((uint32_t*)device_command_sequence.data() +
                                       (write_offset_bytes + sizeof(CQPrefetchCmd)) / sizeof(uint32_t)))
                         ->write_packed);
                uint32_t curr_sub_cmd_data_offset_words =
                    (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                     tt::align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment)) /
                    sizeof(uint32_t);
                for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                    program_command_sequence.launch_messages.push_back(
                        (launch_msg_t*)((uint32_t*)device_command_sequence.data() + curr_sub_cmd_data_offset_words));
                    curr_sub_cmd_data_offset_words += launch_msg_size_words;
                }
            }
        }
    }

    bool has_multicast_launch_cmds() const { return multicast_cmds.sub_cmds.size() > 0; }
    bool has_unicast_launch_cmds() const { return unicast_cmds.sub_cmds.size() > 0; }

private:
    template <typename T>
    struct LaunchMessageCmds {
        std::vector<std::pair<const void*, uint32_t>> data;
        std::vector<T> sub_cmds;
        std::vector<std::pair<uint32_t, uint32_t>> payload;
    };

    static constexpr uint32_t launch_msg_sizeB = sizeof(launch_msg_t);

    LaunchMessageCmds<CQDispatchWritePackedMulticastSubCmd> multicast_cmds;
    LaunchMessageCmds<CQDispatchWritePackedUnicastSubCmd> unicast_cmds;
};

class GoSignalGenerator {
public:
    // Determine the size of the go signal commands.
    void size_commands(
        DeviceCommandCalculator& calculator,
        IDevice* device,
        SubDeviceId sub_device_id,
        const ProgramTransferInfo& program_transfer_info) {
        // if dispatch_s is enabled have dispatch_d send a semaphore update to dispatch_s (this will include a write
        // barrier on dispatch_d if program is active) if not,  check if the program is active on workers. If active,
        // have dispatch_d issue a write barrier either dispatch_s or dispatch_d will send the go signal
        // (go_signal_mcast command)
        if (tt_metal::MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            calculator.add_notify_dispatch_s_go_signal_cmd();
        } else {
            // Wait Noc Write Barrier, wait for binaries/configs and launch_msg to be written to worker cores
            if (program_transfer_info.num_active_cores > 0) {
                calculator.add_dispatch_wait();
            }
        }
        calculator.add_dispatch_go_signal_mcast();
    }

    // Assemble the go signal commands into the device command sequence.
    void assemble_commands(
        ProgramCommandSequence& program_command_sequence,
        HostMemDeviceCommand& device_command_sequence,
        const CommandConstants& constants,
        IDevice* device,
        SubDeviceId sub_device_id,
        const ProgramTransferInfo& program_transfer_info,
        bool has_multicast_launch_cmds,
        bool has_unicast_launch_cmds) {
        const auto& noc_data_start_idx =
            device->noc_data_start_index(sub_device_id, has_multicast_launch_cmds, has_unicast_launch_cmds);
        const auto& num_noc_mcast_txns = has_multicast_launch_cmds ? device->num_noc_mcast_txns(sub_device_id) : 0;
        const auto& num_noc_unicast_txns = has_unicast_launch_cmds ? device->num_noc_unicast_txns(sub_device_id) : 0;
        DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
        auto sub_device_index = *sub_device_id;
        if (tt_metal::MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            // dispatch_d signals dispatch_s to send the go signal, use a barrier if there are cores active
            uint16_t index_bitmask = 0;
            index_bitmask |= 1 << sub_device_index;
            device_command_sequence.add_notify_dispatch_s_go_signal_cmd(
                program_transfer_info.num_active_cores > 0, index_bitmask);
            dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
        } else {
            // Wait Noc Write Barrier, wait for binaries/configs and launch_msg to be written to worker cores
            if (program_transfer_info.num_active_cores > 0) {
                device_command_sequence.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);
            }
        }
        go_msg_t run_program_go_signal;
        run_program_go_signal.signal = RUN_MSG_GO;
        // Dispatch X/Y resolved when the program is enqueued
        run_program_go_signal.master_x = 0;
        run_program_go_signal.master_y = 0;
        run_program_go_signal.dispatch_message_offset = MetalContext::instance()
                                                            .dispatch_mem_map(constants.dispatch_core_type)
                                                            .get_dispatch_message_update_offset(sub_device_index);
        uint32_t write_offset_bytes = device_command_sequence.write_offset_bytes();
        // Num Workers Resolved when the program is enqueued
        device_command_sequence.add_dispatch_go_signal_mcast(
            0,
            *reinterpret_cast<uint32_t*>(&run_program_go_signal),
            MetalContext::instance()
                .dispatch_mem_map(constants.dispatch_core_type)
                .get_dispatch_stream_index(sub_device_index),
            num_noc_mcast_txns,
            num_noc_unicast_txns,
            noc_data_start_idx,
            dispatcher_for_go_signal);
        program_command_sequence.mcast_go_signal_cmd_ptr =
            &((CQDispatchCmd*)((uint32_t*)device_command_sequence.data() +
                               (write_offset_bytes + sizeof(CQPrefetchCmd)) / sizeof(uint32_t)))
                 ->mcast;
    }
};

void assemble_device_commands(
    ProgramCommandSequence& program_command_sequence,
    ProgramImpl& program,
    IDevice* device,
    SubDeviceId sub_device_id) {
    CommandConstants constants;
    auto dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    constants.dispatch_core_type = dispatch_core_config.get_core_type();
    constants.noc_index = k_dispatch_downstream_noc;
    constants.max_prefetch_command_size =
        MetalContext::instance().dispatch_mem_map(constants.dispatch_core_type).max_prefetch_command_size();
    constants.packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(device);
    BatchedTransfers batched_transfers =
        assemble_runtime_args_commands(program_command_sequence, program, device, constants);

    // Assemble config buffer
    DeviceCommandCalculator program_config_buffer_calculator;

    SemphoreCommandGenerator semaphore_command_generator;
    semaphore_command_generator.size_commands(
        program, device, program_config_buffer_calculator, constants, batched_transfers);

    CircularBufferCommandGenerator circular_buffer_command_generator;
    circular_buffer_command_generator.construct_commands(device, constants, program, batched_transfers);

    BatchedTransferGenerator batched_transfer_generator;
    batched_transfer_generator.construct_commands(batched_transfers, program_config_buffer_calculator);

    program_command_sequence.program_config_buffer_command_sequence =
        HostMemDeviceCommand(program_config_buffer_calculator.write_offset_bytes());
    batched_transfer_generator.assemble_commands(
        program_command_sequence, program_command_sequence.program_config_buffer_command_sequence);
    semaphore_command_generator.assemble_unicast_commands(
        program_command_sequence.program_config_buffer_command_sequence, program, constants);
    // Ensure that we use the correct amount of space for each command sequence
    TT_ASSERT(
        program_command_sequence.program_config_buffer_command_sequence.size_bytes() ==
        program_command_sequence.program_config_buffer_command_sequence.write_offset_bytes());

    // Assemble binary
    const auto& program_transfer_info = program.get_program_transfer_info();
    if (program_transfer_info.kernel_bins.size()) {
        TT_FATAL(
            program.get_kernels_buffer(device).get(), "Expected Kernel Binary Buffer to be allocated for program.");
    }
    ProgramBinaryCommandGenerator program_binary_command_generator;
    DeviceCommandCalculator program_binary_calculator;
    program_binary_command_generator.size_commands(
        device,
        program,
        program_transfer_info,
        program.get_kernels_buffer(device),
        constants,
        program_binary_calculator);
    program_command_sequence.program_binary_command_sequence =
        HostMemDeviceCommand(program_binary_calculator.write_offset_bytes());
    program_binary_command_generator.assemble_commands(program_command_sequence.program_binary_command_sequence);
    TT_ASSERT(
        program_command_sequence.program_binary_command_sequence.size_bytes() ==
        program_command_sequence.program_binary_command_sequence.write_offset_bytes());

    // Assemble launch message
    LaunchMessageGenerator launch_message_generator;
    DeviceCommandCalculator launch_message_calculator;
    launch_message_generator.construct_commands(device, program, launch_message_calculator, constants, sub_device_id);
    program_command_sequence.launch_msg_command_sequence =
        HostMemDeviceCommand(launch_message_calculator.write_offset_bytes());
    launch_message_generator.assemble_commands(
        program_command_sequence, program_command_sequence.launch_msg_command_sequence, constants);
    TT_ASSERT(
        program_command_sequence.launch_msg_command_sequence.size_bytes() ==
        program_command_sequence.launch_msg_command_sequence.write_offset_bytes());

    // Assemble go signal
    GoSignalGenerator go_signal_generator;
    DeviceCommandCalculator go_signal_calculator;
    go_signal_generator.size_commands(go_signal_calculator, device, sub_device_id, program_transfer_info);
    program_command_sequence.go_msg_command_sequence = HostMemDeviceCommand(go_signal_calculator.write_offset_bytes());
    go_signal_generator.assemble_commands(
        program_command_sequence,
        program_command_sequence.go_msg_command_sequence,
        constants,
        device,
        sub_device_id,
        program_transfer_info,
        launch_message_generator.has_multicast_launch_cmds(),
        launch_message_generator.has_unicast_launch_cmds());
    TT_ASSERT(
        program_command_sequence.go_msg_command_sequence.size_bytes() ==
        program_command_sequence.go_msg_command_sequence.write_offset_bytes());
}

void initialize_worker_config_buf_mgr(WorkerConfigBufferMgr& config_buffer_mgr, uint32_t worker_l1_unreserved_start) {
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        uint32_t ringbuffer_size;
        if (hal.get_programmable_core_type(index) == tt::tt_metal::HalProgrammableCoreType::TENSIX) {
            ringbuffer_size =
                worker_l1_unreserved_start -
                hal.get_dev_addr(hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
        } else {
            ringbuffer_size =
                hal.get_dev_size(hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
        }
        config_buffer_mgr.init_add_buffer(
            hal.get_dev_addr(hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG),
            ringbuffer_size);
    }
    // Subtract 1 from the number of entries, so the watcher can read information (e.g. fired asserts) from the
    // previous launch message.
    config_buffer_mgr.init_add_buffer(0, launch_msg_buffer_num_entries - 1);
    // There's no ring buffer for active ethernet binaries, so keep track of them separately.
    config_buffer_mgr.init_add_buffer(0, 1);
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

        // TODO: config_buffer_mgr is stateful so code below restores original reservation state
        // pull state out of the config_buffer_mgr
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

    // TODO.  This code is needlessly complex due to enqueue program and
    // binary writing being intertwined.  Separate out writing kernel
    // binaries into program compile/finalize.  The sync below is confusing
    // and not needed (just need a barrier on DRAM write)
    if (program_binary_status != ProgramBinaryStatus::Committed) {
        // Insert a stall before writing any program configs when binaries are in flight
        dispatch_md.stall_first = true;
        dispatch_md.stall_before_program = false;
        // Wait on all previous workers before writing kernel binaries to workers
        dispatch_md.sync_count = expected_num_workers_completed;
    }

    // Remove launch buffer from config addrs, since it's not a real core.
    dispatch_md.kernel_config_addrs = std::vector<ConfigBufferEntry>(
        std::make_move_iterator(reservation.second.begin()), std::make_move_iterator(reservation.second.end() - 2));
}

void update_program_dispatch_commands(
    ProgramImpl& program,
    ProgramCommandSequence& cached_program_command_sequence,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    CoreType dispatch_core_type,
    SubDeviceId sub_device_id,
    const ProgramDispatchMetadata& dispatch_md,
    ProgramBinaryStatus program_binary_status,
    std::pair<bool, int> unicast_go_signal_update) {
    uint32_t i = 0;
    ZoneScopedN("program_loaded_on_device");

    static constexpr uint32_t wait_count_offset = (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, wait.count));
    static constexpr uint32_t tensix_l1_write_offset_offset =
        (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.offset1));
    static constexpr uint32_t eth_l1_write_offset_offset =
        (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.offset2));
    static constexpr uint32_t program_host_id_offset =
        (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.program_host_id));
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
    const auto& hal = MetalContext::instance().hal();
    cached_program_command_sequence.preamble_command_sequence.update_cmd_sequence(
        tensix_l1_write_offset_offset,
        &dispatch_md.kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)],
        sizeof(uint32_t));
    // May truncate to fit the space.
    static_assert(
        std::is_same_v<uint16_t, decltype(std::declval<CQDispatchCmd>().set_write_offset.program_host_id)>,
        "program_host_id type should be uint16_t");
    uint16_t runtime_id = program.get_runtime_id();
    cached_program_command_sequence.preamble_command_sequence.update_cmd_sequence(
        program_host_id_offset, &runtime_id, sizeof(runtime_id));
    if (hal.get_programmable_core_type_count() >= 2) {
        cached_program_command_sequence.preamble_command_sequence.update_cmd_sequence(
            eth_l1_write_offset_offset,
            &dispatch_md.kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)],
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
    // Update RTAs.
    for (auto& rta_update : cached_program_command_sequence.rta_updates) {
        memcpy(rta_update.dst, rta_update.src, rta_update.size);
    }

    // Update launch messages
    for (auto& launch_msg : cached_program_command_sequence.launch_messages) {
        for (uint32_t i = 0; i < dispatch_md.kernel_config_addrs.size(); i++) {
            launch_msg->kernel_config.kernel_config_base[i] = dispatch_md.kernel_config_addrs[i].addr;
        }
        launch_msg->kernel_config.host_assigned_id = program.get_runtime_id();
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
    run_program_go_signal.dispatch_message_offset = MetalContext::instance()
                                                        .dispatch_mem_map(dispatch_core_type)
                                                        .get_dispatch_message_update_offset(*sub_device_id);
    cached_program_command_sequence.mcast_go_signal_cmd_ptr->go_signal =
        *reinterpret_cast<uint32_t*>(&run_program_go_signal);
    cached_program_command_sequence.mcast_go_signal_cmd_ptr->wait_count = expected_num_workers_completed;
    // Update the number of unicast txns based on user provided parameter
    // This is required when a MeshWorkload uses ethernet cores on a set of devices
    // where the number of active eth cores is heterogenous across devices.
    // Update the number of unicast txns to eth cores to match the minimum number of cores
    // across devices (specified by user)
    if (unicast_go_signal_update.first) {
        TT_FATAL(
            unicast_go_signal_update.second > 0,
            "Must specify a valid number of cores to unicast the go signal to when updating dispatch commands");
        cached_program_command_sequence.mcast_go_signal_cmd_ptr->num_unicast_txns = unicast_go_signal_update.second;
    }
}

void write_program_command_sequence(
    const ProgramCommandSequence& program_command_sequence,
    SystemMemoryManager& manager,
    uint32_t command_queue_id,
    CoreType dispatch_core_type,
    bool stall_first,
    bool stall_before_program) {
    // Check if it's possible to write all commands in a single fetch queue entry
    uint32_t one_shot_fetch_size = program_command_sequence.get_one_shot_fetch_size(stall_first, stall_before_program);
    bool one_shot = one_shot_fetch_size <=
                    MetalContext::instance().dispatch_mem_map(dispatch_core_type).max_prefetch_command_size();
    if (one_shot) {
        manager.issue_queue_reserve(one_shot_fetch_size, command_queue_id);
    }
    uint32_t one_shot_write_ptr = manager.get_issue_queue_write_ptr(command_queue_id);

    auto write_data_to_cq = [&](void* data, uint32_t size_bytes) {
        if (!size_bytes) {
            return;
        }

        if (one_shot) {
            // Already reserved. Write only. Defer push back until all commands are written
            manager.cq_write(data, size_bytes, one_shot_write_ptr);
            one_shot_write_ptr += size_bytes;
        } else {
            manager.issue_queue_reserve(size_bytes, command_queue_id);
            manager.cq_write(data, size_bytes, manager.get_issue_queue_write_ptr(command_queue_id));
            manager.issue_queue_push_back(size_bytes, command_queue_id);
            manager.fetch_queue_reserve_back(command_queue_id);
            manager.fetch_queue_write(size_bytes, command_queue_id);
        }
    };

    // Write the preamble
    write_data_to_cq(
        program_command_sequence.preamble_command_sequence.data(),
        program_command_sequence.preamble_command_sequence.size_bytes());

    const auto curr_stall_seq_idx = program_command_sequence.current_stall_seq_idx;
    if (stall_first) {
        // Must stall before writing kernel config data
        write_data_to_cq(
            program_command_sequence.stall_command_sequences[curr_stall_seq_idx].data(),
            program_command_sequence.stall_command_sequences[curr_stall_seq_idx].size_bytes());
    }

    // TODO: We can pack multiple RT args into one fetch q entry
    for (const auto& cmds : program_command_sequence.runtime_args_command_sequences) {
        write_data_to_cq(cmds.data(), cmds.size_bytes());
    }

    // Write the program config buffer
    write_data_to_cq(
        program_command_sequence.program_config_buffer_command_sequence.data(),
        program_command_sequence.program_config_buffer_command_sequence.size_bytes());

    // Need to stall before writing the program binary?
    if (stall_before_program) {
        // Didn't stall before kernel config data, stall before remaining commands
        write_data_to_cq(
            program_command_sequence.stall_command_sequences[curr_stall_seq_idx].data(),
            program_command_sequence.stall_command_sequences[curr_stall_seq_idx].size_bytes());
    }

    // Write the program binary
    write_data_to_cq(
        program_command_sequence.program_binary_command_sequence.data(),
        program_command_sequence.program_binary_command_sequence.size_bytes());

    // Write the launch message
    write_data_to_cq(
        program_command_sequence.launch_msg_command_sequence.data(),
        program_command_sequence.launch_msg_command_sequence.size_bytes());

    // Write the go signal
    write_data_to_cq(
        program_command_sequence.go_msg_command_sequence.data(),
        program_command_sequence.go_msg_command_sequence.size_bytes());

    if (one_shot) {
        manager.issue_queue_push_back(one_shot_fetch_size, command_queue_id);
        manager.fetch_queue_reserve_back(command_queue_id);
        manager.fetch_queue_write(one_shot_fetch_size, command_queue_id);
    }
}

KernelHandle get_device_local_kernel_handle(KernelHandle kernel_handle) {
    // Device local Kernel Handle/Kernel Ids are 16 bit. The top 16 bits of
    // the Kernel Handle may encode device coordinates when MeshWorkloads are
    // being dispatched.
    return kernel_handle & 0xffff;
}

template <typename WorkloadType, typename DeviceType>
uint32_t program_base_addr_on_core(
    WorkloadType& workload, DeviceType generic_device, HalProgrammableCoreType programmable_core_type) {
    uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(programmable_core_type);
    const auto& sub_device_ids = workload.determine_sub_device_ids(generic_device);
    // TODO: This restriction can be lifted once this function is changed to return a vector of addresses
    // Addresses are not the same across sub-devices
    TT_FATAL(
        sub_device_ids.size() == 1, "get_sem_base_addr currently only supports programs spanning a single sub-device");
    auto sub_device_index = **sub_device_ids.begin();
    auto cq = workload.get_last_used_command_queue();
    return cq ? (cq->get_config_buffer_mgr(sub_device_index).get_last_slot_addr(programmable_core_type))
              : MetalContext::instance().hal().get_dev_addr(programmable_core_type, HalL1MemAddrType::KERNEL_CONFIG);
}

void reset_config_buf_mgrs_and_expected_workers(
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgrs,
    DispatchArray<uint32_t>& expected_num_workers_completed,
    uint32_t num_entries_to_reset,
    uint32_t worker_l1_unreserved_start) {
    for (uint32_t i = 0; i < num_entries_to_reset; ++i) {
        config_buffer_mgrs[i] = WorkerConfigBufferMgr();
        initialize_worker_config_buf_mgr(config_buffer_mgrs[i], worker_l1_unreserved_start);
    }
    std::fill(expected_num_workers_completed.begin(), expected_num_workers_completed.begin() + num_entries_to_reset, 0);
}

void reset_worker_dispatch_state_on_device(
    IDevice* device,
    SystemMemoryManager& manager,
    uint8_t cq_id,
    CoreCoord dispatch_core,
    const DispatchArray<uint32_t>& expected_num_workers_completed,
    bool reset_launch_msg_state) {
    auto num_sub_devices = device->num_sub_devices();
    if (reset_launch_msg_state) {
        uint32_t pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    }

    tt::tt_metal::DeviceCommandCalculator calculator;
    if (reset_launch_msg_state) {
        for (int i = 0; i < num_sub_devices; ++i) {
            calculator.add_dispatch_go_signal_mcast();
        }
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            calculator.add_notify_dispatch_s_go_signal_cmd();
        }
    }
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        if (MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher()) {
            calculator.add_dispatch_wait();
        }
        calculator.add_dispatch_wait();
    }

    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (reset_launch_msg_state) {
        if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            uint16_t index_bitmask = 0;
            for (uint32_t i = 0; i < num_sub_devices; ++i) {
                index_bitmask |= 1 << i;
            }
            command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
            dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SLAVE;
        }
        go_msg_t reset_launch_message_read_ptr_go_signal;
        reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
        reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)dispatch_core.x;
        reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)dispatch_core.y;
        for (uint32_t i = 0; i < num_sub_devices; ++i) {
            reset_launch_message_read_ptr_go_signal.dispatch_message_offset =
                MetalContext::instance().dispatch_mem_map().get_dispatch_message_update_offset(i);
            // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
            SubDeviceId sub_device_id(static_cast<uint8_t>(i));
            command_sequence.add_dispatch_go_signal_mcast(
                expected_num_workers_completed[i],
                *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal),
                MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(i),
                device->num_noc_mcast_txns(sub_device_id),
                device->num_noc_unicast_txns(sub_device_id),
                device->noc_data_start_index(sub_device_id),
                dispatcher_for_go_signal);
        }
    }
    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed
    // this step, before sending kernel config data to workers or notifying dispatch_s that its safe to send the
    // go_signal. Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        SubDeviceId sub_device_id(static_cast<uint8_t>(i));
        uint32_t expected_num_workers = expected_num_workers_completed[i];
        if (reset_launch_msg_state) {
            expected_num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id) +
                                    device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
        if (MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM,
                0,
                MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(i),
                expected_num_workers,
                1);
        }
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM,
            0,
            MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(i),
            expected_num_workers);
    }
    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void set_num_worker_sems_on_dispatch(
    IDevice* /*device*/, SystemMemoryManager& manager, uint8_t cq_id, uint32_t num_worker_sems) {
    // Not needed for regular dispatch kernel
    if (!MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        return;
    }
    tt::tt_metal::DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_num_worker_sems();
    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();
    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    command_sequence.add_dispatch_set_num_worker_sems(num_worker_sems, DispatcherSelect::DISPATCH_SLAVE);
    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void set_go_signal_noc_data_on_dispatch(
    IDevice* /*device*/,
    const vector_aligned<uint32_t>& go_signal_noc_data,
    SystemMemoryManager& manager,
    uint8_t cq_id) {
    tt::tt_metal::DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_go_signal_noc_data(go_signal_noc_data.size());
    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();
    void* cmd_region = manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    DispatcherSelect dispatcher_for_go_signal =
        MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled() ? DispatcherSelect::DISPATCH_SLAVE
                                                                                   : DispatcherSelect::DISPATCH_MASTER;
    command_sequence.add_dispatch_set_go_signal_noc_data(go_signal_noc_data, dispatcher_for_go_signal);
    manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    manager.fetch_queue_reserve_back(cq_id);
    manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

template uint32_t program_base_addr_on_core<ProgramImpl, IDevice*>(ProgramImpl&, IDevice*, HalProgrammableCoreType);
template uint32_t program_base_addr_on_core<distributed::MeshWorkload, distributed::MeshDevice*>(
    distributed::MeshWorkload&, distributed::MeshDevice*, HalProgrammableCoreType);
}  // namespace program_dispatch

}  // namespace tt::tt_metal
