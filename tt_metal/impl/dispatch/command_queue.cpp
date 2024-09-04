// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <malloc.h>

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "allocator/allocator.hpp"
#include "debug_tools.hpp"
#include "dev_msgs.h"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/impl/event/event.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;

std::mutex finish_mutex;
std::condition_variable finish_cv;

namespace tt::tt_metal {

namespace detail {

bool DispatchStateCheck(bool isFastDispatch) {
    static bool fd = isFastDispatch;
    TT_FATAL(fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!");
    return fd;
}

void SetLazyCommandQueueMode(bool lazy) {
    DispatchStateCheck(true);
    LAZY_COMMAND_QUEUE_MODE = lazy;
}
}  // namespace detail

enum DispatchWriteOffsets {
    DISPATCH_WRITE_OFFSET_ZERO = 0,
    DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE = 1,
    DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE = 2,
};

// TODO: Delete entries when programs are deleted to save memory
thread_local std::unordered_map<uint64_t, EnqueueProgramCommand::CachedProgramCommandSequence>
    EnqueueProgramCommand::cached_program_command_sequences = {};

// EnqueueReadBufferCommandSection

EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    Buffer& buffer,
    void* dst,
    SystemMemoryManager& manager,
    uint32_t expected_num_workers_completed,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    dst(dst),
    manager(manager),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    src_page_index(src_page_index),
    pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {
    TT_ASSERT(buffer.is_dram() or buffer.is_l1(), "Trying to read an invalid buffer");

    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void EnqueueReadInterleavedBufferCommand::add_prefetch_relay(HugepageDeviceCommand& command) {
    uint32_t padded_page_size = this->buffer.aligned_page_size();
    command.add_prefetch_relay_paged(
        this->buffer.is_dram(), this->src_page_index, this->buffer.address(), padded_page_size, this->pages_to_read);
}

void EnqueueReadShardedBufferCommand::add_prefetch_relay(HugepageDeviceCommand& command) {
    uint32_t padded_page_size = this->buffer.aligned_page_size();
    const CoreCoord physical_core =
        this->buffer.device()->physical_core_from_logical_core(this->core, this->buffer.core_type());
    command.add_prefetch_relay_linear(
        this->device->get_noc_unicast_encoding(this->noc_index, physical_core),
        padded_page_size * this->pages_to_read,
        this->bank_base_address);
}

void EnqueueReadBufferCommand::process() {
    // accounts for padding
    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_STALL
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait_with_prefetch_stall(
        true, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);

    uint32_t padded_page_size = this->buffer.aligned_page_size();
    bool flush_prefetch = false;
    command_sequence.add_dispatch_write_host(flush_prefetch, this->pages_to_read * padded_page_size, false);

    this->add_prefetch_relay(command_sequence);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

// EnqueueWriteBufferCommand section

EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    const Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    bool issue_wait,
    uint32_t expected_num_workers_completed,
    uint32_t bank_base_address,
    uint32_t padded_page_size,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    manager(manager),
    issue_wait(issue_wait),
    src(src),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    bank_base_address(bank_base_address),
    padded_page_size(padded_page_size),
    dst_page_index(dst_page_index),
    pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(buffer.is_dram() or buffer.is_l1(), "Trying to write to an invalid buffer");
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void EnqueueWriteInterleavedBufferCommand::add_dispatch_write(HugepageDeviceCommand& command_sequence) {
    uint8_t is_dram = uint8_t(this->buffer.is_dram());
    TT_ASSERT(
        this->dst_page_index <= 0xFFFF,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    uint16_t start_page = uint16_t(this->dst_page_index & 0xFFFF);
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch, is_dram, start_page, this->bank_base_address, this->padded_page_size, this->pages_to_write);
}

void EnqueueWriteInterleavedBufferCommand::add_buffer_data(HugepageDeviceCommand& command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;

    uint32_t full_page_size = this->buffer.aligned_page_size();  // this->padded_page_size could be a partial page if
                                                                 // buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = this->padded_page_size < full_page_size;

    uint32_t buffer_addr_offset = this->bank_base_address - this->buffer.address();
    uint32_t num_banks = this->device->num_banks(this->buffer.buffer_type());

    // TODO: Consolidate
    if (write_partial_pages) {
        uint32_t padding = full_page_size - this->buffer.page_size();
        uint32_t unpadded_src_offset = buffer_addr_offset;
        uint32_t src_address_offset = unpadded_src_offset;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
             sysmem_address_offset += this->padded_page_size) {
            uint32_t page_size_to_copy = this->padded_page_size;
            if (src_address_offset + this->padded_page_size > buffer.page_size()) {
                // last partial page being copied from unpadded src buffer
                page_size_to_copy -= padding;
            }
            command_sequence.add_data((char*)this->src + src_address_offset, page_size_to_copy, this->padded_page_size);
            src_address_offset += page_size_to_copy;
        }
    } else {
        uint32_t unpadded_src_offset =
            (((buffer_addr_offset / this->padded_page_size) * num_banks) + this->dst_page_index) *
            this->buffer.page_size();
        if (this->buffer.page_size() % this->buffer.alignment() != 0 and
            this->buffer.page_size() != this->buffer.size()) {
            // If page size is not aligned, we cannot do a contiguous write
            uint32_t src_address_offset = unpadded_src_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                 sysmem_address_offset += this->padded_page_size) {
                command_sequence.add_data(
                    (char*)this->src + src_address_offset, this->buffer.page_size(), this->padded_page_size);
                src_address_offset += this->buffer.page_size();
            }
        } else {
            command_sequence.add_data((char*)this->src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void EnqueueWriteShardedBufferCommand::add_dispatch_write(HugepageDeviceCommand& command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    const CoreCoord physical_core =
        this->buffer.device()->physical_core_from_logical_core(this->core, this->buffer.core_type());
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_linear(
        flush_prefetch,
        0,
        this->device->get_noc_unicast_encoding(this->noc_index, physical_core),
        this->bank_base_address,
        data_size_bytes);
}

void EnqueueWriteShardedBufferCommand::add_buffer_data(HugepageDeviceCommand& command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    if (this->buffer_page_mapping.has_value()) {
        const auto& page_mapping = this->buffer_page_mapping.value();
        uint8_t* dst = command_sequence.reserve_space<uint8_t*, true>(data_size_bytes);
        // TODO: Expose getter for cmd_write_offsetB?
        uint32_t dst_offset = dst - (uint8_t*)command_sequence.data();
        for (uint32_t dev_page = this->dst_page_index; dev_page < this->dst_page_index + this->pages_to_write;
             ++dev_page) {
            auto& host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                command_sequence.update_cmd_sequence(
                    dst_offset,
                    (char*)this->src + host_page.value() * this->buffer.page_size(),
                    this->buffer.page_size());
            }
            dst_offset += this->padded_page_size;
        }
    } else {
        if (this->buffer.page_size() != this->padded_page_size and this->buffer.page_size() != this->buffer.size()) {
            uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
            for (uint32_t i = 0; i < this->pages_to_write; ++i) {
                command_sequence.add_data(
                    (char*)this->src + unpadded_src_offset, this->buffer.page_size(), this->padded_page_size);
                unpadded_src_offset += this->buffer.page_size();
            }
        } else {
            uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
            command_sequence.add_data((char*)this->src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void EnqueueWriteBufferCommand::process() {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;

    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + (CQ_DISPATCH_CMD_WRITE_PAGED or
                                         // CQ_DISPATCH_CMD_WRITE_LINEAR)
        data_size_bytes;
    if (this->issue_wait) {
        cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (this->issue_wait) {
        command_sequence.add_dispatch_wait(false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);
    }

    this->add_dispatch_write(command_sequence);

    uint32_t full_page_size = this->buffer.aligned_page_size();  // this->padded_page_size could be a partial page if
                                                                 // buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = this->padded_page_size < full_page_size;

    this->add_buffer_data(command_sequence);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

inline uint32_t get_packed_write_max_unicast_sub_cmds(Device* device) {
    return uint32_t(device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);
}

// EnqueueProgramCommand Section

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    Program& program,
    CoreCoord& dispatch_core,
    SystemMemoryManager& manager,
    uint32_t expected_num_workers_completed) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    manager(manager),
    expected_num_workers_completed(expected_num_workers_completed),
    program(program),
    dispatch_core(dispatch_core) {
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    this->packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
}

void EnqueueProgramCommand::assemble_preamble_commands(std::vector<ConfigBufferEntry>& kernel_config_addrs) {
    constexpr uint32_t uncached_cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_SET_WRITE_OFFSET

    this->cached_program_command_sequences[program.id].preamble_command_sequence =
        HostMemDeviceCommand(uncached_cmd_sequence_sizeB);

    // Send write offsets
    if (hal.get_programmable_core_type_count() >= 2) {
        this->cached_program_command_sequences[program.id].preamble_command_sequence.add_dispatch_set_write_offsets(
            0,
            kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)].addr,
            kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)].addr);
    } else {
        this->cached_program_command_sequences[program.id].preamble_command_sequence.add_dispatch_set_write_offsets(
            0, kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)].addr, 0);
    }
}

void EnqueueProgramCommand::assemble_stall_commands(bool prefetch_stall) {
    if (prefetch_stall) {
        // Wait command so previous program finishes
        // Wait command with barrier for binaries to commit to DRAM
        // Prefetch stall to prevent prefetcher picking up incomplete binaries from DRAM
        constexpr uint32_t uncached_cmd_sequence_sizeB =
            CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_STALL

        this->cached_program_command_sequences[program.id].stall_command_sequence =
            HostMemDeviceCommand(uncached_cmd_sequence_sizeB);

        // Wait for Noc Write Barrier
        // wait for binaries to commit to dram, also wait for previous program to be done
        // Wait Noc Write Barrier, wait for binaries to be written to worker cores
        // Stall to allow binaries to commit to DRAM first
        // TODO: this can be removed for all but the first program run
        this->cached_program_command_sequences[program.id].stall_command_sequence.add_dispatch_wait_with_prefetch_stall(
            true, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);
    } else {
        // Wait command so previous program finishes
        constexpr uint32_t cached_cmd_sequence_sizeB =
            CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

        this->cached_program_command_sequences[program.id].stall_command_sequence =
            HostMemDeviceCommand(cached_cmd_sequence_sizeB);
        this->cached_program_command_sequences[program.id].stall_command_sequence.add_dispatch_wait(
            false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);
    }
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
    uint32_t max_prefetch_size =
        max_prefetch_cmd_size - sizeof(CQPrefetchCmd) - PCIE_ALIGNMENT - sizeof(CQDispatchCmd) - L1_ALIGNMENT;
    uint32_t max_prefetch_num_packed_cmds =
        no_stride ? (max_prefetch_size - align(data_size * sizeof(uint32_t), L1_ALIGNMENT)) / sub_cmd_sizeB
                  : max_prefetch_size / (align(data_size * sizeof(uint32_t), L1_ALIGNMENT) + sub_cmd_sizeB);

    uint32_t packed_write_max_multicast_sub_cmds =
        get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);
    return std::min(
        max_prefetch_num_packed_cmds,
        is_unicast ? packed_write_max_unicast_sub_cmds : packed_write_max_multicast_sub_cmds);
};

template <typename PackedSubCmd>
uint32_t insert_write_packed_payloads(
    const uint32_t num_sub_cmds,
    const uint32_t sub_cmd_sizeB,
    const uint32_t max_prefetch_command_size,
    const uint32_t packed_write_max_unicast_sub_cmds,
    std::vector<std::pair<uint32_t, uint32_t>>& packed_cmd_payloads) {
    const uint32_t aligned_sub_cmd_sizeB = align(sub_cmd_sizeB, L1_ALIGNMENT);
    const uint32_t max_packed_sub_cmds_per_cmd = get_max_write_packed_sub_cmds<PackedSubCmd>(
        aligned_sub_cmd_sizeB, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, false);
    uint32_t rem_num_sub_cmds = num_sub_cmds;
    uint32_t cmd_payload_sizeB = 0;
    while (rem_num_sub_cmds != 0) {
        const uint32_t num_sub_cmds_in_cmd = std::min(max_packed_sub_cmds_per_cmd, rem_num_sub_cmds);
        const uint32_t aligned_data_sizeB = aligned_sub_cmd_sizeB * num_sub_cmds_in_cmd;
        const uint32_t dispatch_cmd_sizeB =
            align(sizeof(CQDispatchCmd) + num_sub_cmds_in_cmd * sizeof(PackedSubCmd), L1_ALIGNMENT);
        packed_cmd_payloads.emplace_back(num_sub_cmds_in_cmd, dispatch_cmd_sizeB + aligned_data_sizeB);
        cmd_payload_sizeB += align(sizeof(CQPrefetchCmd) + packed_cmd_payloads.back().second, PCIE_ALIGNMENT);
        rem_num_sub_cmds -= num_sub_cmds_in_cmd;
    }
    return cmd_payload_sizeB;
}

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
            uint32_t sub_cmd_sizeB =
                is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB = sizeof(CQDispatchCmd) + align(num_packed_cmds * sub_cmd_sizeB, L1_ALIGNMENT);
            uint32_t aligned_runtime_data_sizeB =
                (no_stride ? 1 : num_packed_cmds) * align(runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT);
            return dispatch_cmd_sizeB + aligned_runtime_data_sizeB;
        };
    thread_local static auto get_runtime_args_data_offset =
        [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast) {
            uint32_t sub_cmd_sizeB =
                is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB = sizeof(CQDispatchCmd) + align(num_packed_cmds * sub_cmd_sizeB, L1_ALIGNMENT);
            return sizeof(CQPrefetchCmd) + dispatch_cmd_sizeB;
        };

    constexpr bool unicast = std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value;

    uint32_t num_packed_cmds_in_seq = sub_cmds.size();
    uint32_t max_packed_cmds = get_max_write_packed_sub_cmds<PackedSubCmd>(
        max_runtime_args_len, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, no_stride);
    uint32_t offset_idx = 0;
    if (no_stride) {
        TT_FATAL(max_packed_cmds >= num_packed_cmds_in_seq);
    }
    while (num_packed_cmds_in_seq != 0) {
        // Generate the device command
        uint32_t num_packed_cmds = std::min(num_packed_cmds_in_seq, max_packed_cmds);
        uint32_t rt_payload_sizeB =
            get_runtime_payload_sizeB(num_packed_cmds, max_runtime_args_len, unicast, no_stride);
        uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + rt_payload_sizeB, PCIE_ALIGNMENT);
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
        const uint32_t data_inc = align(max_runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT);
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

// Generate command sequence for unique (unicast) and common (multicast) runtime args
void EnqueueProgramCommand::assemble_runtime_args_commands() {
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();

    // Note: each sub_cmd contain data for multiple kernels (DM*, COMPUTE)
    // the outer vector counts through the kernels, the inner vectors contains the data for each kernel
    std::vector<CQDispatchWritePackedUnicastSubCmd> unique_sub_cmds;
    std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>> unique_rt_data_and_sizes;
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>> unique_rt_args_data;

    std::variant<std::vector<CQDispatchWritePackedMulticastSubCmd>, std::vector<CQDispatchWritePackedUnicastSubCmd>>
        common_sub_cmds;
    std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>> common_rt_data_and_sizes;
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>> common_rt_args_data;

    this->cached_program_command_sequences[program.id].runtime_args_command_sequences = {};

    uint32_t command_count = 0;
    for (uint32_t programmable_core_type_index = 0;
         programmable_core_type_index < hal.get_programmable_core_type_count();
         programmable_core_type_index++) {
        for (auto& kg : program.get_kernel_groups(programmable_core_type_index)) {
            if (kg.total_rta_size != 0) {
                // Reserve 2x for unique rtas as we pontentially split the cmds due to not fitting in one prefetch cmd
                command_count += 2;
            }
        }
        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            uint32_t common_size = program.get_program_config(programmable_core_type_index).crta_sizes[dispatch_class];
            if (common_size != 0) {
                command_count++;
            }
        }
    }

    this->cached_program_command_sequences[program.id].runtime_args_command_sequences.reserve(command_count);
    // Unique Runtime Args (Unicast)
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        if (hal.get_programmable_core_type(index) == HalProgrammableCoreType::IDLE_ETH) {
            // Fast dispatch not supported on IDLE_ETH yet
            // TODO: can't just loop here as code below confuses ACTIVE/IDLE
            continue;
        }
        CoreType core_type = hal.get_core_type(index);

        for (auto& kg : program.get_kernel_groups(index)) {
            if (kg.total_rta_size != 0) {
                for (const CoreRange& core_range : kg.core_ranges.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            CoreCoord core_coord(x, y);

                            unique_rt_args_data.resize(unique_rt_args_data.size() + 1);
                            unique_rt_data_and_sizes.resize(unique_rt_data_and_sizes.size() + 1);
                            for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
                                auto& optional_id = kg.kernel_ids[dispatch_class];
                                if (optional_id) {
                                    auto kernel = detail::GetKernel(program, optional_id.value());
                                    if (!kernel->cores_with_runtime_args().empty()) {
                                        const auto& runtime_args_data = kernel->runtime_args(core_coord);
                                        unique_rt_args_data.back().emplace_back(kernel->runtime_args_data(core_coord));
                                        TT_ASSERT(
                                            runtime_args_data.size() * sizeof(uint32_t) <=
                                            kg.rta_sizes[dispatch_class]);
                                        unique_rt_data_and_sizes.back().emplace_back(
                                            runtime_args_data.data(),
                                            runtime_args_data.size() * sizeof(uint32_t),
                                            kg.rta_sizes[dispatch_class]);
                                    }
                                }
                            }

                            CoreCoord physical_core = device->physical_core_from_logical_core(core_coord, core_type);
                            unique_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, physical_core)});
                        }
                    }
                }
                uint32_t rta_offset = program.get_program_config(index).rta_offset;
                generate_runtime_args_cmds(
                    this->cached_program_command_sequences[program.id].runtime_args_command_sequences,
                    rta_offset,
                    unique_sub_cmds,
                    unique_rt_data_and_sizes,
                    kg.total_rta_size / sizeof(uint32_t),
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

        for (int dispatch_class = 0; dispatch_class < DISPATCH_CLASS_MAX; dispatch_class++) {
            uint32_t common_size = program.get_program_config(index).crta_sizes[dispatch_class];
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
                            CoreCoord physical_core = device->ethernet_core_from_logical_core(core_coord);
                            unicast_sub_cmd.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, physical_core)});
                        }
                    } else {
                        vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                            device->extract_dst_noc_multicast_info<std::vector<CoreRange>>(
                                kernel->logical_coreranges(), core_type);
                        common_sub_cmds.emplace<std::vector<CQDispatchWritePackedMulticastSubCmd>>(
                            std::vector<CQDispatchWritePackedMulticastSubCmd>());
                        auto& multicast_sub_cmd =
                            std::get<std::vector<CQDispatchWritePackedMulticastSubCmd>>(common_sub_cmds);
                        multicast_sub_cmd.reserve(dst_noc_multicast_info.size());
                        for (const auto& mcast_dests : dst_noc_multicast_info) {
                            multicast_sub_cmd.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                                .noc_xy_addr = this->device->get_noc_multicast_encoding(
                                    this->noc_index, std::get<CoreRange>(mcast_dests.first)),
                                .num_mcast_dests = mcast_dests.second});
                        }
                    }
                }
            }

            if (common_size != 0) {
                uint32_t crta_offset = program.get_program_config(index).crta_offsets[dispatch_class];

                // Common rtas are always expected to fit in one prefetch cmd
                // TODO: use a linear write instead of a packed-write
                std::visit(
                    [&](auto&& sub_cmds) {
                        generate_runtime_args_cmds(
                            this->cached_program_command_sequences[program.id].runtime_args_command_sequences,
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
            }

            for (auto& data_per_kernel : common_rt_data_and_sizes) {
                for (auto& data_and_sizes : data_per_kernel) {
                    RecordDispatchData(program, DISPATCH_DATA_RTARGS, std::get<1>(data_and_sizes));
                }
            }
            common_rt_data_and_sizes.clear();
            common_rt_args_data.clear();
        }
    }

    uint32_t runtime_args_fetch_size_bytes = 0;
    for (const auto& cmds : this->cached_program_command_sequences[program.id].runtime_args_command_sequences) {
        // BRISC, NCRISC, TRISC...
        runtime_args_fetch_size_bytes += cmds.size_bytes();
    }
    this->cached_program_command_sequences[program.id].runtime_args_fetch_size_bytes = runtime_args_fetch_size_bytes;
}

void EnqueueProgramCommand::assemble_device_commands(
    bool is_cached, std::vector<ConfigBufferEntry>& kernel_config_addrs) {
    auto& cached_program_command_sequence = this->cached_program_command_sequences[this->program.id];
    if (not is_cached) {
        // Calculate size of command and fill program indices of data to update
        // TODO: Would be nice if we could pull this out of program
        uint32_t cmd_sequence_sizeB = 0;
        const uint32_t max_prefetch_command_size =
            dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();

        // Multicast Semaphore Cmd
        uint32_t num_multicast_semaphores = program.program_transfer_info.multicast_semaphores.size();
        std::vector<std::vector<CQDispatchWritePackedMulticastSubCmd>> multicast_sem_sub_cmds(num_multicast_semaphores);
        std::vector<std::vector<std::pair<const void*, uint32_t>>> multicast_sem_data(num_multicast_semaphores);
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> multicast_sem_payload(num_multicast_semaphores);
        std::vector<std::pair<uint32_t, uint32_t>> multicast_sem_dst_size;
        multicast_sem_dst_size.reserve(num_multicast_semaphores);
        if (num_multicast_semaphores > 0) {
            uint32_t i = 0;
            for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.multicast_semaphores) {
                // TODO: loop over things inside transfer_info[i]
                uint32_t write_packed_len = transfer_info_vec[0].data.size();
                multicast_sem_dst_size.emplace_back(std::make_pair(dst, write_packed_len * sizeof(uint32_t)));

                for (const auto& transfer_info : transfer_info_vec) {
                    for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                        TT_ASSERT(
                            transfer_info.data.size() == write_packed_len,
                            "Not all data vectors in write packed semaphore cmd equal in len");
                        multicast_sem_sub_cmds[i].emplace_back(CQDispatchWritePackedMulticastSubCmd{
                            .noc_xy_addr = this->device->get_noc_multicast_encoding(
                                this->noc_index, std::get<CoreRange>(dst_noc_info.first)),
                            .num_mcast_dests = dst_noc_info.second});
                        multicast_sem_data[i].emplace_back(
                            transfer_info.data.data(), transfer_info.data.size() * sizeof(uint32_t));
                    }
                }
                cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
                    multicast_sem_sub_cmds[i].size(),
                    multicast_sem_dst_size.back().second,
                    max_prefetch_command_size,
                    this->packed_write_max_unicast_sub_cmds,
                    multicast_sem_payload[i]);
                i++;
            }
        }

        // Unicast Semaphore Cmd
        uint32_t num_unicast_semaphores = program.program_transfer_info.unicast_semaphores.size();
        std::vector<std::vector<CQDispatchWritePackedUnicastSubCmd>> unicast_sem_sub_cmds(num_unicast_semaphores);
        std::vector<std::vector<std::pair<const void*, uint32_t>>> unicast_sem_data(num_unicast_semaphores);
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> unicast_sem_payload(num_unicast_semaphores);
        std::vector<std::pair<uint32_t, uint32_t>> unicast_sem_dst_size;
        unicast_sem_dst_size.reserve(num_unicast_semaphores);
        if (num_unicast_semaphores > 0) {
            uint32_t i = 0;
            for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.unicast_semaphores) {
                // TODO: loop over things inside transfer_info[i]
                uint32_t write_packed_len = transfer_info_vec[0].data.size();
                unicast_sem_dst_size.emplace_back(std::make_pair(dst, write_packed_len * sizeof(uint32_t)));

                for (const auto& transfer_info : transfer_info_vec) {
                    for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                        TT_ASSERT(
                            transfer_info.data.size() == write_packed_len,
                            "Not all data vectors in write packed semaphore cmd equal in len");
                        unicast_sem_sub_cmds[i].emplace_back(CQDispatchWritePackedUnicastSubCmd{
                            .noc_xy_addr = this->device->get_noc_unicast_encoding(
                                this->noc_index, std::get<CoreCoord>(dst_noc_info.first))});
                        unicast_sem_data[i].emplace_back(
                            transfer_info.data.data(), transfer_info.data.size() * sizeof(uint32_t));
                    }
                }
                cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedUnicastSubCmd>(
                    unicast_sem_sub_cmds[i].size(),
                    unicast_sem_dst_size.back().second,
                    max_prefetch_command_size,
                    this->packed_write_max_unicast_sub_cmds,
                    unicast_sem_payload[i]);
                i++;
            }
        }

        const auto& circular_buffers_unique_coreranges = program.circular_buffers_unique_coreranges();
        const uint16_t num_multicast_cb_sub_cmds = circular_buffers_unique_coreranges.size();
        std::vector<std::pair<uint32_t, uint32_t>> mcast_cb_payload;
        uint16_t cb_config_size_bytes = 0;
        uint32_t aligned_cb_config_size_bytes = 0;
        std::vector<std::vector<uint32_t>> cb_config_payloads(
            num_multicast_cb_sub_cmds,
            std::vector<uint32_t>(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * NUM_CIRCULAR_BUFFERS, 0));
        std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_cb_config_sub_cmds;
        std::vector<std::pair<const void*, uint32_t>> multicast_cb_config_data;
        if (num_multicast_cb_sub_cmds > 0) {
            multicast_cb_config_sub_cmds.reserve(num_multicast_cb_sub_cmds);
            multicast_cb_config_data.reserve(num_multicast_cb_sub_cmds);
            cached_program_command_sequence.circular_buffers_on_core_ranges.resize(num_multicast_cb_sub_cmds);
            uint32_t i = 0;
            uint32_t max_overall_base_index = 0;
            for (const CoreRange& core_range : circular_buffers_unique_coreranges) {
                const CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start_coord);
                const CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end_coord);

                const uint32_t num_receivers = core_range.size();
                auto& cb_config_payload = cb_config_payloads[i];
                uint32_t max_base_index = 0;
                const auto& circular_buffers_on_corerange = program.circular_buffers_on_corerange(core_range);
                cached_program_command_sequence.circular_buffers_on_core_ranges[i].reserve(
                    circular_buffers_on_corerange.size());
                for (const shared_ptr<CircularBuffer>& cb : circular_buffers_on_corerange) {
                    cached_program_command_sequence.circular_buffers_on_core_ranges[i].emplace_back(cb);
                    const uint32_t cb_address = cb->address() >> 4;
                    const uint32_t cb_size = cb->size() >> 4;
                    for (const auto& buffer_index : cb->buffer_indices()) {
                        // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                        // cb config payload
                        const uint32_t base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * (uint32_t)buffer_index;
                        cb_config_payload[base_index] = cb_address;
                        cb_config_payload[base_index + 1] = cb_size;
                        cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                        cb_config_payload[base_index + 3] = cb->page_size(buffer_index) >> 4;
                        max_base_index = std::max(max_base_index, base_index);
                    }
                }
                multicast_cb_config_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                    .noc_xy_addr = this->device->get_noc_multicast_encoding(
                        this->noc_index, CoreRange(physical_start, physical_end)),
                    .num_mcast_dests = (uint32_t)core_range.size()});
                multicast_cb_config_data.emplace_back(
                    cb_config_payload.data(),
                    (max_base_index + UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG) * sizeof(uint32_t));
                max_overall_base_index = std::max(max_overall_base_index, max_base_index);
                i++;
            }
            cb_config_size_bytes =
                (max_overall_base_index + UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG) * sizeof(uint32_t);
            aligned_cb_config_size_bytes = align(cb_config_size_bytes, L1_ALIGNMENT);
            cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
                num_multicast_cb_sub_cmds,
                cb_config_size_bytes,
                max_prefetch_command_size,
                this->packed_write_max_unicast_sub_cmds,
                mcast_cb_payload);
        }

        // Program Binaries and Go Signals
        // Get launch msg data while getting size of cmds
        std::vector<std::vector<CQPrefetchRelayPagedPackedSubCmd>> kernel_bins_prefetch_subcmds;
        std::vector<std::vector<CQDispatchWritePackedLargeSubCmd>> kernel_bins_dispatch_subcmds;
        std::vector<uint32_t> kernel_bins_write_packed_large_data_aligned_sizeB;
        std::vector<HostMemDeviceCommand> kernel_bins_unicast_cmds;
        const uint32_t max_length_per_sub_cmd = dispatch_constants::get(this->dispatch_core_type).scratch_db_size() / 2;
        const uint32_t max_paged_length_per_sub_cmd =
            max_length_per_sub_cmd / HostMemDeviceCommand::PROGRAM_PAGE_SIZE * HostMemDeviceCommand::PROGRAM_PAGE_SIZE;
        for (const auto& [cores, num_mcast_dests, kg_transfer_info] : program.program_transfer_info.kernel_bins) {
            bool write_linear;
            uint32_t noc_encoding;
            std::visit(
                [&](auto&& cores) {
                    using T = std::decay_t<decltype(cores)>;
                    if constexpr (std::is_same_v<T, CoreRange>) {
                        noc_encoding = this->device->get_noc_multicast_encoding(this->noc_index, cores);
                        write_linear = false;
                    } else {
                        noc_encoding = this->device->get_noc_unicast_encoding(this->noc_index, cores);
                        write_linear = true;
                    }
                },
                cores);
            for (uint32_t kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
                if (write_linear) {
                    kernel_bins_unicast_cmds.emplace_back(2 * CQ_PREFETCH_CMD_BARE_MIN_SIZE);
                    cmd_sequence_sizeB += 2 * CQ_PREFETCH_CMD_BARE_MIN_SIZE;
                    kernel_bins_unicast_cmds.back().add_dispatch_write_linear(
                        false,            // flush_prefetch
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
                        const uint32_t num_banks = this->device->num_banks(this->program.kernels_buffer->buffer_type());
                        page_offset = kg_transfer_info.page_offsets[kernel_idx] % num_banks;
                        uint32_t num_full_pages_written_per_bank =
                            kg_transfer_info.page_offsets[kernel_idx] / num_banks;
                        base_address = this->program.kernels_buffer->address() +
                                       num_full_pages_written_per_bank * this->program.kernels_buffer->page_size();
                    } else {
                        base_address = this->program.kernels_buffer->address();
                        page_offset = kg_transfer_info.page_offsets[kernel_idx];
                    }

                    kernel_bins_unicast_cmds.back().add_prefetch_relay_paged(
                        true,  // is_dram
                        page_offset,
                        base_address,
                        this->program.kernels_buffer->page_size(),
                        relayed_bytes / this->program.kernels_buffer->page_size(),
                        length_adjust);
                } else {
                    uint32_t base_address = this->program.kernels_buffer->address();
                    uint32_t page_offset = kg_transfer_info.page_offsets[kernel_idx];
                    uint32_t dst_addr = kg_transfer_info.dst_base_addrs[kernel_idx];
                    uint32_t aligned_length = align(kg_transfer_info.lengths[kernel_idx], DRAM_ALIGNMENT);
                    uint32_t padding = aligned_length - kg_transfer_info.lengths[kernel_idx];
                    while (aligned_length != 0) {
                        if (kernel_bins_dispatch_subcmds.empty() ||
                            kernel_bins_dispatch_subcmds.back().size() ==
                                CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS) {
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
                        kernel_bins_dispatch_subcmds.back().emplace_back(CQDispatchWritePackedLargeSubCmd{
                            .noc_xy_addr = noc_encoding,
                            .addr = dst_addr,
                            .length = (uint16_t)write_length,
                            .num_mcast_dests = (uint8_t)num_mcast_dests,
                            .flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_NONE});
                        RecordDispatchData(
                            program, DISPATCH_DATA_BINARY, write_length, kg_transfer_info.riscvs[kernel_idx]);
                        dst_addr += write_length;

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
            // Unlink the last subcmd of the current core range
            if (!write_linear) {
                kernel_bins_dispatch_subcmds.back().back().flags |= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
            }
        }
        for (uint32_t i = 0; i < kernel_bins_dispatch_subcmds.size(); ++i) {
            cmd_sequence_sizeB += align(
                ((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd))) +
                    kernel_bins_dispatch_subcmds[i].size() * sizeof(CQDispatchWritePackedLargeSubCmd),
                PCIE_ALIGNMENT);
            cmd_sequence_sizeB += align(
                kernel_bins_prefetch_subcmds[i].size() * sizeof(CQPrefetchRelayPagedPackedSubCmd) +
                    sizeof(CQPrefetchCmd),
                PCIE_ALIGNMENT);
        }

        // Wait Cmd
        if (program.program_transfer_info.num_active_cores > 0) {
            cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
        }

        std::vector<std::pair<const void*, uint32_t>> multicast_go_signal_data;
        std::vector<std::pair<const void*, uint32_t>> unicast_go_signal_data;
        std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_go_signal_sub_cmds;
        std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_go_signal_sub_cmds;
        std::vector<std::pair<uint32_t, uint32_t>> multicast_go_signals_payload;
        std::vector<std::pair<uint32_t, uint32_t>> unicast_go_signals_payload;
        constexpr uint32_t go_signal_sizeB = sizeof(launch_msg_t);
        constexpr uint32_t aligned_go_signal_sizeB = align(go_signal_sizeB, L1_ALIGNMENT);
        constexpr uint32_t go_signal_size_words = aligned_go_signal_sizeB / sizeof(uint32_t);

        // TODO: eventually the code below could be structured to loop over programmable_indices
        // and check for mcast/unicast
        uint32_t programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        for (KernelGroup& kernel_group : program.get_kernel_groups(programmable_core_index)) {
            kernel_group.launch_msg.kernel_config.mode = DISPATCH_MODE_DEV;
            kernel_group.launch_msg.kernel_config.dispatch_core_x = this->dispatch_core.x;
            kernel_group.launch_msg.kernel_config.dispatch_core_y = this->dispatch_core.y;
            for (uint32_t i = 0; i < kernel_config_addrs.size(); i++) {
                kernel_group.launch_msg.kernel_config.kernel_config_base[i] = kernel_config_addrs[i].addr;
            }
            kernel_group.launch_msg.kernel_config.host_assigned_id = program.get_runtime_id();
            const void* launch_message_data = (const void*)(&kernel_group.launch_msg);
            for (const CoreRange& core_range : kernel_group.core_ranges.ranges()) {
                CoreCoord physical_start =
                    device->physical_core_from_logical_core(core_range.start_coord, kernel_group.get_core_type());
                CoreCoord physical_end =
                    device->physical_core_from_logical_core(core_range.end_coord, kernel_group.get_core_type());

                multicast_go_signal_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                    .noc_xy_addr = this->device->get_noc_multicast_encoding(
                        this->noc_index, CoreRange(physical_start, physical_end)),
                    .num_mcast_dests = (uint32_t)core_range.size()});

                multicast_go_signal_data.emplace_back(launch_message_data, go_signal_sizeB);
            }
        }
        if (multicast_go_signal_sub_cmds.size() > 0) {
            cmd_sequence_sizeB += insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
                multicast_go_signal_sub_cmds.size(),
                go_signal_sizeB,
                max_prefetch_command_size,
                this->packed_write_max_unicast_sub_cmds,
                multicast_go_signals_payload);
        }

        programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
        // TODO: ugly, can be fixed by looping over indices w/ some work
        if (programmable_core_index != -1) {
            for (KernelGroup& kernel_group : program.get_kernel_groups(programmable_core_index)) {
                kernel_group.launch_msg.kernel_config.mode = DISPATCH_MODE_DEV;
                kernel_group.launch_msg.kernel_config.dispatch_core_x = this->dispatch_core.x;
                kernel_group.launch_msg.kernel_config.dispatch_core_y = this->dispatch_core.y;
                for (uint32_t i = 0; i < kernel_config_addrs.size(); i++) {
                    kernel_group.launch_msg.kernel_config.kernel_config_base[i] = kernel_config_addrs[i].addr;
                }
                kernel_group.launch_msg.kernel_config.host_assigned_id = program.get_runtime_id();
                const void* launch_message_data = (const launch_msg_t*)(&kernel_group.launch_msg);
                for (const CoreRange& core_range : kernel_group.core_ranges.ranges()) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                            CoreCoord physical_coord = device->physical_core_from_logical_core(
                                CoreCoord({x, y}), kernel_group.get_core_type());
                            unicast_go_signal_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                                .noc_xy_addr =
                                    this->device->get_noc_unicast_encoding(this->noc_index, physical_coord)});
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
                this->packed_write_max_unicast_sub_cmds,
                unicast_go_signals_payload);
        }

        cached_program_command_sequence.program_command_sequence = HostMemDeviceCommand(cmd_sequence_sizeB);

        auto& program_command_sequence = cached_program_command_sequence.program_command_sequence;

        // Semaphores
        // Multicast Semaphore Cmd
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        for (uint32_t i = 0; i < num_multicast_semaphores; ++i) {
            uint32_t curr_sub_cmd_idx = 0;
            for (const auto& [num_sub_cmds_in_cmd, multicast_sem_payload_sizeB] : multicast_sem_payload[i]) {
                program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                    num_sub_cmds_in_cmd,
                    multicast_sem_dst_size[i].first + program.get_program_config(index).sem_offset,
                    multicast_sem_dst_size[i].second,
                    multicast_sem_payload_sizeB,
                    multicast_sem_sub_cmds[i],
                    multicast_sem_data[i],
                    this->packed_write_max_unicast_sub_cmds,
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
                program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                    num_sub_cmds_in_cmd,
                    unicast_sem_dst_size[i].first + program.get_program_config(index).sem_offset,
                    unicast_sem_dst_size[i].second,
                    unicast_sem_payload_sizeB,
                    unicast_sem_sub_cmds[i],
                    unicast_sem_data[i],
                    this->packed_write_max_unicast_sub_cmds,
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
            cached_program_command_sequence.cb_configs_payloads.reserve(num_multicast_cb_sub_cmds);
            const uint32_t cb_config_size_words = aligned_cb_config_size_bytes / sizeof(uint32_t);
            for (const auto& [num_sub_cmds_in_cmd, mcast_cb_payload_sizeB] : mcast_cb_payload) {
                uint32_t write_offset_bytes = program_command_sequence.write_offset_bytes();
                program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                    num_sub_cmds_in_cmd,
                    program.get_program_config(index).cb_offset,
                    cb_config_size_bytes,
                    mcast_cb_payload_sizeB,
                    multicast_cb_config_sub_cmds,
                    multicast_cb_config_data,
                    this->packed_write_max_unicast_sub_cmds,
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
                     align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT)) /
                    sizeof(uint32_t);
                for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                    cached_program_command_sequence.cb_configs_payloads.push_back(
                        (uint32_t*)program_command_sequence.data() + curr_sub_cmd_data_offset_words);
                    curr_sub_cmd_data_offset_words += cb_config_size_words;
                }
            }
        }

        // All Previous Cmds Up to This Point Go Into the Kernel Config Buffer
        cached_program_command_sequence.program_config_buffer_data_size_bytes = program_command_sequence.write_offset_bytes();

        // Program Binaries
        for (const auto& kernel_bins_unicast_cmd : kernel_bins_unicast_cmds) {
            program_command_sequence.add_data(
                kernel_bins_unicast_cmd.data(),
                kernel_bins_unicast_cmd.size_bytes(),
                kernel_bins_unicast_cmd.size_bytes());
        }
        for (uint32_t i = 0; i < kernel_bins_dispatch_subcmds.size(); ++i) {
            program_command_sequence.add_dispatch_write_packed_large(
                DRAM_ALIGNMENT, kernel_bins_dispatch_subcmds[i].size(), kernel_bins_dispatch_subcmds[i]);
            program_command_sequence.add_prefetch_relay_paged_packed(
                kernel_bins_write_packed_large_data_aligned_sizeB[i],
                kernel_bins_prefetch_subcmds[i],
                kernel_bins_prefetch_subcmds[i].size());
        }

        // Wait Noc Write Barrier, wait for binaries/configs to be written to worker cores
        if (program.program_transfer_info.num_active_cores > 0) {
            program_command_sequence.add_dispatch_wait(true, DISPATCH_MESSAGE_ADDR, 0, 0, false, false);
        }

        // Go Signals
        cached_program_command_sequence.go_signals.reserve(
            multicast_go_signal_sub_cmds.size() + unicast_go_signal_sub_cmds.size());
        if (multicast_go_signal_sub_cmds.size() > 0) {
            uint32_t curr_sub_cmd_idx = 0;
            for (const auto& [num_sub_cmds_in_cmd, multicast_go_signal_payload_sizeB] : multicast_go_signals_payload) {
                uint32_t write_offset_bytes = program_command_sequence.write_offset_bytes();
                program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                    num_sub_cmds_in_cmd,
                    hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalMemAddrType::LAUNCH),
                    go_signal_sizeB,
                    multicast_go_signal_payload_sizeB,
                    multicast_go_signal_sub_cmds,
                    multicast_go_signal_data,
                    this->packed_write_max_unicast_sub_cmds,
                    curr_sub_cmd_idx);
                curr_sub_cmd_idx += num_sub_cmds_in_cmd;
                uint32_t curr_sub_cmd_data_offset_words =
                    (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                     align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT)) /
                    sizeof(uint32_t);
                for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                    cached_program_command_sequence.go_signals.push_back(
                        (launch_msg_t*)((uint32_t*)program_command_sequence.data() + curr_sub_cmd_data_offset_words));
                    curr_sub_cmd_data_offset_words += go_signal_size_words;
                }
            }
        }

        if (unicast_go_signal_sub_cmds.size() > 0) {
            uint32_t curr_sub_cmd_idx = 0;
            for (const auto& [num_sub_cmds_in_cmd, unicast_go_signal_payload_sizeB] : unicast_go_signals_payload) {
                uint32_t write_offset_bytes = program_command_sequence.write_offset_bytes();
                program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                    num_sub_cmds_in_cmd,
                    hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalMemAddrType::LAUNCH),
                    go_signal_sizeB,
                    unicast_go_signal_payload_sizeB,
                    unicast_go_signal_sub_cmds,
                    unicast_go_signal_data,
                    this->packed_write_max_unicast_sub_cmds,
                    curr_sub_cmd_idx);
                curr_sub_cmd_idx += num_sub_cmds_in_cmd;
                uint32_t curr_sub_cmd_data_offset_words =
                    (write_offset_bytes + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) +
                     align(num_sub_cmds_in_cmd * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT)) /
                    sizeof(uint32_t);
                for (uint32_t i = 0; i < num_sub_cmds_in_cmd; ++i) {
                    cached_program_command_sequence.go_signals.push_back(
                        (launch_msg_t*)((uint32_t*)program_command_sequence.data() + curr_sub_cmd_data_offset_words));
                    curr_sub_cmd_data_offset_words += go_signal_size_words;
                }
            }
        }
    } else {
        uint32_t i = 0;
        ZoneScopedN("program_loaded_on_device");
        for (const auto& cbs_on_core_range : cached_program_command_sequence.circular_buffers_on_core_ranges) {
            uint32_t* cb_config_payload = cached_program_command_sequence.cb_configs_payloads[i];
            for (const shared_ptr<CircularBuffer>& cb : cbs_on_core_range) {
                const uint32_t cb_address = cb->address() >> 4;
                const uint32_t cb_size = cb->size() >> 4;
                for (const auto& buffer_index : cb->buffer_indices()) {
                    // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                    // cb config payload
                    uint32_t base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * (uint32_t)buffer_index;
                    cb_config_payload[base_index] = cb_address;
                    cb_config_payload[base_index + 1] = cb_size;
                    cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                    cb_config_payload[base_index + 3] = cb->page_size(buffer_index) >> 4;
                }
            }
            i++;
        }
        uint32_t go_signal_count = 0;
        for (auto& go_signal : cached_program_command_sequence.go_signals) {
            go_signal->kernel_config.dispatch_core_x = this->dispatch_core.x;
            go_signal->kernel_config.dispatch_core_y = this->dispatch_core.y;
            for (uint32_t i = 0; i < kernel_config_addrs.size(); i++) {
                go_signal->kernel_config.kernel_config_base[i] = kernel_config_addrs[i].addr;
            }

            go_signal_count++;
            go_signal->kernel_config.host_assigned_id = program.get_runtime_id();
        }
    }
}

void EnqueueProgramCommand::process() {
    bool is_cached = true;
    if (not program.is_finalized()) {
        program.finalize();
        is_cached = false;
    }

    const std::pair<ConfigBufferSync, std::vector<ConfigBufferEntry>&> reservation =
        this->manager.get_config_buffer_mgr().reserve(program.program_config_sizes_);
    bool stall_first = reservation.first.need_sync;
    // Note: since present implementation always stalls, we always free up to "now"
    this->manager.get_config_buffer_mgr().free(reservation.first.sync_count);
    this->manager.get_config_buffer_mgr().alloc(
        this->expected_num_workers_completed + program.program_transfer_info.num_active_cores);

    std::vector<ConfigBufferEntry>& kernel_config_addrs = reservation.second;

    // Calculate all commands size and determine how many fetch q entries to use
    // Preamble, some waits and stalls
    // can be written directly to the issue queue
    if (not is_cached) {
        this->assemble_preamble_commands(kernel_config_addrs);
        this->assemble_stall_commands(true);
        // Runtime Args Command Sequence
        this->assemble_runtime_args_commands();

        // Record kernel groups in this program, only need to do it once.
        for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
            CoreType core_type = hal.get_core_type(index);
            RecordKernelGroups(program, core_type, program.get_kernel_groups(index));
        }
    } else {
        static constexpr uint32_t wait_count_offset = (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, wait.count));
        static constexpr uint32_t tensix_l1_write_offset_offset =
            (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.offset1));
        static constexpr uint32_t eth_l1_write_offset_offset =
            (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, set_write_offset.offset2));
        TT_ASSERT(
            this->cached_program_command_sequences.find(program.id) != this->cached_program_command_sequences.end(),
            "Program cache hit, but no stored command sequence");

        this->cached_program_command_sequences[program.id].stall_command_sequence.update_cmd_sequence(
            wait_count_offset, &this->expected_num_workers_completed, sizeof(uint32_t));

        this->cached_program_command_sequences[program.id].preamble_command_sequence.update_cmd_sequence(
            tensix_l1_write_offset_offset,
            &kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)],
            sizeof(uint32_t));
        if (hal.get_programmable_core_type_count() >= 2) {
            this->cached_program_command_sequences[program.id].preamble_command_sequence.update_cmd_sequence(
                eth_l1_write_offset_offset,
                &kernel_config_addrs[hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)],
                sizeof(uint32_t));
        }
    }
    RecordProgramRun(program);

    // Main Command Sequence
    this->assemble_device_commands(is_cached, kernel_config_addrs);

    const auto& cached_program_command_sequence = this->cached_program_command_sequences[program.id];

    uint32_t preamble_fetch_size_bytes = cached_program_command_sequence.preamble_command_sequence.size_bytes();

    uint32_t stall_fetch_size_bytes = cached_program_command_sequence.stall_command_sequence.size_bytes();

    uint32_t runtime_args_fetch_size_bytes = cached_program_command_sequence.runtime_args_fetch_size_bytes;

    uint32_t program_fetch_size_bytes = cached_program_command_sequence.program_command_sequence.size_bytes();

    uint32_t program_config_buffer_data_size_bytes = cached_program_command_sequence.program_config_buffer_data_size_bytes;

    uint32_t program_rem_fetch_size_bytes = program_fetch_size_bytes - program_config_buffer_data_size_bytes;

    uint8_t* program_command_sequence_data = (uint8_t*)cached_program_command_sequence.program_command_sequence.data();

    uint32_t total_fetch_size_bytes =
        stall_fetch_size_bytes + preamble_fetch_size_bytes + runtime_args_fetch_size_bytes + program_fetch_size_bytes;

    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    if (total_fetch_size_bytes <= dispatch_constants::get(dispatch_core_type).max_prefetch_command_size()) {
        this->manager.issue_queue_reserve(total_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);

        this->manager.cq_write(
            cached_program_command_sequence.preamble_command_sequence.data(), preamble_fetch_size_bytes, write_ptr);
        write_ptr += preamble_fetch_size_bytes;

        if (stall_first) {
            // Must stall before writing runtime args
            this->manager.cq_write(
                cached_program_command_sequence.stall_command_sequence.data(), stall_fetch_size_bytes, write_ptr);
            write_ptr += stall_fetch_size_bytes;
        }

        for (const auto& cmds : cached_program_command_sequence.runtime_args_command_sequences) {
            this->manager.cq_write(cmds.data(), cmds.size_bytes(), write_ptr);
            write_ptr += cmds.size_bytes();
        }

        if (not stall_first) {
            if (program_config_buffer_data_size_bytes > 0) {
                this->manager.cq_write(program_command_sequence_data, program_config_buffer_data_size_bytes, write_ptr);
                program_command_sequence_data += program_config_buffer_data_size_bytes;
                write_ptr += program_config_buffer_data_size_bytes;
            }

            // Didn't stall before kernel config data, stall before remaining commands
            this->manager.cq_write(
                cached_program_command_sequence.stall_command_sequence.data(), stall_fetch_size_bytes, write_ptr);
            write_ptr += stall_fetch_size_bytes;

            this->manager.cq_write(program_command_sequence_data, program_rem_fetch_size_bytes, write_ptr);
        } else {
            this->manager.cq_write(program_command_sequence_data, program_fetch_size_bytes, write_ptr);
        }

        this->manager.issue_queue_push_back(total_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for entire program
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(total_fetch_size_bytes, this->command_queue_id);

        // TODO: We are making a lot of fetch queue entries here, we can pack multiple commands into one fetch q entry
    } else {
        this->manager.issue_queue_reserve(preamble_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(
            cached_program_command_sequence.preamble_command_sequence.data(), preamble_fetch_size_bytes, write_ptr);
        this->manager.issue_queue_push_back(preamble_fetch_size_bytes, this->command_queue_id);
        // One fetch queue entry for just the wait and stall, very inefficient
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(preamble_fetch_size_bytes, this->command_queue_id);

        if (stall_first) {
            // Must stall before writing kernel config data
            this->manager.issue_queue_reserve(stall_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(
                cached_program_command_sequence.stall_command_sequence.data(), stall_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(stall_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for just the wait and stall, very inefficient
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(stall_fetch_size_bytes, this->command_queue_id);
        }

        // TODO: We can pack multiple RT args into one fetch q entry
        for (const auto& cmds : cached_program_command_sequence.runtime_args_command_sequences) {
            uint32_t fetch_size_bytes = cmds.size_bytes();
            this->manager.issue_queue_reserve(fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(cmds.data(), fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for each runtime args write location, e.g. BRISC/NCRISC/TRISC/ERISC
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
        }

        // Insert a stall between program data that goes on the ring buffer and the rest of the data
        // Otherwise write all data in 1 prefetch entry
        if (not stall_first) {
            if (program_config_buffer_data_size_bytes > 0) {
                this->manager.issue_queue_reserve(program_config_buffer_data_size_bytes, this->command_queue_id);
                write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
                this->manager.cq_write(program_command_sequence_data, program_config_buffer_data_size_bytes, write_ptr);
                this->manager.issue_queue_push_back(program_config_buffer_data_size_bytes, this->command_queue_id);
                this->manager.fetch_queue_reserve_back(this->command_queue_id);
                this->manager.fetch_queue_write(program_config_buffer_data_size_bytes, this->command_queue_id);
                program_command_sequence_data += program_config_buffer_data_size_bytes;
            }

            // Didn't stall before kernel config data, stall before remaining commands
            this->manager.issue_queue_reserve(stall_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(
                cached_program_command_sequence.stall_command_sequence.data(), stall_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(stall_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for just the wait and stall, very inefficient
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(stall_fetch_size_bytes, this->command_queue_id);

            this->manager.issue_queue_reserve(program_rem_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(program_command_sequence_data, program_rem_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(program_rem_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for rest of program commands
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(program_rem_fetch_size_bytes, this->command_queue_id);
        } else {
            this->manager.issue_queue_reserve(program_fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(program_command_sequence_data, program_fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(program_fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for rest of program commands
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(program_fetch_size_bytes, this->command_queue_id);
        }
    }

    // Front load generating and caching stall_commands without stall during program loading stage
    if (not is_cached) {
        this->assemble_stall_commands(false);
    }
}

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id,
    Device* device,
    NOC noc_index,
    SystemMemoryManager& manager,
    uint32_t event_id,
    uint32_t expected_num_workers_completed,
    bool clear_count,
    bool write_barrier) :
    command_queue_id(command_queue_id),
    device(device),
    noc_index(noc_index),
    manager(manager),
    event_id(event_id),
    expected_num_workers_completed(expected_num_workers_completed),
    clear_count(clear_count),
    write_barrier(write_barrier) {}

void EnqueueRecordEventCommand::process() {
    std::vector<uint32_t> event_payload(dispatch_constants::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = this->event_id;

    uint8_t num_hw_cqs =
        this->device->num_hw_cqs();  // Device initialize asserts that there can only be a maximum of 2 HW CQs
    uint32_t packed_event_payload_sizeB =
        align(sizeof(CQDispatchCmd) + num_hw_cqs * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT) +
        (align(dispatch_constants::EVENT_PADDED_SIZE, L1_ALIGNMENT) * num_hw_cqs);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, PCIE_ALIGNMENT);

    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED + unicast subcmds + event
                              // payload
        align(
            sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
            PCIE_ALIGNMENT);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID

    if (not device->is_mmio_capable()) {
        cmd_sequence_sizeB +=
            CQ_PREFETCH_CMD_BARE_MIN_SIZE *
            num_hw_cqs;  // CQ_DISPATCH_REMOTE_WRITE (number of writes = number of prefetch_h cores on this CQ)
    }

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait(
        this->write_barrier, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed, this->clear_count);

    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_hw_cqs);
    std::vector<std::pair<const void*, uint32_t>> event_payloads(num_hw_cqs);

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair dispatch_location;
        if (device->is_mmio_capable()) {
            dispatch_location = dispatch_core_manager::instance().dispatcher_core(this->device->id(), channel, cq_id);
        } else {
            dispatch_location = dispatch_core_manager::instance().dispatcher_d_core(this->device->id(), channel, cq_id);
        }

        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{
            .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, dispatch_physical_core)};
        event_payloads[cq_id] = {event_payload.data(), event_payload.size() * sizeof(uint32_t)};
    }

    uint32_t address = this->command_queue_id == 0 ? CQ0_COMPLETION_LAST_EVENT : CQ1_COMPLETION_LAST_EVENT;
    const uint32_t packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_hw_cqs,
        address,
        dispatch_constants::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads,
        packed_write_max_unicast_sub_cmds);

    if (not device->is_mmio_capable()) {
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair prefetch_location =
                dispatch_core_manager::instance().prefetcher_core(this->device->id(), channel, cq_id);
            CoreCoord prefetch_physical_core = get_physical_core_coordinate(prefetch_location, core_type);
            command_sequence.add_dispatch_write_remote(
                this->event_id,
                this->device->get_noc_unicast_encoding(this->noc_index, prefetch_physical_core),
                address);
        }
    }

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_host<true>(
        flush_prefetch, dispatch_constants::EVENT_PADDED_SIZE, true, event_payload.data());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id,
    Device* device,
    SystemMemoryManager& manager,
    const Event& sync_event,
    bool clear_count) :
    command_queue_id(command_queue_id),
    device(device),
    manager(manager),
    sync_event(sync_event),
    clear_count(clear_count) {
    this->dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
    // TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
    //     "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
    //     event, command_queue_id);
}

void EnqueueWaitForEventCommand::process() {
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                                                                  // or CQ_PREFETCH_CMD_WAIT_FOR_EVENT

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);
    uint32_t last_completed_event_address =
        sync_event.cq_id == 0 ? CQ0_COMPLETION_LAST_EVENT : CQ1_COMPLETION_LAST_EVENT;
    if (this->device->is_mmio_capable()) {
        command_sequence.add_dispatch_wait(false, last_completed_event_address, sync_event.event_id, this->clear_count);
    } else {
        command_sequence.add_prefetch_wait_for_event(sync_event.event_id, last_completed_event_address);
    }
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

EnqueueTraceCommand::EnqueueTraceCommand(
    uint32_t command_queue_id,
    Device* device,
    SystemMemoryManager& manager,
    Buffer& buffer,
    uint32_t& expected_num_workers_completed) :
    command_queue_id(command_queue_id),
    buffer(buffer),
    device(device),
    manager(manager),
    expected_num_workers_completed(expected_num_workers_completed),
    clear_count(true) {}

void EnqueueTraceCommand::process() {
    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_EXEC_BUF

    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait(
        false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed, this->clear_count);

    if (this->clear_count) {
        this->expected_num_workers_completed = 0;
    }

    uint32_t page_size = buffer.page_size();
    uint32_t page_size_log2 = __builtin_ctz(page_size);
    TT_ASSERT((page_size & (page_size - 1)) == 0, "Page size must be a power of 2");

    command_sequence.add_prefetch_exec_buf(buffer.address(), page_size_log2, buffer.num_pages());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    const bool stall_prefetcher = true;
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id, stall_prefetcher);
}

EnqueueTerminateCommand::EnqueueTerminateCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager) :
    command_queue_id(command_queue_id), device(device), manager(manager) {}

void EnqueueTerminateCommand::process() {
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand dispatch_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_command_sequence.add_dispatch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);

    cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

// HWCommandQueue section
HWCommandQueue::HWCommandQueue(Device* device, uint32_t id, NOC noc_index) :
    manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->noc_index = noc_index;
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();
    if (tt::Cluster::instance().is_galaxy_cluster()) {
        // Galaxy puts 4 devices per host channel until umd can provide one channel per device.
        this->size_B = this->size_B / 4;
    }

    CoreCoord enqueue_program_dispatch_core;
    if (device->is_mmio_capable()) {
        enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_core(device->id(), channel, id);
    } else {
        enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_d_core(device->id(), channel, id);
    }
    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    this->physical_enqueue_program_dispatch_core =
        device->physical_core_from_logical_core(enqueue_program_dispatch_core, core_type);

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::instance().completion_queue_writer_core(device->id(), channel, this->id);

    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    // Set the affinity of the completion queue reader.
    set_device_thread_affinity(this->completion_queue_thread, device->worker_thread_core);
    this->expected_num_workers_completed = 0;
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {
        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted "
            "commands: {}",
            this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->set_exit_condition();
        this->completion_queue_thread.join();
    }
}

void HWCommandQueue::increment_num_entries_in_completion_q() {
    // Increment num_entries_in_completion_q and inform reader thread
    // that there is work in the completion queue to process
    this->num_entries_in_completion_q++;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

void HWCommandQueue::set_exit_condition() {
    this->exit_condition = true;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking) {
    command.process();
    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion
// region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Read Buffer cannot be used with tracing");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());

    uint32_t padded_page_size = buffer.aligned_page_size();
    uint32_t pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        bool width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
        std::optional<BufferPageMapping> buffer_page_mapping = std::nullopt;
        if (width_split) {
            buffer_page_mapping = generate_buffer_page_mapping(buffer);
        }
        // Note that the src_page_index is the device page idx, not the host page idx
        // Since we read core by core we are reading the device pages sequentially
        const auto& cores = width_split ? buffer_page_mapping.value().all_cores_
                                        : corerange_to_cores(
                                              buffer.shard_spec().grid(),
                                              buffer.num_cores(),
                                              buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();
        bool linear_page_copy = true;
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            uint32_t num_pages_to_read;
            if (width_split) {
                num_pages_to_read =
                    buffer_page_mapping.value().core_shard_shape_[core_id][0] * buffer.shard_spec().shape_in_pages()[1];
            } else {
                num_pages_to_read = std::min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages_to_read;
            }
            uint32_t bank_base_address = buffer.address();
            if (buffer.is_dram()) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            if (num_pages_to_read > 0) {
                if (width_split) {
                    uint32_t host_page = buffer_page_mapping.value().core_host_page_indices_[core_id][0];
                    src_page_index = buffer_page_mapping.value().host_page_to_dev_page_mapping_[host_page];
                    unpadded_dst_offset = host_page * buffer.page_size();
                } else {
                    unpadded_dst_offset = src_page_index * buffer.page_size();
                }

                auto command = EnqueueReadShardedBufferCommand(
                    this->id,
                    this->device,
                    this->noc_index,
                    buffer,
                    dst,
                    this->manager,
                    this->expected_num_workers_completed,
                    cores[core_id],
                    bank_base_address,
                    src_page_index,
                    num_pages_to_read);

                this->issued_completion_q_reads.push(detail::CompletionReaderVariant(
                    std::in_place_type<detail::ReadBufferDescriptor>,
                    buffer.buffer_layout(),
                    buffer.page_size(),
                    padded_page_size,
                    dst,
                    unpadded_dst_offset,
                    num_pages_to_read,
                    src_page_index,
                    width_split ? (*buffer_page_mapping).dev_page_to_host_page_mapping_
                                : vector<std::optional<uint32_t>>()));

                src_page_index += num_pages_to_read;
                this->enqueue_command(command, false);
                this->increment_num_entries_in_completion_q();
            }
        }
        if (blocking) {
            this->finish();
        }
    } else {
        // this is a streaming command so we don't need to break down to multiple
        auto command = EnqueueReadInterleavedBufferCommand(
            this->id,
            this->device,
            this->noc_index,
            buffer,
            dst,
            this->manager,
            this->expected_num_workers_completed,
            src_page_index,
            pages_to_read);

        this->issued_completion_q_reads.push(detail::CompletionReaderVariant(
            std::in_place_type<detail::ReadBufferDescriptor>,
            buffer.buffer_layout(),
            buffer.page_size(),
            padded_page_size,
            dst,
            unpadded_dst_offset,
            pages_to_read,
            src_page_index));
        this->enqueue_command(command, blocking);
        this->increment_num_entries_in_completion_q();
    }
}

void HWCommandQueue::enqueue_write_buffer(
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<const Buffer>> buffer,
    HostDataType src,
    bool blocking) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    std::visit(
        [this, &buffer, &blocking](auto&& data) {
            using T = std::decay_t<decltype(data)>;
            std::visit(
                [this, &buffer, &blocking, &data](auto&& b) {
                    using type_buf = std::decay_t<decltype(b)>;
                    if constexpr (std::is_same_v<T, const void*>) {
                        if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                            this->enqueue_write_buffer(*b, data, blocking);
                        } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                            this->enqueue_write_buffer(b.get(), data, blocking);
                        }
                    } else {
                        if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                            this->enqueue_write_buffer(*b, data.get()->data(), blocking);
                        } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                            this->enqueue_write_buffer(b.get(), data.get()->data(), blocking);
                        }
                    }
                },
                buffer);
        },
        src);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::instance().get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Write Buffer cannot be used with tracing");

    uint32_t padded_page_size = buffer.aligned_page_size();

    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    uint32_t max_data_sizeB =
        max_prefetch_command_size - (CQ_PREFETCH_CMD_BARE_MIN_SIZE * 2);  // * 2 to account for issue

    uint32_t dst_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        const bool width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
        std::optional<BufferPageMapping> buffer_page_mapping = std::nullopt;
        if (width_split) {
            buffer_page_mapping = generate_buffer_page_mapping(buffer);
        }
        const auto& cores = width_split ? buffer_page_mapping.value().all_cores_
                                        : corerange_to_cores(
                                              buffer.shard_spec().grid(),
                                              buffer.num_cores(),
                                              buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
        TT_FATAL(
            max_data_sizeB >= padded_page_size,
            "Writing padded page size > {} is currently unsupported for sharded tensors.",
            max_data_sizeB);
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();

        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            // Skip writing the padded pages along the bottom
            // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
            // Alternative write each page row into separate commands, or have a strided linear write
            uint32_t num_pages;
            if (width_split) {
                num_pages =
                    buffer_page_mapping.value().core_shard_shape_[core_id][0] * buffer.shard_spec().shape_in_pages()[1];
                if (num_pages == 0) {
                    continue;
                }
                dst_page_index = buffer_page_mapping.value().host_page_to_dev_page_mapping_
                                     [buffer_page_mapping.value().core_host_page_indices_[core_id][0]];
            } else {
                num_pages = std::min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages;
            }
            uint32_t curr_page_idx_in_shard = 0;
            uint32_t bank_base_address = buffer.address();
            if (buffer.is_dram()) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            while (num_pages != 0) {
                // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
                uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
                bool issue_wait = dst_page_index == 0;  // only stall for the first write of the buffer
                if (issue_wait) {
                    // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                    data_offset_bytes *= 2;
                }
                uint32_t space_available_bytes = std::min(
                    command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
                int32_t num_pages_available =
                    (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(padded_page_size);

                uint32_t pages_to_write = std::min(num_pages, (uint32_t)num_pages_available);
                if (pages_to_write > 0) {
                    uint32_t address = bank_base_address + curr_page_idx_in_shard * padded_page_size;

                    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

                    auto command = EnqueueWriteShardedBufferCommand(
                        this->id,
                        this->device,
                        this->noc_index,
                        buffer,
                        src,
                        this->manager,
                        issue_wait,
                        this->expected_num_workers_completed,
                        address,
                        buffer_page_mapping,
                        cores[core_id],
                        padded_page_size,
                        dst_page_index,
                        pages_to_write);

                    this->enqueue_command(command, false);
                    curr_page_idx_in_shard += pages_to_write;
                    num_pages -= pages_to_write;
                    dst_page_index += pages_to_write;
                } else {
                    this->manager.wrap_issue_queue_wr_ptr(this->id);
                }
            }
        }
    } else {
        uint32_t total_pages_to_write = buffer.num_pages();
        bool write_partial_pages = padded_page_size > max_data_sizeB;
        uint32_t page_size_to_write = padded_page_size;
        uint32_t padded_buffer_size = buffer.num_pages() * padded_page_size;
        if (write_partial_pages) {
            TT_FATAL(buffer.num_pages() == 1, "TODO: add support for multi-paged buffer with page size > 64KB");
            uint32_t partial_size = dispatch_constants::BASE_PARTIAL_PAGE_SIZE;
            while (padded_buffer_size % partial_size != 0) {
                partial_size += PCIE_ALIGNMENT;
            }
            page_size_to_write = partial_size;
            total_pages_to_write = padded_buffer_size / page_size_to_write;
        }

        const uint32_t num_banks = this->device->num_banks(buffer.buffer_type());
        uint32_t num_pages_round_robined = buffer.num_pages() / num_banks;
        uint32_t num_banks_with_residual_pages = buffer.num_pages() % num_banks;
        uint32_t num_partial_pages_per_page = padded_page_size / page_size_to_write;
        uint32_t num_partials_round_robined = num_partial_pages_per_page * num_pages_round_robined;

        uint32_t max_num_pages_to_write = write_partial_pages
                                              ? (num_pages_round_robined > 0 ? (num_banks * num_partials_round_robined)
                                                                             : num_banks_with_residual_pages)
                                              : total_pages_to_write;

        uint32_t bank_base_address = buffer.address();

        uint32_t num_full_pages_written = 0;
        while (total_pages_to_write > 0) {
            uint32_t data_offsetB = CQ_PREFETCH_CMD_BARE_MIN_SIZE; // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
            bool issue_wait =
                (dst_page_index == 0 and
                 bank_base_address == buffer.address());  // only stall for the first write of the buffer
            if (issue_wait) {
                data_offsetB *= 2;  // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            }

            uint32_t space_availableB = std::min(
                command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
            int32_t num_pages_available =
                (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(page_size_to_write);

            if (num_pages_available <= 0) {
                this->manager.wrap_issue_queue_wr_ptr(this->id);
                continue;
            }

            uint32_t num_pages_to_write =
                std::min(std::min((uint32_t)num_pages_available, max_num_pages_to_write), total_pages_to_write);

            // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
            // To handle larger page offsets move bank base address up and update page offset to be relative to the new
            // bank address
            if (dst_page_index > 0xFFFF or (num_pages_to_write == max_num_pages_to_write and write_partial_pages)) {
                uint32_t num_banks_to_use = write_partial_pages ? max_num_pages_to_write : num_banks;
                uint32_t residual = dst_page_index % num_banks_to_use;
                uint32_t num_pages_written_per_bank = dst_page_index / num_banks_to_use;
                bank_base_address += num_pages_written_per_bank * page_size_to_write;
                dst_page_index = residual;
            }

            tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", this->id);

            auto command = EnqueueWriteInterleavedBufferCommand(
                this->id,
                this->device,
                this->noc_index,
                buffer,
                src,
                this->manager,
                issue_wait,
                this->expected_num_workers_completed,
                bank_base_address,
                page_size_to_write,
                dst_page_index,
                num_pages_to_write);
            this->enqueue_command(
                command, false);  // don't block until the entire src data is enqueued in the issue queue

            total_pages_to_write -= num_pages_to_write;
            dst_page_index += num_pages_to_write;
        }
    }

    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
    if (not program.is_finalized()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing should only be used when programs have been cached");
        if (program.kernels_buffer != nullptr) {
            this->enqueue_write_buffer(
                *program.kernels_buffer, program.program_transfer_info.binary_data.data(), false);
        }
    }
#ifdef DEBUG
    if (tt::llrt::OptionsG.get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (program.kernels_buffer != nullptr) {
            const auto& buffer = program.kernels_buffer;
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            this->enqueue_read_buffer(*program.kernels_buffer, read_data.data(), true);
            TT_FATAL(
                program.program_transfer_info.binary_data == read_data,
                "Binary for program to be executed is corrupted. Another program likely corrupted this binary");
        }
    }
#endif

    // Snapshot of expected workers from previous programs, used for dispatch_wait cmd generation.
    uint32_t expected_workers_completed = this->manager.get_bypass_mode() ? this->trace_ctx->num_completion_worker_cores
                                                                          : this->expected_num_workers_completed;
    if (this->manager.get_bypass_mode()) {
        this->trace_ctx->num_completion_worker_cores += program.program_transfer_info.num_active_cores;
    } else {
        this->expected_num_workers_completed += program.program_transfer_info.num_active_cores;
    }

    auto command = EnqueueProgramCommand(
        this->id,
        this->device,
        this->noc_index,
        program,
        this->physical_enqueue_program_dispatch_core,
        this->manager,
        expected_workers_completed);
    this->enqueue_command(command, blocking);

    if (program.has_multi_device_dependencies() and not this->device->is_mmio_capable() and
        tt::Cluster::instance().is_galaxy_cluster() and not this->tid.has_value()) {
        // Issue #19078 - Temporary workaround to avoid deadlocks on Galaxy, until Ethernet Routing Fabric supports VCs:
        // For programs that require syncs between devices (ex: CCLs), it must be ensured that all devices in a tunnel
        // receive the full set of program commands. Due to demux being a shared resource (it has a single input queue)
        // and cannot toggle its output queue id, until a txn is completed (prefetch_d corresponding to the current
        // packet is unblocked), it is possible that all devices do not get the program commands and enter a deadlock
        // (dispatch_d gets blocked waiting for the multi-device program to complete, causing prefetch_d to
        // backpressure, as its picked up other commands -> demux has CCL program commands for other devices in its
        // queue, but is blocked sending a downstream command to the backpressured prefetch_d). To resolve this,
        // prefetch_h for all devices involved in the multi-device program will stall sending commands, until dispatch_d
        // has notified prefetch_h that workers have completed execution (all chips got the program commands, and there
        // is no further scope of a deadlock).
        // This pipeline flush does not need to be issued when using trace, since prefetch_h will stall sending pages to
        // prefetch_d until it has been notified of trace completion (due to cmddat_q reuse). Additionally, events can
        // currently not be traced, thus this is skipped during trace capture.
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
        this->enqueue_wait_for_event(event);
    }

#ifdef DEBUG
    if (tt::llrt::OptionsG.get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (program.kernels_buffer != nullptr) {
            const auto& buffer = program.kernels_buffer;
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            this->enqueue_read_buffer(*program.kernels_buffer, read_data.data(), true);
            TT_FATAL(
                program.program_transfer_info.binary_data == read_data,
                "Binary for program that executed is corrupted. This program likely corrupted its own binary.");
        }
    }
#endif

    log_trace(
        tt::LogMetal,
        "Created EnqueueProgramCommand (active_cores: {} bypass_mode: {} expected_workers_completed: {})",
        program.program_transfer_info.num_active_cores,
        this->manager.get_bypass_mode(),
        expected_workers_completed);
}

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Record Event cannot be used with tracing");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true;  // what does this mean???

    auto command = EnqueueRecordEventCommand(
        this->id,
        this->device,
        this->noc_index,
        this->manager,
        event->event_id,
        this->expected_num_workers_completed,
        clear_count,
        true);
    this->enqueue_command(command, false);

    if (clear_count) {
        this->expected_num_workers_completed = 0;
    }
    this->issued_completion_q_reads.push(
        detail::CompletionReaderVariant(std::in_place_type<detail::ReadEventDescriptor>, event->event_id));
    this->increment_num_entries_in_completion_q();
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> sync_event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    auto command = EnqueueWaitForEventCommand(this->id, this->device, this->manager, *sync_event, clear_count);
    this->enqueue_command(command, false);

    if (clear_count) {
        this->manager.reset_event_id(this->id);
    }
}

void HWCommandQueue::enqueue_trace(const uint32_t trace_id, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_trace");

    auto trace_inst = this->device->get_trace(trace_id);
    auto command = EnqueueTraceCommand(
        this->id, this->device, this->manager, *trace_inst->buffer, this->expected_num_workers_completed);

    this->enqueue_command(command, false);

    // Increment the exepected worker cores counter due to trace programs completions
    this->expected_num_workers_completed += trace_inst->desc->num_completion_worker_cores;

    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::copy_into_user_space(
    const detail::ReadBufferDescriptor& read_buffer_descriptor, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, dev_page_to_host_page_mapping, dst, dst_offset, num_pages_read, cur_dev_page_id] =
        read_buffer_descriptor;

    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;
    uint32_t dev_page_id = cur_dev_page_id;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    std::optional<uint32_t> host_page_id = std::nullopt;
    uint32_t offset_in_completion_q_data = sizeof(CQDispatchCmd);

    uint32_t pad_size_bytes = padded_page_size - page_size;

    while (remaining_bytes_to_read != 0) {
        this->manager.completion_queue_wait_front(this->id, this->exit_condition);

        if (this->exit_condition) {
            break;
        }

        uint32_t completion_queue_write_ptr_and_toggle =
            get_cq_completion_wr_ptr<true>(this->device->id(), this->id, this->manager.get_cq_size());
        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
        uint32_t completion_q_read_toggle = this->manager.get_completion_queue_read_toggle(this->id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue =
                this->manager.get_completion_queue_limit(this->id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(remaining_bytes_to_read, bytes_avail_in_completion_queue);
        uint32_t num_pages_xfered =
            (bytes_xfered + dispatch_constants::TRANSFER_PAGE_SIZE - 1) / dispatch_constants::TRANSFER_PAGE_SIZE;

        remaining_bytes_to_read -= bytes_xfered;

        if (dev_page_to_host_page_mapping.empty()) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            if (page_size == padded_page_size) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data;
                tt::Cluster::instance().read_sysmem(
                    contiguous_dst,
                    data_bytes_xfered,
                    completion_q_read_ptr + offset_in_completion_q_data,
                    mmio_device_id,
                    channel);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset_bytes = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint32_t dst_offset_bytes = 0;

                while (src_offset_bytes < bytes_xfered) {
                    uint32_t src_offset_increment = padded_page_size;
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = num_bytes_to_copy;
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes;
                                // Only pad data left in queue
                            } else {
                                offset_in_completion_q_data = pad_size_bytes - rem_bytes_in_cq;
                            }
                        }
                    } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    tt::Cluster::instance().read_sysmem(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        num_bytes_to_copy,
                        completion_q_read_ptr + src_offset_bytes,
                        mmio_device_id,
                        channel);

                    src_offset_bytes += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else {
            uint32_t src_offset_bytes = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint32_t dst_offset_bytes = contig_dst_offset;
            uint32_t num_bytes_to_copy = 0;

            while (src_offset_bytes < bytes_xfered) {
                uint32_t src_offset_increment = padded_page_size;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = num_bytes_to_copy;
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        dev_page_id++;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes;
                            offset_in_completion_q_data = 0;
                            // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq);
                        }
                    }
                    if (!host_page_id.has_value()) {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    host_page_id = dev_page_to_host_page_mapping[dev_page_id];
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        dev_page_id++;
                    }
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = host_page_id.value() * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    host_page_id = dev_page_to_host_page_mapping[dev_page_id];
                    dev_page_id++;
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = host_page_id.value() * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                }

                tt::Cluster::instance().read_sysmem(
                    (char*)(uint64_t(dst) + dst_offset_bytes),
                    num_bytes_to_copy,
                    completion_q_read_ptr + src_offset_bytes,
                    mmio_device_id,
                    channel);

                src_offset_bytes += src_offset_increment;
            }
            dst_offset_bytes += num_bytes_to_copy;
            contig_dst_offset = dst_offset_bytes;
        }
        this->manager.completion_queue_pop_front(num_pages_xfered, this->id);
    }
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        {
            std::unique_lock<std::mutex> lock(this->reader_thread_cv_mutex);
            this->reader_thread_cv.wait(lock, [this] {
                return this->num_entries_in_completion_q > this->num_completed_completion_q_reads or
                       this->exit_condition;
            });
        }
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            ZoneScopedN("CompletionQueueReader");
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {
                ZoneScopedN("CompletionQueuePopulated");
                auto read_descriptor = *(this->issued_completion_q_reads.pop());
                {
                    ZoneScopedN("CompletionQueueWait");
                    this->manager.completion_queue_wait_front(
                        this->id, this->exit_condition);  // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN
                }
                if (this->exit_condition) {  // Early exit
                    return;
                }

                std::visit(
                    [&](auto&& read_descriptor) {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            ZoneScopedN("CompletionQueueReadData");
                            this->copy_into_user_space(read_descriptor, mmio_device_id, channel);
                        } else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            ZoneScopedN("CompletionQueueReadEvent");
                            uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
                            thread_local static std::vector<uint32_t> dispatch_cmd_and_event(
                                (sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(),
                                sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
                                read_ptr,
                                mmio_device_id,
                                channel);
                            uint32_t event_completed =
                                dispatch_cmd_and_event.at(sizeof(CQDispatchCmd) / sizeof(uint32_t));

                            TT_ASSERT(
                                event_completed == read_descriptor.event_id,
                                "Event Order Issue: expected to read back completion signal for event {} but got {}!",
                                read_descriptor.event_id,
                                event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, read_descriptor.get_global_event_id());
                            log_trace(
                                LogAlways,
                                "Completion queue popped event {} (global: {})",
                                event_completed,
                                read_descriptor.get_global_event_id());
                        }
                    },
                    read_descriptor);
            }
            this->num_completed_completion_q_reads += num_events_to_read;
            {
                std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex);
                this->reads_processed_cv.notify_one();
            }
        } else if (this->exit_condition) {
            return;
        }
    }
}

void HWCommandQueue::finish() {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event);
    if (tt::llrt::OptionsG.get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->dprint_server_hang = true;
                this->set_exit_condition();
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->illegal_noc_txn_hang = true;
                this->set_exit_condition();
                return;
            }
        }
    } else {
        std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex);
        this->reads_processed_cv.wait(
            lock, [this] { return this->num_entries_in_completion_q == this->num_completed_completion_q_reads; });
    }
}

volatile bool HWCommandQueue::is_dprint_server_hung() { return dprint_server_hang; }

volatile bool HWCommandQueue::is_noc_hung() { return illegal_noc_txn_hang; }

void HWCommandQueue::record_begin(const uint32_t tid, std::shared_ptr<detail::TraceDescriptor> ctx) {
    // Issue event as a barrier and a counter reset
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event, true);
    // Record commands using bypass mode
    this->tid = tid;
    this->trace_ctx = ctx;
    this->manager.set_bypass_mode(true, true);  // start
}

void HWCommandQueue::record_end() {
    this->tid = std::nullopt;
    this->trace_ctx = nullptr;
    this->manager.set_bypass_mode(false, false);  // stop
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager.get_bypass_mode(), "Terminate cannot be used with tracing");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id);
    auto command = EnqueueTerminateCommand(this->id, this->device, this->manager);
    this->enqueue_command(command, false);
}

void EnqueueAddBufferToProgramImpl(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    Program* program) {
    std::visit(
        [program](auto&& b) {
            using buffer_type = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
                if (program != nullptr) {
                    program->add_buffer(b);
                }
            }
        },
        buffer);
}

void EnqueueAddBufferToProgram(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    Program* program,
    bool blocking) {
    EnqueueAddBufferToProgramImpl(buffer, program);
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit(
            [&resolved_runtime_args](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, Buffer*>) {
                    resolved_runtime_args.push_back(a->address());
                } else {
                    resolved_runtime_args.push_back(a);
                }
            },
            arg);
    }
    runtime_args_md.kernel->set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueSetRuntimeArgs(
    CommandQueue& cq,
    const std::shared_ptr<Kernel> kernel,
    const CoreCoord& core_coord,
    std::shared_ptr<RuntimeArgs> runtime_args_ptr,
    bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata{
        .core_coord = core_coord,
        .runtime_args_ptr = runtime_args_ptr,
        .kernel = kernel,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer->address();
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::GET_BUF_ADDR, .blocking = blocking, .shadow_buffer = buffer, .dst = dst_buf_addr});
}

void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md) {
    Buffer* buffer = alloc_md.buffer;
    uint32_t allocated_addr;
    if (is_sharded(buffer->buffer_layout())) {
        allocated_addr = allocator::allocate_buffer(
            *(buffer->device()->allocator_),
            buffer->shard_spec().size() * buffer->num_cores() * buffer->page_size(),
            buffer->page_size(),
            buffer->buffer_type(),
            alloc_md.bottom_up,
            buffer->num_cores());
    } else {
        allocated_addr = allocator::allocate_buffer(
            *(buffer->device()->allocator_),
            buffer->size(),
            buffer->page_size(),
            buffer->buffer_type(),
            alloc_md.bottom_up,
            std::nullopt);
    }
    buffer->set_address(static_cast<uint64_t>(allocated_addr));
}

void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking) {
    auto alloc_md = AllocBufferMetadata{
        .buffer = buffer,
        .allocator = *(buffer->device()->allocator_),
        .bottom_up = bottom_up,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md) {
    allocator::deallocate_buffer(alloc_md.allocator, alloc_md.device_address, alloc_md.buffer_type);
}

void EnqueueDeallocateBuffer(
    CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking) {
    // Need to explictly pass in relevant buffer attributes here, since the Buffer* ptr can be deallocated a this point
    auto alloc_md = AllocBufferMetadata{
        .allocator = allocator,
        .buffer_type = buffer_type,
        .device_address = device_address,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::DEALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueReadBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    vector<uint32_t>& dst,
    bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    Buffer& b = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                    ? *(std::get<std::shared_ptr<Buffer>>(buffer))
                    : std::get<std::reference_wrapper<Buffer>>(buffer).get();
    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    std::visit(
        [&dst](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>>) {
                dst.resize(b.get().page_size() * b.get().num_pages() / sizeof(uint32_t));
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
                dst.resize(b->page_size() * b->num_pages() / sizeof(uint32_t));
            }
        },
        buffer);

    // TODO(agrebenisan): Move to deprecated
    EnqueueReadBuffer(cq, buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    vector<uint32_t>& src,
    bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, buffer, src.data(), blocking);
}

void EnqueueReadBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER, .blocking = blocking, .buffer = buffer, .dst = dst});
}

void EnqueueReadBufferImpl(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    bool blocking) {
    std::visit(
        [&cq, dst, blocking](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking);
            }
        },
        buffer);
}

void EnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER, .blocking = blocking, .buffer = buffer, .src = src});
}

void EnqueueWriteBufferImpl(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking) {
    std::visit(
        [&cq, src, blocking](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.hw_command_queue().enqueue_write_buffer(b, src, blocking);
            }
        },
        buffer);
}

void EnqueueProgram(
    CommandQueue& cq, Program* program, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::ENQUEUE_PROGRAM, .blocking = blocking, .program = program});
}

void EnqueueProgramImpl(
    CommandQueue& cq, Program* program, bool blocking) {
    ZoneScoped;
    if (program != nullptr) {
        Device* device = cq.device();
        detail::CompileProgram(device, *program);
        program->allocate_circular_buffers();
        detail::ValidateCircularBufferRegion(*program, device);
        cq.hw_command_queue().enqueue_program(*program, blocking);
        // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem
        // leaks on device.
        program->release_buffers();
    }
}

void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueRecordEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    cq.hw_command_queue().enqueue_record_event(event);
}

void EnqueueWaitForEvent(CommandQueue& cq, std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueWaitForEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    event->wait_until_ready();  // Block until event populated. Worker thread.
    log_trace(
        tt::LogMetal,
        "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(),
        event->cq_id,
        event->event_id,
        cq.device()->id(),
        cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}

void EventSynchronize(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    log_trace(
        tt::LogMetal,
        "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})",
        event->device->id(),
        event->cq_id,
        event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::OptionsG.get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_FATAL(
                false,
                "Command Queue could not complete EventSynchronize. See {} for details.",
                tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool EventQuery(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        event->device->id(),
        event->cq_id,
        event->event_id);
    return event_completed;
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{.type = EnqueueCommandType::FINISH, .blocking = true});
    TT_ASSERT(
        !(cq.device()->hw_command_queue(cq.id()).is_dprint_server_hung()),
        "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(
        !(cq.device()->hw_command_queue(cq.id()).is_noc_hung()),
        "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
        tt::watcher_get_log_file_name());
}

void FinishImpl(CommandQueue& cq) { cq.hw_command_queue().finish(); }

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(
        cq.device()->get_trace(trace_id) != nullptr,
        "Trace instance " + std::to_string(trace_id) + " must exist on device");
    cq.run_command(
        CommandInterface{.type = EnqueueCommandType::ENQUEUE_TRACE, .blocking = blocking, .trace_id = trace_id});
}

void EnqueueTraceImpl(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    cq.hw_command_queue().enqueue_trace(trace_id, blocking);
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device), cq_id(id), mode(mode), worker_state(CommandQueueState::IDLE) {
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace& trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (not this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() { return this->device()->hw_command_queue(this->cq_id); }

void CommandQueue::dump() {
    int cid = 0;
    log_info(LogMetalTrace, "Dumping {}, mode={}", this->name(), this->get_mode());
    for (const auto& cmd : this->worker_queue) {
        log_info(LogMetalTrace, "[{}]: {}", cid, cmd.type);
        cid++;
    }
}

std::string CommandQueue::name() {
    if (this->mode == CommandQueueMode::TRACE) {
        return "TraceQueue";
    }
    return "CQ" + std::to_string(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "{} WFI start", this->name());
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "{} WFI complete", this->name());
}

void CommandQueue::set_mode(const CommandQueueMode& mode) {
    TT_ASSERT(
        not this->trace_mode(),
        "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "{} started worker thread", this->name());
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "{} stopped worker thread", this->name());
}

void CommandQueue::run_worker() {
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(const CommandInterface& command) {
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
    if (this->async_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id) {
            // In async mode when parent pushes cmd, feed worker through queue.
            this->worker_queue.push(command);
            bool blocking = command.blocking.has_value() and *command.blocking;
            if (blocking) {
                TT_ASSERT(not this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        } else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(
                std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id,
                "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else if (this->trace_mode()) {
        // In trace mode push to the trace queue
        this->worker_queue.push(command);
    } else if (this->passthrough_mode()) {
        this->run_command_impl(command);
    } else {
        TT_THROW("Unsupported CommandQueue mode!");
    }
}

void CommandQueue::run_command_impl(const CommandInterface& command) {
    log_trace(LogDispatch, "{} running {}", this->name(), command.type);
    switch (command.type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueReadBufferImpl(*this, command.buffer.value(), command.dst.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
            TT_ASSERT(command.src.has_value(), "Must provide a src!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueWriteBufferImpl(*this, command.buffer.value(), command.src.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueAllocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::DEALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueDeallocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::GET_BUF_ADDR:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
            TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
            EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
            break;
        case EnqueueCommandType::SET_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program != nullptr, "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), command.program);
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program != nullptr, "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, command.program, command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_TRACE:
            EnqueueTraceImpl(*this, command.trace_id.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueRecordEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueWaitForEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::FINISH: FinishImpl(*this); break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default: TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "{} running {} complete", this->name(), command.type);
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type) {
    switch (type) {
        case CommandQueue::CommandQueueMode::PASSTHROUGH: os << "PASSTHROUGH"; break;
        case CommandQueue::CommandQueueMode::ASYNC: os << "ASYNC"; break;
        case CommandQueue::CommandQueueMode::TRACE: os << "TRACE"; break;
        default: TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}
