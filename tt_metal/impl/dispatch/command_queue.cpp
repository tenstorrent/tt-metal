// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>
#include <variant>

#include "allocator/allocator.hpp"
#include "assert.hpp"
#include "debug_tools.hpp"
#include "dev_msgs.h"
#include "tt_metal/common/logger.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;

std::mutex finish_mutex;
std::condition_variable finish_cv;

namespace tt::tt_metal {

uint32_t get_noc_unicast_encoding(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

void align_commands_vector(vector<uint32_t> &commands, uint32_t byte_alignment) {
    commands.resize(align(commands.size(), byte_alignment / sizeof(uint32_t)), 0);
}

// EnqueueReadBufferCommandSection

EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    void* dst,
    SystemMemoryManager& manager,
    uint32_t expected_num_workers_completed,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id),
    dst(dst),
    manager(manager),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    src_page_index(src_page_index),
    pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {

    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to read an invalid buffer");

    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void EnqueueReadInterleavedBufferCommand::add_prefetch_relay(DeviceCommand &command) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    command.add_prefetch_relay_paged(
        this->buffer.buffer_type() == BufferType::DRAM, this->src_page_index, this->buffer.address(), padded_page_size, this->pages_to_read);
}

void EnqueueReadShardedBufferCommand::add_prefetch_relay(DeviceCommand &command) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    CoreCoord logical_core = this->buffer_page_mapping.all_cores_[this->buffer_page_mapping.dev_page_to_core_mapping_[this->src_page_index]];
    CoreCoord core = this->buffer.device()->worker_core_from_logical_core(logical_core);
    command.add_prefetch_relay_linear(
        get_noc_unicast_encoding(core),
        padded_page_size * this->pages_to_read,
        this->buffer.address() + this->buffer_page_mapping.host_page_to_local_shard_page_mapping_[this->buffer_page_mapping.dev_page_to_host_page_mapping_[this->src_page_index]] * padded_page_size
    );
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_commands() {
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_STALL
        CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    DeviceCommand command(cmd_sequence_sizeB);

    command.add_dispatch_wait_with_prefetch_stall(true, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);

    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    bool flush_prefetch = false;
    command.add_dispatch_write_host(flush_prefetch, this->pages_to_read * padded_page_size);

    this->add_prefetch_relay(command);

    return command;
}

void EnqueueReadBufferCommand::process() {
    DeviceCommand command_sequence = this->assemble_device_commands();

    uint32_t fetch_size_bytes = command_sequence.size_bytes();

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    // Wrap issue queue
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
    if (write_ptr + align(fetch_size_bytes, 32) > command_issue_limit) {
        this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    }

    this->manager.cq_write(command_sequence.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
}

// EnqueueWriteBufferCommand section

EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
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
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void EnqueueWriteInterleavedBufferCommand::add_dispatch_write(DeviceCommand &command_sequence, void *data_to_write) {
    uint8_t is_dram = uint8_t(this->buffer.buffer_type() == BufferType::DRAM);
    TT_ASSERT(this->dst_page_index <= 0xFFFF, "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    uint16_t start_page = uint16_t(this->dst_page_index & 0xFFFF);
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch, is_dram, start_page, this->bank_base_address, this->padded_page_size, this->pages_to_write, data_to_write);
}

void EnqueueWriteShardedBufferCommand::add_dispatch_write(DeviceCommand &command_sequence, void *data_to_write) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    CoreCoord logical_core = this->buffer_page_mapping.all_cores_[this->buffer_page_mapping.dev_page_to_core_mapping_[this->dst_page_index]];
    CoreCoord core = this->buffer.device()->worker_core_from_logical_core(logical_core);

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_linear(
        flush_prefetch, 0, get_noc_unicast_encoding(core), this->bank_base_address, data_size_bytes, data_to_write);
}

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_commands() {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;

    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_RELAY_INLINE + (CQ_DISPATCH_CMD_WRITE_PAGED or CQ_DISPATCH_CMD_WRITE_LINEAR)
        data_size_bytes;
    if (this->issue_wait) {
        cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE; // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }
    DeviceCommand command_sequence(cmd_sequence_sizeB);

    if (this->issue_wait) {
        command_sequence.add_dispatch_wait(false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);
    }

    std::vector<uint32_t> data_to_write(data_size_bytes / sizeof(uint32_t), 0);

    uint32_t full_page_size = align(this->buffer.page_size(), 32); // this->padded_page_size could be a partial page if buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = this->padded_page_size < full_page_size;

    uint32_t buffer_addr_offset = this->bank_base_address - this->buffer.address();
    uint32_t num_banks = is_sharded(this->buffer.buffer_layout()) ? 0 : this->device->num_banks(this->buffer.buffer_type());

    // TODO: Consolidate
    if (write_partial_pages) {
        uint32_t padding = full_page_size - this->buffer.page_size();
        uint32_t unpadded_src_offset = buffer_addr_offset;
        uint32_t src_address_offset = unpadded_src_offset;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes; sysmem_address_offset += this->padded_page_size) {
            uint32_t sysmem_address_offset_words = sysmem_address_offset / sizeof(uint32_t);
            uint32_t page_size_to_copy = this->padded_page_size;
            if (src_address_offset + this->padded_page_size > buffer.page_size()) {
                // last partial page being copied from unpadded src buffer
                page_size_to_copy -= padding;
            }
            memcpy(data_to_write.data() + sysmem_address_offset_words, (char*)this->src + src_address_offset, page_size_to_copy);
            src_address_offset += page_size_to_copy;
        }
    } else {
        uint32_t unpadded_src_offset = ( ((buffer_addr_offset/this->padded_page_size) * num_banks) + this->dst_page_index) * this->buffer.page_size();
        if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
            // If page size is not 32B-aligned, we cannot do a contiguous write
            uint32_t src_address_offset = unpadded_src_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes; sysmem_address_offset += this->padded_page_size) {
                uint32_t sysmem_address_offset_words = sysmem_address_offset / sizeof(uint32_t);
                memcpy(data_to_write.data() + sysmem_address_offset_words, (char*)this->src + src_address_offset, this->buffer.page_size());
                src_address_offset += this->buffer.page_size();
            }
        } else {
            memcpy(data_to_write.data(), (char*)this->src + unpadded_src_offset, data_size_bytes);
        }
    }

    this->add_dispatch_write(command_sequence, data_to_write.data());

    return command_sequence;
}

void EnqueueWriteBufferCommand::process() {
    DeviceCommand command_sequence = this->assemble_device_commands();

    uint32_t fetch_size_bytes = command_sequence.size_bytes();

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);

    // We already checked for issue queue wrap when setting up the cmd, no need to check again here

    this->manager.cq_write(command_sequence.data(), fetch_size_bytes, write_ptr);


    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
}

// EnqueueProgramCommand Section

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    Program& program,
    SystemMemoryManager& manager,
    uint32_t expected_num_workers_completed) :
    command_queue_id(command_queue_id),
    manager(manager),
    expected_num_workers_completed(expected_num_workers_completed),
    program(program) {
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

const DeviceCommand EnqueueProgramCommand::assemble_preamble_commands() {
    if (not program.loaded_onto_device) {
        // Wait command so previous program finishes
        // Wait command with barrier for binaries to commit to DRAM
        // Prefetch stall to prevent prefetcher picking up incomplete binaries from DRAM
        uint32_t cmd_sequence_sizeB =
            CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_STALL

        DeviceCommand command_sequence(cmd_sequence_sizeB);

        // Wait for Noc Write Barrier
        // wait for binaries to commit to dram, also wait for previous program to be done
        // Wait Noc Write Barrier, wait for binaries to be written to worker cores
        // Stall to allow binaries to commit to DRAM first
        // TODO: this can be removed for all but the first program run
        command_sequence.add_dispatch_wait_with_prefetch_stall(
            true, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);

        program.loaded_onto_device = true;
        return command_sequence;
    } else {
        // Wait command so previous program finishes
        uint32_t cmd_sequence_sizeB =
            CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

        DeviceCommand command_sequence(cmd_sequence_sizeB);
        command_sequence.add_dispatch_wait(false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed);
        return command_sequence;
    }
}

// Generate command sequence for unique (unicast) and common (multicast) runtime args
const vector<DeviceCommand> EnqueueProgramCommand::assemble_runtime_args_commands() {
    program.update_runtime_args_transfer_info(this->device);

    vector<DeviceCommand> runtime_args_command_sequences;

    CoreType dispatch_core_type = dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();

    auto get_runtime_payload_sizeB = [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast){
        uint32_t sub_cmd_sizeB = is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
        uint32_t dispatch_cmd_sizeB = align(sizeof(CQDispatchCmd) + num_packed_cmds * sub_cmd_sizeB, L1_ALIGNMENT);
        uint32_t aligned_runtime_data_sizeB = num_packed_cmds * align(runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT);
        return dispatch_cmd_sizeB + aligned_runtime_data_sizeB;
    };

    // Unicast unique runtime args per core.
    for (const auto& [dst, transfer_info] : program.program_transfer_info.unicast_runtime_args) {
        uint32_t num_packed_cmds = transfer_info.size();
        uint32_t packed_cmd_idx = 0;
        while (packed_cmd_idx < num_packed_cmds) {
            uint32_t runtime_args_len = transfer_info[packed_cmd_idx].data.size();

            uint32_t num_packed_cmds_in_seq = 0;

            for (int i = packed_cmd_idx; i < num_packed_cmds; i++) {
                TT_ASSERT(transfer_info[i].dst_noc_info.size() == 1, "Not supporting CoreRangeSet for runtime args");
                uint32_t max_runtime_args_len = std::max(runtime_args_len, (uint32_t)transfer_info[i].data.size());
                uint32_t next_cmd_sequence_sizeB = align(
                    sizeof(CQPrefetchCmd) + get_runtime_payload_sizeB(num_packed_cmds_in_seq + 1, max_runtime_args_len, true), PCIE_ALIGNMENT);


                if (next_cmd_sequence_sizeB > max_prefetch_command_size) {
                    // If we hit this, then we send multiple write packed commands for each RT dst.
                    // TODO: Revisit RT arg command sequence construction to be more performant
                    break;
                }
                runtime_args_len = max_runtime_args_len;
                num_packed_cmds_in_seq++;
            }

            uint32_t rt_payload_sizeB = get_runtime_payload_sizeB(num_packed_cmds_in_seq, runtime_args_len, true);
            uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + rt_payload_sizeB, PCIE_ALIGNMENT);
            DeviceCommand command_sequence(cmd_sequence_sizeB);

            std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds;
            std::vector<const void *> rt_data_and_sizes;

            for (int i = packed_cmd_idx; i < (packed_cmd_idx + num_packed_cmds_in_seq); i++) {
                unicast_sub_cmds.emplace_back(
                    CQDispatchWritePackedUnicastSubCmd{.noc_xy_addr = transfer_info[i].dst_noc_info[0].first});
                rt_data_and_sizes.emplace_back(transfer_info[i].data.data());
            }

            command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_packed_cmds_in_seq,
                dst,
                runtime_args_len * sizeof(uint32_t),
                rt_payload_sizeB,
                unicast_sub_cmds,
                rt_data_and_sizes);
            runtime_args_command_sequences.emplace_back(command_sequence);

            packed_cmd_idx += num_packed_cmds_in_seq;
        }
    }

    // Muticast common runtime args (shared by all cores)
    for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.multicast_runtime_args) {
        uint32_t num_packed_cmds = 0;
        uint32_t runtime_args_len = transfer_info_vec[0].data.size();

        for (const auto& transfer_info: transfer_info_vec) {
            for (const auto& dst_noc_info: transfer_info.dst_noc_info) {
                TT_ASSERT(transfer_info.data.size() == runtime_args_len, "Not all data vectors in write packed common rtargs cmd equal in len");
                num_packed_cmds += 1;
            }
        }

        uint32_t mcast_payload_sizeB = get_runtime_payload_sizeB(num_packed_cmds, runtime_args_len, false);
        uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + mcast_payload_sizeB, PCIE_ALIGNMENT);

        DeviceCommand command_sequence(cmd_sequence_sizeB);

        std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_sub_cmds;
        std::vector<const void *> rtarg_data;

        for (const auto& transfer_info: transfer_info_vec) {
            for (const auto& dst_noc_info: transfer_info.dst_noc_info) {
                multicast_sub_cmds.emplace_back(
                    CQDispatchWritePackedMulticastSubCmd{.noc_xy_addr = dst_noc_info.first, .num_mcast_dests = dst_noc_info.second}
                );
                rtarg_data.emplace_back(transfer_info.data.data());
            }
        }

        command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
            num_packed_cmds,
            dst,
            runtime_args_len * sizeof(uint32_t),
            mcast_payload_sizeB,
            multicast_sub_cmds,
            rtarg_data
        );
        runtime_args_command_sequences.emplace_back(command_sequence);
    }

    return runtime_args_command_sequences;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_commands() {
    if (program.cached_device_commands.size() == 0) {
        // Calculate size of command and fill program indices of data to update
        // TODO: Would be nice if we could pull this out of program
        uint32_t cmd_sequence_sizeB = 0;

        for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.multicast_semaphores) {
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            for (const auto& transfer_info : transfer_info_vec) {
                for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                    TT_ASSERT(
                        transfer_info.data.size() == write_packed_len,
                        "Not all data vectors in write packed semaphore cmd equal in len");
                    num_packed_cmds += 1;
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT);
            uint32_t mcast_payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + mcast_payload_sizeB, PCIE_ALIGNMENT);
        }

        for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.unicast_semaphores) {
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            for (const auto& transfer_info : transfer_info_vec) {
                for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                    TT_ASSERT(
                        transfer_info.data.size() == write_packed_len,
                        "Not all data vectors in write packed semaphore cmd equal in len");
                    num_packed_cmds += 1;
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT);
            uint32_t ucast_payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + ucast_payload_sizeB, PCIE_ALIGNMENT);
        }

        program.command_indices.cb_configs_payload_start =
            (cmd_sequence_sizeB + CQ_PREFETCH_CMD_BARE_MIN_SIZE) / sizeof(uint32_t);
        for (const CoreRange& core_range : program.circular_buffers_unique_coreranges()) {
            cmd_sequence_sizeB += align(
                CQ_PREFETCH_CMD_BARE_MIN_SIZE +
                    (UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * NUM_CIRCULAR_BUFFERS * sizeof(uint32_t)),
                PCIE_ALIGNMENT);
        }

        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            const auto& kg_transfer_info = program.program_transfer_info.kernel_bins[buffer_idx];
            for (int kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
                for (const pair<uint32_t, uint32_t>& dst_noc_info : kg_transfer_info.dst_noc_info) {
                    cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
                    cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
                }
            }
        }

        if (program.program_transfer_info.num_active_cores > 0) {
            cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
        }

        for (const auto& transfer_info : program.program_transfer_info.go_signals) {
            for (const pair<uint32_t, uint32_t>& dst_noc_info : transfer_info.dst_noc_info) {
                cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
                cmd_sequence_sizeB += align(transfer_info.data.size() * sizeof(uint32_t), PCIE_ALIGNMENT);
            }
        }

        DeviceCommand command_sequence(cmd_sequence_sizeB);

        // Semaphores
        // Multicast Semaphore Cmd
        for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.multicast_semaphores) {
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_sub_cmds;
            std::vector<const void*> sem_data;

            for (const auto& transfer_info : transfer_info_vec) {
                for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                    num_packed_cmds += 1;
                    multicast_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                        .noc_xy_addr = dst_noc_info.first, .num_mcast_dests = dst_noc_info.second});
                    sem_data.emplace_back(transfer_info.data.data());
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT);
            uint32_t payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;

            command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_packed_cmds, dst, write_packed_len * sizeof(uint32_t), payload_sizeB, multicast_sub_cmds, sem_data);
        }

        // Unicast Semaphore Cmd
        for (const auto& [dst, transfer_info_vec] : program.program_transfer_info.unicast_semaphores) {
            // TODO: loop over things inside transfer_info[i]
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds;
            std::vector<const void*> sem_data;

            for (const auto& transfer_info : transfer_info_vec) {
                for (const auto& dst_noc_info : transfer_info.dst_noc_info) {
                    num_packed_cmds += 1;
                    unicast_sub_cmds.emplace_back(
                        CQDispatchWritePackedUnicastSubCmd{.noc_xy_addr = dst_noc_info.first});
                    sem_data.emplace_back(transfer_info.data.data());
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT);
            uint32_t payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;

            command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_packed_cmds, dst, write_packed_len * sizeof(uint32_t), payload_sizeB, unicast_sub_cmds, sem_data);
        }

        // CB Configs commands
        for (const CoreRange& core_range : program.circular_buffers_unique_coreranges()) {
            std::vector<uint32_t> cb_config_payload(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * NUM_CIRCULAR_BUFFERS, 0);
            CoreCoord physical_start = device->physical_core_from_logical_core(core_range.start, CoreType::WORKER);
            CoreCoord physical_end = device->physical_core_from_logical_core(core_range.end, CoreType::WORKER);

            uint32_t dst_noc_multicast_encoding =
                NOC_MULTICAST_ENCODING(physical_start.x, physical_start.y, physical_end.x, physical_end.y);

            uint32_t num_receivers = core_range.size();

            for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers_on_corerange(core_range)) {
                for (const auto buffer_index : cb->buffer_indices()) {
                    // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                    // cb config payload
                    uint32_t base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * (uint32_t)buffer_index;
                    cb_config_payload[base_index] = cb->address() >> 4;
                    cb_config_payload[base_index + 1] = cb->size() >> 4;
                    cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                    cb_config_payload[base_index + 3] = (cb->size() / cb->num_pages(buffer_index)) >> 4;
                }
            }
            command_sequence.add_dispatch_write_linear(
                true,  // flush_prefetch
                num_receivers,
                dst_noc_multicast_encoding,
                CIRCULAR_BUFFER_CONFIG_BASE,
                UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * NUM_CIRCULAR_BUFFERS * sizeof(uint32_t),
                cb_config_payload.data());
        }

        // Program Binaries
        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            const auto& kg_transfer_info = program.program_transfer_info.kernel_bins[buffer_idx];
            for (int kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
                for (const pair<uint32_t, uint32_t>& dst_noc_info : kg_transfer_info.dst_noc_info) {
                    command_sequence.add_dispatch_write_linear(
                        false,                // flush_prefetch
                        dst_noc_info.second,  // num_mcast_dests
                        dst_noc_info.first,   // noc_xy_addr
                        kg_transfer_info.dst_base_addrs[kernel_idx],
                        align(kg_transfer_info.lengths[kernel_idx], NOC_DRAM_ALIGNMENT_BYTES));
                    // Difference between prefetch total relayed pages and dispatch write linear
                    uint32_t relayed_bytes =
                        align(kg_transfer_info.lengths[kernel_idx], DeviceCommand::PROGRAM_PAGE_SIZE);
                    // length_adjust needs to be aligned to NOC_DRAM_ALIGNMENT
                    uint16_t length_adjust =
                        uint16_t(relayed_bytes - align(kg_transfer_info.lengths[kernel_idx], NOC_DRAM_ALIGNMENT_BYTES));

                    uint32_t base_address, page_offset;
                    if (kg_transfer_info.page_offsets[kernel_idx] > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
                        const uint32_t num_banks =
                            this->device->num_banks(this->program.kg_buffers[buffer_idx]->buffer_type());
                        page_offset = kg_transfer_info.page_offsets[kernel_idx] % num_banks;
                        uint32_t num_full_pages_written_per_bank =
                            kg_transfer_info.page_offsets[kernel_idx] / num_banks;
                        base_address =
                            this->program.kg_buffers[buffer_idx]->address() +
                            num_full_pages_written_per_bank * this->program.kg_buffers[buffer_idx]->page_size();
                    } else {
                        base_address = this->program.kg_buffers[buffer_idx]->address();
                        page_offset = kg_transfer_info.page_offsets[kernel_idx];
                    }

                    command_sequence.add_prefetch_relay_paged(
                        true,  // is_dram
                        page_offset,
                        base_address,
                        this->program.kg_buffers[buffer_idx]->page_size(),
                        relayed_bytes / this->program.kg_buffers[buffer_idx]->page_size(),
                        length_adjust);
                }
            }
        }

        // Wait Noc Write Barrier, wait for binaries to be written to worker cores
        if (program.program_transfer_info.num_active_cores > 0) {
            // Wait Noc Write Barrier, wait for binaries to be written to worker cores
            // TODO: any way to not have dispatcher poll the addr here?
            command_sequence.add_dispatch_wait(true, DISPATCH_MESSAGE_ADDR, 0);
        }

        // Go Signals
        for (const auto& transfer_info : program.program_transfer_info.go_signals) {
            for (const pair<uint32_t, uint32_t>& dst_noc_info : transfer_info.dst_noc_info) {
                command_sequence.add_dispatch_write_linear(
                    true,                 // flush_prefetch
                    dst_noc_info.second,  // num_mcast_dests
                    dst_noc_info.first,   // noc_xy_addr
                    transfer_info.dst_base_addr,
                    transfer_info.data.size() * sizeof(uint32_t),
                    transfer_info.data.data());
            }
        }
        // TODO: add GO for FD2.1
        program.cached_device_commands = command_sequence.cmd_vector();
        return command_sequence;
    } else {
        DeviceCommand command_sequence(program.cached_device_commands.size() * sizeof(uint32_t));
        command_sequence.update_cmd_sequence(
            0, program.cached_device_commands.data(), program.cached_device_commands.size() * sizeof(uint32_t));

        uint32_t cb_payload_index = program.command_indices.cb_configs_payload_start;
        for (const CoreRange& core_range : program.circular_buffers_unique_coreranges()) {
            std::vector<uint32_t> cb_config_payload(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * NUM_CIRCULAR_BUFFERS, 0);
            for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers_on_corerange(core_range)) {
                for (const auto buffer_index : cb->buffer_indices()) {
                    // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                    // cb config payload
                    uint32_t base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * (uint32_t)buffer_index;
                    cb_config_payload[base_index] = cb->address() >> 4;
                    cb_config_payload[base_index + 1] = cb->size() >> 4;
                    cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                    cb_config_payload[base_index + 3] = (cb->size() / cb->num_pages(buffer_index)) >> 4;
                }
            }
            command_sequence.update_cmd_sequence(
                cb_payload_index, cb_config_payload.data(), cb_config_payload.size() * sizeof(uint32_t));
            cb_payload_index += (align(
                                    CQ_PREFETCH_CMD_BARE_MIN_SIZE + (UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG *
                                                                     NUM_CIRCULAR_BUFFERS * sizeof(uint32_t)),
                                    PCIE_ALIGNMENT)) /
                                sizeof(uint32_t);
        }

        return command_sequence;
    }
}

void EnqueueProgramCommand::process() {
    // Calculate all commands size and determine how many fetch q entries to use

    // Preamble, some waits and stalls
    DeviceCommand preamble_command_sequence = this->assemble_preamble_commands();

    uint32_t preamble_fetch_size_bytes = preamble_command_sequence.size_bytes();

    // Runtime Args Command Sequence
    vector<DeviceCommand> runtime_args_command_sequences = this->assemble_runtime_args_commands();
    uint32_t runtime_args_fetch_size_bytes = 0;
    for (const auto& cmds : runtime_args_command_sequences) {
        // BRISC, NCRISC, TRISC...
        runtime_args_fetch_size_bytes += cmds.size_bytes();
    }

    // Main Command Sequence
    DeviceCommand program_command_sequence = this->assemble_device_commands();
    uint32_t program_fetch_size_bytes = program_command_sequence.size_bytes();

    uint32_t total_fetch_size_bytes =
        preamble_fetch_size_bytes + runtime_args_fetch_size_bytes + program_fetch_size_bytes;

    CoreType dispatch_core_type = dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    if (total_fetch_size_bytes <= dispatch_constants::get(dispatch_core_type).max_prefetch_command_size()) {
        this->manager.fetch_queue_reserve_back(this->command_queue_id);

        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        // Wrap issue queue
        uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
        if (write_ptr + align(total_fetch_size_bytes, 32) > command_issue_limit) {
            this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        }

        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(preamble_command_sequence.data(), preamble_fetch_size_bytes, write_ptr);
        this->manager.issue_queue_push_back(preamble_fetch_size_bytes, this->command_queue_id);

        for (const auto& cmds : runtime_args_command_sequences) {
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(cmds.data(), cmds.size_bytes(), write_ptr);
            this->manager.issue_queue_push_back(cmds.size_bytes(), this->command_queue_id);
        }
        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(program_command_sequence.data(), program_fetch_size_bytes, write_ptr);
        this->manager.issue_queue_push_back(program_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for entire program
        this->manager.fetch_queue_write(total_fetch_size_bytes, this->command_queue_id);
    } else {
        this->manager.fetch_queue_reserve_back(this->command_queue_id);

        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        // Wrap issue queue
        uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
        if (write_ptr + align(preamble_fetch_size_bytes, 32) > command_issue_limit) {
            this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        }

        this->manager.cq_write(preamble_command_sequence.data(), preamble_fetch_size_bytes, write_ptr);
        this->manager.issue_queue_push_back(preamble_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for just the wait and stall, very inefficient
        this->manager.fetch_queue_write(preamble_fetch_size_bytes, this->command_queue_id);

        for (const auto& cmds : runtime_args_command_sequences) {
            uint32_t fetch_size_bytes = cmds.size_bytes();

            this->manager.fetch_queue_reserve_back(this->command_queue_id);

            uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            // Wrap issue queue
            uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
            if (write_ptr + align(fetch_size_bytes, 32) > command_issue_limit) {
                this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
                write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            }

            this->manager.cq_write(cmds.data(), fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

            // One fetch queue entry for each runtime args write location, e.g. BRISC/NCRISC/TRISC/ERISC
            this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
        }
        this->manager.fetch_queue_reserve_back(this->command_queue_id);

        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        // Wrap issue queue
        command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
        if (write_ptr + align(program_fetch_size_bytes, 32) > command_issue_limit) {
            this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        }

        this->manager.cq_write(program_command_sequence.data(), program_fetch_size_bytes, write_ptr);
        this->manager.issue_queue_push_back(program_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for rest of program commands
        this->manager.fetch_queue_write(program_fetch_size_bytes, this->command_queue_id);
    }
}

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id,
    Device* device,
    SystemMemoryManager& manager,
    uint32_t event_id,
    uint32_t expected_num_workers_completed,
    bool clear_count) :
    command_queue_id(command_queue_id),
    device(device),
    manager(manager),
    event_id(event_id),
    expected_num_workers_completed(expected_num_workers_completed),
    clear_count(clear_count) {}

const DeviceCommand EnqueueRecordEventCommand::assemble_device_commands() {
    std::vector<uint32_t> event_payload(dispatch_constants::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = this->event_id;

    uint8_t num_hw_cqs = this->device->num_hw_cqs(); // Device initialize asserts that there can only be a maximum of 2 HW CQs
    uint32_t packed_event_payload_sizeB = align(sizeof(CQDispatchCmd) + num_hw_cqs * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT)
        + (align(dispatch_constants::EVENT_PADDED_SIZE, L1_ALIGNMENT) * num_hw_cqs);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, PCIE_ALIGNMENT);

    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB + // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED + unicast subcmds + event payload
        align(CQ_PREFETCH_CMD_BARE_MIN_SIZE + dispatch_constants::EVENT_PADDED_SIZE, PCIE_ALIGNMENT); // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID

    DeviceCommand command_sequence(cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait(
        false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed, this->clear_count);

    CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_hw_cqs);
    std::vector<const void *> event_payloads(num_hw_cqs);

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(this->device->id(), channel, cq_id);
        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{.noc_xy_addr = get_noc_unicast_encoding(dispatch_physical_core)};
        event_payloads[cq_id] = event_payload.data();
    }

    uint32_t address = this->command_queue_id == 0 ? CQ0_COMPLETION_LAST_EVENT : CQ1_COMPLETION_LAST_EVENT;
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_hw_cqs,
        address,
        dispatch_constants::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads
    );

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_host(flush_prefetch, dispatch_constants::EVENT_PADDED_SIZE, event_payload.data());

    return command_sequence;
}

void EnqueueRecordEventCommand::process() {
    DeviceCommand command_sequence = this->assemble_device_commands();

    uint32_t fetch_size_bytes = command_sequence.size_bytes();

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(command_sequence.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
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
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
    // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
    // TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
    //     "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
    //     event, command_queue_id);
}

const DeviceCommand EnqueueWaitForEventCommand::assemble_device_commands() {
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE; // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

    DeviceCommand command_sequence(cmd_sequence_sizeB);

    uint32_t last_completed_event_address = sync_event.cq_id == 0 ? CQ0_COMPLETION_LAST_EVENT : CQ1_COMPLETION_LAST_EVENT;
    command_sequence.add_dispatch_wait(false, last_completed_event_address, sync_event.event_id, this->clear_count);

    return command_sequence;
}

void EnqueueWaitForEventCommand::process() {
    DeviceCommand command_sequence = this->assemble_device_commands();

    uint32_t fetch_size_bytes = command_sequence.size_bytes();

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    // Wrap issue queue
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
    if (write_ptr + align(fetch_size_bytes, 32) > command_issue_limit) {
        this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    }
    this->manager.cq_write(command_sequence.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
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

const DeviceCommand EnqueueTraceCommand::assemble_device_commands() {
    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_EXEC_BUF

    DeviceCommand command_sequence(cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait(
        false, DISPATCH_MESSAGE_ADDR, this->expected_num_workers_completed, this->clear_count);

    if (this->clear_count) {
        this->expected_num_workers_completed = 0;
    }

    uint32_t page_size = buffer.page_size();
    uint32_t page_size_log2 = __builtin_ctz(page_size);
    TT_ASSERT((page_size & (page_size - 1)) == 0, "Page size must be a power of 2");

    command_sequence.add_prefetch_exec_buf(buffer.address(), page_size_log2, buffer.num_pages());

    return command_sequence;
}

void EnqueueTraceCommand::process() {
    DeviceCommand command_sequence = this->assemble_device_commands();

    uint32_t fetch_size_bytes = command_sequence.size_bytes();

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    // Wrap issue queue
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->command_queue_id);
    if (write_ptr + align(fetch_size_bytes, 32) > command_issue_limit) {
        this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    }
    this->manager.cq_write(command_sequence.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
    // log_trace(LogDispatch, "EnqueueTraceCommand issued write_ptr={}, fetch_size={}, commands={}", write_ptr, fetch_size_bytes, this->commands);
}

EnqueueTerminateCommand::EnqueueTerminateCommand(
    uint32_t command_queue_id,
    Device* device,
    SystemMemoryManager& manager) :
    command_queue_id(command_queue_id), device(device), manager(manager) {}

const DeviceCommand EnqueueTerminateCommand::assemble_device_commands() {
    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE + // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_TERMINATE

    DeviceCommand command_sequence(cmd_sequence_sizeB);
    command_sequence.add_dispatch_terminate();
    command_sequence.add_prefetch_terminate();

    return command_sequence;
}

void EnqueueTerminateCommand::process() {
    DeviceCommand command_sequence = this->assemble_device_commands();

    uint32_t fetch_size_bytes = command_sequence.size_bytes();

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(command_sequence.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
}


// HWCommandQueue section
HWCommandQueue::HWCommandQueue(Device* device, uint32_t id) :
    manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    this->expected_num_workers_completed = 0;
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {

        // TODO: SEND THE TERMINATE CMD?

        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted commands: {}", this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->exit_condition = true;
        this->completion_queue_thread.join();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking) {
    command.process();
    if (blocking) {
        this->finish();
    }
}

// TODO: Currently converting page ordering from interleaved to sharded and then doing contiguous read/write
//  Look into modifying command to do read/write of a page at a time to avoid doing copy
void convert_interleaved_to_sharded_on_host(void * swapped, const void* host, const Buffer& buffer) {
    const uint32_t num_pages = buffer.num_pages();
    const uint32_t page_size = buffer.page_size();

    std::set<uint32_t> pages_seen;
    uint32_t shard_width_in_pages = buffer.shard_spec().tensor_shard_spec.shape[1] / buffer.shard_spec().page_shape[1];
    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {
        uint32_t local_num_pages;
        auto host_page_id = page_id;
        auto dev_page_id = buffer_page_mapping.host_page_to_dev_page_mapping_[host_page_id];
        TT_ASSERT(host_page_id < num_pages and host_page_id >= 0);
        memcpy((char*)swapped + dev_page_id * page_size, (char*)host + host_page_id * page_size, page_size);
    }
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    CoreType dispatch_core_type = dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
        // Note that the src_page_index is the device page idx, not the host page idx
        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            uint32_t num_pages_to_read = buffer_page_mapping.core_host_page_indices_[core_id].size();
            if (num_pages_to_read > 0) {
                uint32_t host_page = buffer_page_mapping.core_host_page_indices_[core_id][0];
                src_page_index = buffer_page_mapping.host_page_to_dev_page_mapping_[host_page];
                unpadded_dst_offset = host_page * buffer.page_size();

                auto command = EnqueueReadShardedBufferCommand(
                    this->id, this->device, buffer, dst, this->manager, this->expected_num_workers_completed, buffer_page_mapping, src_page_index, num_pages_to_read);

                this->issued_completion_q_reads.push(
                    detail::ReadBufferDescriptor(buffer, padded_page_size, dst, unpadded_dst_offset, num_pages_to_read, src_page_index)
                );
                this->num_entries_in_completion_q++;

                this->enqueue_command(command, false);
            }
        }
        if (blocking) {
            this->finish();
        } else {
            std::shared_ptr<Event> event = std::make_shared<Event>();
            this->enqueue_record_event(event);
        }
    } else {
        // this is a streaming command so we don't need to break down to multiple
        auto command = EnqueueReadInterleavedBufferCommand(
            this->id, this->device, buffer, dst, this->manager, this->expected_num_workers_completed, src_page_index, pages_to_read);

        this->issued_completion_q_reads.push(
            detail::ReadBufferDescriptor(buffer, padded_page_size, dst, unpadded_dst_offset, pages_to_read, src_page_index)
        );
        this->num_entries_in_completion_q++;

        this->enqueue_command(command, blocking);
        if (not blocking) { // should this be unconditional?
            std::shared_ptr<Event> event = std::make_shared<Event>();
            this->enqueue_record_event(event);
        }
    }
}

void HWCommandQueue::enqueue_write_buffer(std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<const Buffer>> buffer, HostDataType src, bool blocking) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    std::visit ([this, &buffer, &blocking](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        std::visit ([this, &buffer, &blocking, &data](auto&& b) {
            using type_buf = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, const void*>) {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data, blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data, blocking);
                }
            } else {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data.get() -> data(), blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data.get() -> data(), blocking);
                }
            }
        }, buffer);
    }, src);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");

    uint32_t padded_page_size = align(buffer.page_size(), 32);

    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    CoreType dispatch_core_type = dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    uint32_t max_data_sizeB = max_prefetch_command_size - ((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) * 2); // * 2 to account for issue

    uint32_t dst_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
        TT_ASSERT(max_data_sizeB >= padded_page_size);
        const void* remapped_src = (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
                                    buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED)
                                       ? convert_interleaved_to_sharded_on_host(src, buffer, buffer_page_mapping)
                                       : src;

        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            std::vector<uint32_t> core_pages;
            for (uint32_t i=0; i < buffer_page_mapping.dev_page_to_core_mapping_.size(); i++) {
                if (buffer_page_mapping.dev_page_to_core_mapping_[i] == core_id) {
                    core_pages.push_back(i);
                }
            }

            uint32_t num_pages = core_pages.size();
            uint32_t curr_page_idx_in_shard = 0;
            while (num_pages != 0) {
                uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)); // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
                bool issue_wait = dst_page_index == 0; // only stall for the first write of the buffer
                if (issue_wait) {
                    data_offset_bytes *= 2; // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                }
                uint32_t space_available_bytes = std::min(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
                int32_t num_pages_available =
                    (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(padded_page_size);

                uint32_t pages_to_write = std::min(num_pages, (uint32_t)num_pages_available);
                if (pages_to_write > 0) {
                    uint32_t bank_base_address = buffer.address() + curr_page_idx_in_shard * padded_page_size;
                    // Technically we are going through dst pages in order
                    dst_page_index = core_pages[curr_page_idx_in_shard];

                    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

                    auto command = EnqueueWriteShardedBufferCommand(
                        this->id, this->device, buffer, remapped_src, this->manager, issue_wait, this->expected_num_workers_completed, bank_base_address, buffer_page_mapping, padded_page_size, dst_page_index, pages_to_write);

                    this->enqueue_command(command, false);
                    curr_page_idx_in_shard += pages_to_write;
                    num_pages -= pages_to_write;
                } else {
                    this->manager.wrap_issue_queue_wr_ptr(this->id);
                }
            }
        }
        if (remapped_src != src) {
            free((void*)remapped_src);
        }
    } else {

        uint32_t total_pages_to_write = buffer.num_pages();
        bool write_partial_pages = padded_page_size > max_data_sizeB;
        uint32_t page_size_to_write = padded_page_size;
        uint32_t padded_buffer_size = buffer.num_pages() * padded_page_size;
        if (write_partial_pages) {
            TT_ASSERT(buffer.num_pages() == 1, "TODO: add support for multi-paged buffer with page size > 64KB");
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

        uint32_t max_num_pages_to_write = write_partial_pages ?
            ( num_pages_round_robined > 0 ? (num_banks * num_partials_round_robined) : num_banks_with_residual_pages ) : total_pages_to_write;

        uint32_t bank_base_address = buffer.address();

        uint32_t num_full_pages_written = 0;
        while (total_pages_to_write > 0) {
            uint32_t data_offsetB = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)); // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
            bool issue_wait = (dst_page_index == 0 and bank_base_address == buffer.address()); // only stall for the first write of the buffer
            if (issue_wait) {
                data_offsetB *= 2; // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            }

            uint32_t space_availableB = std::min(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
            int32_t num_pages_available = (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(page_size_to_write);

            if (num_pages_available <= 0) {
                this->manager.wrap_issue_queue_wr_ptr(this->id);
                continue;
            }

            uint32_t num_pages_to_write = std::min(std::min((uint32_t)num_pages_available, max_num_pages_to_write), total_pages_to_write);

            // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
            // To handle larger page offsets move bank base address up and update page offset to be relative to the new bank address
            if (dst_page_index > 0xFFFF or (num_pages_to_write == max_num_pages_to_write and write_partial_pages)) {
                uint32_t num_banks_to_use = write_partial_pages ? max_num_pages_to_write : num_banks;
                uint32_t residual = dst_page_index % num_banks_to_use;
                uint32_t num_pages_written_per_bank = dst_page_index / num_banks_to_use;
                bank_base_address += num_pages_written_per_bank * page_size_to_write;
                dst_page_index = residual;
            }

            tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", this->id);

            auto command = EnqueueWriteInterleavedBufferCommand(
                this->id, this->device, buffer, src, this->manager, issue_wait, this->expected_num_workers_completed, bank_base_address, page_size_to_write, dst_page_index, num_pages_to_write);
            this->enqueue_command(command, false); // don't block until the entire src data is enqueued in the issue queue

            total_pages_to_write -= num_pages_to_write;
            dst_page_index += num_pages_to_write;
        }
    }

    if (blocking) {
        this->finish();
    } else {
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
    }
}

void HWCommandQueue::enqueue_program(
    Program& program, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");

    this->force_commands([&]() {
        if (not program.loaded_onto_device) {
            TT_ASSERT(program.program_transfer_info.kernel_bins.size() == program.kg_buffers.size());
            for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
                this->enqueue_write_buffer(*program.kg_buffers[buffer_idx], program.program_transfer_info.kernel_bins[buffer_idx].data.data(), false);
            }
        }
    });

    auto command = EnqueueProgramCommand(this->id, this->device, program, this->manager, this->expected_num_workers_completed);
    this->enqueue_command(command, blocking);

    if (this->manager.get_bypass_mode()) {
        this->trace_ctx->num_completion_worker_cores += program.program_transfer_info.num_active_cores;
        this->trace_ctx->owned_buffer_pool.insert(this->trace_ctx->owned_buffer_pool.end(), program.kg_buffers.begin(), program.kg_buffers.end());
    } else {
        this->expected_num_workers_completed += program.program_transfer_info.num_active_cores;
    }
}

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true; // what does this mean???

    if (this->manager.get_bypass_mode()) {
        TT_FATAL(this->trace_ctx != nullptr, "A trace context must be present in bypass mode!");
        event->event_id = this->trace_ctx->relative_event_id(event->event_id);
    }
    auto command = EnqueueRecordEventCommand(this->id, this->device, this->manager, event->event_id, this->expected_num_workers_completed, clear_count);
    this->enqueue_command(command, false);

    if (clear_count) {
        this->expected_num_workers_completed = 0;
    }
    if (this->manager.get_bypass_mode()) {
        this->trace_ctx->traced_completion_q_reads.push(detail::ReadEventDescriptor(event->event_id));
        this->trace_ctx->num_completion_q_reads++;
    } else {
        this->issued_completion_q_reads.push(detail::ReadEventDescriptor(event->event_id));
        this->num_entries_in_completion_q++;
    }
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> sync_event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    auto command = EnqueueWaitForEventCommand(this->id, this->device, this->manager, *sync_event, clear_count);
    this->enqueue_command(command, false);

    if (clear_count) {
        this->manager.reset_event_id(this->id);
    }
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event);
}

void HWCommandQueue::enqueue_trace(const uint32_t trace_id, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_trace");

    auto trace_inst = Trace::get_instance(trace_id);
    auto command = EnqueueTraceCommand(this->id, this->device, this->manager, *trace_inst.buffer, this->expected_num_workers_completed);

    // Emit the completion queue entries from the trace
    auto& cmpl_q = trace_inst.desc->traced_completion_q_reads;
    uint32_t num_events = 0;
    uint32_t event_id = this->manager.get_next_event(this->id);
    for (auto read_descriptor : cmpl_q) {
        std::visit(
            [&](auto&& read_descriptor) {
                using T = std::decay_t<decltype(read_descriptor)>;
                if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                    TT_THROW("Device trace does not support ReadBuffer commands, please perform on Host instead!");
                } else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                    read_descriptor.set_global_offset(event_id);
                    this->issued_completion_q_reads.push(read_descriptor);
                    this->num_entries_in_completion_q++;
                    num_events++;
                }
            },
            read_descriptor);

    }
    this->enqueue_command(command, false);

    // Increment the global event counter due to trace emitting events in a batch
    this->manager.increment_event_id(this->id, num_events);

    // Increment the exepected worker cores counter due to trace programs completions
    this->expected_num_workers_completed += trace_inst.desc->num_completion_worker_cores;

    if (blocking) {
        this->finish();
    } else {
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
    }
}

void HWCommandQueue::copy_into_user_space(const detail::ReadBufferDescriptor &read_buffer_descriptor, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, dev_page_to_host_page_mapping, dst, dst_offset, num_pages_read, cur_dev_page_id] = read_buffer_descriptor;

    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;
    uint32_t dev_page_id = cur_dev_page_id;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    uint32_t offset_in_completion_q_data = (sizeof(CQDispatchCmd) / sizeof(uint32_t));

    uint32_t pad_size_bytes = padded_page_size - page_size;
    uint32_t padded_page_increment = (padded_page_size / sizeof(uint32_t));
    uint32_t page_increment = (page_size / sizeof(uint32_t));

    static std::vector<uint32_t> completion_q_data;

    while (remaining_bytes_to_read != 0) {
        this->manager.completion_queue_wait_front(this->id, this->exit_condition);

        if (this->exit_condition) {
            break;
        }

        uint32_t completion_queue_write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(
            this->device->id(), this->id, this->manager.get_cq_size());
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
            bytes_avail_in_completion_queue = this->manager.get_completion_queue_limit(this->id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(remaining_bytes_to_read, bytes_avail_in_completion_queue);
        uint32_t num_pages_xfered = (bytes_xfered + dispatch_constants::TRANSFER_PAGE_SIZE - 1) / dispatch_constants::TRANSFER_PAGE_SIZE;

        completion_q_data.resize(bytes_xfered / sizeof(uint32_t));

        tt::Cluster::instance().read_sysmem(
            completion_q_data.data(), bytes_xfered, completion_q_read_ptr, mmio_device_id, channel);

        this->manager.completion_queue_pop_front(num_pages_xfered, this->id);

        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_layout == TensorMemoryLayout::INTERLEAVED or
            buffer_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            if ((page_size % 32) == 0) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data * sizeof(uint32_t);
                memcpy(contiguous_dst, completion_q_data.data() + offset_in_completion_q_data, data_bytes_xfered);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint32_t dst_offset_bytes = 0;

                while (src_offset < completion_q_data.size()) {

                    uint32_t src_offset_increment = padded_page_increment;
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = (completion_q_data.size() - src_offset) * sizeof(uint32_t);
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = (num_bytes_to_copy/sizeof(uint32_t));
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes / sizeof(uint32_t);
                            // Only pad data left in queue
                            } else {
                                offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq) / sizeof(uint32_t);
                            }
                        }
                    } else if (src_offset + padded_page_increment >= completion_q_data.size()) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = (completion_q_data.size() - src_offset) * sizeof(uint32_t);
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size );
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_increment - num_bytes_remaining / sizeof(uint32_t);
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    memcpy(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        completion_q_data.data() + src_offset,
                        num_bytes_to_copy
                    );

                    src_offset += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else if (
            buffer_layout == TensorMemoryLayout::WIDTH_SHARDED or
            buffer_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            uint32_t src_offset = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint32_t dst_offset_bytes = contig_dst_offset;

            while(src_offset < completion_q_data.size()) {

                uint32_t src_offset_increment = padded_page_increment;
                uint32_t num_bytes_to_copy;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = (completion_q_data.size() - src_offset) * sizeof(uint32_t);
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = (num_bytes_to_copy/sizeof(uint32_t));
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        dev_page_id++;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes / sizeof(uint32_t);
                            offset_in_completion_q_data = 0;
                        // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq) / sizeof(uint32_t);
                        }
                    }
                } else if (src_offset + padded_page_increment >= completion_q_data.size()) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    uint32_t host_page_id = dev_page_to_host_page_mapping[dev_page_id];
                    dst_offset_bytes = host_page_id * page_size;
                    uint32_t num_bytes_remaining = (completion_q_data.size() - src_offset) * sizeof(uint32_t);
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_increment - num_bytes_remaining / sizeof(uint32_t);
                        dev_page_id++;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    uint32_t host_page_id = dev_page_to_host_page_mapping[dev_page_id];
                    dst_offset_bytes = host_page_id * page_size;
                    dev_page_id++;
                }

                memcpy(
                        (char*)(uint64_t(dst) + dst_offset_bytes),
                        completion_q_data.data() + src_offset,
                        num_bytes_to_copy
                    );
                src_offset += src_offset_increment;
                dst_offset_bytes += num_bytes_to_copy;
            }
            contig_dst_offset = dst_offset_bytes;
        }
    }
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {

                std::variant<detail::ReadBufferDescriptor, detail::ReadEventDescriptor> read_descriptor = *(this->issued_completion_q_reads.pop());

                this->manager.completion_queue_wait_front(this->id, this->exit_condition); // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN

                if (this->exit_condition) {  // Early exit
                    return;
                }

                std::visit(
                    [&](auto&& read_descriptor)
                    {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            this->copy_into_user_space(read_descriptor, mmio_device_id, channel);
                        }
                        else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
                            static std::vector<uint32_t> dispatch_cmd_and_event((sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(), sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE, read_ptr, mmio_device_id, channel);
                            uint32_t event_completed = dispatch_cmd_and_event.at(sizeof(CQDispatchCmd) / sizeof(uint32_t));
                            TT_ASSERT(event_completed == read_descriptor.event_id, "Event Order Issue: expected to read back completion signal for event {} but got {}!", read_descriptor.event_id, event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, read_descriptor.get_global_event_id());
                            log_trace(LogAlways, "Completion queue popped event {} (global: {})", event_completed, read_descriptor.get_global_event_id());
                        }
                    },
                    read_descriptor
                );
            }
            this->num_completed_completion_q_reads += num_events_to_read;
        } else if (this->exit_condition) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
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
                this->exit_condition = true;
                this->dprint_server_hang = true;
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->illegal_noc_txn_hang = true;
                return;
            }
        }
    } else {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads);
    }
}

volatile bool HWCommandQueue::is_dprint_server_hung() {
    return dprint_server_hang;
}

volatile bool HWCommandQueue::is_noc_hung() {
    return illegal_noc_txn_hang;
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id);
    auto command = EnqueueTerminateCommand(this->id, this->device, this->manager);
    this->enqueue_command(command, false);
}

void EnqueueAddBufferToProgramImpl(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program) {
    std::visit([program] (auto&& b) {
        using buffer_type = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
            std::visit([&b] (auto&& p) {
                using program_type = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<program_type, std::reference_wrapper<Program>>) {
                    p.get().add_buffer(b);
                }
                else {
                    p->add_buffer(b);
                }
            }, program);
        }
    }, buffer);
}

void EnqueueAddBufferToProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program, bool blocking) {
    EnqueueAddBufferToProgramImpl(buffer, program);
    // cq.run_command(CommandInterface{
    //     .type = EnqueueCommandType::ADD_BUFFER_TO_PROGRAM,
    //     .blocking = blocking,
    //     .buffer = buffer,
    //     .program = program,
    // });
}

void EnqueueUpdateRuntimeArgsImpl (const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    auto& kernel_runtime_args = runtime_args_md.kernel->runtime_args(runtime_args_md.core_coord);
    for (const auto& idx : runtime_args_md.update_idx) {
        kernel_runtime_args[idx] = resolved_runtime_args[idx];
    }
}

void EnqueueUpdateRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
            .update_idx = update_idx,
    };
    EnqueueUpdateRuntimeArgsImpl(runtime_args_md);
    // cq.run_command( CommandInterface {
    //     .type = EnqueueCommandType::UPDATE_RUNTIME_ARGS,
    //     .blocking = blocking,
    //     .runtime_args_md = runtime_args_md,
    // });
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    runtime_args_md.kernel -> set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueSetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
    };
    EnqueueSetRuntimeArgsImpl(runtime_args_md);
    // cq.run_command( CommandInterface {
    //     .type = EnqueueCommandType::SET_RUNTIME_ARGS,
    //     .blocking = blocking,
    //     .runtime_args_md = runtime_args_md,
    // });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer->address();
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    EnqueueGetBufferAddrImpl(dst_buf_addr, buffer);
    // cq.run_command( CommandInterface {
    //     .type = EnqueueCommandType::GET_BUF_ADDR,
    //     .blocking = blocking,
    //     .shadow_buffer = buffer,
    //     .dst = dst_buf_addr
    // });
}

void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md) {
    Buffer* buffer = alloc_md.buffer;
    uint32_t allocated_addr;
    if(is_sharded(buffer->buffer_layout())) {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->shard_spec().size() * buffer->num_cores() * buffer->page_size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, buffer->num_cores());
    }
    else {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, std::nullopt);
    }
    buffer->set_address(static_cast<uint64_t>(allocated_addr));
}

void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking) {
    auto alloc_md = AllocBufferMetadata{
        .buffer = buffer,
        .allocator = *(buffer->device()->allocator_),
        .bottom_up = bottom_up,
    };
    EnqueueAllocateBufferImpl(alloc_md);
    // cq.run_command(CommandInterface {
    //     .type = EnqueueCommandType::ALLOCATE_BUFFER,
    //     .blocking = blocking,
    //     .alloc_md = alloc_md,
    // });
}

void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md) {
    allocator::deallocate_buffer(alloc_md.allocator, alloc_md.device_address, alloc_md.buffer_type);
}

void EnqueueDeallocateBuffer(CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking) {
    // Need to explictly pass in relevant buffer attributes here, since the Buffer* ptr can be deallocated a this point
    auto alloc_md = AllocBufferMetadata {
        .allocator = allocator,
        .buffer_type = buffer_type,
        .device_address = device_address,
    };
    EnqueueDeallocateBufferImpl(alloc_md);
    // cq.run_command(CommandInterface{
    //     .type = EnqueueCommandType::DEALLOCATE_BUFFER,
    //     .blocking = blocking,
    //     .alloc_md = alloc_md,
    // });
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& dst, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    Buffer & b = std::holds_alternative<std::shared_ptr<Buffer>>(buffer) ? *(std::get< std::shared_ptr<Buffer> > ( buffer )) :
                                                                            std::get<std::reference_wrapper<Buffer>>(buffer).get();
    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    std::visit ( [&dst](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>>) {
            dst.resize(b.get().page_size() * b.get().num_pages() / sizeof(uint32_t));
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
            dst.resize(b->page_size() * b->num_pages() / sizeof(uint32_t));
        }
    }, buffer);

    // TODO(agrebenisan): Move to deprecated
    EnqueueReadBuffer(cq, buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& src, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .dst = dst
    });
}

void EnqueueReadBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    std::visit ( [&cq, dst, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer> > ) {
            cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking);
        }
    }, buffer);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .src = src
    });
}

void EnqueueWriteBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    std::visit ( [&cq, src, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>> ) {
            cq.hw_command_queue().enqueue_write_buffer(b, src, blocking);
        }
    }, buffer);
}

void EnqueueProgram(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    detail::DispatchStateCheck(true);
    if (cq.get_mode() != CommandQueue::CommandQueueMode::TRACE) {
        TT_FATAL(cq.id() == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_PROGRAM,
        .blocking = blocking,
        .program = program
    });
}

void EnqueueProgramImpl(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    ZoneScoped;
    std::visit ( [&cq, blocking](auto&& program) {
        ZoneScoped;
        using T = std::decay_t<decltype(program)>;
        Device * device = cq.device();
        if constexpr (std::is_same_v<T, std::reference_wrapper<Program>>) {
            detail::CompileProgram(device, program);
            program.get().allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(program, device);
            cq.hw_command_queue().enqueue_program(program, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program.get().release_buffers();
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Program>>) {
            detail::CompileProgram(device, *program);
            program->allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(*program, device);
            cq.hw_command_queue().enqueue_program(*program, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program->release_buffers();
        }
    }, program);
}

void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event) {
    TT_ASSERT(event->device == nullptr, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->event_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->cq_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");

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
    event->wait_until_ready(); // Block until event populated. Worker thread.
    log_trace(tt::LogMetal, "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(), event->cq_id, event->event_id, cq.device()->id(), cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}

void EventSynchronize(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready(); // Block until event populated. Parent thread.
    log_trace(tt::LogMetal, "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})", event->device->id(), event->cq_id, event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::OptionsG.get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_ASSERT(false, "Command Queue could not complete EventSynchronize. See {} for details.", tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool EventQuery(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready(); // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(tt::LogMetal, "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed, event->device->id(), event->cq_id, event->event_id);
    return event_completed;
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::FINISH,
        .blocking = true
    });
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_dprint_server_hung()),
              "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_noc_hung()),
              "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
               tt::watcher_get_log_file_name());
}

void FinishImpl(CommandQueue& cq) {
    cq.hw_command_queue().finish();
}

CommandQueue& BeginTrace(Trace& trace) {
    log_debug(LogMetalTrace, "Begin trace capture");
    trace.begin_capture();
    return trace.queue();
}

void EndTrace(Trace& trace) {
    trace.end_capture();
    log_debug(LogMetalTrace, "End trace capture");
}

uint32_t InstantiateTrace(Trace& trace, CommandQueue& cq) {
    uint32_t trace_id = trace.instantiate(cq);
    return trace_id;
}

void ReleaseTrace(uint32_t trace_id) {
    if (trace_id == -1) {
        Trace::release_all();
    } else if (Trace::has_instance(trace_id)) {
        Trace::remove_instance(trace_id);
    }
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(Trace::has_instance(trace_id), "Trace instance " + std::to_string(trace_id) + " must exist on device");
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking,
        .trace_id = trace_id
    });
}

void EnqueueTraceImpl(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    cq.hw_command_queue().enqueue_trace(trace_id, blocking);
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device),
    cq_id(id),
    mode(mode),
    worker_state(CommandQueueState::IDLE) {
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
    worker_state(CommandQueueState::IDLE) {
}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (not this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() {
    return this->device()->hw_command_queue(this->cq_id);
}

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
    TT_ASSERT(not this->trace_mode(), "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
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
            TT_ASSERT(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id, "Only main thread or worker thread can run commands through the SW command queue");
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
        case EnqueueCommandType::UPDATE_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueUpdateRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), command.program.value());
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, command.program.value(), command.blocking.value());
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
        case EnqueueCommandType::FINISH:
            FinishImpl(*this);
            break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default:
            TT_THROW("Invalid command type");
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
