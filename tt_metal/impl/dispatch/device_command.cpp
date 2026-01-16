// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_command.hpp"

#include <cstring>

#include <random>
#include <chrono>

#include <tt_stl/aligned_allocator.hpp>
#include <tt_stl/assert.hpp>
#include "dispatch/kernels/cq_commands.hpp"
#include "dispatch/memcpy.hpp"
#include "dispatch_settings.hpp"
#include "tt_align.hpp"

namespace tt::tt_metal {

template <bool hugepage_write>
DeviceCommand<hugepage_write>::DeviceCommand(void* cmd_region, uint32_t cmd_sequence_sizeB) :
    cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_region(cmd_region) {
    TT_FATAL(
        cmd_sequence_sizeB % sizeof(uint32_t) == 0,
        "Command sequence size B={} is not {}-byte aligned",
        cmd_sequence_sizeB,
        sizeof(uint32_t));
}

template <bool hugepage_write>
template <bool hp_w, typename std::enable_if_t<!hp_w, int>>
DeviceCommand<hugepage_write>::DeviceCommand(uint32_t cmd_sequence_sizeB) : cmd_sequence_sizeB(cmd_sequence_sizeB) {
    TT_FATAL(
        cmd_sequence_sizeB % sizeof(uint32_t) == 0,
        "Command sequence size B={} is not {}-byte aligned",
        cmd_sequence_sizeB,
        sizeof(uint32_t));
    this->cmd_region_vector.resize(cmd_sequence_sizeB / sizeof(uint32_t), 0);
    this->cmd_region = this->cmd_region_vector.data();
}

template <bool hugepage_write>
DeviceCommand<hugepage_write>& DeviceCommand<hugepage_write>::operator=(const DeviceCommand& other) {
    if (this == &other) {
        return *this;
    }
    this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
    this->cmd_write_offsetB = other.cmd_write_offsetB;
    this->cmd_region_vector = other.cmd_region_vector;
    this->deepcopy(other);
    return *this;
}

template <bool hugepage_write>
DeviceCommand<hugepage_write>& DeviceCommand<hugepage_write>::operator=(DeviceCommand&& other) noexcept {
    this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
    this->cmd_write_offsetB = other.cmd_write_offsetB;
    this->cmd_region_vector = std::move(other.cmd_region_vector);
    if constexpr (hugepage_write) {
        this->deepcopy(other);
    } else {
        this->cmd_region = this->cmd_region_vector.data();
    }

    return *this;
}

template <bool hugepage_write>
DeviceCommand<hugepage_write>::DeviceCommand(const DeviceCommand& other) :
    cmd_sequence_sizeB(other.cmd_sequence_sizeB),
    cmd_write_offsetB(other.cmd_write_offsetB),
    cmd_region_vector(other.cmd_region_vector) {
    this->deepcopy(other);
}

template <bool hugepage_write>
DeviceCommand<hugepage_write>::DeviceCommand(DeviceCommand&& other) noexcept :
    cmd_sequence_sizeB(other.cmd_sequence_sizeB),
    cmd_write_offsetB(other.cmd_write_offsetB),
    cmd_region_vector(std::move(other.cmd_region_vector)) {
    if constexpr (hugepage_write) {
        this->deepcopy(other);
    } else {
        this->cmd_region = this->cmd_region_vector.data();
    }
}

template <bool hugepage_write>
uint32_t DeviceCommand<hugepage_write>::size_bytes() const {
    return this->cmd_sequence_sizeB;
}

template <bool hugepage_write>
void* DeviceCommand<hugepage_write>::data() const {
    return this->cmd_region;
}

template <bool hugepage_write>
uint32_t DeviceCommand<hugepage_write>::write_offset_bytes() const {
    return this->cmd_write_offsetB;
}

template <bool hugepage_write>
vector_aligned<uint32_t> DeviceCommand<hugepage_write>::cmd_vector() const {
    return this->cmd_region_vector;
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_wait(
    uint32_t flags, uint32_t address, uint32_t stream, uint32_t count, uint8_t dispatcher_type) {
    auto initialize_wait_cmds = [&](CQPrefetchCmd* relay_wait, CQDispatchCmd* wait_cmd) {
        relay_wait->base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_wait->relay_inline.dispatcher_type = dispatcher_type;
        relay_wait->relay_inline.length = sizeof(CQDispatchCmd);
        relay_wait->relay_inline.stride =
            tt::align(sizeof(CQDispatchCmd) + sizeof(CQPrefetchCmd), this->pcie_alignment);

        wait_cmd->base.cmd_id = CQ_DISPATCH_CMD_WAIT;
        wait_cmd->wait.flags = flags;
        wait_cmd->wait.addr = address;
        wait_cmd->wait.count = count;
        wait_cmd->wait.stream = stream;
    };
    CQPrefetchCmd* relay_wait_dst = this->reserve_space<CQPrefetchCmd*>(sizeof(CQPrefetchCmd));
    CQDispatchCmd* wait_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_wait{};
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd wait_cmd{};
        initialize_wait_cmds(&relay_wait, &wait_cmd);
        this->memcpy(relay_wait_dst, &relay_wait, sizeof(CQPrefetchCmd));
        this->memcpy(wait_cmd_dst, &wait_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_wait_cmds(relay_wait_dst, wait_cmd_dst);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_wait_with_prefetch_stall(
    uint32_t flags, uint32_t address, uint32_t stream, uint32_t count) {
    this->add_dispatch_wait(flags | CQ_DISPATCH_CMD_WAIT_FLAG_NOTIFY_PREFETCH, address, stream, count);
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_stall_cmd = [&](CQPrefetchCmd* stall_cmd) {
        *stall_cmd = {};
        stall_cmd->base.cmd_id = CQ_PREFETCH_CMD_STALL;
    };
    CQPrefetchCmd* stall_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd stall_cmd{};
        initialize_stall_cmd(&stall_cmd);
        this->memcpy(stall_cmd_dst, &stall_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_stall_cmd(stall_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_relay_linear(uint32_t noc_xy_addr, DeviceAddr lengthB, uint32_t addr) {
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmdLarge), this->pcie_alignment);
    auto initialize_relay_linear_cmd = [&](CQPrefetchCmdLarge* relay_linear_cmd) {
        relay_linear_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;
        relay_linear_cmd->relay_linear.noc_xy_addr = noc_xy_addr;
        relay_linear_cmd->relay_linear.length = lengthB;
        relay_linear_cmd->relay_linear.addr = addr;
    };
    CQPrefetchCmdLarge* relay_linear_cmd_dst = this->reserve_space<CQPrefetchCmdLarge*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmdLarge relay_linear_cmd{};
        initialize_relay_linear_cmd(&relay_linear_cmd);
        this->memcpy(relay_linear_cmd_dst, &relay_linear_cmd, sizeof(CQPrefetchCmdLarge));
    } else {
        initialize_relay_linear_cmd(relay_linear_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_relay_paged(
    uint8_t is_dram,
    uint8_t start_page,
    uint32_t base_addr,
    uint32_t page_size,
    uint32_t pages,
    uint16_t length_adjust) {
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_relay_paged_cmd = [&](CQPrefetchCmd* relay_paged_cmd) {
        TT_ASSERT((length_adjust & CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK) == length_adjust);
        relay_paged_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;
        relay_paged_cmd->relay_paged.start_page = start_page;
        relay_paged_cmd->relay_paged.is_dram_and_length_adjust =
            (is_dram << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT) |
            (length_adjust & CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK);
        relay_paged_cmd->relay_paged.base_addr = base_addr;
        relay_paged_cmd->relay_paged.page_size = page_size;
        relay_paged_cmd->relay_paged.pages = pages;
    };
    CQPrefetchCmd* relay_paged_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_paged_cmd{};
        initialize_relay_paged_cmd(&relay_paged_cmd);
        this->memcpy(relay_paged_cmd_dst, &relay_paged_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_relay_paged_cmd(relay_paged_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_relay_paged_packed(
    uint32_t length,
    const std::vector<CQPrefetchRelayPagedPackedSubCmd>& sub_cmds,
    uint16_t num_sub_cmds,
    uint32_t offset_idx) {
    static_assert(sizeof(CQPrefetchRelayPagedPackedSubCmd) % sizeof(uint32_t) == 0);

    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQPrefetchRelayPagedPackedSubCmd);
    uint32_t increment_sizeB = tt::align(sub_cmds_sizeB + sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_relay_paged_cmd = [&](CQPrefetchCmd* relay_paged_cmd) {
        relay_paged_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED_PACKED;
        relay_paged_cmd->relay_paged_packed.total_length = length;
        relay_paged_cmd->relay_paged_packed.stride = increment_sizeB;
        relay_paged_cmd->relay_paged_packed.count = num_sub_cmds;
    };
    CQPrefetchCmd* relay_paged_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_paged_cmd{};
        initialize_relay_paged_cmd(&relay_paged_cmd);
        this->memcpy(relay_paged_cmd_dst, &relay_paged_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_relay_paged_cmd(relay_paged_cmd_dst);
    }

    this->memcpy((char*)relay_paged_cmd_dst + sizeof(CQPrefetchCmd), &sub_cmds[offset_idx], sub_cmds_sizeB);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_paged_to_ringbuffer(
    const CQPrefetchPagedToRingbufferCmd& paged_to_ringbuffer_info) {
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_paged_to_ringbuffer_cmd = [&](CQPrefetchCmd* paged_to_ringbuffer_cmd) {
        paged_to_ringbuffer_cmd->base.cmd_id = CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER;
        paged_to_ringbuffer_cmd->paged_to_ringbuffer = paged_to_ringbuffer_info;
    };
    CQPrefetchCmd* paged_to_ringbuffer_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd paged_to_ringbuffer_cmd{};
        initialize_paged_to_ringbuffer_cmd(&paged_to_ringbuffer_cmd);
        this->memcpy(paged_to_ringbuffer_cmd_dst, &paged_to_ringbuffer_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_paged_to_ringbuffer_cmd(paged_to_ringbuffer_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_set_ringbuffer_offset(uint32_t offset, bool update_wp) {
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_set_ringbuffer_offset_cmd = [&](CQPrefetchCmd* set_ringbuffer_offset_cmd) {
        set_ringbuffer_offset_cmd->base.cmd_id = CQ_PREFETCH_CMD_SET_RINGBUFFER_OFFSET;
        set_ringbuffer_offset_cmd->set_ringbuffer_offset.offset = offset;
        set_ringbuffer_offset_cmd->set_ringbuffer_offset.update_wp = update_wp;
    };
    CQPrefetchCmd* set_ringbuffer_offset_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd set_ringbuffer_offset_cmd{};
        initialize_set_ringbuffer_offset_cmd(&set_ringbuffer_offset_cmd);
        this->memcpy(set_ringbuffer_offset_cmd_dst, &set_ringbuffer_offset_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_set_ringbuffer_offset_cmd(set_ringbuffer_offset_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_relay_ringbuffer(
    uint32_t num_sub_cmds, const std::vector<CQPrefetchRelayRingbufferSubCmd>& sub_cmds, uint32_t offset_idx) {
    static_assert(sizeof(CQPrefetchRelayRingbufferSubCmd) % sizeof(uint32_t) == 0);

    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQPrefetchRelayRingbufferSubCmd);
    uint32_t increment_sizeB = tt::align(sub_cmds_sizeB + sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_relay_ringbuffer_cmd = [&](CQPrefetchCmd* relay_ringbuffer_cmd) {
        relay_ringbuffer_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_RINGBUFFER;
        relay_ringbuffer_cmd->relay_ringbuffer.count = num_sub_cmds;
        relay_ringbuffer_cmd->relay_ringbuffer.stride = increment_sizeB;
    };

    CQPrefetchCmd* relay_ringbuffer_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_ringbuffer_cmd{};
        initialize_relay_ringbuffer_cmd(&relay_ringbuffer_cmd);
        this->memcpy(relay_ringbuffer_cmd_dst, &relay_ringbuffer_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_relay_ringbuffer_cmd(relay_ringbuffer_cmd_dst);
    }

    this->memcpy((char*)relay_ringbuffer_cmd_dst + sizeof(CQPrefetchCmd), &sub_cmds[offset_idx], sub_cmds_sizeB);
}

template <bool hugepage_write>
template <bool flush_prefetch, bool inline_data>
void DeviceCommand<hugepage_write>::add_dispatch_write_linear(
    uint8_t num_mcast_dests,
    uint32_t noc_xy_addr,
    DeviceAddr addr,
    DeviceAddr data_sizeB,
    const void* data,
    uint32_t write_offset_index) {
    // payload to prefetch relay inline will always be limited to the length of the cmddat queue, i.e. < 64 bits
    // Conversely if data_sizeB > 32-bits, then the data won't be inline
    uint32_t payload_sizeB = static_cast<uint32_t>(sizeof(CQDispatchCmdLarge) + (flush_prefetch ? data_sizeB : 0));
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_cmd = [&](CQDispatchCmdLarge* write_cmd) {
        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
        write_cmd->write_linear.num_mcast_dests = num_mcast_dests;
        write_cmd->write_linear.write_offset_index = write_offset_index;
        write_cmd->write_linear.noc_xy_addr = noc_xy_addr;
        write_cmd->write_linear.addr = addr;
        write_cmd->write_linear.length = data_sizeB;
    };
    CQDispatchCmdLarge* write_cmd_dst = this->reserve_space<CQDispatchCmdLarge*>(sizeof(CQDispatchCmdLarge));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmdLarge write_cmd{};
        initialize_write_cmd(&write_cmd);
        this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmdLarge));
    } else {
        initialize_write_cmd(write_cmd_dst);
    }

    // Case 1: flush_prefetch
    //  a) there is inline_data: data is provided here and follows prefetch relay inline and cq dispatch write
    //  linear so total increment size is:
    //          tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge) + data_sizeB, pcie_alignment)
    //  b) don't have inline_data: next command should be to add_data (don't do aligned increment) so total
    //  increment size is:
    //          sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge)
    // Case 2: !flush_prefetch: no data, increment size is:
    //          tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge), pcie_alignment)
    // Note that adding prefetch_relay_inline and writing the dispatch command already increment cmd_write_offsetB
    // via calls to reserve_space
    if constexpr (flush_prefetch) {
        if constexpr (inline_data) {
            TT_ASSERT(data != nullptr);  // compiled out?
            this->add_data(data, data_sizeB, data_sizeB);
            // this->cmd_write_offsetB has been incremented by sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmdLarge) +
            // data_sizeB need to ensure this is aligned for next cmds to be written at the correct location
            this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
        }
    } else {
        // Need to make sure next command that flushes prefetch is written to correctly aligned location
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }
}

template <bool hugepage_write>
template <bool flush_prefetch, bool inline_data>
void DeviceCommand<hugepage_write>::add_dispatch_write_linear_h(
    uint8_t num_mcast_dests,
    uint32_t noc_xy_addr,
    DeviceAddr addr,
    DeviceAddr data_sizeB,
    const void* data,
    uint32_t write_offset_index) {
    uint32_t payload_sizeB = sizeof(CQDispatchCmdLarge) + (flush_prefetch ? data_sizeB : 0);
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_cmd = [&](CQDispatchCmdLarge* write_cmd) {
        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H;
        write_cmd->write_linear.num_mcast_dests = num_mcast_dests;
        write_cmd->write_linear.write_offset_index = write_offset_index;
        write_cmd->write_linear.noc_xy_addr = noc_xy_addr;
        write_cmd->write_linear.addr = addr;
        write_cmd->write_linear.length = data_sizeB;
    };
    CQDispatchCmdLarge* write_cmd_dst = this->reserve_space<CQDispatchCmdLarge*>(sizeof(CQDispatchCmdLarge));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmdLarge write_cmd{};
        initialize_write_cmd(&write_cmd);
        this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmdLarge));
    } else {
        initialize_write_cmd(write_cmd_dst);
    }

    if constexpr (flush_prefetch && inline_data) {
        TT_ASSERT(data != nullptr);
        this->add_data(data, data_sizeB, data_sizeB);
    }

    if constexpr (!flush_prefetch) {
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }
}

// Explicit template instantiations for add_dispatch_write_linear_h
template void DeviceCommand<true>::add_dispatch_write_linear_h<false, false>(
    uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_go_signal_mcast(
    uint32_t wait_count,
    uint32_t go_signal,
    uint32_t wait_stream,
    uint8_t multicast_go_offset,
    uint8_t num_unicast_txns,
    uint8_t noc_data_start_index,
    DispatcherSelect dispatcher_type) {
    TT_ASSERT(
        num_unicast_txns <= std::numeric_limits<uint8_t>::max(),
        "Number of unicast destinations {} exceeds maximum {}",
        num_unicast_txns,
        std::numeric_limits<uint8_t>::max());
    uint32_t lengthB = sizeof(CQDispatchCmd);
    TT_ASSERT(
        lengthB <= (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE),
        "Data for go signal mcast must fit within one page");
    this->add_prefetch_relay_inline(true, lengthB, dispatcher_type);
    auto initialize_mcast_cmd = [&](CQDispatchCmd* mcast_cmd) {
        *mcast_cmd = {};
        mcast_cmd->base.cmd_id = CQ_DISPATCH_CMD_SEND_GO_SIGNAL;
        mcast_cmd->mcast.go_signal = go_signal;
        mcast_cmd->mcast.wait_count = wait_count;
        mcast_cmd->mcast.multicast_go_offset = multicast_go_offset;
        mcast_cmd->mcast.num_unicast_txns = num_unicast_txns;
        mcast_cmd->mcast.noc_data_start_index = noc_data_start_index;
        mcast_cmd->mcast.wait_stream = wait_stream;
    };
    CQDispatchCmd* mcast_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd mcast_cmd{};
        initialize_mcast_cmd(&mcast_cmd);
        this->memcpy(mcast_cmd_dst, &mcast_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_mcast_cmd(mcast_cmd_dst);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_notify_dispatch_s_go_signal_cmd(uint8_t wait, uint16_t index_bitmask) {
    // Command to have dispatch_master send a notification to dispatch_subordinate
    this->add_prefetch_relay_inline(true, sizeof(CQDispatchCmd), DispatcherSelect::DISPATCH_MASTER);
    auto initialize_sem_update_cmd = [&](CQDispatchCmd* sem_update_cmd) {
        *sem_update_cmd = {};
        sem_update_cmd->base.cmd_id = CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL;
        sem_update_cmd->notify_dispatch_s_go_signal.wait = wait;
        sem_update_cmd->notify_dispatch_s_go_signal.index_bitmask = index_bitmask;
    };
    CQDispatchCmd* dispatch_s_sem_update_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));
    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd dispatch_s_sem_update_cmd{};
        initialize_sem_update_cmd(&dispatch_s_sem_update_cmd);
        this->memcpy(dispatch_s_sem_update_dst, &dispatch_s_sem_update_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_sem_update_cmd(dispatch_s_sem_update_dst);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
template <bool inline_data>
void DeviceCommand<hugepage_write>::add_dispatch_write_paged(
    bool flush_prefetch,
    uint8_t is_dram,
    uint16_t start_page,
    uint32_t base_addr,
    uint32_t page_size,
    uint32_t pages,
    const void* data) {
    uint32_t data_sizeB = page_size * pages;
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + (flush_prefetch ? data_sizeB : 0);
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_cmd = [&](CQDispatchCmd* write_cmd) {
        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
        write_cmd->write_paged.is_dram = is_dram;
        write_cmd->write_paged.start_page = start_page;
        write_cmd->write_paged.base_addr = base_addr;
        write_cmd->write_paged.page_size = page_size;
        write_cmd->write_paged.pages = pages;
    };
    CQDispatchCmd* write_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_cmd{};
        initialize_write_cmd(&write_cmd);
        this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_write_cmd(write_cmd_dst);
    }

    if (inline_data) {
        TT_ASSERT(data != nullptr);  // compiled out?
        this->add_data(data, data_sizeB, data_sizeB);
        // Increment wr offset to aligned address only if data is inline. Out-of-line data will
        // follow right after the command, so defer alignment to after it.
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }
}

template <bool hugepage_write>
template <bool inline_data>
void DeviceCommand<hugepage_write>::add_dispatch_write_host(
    bool flush_prefetch, uint64_t data_sizeB, bool is_event, uint16_t pad1, const void* data) {
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + (flush_prefetch ? data_sizeB : 0);
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_cmd = [&](CQDispatchCmd* write_cmd) {
        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
        write_cmd->write_linear_host.is_event = is_event;
        // This padding value is checked on the host side for event commands.
        write_cmd->write_linear_host.pad1 = pad1;
        write_cmd->write_linear_host.pad2 = DeviceCommand::random_padding_value();
        write_cmd->write_linear_host.length =
            sizeof(CQDispatchCmd) +
            data_sizeB;  // CQ_DISPATCH_CMD_WRITE_LINEAR_HOST writes dispatch cmd back to completion queue
    };
    CQDispatchCmd* write_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_cmd{};
        initialize_write_cmd(&write_cmd);
        this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_write_cmd(write_cmd_dst);
    }

    if (inline_data) {
        TT_ASSERT(data != nullptr);  // compiled out?
        this->add_data(data, data_sizeB, data_sizeB);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages) {
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_exec_buf_cmd = [&](CQPrefetchCmd* exec_buf_cmd) {
        exec_buf_cmd->base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF;
        exec_buf_cmd->exec_buf.base_addr = base_addr;
        exec_buf_cmd->exec_buf.log_page_size = log_page_size;
        exec_buf_cmd->exec_buf.pages = pages;
    };
    CQPrefetchCmd* exec_buf_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd exec_buf_cmd{};
        initialize_exec_buf_cmd(&exec_buf_cmd);
        this->memcpy(exec_buf_cmd_dst, &exec_buf_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_exec_buf_cmd(exec_buf_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_set_num_worker_sems(
    const uint32_t num_worker_sems, DispatcherSelect dispatcher_type) {
    this->add_prefetch_relay_inline(true, sizeof(CQDispatchCmd), dispatcher_type);
    auto initialize_set_num_worker_sems_cmd = [&](CQDispatchCmd* set_num_worker_sems_cmd) {
        set_num_worker_sems_cmd->base.cmd_id = CQ_DISPATCH_SET_NUM_WORKER_SEMS;
        set_num_worker_sems_cmd->set_num_worker_sems.num_worker_sems = num_worker_sems;
    };
    CQDispatchCmd* set_num_worker_sems_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));
    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd set_num_worker_sems_cmd{};
        initialize_set_num_worker_sems_cmd(&set_num_worker_sems_cmd);
        this->memcpy(set_num_worker_sems_cmd_dst, &set_num_worker_sems_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_set_num_worker_sems_cmd(set_num_worker_sems_cmd_dst);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_set_go_signal_noc_data(
    const vector_aligned<uint32_t>& noc_mcast_unicast_data, DispatcherSelect dispatcher_type) {
    TT_ASSERT(
        noc_mcast_unicast_data.size() <= DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES,
        "Number of words {} exceeds maximum {}",
        noc_mcast_unicast_data.size(),
        DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES);
    auto data_sizeB = noc_mcast_unicast_data.size() * sizeof(uint32_t);
    uint32_t lengthB = sizeof(CQDispatchCmd) + data_sizeB;
    if (dispatcher_type == DispatcherSelect::DISPATCH_SUBORDINATE) {
        constexpr uint32_t dispatch_page_size = 1 << DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
        TT_FATAL(
            lengthB <= dispatch_page_size,
            "Data to set go signal noc data {} must fit within one dispatch page {} when sending to dispatch_s",
            lengthB,
            dispatch_page_size);
    }
    this->add_prefetch_relay_inline(true, lengthB, dispatcher_type);
    auto initialize_set_go_signal_noc_data_cmd = [&](CQDispatchCmd* set_go_signal_noc_data_cmd) {
        set_go_signal_noc_data_cmd->base.cmd_id = CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA;
        set_go_signal_noc_data_cmd->set_go_signal_noc_data.num_words = noc_mcast_unicast_data.size();
    };
    CQDispatchCmd* set_go_signal_noc_data_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));
    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd set_go_signal_noc_data_cmd{};
        initialize_set_go_signal_noc_data_cmd(&set_go_signal_noc_data_cmd);
        this->memcpy(set_go_signal_noc_data_cmd_dst, &set_go_signal_noc_data_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_set_go_signal_noc_data_cmd(set_go_signal_noc_data_cmd_dst);
    }
    uint32_t* noc_mcast_unicast_data_dst = this->reserve_space<uint32_t*>(data_sizeB);
    if (data_sizeB > 0) {
        this->memcpy(noc_mcast_unicast_data_dst, noc_mcast_unicast_data.data(), data_sizeB);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_set_write_offsets(tt::stl::Span<const uint32_t> write_offsets) {
    TT_ASSERT(write_offsets.size() <= CQ_DISPATCH_MAX_WRITE_OFFSETS);
    size_t data_sizeB = write_offsets.size() * sizeof(uint32_t);
    size_t cmd_size = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(true, cmd_size);
    auto initialize_write_offset_cmd = [&](CQDispatchCmd* write_offset_cmd) {
        *write_offset_cmd = {};
        write_offset_cmd->base.cmd_id = CQ_DISPATCH_CMD_SET_WRITE_OFFSET;
        write_offset_cmd->set_write_offset.offset_count = write_offsets.size();
    };
    CQDispatchCmd* write_offset_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_offset_cmd{};
        initialize_write_offset_cmd(&write_offset_cmd);
        this->memcpy(write_offset_cmd_dst, &write_offset_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_write_offset_cmd(write_offset_cmd_dst);
    }
    uint32_t* write_offsets_dst = this->reserve_space<uint32_t*>(data_sizeB);
    memcpy(write_offsets_dst, write_offsets.data(), data_sizeB);
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_terminate(DispatcherSelect dispatcher_type) {
    this->add_prefetch_relay_inline(true, sizeof(CQDispatchCmd), dispatcher_type);
    auto initialize_terminate_cmd = [&](CQDispatchCmd* terminate_cmd) {
        *terminate_cmd = {};
        terminate_cmd->base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    };
    CQDispatchCmd* terminate_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd terminate_cmd{};
        initialize_terminate_cmd(&terminate_cmd);
        this->memcpy(terminate_cmd_dst, &terminate_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_terminate_cmd(terminate_cmd_dst);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_terminate() {
    uint32_t increment_sizeB = tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    auto initialize_terminate_cmd = [&](CQPrefetchCmd* terminate_cmd) {
        *terminate_cmd = {};
        terminate_cmd->base.cmd_id = CQ_PREFETCH_CMD_TERMINATE;
    };
    CQPrefetchCmd* terminate_cmd_dst = this->reserve_space<CQPrefetchCmd*>(increment_sizeB);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd terminate_cmd{};
        initialize_terminate_cmd(&terminate_cmd);
        this->memcpy(terminate_cmd_dst, &terminate_cmd, sizeof(CQPrefetchCmd));
    } else {
        initialize_terminate_cmd(terminate_cmd_dst);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_exec_buf_end() {
    auto initialize_prefetch_exec_buf_end_cmd = [&](CQPrefetchCmd* exec_buf_end_cmd) {
        // prefetch exec_buf_end behaves as a relay_inline
        exec_buf_end_cmd->base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF_END;
        exec_buf_end_cmd->relay_inline.length = sizeof(CQDispatchCmd);
        exec_buf_end_cmd->relay_inline.stride =
            tt::align(sizeof(CQDispatchCmd) + sizeof(CQPrefetchCmd), this->pcie_alignment);
    };
    auto initialize_dispatch_exec_buf_end_cmd = [&](CQDispatchCmd* exec_buf_end_cmd) {
        exec_buf_end_cmd->base.cmd_id = CQ_DISPATCH_CMD_EXEC_BUF_END;
    };

    CQPrefetchCmd* prefetch_exec_buf_end_cmd_dst = this->reserve_space<CQPrefetchCmd*>(sizeof(CQPrefetchCmd));
    CQDispatchCmd* dispatch_exec_buf_end_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd prefetch_exec_buf_end_cmd{};
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd dispatch_exec_buf_end_cmd{};
        initialize_prefetch_exec_buf_end_cmd(&prefetch_exec_buf_end_cmd);
        initialize_dispatch_exec_buf_end_cmd(&dispatch_exec_buf_end_cmd);
        this->memcpy(prefetch_exec_buf_end_cmd_dst, &prefetch_exec_buf_end_cmd, sizeof(CQPrefetchCmd));
        this->memcpy(dispatch_exec_buf_end_cmd_dst, &dispatch_exec_buf_end_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_prefetch_exec_buf_end_cmd(prefetch_exec_buf_end_cmd_dst);
        initialize_dispatch_exec_buf_end_cmd(dispatch_exec_buf_end_cmd_dst);
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::update_cmd_sequence(
    uint32_t cmd_offsetB, const void* new_data, uint32_t data_sizeB) {
    this->memcpy((char*)this->cmd_region + cmd_offsetB, new_data, data_sizeB);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_data(
    const void* data, uint32_t data_size_to_copyB, uint32_t cmd_write_offset_incrementB) {
    this->validate_cmd_write(cmd_write_offset_incrementB);
    this->memcpy((uint8_t*)this->cmd_region + this->cmd_write_offsetB, data, data_size_to_copyB);
    this->cmd_write_offsetB += cmd_write_offset_incrementB;
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::align_write_offset() {
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
template <typename PackedSubCmd>
void DeviceCommand<hugepage_write>::add_dispatch_write_packed(
    uint8_t type,
    uint16_t num_sub_cmds,
    uint32_t common_addr,
    uint16_t packed_data_sizeB,
    uint32_t payload_sizeB,
    const std::vector<PackedSubCmd>& sub_cmds,
    const std::vector<std::pair<const void*, uint32_t>>& data_collection,
    uint32_t packed_write_max_unicast_sub_cmds,
    const uint32_t offset_idx,
    const bool no_stride,
    uint32_t write_offset_index) {
    static_assert(
        std::is_same_v<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd> or
        std::is_same_v<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>);
    bool multicast = std::is_same_v<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>;

    uint32_t packed_write_max_multicast_sub_cmds =
        get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);
    uint32_t max_num_packed_sub_cmds = std::is_same_v<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>
                                           ? packed_write_max_unicast_sub_cmds
                                           : packed_write_max_multicast_sub_cmds;
    TT_FATAL(
        num_sub_cmds <= max_num_packed_sub_cmds,
        "Max number of packed sub commands are {} but requesting {}",
        max_num_packed_sub_cmds,
        num_sub_cmds);

    TT_ASSERT((type & ~CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_MASK) == 0, "Invalid type {}", type);

    constexpr bool flush_prefetch = true;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_packed_cmd = [&](CQDispatchCmd* write_packed_cmd) {
        write_packed_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
        write_packed_cmd->write_packed.flags =
            type | (multicast ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE) |
            (no_stride ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE);
        write_packed_cmd->write_packed.count = num_sub_cmds;
        write_packed_cmd->write_packed.write_offset_index = write_offset_index;
        write_packed_cmd->write_packed.addr = common_addr;
        write_packed_cmd->write_packed.size = packed_data_sizeB;
    };
    CQDispatchCmd* write_packed_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_packed_cmd{};
        initialize_write_packed_cmd(&write_packed_cmd);
        this->memcpy(write_packed_cmd_dst, &write_packed_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_write_packed_cmd(write_packed_cmd_dst);
    }

    static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(PackedSubCmd);
    this->memcpy((char*)this->cmd_region + this->cmd_write_offsetB, &sub_cmds[offset_idx], sub_cmds_sizeB);

    uint32_t increment_sizeB =
        tt::align(sub_cmds_sizeB, this->l1_alignment);  // this assumes CQDispatchCmd is L1 aligned
    this->cmd_write_offsetB += increment_sizeB;

    // copy the actual data
    increment_sizeB = tt::align(packed_data_sizeB, this->l1_alignment);
    uint32_t num_data_copies = no_stride ? 1 : num_sub_cmds;
    for (uint32_t i = offset_idx; i < offset_idx + num_data_copies; ++i) {
        if (data_collection[i].first) {
            this->memcpy(
                (char*)this->cmd_region + this->cmd_write_offsetB, data_collection[i].first, data_collection[i].second);
        }
        // Always advanced by the packed stride, even if no data was copied
        this->cmd_write_offsetB += increment_sizeB;
    }

    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

// Tuple in data_collection is:
//  0:address, 1:size, 2:stride
template <bool hugepage_write>
template <typename PackedSubCmd>
void DeviceCommand<hugepage_write>::add_dispatch_write_packed(
    uint8_t type,
    uint16_t num_sub_cmds,
    uint32_t common_addr,
    uint16_t packed_data_sizeB,
    uint32_t payload_sizeB,
    const std::vector<PackedSubCmd>& sub_cmds,
    const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>& data_collection,
    uint32_t packed_write_max_unicast_sub_cmds,
    const uint32_t offset_idx,
    const bool no_stride,
    uint32_t write_offset_index) {
    static_assert(
        std::is_same_v<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd> or
        std::is_same_v<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>);
    bool multicast = std::is_same_v<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>;

    uint32_t packed_write_max_multicast_sub_cmds =
        get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);
    uint32_t max_num_packed_sub_cmds = std::is_same_v<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>
                                           ? packed_write_max_unicast_sub_cmds
                                           : packed_write_max_multicast_sub_cmds;
    TT_ASSERT(
        num_sub_cmds <= max_num_packed_sub_cmds,
        "Max number of packed sub commands are {} but requesting {}",
        max_num_packed_sub_cmds,
        num_sub_cmds);
    TT_ASSERT((type & ~CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_MASK) == 0, "Invalid type {}", type);

    constexpr bool flush_prefetch = true;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_packed_cmd = [&](CQDispatchCmd* write_packed_cmd) {
        write_packed_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
        write_packed_cmd->write_packed.flags =
            type | (multicast ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE) |
            (no_stride ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE);
        write_packed_cmd->write_packed.count = num_sub_cmds;
        write_packed_cmd->write_packed.write_offset_index = write_offset_index;
        write_packed_cmd->write_packed.addr = common_addr;
        write_packed_cmd->write_packed.size = packed_data_sizeB;
    };
    CQDispatchCmd* write_packed_cmd_dst = this->reserve_space<CQDispatchCmd*>(sizeof(CQDispatchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_packed_cmd{};
        initialize_write_packed_cmd(&write_packed_cmd);
        this->memcpy(write_packed_cmd_dst, &write_packed_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_write_packed_cmd(write_packed_cmd_dst);
    }

    static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(PackedSubCmd);
    this->memcpy((char*)this->cmd_region + this->cmd_write_offsetB, &sub_cmds[offset_idx], sub_cmds_sizeB);

    uint32_t increment_sizeB =
        tt::align(sub_cmds_sizeB, this->l1_alignment);  // this assumes CQDispatchCmd is L1 aligned
    this->cmd_write_offsetB += increment_sizeB;

    // copy the actual data
    increment_sizeB = tt::align(packed_data_sizeB, this->l1_alignment);
    uint32_t num_data_copies = no_stride ? 1 : num_sub_cmds;
    for (uint32_t i = offset_idx; i < offset_idx + num_data_copies; ++i) {
        uint32_t offset = 0;
        for (const auto& data : data_collection[i]) {
            if (std::get<0>(data)) {
                this->memcpy(
                    (char*)this->cmd_region + this->cmd_write_offsetB + offset, std::get<0>(data), std::get<1>(data));
            }
            // Always advanced by the stride, even if no data was copied
            offset += std::get<2>(data);
        }
        this->cmd_write_offsetB += increment_sizeB;
    }

    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

// Add write packed large, with no data.
template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_write_packed_large(
    uint8_t type,
    uint16_t alignment,
    uint16_t num_sub_cmds,
    const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
    const uint32_t offset_idx,
    uint32_t write_offset_index) {
    constexpr bool flush_prefetch = false;
    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQDispatchWritePackedLargeSubCmd);
    uint32_t payload_size = tt::align(sizeof(CQDispatchCmd) + sub_cmds_sizeB, this->l1_alignment);
    this->add_dispatch_write_packed_large_internal(
        type, flush_prefetch, alignment, payload_size, num_sub_cmds, sub_cmds, offset_idx, write_offset_index);
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

// Add write packed large, with data inlined.
template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_write_packed_large(
    uint8_t type,
    uint16_t alignment,
    uint16_t num_sub_cmds,
    const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
    const std::vector<tt::stl::Span<const uint8_t>>& data_collection,
    std::vector<uint8_t*>*
        data_collection_buffer_ptr,  // optional. Stores the location each data segment was written to
    const uint32_t offset_idx,
    uint32_t write_offset_index) {
    constexpr bool flush_prefetch = true;
    size_t data_collection_size = 0;
    for (const auto& data : data_collection) {
        data_collection_size += data.size();
    }
    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQDispatchWritePackedLargeSubCmd);
    uint32_t payload_size = tt::align(
        tt::align(sizeof(CQDispatchCmd) + sub_cmds_sizeB, this->l1_alignment) + data_collection_size,
        this->l1_alignment);
    this->add_dispatch_write_packed_large_internal(
        type, flush_prefetch, alignment, payload_size, num_sub_cmds, sub_cmds, offset_idx, write_offset_index);

    if (data_collection_buffer_ptr != nullptr) {
        data_collection_buffer_ptr->resize(data_collection.size());
    }
    for (size_t i = 0; i < data_collection.size(); i++) {
        if (data_collection_buffer_ptr) {
            data_collection_buffer_ptr->at(i) = (uint8_t*)this->cmd_region + this->cmd_write_offsetB;
        }
        this->add_data(data_collection[i].data(), data_collection[i].size(), data_collection[i].size());
    }
    this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_prefetch_relay_inline(
    bool flush, uint32_t lengthB, DispatcherSelect dispatcher_type) {
    if (!flush) {
        uint32_t dispatch_page_size = 1
                                      << (dispatcher_type == DispatcherSelect::DISPATCH_MASTER
                                              ? DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE
                                              : DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
        TT_ASSERT(
            lengthB <= dispatch_page_size,
            "Data to relay for inline no flush {} must fit within one dispatch page {}",
            lengthB,
            dispatch_page_size);
    }
    auto initialize_relay_write = [&](CQPrefetchCmd* relay_write) {
        relay_write->base.cmd_id = flush ? CQ_PREFETCH_CMD_RELAY_INLINE : CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
        relay_write->relay_inline.dispatcher_type = (uint8_t)(dispatcher_type);
        relay_write->relay_inline.length = lengthB;
        relay_write->relay_inline.stride = tt::align(sizeof(CQPrefetchCmd) + lengthB, this->pcie_alignment);
    };
    CQPrefetchCmd* relay_write_dst = this->reserve_space<CQPrefetchCmd*>(sizeof(CQPrefetchCmd));

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_write{};
        initialize_relay_write(&relay_write);
        this->memcpy(relay_write_dst, &relay_write, sizeof(CQPrefetchCmd));
    } else {
        initialize_relay_write(relay_write_dst);
    }
}

// Write packed large cmd and subcmds, but not data.
template <bool hugepage_write>
void DeviceCommand<hugepage_write>::add_dispatch_write_packed_large_internal(
    uint8_t type,
    bool flush_prefetch,
    uint16_t alignment,
    uint32_t payload_sizeB,
    uint16_t num_sub_cmds,
    const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
    const uint32_t offset_idx,
    uint32_t write_offset_index) {
    TT_ASSERT(
        num_sub_cmds <= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS,
        "Cannot fit {} sub cmds in one CQDispatchWritePackedLargeCmd",
        num_sub_cmds);
    static_assert(sizeof(CQDispatchWritePackedLargeSubCmd) % sizeof(uint32_t) == 0);
    uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQDispatchWritePackedLargeSubCmd);
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    auto initialize_write_packed_large_cmd = [&](CQDispatchCmd* write_packed_large_cmd) {
        write_packed_large_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED_LARGE;
        write_packed_large_cmd->write_packed_large.type = type;
        write_packed_large_cmd->write_packed_large.count = num_sub_cmds;
        write_packed_large_cmd->write_packed_large.alignment = alignment;
        write_packed_large_cmd->write_packed_large.write_offset_index = write_offset_index;
    };
    uint32_t sub_cmd_size = tt::align(sizeof(CQDispatchCmd) + sub_cmds_sizeB, this->l1_alignment);
    CQDispatchCmd* write_packed_large_cmd_dst = this->reserve_space<CQDispatchCmd*>(sub_cmd_size);
    char* write_packed_large_sub_cmds_dst = (char*)write_packed_large_cmd_dst + sizeof(CQDispatchCmd);

    if constexpr (hugepage_write) {
        alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_packed_large_cmd{};
        initialize_write_packed_large_cmd(&write_packed_large_cmd);
        this->memcpy(write_packed_large_cmd_dst, &write_packed_large_cmd, sizeof(CQDispatchCmd));
    } else {
        initialize_write_packed_large_cmd(write_packed_large_cmd_dst);
    }

    this->memcpy(write_packed_large_sub_cmds_dst, &sub_cmds[offset_idx], sub_cmds_sizeB);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::validate_cmd_write(uint32_t data_sizeB) const {
    uint32_t data_endB = this->cmd_write_offsetB + data_sizeB;
    TT_ASSERT(
        data_endB <= this->cmd_sequence_sizeB,
        "Out of bounds command sequence write: attemping to write {} B but only {} B available",
        data_sizeB,
        this->cmd_sequence_sizeB - this->cmd_write_offsetB);
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::deepcopy(const DeviceCommand& other) {
    if (other.cmd_region_vector.empty() and other.cmd_region != nullptr) {
        this->cmd_region = other.cmd_region;
    } else if (not other.cmd_region_vector.empty()) {
        TT_ASSERT(other.cmd_region != nullptr);
        this->cmd_region = this->cmd_region_vector.data();
        this->memcpy(this->cmd_region, other.cmd_region_vector.data(), this->cmd_sequence_sizeB);
    }
}

template <bool hugepage_write>
void DeviceCommand<hugepage_write>::memcpy(void* __restrict dst, const void* __restrict src, size_t n) {
    if constexpr (hugepage_write) {
        memcpy_to_device(dst, src, n);
    } else {
        std::memcpy(dst, src, n);
    }
}

namespace {
uint32_t random_padding_value_cached = []() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1u, 0xFFFFFFFFu);
    return dist(gen);
}();
}  // namespace

template <bool hugepage_write>
uint32_t DeviceCommand<hugepage_write>::random_padding_value() {
    return random_padding_value_cached;
}

// clang-format off
template class DeviceCommand<true>;
template class DeviceCommand<false>;

template DeviceCommand<false>::DeviceCommand(uint32_t);

template void DeviceCommand<true>::add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedUnicastSubCmd>&, const std::vector<std::pair<const void*, uint32_t>>&, uint32_t, const uint32_t, const bool, uint32_t);
template void DeviceCommand<true>::add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedMulticastSubCmd>&, const std::vector<std::pair<const void*, uint32_t>>&, uint32_t, const uint32_t, const bool, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedUnicastSubCmd>&, const std::vector<std::pair<const void*, uint32_t>>&, uint32_t, const uint32_t, const bool, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedMulticastSubCmd>&, const std::vector<std::pair<const void*, uint32_t>>&, uint32_t, const uint32_t, const bool, uint32_t);

template void DeviceCommand<true>::add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedUnicastSubCmd>&, const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>&, uint32_t, const uint32_t, const bool, uint32_t);
template void DeviceCommand<true>::add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedMulticastSubCmd>&, const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>&, uint32_t, const uint32_t, const bool, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedUnicastSubCmd>&, const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>&, uint32_t, const uint32_t, const bool, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(uint8_t, uint16_t, uint32_t, uint16_t, uint32_t, const std::vector<CQDispatchWritePackedMulticastSubCmd>&, const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>&, uint32_t, const uint32_t, const bool, uint32_t);

template void DeviceCommand<true>::add_dispatch_write_host<false>(bool, uint64_t, bool, uint16_t, const void*);
template void DeviceCommand<true>::add_dispatch_write_host<true>(bool, uint64_t, bool, uint16_t, const void*);
template void DeviceCommand<false>::add_dispatch_write_host<false>(bool, uint64_t, bool, uint16_t, const void*);
template void DeviceCommand<false>::add_dispatch_write_host<true>(bool, uint64_t, bool, uint16_t, const void*);

template void DeviceCommand<true>::add_dispatch_write_paged<false>(bool, uint8_t, uint16_t, uint32_t, uint32_t, uint32_t, const void*);
template void DeviceCommand<true>::add_dispatch_write_paged<true>(bool, uint8_t, uint16_t, uint32_t, uint32_t, uint32_t, const void*);
template void DeviceCommand<false>::add_dispatch_write_paged<false>(bool, uint8_t, uint16_t, uint32_t, uint32_t, uint32_t, const void*);
template void DeviceCommand<false>::add_dispatch_write_paged<true>(bool, uint8_t, uint16_t, uint32_t, uint32_t, uint32_t, const void*);

template void DeviceCommand<true>::add_dispatch_write_linear<true, false>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<true>::add_dispatch_write_linear<true, true>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<true>::add_dispatch_write_linear<false, false>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<true>::add_dispatch_write_linear<false, true>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_linear<true, false>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_linear<true, true>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_linear<false, false>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
template void DeviceCommand<false>::add_dispatch_write_linear<false, true>(uint8_t, uint32_t, DeviceAddr, DeviceAddr, const void*, uint32_t);
// clang-format on
}  // namespace tt::tt_metal
