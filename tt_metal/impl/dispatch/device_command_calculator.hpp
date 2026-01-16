// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_align.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"

namespace tt::tt_metal {
class DeviceCommandCalculator {
public:
    uint32_t write_offset_bytes() const { return this->cmd_write_offsetB; }

    void add_dispatch_wait() {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_wait_with_prefetch_stall() {
        this->add_dispatch_wait();
        this->cmd_write_offsetB += sizeof(CQPrefetchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }
    void add_prefetch_relay_linear() {
        this->cmd_write_offsetB += sizeof(CQPrefetchCmdLarge);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_prefetch_stall() {
        this->cmd_write_offsetB += sizeof(CQPrefetchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    template <bool pcie_aligned = true>
    void add_data(uint32_t cmd_write_offset_incrementB) {
        this->cmd_write_offsetB += cmd_write_offset_incrementB;
        if constexpr (not pcie_aligned) {
            this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
        }
    }

    void add_alignment() { this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment); }

    template <bool flush_prefetch = true, bool inline_data = false>
    void add_dispatch_write_linear(uint32_t data_sizeB) {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmdLarge);

        if constexpr (flush_prefetch) {
            if constexpr (inline_data) {
                this->add_data(data_sizeB);
                // this->cmd_write_offsetB has been incremented by sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) +
                // data_sizeB need to ensure this is aligned for next cmds to be written at the correct location
                this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
            }
        } else {
            // Need to make sure next command that flushes prefetch is written to correctly aligned location
            this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
        }
    }

    // Calculator sizing for CQ_DISPATCH_CMD_WRITE_LINEAR_H (dispatch_h linear write)
    // Mirrors add_dispatch_write_linear for sizing/alignment purposes.
    template <bool flush_prefetch = true, bool inline_data = false>
    void add_dispatch_write_linear_h(uint32_t data_sizeB) {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmdLarge);

        if constexpr (flush_prefetch) {
            if constexpr (inline_data) {
                this->add_data(data_sizeB);
                this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
            }
        } else {
            this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
        }
    }

    void add_dispatch_write_linear_host_event(uint32_t data_sizeB) {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd) + data_sizeB;
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_write_linear_host() {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_go_signal_mcast() {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_notify_dispatch_s_go_signal_cmd() {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_set_num_worker_sems() {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_set_go_signal_noc_data(uint32_t num_words) {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd) + num_words * sizeof(uint32_t);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_set_write_offsets(uint32_t num_offsets) {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd) + num_offsets * sizeof(uint32_t);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_terminate() {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_prefetch_terminate() { this->cmd_write_offsetB += tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment); }

    template <bool inline_data = false>
    void add_dispatch_write_paged(uint32_t page_size, uint32_t pages) {
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);
        if constexpr (inline_data) {
            uint32_t data_sizeB = page_size * pages;
            this->add_data(data_sizeB);
            this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
        }
    }

    void add_prefetch_relay_paged() {
        this->cmd_write_offsetB += tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    }

    void add_prefetch_relay_paged_packed(uint16_t num_sub_cmds) {
        static_assert(sizeof(CQPrefetchRelayPagedPackedSubCmd) % sizeof(uint32_t) == 0);

        uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQPrefetchRelayPagedPackedSubCmd);
        uint32_t increment_sizeB = tt::align(sub_cmds_sizeB + sizeof(CQPrefetchCmd), this->pcie_alignment);
        this->cmd_write_offsetB += increment_sizeB;
    }

    void add_prefetch_relay_ringbuffer(uint16_t num_sub_cmds) {
        static_assert(sizeof(CQPrefetchRelayRingbufferSubCmd) % sizeof(uint32_t) == 0);

        uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQPrefetchRelayRingbufferSubCmd);
        uint32_t increment_sizeB = tt::align(sub_cmds_sizeB + sizeof(CQPrefetchCmd), this->pcie_alignment);
        this->cmd_write_offsetB += increment_sizeB;
    }

    void add_prefetch_set_ringbuffer_offset() {
        this->cmd_write_offsetB += tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    }

    void add_prefetch_paged_to_ringbuffer() {
        this->cmd_write_offsetB += tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment);
    }

    void add_prefetch_exec_buf() { this->cmd_write_offsetB += tt::align(sizeof(CQPrefetchCmd), this->pcie_alignment); }

    void add_prefetch_exec_buf_end() {
        this->cmd_write_offsetB += tt::align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), this->pcie_alignment);
    }

    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint16_t num_sub_cmds,
        uint16_t packed_data_sizeB,
        uint32_t packed_write_max_unicast_sub_cmds,
        const bool no_stride = false) {
        static_assert(
            std::is_same_v<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd> or
            std::is_same_v<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>);

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

        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += sizeof(CQDispatchCmd);

        static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(PackedSubCmd);

        uint32_t increment_sizeB =
            tt::align(sub_cmds_sizeB, this->l1_alignment);  // this assumes CQDispatchCmd is L1 aligned
        this->cmd_write_offsetB += increment_sizeB;

        // copy the actual data
        increment_sizeB = tt::align(packed_data_sizeB, this->l1_alignment);
        uint32_t num_data_copies = no_stride ? 1 : num_sub_cmds;
        this->cmd_write_offsetB += num_data_copies * increment_sizeB;

        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_write_packed_large(uint16_t num_sub_cmds) {
        TT_ASSERT(
            num_sub_cmds <= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS,
            "Cannot fit {} sub cmds in one CQDispatchWritePackedLargeCmd",
            num_sub_cmds);
        static_assert(sizeof(CQDispatchWritePackedLargeSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQDispatchWritePackedLargeSubCmd);
        uint32_t payload_size = tt::align(sizeof(CQDispatchCmd) + sub_cmds_sizeB, this->l1_alignment);
        this->add_prefetch_relay_inline();
        uint32_t payload_dst_size =
            tt::align(sizeof(CQPrefetchCmd) + payload_size, this->pcie_alignment) - sizeof(CQPrefetchCmd);
        this->cmd_write_offsetB += payload_dst_size;
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    void add_dispatch_write_packed_large(uint16_t num_sub_cmds, uint32_t payload_sizeB) {
        TT_ASSERT(
            num_sub_cmds <= CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS,
            "Cannot fit {} sub cmds in one CQDispatchWritePackedLargeCmd",
            num_sub_cmds);
        static_assert(sizeof(CQDispatchWritePackedLargeSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(CQDispatchWritePackedLargeSubCmd);
        uint32_t payload_size = tt::align(sizeof(CQDispatchCmd) + sub_cmds_sizeB, this->l1_alignment);
        this->add_prefetch_relay_inline();
        this->cmd_write_offsetB += payload_size;
        this->cmd_write_offsetB += tt::align(payload_sizeB, this->l1_alignment);
        this->cmd_write_offsetB = tt::align(this->cmd_write_offsetB, this->pcie_alignment);
    }

    template <typename PackedSubCmd>
    uint32_t get_max_write_packed_sub_cmds(
        uint32_t data_size,
        uint32_t max_prefetch_cmd_size,
        uint32_t packed_write_max_unicast_sub_cmds,
        bool no_stride) const;

    // Divide the sub commands into multiple dispatch commands if the number of sub commands exceeds the maximum number
    // of sub commands that can be written in a single dispatch command.
    template <typename PackedSubCmd>
    void insert_write_packed_payloads(
        uint32_t num_sub_cmds,
        uint32_t sub_cmd_sizeB,
        uint32_t max_prefetch_command_size,
        uint32_t packed_write_max_unicast_sub_cmds,
        std::vector<std::pair<uint32_t, uint32_t>>& packed_cmd_payloads);

    // Clear calculator state
    void clear() { this->cmd_write_offsetB = 0; }

    // Update state
    void update_write_offset_bytes(uint32_t update_valueB) { this->cmd_write_offsetB += update_valueB; }

private:
    void add_prefetch_relay_inline() { this->cmd_write_offsetB += sizeof(CQPrefetchCmd); }
    uint32_t cmd_write_offsetB = 0;
    uint32_t pcie_alignment =
        tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST);
    uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
};
}  // namespace tt::tt_metal
