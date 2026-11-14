// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/device_types.hpp>    // for tt::ChipId
#include <tt-umd/io_window/io_window.hpp>  // for tt::umd::IoWindow
#include <tt-umd/types/xy_pair.hpp>        // for tt_cxy_pair

#include "aux_region_allocator.hpp"
#include "system_memory_cq_interface.hpp"

namespace tt::tt_metal {

class MetalContext;
class MetalEnvImpl;

// Where a device's command queues live and how host-side changes reach the device. SystemMemoryManager owns the ring
// arithmetic for each CQ and calls into one backing, chosen once at construction:
// - HugepageCqBacking: the CQs are in a PCIe hugepage the device reads directly.
// - DramCqBacking: the CQs are in device DRAM; the host edits a staging copy and syncs the changed ranges. For
//   simulators and other setups without a PCIe hugepage.
// - MockCqBacking: mock and emulated devices; a host buffer, so the ring code runs unchanged, and no device.
class CqBacking {
public:
    static std::unique_ptr<CqBacking> create(
        MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs);

    virtual ~CqBacking() = default;
    CqBacking(const CqBacking&) = delete;
    CqBacking& operator=(const CqBacking&) = delete;
    CqBacking(CqBacking&&) = delete;
    CqBacking& operator=(CqBacking&&) = delete;

    // DRAM bank that holds DRAM-backed CQs.
    static constexpr uint32_t kDramBankId = 0;

    MetalEnvImpl& env() const { return env_; }

    MetalContext& context() const { return ctx_; }

    ChipId device_id() const { return device_id_; }

    // Bytes per CQ, after the aux region reservation.
    uint32_t cq_size() const { return cq_size_; }

    uint32_t issue_alignment() const { return issue_alignment_; }

    bool is_dram_backed() const { return is_dram_backed_; }

    AuxRegionAllocator& aux_region_allocator() { return aux_region_allocator_; }

    // Host pointer for a CQ address, the address the device uses for the same byte.
    char* host_ptr(uint32_t cq_addr) const { return host_base_ + (cq_addr - cq_addr_base_); }

    // CQ address where cq_id's region starts, for the device on the given host channel.
    uint32_t cq_base_addr(uint8_t cq_id, uint16_t channel) const;

    // Reads size_B bytes of this device's CQ memory, as the device last wrote it, at a CQ address.
    virtual void read_cq_memory(void* dst, uint32_t size_B, uint32_t cq_addr) const = 0;

    // Writes the CQ memory of device_id, serviced by this device on the given host channel. The hugepage backing
    // addresses it by channel, the DRAM backing by device_id; each ignores the other.
    virtual void write_cq_memory(
        ChipId device_id, uint16_t channel, const void* src, uint32_t size_B, uint32_t cq_addr) const = 0;

    // Reads a host-side CQ pointer (a CommandQueueHostAddrType address) of cq_id. Pointers the device writes sit past
    // its channel offset in a shared hugepage; pointers the host writes don't.
    virtual uint32_t read_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, bool device_written) const = 0;

    // Builds the per-CQ ring state the manager starts from.
    virtual std::vector<SystemMemoryCQInterface> make_cq_interfaces(uint8_t num_hw_cqs) const = 0;

    // dst comes from host_ptr().
    virtual void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const = 0;

    // Called after the host pushed size_B bytes starting at start_addr; wr_ptr_16B is the new issue write pointer.
    virtual void issue_queue_pushed(uint8_t cq_id, uint32_t start_addr, uint32_t size_B, uint32_t wr_ptr_16B) = 0;

    // Makes the completion queue of cq_id readable through host_ptr() starting at its base address.
    virtual void sync_completion_queue(uint8_t cq_id, const SystemMemoryCQInterface& cq_interface) const = 0;

    // Tells the device how far the host has read the completion queue.
    virtual void send_completion_queue_read_ptr(uint8_t cq_id, uint32_t read_ptr_and_toggle) const = 0;

    // Waits for room for one fetch queue entry.
    virtual void fetch_queue_reserve_back(uint8_t cq_id) = 0;

    // Publishes command_size_B bytes of the issue queue to the prefetcher.
    virtual void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher) = 0;

    // Waits until the device has written past the host's read position; returns the write pointer and toggle.
    virtual uint32_t completion_queue_wait_front(
        uint8_t cq_id, const SystemMemoryCQInterface& cq_interface, std::atomic<bool>& exit_condition) const = 0;

protected:
    CqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id) : env_(env), ctx_(ctx), device_id_(device_id) {}

    // Where the CQs are and how they are addressed; each backing's constructor sets it once.
    struct Layout {
        // Host pointer of CQ address cq_addr_base.
        char* host_base = nullptr;
        uint32_t cq_addr_base = 0;
        uint32_t cq_size = 0;
        uint32_t issue_alignment = 0;
        bool is_dram_backed = false;
        AuxRegionAllocator aux_region_allocator;
    };

    void set_layout(Layout layout);

    MetalEnvImpl& env_;
    MetalContext& ctx_;
    const ChipId device_id_;

private:
    char* host_base_ = nullptr;
    uint32_t cq_addr_base_ = 0;
    uint32_t cq_size_ = 0;
    uint32_t issue_alignment_ = 0;
    bool is_dram_backed_ = false;
    AuxRegionAllocator aux_region_allocator_;
};

}  // namespace tt::tt_metal
