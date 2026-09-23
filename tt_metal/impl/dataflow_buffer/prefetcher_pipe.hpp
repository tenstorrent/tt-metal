// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

namespace tt::tt_metal {

class DriscL1Allocation;
class IDevice;

namespace distributed {
class MeshDevice;
}

namespace experimental {

// Byte layout of one config page in a space. Every page in a space is sized for the space's
// max_receivers_per_pipe x credit_lane_capacity; a pipe with fewer receivers writes its real count
// into word[1] and leaves the tail unused (device init reads the block offsets from the page, so a
// max-sized page serves any smaller pipe). See remote_dfb_config_layout.h.
struct PrefetcherPipeConfigPageLayout {
    uint32_t noc_xy_offset = 0;
    uint32_t sent_offset = 0;   // SENT block base: [sent | wr_cursor] slot per (receiver, lane)
    uint32_t acked_offset = 0;  // ACKED block base: [acked] slot per (receiver, lane)
    uint32_t page_size = 0;
};

// Implementation of PrefetcherPipeSpace: the persistent L1 reservation (one ring allocation and
// one config-page allocation over sender_cores ∪ receiver_domain), the claim table that keeps
// live pipes on disjoint cores, and the page writer. Owned by exactly one PrefetcherPipeSpace
// handle. Pipes carved from it point back at it without owning it: the space must outlive its
// pipes. If it does not, the destructor detaches every live pipe (so a later use of that pipe
// fails with a clear error rather than dereferencing freed memory) and logs the violation.
class PrefetcherPipeSpaceImpl {
public:
    PrefetcherPipeSpaceImpl(const distributed::MeshDevice& device, PrefetcherPipeSpaceConfig config);

    PrefetcherPipeSpaceImpl(const PrefetcherPipeSpaceImpl&) = delete;
    PrefetcherPipeSpaceImpl& operator=(const PrefetcherPipeSpaceImpl&) = delete;
    PrefetcherPipeSpaceImpl(PrefetcherPipeSpaceImpl&&) = delete;
    PrefetcherPipeSpaceImpl& operator=(PrefetcherPipeSpaceImpl&&) = delete;
    ~PrefetcherPipeSpaceImpl();

    uint32_t buffer_address() const { return data_address_; }
    uint32_t config_address() const { return config_address_; }
    uint32_t ring_size() const { return config_.ring_size; }
    uint32_t config_page_size() const { return layout_.page_size; }
    uint32_t max_receivers_per_pipe() const { return config_.max_receivers_per_pipe; }
    // Credit lane slots allocated per receiver in every page (Quasar reserves headroom).
    uint32_t credit_lane_capacity() const { return credit_lane_capacity_; }
    const PrefetcherPipeConfigPageLayout& layout() const { return layout_; }

    const CoreRangeSet& sender_cores() const { return config_.sender_cores; }
    uint32_t num_dram_senders() const { return config_.num_dram_senders; }
    const CoreRangeSet& receiver_domain() const { return config_.receiver_domain; }
    const CoreRangeSet& reservation_cores() const { return reservation_cores_; }
    CoreRangeSet unclaimed_cores() const;
    bool is_claimed(const CoreCoord& core) const { return claimed_.contains(core); }
    const distributed::MeshDevice* get_device() const { return device_; }

    PrefetcherPipe create_pipe(CoreCoord sender, const CoreRangeSet& receivers);
    std::vector<PrefetcherPipe> create_pipes(std::span<const std::pair<CoreCoord, CoreRangeSet>> pipes);
    void set_dram_sender_cores(std::span<const CoreCoord> dram_senders);
    void validate_dram_carves(std::span<const std::pair<CoreCoord, CoreRangeSet>> pipes) const;
    PrefetcherPipe create_dram_sender_pipe(
        CoreCoord dram_sender,
        const CoreRangeSet& receivers,
        uint32_t recv_index_base = 0,
        uint64_t tensor_prefetcher_factory_id = 0,
        uint32_t tensor_prefetcher_factory_num_pipes = 0);

    // Used by PrefetcherPipeImpl. `pending` holds cores claimed earlier in the same create_pipes
    // batch (validation runs before any claim is taken).
    void validate_carve(
        CoreCoord sender, const CoreRangeSet& receivers, const std::unordered_set<CoreCoord>* pending) const;
    void validate_dram_carve(CoreCoord sender, const CoreRangeSet& receivers) const;
    // `owner` is the pipe taking / releasing the claim; the space tracks it so it can detach the
    // pipe if the space is destroyed first.
    void claim(const CoreRangeSet& cores, PrefetcherPipeImpl& owner);
    void unclaim(const CoreRangeSet& cores, PrefetcherPipeImpl& owner) noexcept;

    // Write `page` at config_address() on every core in `cores`: one NOC multicast per rectangle,
    // unicast for single-core rectangles.
    void write_page(const CoreRangeSet& cores, const std::vector<uint32_t>& page) const;
    // Write per-core pages, grouping cores with identical bytes into multicast rectangles.
    void write_pages(const std::unordered_map<CoreCoord, std::vector<uint32_t>>& pages) const;

private:
    friend class PrefetcherPipeImpl;
    // Shared by validate_dram_carve and validate_dram_carves: everything a DRAM carve must satisfy
    // except that its sender is reserved, which the two forms check at different points.
    void validate_dram_pipe_geometry(CoreCoord sender, const CoreRangeSet& receivers) const;
    void setup_reservation();
    void release_allocations() noexcept;

    const distributed::MeshDevice* device_ = nullptr;
    PrefetcherPipeSpaceConfig config_;
    CoreRangeSet reservation_cores_;
    uint32_t credit_lane_capacity_ = 1;
    PrefetcherPipeConfigPageLayout layout_;
    uint64_t data_allocation_id_ = 0;
    uint64_t config_allocation_id_ = 0;
    uint32_t data_address_ = 0;
    uint32_t config_address_ = 0;
    std::unordered_set<CoreCoord> claimed_;
    std::unordered_set<PrefetcherPipeImpl*> live_pipes_;
    std::unordered_set<CoreCoord> claimed_dram_senders_;
    std::unordered_map<CoreCoord, std::shared_ptr<DriscL1Allocation>> dram_sender_allocations_;
};

// Implementation of the PrefetcherPipe host object declared in
// tt-metalium/experimental/prefetcher_pipe.hpp: one sender -> receivers mapping carved from a
// space. It claims its cores in the space, composes each core's config page and stamps those
// pages onto the device; addresses come from the space. A PrefetcherPipe holds one of these and
// forwards to it, and the host runtime records a binding by pointing at one. That pointer is only
// as good as the handle: destroying the PrefetcherPipe destroys this object and releases the claim,
// which is why the handle must outlive every Program bound to it. Moving a handle hands this
// object to the new owner without relocating it, so destruction is the only way to get there.
//
// The space pointer is non-owning (the PrefetcherPipeSpace handle owns the space). Every
// accessor that needs the space goes through space(), which rejects a pipe whose space has
// already been destroyed.
class PrefetcherPipeImpl {
public:
    PrefetcherPipeImpl(PrefetcherPipeSpaceImpl& space, CoreCoord sender_core, const CoreRangeSet& receiver_cores);
    PrefetcherPipeImpl(
        PrefetcherPipeSpaceImpl& space,
        CoreCoord dram_sender,
        const CoreRangeSet& receiver_cores,
        std::shared_ptr<DriscL1Allocation> drisc_config_page,
        uint32_t recv_index_base,
        uint64_t tensor_prefetcher_factory_id,
        uint32_t tensor_prefetcher_factory_num_pipes);

    PrefetcherPipeImpl(const PrefetcherPipeImpl&) = delete;
    PrefetcherPipeImpl& operator=(const PrefetcherPipeImpl&) = delete;
    PrefetcherPipeImpl(PrefetcherPipeImpl&&) = delete;
    PrefetcherPipeImpl& operator=(PrefetcherPipeImpl&&) = delete;
    ~PrefetcherPipeImpl();

    uint32_t buffer_address() const { return space().buffer_address(); }
    uint32_t config_address() const { return space().config_address(); }
    uint32_t ring_size() const { return space().ring_size(); }
    // Active pipe-consumer lanes (matches relay num_producers when multi-producer).
    uint32_t num_credit_lanes() const { return active_credit_lanes_; }
    // Slots allocated in the config page (Quasar may reserve headroom above active).
    uint32_t credit_lane_capacity() const { return space().credit_lane_capacity(); }
    // Set active lanes from consumer geometry. Host state only: dispatch packs the value into
    // every bound program's kernel-config slot (ordered with that program), nothing in
    // persistent L1 is written. May upgrade from the carve-time default of 1, or no-op if
    // already equal. Reprogramming to a different value after arming is rejected.
    void set_active_credit_lanes(uint32_t num_lanes);
    // The checks set_active_credit_lanes(num_lanes) would apply if the pipe were currently armed
    // with `from_lanes`; throws on rejection, no state change. Lets a program preflight a batch of
    // bindings (which may arm the same pipe more than once) before committing any of them.
    void validate_credit_lane_transition(uint32_t from_lanes, uint32_t num_lanes) const;
    // Lane mode (num_lanes > 1) needs an exact entry ring whose entry count is a multiple of
    // num_lanes; throws otherwise. Call before set_active_credit_lanes so a rejected bind
    // does not leave the persistent pipe re-armed.
    void validate_lane_geometry(uint32_t entry_size, uint32_t num_lanes) const;

    uint32_t config_page_size() const { return space().config_page_size(); }
    uint32_t credit_reset_offset() const { return space().layout().sent_offset; }
    uint32_t credit_reset_size() const { return space().config_page_size() - space().layout().sent_offset; }
    const std::vector<uint32_t>& config_page(const CoreCoord& core) const;

    const CoreRangeSet& sender_cores() const { return sender_cores_; }
    const CoreRangeSet& receiver_cores() const { return receiver_cores_; }
    const CoreRangeSet& all_cores() const { return all_cores_; }
    CoreCoord sender_core() const { return sender_core_; }
    const distributed::MeshDevice* get_device() const { return space().get_device(); }
    // The space this pipe was carved from. Throws if that space has been destroyed.
    const PrefetcherPipeSpaceImpl& space() const;
    SenderCoreType sender_core_type() const;
    uint32_t initial_entry_size() const { return initial_entry_size_; }
    uint64_t identity() const { return identity_; }
    uint64_t tensor_prefetcher_factory_id() const { return tensor_prefetcher_factory_id_; }
    // How many pipes the CreatePrefetcherPipesForTensorPrefetcher call that made this one returned.
    uint32_t tensor_prefetcher_factory_num_pipes() const { return tensor_prefetcher_factory_num_pipes_; }
    uint32_t recv_index_base() const { return recv_index_base_; }
    DeviceAddr sender_state_drisc_l1_base() const;

private:
    friend class PrefetcherPipeSpaceImpl;

    void build_config_pages();
    void build_dram_sender_config_pages();
    // Called by the space's destructor when it is destroyed while this pipe is still alive.
    void detach_from_space() noexcept;

    PrefetcherPipeSpaceImpl* space_ = nullptr;
    uint64_t identity_ = 0;
    CoreCoord sender_core_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    bool claimed_ = false;
    // Active lanes for striping / wait_front (per-program kernel-config slot); set from the
    // receiver kernel's thread count when a Program binds the pipe.
    uint32_t active_credit_lanes_ = 1;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> config_pages_;
    SenderCoreType sender_core_type_{};
    uint32_t initial_entry_size_ = 0;
    uint32_t recv_index_base_ = 0;
    uint64_t tensor_prefetcher_factory_id_ = 0;
    uint32_t tensor_prefetcher_factory_num_pipes_ = 0;
    std::shared_ptr<DriscL1Allocation> drisc_config_page_;
};

}  // namespace experimental
}  // namespace tt::tt_metal
