// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/dataflow_buffer/dataflow_buffer.hpp"

namespace tt::tt_metal {

class Buffer;
class IDevice;
class Program;

namespace experimental {

class CrossNodeDFB {
public:
    // sender_receiver_mapping: M (sender_core, receiver_CoreRangeSet) pairs.
    // Topology rules:
    //   - No duplicate sender cores.
    //   - No duplicate receiver cores within a sender's set.
    //   - No receiver core appears in more than one sender's set (disjoint receivers).
    //   - Sender and receiver sets are disjoint (no core plays both roles).
    CrossNodeDFB(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
        uint32_t entry_size,
        uint32_t num_entries,
        BufferType buffer_type = BufferType::L1);

    // Borrowed-data constructor: uses `data_buffer` for the data ring. Caller owns
    // `data_buffer` for the CrossNodeDFB lifetime; this object only records its address.
    // The config Buffer is owned by CrossNodeDFB
    CrossNodeDFB(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
        uint32_t entry_size,
        uint32_t num_entries,
        Buffer& data_buffer);

    CrossNodeDFB(const CrossNodeDFB&) = default;
    CrossNodeDFB& operator=(const CrossNodeDFB&) = default;
    CrossNodeDFB(CrossNodeDFB&&) noexcept = default;
    CrossNodeDFB& operator=(CrossNodeDFB&&) noexcept = default;

    uint32_t buffer_address() const;
    const Buffer& config_buffer() const;
    uint32_t config_address() const;
    uint32_t entry_size() const;
    uint32_t num_entries() const;
    // Bytes per core of the data ring (= entry_size * num_entries).
    uint32_t ring_size() const { return entry_size_ * num_entries_; }

    uint32_t config_page_size() const { return config_page_size_; }
    // Byte offset of the credit window within each config page.
    uint32_t credit_reset_offset() const { return credit_reset_offset_; }
    uint32_t credit_reset_size() const { return credit_reset_size_; }
    // Per-core host config page (page-relative words 5–7). Missing cores are not participants.
    const std::vector<uint32_t>& config_page(const CoreCoord& core) const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    IDevice* get_device() const { return device_; }

    // Retarget the data ring to `data_buffer` and rebuild host config pages in place.
    // The next program launch picks up the updates.
    void retarget_data_buffer(Buffer& data_buffer);

private:
    void setup_buffers(BufferType buffer_type);
    void setup_buffers_with_borrowed_data(Buffer& data_buffer);
    void validate_data_buffer(Buffer& data_buffer) const;
    // Allocate the program-owned sharded config Buffer. Its contents are populated at launch.
    void allocate_config_buffer(BufferType config_buffer_type);
    // Build host config page images (Create or UpdateDynamic).
    void rebuild_config_pages();
    // Point at an external data ring address; drops any CrossNode-owned ring.
    void set_data_address(uint32_t data_address);

    // Owned allocation when CrossNode creates the data ring; empty when borrowing.
    distributed::AnyBuffer owned_dfb_buffer_;
    // Program-owned config allocation, sharded over every participant core.
    // Launch dispatch materializes the current host pages.
    distributed::AnyBuffer config_buffer_;
    // Data-ring L1 address. From owned_dfb_buffer_ when owning; from the caller's Buffer
    // when borrowing. Device config/relays only need this address.
    uint32_t data_address_ = 0;
    IDevice* device_ = nullptr;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t entry_size_ = 0;
    uint32_t num_entries_ = 0;
    uint32_t config_page_size_ = 0;
    uint32_t credit_reset_offset_ = 0;
    uint32_t credit_reset_size_ = 0;
    uint32_t max_num_receivers_per_sender_ = 0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> config_pages_;
};

/**
 * @brief Allocates a CrossNodeDFB and wires it into `program` on all mapping cores.
 *
 * Same-program only: creates the data/config Buffers and host config pages, stores the host
 * object in the program, and returns the dense `remote_dfb_id` for kernel compile-time
 * args. CrossNodeDFB and GlobalCircularBuffer are mutually exclusive within a program.
 *
 * For UpdateDynamic, pass a replacement data Buffer with matching shard layout.
 *
 * @return Runtime-assigned CrossNode slot id (0-based, ascending per program).
 */
uint8_t CreateCrossNodeDFB(
    Program& program,
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Creates a CrossNodeDFB backed by a user-supplied sharded L1 data buffer
 * and wires it into `program` on all mapping cores.
 *
 * Config pages are host-only until launch writes the dedicated config Buffer.
 * `data_buffer` must match the shard layout
 * Create would allocate (HEIGHT_SHARDED over senders∪receivers, page_size = entry_size *
 * num_entries). Caller keeps `data_buffer` alive for the CrossNodeDFB lifetime;
 * CrossNode only records its address.
 *
 * @return Runtime-assigned CrossNode slot id.
 */
uint8_t CreateCrossNodeDFB(
    Program& program,
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    Buffer& data_buffer);

/**
 * @brief Create and register the normal local DFB used to relay a CrossNodeDFB to TRISC.
 *
 * The local DFB borrows the CrossNode data ring and is recorded as the sole typed
 * relay relationship for `remote_dfb_id`. The CrossNodeDFB must already be created
 * in this program with that slot.
 *
 * @return Program-unique host DFB id (distinct from CrossNode `remote_dfb_id`).
 */
uint32_t CreateCrossNodeRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t remote_dfb_id);

/**
 * @brief Update the data-ring address of a CrossNodeDFB slot without recompiling.
 *
 * `remote_dfb_id` is the slot returned by CreateCrossNodeDFB. `buffer` must match the
 * shard layout of the existing data ring (HEIGHT_SHARDED over all_cores, page_size =
 * entry_size * num_entries). Config pages are rebuilt but device state is unchanged
 * until the next program launch. Re-launching a Program with CrossNodeDFBs must be
 * ordered after its previous launch has completed because launches reuse the same
 * config and data Buffers. Queue order provides this on one CQ; across CQs, the user
 * must insert an event dependency. Analogous to UpdateDynamicCircularBufferAddress.
 */
void UpdateDynamicCrossNodeDFBAddress(Program& program, uint8_t remote_dfb_id, Buffer& buffer);

}  // namespace experimental
}  // namespace tt::tt_metal
