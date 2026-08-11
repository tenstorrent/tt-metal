// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
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

    // Borrowed-data constructor: uses `data_buffer` for the data ring (must outlive
    // this object via shared ownership) and still allocates the config sideband.
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

    // The data ring (sharded over all_cores, one ring FIFO page per core; receivers use theirs).
    const Buffer& dfb_buffer() const;
    // Non-const overload so this ring can be handed to UpdateDynamicCrossNodeDFBAddress,
    // which needs shared ownership of the buffer to borrow it.
    Buffer& dfb_buffer();

    // The config sideband (sharded over all_cores = senders ∪ receivers, one page per core).
    const Buffer& config_buffer() const;

    uint32_t buffer_address() const;
    uint32_t config_address() const;
    uint32_t credit_reset_address() const;
    uint32_t credit_reset_size() const;
    uint32_t entry_size() const;
    uint32_t num_entries() const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    IDevice* get_device() const { return device_; }

    // Retarget the data ring to `data_buffer` and rewrite config pages in place.
    // Used by UpdateDynamicCrossNodeDFBAddress; config sideband address is unchanged.
    void retarget_data_buffer(Buffer& data_buffer);

private:
    void setup_buffers(BufferType buffer_type);
    void setup_buffers_with_borrowed_data(Buffer& data_buffer);
    void validate_data_buffer(Buffer& data_buffer) const;
    // Create path: allocate a new config sideband, then fill and write pages.
    void allocate_config_and_write_pages(BufferType config_buffer_type);
    // Fill and write config pages for the current data/config buffers (Create or UpdateDynamic).
    void write_config_pages();

    distributed::AnyBuffer dfb_buffer_;
    distributed::AnyBuffer config_buffer_;
    IDevice* device_ = nullptr;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t entry_size_ = 0;
    uint32_t num_entries_ = 0;
    uint32_t credit_reset_offset_ = 0;
    uint32_t credit_reset_size_ = 0;
    uint32_t max_num_receivers_per_sender_ = 0;
};

/**
 * @brief Allocates a CrossNodeDFB and wires it into `program` on all mapping cores.
 *
 * Same-program only: creates the data/config buffers, stores the host object in the
 * program, and returns the dense `remote_dfb_id` for kernel compile-time args.
 * CrossNodeDFB and GlobalCircularBuffer are mutually exclusive within a program.
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
 * Config is still runtime-allocated. `data_buffer` must match the shard layout Create
 * would allocate (HEIGHT_SHARDED over senders∪receivers, page_size = entry_size *
 * num_entries). The buffer must remain alive for the CrossNodeDFB lifetime (keep a
 * shared_ptr / AnyBuffer in scope).
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
 * entry_size * num_entries). Config sideband stays in place; only fifo_start addresses
 * and relay borrowed bases are retargeted. Analogous to UpdateDynamicCircularBufferAddress.
 */
void UpdateDynamicCrossNodeDFBAddress(Program& program, uint8_t remote_dfb_id, Buffer& buffer);

}  // namespace experimental
}  // namespace tt::tt_metal
