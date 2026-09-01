// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/dataflow_buffer/dataflow_buffer.hpp"

#include <variant>

namespace tt::tt_metal {

class Buffer;
class IDevice;
class Program;

namespace experimental {

class PersistentDFB {
public:
    /**
     * Host object for a durable cross-program remote DFB.
     *
     * Lifetime: Create allocates the data ring + config Buffer and writes initial config
     * pages once. Keep this object alive for the entire time any program Attaches / uses
     * it, destroying it frees the ring and config. The runtime does not fence peers;
     * only destroy (or let it go out of scope) after every peer program has Finished.
     *
     * Host programming model:
     *   auto pdfb = CreatePersistentDFB(device, mapping, ring_size);
     *   AttachPersistentDFB(program, pdfb, sender_cores, entry_size);  // or all receivers
     *   // optional: CreatePersistentRelayDataflowBuffer(program, receivers, cfg, id);
     *
     * Device kernel flows (sender / receiver / relay) are documented on the device API:
     *   tt_metal/hw/inc/api/dataflow/persistent_dfb.h
     *
     */
    PersistentDFB(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
        uint32_t ring_size,
        BufferType buffer_type = BufferType::L1);

    PersistentDFB(const PersistentDFB&) = delete;
    PersistentDFB& operator=(const PersistentDFB&) = delete;
    PersistentDFB(PersistentDFB&&) = delete;
    PersistentDFB& operator=(PersistentDFB&&) = delete;

    uint32_t buffer_address() const;
    const Buffer& config_buffer() const;
    uint32_t config_address() const;
    uint32_t ring_size() const { return ring_size_; }

    uint32_t config_page_size() const { return config_page_size_; }
    uint32_t credit_reset_offset() const { return credit_reset_offset_; }
    uint32_t credit_reset_size() const { return credit_reset_size_; }
    const std::vector<uint32_t>& config_page(const CoreCoord& core) const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    IDevice* get_device() const { return device_; }

private:
    void setup_buffers(BufferType buffer_type);
    void allocate_config_buffer(BufferType config_buffer_type);
    void build_config_pages();
    void write_config_to_device();

    distributed::AnyBuffer data_buffer_;
    distributed::AnyBuffer config_buffer_;
    uint32_t data_address_ = 0;
    IDevice* device_ = nullptr;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t ring_size_ = 0;
    uint32_t config_page_size_ = 0;
    uint32_t credit_reset_offset_ = 0;
    uint32_t credit_reset_size_ = 0;
    uint32_t max_num_receivers_per_sender_ = 0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> config_pages_;
};

/**
 * @brief Create a PersistentDFB host object with internal data ring + config Buffer.
 *
 * Config pages are written to device L1 at Create (safe-point initial write).
 * Caller keeps the object alive for cross-program persistence; Attach wires programs
 * to the same ring/config addresses.
 */
PersistentDFB CreatePersistentDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t ring_size,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Attach a PersistentDFB to `program` on the given cores (non-owning).
 *
 * `cores` must be a non-empty role-complete subset of the PersistentDFB's mapping
 * cores: if any sender is present, all senders must be present; likewise for
 * receivers. This prevents one PersistentDFB role from being split across Programs.
 * Returns an independent persistent_dfb_id in [0, 255).
 *
 * WH/BH: on each sender core, only one DM (BRISC or NCRISC) may own PersistentDFB
 * credit / resize / push for that Attach. Both DMs can run on the same physical
 * core, but dual-DM ownership races on local sent counters and the checkpoint
 * cursor. Host binding / kernel placement should pin a single sender DM owner
 * until Attach can enforce this.
 *
 * @param entry_size Dense entry size for this Program execution epoch.
 */
uint8_t AttachPersistentDFB(
    Program& program, PersistentDFB& persistent_dfb, const CoreRangeSet& cores, uint32_t entry_size);

/**
 * @brief Create and register the local DFB used to relay a PersistentDFB to TRISC.
 *
 * The local DFB borrows the Persistent data ring. `persistent_dfb_id` must already be
 * Attached on `receiver_core_spec`. Relay entry_size / depth must match this Attach's
 * dense entry_size and `ring_size / entry_size`.
 *
 * @return Program-unique host DFB id (distinct from Persistent `persistent_dfb_id`).
 */
uint32_t CreatePersistentRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t persistent_dfb_id);

}  // namespace experimental
}  // namespace tt::tt_metal
