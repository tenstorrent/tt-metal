// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
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
class DriscL1Allocation;
class IDevice;
class Program;

namespace experimental {

// Forward declarations for the DRAM-sender extension defined in
// tt-metalium/experimental/persistent_dfb.hpp. DRAM-sender mode is opt-in and is not part of
// the public PersistentDFB API surface; existing callers see the original interface unchanged.
namespace persistent_dfb_dram_sender {
struct PersistentDfbDramSenderInternals;
}  // namespace persistent_dfb_dram_sender

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
     *   auto pdfb = CreatePersistentDFB(device, mapping, entry_size, num_entries);
     *   AttachPersistentDFB(program, pdfb, sender_cores);     // or receivers / subset
     *   // optional: CreatePersistentRelayDataflowBuffer(program, receivers, cfg, id);
     *
     * Device kernel flows (sender / receiver / relay) are documented on the device API:
     *   tt_metal/hw/inc/api/dataflow/persistent_dfb.h
     *
     */
    PersistentDFB(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
        uint32_t entry_size,
        uint32_t num_entries,
        BufferType buffer_type = BufferType::L1);

    PersistentDFB(const PersistentDFB&) = delete;
    PersistentDFB& operator=(const PersistentDFB&) = delete;
    PersistentDFB(PersistentDFB&&) = delete;
    PersistentDFB& operator=(PersistentDFB&&) = delete;

    uint32_t buffer_address() const;
    const Buffer& config_buffer() const;
    uint32_t config_address() const;
    uint32_t entry_size() const;
    uint32_t num_entries() const;
    uint32_t ring_size() const { return entry_size_ * num_entries_; }

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
    // Tag selecting the DRAM-sender constructor. Private so the only way in is
    // CreatePersistentDFBForTensorPrefetcher, which owns the bank -> sender-core mapping.
    struct DramSenderTag {};

    friend struct persistent_dfb_dram_sender::PersistentDfbDramSenderInternals;

    /**
     * DRAM-sender PersistentDFB: senders are programmable DRAM cores (Blackhole DRISCs)
     * rather than worker cores.
     *
     * Differences from the worker-sender ctor above:
     *   - The data ring and the config buffer are sharded over receivers only. DRAM cores
     *     hold no ring slice, and their logical coords are not worker coords so they cannot
     *     share a sharded buffer with them.
     *   - Each sender's config page lives in DRISC L1 (allocated from the per-mesh
     *     DriscL1Arena), written directly over NOC, and is never Attached: DRAM cores are
     *     not dispatched to, so the DRISC kernel builds its sender interface from an
     *     explicit config-page address instead of a launch-message slot.
     *   - Credit counters cross L1 address spaces. The sender's remote-counter base and each
     *     receiver's ack target are page-relative offsets the host computes so that a
     *     sender's `base + 2*r*L1_ALIGNMENT` lands on receiver r's own page, and a receiver's
     *     ack lands in DRISC L1.
     */
    PersistentDFB(
        distributed::MeshDevice* mesh_device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
        uint32_t entry_size,
        uint32_t num_entries,
        BufferType buffer_type,
        DramSenderTag);

    void setup_buffers(BufferType buffer_type);
    void allocate_config_buffer(BufferType config_buffer_type);
    void build_config_pages();
    void write_config_to_device();

    // DRAM-sender only: allocate the DRISC-L1 sender blocks, build the receiver config pages
    // (whose ack targets point into DRISC L1), and stamp both sides.
    void setup_dram_sender_buffers(distributed::MeshDevice* mesh_device, BufferType buffer_type);
    void build_dram_sender_receiver_config_pages();
    void initialize_dram_sender_state(distributed::MeshDevice* mesh_device);

    distributed::AnyBuffer data_buffer_;
    distributed::AnyBuffer config_buffer_;
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

    // ---- DRAM-sender state (all zero / empty for a worker-sender PersistentDFB) ----
    uint8_t sender_core_type_value_ = 0;  // experimental::SenderCoreType
    // Base of this object's block in the DRISC L1 arena. Uniform across banks: every sender
    // core plants its prefix + config page at the same L1 offset.
    DeviceAddr sender_state_drisc_l1_base_ = 0;
    std::shared_ptr<DriscL1Allocation> drisc_sender_state_alloc_;
    // Physical worker NOC XY of each sender's receivers, in bank-local slab order.
    std::vector<std::vector<CoreCoord>> receiver_coords_per_sender_;
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
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Attach a PersistentDFB to `program` on the given cores (non-owning).
 *
 * `cores` must be a non-empty subset of the PersistentDFB's mapping cores.
 * Returns an independent persistent_dfb_id in [0, MAX_PERSISTENT_DFBS).
 *
 * WH/BH: on each sender core, only one DM (BRISC or NCRISC) may own PersistentDFB
 * credit / resize / push for that Attach. Both DMs can run on the same physical
 * core, but dual-DM ownership races on local sent counters and the checkpoint
 * cursor. Host binding / kernel placement should pin a single sender DM owner
 * until Attach can enforce this.
 *
 * @param entry_size_override When set, overrides the dense-slot entry_size (defaults to object's).
 */
uint8_t AttachPersistentDFB(
    Program& program,
    PersistentDFB& persistent_dfb,
    const CoreRangeSet& cores,
    std::optional<uint32_t> entry_size_override = std::nullopt);

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
