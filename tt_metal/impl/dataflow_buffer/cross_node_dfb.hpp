// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>

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

    static constexpr auto attribute_names =
        std::forward_as_tuple("sender_receiver_core_mapping", "entry_size", "num_entries", "buffer_type");
    auto attribute_values() const {
        return std::make_tuple(sender_receiver_mapping_, entry_size_, num_entries_, dfb_buffer().buffer_type());
    }

private:
    void setup_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);
    void setup_buffers_with_borrowed_data(Buffer& data_buffer, uint32_t max_num_receivers_per_sender);
    void allocate_config_and_write_pages(uint32_t max_num_receivers_per_sender, BufferType config_buffer_type);

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
};

/**
 * @brief Allocates a CrossNodeDFB in L1 on the device.
 *
 * @param device The device to create the CrossNodeDFB on.
 * @param sender_receiver_mapping M (sender, receivers) pairs; disjoint receiver sets.
 * @param entry_size Size of one entry in bytes; must be a multiple of L1_ALIGNMENT.
 * @param num_entries Number of entries per receiver ring.
 * @param buffer_type L1 or L1_SMALL (default L1).
 */
CrossNodeDFB CreateCrossNodeDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Creates a CrossNodeDFB backed by a user-supplied sharded L1 data buffer.
 *
 * Config is still runtime-allocated. `data_buffer` must match the shard layout Create
 * would allocate (HEIGHT_SHARDED over senders∪receivers, page_size = entry_size *
 * num_entries). The buffer must remain alive for the CrossNodeDFB lifetime (keep a
 * shared_ptr / AnyBuffer in scope).
 */
CrossNodeDFB CreateCrossNodeDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    Buffer& data_buffer);

/**
 * @brief Attach a CrossNodeDFB to a program on the specified cores.
 *
 * Assigns a runtime-generated slot index (0-based, ascending) per CrossNodeDFB
 * instance within the program. Re-attaching the same CrossNodeDFB to additional
 * cores reuses the same slot. CrossNodeDFB and GlobalCircularBuffer are mutually
 * exclusive within a program — attaching a CrossNodeDFB to a program that already
 * has a GlobalCB (or vice versa) will throw.
 *
 * CrossNodeDFB state is same-program only: interfaces are (re)initialized from the
 * kernel-config region on each program launch / init, and are not persisted across
 * programs.
 *
 * @param program       The program to attach to.
 * @param core_spec     Cores to configure (must be senders or receivers of gdfb).
 * @param gdfb          The CrossNodeDFB.
 * @param relay_dfb_name  Optional accessor name of the local DFB in the kernel that
 *                  should be relayed for compute. Resolved to a logical handle at JIT
 *                  finalization time (Phase 3: host L1 aliasing + align emit).
 * @return Runtime-assigned slot index for kernel compile-time args. Re-attaching the
 *         same CrossNodeDFB returns the same slot.
 */
uint8_t AttachCrossNodeDFB(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CrossNodeDFB& gdfb,
    const std::optional<std::string>& relay_dfb_name = std::nullopt);

/**
 * @brief Update the address of a dynamic CrossNodeDFB without recompiling.
 *
 * Analogous to UpdateDynamicCircularBufferAddress for GlobalCBs.
 */
void UpdateDynamicCrossNodeDFBAddress(Program& program, const CrossNodeDFB& gdfb);

}  // namespace experimental
}  // namespace tt::tt_metal

namespace std {
template <>
struct hash<tt::tt_metal::experimental::CrossNodeDFB> {
    std::size_t operator()(const tt::tt_metal::experimental::CrossNodeDFB& gdfb) const;
};
}  // namespace std
