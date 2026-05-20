// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {

class Buffer;
class IDevice;

namespace experimental {

// Domain that the sender cores live in. Worker = standard sharded GCB where senders are
// worker cores hosting their own slice of the cb_buffer_ in L1. Dram = senders are
// programmable DRAM cores (Blackhole DRISCs) that own their staging L1 separately; the
// cb_buffer_ is sharded over receivers only, and the receiver-side config_buffer slot
// for `remote_pages_addr_override` points at DRISC L1 so the receiver's pages_acked
// NoC-inc lands on the DRISC side.
enum class SenderCoreType : uint8_t {
    Worker = 0,
    Dram = 1,
};

class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1,
        SenderCoreType sender_core_type = SenderCoreType::Worker);

    GlobalCircularBuffer(const GlobalCircularBuffer&) = default;
    GlobalCircularBuffer& operator=(const GlobalCircularBuffer&) = default;

    GlobalCircularBuffer(GlobalCircularBuffer&&) noexcept = default;
    GlobalCircularBuffer& operator=(GlobalCircularBuffer&&) noexcept = default;

    const Buffer& cb_buffer() const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    DeviceAddr buffer_address() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    IDevice* get_device() const { return this->device_; }

    // Sender-domain accessors. For Worker GCBs the DRAM-specific ones return 0 / empty.
    SenderCoreType sender_core_type() const { return sender_core_type_; }
    // DRAM-only: physical worker NOC XY for each sender's receivers (used by the DRISC
    // kernel's runtime args).
    const std::vector<std::vector<CoreCoord>>& receiver_coords_per_sender() const {
        return receiver_coords_per_sender_;
    }
    // DRAM-only: DRISC unreserved L1 base where the sender's pages_sent/acked counters live.
    // The factory builds the DRISC L1 layout starting at this address.
    DeviceAddr pages_sent_drisc_l1_base() const { return pages_sent_drisc_l1_base_; }
    // DRAM-only: worker-L1 offset (inside the receiver's config page) where the
    // receiver's local pages_sent counter lives. The DRISC NoC-incs to this address.
    DeviceAddr pages_sent_worker_l1_base() const { return pages_sent_worker_l1_base_; }

    static constexpr auto attribute_names =
        std::forward_as_tuple("sender_receiver_core_mapping", "size", "buffer_type", "sender_core_type");
    auto attribute_values() const {
        return std::make_tuple(
            this->sender_receiver_core_mapping_,
            this->size_,
            cb_buffer_.get_buffer()->buffer_type(),
            this->sender_core_type_);
    }

private:
    void setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);

    // GlobalCircularBuffer is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    distributed::AnyBuffer cb_buffer_;
    distributed::AnyBuffer cb_config_buffer_;
    IDevice* device_;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t size_ = 0;
    SenderCoreType sender_core_type_ = SenderCoreType::Worker;
    // DRAM-sender-only metadata (zero/empty for Worker senders).
    std::vector<std::vector<CoreCoord>> receiver_coords_per_sender_;
    DeviceAddr pages_sent_drisc_l1_base_ = 0;
    DeviceAddr pages_sent_worker_l1_base_ = 0;
};

/**
 * @brief Allocates a global circular buffer in L1 on the device.
 *
 * @param device The device to create the global circular buffer on.
 * @param sender_receiver_core_mapping The mapping of remote sender to remote receiver cores for the circular buffer.
 * @param size Size of the global circular buffer per core in bytes.
 * @param buffer_type Buffer type to store the global circular buffer. Can only be an L1 buffer type.\
 * @return The allocated global circular buffer.
 */
GlobalCircularBuffer CreateGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    SenderCoreType sender_core_type = SenderCoreType::Worker);

/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores using the address space of the
 * global circular bufferand adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @param global_circular_buffer Global circular buffer to use the address space and configuration of.
 * @return CBHandle representing the Circular Buffer ID.
 */
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer);

/**
 * @brief Updates the address of a dynamic global circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated global L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer);

}  // namespace experimental

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::experimental::GlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const;
};

}  // namespace std
