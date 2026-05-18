// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {

class Buffer;
class CircularBufferConfig;
class IDevice;
class Program;

namespace experimental {

// Like GlobalCircularBuffer, but with senders on DRAM cores (DRISC) instead of worker cores.
//
// For the prototype we only allocate sharded data/config buffers on the receiver worker cores;
// the sender side (DRISC) is hand-managed via runtime args (see receiver_coords_per_sender()).
// The receiver-side config carries each sender's DRAM virtual coord so receiver ack semaphores
// route back to the DRISC.
class DramSenderGlobalCircularBuffer {
public:
    DramSenderGlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& dram_sender_to_worker_receivers,
        uint32_t size,
        BufferType buffer_type = BufferType::L1);

    DramSenderGlobalCircularBuffer(const DramSenderGlobalCircularBuffer&) = default;
    DramSenderGlobalCircularBuffer& operator=(const DramSenderGlobalCircularBuffer&) = default;
    DramSenderGlobalCircularBuffer(DramSenderGlobalCircularBuffer&&) noexcept = default;
    DramSenderGlobalCircularBuffer& operator=(DramSenderGlobalCircularBuffer&&) noexcept = default;

    const Buffer& cb_buffer() const;

    const CoreRangeSet& sender_cores() const;    // logical DRAM cores
    const CoreRangeSet& receiver_cores() const;  // logical worker cores
    DeviceAddr buffer_address() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    // Physical worker NOC XY of each sender's receivers, ordered to match the receiver config.
    // Used by the DRISC kernel runtime args.
    const std::vector<std::vector<CoreCoord>>& receiver_coords_per_sender() const;
    // L1 address (in DRISC unreserved space) where the DRISC sender's local pages_sent/pages_acked
    // counters live. The receivers' ack NOC-incs land here. The DRISC kernel uses this as the
    // base for its own pages_sent counters.
    DeviceAddr pages_sent_drisc_l1_base() const;
    // L1 address (in receiver worker L1, inside the GCB config buffer page) where each receiver's
    // local pages_sent counter lives. The DRISC kernel uses this as the NOC-inc target when it
    // pushes pages — receivers read pages_sent from here locally.
    DeviceAddr pages_sent_worker_l1_base() const;
    IDevice* get_device() const { return device_; }

    static constexpr auto attribute_names =
        std::forward_as_tuple("sender_receiver_core_mapping", "size", "buffer_type");
    auto attribute_values() const {
        return std::make_tuple(sender_receiver_core_mapping_, size_, cb_buffer_.get_buffer()->buffer_type());
    }

private:
    void setup_receiver_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);

    distributed::AnyBuffer cb_buffer_;
    distributed::AnyBuffer cb_config_buffer_;
    IDevice* device_;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;                                       // logical DRAM
    CoreRangeSet receiver_cores_;                                     // logical worker
    std::vector<std::vector<CoreCoord>> receiver_coords_per_sender_;  // translated worker NOC XY
    DeviceAddr pages_sent_drisc_l1_base_ = 0;
    DeviceAddr pages_sent_worker_l1_base_ = 0;
    uint32_t size_ = 0;
};

DramSenderGlobalCircularBuffer CreateDramSenderGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& dram_sender_to_worker_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Receiver-side attachment of a DramSenderGlobalCircularBuffer to a worker-core program. The
// sender side (DRISC) is hand-managed by the kernel and intentionally not attached here.
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const DramSenderGlobalCircularBuffer& dram_sender_global_circular_buffer);

}  // namespace experimental

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::experimental::DramSenderGlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::experimental::DramSenderGlobalCircularBuffer& cb) const;
};

}  // namespace std
