// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {

class Buffer;
class IDevice;
// Impl-only DRISC L1 arena type; held as a private shared_ptr so this header doesn't have to
// include the impl arena header.
class DriscL1Allocation;

namespace experimental {

namespace global_circular_buffer_dram_sender {
struct GlobalCircularBufferDramSenderInternals;
}  // namespace global_circular_buffer_dram_sender

// Impl-only backing state for tt-metalium/global_circular_buffer.hpp's GlobalCircularBuffer. The
// public header holds this by std::shared_ptr (copies of a GlobalCircularBuffer share the same
// backing buffers, matching the pre-pimpl AnyBuffer/DriscL1Allocation shared-copy semantics), and
// forwards its public surface to the methods here. Callers inside tt_metal/ that need the
// INTERNAL-only surface (all_cores(), buffer_address(), get_device(), attribute_names/values) go
// through GlobalCircularBuffer::impl().
class GlobalCircularBufferImpl {
public:
    GlobalCircularBufferImpl(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type);

    GlobalCircularBufferImpl(const GlobalCircularBufferImpl&) = default;
    GlobalCircularBufferImpl& operator=(const GlobalCircularBufferImpl&) = default;
    GlobalCircularBufferImpl(GlobalCircularBufferImpl&&) noexcept = default;
    GlobalCircularBufferImpl& operator=(GlobalCircularBufferImpl&&) noexcept = default;

    const Buffer& cb_buffer() const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    DeviceAddr buffer_address() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    IDevice* get_device() const { return this->device_; }

    static constexpr auto attribute_names =
        std::forward_as_tuple("sender_receiver_core_mapping", "size", "buffer_type");
    auto attribute_values() const {
        return std::make_tuple(
            this->sender_receiver_core_mapping_, this->size_, cb_buffer_.get_buffer()->buffer_type());
    }

private:
    void setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);
    // Allocates and writes the per-GCB sender state block in DRISC L1. DRAM-sender flavour only.
    void initialize_dram_sender_state_block(
        distributed::MeshDevice* mesh_device, uint32_t max_num_receivers_per_sender);

    // Tag for the private experimental DRAM-sender constructor; only the experimental
    // factory (a friend) can name this type. Takes MeshDevice because the DRAM-sender
    // path relies on the per-mesh DriscL1Arena for pages_sent placement.
    struct DramSenderTag {};
    GlobalCircularBufferImpl(
        distributed::MeshDevice* mesh_device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type,
        DramSenderTag);

    // GlobalCircularBufferImpl is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    distributed::AnyBuffer cb_buffer_;
    distributed::AnyBuffer cb_config_buffer_;
    IDevice* device_;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t size_ = 0;
    // Private experimental DRAM-sender metadata. `sender_core_type_value_` is stored as
    // uint8_t (0=Worker, 1=Dram) so the SenderCoreType enum stays in the experimental
    // header. Accessed only through the friend struct in
    // tt-metalium/experimental/global_circular_buffer.hpp.
    uint8_t sender_core_type_value_ = 0;
    // Base of the per-receiver pages_sent/pages_acked counters in DRISC L1. Carved
    // from the front of the combined sender-state allocation below.
    DeviceAddr pages_sent_drisc_l1_base_ = 0;
    DeviceAddr pages_sent_worker_l1_base_ = 0;
    // DRISC L1 base of the per-GCB "sender state block" (RemoteSenderCBInterface
    // bytes + sender config block + receiver NOC XY table). Pre-initialized at GCB
    // construction; on each request that targets this GCB the Tensor prefetcher
    // kernel loads the RemoteSenderCBInterface region into its static cb_interface[]
    // slot, runs the chunk loop, and writes fifo_wr_ptr back so the ring offset
    // survives multi-GCB request switching. Layout in
    // tt_metal/impl/buffers/dram_sender_state_block.hpp.
    DeviceAddr sender_state_drisc_l1_base_ = 0;
    std::vector<std::vector<CoreCoord>> receiver_coords_per_sender_;
    // RAII handle for the combined pages_sent + sender-state-block allocation in the
    // per-mesh DriscL1Arena. Held via shared_ptr so copies of the Impl share the same
    // backing range; released when the last GlobalCircularBuffer copy goes out of scope.
    // Empty for worker-sender GCBs.
    std::shared_ptr<::tt::tt_metal::DriscL1Allocation> drisc_sender_state_alloc_;

    friend struct global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals;
};

}  // namespace experimental
}  // namespace tt::tt_metal
