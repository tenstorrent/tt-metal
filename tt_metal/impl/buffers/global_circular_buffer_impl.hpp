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

namespace experimental {

// Impl-only backing state for tt-metalium/global_circular_buffer.hpp's GlobalCircularBuffer.
// The public type holds a unique_ptr to this impl and copies by value (deep-copy of *impl_).
// MeshBuffer members still share device resources across copies.
// Callers inside tt_metal/ that need the INTERNAL-only surface (all_cores(), get_device())
// go through GlobalCircularBuffer::impl().
class GlobalCircularBufferImpl {
public:
    GlobalCircularBufferImpl(
        distributed::MeshDevice& device,
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
        return std::make_tuple(this->sender_receiver_core_mapping_, this->size_, cb_buffer().buffer_type());
    }

private:
    void setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);

    // GlobalCircularBufferImpl is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<distributed::MeshBuffer> cb_buffer_;
    std::shared_ptr<distributed::MeshBuffer> cb_config_buffer_;
    distributed::MeshDevice* device_;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t size_ = 0;
};

}  // namespace experimental
}  // namespace tt::tt_metal
