// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "impl/dataflow_buffer/dataflow_buffer.hpp"

#include <variant>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}

namespace experimental {

// Implementation of the PrefetcherPipe host object declared in
// tt-metalium/experimental/prefetcher_pipe.hpp: it owns the persistent L1 allocations, composes
// each core's config page and stamps those pages onto the device. A PrefetcherPipe holds one of
// these and forwards to it, and the host runtime records an Attach by pointing at one. That
// pointer is only as good as the handle: destroying the PrefetcherPipe destroys this object and
// frees the ring, which is why the handle must outlive every Program attached to it. A handle
// cannot be moved or copied, so destruction is the only way to get there.
class PrefetcherPipeImpl {
public:
    PrefetcherPipeImpl(
        distributed::MeshDevice* device,
        CoreCoord sender_core,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        BufferType buffer_type = BufferType::L1);

    PrefetcherPipeImpl(const PrefetcherPipeImpl&) = delete;
    PrefetcherPipeImpl& operator=(const PrefetcherPipeImpl&) = delete;
    PrefetcherPipeImpl(PrefetcherPipeImpl&&) = delete;
    PrefetcherPipeImpl& operator=(PrefetcherPipeImpl&&) = delete;
    ~PrefetcherPipeImpl();

    uint32_t buffer_address() const;
    uint32_t config_address() const;
    uint32_t ring_size() const { return ring_size_; }

    uint32_t config_page_size() const { return config_page_size_; }
    uint32_t credit_reset_offset() const { return credit_reset_offset_; }
    uint32_t credit_reset_size() const { return credit_reset_size_; }
    const std::vector<uint32_t>& config_page(const CoreCoord& core) const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    CoreCoord sender_core() const { return sender_core_; }
    distributed::MeshDevice* get_device() const { return device_; }

private:
    void setup_buffers(BufferType buffer_type);
    void build_config_pages();
    void write_config_to_device();
    void release_allocations() noexcept;

    uint64_t data_allocation_id_ = 0;
    uint64_t config_allocation_id_ = 0;
    uint32_t data_address_ = 0;
    uint32_t config_address_ = 0;
    distributed::MeshDevice* device_ = nullptr;
    CoreCoord sender_core_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t ring_size_ = 0;
    uint32_t config_page_size_ = 0;
    uint32_t credit_reset_offset_ = 0;
    uint32_t credit_reset_size_ = 0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> config_pages_;
};

/**
 * @brief Create and register the local DFB used to relay a PrefetcherPipe to TRISC.
 *
 * The local DFB borrows the PrefetcherPipe data ring. `prefetcher_pipe_id` must already be
 * Attached on `receiver_core_spec`. Relay entry_size / depth must match this Attach's
 * dense entry_size and `ring_size / entry_size`.
 *
 * Declared here rather than alongside the rest of the host API because it takes a DFB config,
 * which has no public header yet.
 *
 * @return Program-unique host DFB id (distinct from `prefetcher_pipe_id`).
 */
uint32_t CreatePrefetcherPipeRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t prefetcher_pipe_id);

}  // namespace experimental
}  // namespace tt::tt_metal
