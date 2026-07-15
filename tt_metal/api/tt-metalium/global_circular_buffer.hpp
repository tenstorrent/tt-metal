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
// Impl-only DRISC L1 arena types; held by GlobalCircularBuffer as a private
// shared_ptr so the GCB doesn't have to include the impl arena header.
class DriscL1Allocation;

namespace experimental {

// Forward declarations for the experimental DRAM-sender extension defined in
// tt-metalium/experimental/global_circular_buffer.hpp. The DRAM-sender feature is an
// opt-in mode that is not part of the public GlobalCircularBuffer API surface; existing
// callers continue to see the original public interface unchanged.
class GlobalCircularBuffer;
enum class SenderCoreType : uint8_t;
namespace global_circular_buffer_dram_sender {
struct GlobalCircularBufferDramSenderInternals;
}  // namespace global_circular_buffer_dram_sender

class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1);

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
    GlobalCircularBuffer(
        distributed::MeshDevice* mesh_device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type,
        DramSenderTag);

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
    // per-mesh DriscL1Arena. Held via shared_ptr so copies of the GCB share the same
    // backing range; released when the last GCB copy goes out of scope. Empty for
    // worker-sender GCBs.
    std::shared_ptr<::tt::tt_metal::DriscL1Allocation> drisc_sender_state_alloc_;

    friend struct global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals;
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
    BufferType buffer_type = BufferType::L1);

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
