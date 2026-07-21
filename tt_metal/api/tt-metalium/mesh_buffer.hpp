// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <atomic>
#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal::distributed {

// Specifies how a buffer is laid out across Memory Banks within a single device.
struct DeviceLocalBufferConfig {
    DeviceAddr page_size = 0;

    // Can be DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE.
    BufferType buffer_type = BufferType::DRAM;

    BufferShardingArgs sharding_args;

    // The direction in which memory for this buffer is allocated.
    std::optional<bool> bottom_up;

    // Optional: Specify the worker sub device this buffer will be allocated on
    std::optional<SubDeviceId> sub_device_id = std::nullopt;
};

// Specifies MeshBuffer that is replicated across the virtual mesh.
// Write APIs for replicated buffers will write the same data to all devices in the virtual mesh.
struct ReplicatedBufferConfig {
    // Each device will get a buffer of this size.
    DeviceAddr size = 0;
};

// Specifies sharded MeshBuffer.
// Write APIs for sharded buffers will split the data so that each device in the virtual mesh will only get a fraction
// of the data.
struct ShardedBufferConfig {
    // Note: Only 2D sharding and replication is supported by the APIs exposed through this struct.
    // This interface will likely change over time depending on the status of native ND sharding.
    // Global buffer size. Each device will get a fraction of this size.
    DeviceAddr global_size = 0;

    // Global shape of the buffer; at metal-level, we expect the shape to be aligned with the mesh shape.
    Shape2D global_buffer_shape = {0, 0};

    // Shard shape, sent to each device.
    Shape2D shard_shape = {0, 0};

    // Orientation of the shards in a mesh.
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    // Computes the number of bytes per datum in the sharded buffer.
    uint32_t compute_datum_size_bytes() const;

    std::pair<bool, bool> replicated_dims() const;

    Shape2D physical_shard_shape() const;
};

enum class MeshBufferLayout : uint8_t { REPLICATED, SHARDED };
using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

class MeshBuffer;

}  // namespace tt::tt_metal::distributed

// Forward declaration for experimental per-core allocation friend
namespace tt::tt_metal::experimental::per_core_allocation {
std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_on_single_device(
    const tt::tt_metal::distributed::MeshBufferConfig& mesh_buffer_config,
    const tt::tt_metal::distributed::DeviceLocalBufferConfig& device_local_config,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::distributed::MeshCoordinate& coord);
}  // namespace tt::tt_metal::experimental::per_core_allocation

namespace tt::tt_metal::distributed {

// MeshBuffer allocates a buffer across a mesh of devices according to the specified configuration: either full
// replication, or 2D sharding. The allocation is done in lock-step across all devices in the mesh.
class MeshBuffer {
public:
    class PendingEventRegistration {
    public:
        PendingEventRegistration(const PendingEventRegistration&) = delete;
        PendingEventRegistration& operator=(const PendingEventRegistration&) = delete;
        PendingEventRegistration(PendingEventRegistration&& other) noexcept;
        PendingEventRegistration& operator=(PendingEventRegistration&& other) noexcept;
        ~PendingEventRegistration();

        // Publishes the completion event and releases this registration. The event
        // must be ordered after all work that references the buffer.
        void publish(const MeshEvent& event);

    private:
        explicit PendingEventRegistration(const MeshBuffer* buffer) : buffer_(buffer) {}
        void release();

        const MeshBuffer* buffer_ = nullptr;
        friend class MeshBuffer;
    };

    static std::shared_ptr<MeshBuffer> create(
        const MeshBufferConfig& mesh_buffer_config,
        const DeviceLocalBufferConfig& device_local_config,
        MeshDevice* mesh_device,
        std::optional<DeviceAddr> address = std::nullopt);
    ~MeshBuffer();

    // MeshBuffer manages device memory and owns the backing allocation. Copying would create
    // multiple owners of the same device memory, leading to double-free on destruction.
    MeshBuffer(const MeshBuffer&) = delete;
    MeshBuffer& operator=(const MeshBuffer&) = delete;
    MeshBuffer(MeshBuffer&& other) noexcept;
    MeshBuffer& operator=(MeshBuffer&& other) noexcept;

    // Returns true if the MeshBuffer is allocated. Note that MeshBuffer is created in the allocated state; either the
    // destructor or the `deallocate` method deallocate the MeshBuffer.
    bool is_allocated() const;

    // Deallocates the MeshBuffer.
    // TODO: Re-consider a need for explicit deallocation methods, as opposed to relying on RAII to clean up the
    // resources.
    void deallocate();

    // Throws an exception if the corresponding MeshDevice is already deallocated
    MeshDevice* device() const;
    DeviceAddr size() const;
    DeviceAddr device_local_size() const { return device_local_size_; }
    DeviceAddr address() const { return address_; };

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const { return config_; }

    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const { return device_local_config_; }

    Buffer* get_device_buffer(const MeshCoordinate& device_coord) const;

    // TODO: Remove this method, once there is no need to interop MeshBuffer with Buffer.
    // The reference buffer allows "casting" the MeshBuffer to a buffer allocated on a
    // single device. This allows users of this object that only need to query single device
    // attributes to do so without having to keep track of MeshDevice attributes.
    Buffer* get_reference_buffer() const;
    // The backing buffer represents the buffer object keeping the MeshBuffer alive/allocated
    // at its specific address. The backing buffer will not be populated if an address was passed
    // into the creation API.
    Buffer* get_backing_buffer() const;

    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;
    uint32_t page_size() const { return device_local_config_.page_size; }
    uint32_t num_pages() const { return page_size() == 0 ? 0 : device_local_size_ / page_size(); }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pending Event Tracking for Multi-CQ Safety
    //
    // In multi-CQ scenarios, operations on one CQ may reference a buffer while another CQ
    // deallocates and reallocates the same address. To prevent this race, we track the latest
    // pending full-mesh event per supported CQ as a packed (quiesce_epoch, event_id) uint64_t.
    // Deallocation rejects unsupported CQ IDs rather than silently skipping their events.
    // The buffer's address cannot be safely reused until all pending events complete.

    // Acquires an in-flight publisher before work using this buffer is dispatched.
    // Deallocation blocks until every acquired registration publishes an event or
    // is released. Returns nullopt once deallocation has started.
    [[nodiscard]] std::optional<PendingEventRegistration> try_acquire_pending_event_registration() const;

    // Waits for all pending events to complete. Called during deallocation to ensure
    // no operations are still in-flight before releasing the buffer address.
    void wait_for_pending_events();

    // Returns true if there are pending events that must complete before deallocation.
    bool has_pending_events() const;

private:
    // Creates an owning `MeshBuffer`, backed by an allocation made through `backing_buffer`.
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device,
        std::shared_ptr<Buffer> backing_buffer) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(backing_buffer->address()),
        device_local_size_(device_local_size),
        buffers_(MeshShape(mesh_device->shape())),
        state_(OwnedBufferState{std::move(backing_buffer)}) {}

    // Creates a non-owning `MeshBuffer` as "view" over an existing `address`.
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr address,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(address),
        device_local_size_(device_local_size),
        buffers_(MeshShape(mesh_device->shape())),
        state_(ExternallyOwnedState{}) {}

    void initialize_device_buffers();
    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    std::weak_ptr<MeshDevice> mesh_device_;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;

    DistributedMeshContainer<std::shared_ptr<Buffer>> buffers_;

    // `MeshBufferState` specifies the state of the MeshBuffer. It can either be:
    // 1. Owned - a single device buffer is responsible for providing the address for the entire mesh buffer.
    // 2. Externally owned - the MeshBuffer was created as a view over an existing address.
    // 3. Deallocated - the MeshBuffer is in the deallocated state.
    struct OwnedBufferState {
        std::shared_ptr<Buffer> backing_buffer;
    };
    struct ExternallyOwnedState {};
    struct DeallocatedState {};
    using MeshBufferState = std::variant<OwnedBufferState, ExternallyOwnedState, DeallocatedState>;
    MeshBufferState state_;

    // Pending event tracking for multi-CQ safety.
    // Stores the latest in-flight full-mesh event per supported CQ. 0 = no pending event.
    // Each slot stores (quiesce_epoch << 32 | event_id) so that
    // wait_for_pending_events() can detect stale events from a previous quiesce
    // cycle without risking an infinite spin on reset counters.
    // IDs are monotonically increasing within a cycle; CAS-updated so only the
    // latest is kept.
    // This currently supports the two-CQ FD configurations used by Wormhole/Blackhole;
    // add_pending_event() fails fast if a runtime CQ id is outside this fixed array.
    //
    // Each slot is padded to a full cache line (64 bytes) to prevent false sharing:
    // CQ0 and CQ1 are written by different dispatch threads; without padding they
    // would share a cache line and cause unnecessary coherence traffic.
    static constexpr size_t kMaxMeshCQs = 2;
    struct alignas(64) CacheLinePaddedEventId {
        std::atomic<uint64_t> value{};
    };
    mutable std::array<CacheLinePaddedEventId, kMaxMeshCQs> pending_event_ids_{};

    // Pack/unpack helpers for the 64-bit (epoch, event_id) encoding.
    static uint64_t pack_epoch_event(uint32_t epoch, uint32_t event_id) {
        return (static_cast<uint64_t>(epoch) << 32) | event_id;
    }
    static uint32_t unpack_epoch(uint64_t packed) { return static_cast<uint32_t>(packed >> 32); }
    static uint32_t unpack_event_id(uint64_t packed) { return static_cast<uint32_t>(packed); }

    // Adds an event while a PendingEventRegistration is active. Publishing the
    // event before releasing the registration makes it visible to deallocation
    // before the pending-event drain can run.
    void add_pending_event(const MeshEvent& event) const;
    void release_pending_event_registration() const;
    void wait_for_pending_event_registrations() const;

    // Terminal close gate for device-memory lifetime. A publisher increments the
    // count before dispatch and rechecks the gate. Deallocation closes the gate,
    // waits for the count to reach zero, drains events, then releases the address.
    std::atomic_flag deallocation_started_ = ATOMIC_FLAG_INIT;
    mutable std::atomic<uint32_t> active_event_publishers_{0};

    friend std::shared_ptr<MeshBuffer> tt::tt_metal::experimental::per_core_allocation::create_on_single_device(
        const tt::tt_metal::distributed::MeshBufferConfig&,
        const tt::tt_metal::distributed::DeviceLocalBufferConfig&,
        tt::tt_metal::distributed::MeshDevice*,
        const tt::tt_metal::distributed::MeshCoordinate&);
};

class AnyBuffer {
public:
    AnyBuffer() = default;
    static AnyBuffer create(
        const tt::tt_metal::ShardedBufferConfig& config, std::optional<uint64_t> address = std::nullopt);
    static AnyBuffer create(
        const tt::tt_metal::InterleavedBufferConfig& config, std::optional<uint64_t> address = std::nullopt);

    Buffer* get_buffer() const;
    bool is_mesh_buffer() const;
    std::shared_ptr<MeshBuffer> get_mesh_buffer() const;

private:
    AnyBuffer(std::shared_ptr<Buffer> buffer);
    AnyBuffer(std::shared_ptr<MeshBuffer> buffer);

    Buffer* buffer_ = nullptr;
    std::variant<std::shared_ptr<Buffer>, std::shared_ptr<distributed::MeshBuffer>> holder_;
};

}  // namespace tt::tt_metal::distributed
