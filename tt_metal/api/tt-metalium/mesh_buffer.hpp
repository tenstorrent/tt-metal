// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.hpp"
#include "buffer_constants.hpp"
#include "mesh_coord.hpp"
#include "mesh_device.hpp"
#include "mesh_device_view.hpp"
#include "shape2d.hpp"

namespace tt::tt_metal::distributed {

// Specifies how a buffer is laid out across Memory Banks within a single device.
struct DeviceLocalBufferConfig {
    DeviceAddr page_size = 0;

    // Can be DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE.
    BufferType buffer_type = BufferType::DRAM;

    // Can be INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED.
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;

    // Must be set for sharded buffer layouts.
    std::optional<ShardSpecBuffer> shard_parameters;

    // The direction in which memory for this buffer is allocated.
    bool bottom_up = false;
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

// MeshBuffer allocates a buffer across a mesh of devices according to the specified configuration: either full
// replication, or 2D sharding. The allocation is done in lock-step across all devices in the mesh.
class MeshBuffer : Buffer {
public:
    static std::shared_ptr<MeshBuffer> create(
        const MeshBufferConfig& mesh_buffer_config,
        const DeviceLocalBufferConfig& device_local_layout,
        MeshDevice* mesh_device,
        std::optional<DeviceAddr> address = std::nullopt);

    IDevice* device() const override { return mesh_device_; }
    Allocator* allocator() const override {
        TT_THROW("Function to be implemented.");
        return nullptr;
    }
    DeviceAddr size() const override;
    // Returns true if the MeshBuffer is allocated. Note that MeshBuffer is created in the allocated state; either the
    // destructor or the `deallocate` method deallocate the MeshBuffer.
    bool is_allocated() const override;

    // Returns address of buffer in the first bank
    uint32_t address() const override { return address_; };

    DeviceAddr page_size() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }
    void set_page_size(DeviceAddr page_size) override { TT_THROW("Function to be implemented."); }

    uint32_t num_pages() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }
    uint32_t num_dev_pages() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }

    BufferType buffer_type() const override {
        TT_THROW("Function to be implemented.");
        return BufferType::DRAM;
    }
    CoreType core_type() const override {
        TT_THROW("Function to be implemented.");
        return CoreType::WORKER;
    }

    bool is_l1() const override {
        TT_THROW("Function to be implemented.");
        return false;
    }
    bool is_dram() const override {
        TT_THROW("Function to be implemented.");
        return false;
    }
    bool is_trace() const override {
        TT_THROW("Function to be implemented.");
        return false;
    }

    bool is_valid_region(const BufferRegion& region) const override {
        TT_THROW("Function to be implemented.");
        return false;
    }
    bool is_valid_partial_region(const BufferRegion& region) const override {
        TT_THROW("Function to be implemented.");
        return false;
    }

    TensorMemoryLayout buffer_layout() const override {
        TT_THROW("Function to be implemented.");
        return TensorMemoryLayout::INTERLEAVED;
    }

    bool bottom_up() const override {
        TT_THROW("Function to be implemented.");
        return false;
    }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const override {
        TT_THROW("Function to be implemented.");
        return {0, 0};
    }

    DeviceAddr page_address(uint32_t bank_id, uint32_t page_index) const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }

    DeviceAddr bank_local_page_address(uint32_t bank_id, uint32_t page_index) const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }
    uint32_t alignment() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }
    DeviceAddr aligned_page_size() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }
    DeviceAddr aligned_size() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }
    DeviceAddr aligned_size_per_bank() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }

    // SHARDED API STARTS HERE
    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    DeviceAddr sharded_page_address(uint32_t bank_id, uint32_t page_index) const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }

    ShardSpecBuffer shard_spec() const override {
        TT_THROW("Function to be implemented.");
        for (auto& [coord, device_buffer] : buffers_) {
            return device_buffer->shard_spec();
        }
    }
    void set_shard_spec(const ShardSpecBuffer& shard_spec) override { TT_THROW("Function to be implemented."); }

    std::optional<uint32_t> num_cores() const override {
        TT_THROW("Function to be implemented.");
        return std::nullopt;
    }

    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping() override {
        TT_THROW("Function to be implemented.");
        for (auto& [coord, device_buffer] : buffers_) {
            return device_buffer->get_buffer_page_mapping();
        }
    }

    std::optional<SubDeviceId> sub_device_id() const override {
        TT_THROW("Function to be implemented.");
        return std::nullopt;
    }
    std::optional<SubDeviceManagerId> sub_device_manager_id() const override {
        TT_THROW("Function to be implemented.");
        return std::nullopt;
    }

    size_t unique_id() const override {
        TT_THROW("Function to be implemented.");
        return 0;
    }

    // Deallocates the MeshBuffer.
    // TODO: Re-consider a need for explicit deallocation methods, as opposed to relying on RAII to clean up the
    // resources.
    void deallocate() override;

    // MeshBuffer-specific APIs
    MeshDevice* mesh_device() const { return mesh_device_; }
    DeviceAddr device_local_size() const { return device_local_size_; }

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const { return config_; }

    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const { return device_local_config_; }

    std::shared_ptr<Buffer> get_device_buffer(const Coordinate& device_coord) const;
    std::shared_ptr<Buffer> get_device_buffer(const MeshCoordinate& device_coord) const;
    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;

private:
    // Creates an owning `MeshBuffer`, backed by an allocation made through `backing_buffer`.
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device,
        std::shared_ptr<Buffer> backing_buffer) :
        buffers_(SimpleMeshShape(mesh_device->shape()), nullptr),
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device),
        address_(backing_buffer->address()),
        device_local_size_(device_local_size),
        state_(OwnedBufferState{std::move(backing_buffer)}) {}

    // Creates a non-owning `MeshBuffer` as "view" over an existing `address`.
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr address,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        buffers_(SimpleMeshShape(mesh_device->shape()), /*fill_value=*/nullptr),
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device),
        address_(address),
        device_local_size_(device_local_size),
        state_(ExternallyOwnedState{}) {}

    void initialize_device_buffers();
    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    MeshDevice* mesh_device_ = nullptr;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;

    MeshContainer<std::shared_ptr<Buffer>> buffers_;

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
};

}  // namespace tt::tt_metal::distributed
