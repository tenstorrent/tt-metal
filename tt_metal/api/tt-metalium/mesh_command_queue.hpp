// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <umd/device/tt_core_coordinates.h>

namespace tt {
namespace tt_metal {
class IDevice;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
namespace distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace distributed
struct ProgramCommandSequence;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {

class MeshEvent;
class MeshTraceDescriptor;
struct MeshBufferReadDescriptor;
struct MeshReadEventDescriptor;
struct MeshCoreDataReadDescriptor;

using MeshCompletionReaderVariant =
    std::variant<MeshBufferReadDescriptor, MeshReadEventDescriptor, MeshCoreDataReadDescriptor>;

// THREAD SAFETY: All methods are thread safe.
class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
protected:
    MeshDevice* mesh_device_ = nullptr;
    uint32_t id_ = 0;

    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id) : mesh_device_(mesh_device), id_(id) {}

public:
    MeshCommandQueue(const MeshCommandQueue& other) = delete;
    MeshCommandQueue& operator=(const MeshCommandQueue& other) = delete;

    virtual ~MeshCommandQueue() = default;

    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    virtual std::optional<MeshTraceId> trace_id() const = 0;
    virtual WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) = 0;
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;

    // Specifies host data to be written to or read from a MeshBuffer shard.
    struct ShardDataTransfer {
        MeshCoordinate shard_coord;
        void* host_data = nullptr;
        std::optional<BufferRegion> region;
    };

    // MeshBuffer Write APIs
    virtual void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer,
        const void* host_data,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt) = 0;
    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
    virtual void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking) = 0;
    virtual void enqueue_write(
        const std::shared_ptr<MeshBuffer>& mesh_buffer, const DistributedHostBuffer& host_buffer, bool blocking) = 0;

    /**
     * @brief Enqueue a write operation to a sub-grid of devices with data format conversion
     *
     * Writes host data to a specific sub-grid range of devices within a MeshBuffer,
     * performing data format conversion from the source format to the target buffer format.
     * This allows writing data in one format (e.g., Float32) to a buffer that stores
     * data in a different format (e.g., BFloat16).
     *
     * Only a limited number of data formats are supported for conversion, including:
     * - Float32 to Float16_b
     * - Identity conversion (Float32 to Float32, Float16_b to Float16_b, etc.)
     *
     * @param buffer The MeshBuffer to write data to
     * @param data_format The target data format for the buffer
     * @param host_data Pointer to the source data on the host
     * @param src_data_format The source data format of the host data
     * @param device_range If the buffer is replicated, the range of mesh coordinates defining the sub-grid to write to.
     * If the buffer is sharded, this is ignored.
     * @param blocking If true, the operation blocks until completion; if false, it's asynchronous
     * @param region Optional buffer region to write to; if not provided, writes to the entire buffer area
     *
     * @note If the identity conversion is not used, the region specified must be a multiple of the buffer data format
     * element size, and the corresponding source size must be a multiple of the source data format element size.
     */
    virtual void enqueue_write_shard_to_sub_grid_with_conversion(
        const MeshBuffer& buffer,
        tt::DataFormat data_format,
        const void* host_data,
        tt::DataFormat src_data_format,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt) = 0;

    /**
     * @brief Enqueue a write operation to multiple shards with data format conversion
     *
     * Writes data to multiple specific shards of a MeshBuffer, performing data format
     * conversion from the source format to the target buffer format. Each shard can
     * receive different data, allowing for fine-grained control over distributed data placement.
     *
     * Only a limited number of data formats are supported for conversion, including:
     * - Float32 to Float16_b
     * - Identity conversion (Float32 to Float32, Float16_b to Float16_b, etc.)
     *
     * @param mesh_buffer Shared pointer to the MeshBuffer to write data to
     * @param data_format The target data format for the buffer
     * @param shard_data_transfers Vector of ShardDataTransfer objects, each specifying
     *                            a shard coordinate, host data pointer, and optional buffer region
     * @param src_data_format The source data format of all host data
     * @param blocking If true, the operation blocks until completion; if false, it's asynchronous
     *
     * @note All host data in the shard_data_transfers must be in the same source format
     * @note The data conversion is performed during the write operation for each shard
     *
     * @note If the identity conversion is not used, each region specified in the shard_data_transfers must be a
     * multiple of the buffer data format element size, and the corresponding source size must be a multiple of the
     * source data format element size.
     * @see ShardDataTransfer for details on specifying per-shard data and regions
     */
    virtual void enqueue_write_shards_with_conversion(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        tt::DataFormat data_format,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        tt::DataFormat src_data_format,
        bool blocking) = 0;

    /**
     * @brief Enqueue a write operation to an entire MeshBuffer with data format conversion
     *
     * Writes host data to all devices in a MeshBuffer, performing data format conversion from the source format to the
     * target buffer format. This operation writes the data using the sharding specification in the buffer.
     *
     * Only a limited number of data formats are supported for conversion, including:
     * - Float32 to Float16_b
     * - Identity conversion (Float32 to Float32, Float16_b to Float16_b, etc.)
     *
     * @param buffer Shared pointer to the MeshBuffer to write data to
     * @param data_format The target data format for the buffer
     * @param host_data Pointer to the source data on the host
     * @param src_data_format The source data format of the host data
     * @param blocking If true, the operation blocks until completion; if false, it's asynchronous
     *
     * @note The data conversion is performed during the write operation
     */
    virtual void enqueue_write_mesh_buffer_with_conversion(
        const std::shared_ptr<MeshBuffer>& buffer,
        tt::DataFormat data_format,
        const void* host_data,
        tt::DataFormat src_data_format,
        bool blocking) = 0;

    /**
     * @brief Enqueue a write operation using a DistributedHostBuffer with data format conversion
     *
     * Writes data from a DistributedHostBuffer to a MeshBuffer, performing data format
     * conversion from the source format to the target buffer format. The DistributedHostBuffer
     * provides per-shard data distribution, allowing different data to be written to each
     * shard in the mesh while maintaining format conversion capabilities.
     *
     * Only a limited number of data formats are supported for conversion, including:
     * - Float32 to Float16_b
     * - Identity conversion (Float32 to Float32, Float16_b to Float16_b, etc.)
     *
     * @param mesh_buffer Shared pointer to the MeshBuffer to write data to
     * @param data_format The target data format for the buffer
     * @param host_buffer Reference to the DistributedHostBuffer containing the source data
     *                    with per-shard data distribution
     * @param src_data_format The source data format of the data in the DistributedHostBuffer
     * @param blocking If true, the operation blocks until completion; if false, it's asynchronous
     *
     * @see DistributedHostBuffer for details on distributed data management
     */
    virtual void enqueue_write_with_conversion(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        tt::DataFormat data_format,
        const DistributedHostBuffer& host_buffer,
        tt::DataFormat src_data_format,
        bool blocking) = 0;

    // MeshBuffer Read APIs
    virtual void enqueue_read_mesh_buffer(
        void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) = 0;
    virtual void enqueue_read_shards(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking) = 0;
    // TODO: does "enqueue" make sense anymore? Return the object by value instead.
    virtual void enqueue_read(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        DistributedHostBuffer& host_buffer,
        const std::optional<std::unordered_set<MeshCoordinate>>& shards,
        bool blocking) = 0;

    virtual MeshEvent enqueue_record_event(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual MeshEvent enqueue_record_event_to_host(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;
    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) = 0;
    virtual void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;
    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;
};

}  // namespace tt::tt_metal::distributed
