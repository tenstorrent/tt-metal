#pragma once

#include <chrono>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

static constexpr uint32_t trace_iters = 10;

struct MeshDescriptor {
    std::optional<distributed::MeshShape> mesh_shape;
    std::optional<distributed::MeshCoordinate> mesh_offset;

    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;

    int num_cqs = 1;
    uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    uint32_t trace_region_size = 1u << 20;  // statically allocate 1 MiB for trace region
    uint32_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
};

struct FabricTestDescriptor {
    uint32_t mesh_id = uint32_t(0);
    ChipId src_chip = 0;
    ChipId dst_chip = 1;
    uint32_t page_size = 0;
    CoreCoord sender_core = {0, 0};
    CoreCoord receiver_core = {0, 0};
};

// Helper to create a sharded buffer config.
// cluster_axis: 0 = shard along height (rows), 1 = shard along width (cols)
inline distributed::ShardedBufferConfig create_sharded_buffer_config(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Shape2D& global_shape,
    uint32_t element_size_bytes,
    int cluster_axis = 0) {
    // Calculate shard shape based on cluster_axis
    Shape2D shard_shape(0, 0);
    if (cluster_axis == 0) {
        // Shard along height (rows) - divide height by num_rows
        shard_shape = Shape2D(
            global_shape.height() / mesh_device->num_rows(),
            global_shape.width()  // width is preserved
        );
    } else {
        // Shard along width (cols) - divide width by num_cols
        shard_shape = Shape2D(
            global_shape.height(),  // height is preserved
            global_shape.width() / mesh_device->num_cols());
    }

    // Calculate global buffer size
    uint32_t global_size = global_shape.height() * global_shape.width() * element_size_bytes;

    return distributed::ShardedBufferConfig{
        .global_size = global_size,
        .global_buffer_shape = global_shape,
        .shard_shape = shard_shape,
        .shard_orientation = (cluster_axis == 0) ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR};
}

inline distributed::MeshCoordinate extract_coord_of_phy_id(
    std::shared_ptr<distributed::MeshDevice> mesh_device, ChipId phy_id) {
    auto view = mesh_device->get_view();
    for (const auto& c : distributed::MeshCoordinateRange(view.shape())) {
        // check if current device id is given phy_id
        if (view.get_device(c)->id() == phy_id) {
            return c;
        }
    }
    TT_FATAL(false, "Physical chip {} is not part of this MeshDevice", phy_id);
    return distributed::MeshCoordinate(0);
};

inline double run_recv_send_workload_trace(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    distributed::MeshWorkload& recv_workload,
    distributed::MeshWorkload& send_workload) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // warm-up run
    distributed::EnqueueMeshWorkload(cq, recv_workload, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(cq, send_workload, /*blocking=*/true);

    // trace iteration
    auto trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < trace_iters; ++i) {
        distributed::EnqueueMeshWorkload(cq, recv_workload, /*blocking=*/false);
        distributed::EnqueueMeshWorkload(cq, send_workload, /*blocking=*/false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    auto t0 = std::chrono::steady_clock::now();
    mesh_device->replay_mesh_trace(cq.id(), trace_id, /*blocking=*/false);
    distributed::Finish(cq);
    auto t1 = std::chrono::steady_clock::now();
    mesh_device->release_mesh_trace(trace_id);

    return std::chrono::duration<double>(t1 - t0).count();
}

inline double run_recv_send_workload_once(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    distributed::MeshWorkload& recv_workload,
    distributed::MeshWorkload& send_workload) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    auto t0 = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, recv_workload, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(cq, send_workload, /*blocking=*/false);
    distributed::Finish(cq);
    auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(t1 - t0).count();
}

inline std::vector<uint32_t> make_src_data(size_t num_words) {
    std::vector<uint32_t> tx(num_words);
    for (size_t i = 0; i < num_words; ++i) {
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    return tx;
}
}  // namespace tt::tt_metal
