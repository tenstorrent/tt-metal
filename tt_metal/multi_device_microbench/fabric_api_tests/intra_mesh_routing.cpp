// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// metalium headers
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/shape2d.hpp>

// umd headers
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include <llrt/tt_cluster.hpp>

#include "llrt.hpp"
#include "system_mesh.hpp"
#include "impl/context/metal_context.hpp"
#include "hostdevcommon/common_values.hpp"
#include "common/tt_backend_api_types.hpp"

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Helper function to create a sharded buffer config similar to Python's mesh_mapper
// cluster_axis: 0 = shard along height (rows), 1 = shard along width (cols)
distributed::ShardedBufferConfig create_sharded_buffer_config(
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

struct TestConfig {
    // If specified, the fixture will open a mesh device with the specified shape and offset.
    // Otherwise, SystemMesh shape with zero offset will be used.
    std::optional<distributed::MeshShape> mesh_shape;
    std::optional<distributed::MeshCoordinate> mesh_offset;

    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;

    int num_cqs = 1;
    uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    uint32_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
    tt_fabric::FabricConfig fabric_config = tt_fabric::FabricConfig::DISABLED;
};

int main() {
    // Initialize test config
    TestConfig config{};

    // Set system mesh shape as default
    config.mesh_shape = distributed::SystemMesh::instance().shape();

    // Extract core config
    auto cluster_type = MetalContext::instance().get_cluster().get_cluster_type();
    bool is_n300_or_t3k_cluster = cluster_type == ClusterType::T3K or cluster_type == ClusterType::N300;
    auto core_type =
        (config.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

    // Create a mesh device
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(config.mesh_shape),
        config.l1_small_size,
        config.trace_region_size,
        config.num_cqs,
        core_type,
        {},
        config.worker_l1_size);

    // Mesh command queue and program setup
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Core range setup
    constexpr CoreCoord core0 = {0, 0};
    constexpr CoreCoord core1 = {0, 1};
    const auto core0_physical_coord = mesh_device->worker_core_from_logical_core(core0);
    const auto core1_physical_coord = mesh_device->worker_core_from_logical_core(core1);

    CoreRange sem_core_range = CoreRange(core0, core1);

    // Check if the environment variable for kernels print is set
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            stderr,
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(0,1) to see the output of "
            "the Data Movement kernels. Command: export TT_METAL_DPRINT_CORES=(0,0),(0,1)\n");
    }

    // Input data preparation
    constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    // Create sharded buffers instead of replicated buffers
    // For a single tile, we'll shard it across the mesh (though for a single tile this is a bit unusual)
    // In practice, you'd have a larger tensor. Here we use cluster_axis=0 to shard along rows
    const Shape2D global_buffer_shape = Shape2D(tt::constants::TILE_HEIGHT * 4, tt::constants::TILE_WIDTH * 2);
    auto sharded_buffer_config =
        create_sharded_buffer_config(mesh_device, global_buffer_shape, sizeof(uint16_t), 0 /* cluster_axis */);

    auto src_dram_buffer = distributed::MeshBuffer::create(sharded_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(sharded_buffer_config, dram_config, mesh_device.get());

    // Core synchronization semaphore setup
    const uint32_t sem_id = CreateSemaphore(program, sem_core_range, 0);

    // Source data preparation and DRAM transfer
    const uint16_t input_data = 14;  // Example input data
    std::vector<uint16_t> src_vec(1, input_data);
    distributed::git(cq, src_dram_buffer, src_vec, false);

    // L1 circular buffer setup
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::UInt16}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::UInt16}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src1_config);

    // Kernels setup
    // Core 0 kernels
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle core0_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fabric_api_tests/kernels/dataflow/reader0.cpp",
        core0,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    KernelHandle core0_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fabric_api_tests/kernels/dataflow/writer0.cpp",
        core0,
        tt::tt_metal::WriterDataMovementConfig{{src0_cb_index, src1_cb_index}});

    // Core 1 kernels
    KernelHandle core1_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fabric_api_tests/kernels/dataflow/reader1.cpp",
        core1,
        tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, src1_cb_index}});
    std::vector<uint32_t> writer_compile_time_args = {src1_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle core1_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fabric_api_tests/kernels/dataflow/writer1.cpp",
        core1,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    // Runtime args setup
    SetRuntimeArgs(program, core0_reader_kernel_id, core0, {src_dram_buffer->address()});
    SetRuntimeArgs(program, core0_writer_kernel_id, core0, {core1_physical_coord.x, core1_physical_coord.y, sem_id});
    SetRuntimeArgs(program, core1_reader_kernel_id, core1, {core0_physical_coord.x, core0_physical_coord.y, sem_id});
    SetRuntimeArgs(program, core1_writer_kernel_id, core1, {dst_dram_buffer->address()});

    // Program enqueue (non-blocking). Wait for completion before reading back.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<uint16_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    fmt::print("Result = {} : Expected = {}\n", result_vec[0], input_data);

    // Teardown mesh device
    mesh_device->close();
    mesh_device.reset();
    if (config.fabric_config != tt_fabric::FabricConfig::DISABLED) {
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
    }
}
