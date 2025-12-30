// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Basically, this is clone of the microbenchmark program `run_unicast_once.cpp`
// Check it for more details.

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/tt_backend_api_types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/mesh_device.hpp>
#include "system_mesh.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

// custom util
#include "buffer_utils.hpp"

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

struct MeshDescriptor {
    // If specified, the fixture will open a mesh device with the specified shape and offset.
    // Otherwise, SystemMesh shape with zero offset will be used.
    std::optional<distributed::MeshShape> mesh_shape;
    std::optional<distributed::MeshCoordinate> mesh_offset;

    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;

    int num_cqs = 1;
    uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
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

inline std::vector<uint32_t> make_src_data(size_t num_words) {
    std::vector<uint32_t> tx(num_words);
    for (size_t i = 0; i < num_words; ++i) {
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    return tx;
}

int main() {
    // Check number of devices
    auto num_devices = GetNumAvailableDevices();
    TT_FATAL(
        num_devices > 1, "Currently {} devices are available. Cannot test number of devices under two.", num_devices);

    // ------------ Setup MeshDevice ------------
    // Initialize test mesh_desc
    MeshDescriptor mesh_desc{};

    // Set system mesh shape as default
    mesh_desc.mesh_shape = distributed::SystemMesh::instance().shape();

    // Extract core mesh_desc
    auto cluster_type = MetalContext::instance().get_cluster().get_cluster_type();
    bool is_n300_or_t3k_cluster = cluster_type == ClusterType::T3K or cluster_type == ClusterType::N300;
    auto core_type =
        (mesh_desc.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

    // Create a mesh device
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(mesh_desc.mesh_shape),
        mesh_desc.l1_small_size,
        mesh_desc.trace_region_size,
        mesh_desc.num_cqs,
        core_type,
        {},
        mesh_desc.worker_l1_size);

    // ------------ Setup Fabric ------------
    // Initialize fabric test descriptor
    FabricTestDescriptor fabric_desc{};

    // Get control plance instance
    const auto& control_plane = MetalContext::instance().get_control_plane();

    // Set fabric node ids. One is composed of {MeshId, ChipId}
    // MeshId is same for both src and dst because it is intra-routing test.
    tt_fabric::FabricNodeId src_fabric_node{tt_fabric::MeshId{fabric_desc.mesh_id}, fabric_desc.src_chip};
    tt_fabric::FabricNodeId dst_fabric_node{tt_fabric::MeshId{fabric_desc.mesh_id}, fabric_desc.dst_chip};

    auto src_phy_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node);
    auto dst_phy_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node);

    auto src_dev = tt::tt_metal::detail::GetActiveDevice(src_phy_id);
    auto dst_dev = tt::tt_metal::detail::GetActiveDevice(dst_phy_id);
    TT_FATAL(src_dev && dst_dev, "Both devices should be valid.");

    auto extract_coord_of_phy_id = [&mesh_device](ChipId phy_id) -> distributed::MeshCoordinate {
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

    distributed::MeshCoordinate src_mesh_coord = extract_coord_of_phy_id(src_phy_id);
    distributed::MeshCoordinate dst_mesh_coord = extract_coord_of_phy_id(dst_phy_id);

    // ------------ Setup MeshBuffer ------------
    constexpr uint32_t page_size = sizeof(uint32_t) * tt::constants::TILE_HW;
    constexpr uint32_t buffer_size = 1u << 20;  // 1MiB

    distributed::DeviceLocalBufferConfig dram_local_config{
        .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig global_config{.size = buffer_size};
    auto src_buf = distributed::MeshBuffer::create(global_config, dram_local_config, mesh_device.get());
    auto dst_buf = distributed::MeshBuffer::create(global_config, dram_local_config, mesh_device.get());

    const auto num_words = buffer_size / 4;
    auto tx_send_data = make_src_data(num_words);
    std::vector<uint32_t> zeros(num_words, 0u);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // blocking write data to buffers.
    distributed::WriteShard(cq, src_buf, tx_send_data, src_mesh_coord, /*blocking=*/true);
    distributed::WriteShard(cq, dst_buf, zeros, dst_mesh_coord, /*blocking=*/true);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // ------------ Setup Workloads ------------
    // Setup receiver program
    Program receiver_program = CreateProgram();
    auto rx_core_range_set = CoreRange(fabric_desc.receiver_core, fabric_desc.receiver_core);

    static GlobalSemaphore global_sema_a = CreateGlobalSemaphore(mesh_device.get(), rx_core_range_set, /*init_val*/ 0);
    static GlobalSemaphore global_sema_b = CreateGlobalSemaphore(mesh_device.get(), rx_core_range_set, /*init_val*/ 0);

    constexpr const char* KERNEL_DIR = "tt_metal/multi_device_microbench/fabric_api_tests/kernels/dataflow/";
    auto rx_wait_kernel = CreateKernel(
        receiver_program,
        std::string(KERNEL_DIR) + "unicast_rx.cpp",
        fabric_desc.receiver_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(receiver_program, rx_wait_kernel, fabric_desc.receiver_core, /*args*/ {global_sema_a.address(), 1u});

    // Setup sender program
    Program sender_program = CreateProgram();
    auto tx_core_range_set = CoreRange(fabric_desc.sender_core, fabric_desc.sender_core);
    auto num_pages = tt::div_up(buffer_size, page_size);
    constexpr auto CB_ID = tt::CBIndex::c_0;

    // CB to buffer local dram read
    auto cb_cfg =
        CircularBufferConfig(8 * page_size, {{CB_ID, tt::DataFormat::Float16}}).set_page_size(CB_ID, page_size);
    CreateCircularBuffer(sender_program, fabric_desc.sender_core, cb_cfg);

    std::vector<uint32_t> reader_cta;
    TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(1u /*SRC_IS_DRAM*/);
    reader_cta.push_back(num_pages);
    reader_cta.push_back(page_size);

    auto reader_kernel = CreateKernel(
        sender_program,
        std::string(KERNEL_DIR) + "unicast_tx_reader_to_cb.cpp",
        fabric_desc.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta});
    tt::tt_metal::SetRuntimeArgs(
        sender_program, reader_kernel, fabric_desc.sender_core, {(uint32_t)src_buf->address()});

    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(num_pages);
    writer_cta.push_back(page_size);

    auto writer_kernel = tt::tt_metal::CreateKernel(
        sender_program,
        std::string(KERNEL_DIR) + "unicast_tx_writer_cb_to_dst.cpp",
        fabric_desc.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta});

    // find available links
    auto links = tt_fabric::get_forwarding_link_indices(src_fabric_node, dst_fabric_node);
    TT_FATAL(!links.empty(), "Need at least one available link from src to dst.");
    uint32_t link_to_use = links[0];

    CoreCoord receiver_coord = dst_dev->worker_core_from_logical_core(fabric_desc.receiver_core);
    std::vector<uint32_t> writer_rta = {
        (uint32_t)dst_buf->address(),      // 0: dst_base (receiver DRAM offset)
        (uint32_t)fabric_desc.mesh_id,     // 1: dst_mesh_id (logical)
        (uint32_t)fabric_desc.dst_chip,    // 2: dst_dev_id  (logical)
        (uint32_t)receiver_coord.x,        // 3: receiver_noc_x
        (uint32_t)receiver_coord.y,        // 4: receiver_noc_y
        (uint32_t)global_sema_b.address()  // 5: receiver L1 semaphore addr
    };
    // Append fabric args (encapsulate routing , link identifiers for fabric traffic)
    tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node,
        dst_fabric_node,
        /*link_idx=*/link_to_use,
        sender_program,
        fabric_desc.sender_core,
        writer_rta);
    SetRuntimeArgs(sender_program, writer_kernel, fabric_desc.sender_core, writer_rta);

    // Enqueue workloads
    distributed::MeshWorkload sender_workload;
    distributed::MeshWorkload receiver_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(src_mesh_coord), std::move(sender_program));
    receiver_workload.add_program(distributed::MeshCoordinateRange(dst_mesh_coord), std::move(receiver_program));

    distributed::EnqueueMeshWorkload(cq, receiver_workload, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(cq, sender_workload, /*blocking=*/false);

    distributed::Finish(cq);

    std::vector<uint32_t> rx_written_data(num_words, 0u);
    distributed::ReadShard(cq, rx_written_data, dst_buf, dst_mesh_coord, /*blocking=*/true);

    // Teardown mesh device
    mesh_device->close();
    mesh_device.reset();
}
