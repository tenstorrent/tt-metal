#include "fabric_benchmark_units.hpp"

using namespace tt;
using namespace tt::tt_metal;

void execute_incremental_packet_size_bench(
    const FabricTestDescriptor& fabric_desc,
    const tt::tt_metal::GlobalSemaphore& cur_global_sema,
    uint32_t page_size,
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    const tt_fabric::FabricNodeId& src_fabric_node,
    const tt_fabric::FabricNodeId& dst_fabric_node,
    std::shared_ptr<distributed::MeshBuffer> src_buf,
    std::shared_ptr<distributed::MeshBuffer> dst_buf,
    std::shared_ptr<distributed::MeshBuffer> device_perf_buf) {
    const auto& control_plane = MetalContext::instance().get_control_plane();

    auto src_phy_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node);
    auto dst_phy_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node);

    auto src_dev = tt::tt_metal::detail::GetActiveDevice(src_phy_id);
    auto dst_dev = tt::tt_metal::detail::GetActiveDevice(dst_phy_id);
    TT_FATAL(src_dev && dst_dev, "Both devices should be valid.");

    distributed::MeshCoordinate src_mesh_coord = extract_coord_of_phy_id(mesh_device, src_phy_id);
    distributed::MeshCoordinate dst_mesh_coord = extract_coord_of_phy_id(mesh_device, dst_phy_id);

    Program receiver_program = CreateProgram();

    constexpr const char* KERNEL_DIR = "tt_metal/multi_device_microbench/fabric_api_tests/kernels/dataflow/";
    auto rx_wait_kernel = CreateKernel(
        receiver_program,
        std::string(KERNEL_DIR) + "unicast_rx.cpp",
        fabric_desc.receiver_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(
        receiver_program, rx_wait_kernel, fabric_desc.receiver_core, /*args*/ {cur_global_sema.address(), 1u});

    // Setup sender program
    Program sender_program = CreateProgram();
    constexpr auto CB_ID = tt::CBIndex::c_0;

    // Allocate single page.
    uint32_t num_cb_total_pages = 1;
    auto data_cb_cfg = CircularBufferConfig(num_cb_total_pages * page_size, {{CB_ID, tt::DataFormat::Float16}})
                           .set_page_size(CB_ID, page_size);
    CreateCircularBuffer(sender_program, fabric_desc.sender_core, data_cb_cfg);
    auto perf_cb_cfg =
        CircularBufferConfig(page_size, {{CB_ID + 1, tt::DataFormat::UInt32}}).set_page_size(CB_ID + 1, page_size);
    CreateCircularBuffer(sender_program, fabric_desc.sender_core, perf_cb_cfg);

    std::vector<uint32_t> reader_cta;
    TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(page_size);

    auto reader_kernel = CreateKernel(
        sender_program,
        std::string(KERNEL_DIR) + "unicast_tx_reader_incremental_packet_size.cpp",
        fabric_desc.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta});
    tt::tt_metal::SetRuntimeArgs(
        sender_program, reader_kernel, fabric_desc.sender_core, {(uint32_t)src_buf->address(), num_cb_total_pages});

    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(page_size);

    auto writer_kernel = tt::tt_metal::CreateKernel(
        sender_program,
        std::string(KERNEL_DIR) + "unicast_tx_writer_incremental_packet_size.cpp",
        fabric_desc.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta});

    // find available links
    auto links = tt_fabric::get_forwarding_link_indices(src_fabric_node, dst_fabric_node);
    TT_FATAL(!links.empty(), "Need at least one available link from src to dst.");
    log_info(tt::LogTest, "Number of links are available : {}", links.size());
    uint32_t link_to_use = links[0];

    CoreCoord receiver_coord = dst_dev->worker_core_from_logical_core(fabric_desc.receiver_core);
    log_info(tt::LogTest, "dst buffer address : {}", dst_buf->address());
    log_info(tt::LogTest, "device perf buffer address : {}", device_perf_buf->address());
    std::vector<uint32_t> writer_rta = {
        (uint32_t)dst_buf->address(),
        (uint32_t)device_perf_buf->address(),
        (uint32_t)fabric_desc.mesh_id,
        (uint32_t)fabric_desc.dst_chip,
        (uint32_t)receiver_coord.x,
        (uint32_t)receiver_coord.y,
        (uint32_t)cur_global_sema.address(),
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

    // Execute workloads and wait until done
    run_recv_send_workload_once(mesh_device, receiver_workload, sender_workload);

    // 1B to 4096B
    std::vector<uint32_t> perf_points(tt::constants::TILE_HW, 0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::ReadShard(cq, perf_points, device_perf_buf, src_mesh_coord, /*blocking=*/true);

    for (uint32_t idx = 0; idx < tt::constants::TILE_HW; idx++) {
        uint32_t pkt_size = (idx + 1) * 4;
        double latency_sec = static_cast<double>(perf_points[idx]) / 1e9 / 10;  // take average of 10 samples
        double GB = static_cast<double>(pkt_size) / 1e9;
        double GB_s = (latency_sec > 0.0) ? (GB / latency_sec) : 0.0;

        log_info(
            tt::LogTest,
            "[Incremental Packet Size Test] Packet Size: {} B, Latency: {:.10f} sec, Throughput: {:.3f} GB/s",
            pkt_size,
            latency_sec,
            GB_s);
    }
}
