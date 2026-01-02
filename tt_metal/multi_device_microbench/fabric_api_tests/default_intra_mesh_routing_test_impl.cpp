#include "fabric_benchmark_units.hpp"

using namespace tt;
using namespace tt::tt_metal;

void execute_default_intra_mesh_routing_bench(
    const FabricTestDescriptor& fabric_desc,
    const tt::tt_metal::GlobalSemaphore& cur_global_sema,
    uint32_t buffer_size,
    uint32_t page_size,
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    const tt_fabric::FabricNodeId& src_fabric_node,
    const tt_fabric::FabricNodeId& dst_fabric_node,
    std::shared_ptr<distributed::MeshBuffer> src_buf,
    std::shared_ptr<distributed::MeshBuffer> dst_buf,
    const std::vector<uint32_t>& tx_send_data) {
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
    auto tx_core_range_set = CoreRange(fabric_desc.sender_core, fabric_desc.sender_core);
    auto num_pages = tt::div_up(buffer_size, page_size);
    log_info(tt::LogTest, "Number of pages to write is {}", num_pages);
    constexpr auto CB_ID = tt::CBIndex::c_0;

    // CB to buffer local dram read
    uint32_t num_cb_total_pages = 16;
    auto cb_cfg = CircularBufferConfig(num_cb_total_pages * page_size, {{CB_ID, tt::DataFormat::Float16}})
                      .set_page_size(CB_ID, page_size);
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
        sender_program, reader_kernel, fabric_desc.sender_core, {(uint32_t)src_buf->address(), num_cb_total_pages});

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
    log_info(tt::LogTest, "Number of links are available : {}", links.size());
    uint32_t link_to_use = links[0];

    uint32_t fabric_max_packet_size = static_cast<uint32_t>(tt_fabric::get_tt_fabric_max_payload_size_bytes());
    log_info(tt::LogTest, "Max packet size byte under tt-fabric : {}B", fabric_max_packet_size);

    CoreCoord receiver_coord = dst_dev->worker_core_from_logical_core(fabric_desc.receiver_core);
    std::vector<uint32_t> writer_rta = {
        (uint32_t)dst_buf->address(),         // 0: dst_base (receiver DRAM offset)
        (uint32_t)fabric_desc.mesh_id,        // 1: dst_mesh_id (logical)
        (uint32_t)fabric_desc.dst_chip,       // 2: dst_dev_id  (logical)
        (uint32_t)receiver_coord.x,           // 3: receiver_noc_x
        (uint32_t)receiver_coord.y,           // 4: receiver_noc_y
        (uint32_t)cur_global_sema.address(),  // 5: receiver L1 semaphore addr
        fabric_max_packet_size,
        num_cb_total_pages};
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
    const double e2e_sec_total = run_recv_send_workload_trace(mesh_device, receiver_workload, sender_workload);

    const auto num_words = buffer_size / sizeof(uint32_t);
    std::vector<uint32_t> rx_written_data(num_words, 0u);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::ReadShard(cq, rx_written_data, dst_buf, dst_mesh_coord, /*blocking=*/true);

    // Verify communication
    TT_FATAL(rx_written_data.size() == tx_send_data.size(), "Tx and Rx data size differs");
    for (size_t i = 0; i < rx_written_data.size(); ++i) {
        TT_FATAL(rx_written_data[i] == tx_send_data[i], "Data mismatch at {}-th word", i);
    }
    log_info(tt::LogTest, "Payloads are successfully transferred.");

    // Print performance metrics
    const double e2e_sec = (trace_iters > 0) ? (e2e_sec_total / static_cast<double>(trace_iters)) : 0.0;
    const uint64_t bytes = static_cast<uint64_t>(buffer_size);
    const double GB = static_cast<double>(bytes) / 1e9;          // gigabytes
    const double GB_s = (e2e_sec > 0.0) ? (GB / e2e_sec) : 0.0;  // GB per second
    const double ms = e2e_sec * 1000.0;

    log_info(tt::LogTest, "=== Performance Metrics ===");
    log_info(tt::LogTest, "  Buffer Size:         {} bytes ({:.6f} GB)", bytes, GB);
    log_info(tt::LogTest, "  Iterations:          {}", trace_iters);
    log_info(tt::LogTest, "  Total Time:          {:.6f} sec", e2e_sec_total);
    log_info(tt::LogTest, "  Avg Time/Iteration:  {:.6f} sec ({:.3f} ms)", e2e_sec, ms);
    log_info(tt::LogTest, "  Throughput:          {:.3f} GB/s", GB_s);
    log_info(tt::LogTest, "===========================");
}
