
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_erisc_data_mover_loopback_with_workers.cpp"

void generate_ct_kernels(
    Program& program,
    Device* device,
    const CoreCoord& worker_core,
    const ttnn::ccl::SenderWorkerAdapterSpec& worker_fabric_connection,
    const mode_variant_t& mode,
    std::size_t edm_buffer_size,
    uint32_t page_plus_header_size,
    uint32_t num_pages_total,
    uint32_t num_pages_per_edm_buffer,
    uint32_t local_worker_fabric_semaphore_id,
    uint32_t local_worker_last_message_semaphore_id,
    uint32_t dram_input_buffer_base_addr,
    bool src_is_dram,
    uint32_t dram_output_buffer_base_addr,
    bool dest_is_dram,
    uint32_t worker_buffer_index_semaphore_id,
    // farthest to closest
    const std::vector<ttnn::ccl::edm_termination_info_t>& edm_termination_infos) {
    const auto& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        num_pages_per_edm_buffer,
        12345};
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_input_buffer_base_addr};

    log_trace(tt::LogTest, "\tSenderReader CT Args");
    for (const auto& arg : sender_worker_reader_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderReader RT Args");
    for (const auto& arg : sender_worker_reader_runtime_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }

    std::vector<uint32_t> sender_worker_writer_compile_args{
        num_pages_per_edm_buffer,
        num_pages_total,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        worker_fabric_connection.num_buffers_per_channel,
        dest_is_dram,
        std::holds_alternative<mcast_send>(mode) ? 1 : 0};
    log_trace(tt::LogTest, "worker_fabric_connection.edm_l1_sem_addr: {}", worker_fabric_connection.edm_l1_sem_addr);
    log_trace(tt::LogTest, "worker_buffer_index_semaphore_id: {}", worker_buffer_index_semaphore_id);
    log_trace(tt::LogTest, "last_message_semaphore_address: {}", local_worker_last_message_semaphore_id);
    log_trace(
        tt::LogTest, "Sender communicating with EDM: x={}, y={}", (uint32_t)edm_noc_core.x, (uint32_t)edm_noc_core.y);
    std::vector<uint32_t> sender_worker_writer_runtime_args{
        worker_fabric_connection.edm_buffer_base_addr,
        worker_fabric_connection.edm_l1_sem_addr,
        local_worker_fabric_semaphore_id,
        (uint32_t)edm_noc_core.x,
        (uint32_t)edm_noc_core.y,
        worker_fabric_connection.num_buffers_per_channel,

        worker_fabric_connection.edm_connection_handshake_addr,
        worker_fabric_connection.edm_worker_location_info_addr,
        edm_buffer_size,
        dram_output_buffer_base_addr,
        local_worker_last_message_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_fabric_connection.persistent_fabric ? 1 : 0,
        worker_fabric_connection.buffer_index_semaphore_id};

    if (std::holds_alternative<mcast_send>(mode)) {
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).distance);
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).range);
    } else {
        sender_worker_writer_runtime_args.push_back(std::get<unicast_send>(mode).distance);
    }

    get_runtime_args_for_edm_termination_infos(edm_termination_infos, sender_worker_writer_runtime_args);

    uint32_t src0_cb_index = CBIndex::c_0;
    log_trace(tt::LogTest, "\tSenderWriter CT Args");
    for (const auto& arg : sender_worker_writer_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderWriter RT Args");
    for (const auto& arg : sender_worker_writer_runtime_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }

    // Just want a dummy DF
    tt::DataFormat df = (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 1024   ? tt::DataFormat::Bfp8
                        : (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 2048 ? tt::DataFormat::Float16
                                                                                     : tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_plus_header_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_plus_header_size);
    CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_core, cb_src0_config);
    auto sender_worker_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/ct_args.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_worker_reader_compile_args});
    auto sender_worker_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/ct_args.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = sender_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(program, sender_worker_reader_kernel, worker_core, sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(program, sender_worker_writer_kernel, worker_core, sender_worker_writer_runtime_args);
}

bool RunCTTest(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,
    std::vector<Program>& programs,
    ttnn::ccl::FabricEriscDatamoverBuilder& chip_0_edm_builder,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    auto& sender_program = programs.at(0);
    std::size_t page_plus_header_size = page_size + sizeof(tt::fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);

    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////

    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages_total,
        .size_bytes = tensor_size_bytes,
        .page_size_bytes = page_size,
        .input_buffer_type = src_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    auto [local_input_buffer, inputs] = build_input_buffer(sender_device, tensor_size_bytes, test_config);

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    auto local_output_buffer = CreateBuffer(InterleavedBufferConfig{
        sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});

    tt_metal::detail::WriteToBuffer(local_output_buffer, all_zeros);

    auto local_input_buffer_address = local_input_buffer->address();
    auto local_output_buffer_address = local_output_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    // EDM Builder Setup
    ////////////////////////////////////////////////////////////////////////////

    static constexpr std::size_t edm_buffer_size = 4096 + PACKET_HEADER_SIZE_BYTES;

    auto chip0_worker_fabric_connection = chip_0_edm_builder.build_connection_to_worker_channel();
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const std::vector<ttnn::ccl::edm_termination_info_t>& edm_termination_infos =
        enable_persistent_fabric ? std::vector<ttnn::ccl::edm_termination_info_t>{}
                                 : std::vector<ttnn::ccl::edm_termination_info_t>{
                                       {1,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).y,
                                        ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address},
                                       {0,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).y,
                                        ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address}};

    TT_ASSERT(
        (enable_persistent_fabric && edm_termination_infos.size() == 0) ||
        (!enable_persistent_fabric && edm_termination_infos.size() > 0));
    generate_ct_kernels(
        sender_program,
        sender_device,
        worker_core,
        chip0_worker_fabric_connection,
        unicast_send{2},  // 2 hops because we are looping back to ourselves
        edm_buffer_size,
        page_plus_header_size,
        num_pages_total,
        pages_per_send,
        local_worker_fabric_semaphore_id,
        local_worker_last_message_semaphore_id,
        local_input_buffer_address,
        src_is_dram,
        local_output_buffer_address,
        dest_is_dram,
        worker_buffer_index_semaphore_id,
        edm_termination_infos);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<Device*> devices = {sender_device};
    if (!enable_persistent_fabric) {
        devices.push_back(receiver_device);
    }
    log_trace(tt::LogTest, "{} programs, {} devices", programs.size(), devices.size());
    run_programs(
        programs,
        devices,
        subdevice_managers.has_value() ? subdevice_managers.value().worker_subdevice_id
                                       : std::optional<std::unordered_map<chip_id_t, SubDeviceId>>{std::nullopt});
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        pass &= run_output_check(all_zeros, inputs, local_output_buffer) == Correctness::Correct;
    }
    return pass;
}

int TestCTEntrypoint(
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    bool enable_persistent_fabric) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    const auto& device_0 = view.get_device(0, 0);
    const auto& device_1 = view.get_device(0, 1);

    const auto& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    chip_id_t device_id = std::numeric_limits<chip_id_t>::max();
    tt_xy_pair eth_receiver_core;
    bool initialized = false;
    tt_xy_pair eth_sender_core;
    do {
        TT_FATAL(eth_sender_core_iter != eth_sender_core_iter_end, "Error");
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != device_1->id());
    TT_ASSERT(device_id == device_1->id());
    // const auto& device_1 = test_fixture.mesh_device_->get_device(device_id);

    std::vector<Program> programs(enable_persistent_fabric ? 1 : 2);
    std::optional<std::vector<Program>> fabric_programs;
    auto& sender_program = programs.at(0);
    if (enable_persistent_fabric) {
        log_info(tt::LogTest, "Enabling persistent fabric");
        fabric_programs = std::vector<Program>(2);
        subdevice_managers = create_subdevices({device_0, device_1});
    }

    auto& fabric_sender_program = enable_persistent_fabric ? fabric_programs->at(0) : sender_program;
    auto& fabric_receiver_program = enable_persistent_fabric ? fabric_programs->at(1) : programs.at(1);
    Device* sender_device = device_0;
    Device* receiver_device = device_1;

    static constexpr std::size_t edm_buffer_size = 4096 + PACKET_HEADER_SIZE_BYTES;
    const chip_id_t local_chip_id = 0;
    const chip_id_t remote_chip_id = 1;
    const auto& edm_config = ttnn::ccl::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);
    auto chip_0_edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
        sender_device,
        fabric_sender_program,
        eth_sender_core,
        local_chip_id,
        remote_chip_id,
        edm_config,
        enable_persistent_fabric);
    auto chip_1_edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
        receiver_device,
        fabric_receiver_program,
        eth_receiver_core,
        remote_chip_id,
        local_chip_id,
        edm_config,
        enable_persistent_fabric);
    // Create the loopback connection on the second device
    chip_1_edm_builder.connect_to_downstream_edm(chip_1_edm_builder);
    auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
        fabric_sender_program, sender_device, chip_0_edm_builder, eth_sender_core, NOC::NOC_0);
    auto remote_edm_kernel = ttnn::ccl::generate_edm_kernel(
        fabric_receiver_program, receiver_device, chip_1_edm_builder, eth_receiver_core, NOC::NOC_0);

    if (enable_persistent_fabric) {
        tt::tt_metal::detail::CompileProgram(sender_device, fabric_sender_program);
        tt::tt_metal::detail::CompileProgram(receiver_device, fabric_receiver_program);
        tt_metal::EnqueueProgram(sender_device->command_queue(), fabric_sender_program, false);
        tt_metal::EnqueueProgram(receiver_device->command_queue(), fabric_receiver_program, false);
    }
    log_trace(tt::LogTest, "{} programs ", programs.size());
    bool success = false;
    try {
        success = RunCTTest(
            device_0,
            device_1,

            eth_sender_core,
            eth_receiver_core,

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram,
            programs,
            chip_0_edm_builder,
            subdevice_managers,
            enable_persistent_fabric);
    } catch (std::exception& e) {
        log_error("Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    if (enable_persistent_fabric) {
        // Run the test twice with a single fabric invocation

        std::vector<Program> second_programs(1);
        try {
            success = RunCTTest(
                device_0,
                device_1,

                eth_sender_core,
                eth_receiver_core,

                page_size,
                num_pages_total,
                src_is_dram,
                dest_is_dram,
                second_programs,
                chip_0_edm_builder,
                subdevice_managers,
                enable_persistent_fabric);
        } catch (std::exception& e) {
            log_error("Caught exception: {}", e.what());
            test_fixture.TearDown();
            return -1;
        }
        // Wait for worker programs to finish

        auto d0_worker_subdevice = device_0->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
        auto d1_worker_subdevice = device_1->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
        auto d0_fabric_subdevice = device_0->get_sub_device_ids()[TEST_EDM_FABRIC_SUBDEVICE_INDEX];
        auto d1_fabric_subdevice = device_1->get_sub_device_ids()[TEST_EDM_FABRIC_SUBDEVICE_INDEX];
        // Teardown the fabric
        tt_metal::Finish(sender_device->command_queue(), {d0_worker_subdevice});
        // tt_metal::Finish(receiver_device->command_queue(), {d1_worker_subdevice});

        // Notify fabric of teardown
        chip_1_edm_builder.teardown_from_host(receiver_device);
        chip_0_edm_builder.teardown_from_host(sender_device);

        // wait for fabric finish
        tt_metal::Finish(sender_device->command_queue(), {d0_fabric_subdevice});
        tt_metal::Finish(receiver_device->command_queue(), {d1_fabric_subdevice});
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

TEST(WorkerFabricEdmDatapath, TestCTArgs) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestCTEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}
