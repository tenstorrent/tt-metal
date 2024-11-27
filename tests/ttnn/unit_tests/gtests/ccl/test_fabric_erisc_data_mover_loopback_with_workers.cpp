
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>

#include "device/tt_arch_types.h"
#include "gtest/gtest.h"
// #include "tt_backend_api_types.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

class T3000TestDevice {
   public:
    T3000TestDevice() : device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 4 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<chip_id_t> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = tt::tt_metal::detail::CreateDevices(ids);

        } else {
            TT_THROW("This suite can only be run on T3000 Wormhole devices");
        }
        device_open = true;
    }
    ~T3000TestDevice() {
        if (device_open) {
            TearDown();
        }
    }

    void TearDown() {
        device_open = false;
        for (auto [device_id, device_ptr] : devices_) {
            tt::tt_metal::CloseDevice(device_ptr);
        }
    }

    std::map<chip_id_t, Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

   private:
    bool device_open;
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    BufferType input_buffer_type;   // = BufferType::L1;
    BufferType output_buffer_type;  // = BufferType::L1;
    tt::DataFormat l1_data_format;  // = tt::DataFormat::Float16_b;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};


enum Correctness { Correct, Incorrect };

struct EthLinkBuilder {
    ttnn::ccl::FabricEriscDatamoverBuilder sender_edm_builder;    // chip_0_edm_builder,
    ttnn::ccl::FabricEriscDatamoverBuilder receiver_edm_builder;  // chip_0_edm_builder,
    tt_xy_pair sender_core;
    tt_xy_pair receiver_core;
    // size_t downstream_edm_buffer_index_semaphore_id;
};

Correctness run_output_check(
    std::vector<uint32_t> const& all_zeros,
    std::vector<uint32_t> const& inputs,
    const std::shared_ptr<Buffer>& output_buffer) {
    constexpr bool debug_mode = true;
    std::vector<uint32_t> readback_data_vec(all_zeros.size(), 0);  // init to 0 data for easier debug

    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    log_info(tt::LogTest, "Checking outputs");
    if (readback_data_vec.size() != inputs.size()) {
        log_error(tt::LogTest, "Output size mismatch: expected {} got {}", inputs.size(), readback_data_vec.size());
        return Correctness::Incorrect;
    }
    bool pass = (readback_data_vec == inputs);
    if (not pass) {
        log_error("Output mismatch");
        if (debug_mode) {
            std::size_t num_printed_mismatches = 0;
            for (size_t i = 0; i < readback_data_vec.size() && num_printed_mismatches < 64; i++) {
                if (readback_data_vec[i] != inputs[i]) {
                    log_error("[{}]: expected {} got {}", i, inputs[i], readback_data_vec[i]);
                    num_printed_mismatches++;
                }
            }
            log_error("... (remaining mismatches omitted)");
        }
    }
    return Correctness::Correct;
};

void run_programs(std::vector<Program>& programs, std::vector<Device*> const& devices) {
    EXPECT_EQ(programs.size(), devices.size());
    const size_t num_programs = programs.size();
    try {
        for (size_t i = 0; i < num_programs; i++) {
            tt::tt_metal::detail::CompileProgram(devices.at(i), programs.at(i));
        }
    } catch (std::exception& e) {
        log_error("Failed compile: {}", e.what());
        throw e;
    }

    log_info(tt::LogTest, "Running...");

    std::vector<std::thread> threads;
    threads.reserve(num_programs);
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        for (size_t i = 0; i < num_programs; i++) {
            threads.emplace_back(std::thread([&] { tt_metal::detail::LaunchProgram(devices.at(i), programs.at(i)); }));
        }

        std::ranges::for_each(threads, [](std::thread& t) { t.join(); });
    } else {
        for (size_t i = 0; i < num_programs; i++) {
            tt_metal::EnqueueProgram(devices.at(i)->command_queue(), programs.at(i), false);
        }

        log_debug(tt::LogTest, "Calling Finish");
        for (size_t i = 0; i < num_programs; i++) {
            tt_metal::Finish(devices.at(i)->command_queue());
        }
    }
}

std::tuple<std::shared_ptr<Buffer>, std::vector<uint32_t>> build_input_buffer(
    Device* first_device, size_t tensor_size_bytes, BankedConfig const& test_config) {
    auto inputs = std::vector<uint32_t>(tensor_size_bytes / sizeof(uint32_t), 0);
    std::iota(inputs.begin(), inputs.end(), 0);

    // Input buffer
    auto local_input_buffer = CreateBuffer(InterleavedBufferConfig{
        first_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});
    tt_metal::detail::WriteToBuffer(local_input_buffer, inputs);
    return {local_input_buffer, inputs};
}

struct EthLinkHop {
    CoreCoord hop_src;
    CoreCoord hop_dest;
};

struct ChipConnection {
    std::vector<EthLinkHop> links;
};

struct unicast_send {
    size_t distance;
};
struct mcast_send {
    size_t distance;
    size_t range;
};


using mode_variant_t = std::variant<mcast_send, unicast_send>;

static constexpr size_t PACKET_HEADER_SIZE_BYTES = sizeof(tt::fabric::PacketHeader);
void generate_sender_worker_kernels(
    Program& program,
    Device* device,
    CoreCoord const& worker_core,
    ttnn::ccl::SenderWorkerAdapterSpec const& worker_fabric_connection,
    mode_variant_t const& mode,
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
    std::vector<ttnn::ccl::edm_termination_info_t> const& edm_termination_infos) {

    auto const& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        num_pages_per_edm_buffer};
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_input_buffer_base_addr};

    log_trace(tt::LogTest, "\tSenderReader CT Args");
    for (auto const& arg : sender_worker_reader_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderReader RT Args");
    for (auto const& arg : sender_worker_reader_runtime_args) {
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
        tt::LogTest,
        "Sender communicating with EDM: x={}, y={}",
        (uint32_t)edm_noc_core.x,
        (uint32_t)edm_noc_core.y);
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
        worker_fabric_connection.buffer_index_semaphore_id};

    if (std::holds_alternative<mcast_send>(mode)) {
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).distance);
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).range);
    } else {
        sender_worker_writer_runtime_args.push_back(std::get<unicast_send>(mode).distance);
    }

    sender_worker_writer_runtime_args.push_back(edm_termination_infos.size());
    for (auto const& info : edm_termination_infos) {
        sender_worker_writer_runtime_args.push_back(info.edm_noc_x);
        sender_worker_writer_runtime_args.push_back(info.edm_noc_y);
        sender_worker_writer_runtime_args.push_back(info.distance);
        sender_worker_writer_runtime_args.push_back(info.termination_addr);
        log_trace(
            tt::LogTest,
            "EDM termination info: x={}, y={}, distance={}, termination_addr={}",
            info.edm_noc_x,
            info.edm_noc_y,
            info.distance,
            info.termination_addr);
    }

    uint32_t src0_cb_index = CBIndex::c_0;
    log_trace(tt::LogTest, "\tSenderWriter CT Args");
    for (auto const& arg : sender_worker_writer_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderWriter RT Args");
    for (auto const& arg : sender_worker_writer_runtime_args) {
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
        "tests/ttnn/unit_tests/gtests/ccl/kernels/fabric_erisc_datamover_sender_worker_reader.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_worker_reader_compile_args});
    auto sender_worker_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/fabric_erisc_datamover_sender_worker_sender.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = sender_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(program, sender_worker_reader_kernel, worker_core, sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(program, sender_worker_writer_kernel, worker_core, sender_worker_writer_runtime_args);
}

bool RunLoopbackTest(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram) {
    std::size_t page_plus_header_size = page_size + sizeof(tt::fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    std::vector<Program> programs(2);
    auto& sender_program = programs.at(0);
    auto& receiver_program = programs.at(1);

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
    const chip_id_t local_chip_id = 0;
    const chip_id_t remote_chip_id = 1;
    auto const& edm_config = ttnn::ccl::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);
    auto chip_0_edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
        sender_device,
        sender_program,
        eth_sender_core,
        local_chip_id,
        remote_chip_id,
        edm_config);
    auto chip0_worker_fabric_connection = chip_0_edm_builder.build_connection_to_worker_channel();
    auto chip_1_edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
        receiver_device,
        receiver_program,
        eth_receiver_core,
        remote_chip_id,
        local_chip_id,
        edm_config);
    // Create the loopback connection on the second device
    chip_1_edm_builder.connect_to_downstream_edm(chip_1_edm_builder);

    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    auto const& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    std::vector<ttnn::ccl::edm_termination_info_t> const& edm_termination_infos = {
        {1,
         sender_device->ethernet_core_from_logical_core(eth_receiver_core).x,
         sender_device->ethernet_core_from_logical_core(eth_receiver_core).y,
         ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address},
        {0,
         sender_device->ethernet_core_from_logical_core(eth_sender_core).x,
         sender_device->ethernet_core_from_logical_core(eth_sender_core).y,
         ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address}};

    generate_sender_worker_kernels(
        sender_program,
        sender_device,
        worker_core,
        chip0_worker_fabric_connection,
        unicast_send{1},
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
    // Build EDMs
    ////////////////////////////////////////////////////////////////////////////
    auto local_edm_kernel =
        ttnn::ccl::generate_edm_kernel(sender_program, sender_device, chip_0_edm_builder, eth_sender_core, NOC::NOC_0);

    auto remote_edm_kernel = ttnn::ccl::generate_edm_kernel(
        receiver_program, receiver_device, chip_1_edm_builder, eth_receiver_core, NOC::NOC_0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    run_programs(programs, {sender_device, receiver_device});
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        pass &= run_output_check(all_zeros, inputs, local_output_buffer) == Correctness::Correct;
    }
    return pass;
}

bool RunLineFabricTest(
    std::vector<tt_metal::Device*> devices,

    const size_t mcast_first_chip,
    const size_t mcast_last_chip,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram) {
    std::size_t page_plus_header_size = page_size + sizeof(tt::fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    static constexpr std::size_t edm_buffer_size = 4096 + PACKET_HEADER_SIZE_BYTES;
    const size_t local_chip_id = 0;
    const size_t remote_chip_id = 1;
    auto programs = std::vector<Program>(devices.size());
    auto program_ptrs = std::vector<Program*>(devices.size());
    std::transform(programs.begin(), programs.end(), program_ptrs.begin(), [](auto& program) { return &program; });

    auto line_fabric = ttnn::ccl::EdmLineFabricOpInterface(devices, program_ptrs, 1);

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

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

    // Input buffer
    auto [local_input_buffer, inputs] = build_input_buffer(devices[0], tensor_size_bytes, test_config);
    auto local_input_buffer_address = local_input_buffer->address();

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    // output buffers
    TT_ASSERT(mcast_first_chip <= mcast_last_chip, "mcast_first_chip must be less than or equal to mcast_last_chip");
    TT_ASSERT(mcast_last_chip < devices.size(), "mcast_last_chip must be less than the number of devices");
    std::vector<std::shared_ptr<Buffer>> output_buffers;
    output_buffers.reserve(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        if (i == 0) {
            output_buffers.push_back(CreateBuffer(InterleavedBufferConfig{
                devices.at(i), test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type}));
        } else {
            output_buffers.push_back(CreateBuffer(InterleavedBufferConfig{
                devices.at(i), test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type}, output_buffers[0]->address())
                );
        }
        tt_metal::detail::WriteToBuffer(output_buffers.back(), all_zeros);
    }
    auto local_output_buffer_address = output_buffers[0]->address();
    bool all_same_addr = std::ranges::all_of(output_buffers, [local_output_buffer_address](auto const& buffer) {
        return buffer->address() == local_output_buffer_address;
    });
    TT_ASSERT(all_same_addr, "All output buffers must have the same address");

    ////////////////////////////////////////////////////////////////////////////
    //   Setup Semaphores and Builders
    ////////////////////////////////////////////////////////////////////////////

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    auto const& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const auto edm_termination_infos = line_fabric.generate_ordered_termination_info_farthest_to_nearest();

    auto chip0_worker_fabric_connection = line_fabric.uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD);

    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    generate_sender_worker_kernels(
        programs[0],
        devices[0],
        worker_core,
        chip0_worker_fabric_connection,
        mcast_send{mcast_first_chip - 1, mcast_last_chip - mcast_first_chip},
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
    // Build EDM Kernels
    ////////////////////////////////////////////////////////////////////////////
    line_fabric.build_kernels();


    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    run_programs(programs, devices);
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {

        // Check all output buffers. Make sure only the buffers in the mcast range are
        // non-zero. All other buffers outside the range should be zero filled
        TT_ASSERT(
            !std::all_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x == 0; }),
            "Input buffer expected to not be all 0");
        for (size_t i = 0; i < output_buffers.size(); i++) {
            bool compare_with_input = (mcast_first_chip <= i && i <= mcast_last_chip);
            auto &golden_tensor = compare_with_input ? inputs : all_zeros;
            pass &= run_output_check(all_zeros, golden_tensor, output_buffers.at(i)) == Correctness::Correct;
        }
    }

    return pass;
}

// RESUME HERE AND IMPLEMENT MCAST TEST
int TestLineFabricEntrypoint(
    const size_t mcast_first_chip,
    const size_t mcast_last_chip,
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on N300 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    T3000TestDevice test_fixture;

    // build a line of devices
    std::vector<Device*> devices = {
        test_fixture.devices_.at(0),
        test_fixture.devices_.at(1),
        test_fixture.devices_.at(2),
        test_fixture.devices_.at(3)};

    bool success = false;
    try {
        success = RunLineFabricTest(
            devices,
            // fabric_hops,

            mcast_first_chip,
            mcast_last_chip,

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram);

    } catch (std::exception& e) {
        log_error("Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

int TestLoopbackEntrypoint(
    const uint32_t page_size, const uint32_t num_pages_total, const bool src_is_dram, const bool dest_is_dram) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on N300 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    T3000TestDevice test_fixture;

    const auto& device_0 = test_fixture.devices_.at(0);

    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
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
    } while (device_id != 1);
    TT_ASSERT(device_id == 1);
    const auto& device_1 = test_fixture.devices_.at(device_id);

    bool success = false;
    try {
        success = RunLoopbackTest(
            device_0,
            device_1,

            eth_sender_core,
            eth_receiver_core,

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram);
    } catch (std::exception& e) {
        log_error("Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

////////////////////////////////////////////////////////////////////
///  MESSAGE COUNT TERMINATION MODE
////////////////////////////////////////////////////////////////////

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_SingleMessage) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender and receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, DISABLED_LineFabricMcast_SingleMessage_SingleSource) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram);

    ASSERT_EQ(result, 0);
}

// Non-functional on harvested parts. Needs testing on unharvested parts.
TEST(WorkerFabricEdmDatapath, DISABLED_LineFabricMcast_ManyMessages_SingleSource) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram);

    ASSERT_EQ(result, 0);
}

// EnablePersistentKernelCache
