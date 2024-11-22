// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/traffic_gen_test.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_common.hpp"

using std::vector;
using namespace tt;
using json = nlohmann::json;


int main(int argc, char **argv) {

    constexpr uint32_t default_tx_x = 0;
    constexpr uint32_t default_tx_y = 0;
    constexpr uint32_t default_rx_x = 0;
    constexpr uint32_t default_rx_y = 3;

    constexpr uint32_t default_mux_x = 0;
    constexpr uint32_t default_mux_y = 1;
    constexpr uint32_t default_demux_x = 0;
    constexpr uint32_t default_demux_y = 2;

    constexpr uint32_t default_prng_seed = 0x100;
    constexpr uint32_t default_data_kb_per_tx = 1024*1024;
    constexpr uint32_t default_max_packet_size_words = 0x100;

    constexpr uint32_t default_tx_queue_start_addr = 0x80000;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = 0xa0000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;
    constexpr uint32_t default_mux_queue_start_addr = 0x80000;
    constexpr uint32_t default_mux_queue_size_bytes = 0x10000;
    constexpr uint32_t default_demux_queue_start_addr = 0x90000;
    constexpr uint32_t default_demux_queue_size_bytes = 0x10000;

    constexpr uint32_t default_test_results_addr = 0x100000;
    constexpr uint32_t default_test_results_size = 0x40000;

    constexpr uint32_t default_tunneler_queue_start_addr = 0x19000;
    constexpr uint32_t default_tunneler_queue_size_bytes = 0x4000; // maximum queue (power of 2)
    constexpr uint32_t default_tunneler_test_results_addr = 0x39000; // 0x8000 * 4 + 0x19000; 0x10000 * 4 + 0x19000 = 0x59000 > 0x40000 (256kB)
    constexpr uint32_t default_tunneler_test_results_size = 0x7000; // 256kB total L1 in ethernet core - 0x39000

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;
    constexpr uint32_t default_rx_disable_header_check = 0;
    constexpr uint32_t default_tx_skip_pkt_content_gen = 0;
    constexpr uint32_t default_check_txrx_timeout = 1;

    constexpr uint32_t src_endpoint_start_id = 0xaa;
    constexpr uint32_t dest_endpoint_start_id = 0xbb;

    constexpr uint32_t num_endpoints = 4;
    constexpr uint32_t num_src_endpoints = num_endpoints;
    constexpr uint32_t num_dest_endpoints = num_endpoints;

    constexpr uint8_t default_tx_pkt_dest_size_choice = 0; // pkt_dest_size_choices_t

    constexpr uint32_t default_tx_data_sent_per_iter_low = 20;
    constexpr uint32_t default_tx_data_sent_per_iter_high = 240;

    constexpr uint32_t default_dump_stat_json = 0;
    constexpr const char* default_output_dir = "/tmp";

    constexpr uint32_t default_test_device_id = 0;

    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  --prng_seed: PRNG seed, default = 0x{:x}", default_prng_seed);
        log_info(LogTest, "  --data_kb_per_tx: Total data in KB per TX endpoint, default = {}", default_data_kb_per_tx);
        log_info(LogTest, "  --max_packet_size_words: Max packet size in words, default = 0x{:x}", default_max_packet_size_words);
        log_info(LogTest, "  --tx_x: X coordinate of the starting TX core, default = {}", default_tx_x);
        log_info(LogTest, "  --tx_y: Y coordinate of the starting TX core, default = {}", default_tx_y);
        log_info(LogTest, "  --rx_x: X coordinate of the starting RX core, default = {}", default_rx_x);
        log_info(LogTest, "  --rx_y: Y coordinate of the starting RX core, default = {}", default_rx_y);
        log_info(LogTest, "  --mux_x: X coordinate of the starting mux core, default = {}", default_mux_x);
        log_info(LogTest, "  --mux_y: Y coordinate of the starting mux core, default = {}", default_mux_y);
        log_info(LogTest, "  --demux_x: X coordinate of the starting demux core, default = {}", default_demux_x);
        log_info(LogTest, "  --demux_y: Y coordinate of the starting demux core, default = {}", default_demux_y);
        log_info(LogTest, "  --tx_queue_start_addr: TX queue start address, default = 0x{:x}", default_tx_queue_start_addr);
        log_info(LogTest, "  --tx_queue_size_bytes: TX queue size in bytes, default = 0x{:x}", default_tx_queue_size_bytes);
        log_info(LogTest, "  --rx_queue_start_addr: RX queue start address, default = 0x{:x}", default_rx_queue_start_addr);
        log_info(LogTest, "  --rx_queue_size_bytes: RX queue size in bytes, default = 0x{:x}", default_rx_queue_size_bytes);
        log_info(LogTest, "  --mux_queue_start_addr: MUX queue start address, default = 0x{:x}", default_mux_queue_start_addr);
        log_info(LogTest, "  --mux_queue_size_bytes: MUX queue size in bytes, default = 0x{:x}", default_mux_queue_size_bytes);
        log_info(LogTest, "  --demux_queue_start_addr: DEMUX queue start address, default = 0x{:x}", default_demux_queue_start_addr);
        log_info(LogTest, "  --demux_queue_size_bytes: DEMUX queue size in bytes, default = 0x{:x}", default_demux_queue_size_bytes);
        log_info(LogTest, "  --test_results_addr: test results buf address, default = 0x{:x}", default_test_results_addr);
        log_info(LogTest, "  --test_results_size: test results buf size, default = 0x{:x}", default_test_results_size);
        log_info(LogTest, "  --timeout_mcycles: Timeout in MCycles, default = {}", default_timeout_mcycles);
        log_info(LogTest, "  --check_txrx_timeout: Check if timeout happens during tx & rx (if enabled, timeout_mcycles will also be used), default = {}", default_check_txrx_timeout);
        log_info(LogTest, "  --rx_disable_data_check: Disable data check on RX, default = {}", default_rx_disable_data_check);
        log_info(LogTest, "  --rx_disable_header_check: Disable header check on RX, default = {}", default_rx_disable_header_check);
        log_info(LogTest, "  --tx_skip_pkt_content_gen: Skip packet content generation during tx, default = {}", default_tx_skip_pkt_content_gen);
        log_info(LogTest, "  --tx_pkt_dest_size_choice: choice for how packet destination and packet size are generated, default = {}", default_tx_pkt_dest_size_choice); // pkt_dest_size_choices_t
        log_info(LogTest, "  --tx_data_sent_per_iter_low: the criteria to determine the amount of tx data sent per iter is low (unit: words); if both 0, then disable counting it in tx kernel, default = {}", default_tx_data_sent_per_iter_low);
        log_info(LogTest, "  --tx_data_sent_per_iter_high: the criteria to determine the amount of tx data sent per iter is high (unit: words); if both 0, then disable counting it in tx kernel, default = {}", default_tx_data_sent_per_iter_high);
        log_info(LogTest, "  --dump_stat_json: Dump stats in json to output_dir, default = {}", default_dump_stat_json);
        log_info(LogTest, "  --output_dir: Output directory, default = {}", default_output_dir);
        log_info(LogTest, "  --device_id: Device on which the test will be run, default = {}", default_test_device_id);
        return 0;
    }

    uint32_t tx_x = test_args::get_command_option_uint32(input_args, "--tx_x", default_tx_x);
    uint32_t tx_y = test_args::get_command_option_uint32(input_args, "--tx_y", default_tx_y);
    uint32_t rx_x = test_args::get_command_option_uint32(input_args, "--rx_x", default_rx_x);
    uint32_t rx_y = test_args::get_command_option_uint32(input_args, "--rx_y", default_rx_y);
    uint32_t mux_x = test_args::get_command_option_uint32(input_args, "--mux_x", default_mux_x);
    uint32_t mux_y = test_args::get_command_option_uint32(input_args, "--mux_y", default_mux_y);
    uint32_t demux_x = test_args::get_command_option_uint32(input_args, "--demux_x", default_demux_x);
    uint32_t demux_y = test_args::get_command_option_uint32(input_args, "--demux_y", default_demux_y);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    uint32_t data_kb_per_tx = test_args::get_command_option_uint32(input_args, "--data_kb_per_tx", default_data_kb_per_tx);
    uint32_t max_packet_size_words = test_args::get_command_option_uint32(input_args, "--max_packet_size_words", default_max_packet_size_words);
    uint32_t tx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--tx_queue_start_addr", default_tx_queue_start_addr);
    uint32_t tx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tx_queue_size_bytes", default_tx_queue_size_bytes);
    uint32_t rx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--rx_queue_start_addr", default_rx_queue_start_addr);
    uint32_t rx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--rx_queue_size_bytes", default_rx_queue_size_bytes);
    uint32_t mux_queue_start_addr = test_args::get_command_option_uint32(input_args, "--mux_queue_start_addr", default_mux_queue_start_addr);
    uint32_t mux_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--mux_queue_size_bytes", default_mux_queue_size_bytes);
    uint32_t demux_queue_start_addr = test_args::get_command_option_uint32(input_args, "--demux_queue_start_addr", default_demux_queue_start_addr);
    uint32_t demux_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--demux_queue_size_bytes", default_demux_queue_size_bytes);
    uint32_t tunneler_queue_start_addr = test_args::get_command_option_uint32(input_args, "--tunneler_queue_start_addr", default_tunneler_queue_start_addr);
    uint32_t tunneler_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tunneler_queue_size_bytes", default_tunneler_queue_size_bytes);
    uint32_t test_results_addr = test_args::get_command_option_uint32(input_args, "--test_results_addr", default_test_results_addr);
    uint32_t test_results_size = test_args::get_command_option_uint32(input_args, "--test_results_size", default_test_results_size);
    uint32_t tunneler_test_results_addr = test_args::get_command_option_uint32(input_args, "--tunneler_test_results_addr", default_tunneler_test_results_addr);
    uint32_t tunneler_test_results_size = test_args::get_command_option_uint32(input_args, "--tunneler_test_results_size", default_tunneler_test_results_size);
    uint32_t timeout_mcycles = test_args::get_command_option_uint32(input_args, "--timeout_mcycles", default_timeout_mcycles);
    uint32_t rx_disable_data_check = test_args::get_command_option_uint32(input_args, "--rx_disable_data_check", default_rx_disable_data_check);
    uint32_t rx_disable_header_check = test_args::get_command_option_uint32(input_args, "--rx_disable_header_check", default_rx_disable_header_check);
    uint32_t tx_skip_pkt_content_gen = test_args::get_command_option_uint32(input_args, "--tx_skip_pkt_content_gen", default_tx_skip_pkt_content_gen);
    uint32_t dump_stat_json = test_args::get_command_option_uint32(input_args, "--dump_stat_json", default_dump_stat_json);
    std::string output_dir = test_args::get_command_option(input_args, "--output_dir", std::string(default_output_dir));
    uint32_t check_txrx_timeout = test_args::get_command_option_uint32(input_args, "--check_txrx_timeout", default_check_txrx_timeout);
    uint8_t tx_pkt_dest_size_choice = (uint8_t) test_args::get_command_option_uint32(input_args, "--tx_pkt_dest_size_choice", default_tx_pkt_dest_size_choice);
    uint32_t tx_data_sent_per_iter_low = test_args::get_command_option_uint32(input_args, "--tx_data_sent_per_iter_low", default_tx_data_sent_per_iter_low);
    uint32_t tx_data_sent_per_iter_high = test_args::get_command_option_uint32(input_args, "--tx_data_sent_per_iter_high", default_tx_data_sent_per_iter_high);

    assert((pkt_dest_size_choices_t)tx_pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE && rx_disable_header_check || (pkt_dest_size_choices_t)tx_pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM);

    uint32_t test_device_id = test_args::get_command_option_uint32(input_args, "--device_id", default_test_device_id);

    bool pass = true;

    std::map<string, string> defines = {
        {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
    };

    try {
        int num_devices = tt_metal::GetNumAvailableDevices();
        if (test_device_id >= num_devices) {
            log_info(LogTest,
                "Device {} is not valid. Highest valid device id = {}.",
                test_device_id, num_devices-1);
            throw std::runtime_error("Invalid Device Id.");
        }
        int device_id_l = test_device_id;

        tt_metal::Device *device = tt_metal::CreateDevice(device_id_l);
        auto const& device_active_eth_cores = device->get_active_ethernet_cores();

        if (device_active_eth_cores.size() == 0) {
            log_info(LogTest,
                "Device {} does not have enough active cores. Need 1 active ethernet core for this test.",
                device_id_l);
            tt_metal::CloseDevice(device);
            throw std::runtime_error("Test cannot run on specified device.");
        }

        auto eth_core_iter = device_active_eth_cores.begin();
        auto [device_id_r, eth_receiver_core] = device->get_connected_ethernet_core(*eth_core_iter);

        tt_metal::Device *device_r = tt_metal::CreateDevice(device_id_r);

        CoreCoord tunneler_logical_core = device->get_ethernet_sockets(device_id_r)[0];
        CoreCoord tunneler_phys_core = device->ethernet_core_from_logical_core(tunneler_logical_core);

        CoreCoord r_tunneler_logical_core = device_r->get_ethernet_sockets(device_id_l)[0];
        CoreCoord r_tunneler_phys_core = device_r->ethernet_core_from_logical_core(r_tunneler_logical_core);

        std::cout<<"Left Tunneler = "<<tunneler_logical_core.str()<<std::endl;
        std::cout<<"Right Tunneler = "<<r_tunneler_logical_core.str()<<std::endl;

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::Program program_r = tt_metal::CreateProgram();

        if (check_txrx_timeout) {
            defines["CHECK_TIMEOUT"] = "";
        }

/*
        std::vector<CoreCoord> tx_phys_core;
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            CoreCoord core = {tx_x+i, tx_y};
            tx_phys_core.push_back(device->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args =
                {
                    src_endpoint_start_id + i, // 0: src_endpoint_id
                    1, // 1: num_dest_endpoints
                    (tx_queue_start_addr >> 4), // 2: queue_start_addr_words
                    (tx_queue_size_bytes >> 4), // 3: queue_size_words
                    ((mux_queue_start_addr + i*mux_queue_size_bytes) >> 4), // 4: remote_rx_queue_start_addr_words
                    (mux_queue_size_bytes >> 4), // 5: remote_rx_queue_size_words
                    (uint32_t)mux_phys_core.x, // 6: remote_rx_x
                    (uint32_t)mux_phys_core.y, // 7: remote_rx_y
                    i, // 8: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 9: tx_network_type
                    test_results_addr, // 10: test_results_addr
                    test_results_size, // 11: test_results_size
                    prng_seed, // 12: prng_seed
                    data_kb_per_tx, // 13: total_data_kb
                    max_packet_size_words, // 14: max_packet_size_words
                    src_endpoint_start_id, // 15: src_endpoint_start_id
                    dest_endpoint_start_id + i, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000 * 4, // 17: timeout_cycles
                    tx_skip_pkt_content_gen, // 18: skip_pkt_content_gen
                    tx_pkt_dest_size_choice, // 19: pkt_dest_size_choice
                    tx_data_sent_per_iter_low, // 20: data_sent_per_iter_low
                    tx_data_sent_per_iter_high // 21: data_sent_per_iter_high
                };

            log_info(LogTest, "run traffic_gen_tx at x={},y={}", core.x, core.y);
            auto kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_tx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_args,
                    .defines = defines
                }
            );
        }

        std::vector<CoreCoord> rx_phys_core;
        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            CoreCoord core = {rx_x+i, rx_y};
            rx_phys_core.push_back(device_r->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args =
                {
                    dest_endpoint_start_id + i, // 0: dest_endpoint_id
                    1, // 1: num_src_endpoints
                    1, // 2: num_dest_endpoints
                    (rx_queue_start_addr >> 4), // 3: queue_start_addr_words
                    (rx_queue_size_bytes >> 4), // 4: queue_size_words
                    (uint32_t)demux_phys_core.x, // 5: remote_tx_x
                    (uint32_t)demux_phys_core.y, // 6: remote_tx_y
                    num_dest_endpoints + i, // 7: remote_tx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
                    test_results_addr, // 9: test_results_addr
                    test_results_size, // 10: test_results_size
                    prng_seed, // 11: prng_seed
                    0, // 12: reserved
                    max_packet_size_words, // 13: max_packet_size_words
                    rx_disable_data_check, // 14: disable data check
                    src_endpoint_start_id + i, // 15: src_endpoint_start_id
                    dest_endpoint_start_id + i, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000 * 4, // 17: timeout_cycles
                    rx_disable_header_check // 18: disable_header_check
                };

            log_info(LogTest, "run traffic_gen_rx at x={},y={}", core.x, core.y);
            auto kernel = tt_metal::CreateKernel(
                program_r,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_rx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_args,
                    .defines = defines
                }
            );
        }
*/
        if (check_txrx_timeout) {
            defines.erase("CHECK_TIMEOUT");
        }

        std::vector<uint32_t> tunneler_l_compile_args =
            {
                tunneler_queue_start_addr, // 2: rx_queue_start_addr_words
                tunneler_queue_start_addr + 1024, // 2: rx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                tunneler_test_results_addr, // 44: test_results_addr
                tunneler_test_results_size, // 45: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 46: timeout_cycles
            };

        auto tunneler_l_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/tt_fabric_router.cpp",
            tunneler_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = tunneler_l_compile_args,
                .defines = defines
            }
        );


        std::vector<uint32_t> tunneler_r_compile_args =
            {
                tunneler_queue_start_addr, // 2: rx_queue_start_addr_words
                tunneler_queue_start_addr + 1024, // 2: rx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                tunneler_test_results_addr, // 44: test_results_addr
                tunneler_test_results_size, // 45: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 46: timeout_cycles
            };

        auto tunneler_r_kernel = tt_metal::CreateKernel(
            program_r,
            "tt_metal/impl/dispatch/kernels/tt_fabric_router.cpp",
            r_tunneler_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = tunneler_r_compile_args,
                .defines = defines
            }
        );



        log_info(LogTest, "Starting test...");

        auto start = std::chrono::system_clock::now();
        tt_metal::detail::LaunchProgram(device, program, false);
        tt_metal::detail::LaunchProgram(device_r, program_r, false);
        tt_metal::detail::WaitProgramDone(device, program);
        tt_metal::detail::WaitProgramDone(device_r, program_r);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);


        pass &= tt_metal::CloseDevice(device);
        pass &= tt_metal::CloseDevice(device_r);

    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::OptionsG.set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
