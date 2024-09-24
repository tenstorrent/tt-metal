// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/traffic_gen_test.hpp"

using namespace tt;


int main(int argc, char **argv) {

    constexpr uint32_t default_tx_x = 0;
    constexpr uint32_t default_tx_y = 0;
    constexpr uint32_t default_rx_x = 0;
    constexpr uint32_t default_rx_y = 3;

    constexpr uint32_t default_mux_x = 0;
    constexpr uint32_t default_mux_y = 1;
    constexpr uint32_t default_demux_x = 0;
    constexpr uint32_t default_demux_y = 2;

    constexpr uint32_t default_tunneler_x = 0;
    constexpr uint32_t default_tunneler_y = 0;

    constexpr uint32_t default_prng_seed = 0x100;
    constexpr uint32_t default_data_kb_per_tx = 16*1024;
    constexpr uint32_t default_max_packet_size_words = 0x100;

    constexpr uint32_t default_tx_queue_start_addr = 0x80000;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = 0xa0000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;
    constexpr uint32_t default_mux_queue_start_addr = 0x80000;
    constexpr uint32_t default_mux_queue_size_bytes = 0x10000;
    constexpr uint32_t default_demux_queue_start_addr = 0x90000;
    constexpr uint32_t default_demux_queue_size_bytes = 0x20000;

    constexpr uint32_t default_tunneler_queue_start_addr = 0x19000;
    constexpr uint32_t default_tunneler_queue_size_bytes = 0x10000;

    constexpr uint32_t default_test_results_addr = 0x100000;
    constexpr uint32_t default_test_results_size = 0x40000;

    constexpr uint32_t default_tunneler_test_results_addr = 0x29000;
    constexpr uint32_t default_tunneler_test_results_size = 0x8000;

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;

    constexpr uint32_t src_endpoint_start_id = 0xaa;
    constexpr uint32_t dest_endpoint_start_id = 0xbb;

    constexpr uint32_t num_src_endpoints = 4;
    constexpr uint32_t num_dest_endpoints = 4;

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
        log_info(LogTest, "  --rx_disable_data_check: Disable data check on RX, default = {}", default_rx_disable_data_check);
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
    uint32_t tunneler_x = test_args::get_command_option_uint32(input_args, "--tunneler_x", default_tunneler_x);
    uint32_t tunneler_y = test_args::get_command_option_uint32(input_args, "--tunneler_y", default_tunneler_y);
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

        CoreCoord mux_core = {mux_x, mux_y};
        CoreCoord mux_phys_core = device->worker_core_from_logical_core(mux_core);

        CoreCoord demux_core = {demux_x, demux_y};
        CoreCoord demux_phys_core = device_r->worker_core_from_logical_core(demux_core);

        std::vector<CoreCoord> tx_phys_core;
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            CoreCoord core = {tx_x+i, tx_y};
            tx_phys_core.push_back(device->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args =
                {
                    src_endpoint_start_id + i, // 0: src_endpoint_id
                    num_dest_endpoints, // 1: num_dest_endpoints
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
                    dest_endpoint_start_id, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000 * 4, // 17: timeout_cycles
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

        // Mux
        std::vector<uint32_t> mux_compile_args =
            {
                0, // 0: reserved
                (mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_src_endpoints, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)tx_phys_core[0].x,
                                      (uint32_t)tx_phys_core[0].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack((uint32_t)tx_phys_core[1].x,
                                      (uint32_t)tx_phys_core[1].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack((uint32_t)tx_phys_core[2].x,
                                      (uint32_t)tx_phys_core[2].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack((uint32_t)tx_phys_core[3].x,
                                      (uint32_t)tx_phys_core[3].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                (tunneler_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)tunneler_phys_core.x, // 10: remote_tx_x
                (uint32_t)tunneler_phys_core.y, // 11: remote_tx_y
                0, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                test_results_addr, // 14: test_results_addr
                test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 16: timeout_cycles
                0, 0, 0, 0, 0, 0, 0, 0 // 17-24: packetize/depacketize settings
            };

        log_info(LogTest, "run mux at x={},y={}", mux_core.x, mux_core.y);
        auto mux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            {mux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_compile_args,
                .defines = defines
            }
        );

        std::vector<uint32_t> tunneler_l_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                1, // 1: tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                packet_switch_4B_pack(r_tunneler_phys_core.x,
                                      r_tunneler_phys_core.y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::ETH), // 4: remote_receiver_0_info
                0, // 5: remote_receiver_1_info
                (tunneler_queue_start_addr >> 4), // 6: remote_receiver_queue_start_addr_words 0
                (tunneler_queue_size_bytes >> 4), // 7: remote_receiver_queue_size_words 0
                0, // 8: remote_receiver_queue_start_addr_words 1
                2, // 9: remote_receiver_queue_size_words 1.
                   // Unused. Setting to 2 to get around size check assertion that does not allow 0.
                packet_switch_4B_pack(mux_phys_core.x,
                                      mux_phys_core.y,
                                      num_dest_endpoints,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 10: remote_sender_0_info
                0, // 11: remote_sender_1_info
                tunneler_test_results_addr, // 12: test_results_addr
                tunneler_test_results_size, // 13: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
                0, // 15:
            };

        auto tunneler_l_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
            tunneler_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = tunneler_l_compile_args,
                .defines = defines
            }
        );


        std::vector<uint32_t> tunneler_r_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                1,  // 1: tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                packet_switch_4B_pack(demux_phys_core.x,
                                      demux_phys_core.y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_receiver_0_info
                0, // 5: remote_receiver_1_info
                (demux_queue_start_addr >> 4), // 6: remote_receiver_queue_start_addr_words 0
                (demux_queue_size_bytes >> 4), // 7: remote_receiver_queue_size_words 0
                0, // 8: remote_receiver_queue_start_addr_words 1
                2, // 9: remote_receiver_queue_size_words 1
                   // Unused. Setting to 2 to get around size check assertion that does not allow 0.
                packet_switch_4B_pack(tunneler_phys_core.x,
                                      tunneler_phys_core.y,
                                      2,
                                      (uint32_t)DispatchRemoteNetworkType::ETH), // 10: remote_sender_0_info
                0, // 11: remote_sender_1_info
                tunneler_test_results_addr, // 12: test_results_addr
                tunneler_test_results_size, // 13: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
                0, // 15:
            };

        auto tunneler_r_kernel = tt_metal::CreateKernel(
            program_r,
            "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
            r_tunneler_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = tunneler_r_compile_args,
                .defines = defines
            }
        );

        std::vector<CoreCoord> rx_phys_core;
        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            CoreCoord core = {rx_x+i, rx_y};
            rx_phys_core.push_back(device_r->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args =
                {
                    dest_endpoint_start_id + i, // 0: dest_endpoint_id
                    num_src_endpoints, // 1: num_src_endpoints
                    num_dest_endpoints, // 2: num_dest_endpoints
                    (rx_queue_start_addr >> 4), // 3: queue_start_addr_words
                    (rx_queue_size_bytes >> 4), // 4: queue_size_words
                    (uint32_t)demux_phys_core.x, // 5: remote_tx_x
                    (uint32_t)demux_phys_core.y, // 6: remote_tx_y
                    i + 1, // 7: remote_tx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
                    test_results_addr, // 9: test_results_addr
                    test_results_size, // 10: test_results_size
                    prng_seed, // 11: prng_seed
                    0, // 12: reserved
                    max_packet_size_words, // 13: max_packet_size_words
                    rx_disable_data_check, // 14: disable data check
                    src_endpoint_start_id, // 15: src_endpoint_start_id
                    dest_endpoint_start_id, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000 * 4, // 17: timeout_cycles
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

        // Demux
        uint32_t dest_map_array[4] = {0, 1, 2, 3};
        uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
        std::vector<uint32_t> demux_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                (demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_dest_endpoints, // 3: demux_fan_out
                packet_switch_4B_pack(rx_phys_core[0].x,
                                      rx_phys_core[0].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                packet_switch_4B_pack(rx_phys_core[1].x,
                                      rx_phys_core[1].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                packet_switch_4B_pack(rx_phys_core[2].x,
                                      rx_phys_core[2].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                packet_switch_4B_pack(rx_phys_core[3].x,
                                      rx_phys_core[3].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                (rx_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words 0
                (rx_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words 0
                (rx_queue_start_addr >> 4), // 10: remote_tx_queue_start_addr_words 1
                (rx_queue_size_bytes >> 4), // 11: remote_tx_queue_size_words 1
                (rx_queue_start_addr >> 4), // 12: remote_tx_queue_start_addr_words 2
                (rx_queue_size_bytes >> 4), // 13: remote_tx_queue_size_words 2
                (rx_queue_start_addr >> 4), // 14: remote_tx_queue_start_addr_words 3
                (rx_queue_size_bytes >> 4), // 15: remote_tx_queue_size_words 3
                (uint32_t)r_tunneler_phys_core.x, // 16: remote_rx_x
                (uint32_t)r_tunneler_phys_core.y, // 17: remote_rx_y
                2, // 18: remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                test_results_addr, // 22: test_results_addr
                test_results_size, // 23: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 24: timeout_cycles
                0, 0, 0, 0, 0 // 25-29: packetize/depacketize settings
            };

        log_info(LogTest, "run demux at x={},y={}", demux_core.x, demux_core.y);
        auto demux_kernel = tt_metal::CreateKernel(
            program_r,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            {demux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_compile_args,
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

        vector<vector<uint32_t>> tx_results;
        vector<vector<uint32_t>> rx_results;

        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            tx_results.push_back(
                tt::llrt::read_hex_vec_from_core(
                    device->id(), tx_phys_core[i], test_results_addr, test_results_size));
            log_info(LogTest, "TX{} status = {}", i, packet_queue_test_status_to_string(tx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (tx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            rx_results.push_back(
                tt::llrt::read_hex_vec_from_core(
                    device_r->id(), rx_phys_core[i], test_results_addr, test_results_size));
            log_info(LogTest, "RX{} status = {}", i, packet_queue_test_status_to_string(rx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (rx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        vector<uint32_t> mux_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), mux_phys_core, test_results_addr, test_results_size);
        log_info(LogTest, "MUX status = {}", packet_queue_test_status_to_string(mux_results[PQ_TEST_STATUS_INDEX]));
        pass &= (mux_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        vector<uint32_t> demux_results =
            tt::llrt::read_hex_vec_from_core(
                device_r->id(), demux_phys_core, test_results_addr, test_results_size);
        log_info(LogTest, "DEMUX status = {}", packet_queue_test_status_to_string(demux_results[PQ_TEST_STATUS_INDEX]));
        pass &= (demux_results[0] == PACKET_QUEUE_TEST_PASS);

        pass &= tt_metal::CloseDevice(device);
        pass &= tt_metal::CloseDevice(device_r);

        if (pass) {
            double total_tx_bw = 0.0;
            uint64_t total_tx_words_sent = 0;
            uint64_t total_rx_words_checked = 0;
            for (uint32_t i = 0; i < num_src_endpoints; i++) {
                uint64_t tx_words_sent = get_64b_result(tx_results[i], PQ_TEST_WORD_CNT_INDEX);
                total_tx_words_sent += tx_words_sent;
                uint64_t tx_elapsed_cycles = get_64b_result(tx_results[i], PQ_TEST_CYCLES_INDEX);
                double tx_bw = ((double)tx_words_sent) * PACKET_WORD_SIZE_BYTES / tx_elapsed_cycles;
                log_info(LogTest,
                         "TX {} words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                         i, tx_words_sent, tx_elapsed_cycles, tx_bw);
                total_tx_bw += tx_bw;
            }
            log_info(LogTest, "Total TX BW = {:.2f} B/cycle", total_tx_bw);
            double total_rx_bw = 0.0;
            for (uint32_t i = 0; i < num_dest_endpoints; i++) {
                uint64_t rx_words_checked = get_64b_result(rx_results[i], PQ_TEST_WORD_CNT_INDEX);
                total_rx_words_checked += rx_words_checked;
                uint64_t rx_elapsed_cycles = get_64b_result(rx_results[i], PQ_TEST_CYCLES_INDEX);
                double rx_bw = ((double)rx_words_checked) * PACKET_WORD_SIZE_BYTES / rx_elapsed_cycles;
                log_info(LogTest,
                         "RX {} words checked = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                         i, rx_words_checked, rx_elapsed_cycles, rx_bw);
                total_rx_bw += rx_bw;
            }
            log_info(LogTest, "Total RX BW = {:.2f} B/cycle", total_rx_bw);
            if (total_tx_words_sent != total_rx_words_checked) {
                log_error(LogTest, "Total TX words sent = {} != Total RX words checked = {}", total_tx_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "Total TX words sent = {} == Total RX words checked = {} -> OK", total_tx_words_sent, total_rx_words_checked);
            }
            uint64_t mux_words_sent = get_64b_result(mux_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t mux_elapsed_cycles = get_64b_result(mux_results, PQ_TEST_CYCLES_INDEX);
            uint64_t mux_iter = get_64b_result(mux_results, PQ_TEST_ITER_INDEX);
            double mux_bw = ((double)mux_words_sent) * PACKET_WORD_SIZE_BYTES / mux_elapsed_cycles;
            double mux_cycles_per_iter = ((double)mux_elapsed_cycles) / mux_iter;
            log_info(LogTest,
                     "MUX words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                     mux_words_sent, mux_elapsed_cycles, mux_bw);
            log_info(LogTest,
                        "MUX iters = {} -> cycles/iter = {:.1f}",
                        mux_iter, mux_cycles_per_iter);
            if (mux_words_sent != total_rx_words_checked) {
                log_error(LogTest, "MUX words sent = {} != Total RX words checked = {}", mux_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "MUX words sent = {} == Total RX words checked = {} -> OK", mux_words_sent, total_rx_words_checked);
            }

            uint64_t demux_words_sent = get_64b_result(demux_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t demux_elapsed_cycles = get_64b_result(demux_results, PQ_TEST_CYCLES_INDEX);
            double demux_bw = ((double)demux_words_sent) * PACKET_WORD_SIZE_BYTES / demux_elapsed_cycles;
            uint64_t demux_iter = get_64b_result(demux_results, PQ_TEST_ITER_INDEX);
            double demux_cycles_per_iter = ((double)demux_elapsed_cycles) / demux_iter;
            log_info(LogTest,
                     "DEMUX words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                     demux_words_sent, demux_elapsed_cycles, demux_bw);
            log_info(LogTest,
                     "DEMUX iters = {} -> cycles/iter = {:.1f}",
                     demux_iter, demux_cycles_per_iter);
            if (demux_words_sent != total_rx_words_checked) {
                log_error(LogTest, "DEMUX words sent = {} != Total RX words checked = {}", demux_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "DEMUX words sent = {} == Total RX words checked = {} -> OK", demux_words_sent, total_rx_words_checked);
            }
        }

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
