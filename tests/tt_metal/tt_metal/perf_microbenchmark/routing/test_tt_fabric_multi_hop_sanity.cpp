// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_fabric/control_plane.hpp"
// #include "tt_metal/impl/dispatch/cq_commands.hpp"
// #include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/tt_fabric_traffic_gen_test.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_common.hpp"
#include "tt_metal/hw/inc/wormhole/eth_l1_address_map.h"
#include "tt_fabric/hw/inc/tt_fabric_interface.h"

using std::vector;
using namespace tt;
using json = nlohmann::json;

int main(int argc, char** argv) {
    constexpr uint32_t default_tx_x = 0;
    constexpr uint32_t default_tx_y = 0;
    constexpr uint32_t default_rx_x = 0;
    constexpr uint32_t default_rx_y = 3;

    constexpr uint32_t default_mux_x = 0;
    constexpr uint32_t default_mux_y = 1;
    constexpr uint32_t default_demux_x = 0;
    constexpr uint32_t default_demux_y = 2;

    constexpr uint32_t default_prng_seed = 0x100;
    constexpr uint32_t default_data_kb_per_tx = 1024 * 1024;
    constexpr uint32_t default_max_packet_size_words = 0x100;

    constexpr uint32_t default_routing_table_start_addr = 0x7EC00;
    constexpr uint32_t default_tx_queue_start_addr = 0x80000;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = 0xa0000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;
    constexpr uint32_t default_tx_signal_address = 0x70000;

    constexpr uint32_t default_test_results_addr = 0x100000;
    constexpr uint32_t default_test_results_size = 0x40000;

    // TODO: use eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE which should be 0x19900, set test results size back
    // to 0x7000
    constexpr uint32_t default_tunneler_queue_size_bytes =
        0x4000;  // max fvc size (send + receive. top half is send buffer, bottom half is receive buffer)
    constexpr uint32_t default_tunneler_test_results_addr =
        0x39000;  // 0x8000 * 4 + 0x19000; 0x10000 * 4 + 0x19000 = 0x59000 > 0x40000 (256kB)
    constexpr uint32_t default_tunneler_test_results_size = 0x6000;  // 256kB total L1 in ethernet core - 0x39000

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;
    constexpr uint32_t default_rx_disable_header_check = 0;
    constexpr uint32_t default_tx_skip_pkt_content_gen = 0;
    constexpr uint32_t default_check_txrx_timeout = 1;

    constexpr uint32_t src_endpoint_start_id = 4;
    constexpr uint32_t dest_endpoint_start_id = 11;

    constexpr uint32_t num_endpoints = 1;
    constexpr uint32_t num_src_endpoints = num_endpoints;
    constexpr uint32_t num_dest_endpoints = num_endpoints;

    constexpr uint8_t default_tx_pkt_dest_size_choice = 0;  // pkt_dest_size_choices_t

    constexpr uint32_t default_tx_data_sent_per_iter_low = 20;
    constexpr uint32_t default_tx_data_sent_per_iter_high = 240;

    constexpr uint32_t default_fabric_command = 1;

    constexpr uint32_t default_dump_stat_json = 0;
    constexpr const char* default_output_dir = "/tmp";

    constexpr uint32_t default_test_device_id_l = 0;
    constexpr uint32_t default_test_device_id_r = -1;

    constexpr uint32_t default_target_address = 0x30000;

    constexpr uint32_t default_atomic_increment = 4;

    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  --prng_seed: PRNG seed, default = 0x{:x}", default_prng_seed);
        log_info(LogTest, "  --data_kb_per_tx: Total data in KB per TX endpoint, default = {}", default_data_kb_per_tx);
        log_info(
            LogTest,
            "  --max_packet_size_words: Max packet size in words, default = 0x{:x}",
            default_max_packet_size_words);
        log_info(LogTest, "  --tx_x: X coordinate of the starting TX core, default = {}", default_tx_x);
        log_info(LogTest, "  --tx_y: Y coordinate of the starting TX core, default = {}", default_tx_y);
        log_info(LogTest, "  --rx_x: X coordinate of the starting RX core, default = {}", default_rx_x);
        log_info(LogTest, "  --rx_y: Y coordinate of the starting RX core, default = {}", default_rx_y);
        log_info(
            LogTest,
            "  --routing_table_start_addr: Routing Table start address, default = 0x{:x}",
            default_routing_table_start_addr);
        log_info(
            LogTest, "  --tx_queue_start_addr: TX queue start address, default = 0x{:x}", default_tx_queue_start_addr);
        log_info(
            LogTest, "  --tx_queue_size_bytes: TX queue size in bytes, default = 0x{:x}", default_tx_queue_size_bytes);
        log_info(
            LogTest, "  --rx_queue_start_addr: RX queue start address, default = 0x{:x}", default_rx_queue_start_addr);
        log_info(
            LogTest, "  --rx_queue_size_bytes: RX queue size in bytes, default = 0x{:x}", default_rx_queue_size_bytes);
        log_info(
            LogTest, "  --test_results_addr: test results buf address, default = 0x{:x}", default_test_results_addr);
        log_info(LogTest, "  --test_results_size: test results buf size, default = 0x{:x}", default_test_results_size);
        log_info(LogTest, "  --timeout_mcycles: Timeout in MCycles, default = {}", default_timeout_mcycles);
        log_info(
            LogTest,
            "  --check_txrx_timeout: Check if timeout happens during tx & rx (if enabled, timeout_mcycles will also be "
            "used), default = {}",
            default_check_txrx_timeout);
        log_info(
            LogTest,
            "  --rx_disable_data_check: Disable data check on RX, default = {}",
            default_rx_disable_data_check);
        log_info(
            LogTest,
            "  --rx_disable_header_check: Disable header check on RX, default = {}",
            default_rx_disable_header_check);
        log_info(
            LogTest,
            "  --tx_skip_pkt_content_gen: Skip packet content generation during tx, default = {}",
            default_tx_skip_pkt_content_gen);
        log_info(
            LogTest,
            "  --tx_pkt_dest_size_choice: choice for how packet destination and packet size are generated, default = "
            "{}",
            default_tx_pkt_dest_size_choice);  // pkt_dest_size_choices_t
        log_info(
            LogTest,
            "  --tx_data_sent_per_iter_low: the criteria to determine the amount of tx data sent per iter is low "
            "(unit: words); if both 0, then disable counting it in tx kernel, default = {}",
            default_tx_data_sent_per_iter_low);
        log_info(
            LogTest,
            "  --tx_data_sent_per_iter_high: the criteria to determine the amount of tx data sent per iter is high "
            "(unit: words); if both 0, then disable counting it in tx kernel, default = {}",
            default_tx_data_sent_per_iter_high);
        log_info(LogTest, "  --dump_stat_json: Dump stats in json to output_dir, default = {}", default_dump_stat_json);
        log_info(LogTest, "  --output_dir: Output directory, default = {}", default_output_dir);
        log_info(
            LogTest, "  --device_id: Device on which the test will be run, default = {}", default_test_device_id_l);
        log_info(
            LogTest, "  --device_id_r: Device on which the test will be run, default = {}", default_test_device_id_r);
        return 0;
    }

    uint32_t tx_x = test_args::get_command_option_uint32(input_args, "--tx_x", default_tx_x);
    uint32_t tx_y = test_args::get_command_option_uint32(input_args, "--tx_y", default_tx_y);
    uint32_t rx_x = test_args::get_command_option_uint32(input_args, "--rx_x", default_rx_x);
    uint32_t rx_y = test_args::get_command_option_uint32(input_args, "--rx_y", default_rx_y);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    uint32_t data_kb_per_tx =
        test_args::get_command_option_uint32(input_args, "--data_kb_per_tx", default_data_kb_per_tx);
    uint32_t max_packet_size_words =
        test_args::get_command_option_uint32(input_args, "--max_packet_size_words", default_max_packet_size_words);
    uint32_t routing_table_start_addr = test_args::get_command_option_uint32(
        input_args, "--routing_table_start_addr", default_routing_table_start_addr);
    uint32_t tx_queue_start_addr =
        test_args::get_command_option_uint32(input_args, "--tx_queue_start_addr", default_tx_queue_start_addr);
    uint32_t tx_queue_size_bytes =
        test_args::get_command_option_uint32(input_args, "--tx_queue_size_bytes", default_tx_queue_size_bytes);
    uint32_t rx_queue_start_addr =
        test_args::get_command_option_uint32(input_args, "--rx_queue_start_addr", default_rx_queue_start_addr);
    uint32_t rx_queue_size_bytes =
        test_args::get_command_option_uint32(input_args, "--rx_queue_size_bytes", default_rx_queue_size_bytes);
    uint32_t tunneler_queue_size_bytes = test_args::get_command_option_uint32(
        input_args, "--tunneler_queue_size_bytes", default_tunneler_queue_size_bytes);
    uint32_t test_results_addr =
        test_args::get_command_option_uint32(input_args, "--test_results_addr", default_test_results_addr);
    uint32_t test_results_size =
        test_args::get_command_option_uint32(input_args, "--test_results_size", default_test_results_size);
    uint32_t tunneler_test_results_addr = test_args::get_command_option_uint32(
        input_args, "--tunneler_test_results_addr", default_tunneler_test_results_addr);
    uint32_t tunneler_test_results_size = test_args::get_command_option_uint32(
        input_args, "--tunneler_test_results_size", default_tunneler_test_results_size);
    uint32_t timeout_mcycles =
        test_args::get_command_option_uint32(input_args, "--timeout_mcycles", default_timeout_mcycles);
    uint32_t rx_disable_data_check =
        test_args::get_command_option_uint32(input_args, "--rx_disable_data_check", default_rx_disable_data_check);
    uint32_t rx_disable_header_check =
        test_args::get_command_option_uint32(input_args, "--rx_disable_header_check", default_rx_disable_header_check);
    uint32_t tx_skip_pkt_content_gen =
        test_args::get_command_option_uint32(input_args, "--tx_skip_pkt_content_gen", default_tx_skip_pkt_content_gen);
    uint32_t dump_stat_json =
        test_args::get_command_option_uint32(input_args, "--dump_stat_json", default_dump_stat_json);
    std::string output_dir = test_args::get_command_option(input_args, "--output_dir", std::string(default_output_dir));
    uint32_t check_txrx_timeout =
        test_args::get_command_option_uint32(input_args, "--check_txrx_timeout", default_check_txrx_timeout);
    uint8_t tx_pkt_dest_size_choice = (uint8_t)test_args::get_command_option_uint32(
        input_args, "--tx_pkt_dest_size_choice", default_tx_pkt_dest_size_choice);
    uint32_t tx_data_sent_per_iter_low = test_args::get_command_option_uint32(
        input_args, "--tx_data_sent_per_iter_low", default_tx_data_sent_per_iter_low);
    uint32_t tx_data_sent_per_iter_high = test_args::get_command_option_uint32(
        input_args, "--tx_data_sent_per_iter_high", default_tx_data_sent_per_iter_high);
    uint32_t fabric_command =
        test_args::get_command_option_uint32(input_args, "--fabric_command", default_fabric_command);
    uint32_t target_address =
        test_args::get_command_option_uint32(input_args, "--target_address", default_target_address);
    uint32_t atomic_increment =
        test_args::get_command_option_uint32(input_args, "--atomic_increment", default_atomic_increment);
    assert(
        (pkt_dest_size_choices_t)tx_pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE &&
            rx_disable_header_check ||
        (pkt_dest_size_choices_t)tx_pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM);

    uint32_t test_device_id_l =
        test_args::get_command_option_uint32(input_args, "--device_id", default_test_device_id_l);
    uint32_t test_device_id_r =
        test_args::get_command_option_uint32(input_args, "--device_id_r", default_test_device_id_r);

    uint32_t tx_signal_address = default_tx_signal_address;

    bool pass = true;

    std::map<string, string> defines = {
        {"FD_CORE_TYPE", std::to_string(0)},  // todo, support dispatch on eth
    };

    try {
        const std::filesystem::path tg_mesh_graph_desc_path =
            std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
            "tt_fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
        auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(tg_mesh_graph_desc_path.string());

        int num_devices = tt_metal::GetNumAvailableDevices();
        if (test_device_id_l >= num_devices) {
            log_info(
                LogTest, "Device {} is not valid. Highest valid device id = {}.", test_device_id_l, num_devices - 1);
            throw std::runtime_error("Invalid Device Id.");
        }

        std::map<chip_id_t, tt_metal::Device*> device_map;

        std::vector<chip_id_t> chip_ids;
        for (unsigned int id = 4; id < 36; id++) {
            chip_ids.push_back(id);
        }
        device_map = tt::tt_metal::detail::CreateDevices(chip_ids);

        log_info(LogTest, "Created {} Devices ...", device_map.size());

        std::map<chip_id_t, tt_metal::Program> program_map;

        for (uint32_t i = 4; i < 36; i++) {
            program_map[i] = tt_metal::CreateProgram();
        }

        log_info(LogTest, "Created Programs ...");

        std::map<chip_id_t, std::vector<CoreCoord>> device_router_map;

        auto const& device_active_eth_cores = device_map[test_device_id_l]->get_active_ethernet_cores();

        if (device_active_eth_cores.size() == 0) {
            log_info(
                LogTest,
                "Device {} does not have enough active cores. Need 1 active ethernet core for this test.",
                test_device_id_l);
            for (auto dev : device_map) {
                tt_metal::CloseDevice(dev.second);
            }

            throw std::runtime_error("Test cannot run on specified device.");
        }

        auto [dev_l_mesh_id, dev_l_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(test_device_id_l);
        auto [dev_r_mesh_id, dev_r_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(test_device_id_r);

        log_info(
            LogTest,
            "Running on Left  Device {} : Fabric Mesh Id {} : Fabric Device Id {}",
            test_device_id_l,
            dev_l_mesh_id,
            dev_l_chip_id);
        log_info(
            LogTest,
            "Running on Right Device {} : Fabric Mesh Id {} : Fabric Device Id {}",
            test_device_id_r,
            dev_r_mesh_id,
            dev_r_chip_id);

        bool router_core_found = false;
        CoreCoord router_logical_core;
        CoreCoord router_phys_core;
        for (auto device : device_map) {
            auto neighbors = device.second->get_ethernet_connected_device_ids();
            std::vector<CoreCoord> device_router_cores;
            std::vector<CoreCoord> device_router_phys_cores;
            uint32_t router_mask = 0;
            for (auto neighbor : neighbors) {
                if (device_map.contains(neighbor)) {
                    if (!router_core_found && device.first == test_device_id_l) {
                        // pick a router so that tx and read in routing tables from this core on the
                        // sender device.
                        router_logical_core = device.second->get_ethernet_sockets(neighbor)[0];
                        router_phys_core = device.second->ethernet_core_from_logical_core(router_logical_core);
                        router_core_found = true;
                    }
                    auto connected_logical_cores = device.second->get_ethernet_sockets(neighbor);
                    for (auto logical_core : connected_logical_cores) {
                        device_router_cores.push_back(logical_core);
                        device_router_phys_cores.push_back(
                            device.second->ethernet_core_from_logical_core(logical_core));
                        router_mask += 0x1 << logical_core.y;
                    }
                } else {
                    log_debug(
                        LogTest,
                        "Device {} skiping Neighbor Device {} since it is not in test device map.",
                        device.first,
                        neighbor);
                }
            }

            log_info(LogTest, "Device {} router_mask = 0x{:04X}", device.first, router_mask);
            uint32_t sem_count = device_router_cores.size();
            device_router_map[device.first] = device_router_phys_cores;
            std::vector<uint32_t> runtime_args = {
                sem_count,    // 0: number of active fabric routers
                router_mask,  // 1: active fabric router mask
            };
            for (auto logical_core : device_router_cores) {
                std::vector<uint32_t> router_compile_args = {
                    (tunneler_queue_size_bytes >> 4),  // 0: rx_queue_size_words
                    tunneler_test_results_addr,        // 1: test_results_addr
                    tunneler_test_results_size,        // 2: test_results_size
                    0,                                 // 3: timeout_cycles
                };
                auto router_kernel = tt_metal::CreateKernel(
                    program_map[device.first],
                    "tt_fabric/impl/kernels/tt_fabric_router.cpp",
                    logical_core,
                    tt_metal::EthernetConfig{
                        .noc = tt_metal::NOC::NOC_0, .compile_args = router_compile_args, .defines = defines});

                tt_metal::SetRuntimeArgs(program_map[device.first], router_kernel, logical_core, runtime_args);

                log_debug(
                    LogTest,
                    "Device {} router added on physical core {}",
                    device.first,
                    device.second->ethernet_core_from_logical_core(logical_core));
            }
        }

        if (check_txrx_timeout) {
            defines["CHECK_TIMEOUT"] = "";
        }

        std::vector<CoreCoord> tx_phys_core;
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            CoreCoord core = {tx_x + i, tx_y};
            tx_phys_core.push_back(device_map[test_device_id_l]->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args = {
                (device_map[test_device_id_l]->id() << 8) + src_endpoint_start_id + i,  // 0: src_endpoint_id
                num_dest_endpoints,                                                     // 1: num_dest_endpoints
                dest_endpoint_start_id,                                                 // 2:
                tx_queue_start_addr,                                                    // 3: queue_start_addr_words
                (tx_queue_size_bytes >> 4),                                             // 4: queue_size_words
                routing_table_start_addr,                                               // 5: routeing table
                0,                                                                      // 6: router_x
                0,                                                                      // 7: router_y
                test_results_addr,                                                      // 8: test_results_addr
                test_results_size,                                                      // 9: test_results_size
                prng_seed,                                                              // 10: prng_seed
                data_kb_per_tx,                                                         // 11: total_data_kb
                max_packet_size_words,                                                  // 12: max_packet_size_words
                timeout_mcycles * 1000 * 1000 * 4,                                      // 13: timeout_cycles
                tx_skip_pkt_content_gen,                                                // 14: skip_pkt_content_gen
                tx_pkt_dest_size_choice,                                                // 15: pkt_dest_size_choice
                tx_data_sent_per_iter_low,                                              // 16: data_sent_per_iter_low
                tx_data_sent_per_iter_high,                                             // 17: data_sent_per_iter_high
                fabric_command,                                                         // 18: fabric command
                target_address,
                atomic_increment,
                tx_signal_address

            };

            // setup runtime args
            std::vector<uint32_t> runtime_args = {
                (device_map[test_device_id_l]->id() << 8) + src_endpoint_start_id + i,  // 0: src_endpoint_id
                0x410,                                                                  // 1: dest_noc_offset
                router_phys_core.x,
                router_phys_core.y,
                (dev_r_mesh_id << 16 | dev_r_chip_id)};

            if (ASYNC_WR == fabric_command) {
                runtime_args.push_back(target_address);
            }

            log_info(LogTest, "run traffic_gen_tx at x={},y={}", core.x, core.y);
            auto kernel = tt_metal::CreateKernel(
                program_map[test_device_id_l],
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen_tx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_args,
                    .defines = defines});

            tt_metal::SetRuntimeArgs(program_map[test_device_id_l], kernel, core, runtime_args);
        }

        if (check_txrx_timeout) {
            defines.erase("CHECK_TIMEOUT");
        }

        log_info(LogTest, "Starting test...");

        auto start = std::chrono::system_clock::now();

        // Initialize tt_fabric router sync semaphore to 0. Routers on each device
        // increment once handshake with ethernet peer has been completed.
        std::vector<uint32_t> zero_buf(1, 0);
        for (auto [device_id, router_phys_cores] : device_router_map) {
            for (auto phys_core : router_phys_cores) {
                tt::llrt::write_hex_vec_to_core(device_id, phys_core, zero_buf, FABRIC_ROUTER_SYNC_SEM);
            }
        }

        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            tt::llrt::write_hex_vec_to_core(test_device_id_l, tx_phys_core[i], zero_buf, tx_signal_address);
        }

        // clear test status to 0. it will be set to non-zero value by tx kernel.
        tt::llrt::write_hex_vec_to_core(test_device_id_l, tx_phys_core[0], zero_buf, test_results_addr);

        for (auto device : device_map) {
            log_info(LogTest, "Launching on {}", device.first);
            tt_metal::detail::LaunchProgram(device.second, program_map[device.first], false);
        }

        // Once all the kernels have been launched on all devices, set the tx_start signal
        // to trigger tx kernels to start sending data.
        std::vector<uint32_t> tx_start(1, 1);
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            tt::llrt::write_hex_vec_to_core(test_device_id_l, tx_phys_core[i], tx_start, tx_signal_address);
        }

        // Wait for tx to return non-zero status.
        while (1) {
            auto tx_status = tt::llrt::read_hex_vec_from_core(test_device_id_l, tx_phys_core[0], test_results_addr, 4);
            if ((tx_status[0] & 0xFFFF) != 0) {
                break;
            }
        }

        log_info(LogTest, "Tx Finished");

        // terminate fabric routers
        for (auto [device_id, router_phys_cores] : device_router_map) {
            for (auto phys_core : router_phys_cores) {
                tt::llrt::write_hex_vec_to_core(device_id, phys_core, zero_buf, FABRIC_ROUTER_SYNC_SEM);
            }
        }

        // wait for all kernels to finish.
        for (auto device : device_map) {
            tt_metal::detail::WaitProgramDone(device.second, program_map[device.first]);
        }

        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end - start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        vector<vector<uint32_t>> tx_results;

        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            tx_results.push_back(tt::llrt::read_hex_vec_from_core(
                device_map[test_device_id_l]->id(), tx_phys_core[i], test_results_addr, 128));
            log_info(
                LogTest,
                "TX{} status = {}",
                i,
                packet_queue_test_status_to_string(tx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (tx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }
        /*
            TODO: Need to add these once control plane api is available to
                  get the first and last hop fabric router core.
                vector<uint32_t> router_results =
                    tt::llrt::read_hex_vec_from_core(
                        device_map[test_device_id_l]->id(), tunneler_phys_core, tunneler_test_results_addr, 128);
                log_info(LogTest, "L Router status = {}",
           packet_queue_test_status_to_string(router_results[PQ_TEST_STATUS_INDEX])); pass &=
           (router_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

                vector<uint32_t> r_router_results =
                    tt::llrt::read_hex_vec_from_core(
                        device_map[test_device_id_r]->id(), r_tunneler_phys_core, tunneler_test_results_addr, 128);
                log_info(LogTest, "R Router status = {}",
           packet_queue_test_status_to_string(r_router_results[PQ_TEST_STATUS_INDEX])); pass &=
           (r_router_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        */
        for (auto active_device : device_map) {
            pass &= tt_metal::CloseDevice(active_device.second);
        }

        if (pass) {
            double total_tx_bw = 0.0;
            uint64_t total_tx_words_sent = 0;
            uint64_t total_rx_words_checked = 0;
            for (uint32_t i = 0; i < num_src_endpoints; i++) {
                uint64_t tx_words_sent = get_64b_result(tx_results[i], PQ_TEST_WORD_CNT_INDEX);
                total_tx_words_sent += tx_words_sent;
                uint64_t tx_elapsed_cycles = get_64b_result(tx_results[i], PQ_TEST_CYCLES_INDEX);
                double tx_bw = ((double)tx_words_sent) * PACKET_WORD_SIZE_BYTES / tx_elapsed_cycles;
                total_tx_bw += tx_bw;
                uint64_t iter = get_64b_result(tx_results[i], PQ_TEST_ITER_INDEX);
                // uint64_t zero_data_sent_iter = get_64b_result(tx_results[i], TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER);
                // uint64_t few_data_sent_iter = get_64b_result(tx_results[i], TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER);
                // uint64_t many_data_sent_iter = get_64b_result(tx_results[i], TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER);
                // uint64_t num_packets = get_64b_result(tx_results[i], TX_TEST_IDX_NPKT);
                // double bytes_per_pkt = static_cast<double>(tx_words_sent) * PACKET_WORD_SIZE_BYTES /
                // static_cast<double>(num_packets);

                log_info(
                    LogTest,
                    "TX {} words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                    i,
                    tx_words_sent,
                    tx_elapsed_cycles,
                    tx_bw);
                // log_info(LogTest, "TX {} packets sent = {}, bytes/packet = {:.2f}, total iter = {}, zero data sent
                // iter = {}, few data sent iter = {}, many data sent iter = {}", i, num_packets, bytes_per_pkt, iter,
                // zero_data_sent_iter, few_data_sent_iter, many_data_sent_iter);
                /*
                                stat[fmt::format("tx_words_sent_{}", i)] = tx_words_sent;
                                stat[fmt::format("tx_elapsed_cycles_{}", i)] = tx_elapsed_cycles;
                                stat[fmt::format("tx_bw_{}", i)] = tx_bw;
                                stat[fmt::format("tx_bytes_per_pkt_{}", i)] = bytes_per_pkt;
                                stat[fmt::format("tx_total_iter_{}", i)] = iter;
                                stat[fmt::format("tx_zero_data_sent_iter_{}", i)] = zero_data_sent_iter;
                                stat[fmt::format("tx_few_data_sent_iter_{}", i)] = few_data_sent_iter;
                                stat[fmt::format("tx_many_data_sent_iter_{}", i)] = many_data_sent_iter;
                */
            }
            log_info(LogTest, "Total TX BW = {:.2f} B/cycle", total_tx_bw);
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
