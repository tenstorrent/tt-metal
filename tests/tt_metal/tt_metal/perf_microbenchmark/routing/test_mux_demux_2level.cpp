// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/traffic_gen_test.hpp"

using namespace tt;


int main(int argc, char **argv) {

    constexpr uint32_t default_prng_seed = 0x100;
    constexpr uint32_t default_data_kb_per_tx = 64*1024;
    constexpr uint32_t default_max_packet_size_words = 0x100;

    constexpr uint32_t default_tx_queue_start_addr = 0x80000;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = 0xa0000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;
    constexpr uint32_t default_mux_queue_start_addr = 0x80000;
    constexpr uint32_t default_mux_queue_size_bytes = 0x10000;
    constexpr uint32_t default_demux_queue_start_addr = 0x90000;
    constexpr uint32_t default_demux_queue_size_bytes = 0x20000;

    constexpr uint32_t default_test_results_addr = 0x100000;
    constexpr uint32_t default_test_results_size = 0x1000;

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;

    constexpr uint32_t src_endpoint_start_id = 0xa0;
    constexpr uint32_t dest_endpoint_start_id = 0xb0;

    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  --prng_seed: PRNG seed, default = 0x{:x}", default_prng_seed);
        log_info(LogTest, "  --data_kb_per_tx: Total data in KB per TX endpoint, default = {}", default_data_kb_per_tx);
        log_info(LogTest, "  --max_packet_size_words: Max packet size in words, default = 0x{:x}", default_max_packet_size_words);
        log_info(LogTest, "  --tx_queue_start_addr: TX queue start address, default = 0x{:x}", default_tx_queue_start_addr);
        log_info(LogTest, "  --tx_queue_size_bytes: TX queue size in bytes, default = 0x{:x}", default_tx_queue_size_bytes);
        log_info(LogTest, "  --rx_queue_start_addr: RX queue start address, default = 0x{:x}", default_rx_queue_start_addr);
        log_info(LogTest, "  --rx_queue_size_bytes: RX queue size in bytes, default = 0x{:x}", default_rx_queue_size_bytes);
        log_info(LogTest, "  --mux_queue_start_addr: MUX queue start address, default = 0x{:x}", default_mux_queue_start_addr);
        log_info(LogTest, "  --mux_queue_size_bytes: MUX queue size in bytes, default = 0x{:x}", default_mux_queue_size_bytes);
        log_info(LogTest, "  --demux_queue_start_addr: DEMUX queue start address, default = 0x{:x}", default_demux_queue_start_addr);
        log_info(LogTest, "  --demux_queue_size_bytes: DEMUX queue size in bytes, default = 0x{:x}", default_demux_queue_size_bytes);
        log_info(LogTest, "  --test_results_addr: test results buffer address, default = 0x{:x}", default_test_results_addr);
        log_info(LogTest, "  --test_results_size: test results buffer size, default = 0x{:x}", default_test_results_size);
        log_info(LogTest, "  --timeout_mcycles: Timeout in MCycles, default = {}", default_timeout_mcycles);
        log_info(LogTest, "  --rx_disable_data_check: Disable data check on RX, default = {}", default_rx_disable_data_check);
        return 0;
    }

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
    uint32_t test_results_addr = test_args::get_command_option_uint32(input_args, "--test_results_addr", default_test_results_addr);
    uint32_t test_results_size = test_args::get_command_option_uint32(input_args, "--test_results_size", default_test_results_size);
    uint32_t timeout_mcycles = test_args::get_command_option_uint32(input_args, "--timeout_mcycles", default_timeout_mcycles);
    uint32_t rx_disable_data_check = test_args::get_command_option_uint32(input_args, "--rx_disable_data_check", default_rx_disable_data_check);

    constexpr uint32_t num_src_endpoints = 16;
    constexpr uint32_t num_dest_endpoints = 16;
    constexpr uint32_t num_mux_l1 = num_src_endpoints/MAX_SWITCH_FAN_IN;
    constexpr uint32_t num_mux_l2 = num_mux_l1/MAX_SWITCH_FAN_IN;
    constexpr uint32_t num_demux_l1 = num_mux_l2;
    constexpr uint32_t num_demux_l2 = num_demux_l1*MAX_SWITCH_FAN_OUT;

    static_assert(num_mux_l2 == 1 && num_demux_l1 == 1 && (num_demux_l2*MAX_SWITCH_FAN_OUT) == num_dest_endpoints,
                 "MAX_SWITCH_FAN_IN and MAX_SWITCH_FAN_OUT expected to be 4");

    bool pass = true;

    std::map<string, string> defines = {
        {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
    };

    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        tt_metal::Program program = tt_metal::CreateProgram();

        constexpr uint32_t tx_x = 0;
        constexpr uint32_t tx_y = 0;
        std::vector<CoreCoord> tx_core;
        std::vector<CoreCoord> tx_phys_core;
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            CoreCoord core = {tx_x+(i%8), tx_y+(i/8)};
            tx_core.push_back(core);
            tx_phys_core.push_back(device->worker_core_from_logical_core(core));
        }

        constexpr uint32_t rx_x = 0;
        constexpr uint32_t rx_y = 4;
        std::vector<CoreCoord> rx_core;
        std::vector<CoreCoord> rx_phys_core;
        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            CoreCoord core = {rx_x+(i%8), rx_y+(i/8)};
            rx_core.push_back(core);
            rx_phys_core.push_back(device->worker_core_from_logical_core(core));
        }

        constexpr uint32_t mux_l1_x = 0;
        constexpr uint32_t mux_l1_y = 2;
        std::vector<CoreCoord> mux_l1_core;
        std::vector<CoreCoord> mux_l1_phys_core;
        for (uint32_t i = 0; i < num_mux_l1; i++) {
            CoreCoord core = {mux_l1_x+i, mux_l1_y};
            mux_l1_core.push_back(core);
            mux_l1_phys_core.push_back(device->worker_core_from_logical_core(core));
        }

        constexpr uint32_t mux_l2_x = 4;
        constexpr uint32_t mux_l2_y = 2;
        CoreCoord mux_l2_core = {mux_l2_x, mux_l2_y};
        CoreCoord mux_l2_phys_core = device->worker_core_from_logical_core(mux_l2_core);

        constexpr uint32_t demux_l1_x = 0;
        constexpr uint32_t demux_l1_y = 3;
        CoreCoord demux_l1_core = {demux_l1_x, demux_l1_y};
        CoreCoord demux_l1_phys_core = device->worker_core_from_logical_core(demux_l1_core);

        constexpr uint32_t demux_l2_x = 1;
        constexpr uint32_t demux_l2_y = 3;
        std::vector<CoreCoord> demux_l2_core;
        std::vector<CoreCoord> demux_l2_phys_core;
        for (uint32_t i = 0; i < num_demux_l2; i++) {
            CoreCoord core = {demux_l2_x+i, demux_l2_y};
            demux_l2_core.push_back(core);
            demux_l2_phys_core.push_back(device->worker_core_from_logical_core(core));
        }

        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            uint32_t mux_index = i / MAX_SWITCH_FAN_IN;
            uint32_t mux_queue_index = i % MAX_SWITCH_FAN_IN;
            std::vector<uint32_t> compile_args =
                {
                    src_endpoint_start_id + i, // 0: src_endpoint_id
                    num_dest_endpoints, // 1: num_dest_endpoints
                    (tx_queue_start_addr >> 4), // 2: queue_start_addr_words
                    (tx_queue_size_bytes >> 4), // 3: queue_size_words
                    ((mux_queue_start_addr + mux_queue_index*mux_queue_size_bytes) >> 4), // 4: remote_rx_queue_start_addr_words
                    (mux_queue_size_bytes >> 4), // 5: remote_rx_queue_size_words
                    (uint32_t)mux_l1_phys_core[mux_index].x, // 6: remote_rx_x
                    (uint32_t)mux_l1_phys_core[mux_index].y, // 7: remote_rx_y
                    mux_queue_index, // 8: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 9: tx_network_type
                    test_results_addr, // 10: test_results_addr
                    test_results_size, // 11: test_results_size
                    prng_seed, // 12: prng_seed
                    data_kb_per_tx, // 13: total_data_kb
                    max_packet_size_words, // 14: max_packet_size_words
                    src_endpoint_start_id, // 15: src_endpoint_start_id
                    dest_endpoint_start_id, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
                };
            log_info(LogTest, "run TX {} at x={},y={} (phys x={},y={})",
                            i, tx_core[i].x, tx_core[i].y, tx_phys_core[i].x, tx_phys_core[i].y);
            auto kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_tx.cpp",
                {tx_core[i]},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_args,
                    .defines = defines,
                }
            );
        }

        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            uint32_t demux_index = i / MAX_SWITCH_FAN_OUT;
            uint32_t demux_queue_index = i % MAX_SWITCH_FAN_OUT;
            std::vector<uint32_t> compile_args =
                {
                    dest_endpoint_start_id + i, // 0: dest_endpoint_id
                    num_src_endpoints, // 1: num_src_endpoints
                    num_dest_endpoints, // 2: num_dest_endpoints
                    (rx_queue_start_addr >> 4), // 3: queue_start_addr_words
                    (rx_queue_size_bytes >> 4), // 4: queue_size_words
                    (uint32_t)demux_l2_phys_core[demux_index].x, // 5: remote_tx_x
                    (uint32_t)demux_l2_phys_core[demux_index].y, // 6: remote_tx_y
                    demux_queue_index, // 7: remote_tx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
                    test_results_addr, // 9: test_results_addr
                    test_results_size, // 10: test_results_size
                    prng_seed, // 11: prng_seed
                    0, // 12: reserved
                    max_packet_size_words, // 13: max_packet_size_words
                    rx_disable_data_check, // 14: disable data check
                    src_endpoint_start_id, // 15: src_endpoint_start_id
                    dest_endpoint_start_id, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
                };
            log_info(LogTest, "run RX {} at x={},y={} (phys x={},y={})",
                    i, rx_core[i].x, rx_core[i].y, rx_phys_core[i].x, rx_phys_core[i].y);
            auto kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_rx.cpp",
                {rx_core[i]},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_args,
                    .defines = defines,
                }
            );
        }

        for (uint32_t i = 0; i < num_mux_l1; i++) {
            uint32_t tx_core_index = i*MAX_SWITCH_FAN_IN;
            uint32_t mux_l2_port_index = i % MAX_SWITCH_FAN_IN;
            std::vector<uint32_t> mux_l1_compile_args =
                {
                    0, // 0: reserved
                    (mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                    (mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                    MAX_SWITCH_FAN_IN, // 3: mux_fan_in
                    packet_switch_4B_pack((uint32_t)tx_phys_core[tx_core_index+0].x,
                                        (uint32_t)tx_phys_core[tx_core_index+0].y,
                                        1,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                    packet_switch_4B_pack((uint32_t)tx_phys_core[tx_core_index+1].x,
                                        (uint32_t)tx_phys_core[tx_core_index+1].y,
                                        1,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                    packet_switch_4B_pack((uint32_t)tx_phys_core[tx_core_index+2].x,
                                        (uint32_t)tx_phys_core[tx_core_index+2].y,
                                        1,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                    packet_switch_4B_pack((uint32_t)tx_phys_core[tx_core_index+3].x,
                                        (uint32_t)tx_phys_core[tx_core_index+3].y,
                                        1,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                    ((mux_queue_start_addr + i*mux_queue_size_bytes) >> 4), // 8: remote_tx_queue_start_addr_words
                    (mux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                    (uint32_t)mux_l2_phys_core.x, // 10: remote_tx_x
                    (uint32_t)mux_l2_phys_core.y, // 11: remote_tx_y
                    mux_l2_port_index, // 12: remote_tx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                    test_results_addr, // 14: test_results_addr
                    test_results_size, // 15: test_results_size
                    timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                    0, 0, 0, 0, 0, 0, 0, 0 // 17-24: packetize/depacketize settings
                };
            log_info(LogTest, "run L1 MUX {} at x={},y={} (phys x={},y={})",
                            i, mux_l1_core[i].x, mux_l1_core[i].y, mux_l1_phys_core[i].x, mux_l1_phys_core[i].y);
            auto mux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                {mux_l1_core[i]},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = mux_l1_compile_args,
                    .defines = defines,
                }
            );
        }

        std::vector<uint32_t> mux_l2_compile_args =
            {
                0, // 0: reserved
                (mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                MAX_SWITCH_FAN_IN, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)mux_l1_phys_core[0].x,
                                      (uint32_t)mux_l1_phys_core[0].y,
                                      MAX_SWITCH_FAN_IN,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack((uint32_t)mux_l1_phys_core[1].x,
                                      (uint32_t)mux_l1_phys_core[1].y,
                                      MAX_SWITCH_FAN_IN,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack((uint32_t)mux_l1_phys_core[2].x,
                                      (uint32_t)mux_l1_phys_core[2].y,
                                      MAX_SWITCH_FAN_IN,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack((uint32_t)mux_l1_phys_core[3].x,
                                      (uint32_t)mux_l1_phys_core[3].y,
                                      MAX_SWITCH_FAN_IN,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                (demux_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                (demux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)demux_l1_phys_core.x, // 10: remote_tx_x
                (uint32_t)demux_l1_phys_core.y, // 11: remote_tx_y
                MAX_SWITCH_FAN_OUT, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
                test_results_addr, // 14: test_results_addr
                test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                0, 0, 0, 0, 0, 0, 0, 0 // 17-24: packetize/depacketize settings
            };
        log_info(LogTest, "run L2 MUX at x={},y={} (phys x={},y={})",
                 mux_l2_core.x, mux_l2_core.y, mux_l2_phys_core.x, mux_l2_phys_core.y);
        auto mux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            {mux_l2_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_l2_compile_args,
                .defines = defines,
            }
        );

        uint32_t demux_l1_dest_map_array[num_dest_endpoints];
        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            demux_l1_dest_map_array[i] = i / MAX_SWITCH_FAN_OUT;
        }
        uint64_t demux_l1_dest_endpoint_output_map = packet_switch_dest_pack(demux_l1_dest_map_array, num_dest_endpoints);
        std::vector<uint32_t> demux_l1_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                (demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                MAX_SWITCH_FAN_OUT, // 3: demux_fan_out
                packet_switch_4B_pack(demux_l2_phys_core[0].x,
                                      demux_l2_phys_core[0].y,
                                      MAX_SWITCH_FAN_OUT,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                packet_switch_4B_pack(demux_l2_phys_core[1].x,
                                      demux_l2_phys_core[1].y,
                                      MAX_SWITCH_FAN_OUT,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                packet_switch_4B_pack(demux_l2_phys_core[2].x,
                                      demux_l2_phys_core[2].y,
                                      MAX_SWITCH_FAN_OUT,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                packet_switch_4B_pack(demux_l2_phys_core[3].x,
                                      demux_l2_phys_core[3].y,
                                      MAX_SWITCH_FAN_OUT,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                (demux_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words 0
                (demux_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words 0
                (demux_queue_start_addr >> 4), // 10: remote_tx_queue_start_addr_words 1
                (demux_queue_size_bytes >> 4), // 11: remote_tx_queue_size_words 1
                (demux_queue_start_addr >> 4), // 12: remote_tx_queue_start_addr_words 2
                (demux_queue_size_bytes >> 4), // 13: remote_tx_queue_size_words 2
                (demux_queue_start_addr >> 4), // 14: remote_tx_queue_start_addr_words 3
                (demux_queue_size_bytes >> 4), // 15: remote_tx_queue_size_words 3
                (uint32_t)mux_l2_phys_core.x, // 16: remote_rx_x
                (uint32_t)mux_l2_phys_core.y, // 17: remote_rx_y
                MAX_SWITCH_FAN_IN, // 18: remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                (uint32_t)(demux_l1_dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                (uint32_t)(demux_l1_dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                test_results_addr, // 22: test_results_addr
                test_results_size, // 23: test_results_size
                timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                0, 0, 0, 0, 0 // 25-29: packetize/depacketize settings
            };

        log_info(LogTest, "run L1 DEMUX at x={},y={} (phys x={},y={})",
                        demux_l1_core.x, demux_l1_core.y, demux_l1_phys_core.x, demux_l1_phys_core.y);
        auto demux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            {demux_l1_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_l1_compile_args,
                .defines = defines,
            }
        );

        // this map applies to all l2 demuxes (since each covers only a subset of rx endpoints)
        uint32_t demux_l2_dest_map_array[num_dest_endpoints];
        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            demux_l2_dest_map_array[i] = i % MAX_SWITCH_FAN_OUT;
        }
        uint64_t demux_l2_dest_endpoint_output_map =
            packet_switch_dest_pack(demux_l2_dest_map_array, num_dest_endpoints);
        for (uint32_t i = 0; i < num_demux_l2; i++) {
            uint32_t rx_index = i*MAX_SWITCH_FAN_OUT;
            std::vector<uint32_t> demux_l2_compile_args =
                {
                    dest_endpoint_start_id, // 0: endpoint_id_start_index
                    (demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                    (demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                    MAX_SWITCH_FAN_OUT, // 3: demux_fan_out
                    packet_switch_4B_pack(rx_phys_core[rx_index+0].x,
                                          rx_phys_core[rx_index+0].y,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                    packet_switch_4B_pack(rx_phys_core[rx_index+1].x,
                                          rx_phys_core[rx_index+1].y,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                    packet_switch_4B_pack(rx_phys_core[rx_index+2].x,
                                          rx_phys_core[rx_index+2].y,
                                          0,
                                          (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                    packet_switch_4B_pack(rx_phys_core[rx_index+3].x,
                                          rx_phys_core[rx_index+3].y,
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
                    (uint32_t)demux_l1_phys_core.x, // 16: remote_rx_x
                    (uint32_t)demux_l1_phys_core.y, // 17: remote_rx_y
                    i, // 18: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
                    (uint32_t)(demux_l2_dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                    (uint32_t)(demux_l2_dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                    test_results_addr, // 22: test_results_addr
                    test_results_size, // 23: test_results_size
                    timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                    0, 0, 0, 0, 0 // 25-29: packetize/depacketize settings
                };

            log_info(LogTest, "run L2 DEMUX at x={},y={} (phys x={},y={})",
                            demux_l2_core[i].x, demux_l2_core[i].y, demux_l2_phys_core[i].x, demux_l2_phys_core[i].y);
            auto demux_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                {demux_l2_core[i]},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = demux_l2_compile_args,
                    .defines = defines,
                }
            );

        }

        log_info(LogTest, "Starting test...");

        auto start = std::chrono::system_clock::now();
        tt_metal::detail::LaunchProgram(device, program);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        vector<vector<uint32_t>> tx_results;
        vector<vector<uint32_t>> rx_results;
        vector<vector<uint32_t>> mux_l1_results;
        vector<uint32_t> mux_l2_results;
        vector<vector<uint32_t>> demux_l2_results;
        vector<uint32_t> demux_l1_results;

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
                    device->id(), rx_phys_core[i], test_results_addr, test_results_size));
            log_info(LogTest, "RX{} status = {}", i, packet_queue_test_status_to_string(rx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (rx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        for (uint32_t i = 0; i < num_mux_l1; i++) {
            mux_l1_results.push_back(
                tt::llrt::read_hex_vec_from_core(
                    device->id(), mux_l1_phys_core[i], test_results_addr, test_results_size)
            );
            log_info(LogTest, "L1 MUX {} status = {}", i, packet_queue_test_status_to_string(mux_l1_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (mux_l1_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        mux_l2_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), mux_l2_phys_core, test_results_addr, test_results_size);
        log_info(LogTest, "L2 MUX status = {}", packet_queue_test_status_to_string(mux_l2_results[PQ_TEST_STATUS_INDEX]));
        pass &= (mux_l2_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        for (uint32_t i = 0; i < num_demux_l2; i++) {
            demux_l2_results.push_back(
                tt::llrt::read_hex_vec_from_core(
                    device->id(), demux_l2_phys_core[i], test_results_addr, test_results_size)
            );
            log_info(LogTest, "L2 DEMUX {} status = {}", i, packet_queue_test_status_to_string(demux_l2_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (demux_l2_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        demux_l1_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), demux_l1_phys_core, test_results_addr, test_results_size);
        log_info(LogTest, "L1 DEMUX status = {}", packet_queue_test_status_to_string(demux_l1_results[0]));
        pass &= (demux_l1_results[0] == PACKET_QUEUE_TEST_PASS);

        pass &= tt_metal::CloseDevice(device);

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

            uint64_t mux_l2_words_sent = get_64b_result(mux_l2_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t mux_l2_elapsed_cycles = get_64b_result(mux_l2_results, PQ_TEST_CYCLES_INDEX);
            uint64_t mux_l2_iter = get_64b_result(mux_l2_results, PQ_TEST_ITER_INDEX);
            double mux_l2_bw = ((double)mux_l2_words_sent) * PACKET_WORD_SIZE_BYTES / mux_l2_elapsed_cycles;
            double mux_l2_cycles_per_iter = ((double)mux_l2_elapsed_cycles) / mux_l2_iter;
            log_info(LogTest,
                     "L2 MUX words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                     mux_l2_words_sent, mux_l2_elapsed_cycles, mux_l2_bw);
            log_info(LogTest,
                    "L2 MUX iter = {}, elapsed cycles = {} -> cycles/iter = {:.2f}",
                    mux_l2_iter, mux_l2_elapsed_cycles, mux_l2_cycles_per_iter);
            if (mux_l2_words_sent != total_rx_words_checked) {
                log_error(LogTest, "L2 MUX words sent = {} != Total RX words checked = {}", mux_l2_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "L2 MUX words sent = {} == Total RX words checked = {} -> OK", mux_l2_words_sent, total_rx_words_checked);
            }

            uint64_t demux_l1_words_sent = get_64b_result(demux_l1_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t demux_l1_elapsed_cycles = get_64b_result(demux_l1_results, PQ_TEST_CYCLES_INDEX);
            uint64_t demux_l1_iter = get_64b_result(demux_l1_results, PQ_TEST_ITER_INDEX);
            double demux_l1_bw = ((double)demux_l1_words_sent) * PACKET_WORD_SIZE_BYTES / demux_l1_elapsed_cycles;
            double demux_l1_cycles_per_iter = ((double)demux_l1_elapsed_cycles) / demux_l1_iter;
            log_info(LogTest,
                     "L1 DEMUX words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                     demux_l1_words_sent, demux_l1_elapsed_cycles, demux_l1_bw);
            log_info(LogTest,
                    "L1 DEMUX iter = {}, elapsed cycles = {} -> cycles/iter = {:.2f}",
                    demux_l1_iter, demux_l1_elapsed_cycles, demux_l1_cycles_per_iter);
            if (demux_l1_words_sent != total_rx_words_checked) {
                log_error(LogTest, "L1 DEMUX words sent = {} != Total RX words checked = {}", demux_l1_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "L1 DEMUX words sent = {} == Total RX words checked = {} -> OK", demux_l1_words_sent, total_rx_words_checked);
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
