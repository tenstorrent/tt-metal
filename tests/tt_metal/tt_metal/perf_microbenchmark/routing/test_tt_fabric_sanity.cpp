// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_fabric/control_plane.hpp"
#include "tt_fabric/mesh_graph.hpp"
//#include "tt_metal/impl/dispatch/cq_commands.hpp"
//#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/tt_fabric_traffic_gen_test.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_common.hpp"
#include "tt_metal/hw/inc/wormhole/eth_l1_address_map.h"
#include "tt_fabric/hw/inc/tt_fabric_interface.h"
#include <numeric>
#include <algorithm>
#include <random>

using std::vector;
using namespace tt;
using namespace tt::tt_fabric;
using json = nlohmann::json;

#define DEFAULT_NUM_HOPS (0xFFFFFFFF)

std::mt19937 global_rng;

std::vector<CoreCoord> eth_chan_to_phys_core = {
    {9, 0},
    {1, 0},
    {8, 0},
    {2, 0},
    {7, 0},
    {3, 0},
    {6, 0},
    {4, 0},
    {9, 6},
    {1, 6},
    {8, 6},
    {2, 6},
    {7, 6},
    {3, 6},
    {6, 6},
    {4, 6}};

// decides if the tx puts the data directly on eth or if a noc hop is allowed as well
bool allow_1st_noc_hop = false;

inline std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate the range
    std::iota(range.begin(), range.end(), start);

    // shuffle the range
    std::shuffle(range.begin(), range.end(), global_rng);

    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

typedef struct test_board {
    std::vector<chip_id_t> physical_chip_ids;
    std::vector<std::pair<chip_id_t, chip_id_t>> unicast_map;
    std::map<chip_id_t, tt_metal::Device*> device_handle_map;
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane;

    test_board(std::string& board_type_) {
        if ("n300" == board_type_) {
            const std::string mesh_graph_descriptor = "n300_mesh_graph_descriptor.yaml";
            uint32_t num_chips = 2;

            if (num_chips != tt_metal::GetNumAvailableDevices()) {
                throw std::runtime_error("Not found the expected 2 chips for n300");
            }

            _init_control_plane(mesh_graph_descriptor);

            for (auto i = 0; i < num_chips; i++) {
                physical_chip_ids.push_back(i);
            }
        } else if ("t3k" == board_type_) {
            const std::string mesh_graph_descriptor = "t3k_mesh_graph_descriptor.yaml";
            uint32_t num_chips = 8;

            if (num_chips != tt_metal::GetNumAvailableDevices()) {
                throw std::runtime_error("Not found the expected 8 chips for t3k");
            }

            _init_control_plane(mesh_graph_descriptor);

            for (auto i = 0; i < num_chips; i++) {
                physical_chip_ids.push_back(i);
            }
        } else if ("glx8" == board_type_) {
            _init_galaxy_board(8);
        } else if ("glx16" == board_type_) {
            _init_galaxy_board(16);
        } else if ("glx32" == board_type_) {
            _init_galaxy_board(32);
        } else {
            throw std::runtime_error("Unsupported board");
        }

        // TODO: should we support odd number of chips (for unicast)?
        if ((physical_chip_ids.size() % 2) != 0) {
            throw std::runtime_error("Odd number of chips detected, not supported currently");
        }

        device_handle_map = tt::tt_metal::detail::CreateDevices(physical_chip_ids);
    }

    void _init_control_plane(const std::string& mesh_graph_descriptor) {
        try {
            const std::filesystem::path mesh_graph_desc_path =
                std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) / "tt_fabric/mesh_graph_descriptors" /
                mesh_graph_descriptor;
            control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
        } catch (const std::exception& e) {
            log_fatal(e.what());
        }
    }

    void _init_galaxy_board(uint32_t num_chips) {
        // TODO: add support for quanta galaxy variant
        const std::string mesh_graph_descriptor = "tg_mesh_graph_descriptor.yaml";
        uint32_t mesh_id = 4;
        uint32_t start_row_idx, start_row_max_idx, num_rows;
        chip_id_t physical_chip_id;

        // do run time check for number of available chips
        if (tt_metal::GetNumAvailableDevices() < 32) {
            throw std::runtime_error("Not a valid galaxy board since it has less than 32 chips");
        }

        _init_control_plane(mesh_graph_descriptor);

        // Init valid and available chip ids
        // consecutive rows of galaxy chips are chosen at random
        // following is the arrangement with virtual mesh_chip_id
        // +----+----+----+---+
        // | 24 | 16 | 8  | 0 |
        // | 25 | 17 | 9  | 1 |
        // | 26 | 18 | 10 | 2 |
        // | 27 | 19 | 11 | 3 |
        // | 28 | 20 | 12 | 4 |
        // | 29 | 21 | 13 | 5 |
        // | 30 | 22 | 14 | 6 |
        // | 31 | 23 | 15 | 7 |
        // +----+----+----+---+

        num_rows = num_chips / 4;

        // highest row index that can be used as the starting row of selected chips
        start_row_max_idx = 8 - num_rows;

        // choose the start row idx randomly from the available options
        start_row_idx = get_random_numbers_from_range(0, start_row_max_idx, 1)[0];

        // populate valid chip and available chip IDs
        for (auto i = start_row_idx; i < (start_row_idx + num_rows); i++) {
            for (auto j = i; j < 32; j += 8) {
                physical_chip_id = control_plane->get_physical_chip_id_from_mesh_chip_id({mesh_id, j});
                physical_chip_ids.push_back(physical_chip_id);
            }
        }
    }

    void generate_unicast_map(uint32_t num_hops) {
        std::unordered_map<chip_id_t, std::vector<chip_id_t>> chip_neighbors;
        std::unordered_map<chip_id_t, std::vector<chip_id_t>> chip_n_hop_neighbors;
        std::vector<std::pair<chip_id_t, uint32_t>> n_hop_neighbors_cnt;
        std::queue<chip_id_t> temp_q;
        uint32_t current_hops;
        chip_id_t current_chip;

        // shuffle chips to induce randomness
        std::shuffle(physical_chip_ids.begin(), physical_chip_ids.end(), global_rng);

        // for default setting, generate a random unicast map
        if (DEFAULT_NUM_HOPS == num_hops) {
            for (auto i = 0; i < physical_chip_ids.size(); i += 2) {
                unicast_map.push_back({physical_chip_ids[i], physical_chip_ids[i + 1]});
            }
            return;
        }

        // for each physical chip id, store the neighbors
        // TDOD: update the logic to find inter-mesh neighbors
        for (auto chip_id : physical_chip_ids) {
            auto neighbors = device_handle_map[chip_id]->get_ethernet_connected_device_ids();
            for (auto neighbor : neighbors) {
                // only append valid chip IDs since the neighbors could include mmio chips (wh galaxy) or
                // could be outside of the board type (in case of partial galaxy configurations)
                if (is_valid_chip_id(neighbor)) {
                    chip_neighbors[chip_id].push_back(neighbor);
                }
            }
        }

        // for each chip id get the neighbors at hops = num_hops using BFS
        for (auto chip_id : physical_chip_ids) {
            std::unordered_map<chip_id_t, bool> visited;
            uint32_t num_prev_chips;
            current_hops = 0;

            temp_q.push(chip_id);
            visited[chip_id] = true;

            while (current_hops < num_hops) {
                // pull out nodes at depth current_hop - 1
                num_prev_chips = temp_q.size();
                for (auto i = 0; i < num_prev_chips; i++) {
                    current_chip = temp_q.front();
                    temp_q.pop();

                    for (auto neighbor : chip_neighbors[current_chip]) {
                        if (visited.contains(neighbor)) {
                            continue;
                        }
                        temp_q.push(neighbor);
                        visited[neighbor] = true;
                    }
                }

                current_hops++;
            }

            // short-circuit if no n hop neighbor found for the given chip
            if (temp_q.empty()) {
                continue;
            }

            while (!temp_q.empty()) {
                chip_n_hop_neighbors[chip_id].push_back(temp_q.front());
                temp_q.pop();
            }

            // shuffle the list of neighbors to induce randomness
            std::shuffle(chip_n_hop_neighbors[chip_id].begin(), chip_n_hop_neighbors[chip_id].end(), global_rng);

            n_hop_neighbors_cnt.push_back({chip_id, chip_n_hop_neighbors[chip_id].size()});
        }

        // error out if no n hop chip pairs exist at all
        if (!n_hop_neighbors_cnt.size()) {
            throw std::runtime_error("No n hop chip pairs found");
        }

        // greedy algo - TODO: still not covering all the possible pairs, need to fix
        // sort the neighbor count vector to first pick the chips with least number of neigbors
        // done to avoid cases where the maximum number of n hop unicast pairs are not found
        auto comp = [](std::pair<chip_id_t, uint32_t> a, std::pair<chip_id_t, uint32_t> b) {
            return a.second < b.second;
        };
        std::sort(n_hop_neighbors_cnt.begin(), n_hop_neighbors_cnt.end(), comp);

        for (auto& [chip_id, neighbor_cnt] : n_hop_neighbors_cnt) {
            uint32_t temp_neighbor_cnt = UINT32_MAX;
            chip_id_t selected_chip_id;

            // check if the key exists, since it could have been erased if the chip id has already been picked
            if (!chip_n_hop_neighbors.contains(chip_id)) {
                continue;
            }

            // select the 1st chip from the neighbors that hasnt been picked already
            for (auto& neighbor_chip_id : chip_n_hop_neighbors[chip_id]) {
                if (!chip_n_hop_neighbors.contains(neighbor_chip_id)) {
                    continue;
                }

                // greedy algo: choose the neighbor with least number of neighbors
                if (chip_n_hop_neighbors[neighbor_chip_id].size() < temp_neighbor_cnt) {
                    selected_chip_id = neighbor_chip_id;
                    temp_neighbor_cnt = chip_n_hop_neighbors[neighbor_chip_id].size();
                }
            }

            /* TODO: re-enable once the algo is fixed
            if (UINT32_MAX == temp_neighbor_cnt)
                throw std::runtime_error("No neighbor found for this chip");
            */

            unicast_map.push_back({chip_id, selected_chip_id});

            // remove selected chip as it should not be picked again
            chip_n_hop_neighbors.erase(selected_chip_id);

            // remove the entry for current chip as it should not be picked again
            chip_n_hop_neighbors.erase(chip_id);
        }
    }

    inline uint32_t get_num_available_devices() { return physical_chip_ids.size(); }

    inline bool is_valid_chip_id(chip_id_t physical_chip_id) {
        auto it = std::find(physical_chip_ids.begin(), physical_chip_ids.end(), physical_chip_id);
        return (it != physical_chip_ids.end());
    }

    inline tt_metal::Device* get_device_handle(chip_id_t physical_chip_id) {
        if (is_valid_chip_id(physical_chip_id)) {
            return device_handle_map[physical_chip_id];
        } else {
            throw std::runtime_error("Invalid physical chip id");
        }
    }

    inline std::pair<mesh_id_t, chip_id_t> get_mesh_chip_id(chip_id_t physical_chip_id) {
        return control_plane->get_mesh_chip_id_from_physical_chip_id(physical_chip_id);
    }

    inline std::vector<std::pair<chip_id_t, chan_id_t>> get_route_to_chip(
        mesh_id_t src_mesh_id,
        chip_id_t src_chip_id,
        mesh_id_t dst_mesh_id,
        chip_id_t dst_chip_id,
        chan_id_t src_chan_id) {
        return control_plane->get_fabric_route(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id, src_chan_id);
    }

} test_board_t;

typedef struct test_device {
    chip_id_t physical_chip_id;
    test_board_t* board_handle;
    tt_metal::Device* device_handle;
    tt_metal::Program program_handle;
    std::vector<CoreCoord> worker_cores;
    std::vector<CoreCoord> router_logical_cores;
    std::vector<CoreCoord> router_physical_cores;
    mesh_id_t mesh_id;
    chip_id_t logical_chip_id;
    uint32_t mesh_chip_id = 0;
    uint32_t router_mask = 0;

    test_device(chip_id_t chip_id_, test_board_t* board_handle_) {
        physical_chip_id = chip_id_;
        board_handle = board_handle_;

        device_handle = board_handle->get_device_handle(physical_chip_id);
        program_handle = tt_metal::CreateProgram();
        std::tie(mesh_id, logical_chip_id) = board_handle->get_mesh_chip_id(physical_chip_id);
        mesh_chip_id = (mesh_id << 16 | logical_chip_id);

        // initalize list of worker cores in 8X8 grid
        // TODO: remove hard-coding
        for (auto i = 0; i < 8; i++) {
            for (auto j = 0; j < 8; j++) {
                worker_cores.push_back(CoreCoord({i, j}));
            }
        }

        // populate router cores
        auto neighbors = device_handle->get_ethernet_connected_device_ids();
        for (auto neighbor : neighbors) {
            if (!(board_handle->is_valid_chip_id(neighbor))) {
                continue;
            }

            auto connected_logical_cores = device_handle->get_ethernet_sockets(neighbor);
            for (auto logical_core : connected_logical_cores) {
                router_logical_cores.push_back(logical_core);
                router_physical_cores.push_back(device_handle->ethernet_core_from_logical_core(logical_core));
                router_mask += 0x1 << logical_core.y;
            }
        }
    }

    void launch_router_kernels(std::vector<uint32_t>& compile_args, std::map<string, string>& defines) {
        uint32_t num_routers = router_logical_cores.size();
        std::vector<uint32_t> zero_buf(1, 0);

        for (auto i = 0; i < num_routers; i++) {
            // setup run time args
            std::vector<uint32_t> runtime_args = {
                num_routers,  // 0: number of active fabric routers
                router_mask,  // 1: active fabric router mask
            };

            // initialize the semaphore
            tt::llrt::write_hex_vec_to_core(
                device_handle->id(), router_physical_cores[i], zero_buf, FABRIC_ROUTER_SYNC_SEM);

            auto kernel = tt_metal::CreateKernel(
                program_handle,
                "tt_fabric/impl/kernels/tt_fabric_router.cpp",
                router_logical_cores[i],
                tt_metal::EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0, .compile_args = compile_args, .defines = defines});

            tt_metal::SetRuntimeArgs(program_handle, kernel, router_logical_cores[i], runtime_args);
        }
    }

    void terminate_router_kernels() {
        std::vector<uint32_t> zero_buf(1, 0);
        for (auto& core : router_physical_cores) {
            tt::llrt::write_hex_vec_to_core(device_handle->id(), core, zero_buf, FABRIC_ROUTER_SYNC_SEM);
        }
    }

    std::vector<CoreCoord> select_random_worker_cores(uint32_t count) {
        std::vector<CoreCoord> result;

        // shuffle the list of cores
        std::shuffle(worker_cores.begin(), worker_cores.end(), global_rng);

        // return and delete the selected cores
        for (auto i = 0; i < count; i++) {
            result.push_back(worker_cores.back());
            worker_cores.pop_back();
        }

        return result;
    }

    inline uint32_t get_endpoint_id(CoreCoord& logical_core) {
        return ((device_handle->id()) << 8) | ((logical_core.x) << 4) | (logical_core.y);
    }

    inline uint32_t get_noc_offset(CoreCoord& logical_core) {
        CoreCoord phys_core = device_handle->worker_core_from_logical_core(logical_core);
        return (phys_core.y << 10) | (phys_core.x << 4);
    }

    inline CoreCoord get_router_core() {
        // TODO: should we return a random core?
        return router_physical_cores.front();
    }

    inline std::vector<std::pair<chip_id_t, chan_id_t>> get_route_to_chip(
        mesh_id_t dst_mesh_id, chip_id_t dst_chip_id, chan_id_t src_chan_id) {
        return board_handle->get_route_to_chip(mesh_id, logical_chip_id, dst_mesh_id, dst_chip_id, src_chan_id);
    }

} test_device_t;

typedef struct test_traffic {
    std::shared_ptr<test_device_t> tx_device;
    std::shared_ptr<test_device_t> rx_device;
    uint32_t num_tx_workers;
    uint32_t num_rx_workers;
    uint32_t target_address;
    std::vector<CoreCoord> tx_logical_cores;
    std::vector<CoreCoord> tx_physical_cores;
    std::vector<CoreCoord> rx_logical_cores;
    std::vector<CoreCoord> rx_physical_cores;
    std::vector<CoreCoord> available_router_cores;
    std::vector<uint32_t> tx_to_rx_map;
    std::vector<std::vector<uint32_t>> rx_to_tx_map;
    std::vector<uint32_t> tx_to_rx_address_map;
    std::vector<std::vector<uint32_t>> rx_to_tx_address_map;
    std::vector<std::vector<uint32_t>> tx_results;
    std::vector<std::vector<uint32_t>> rx_results;
    uint32_t test_results_address;
    uint32_t rx_buf_size;
    uint32_t num_links_to_use;
    uint32_t link_idx = 0;

    test_traffic(
        std::shared_ptr<test_device_t>& tx_device_,
        std::shared_ptr<test_device_t>& rx_device_,
        uint32_t num_src_endpoints,
        uint32_t num_dest_endpoints,
        uint32_t target_address_,
        uint32_t num_hops,
        uint32_t num_links_) {
        tx_device = tx_device_;
        rx_device = rx_device_;
        num_tx_workers = num_src_endpoints;
        num_rx_workers = num_dest_endpoints;
        target_address = target_address_;

        tx_to_rx_map.resize(num_tx_workers);
        rx_to_tx_map.resize(num_rx_workers);
        tx_to_rx_address_map.resize(num_tx_workers);
        rx_to_tx_address_map.resize(num_rx_workers);

        // TODO: keep for now, need to update for multicast/broadcast logic
        if (num_tx_workers < num_rx_workers) {
            throw std::runtime_error("Number of dest endpoints should be less than or equal to src endpoints");
        }

        _get_available_router_cores(num_hops);
        num_links_to_use = std::min(num_links_, (uint32_t)available_router_cores.size());

        _generate_tx_to_rx_mapping();
        _generate_target_addresses();

        tx_logical_cores = tx_device->select_random_worker_cores(num_tx_workers);
        rx_logical_cores = rx_device->select_random_worker_cores(num_rx_workers);

        for (auto core : tx_logical_cores) {
            tx_physical_cores.push_back(tx_device->device_handle->worker_core_from_logical_core(core));
        }

        for (auto core : rx_logical_cores) {
            rx_physical_cores.push_back(rx_device->device_handle->worker_core_from_logical_core(core));
        }
    }

    void launch_kernels(
        std::vector<uint32_t>& tx_compile_args,
        std::vector<uint32_t>& rx_compile_args,
        std::map<string, string>& defines,
        uint32_t fabric_command,
        uint32_t tx_signal_address,
        uint32_t test_results_address_) {
        CoreCoord core, dest_core;
        std::vector<uint32_t> zero_buf(2, 0);
        CoreCoord router_phys_core;
        uint32_t mesh_chip_id = rx_device->mesh_chip_id;

        // update the test results address, which will be used later for polling, collecting results
        test_results_address = test_results_address_;

        // launch tx kernels
        for (auto i = 0; i < num_tx_workers; i++) {
            core = tx_logical_cores[i];
            dest_core = rx_logical_cores[tx_to_rx_map[i]];
            router_phys_core = _assign_router_core();

            // setup runtime args
            std::vector<uint32_t> runtime_args = {
                tx_device->get_endpoint_id(core),      // 0: src_endpoint_id
                rx_device->get_noc_offset(dest_core),  // 1: dest_noc_offset
                router_phys_core.x,                    // 2: router_x
                router_phys_core.y,                    // 3: router_y
                mesh_chip_id,                          // 4: mesh and chip id
                rx_buf_size                            // 5: space in rx's L1
            };

            if (ASYNC_WR == fabric_command) {
                runtime_args.push_back(tx_to_rx_address_map[i]);
            }

            // zero out the signal address
            tt::llrt::write_hex_vec_to_core(
                tx_device->device_handle->id(), tx_physical_cores[i], zero_buf, tx_signal_address);

            log_info(LogTest, "run traffic_gen_tx at x={},y={}", core.x, core.y);
            auto kernel = tt_metal::CreateKernel(
                tx_device->program_handle,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen_tx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = tx_compile_args,
                    .defines = defines});

            tt_metal::SetRuntimeArgs(tx_device->program_handle, kernel, core, runtime_args);
        }

        // launch rx kernels
        for (auto i = 0; i < num_rx_workers; i++) {
            core = rx_logical_cores[i];

            // setup runtime args
            std::vector<uint32_t> runtime_args = {
                rx_to_tx_map[i].size(),  // 0: num tx workers
                rx_buf_size              // 1: space available in L1
            };

            if (ASYNC_WR == fabric_command) {
                // push the src endpoint IDs
                for (auto j : rx_to_tx_map[i]) {
                    runtime_args.push_back(tx_device->get_endpoint_id(tx_logical_cores[j]));
                }

                // push target address per src
                for (auto address : rx_to_tx_address_map[i]) {
                    runtime_args.push_back(address);
                }
            } else if (ATOMIC_INC == fabric_command) {
                tt::llrt::write_hex_vec_to_core(
                    rx_device->device_handle->id(), rx_physical_cores[i], zero_buf, target_address);
            }

            // zero out the test results address, which will be used for polling
            tt::llrt::write_hex_vec_to_core(
                rx_device->device_handle->id(), rx_physical_cores[i], zero_buf, test_results_address);

            log_info(LogTest, "run traffic_gen_rx at x={},y={}", core.x, core.y);
            auto kernel = tt_metal::CreateKernel(
                rx_device->program_handle,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen_rx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = rx_compile_args,
                    .defines = defines});

            tt_metal::SetRuntimeArgs(rx_device->program_handle, kernel, core, runtime_args);
        }
    }

    void notify_tx_workers(uint32_t address) {
        std::vector<uint32_t> start_signal(1, 1);
        for (auto core : tx_physical_cores) {
            tt::llrt::write_hex_vec_to_core(tx_device->device_handle->id(), core, start_signal, address);
        }
    }

    void wait_for_rx_workers_to_finish() {
        for (auto& rx_core : rx_physical_cores) {
            while (true) {
                auto tx_status =
                    tt::llrt::read_hex_vec_from_core(rx_device->device_handle->id(), rx_core, test_results_address, 4);
                if ((tx_status[0] & 0xFFFF) != 0) {
                    break;
                }
            }
        }
    }

    bool collect_results(uint32_t test_results_address) {
        bool pass = true;

        // collect tx results
        for (uint32_t i = 0; i < num_tx_workers; i++) {
            tx_results.push_back(tt::llrt::read_hex_vec_from_core(
                tx_device->device_handle->id(), tx_physical_cores[i], test_results_address, 128));
            log_info(
                LogTest,
                "TX{} status = {}",
                i,
                packet_queue_test_status_to_string(tx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (tx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        // collect rx results
        for (uint32_t i = 0; i < num_rx_workers; i++) {
            rx_results.push_back(tt::llrt::read_hex_vec_from_core(
                rx_device->device_handle->id(), rx_physical_cores[i], test_results_address, 128));
            log_info(
                LogTest,
                "RX{} status = {}",
                i,
                packet_queue_test_status_to_string(rx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (rx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        return pass;
    }

    bool validate_results() {
        bool pass = true;
        uint64_t num_tx_words, num_tx_packets;

        // tally-up data words and number of packets from rx and tx
        for (uint32_t i = 0; i < num_rx_workers; i++) {
            num_tx_words = 0;
            num_tx_packets = 0;

            for (auto j : rx_to_tx_map[i]) {
                num_tx_words += get_64b_result(tx_results[j], PQ_TEST_WORD_CNT_INDEX);
                num_tx_packets += get_64b_result(tx_results[j], TX_TEST_IDX_NPKT);
            }
            pass &= (get_64b_result(rx_results[i], PQ_TEST_WORD_CNT_INDEX) == num_tx_words);
            pass &= (get_64b_result(rx_results[i], TX_TEST_IDX_NPKT) == num_tx_packets);

            if (!pass) {
                break;
            }
        }

        return pass;
    }

    void print_result_summary() {
        double total_tx_bw = 0.0;
        double total_tx_bw_2 = 0.0;
        uint64_t total_tx_words_sent = 0;
        uint64_t total_rx_words_checked = 0;
        uint64_t max_tx_elapsed_cycles = 0;
        for (uint32_t i = 0; i < num_tx_workers; i++) {
            uint64_t tx_words_sent = get_64b_result(tx_results[i], PQ_TEST_WORD_CNT_INDEX);
            total_tx_words_sent += tx_words_sent;
            uint64_t tx_elapsed_cycles = get_64b_result(tx_results[i], PQ_TEST_CYCLES_INDEX);
            double tx_bw = ((double)tx_words_sent) * PACKET_WORD_SIZE_BYTES / tx_elapsed_cycles;
            total_tx_bw += tx_bw;
            uint64_t iter = get_64b_result(tx_results[i], PQ_TEST_ITER_INDEX);
            max_tx_elapsed_cycles = std::max(max_tx_elapsed_cycles, tx_elapsed_cycles);
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
            // log_info(LogTest, "TX {} packets sent = {}, bytes/packet = {:.2f}, total iter = {}, zero data sent iter =
            // {}, few data sent iter = {}, many data sent iter = {}", i, num_packets, bytes_per_pkt, iter,
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
        total_tx_bw_2 = ((double)total_tx_words_sent) * PACKET_WORD_SIZE_BYTES / max_tx_elapsed_cycles;
        for (uint32_t i = 0; i < num_rx_workers; i++) {
            uint64_t words_received = get_64b_result(rx_results[i], PQ_TEST_WORD_CNT_INDEX);
            uint32_t num_tx = rx_to_tx_map[i].size();
            log_info(LogTest, "RX {}, num producers = {}, words received = {}", i, num_tx, words_received);
        }
        // log_info(LogTest, "Total TX BW = {:.2f} B/cycle", total_tx_bw);
        log_info(LogTest, "Total TX BW = {:.2f} B/cycle", total_tx_bw_2);
    }

    // generates mapping of tx core indices to rx core indices
    void _generate_tx_to_rx_mapping() {
        std::vector<uint32_t> tx_workers(num_tx_workers);
        std::vector<uint32_t> rx_workers(num_rx_workers);
        uint32_t tx_idx, rx_idx;

        // populate vectors
        std::iota(tx_workers.begin(), tx_workers.end(), 0);
        std::iota(rx_workers.begin(), rx_workers.end(), 0);

        // shuffle vectors
        std::shuffle(tx_workers.begin(), tx_workers.end(), global_rng);

        // Assign tx to rx. Ensure that atleast one tx is mapped to a rx
        while (tx_workers.size() > 0) {
            std::shuffle(rx_workers.begin(), rx_workers.end(), global_rng);
            for (uint32_t i = 0; (i < num_rx_workers) && (tx_workers.size() > 0); i++) {
                rx_idx = rx_workers[i];
                tx_idx = tx_workers.back();
                tx_workers.pop_back();
                tx_to_rx_map[tx_idx] = rx_idx;
                rx_to_tx_map[rx_idx].push_back(tx_idx);
            }
        }
    }

    // generates mapping of target addresses for tx -> rx writes
    void _generate_target_addresses() {
        uint32_t address, num_tx;
        for (auto i = 0; i < rx_to_tx_map.size(); i++) {
            num_tx = rx_to_tx_map[i].size();
            // currently only max 7 producers per consumer for ASYNC_WR are supported
            // address range for writing data: 0x30000 - 0x100000, allocating max 0x10000 range blocks per writer
            if (num_tx > 7) {
                throw std::runtime_error("currently only max 7 producers per consumer for ASYNC_WR are supported");
            } else if (1 == num_tx) {
                // TODO: remove hard-coding
                // if only one tx set it to 0x30000 to allow more data writes
                tx_to_rx_address_map[rx_to_tx_map[i][0]] = 0x30000;
                rx_to_tx_address_map[i].push_back(0x30000);
                rx_buf_size = 0xd0000;
            } else {
                // TODO: for more than 1 tx, limit the size of test data
                // allocate addresses in the range 0x30000 - 0x100000
                rx_buf_size = 0x10000;
                std::vector<uint32_t> address_prefix = get_random_numbers_from_range(3, 9, num_tx);
                for (auto j = 0; j < num_tx; j++) {
                    address = address_prefix[j] * 0x10000;
                    tx_to_rx_address_map[rx_to_tx_map[i][j]] = address;
                    rx_to_tx_address_map[i].push_back(address);
                }
            }
        }
    }

    void _get_available_router_cores(uint32_t num_hops) {
        // if default num hops then any router core can be used
        if (DEFAULT_NUM_HOPS == num_hops) {
            available_router_cores = tx_device->router_physical_cores;
            return;
        }

        // get the potential routers based on the fabric path
        for (auto i = 0; i < 16; i++) {
            std::vector<std::pair<chip_id_t, chan_id_t>> route;
            std::set<chip_id_t> chips_in_route;
            chips_in_route.insert(tx_device->physical_chip_id);
            try {
                route = tx_device->get_route_to_chip(rx_device->mesh_id, rx_device->logical_chip_id, i);
            } catch (const std::exception& e) {
                continue;
            }

            for (auto& [chip_, chan_] : route) {
                chips_in_route.insert(chip_);
            }

            // including the origin chip, the distinct number of chips should be num_hops + 1
            if (chips_in_route.size() == num_hops + 1) {
                if ((route.size() > num_hops && allow_1st_noc_hop) || route.size() == num_hops) {
                    available_router_cores.push_back(eth_chan_to_phys_core[i]);
                }
            }
        }

        // throw error if no potential router core found
        if (available_router_cores.size() == 0) {
            throw std::runtime_error("No router cores found for specified num hops");
        }
    }

    CoreCoord _assign_router_core() {
        if (link_idx == num_links_to_use) {
            link_idx = 0;
        }
        return available_router_cores[link_idx++];
    }

} test_traffic_t;

int main(int argc, char **argv) {

    constexpr uint32_t default_tx_x = 0;
    constexpr uint32_t default_tx_y = 0;
    constexpr uint32_t default_rx_x = 0;
    constexpr uint32_t default_rx_y = 3;

    constexpr uint32_t default_mux_x = 0;
    constexpr uint32_t default_mux_y = 1;
    constexpr uint32_t default_demux_x = 0;
    constexpr uint32_t default_demux_y = 2;

    constexpr uint32_t default_prng_seed = 0xFFFFFFFF;
    constexpr uint32_t default_data_kb_per_tx = 1024*1024;
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
    constexpr uint32_t default_tunneler_queue_size_bytes = 0x8000;   // maximum queue (power of 2)
    constexpr uint32_t default_tunneler_test_results_addr = 0x39000; // 0x8000 * 4 + 0x19000; 0x10000 * 4 + 0x19000 = 0x59000 > 0x40000 (256kB)
    constexpr uint32_t default_tunneler_test_results_size = 0x6000;  // 256kB total L1 in ethernet core - 0x39000

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;
    constexpr uint32_t default_rx_disable_header_check = 0;
    constexpr uint32_t default_check_txrx_timeout = 1;

    constexpr uint32_t src_endpoint_start_id = 4;
    constexpr uint32_t dest_endpoint_start_id = 11;

    // constexpr uint32_t num_endpoints = 1;
    constexpr uint32_t default_num_src_endpoints = 1;
    constexpr uint32_t default_num_dest_endpoints = 1;

    constexpr uint8_t default_tx_pkt_dest_size_choice = 0; // pkt_dest_size_choices_t

    constexpr uint32_t default_tx_data_sent_per_iter_low = 20;
    constexpr uint32_t default_tx_data_sent_per_iter_high = 240;

    constexpr uint32_t default_fabric_command = 1;

    constexpr uint32_t default_dump_stat_json = 0;
    constexpr const char* default_output_dir = "/tmp";

    constexpr uint32_t default_test_device_id_l = 0xFFFFFFFF;
    constexpr uint32_t default_test_device_id_r = 0xFFFFFFFF;

    constexpr uint32_t default_target_address = 0x30000;

    constexpr uint32_t default_atomic_increment = 4;

    constexpr const char* default_board_type = "glx32";

    constexpr uint32_t default_num_traffic_devices = 0;

    constexpr uint32_t default_num_hops = 0xFFFFFFFF;

    constexpr uint32_t default_num_packets = 0xFFFFFFFF;

    constexpr uint32_t default_num_links = 1;

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
        log_info(
            LogTest,
            "  --routing_table_start_addr: Routing Table start address, default = 0x{:x}",
            default_routing_table_start_addr);
        log_info(LogTest, "  --tx_queue_start_addr: TX queue start address, default = 0x{:x}", default_tx_queue_start_addr);
        log_info(LogTest, "  --tx_queue_size_bytes: TX queue size in bytes, default = 0x{:x}", default_tx_queue_size_bytes);
        log_info(LogTest, "  --rx_queue_start_addr: RX queue start address, default = 0x{:x}", default_rx_queue_start_addr);
        log_info(
            LogTest, "  --rx_queue_size_bytes: RX queue size in bytes, default = 0x{:x}", default_rx_queue_size_bytes);
        log_info(LogTest, "  --test_results_addr: test results buf address, default = 0x{:x}", default_test_results_addr);
        log_info(LogTest, "  --test_results_size: test results buf size, default = 0x{:x}", default_test_results_size);
        log_info(LogTest, "  --timeout_mcycles: Timeout in MCycles, default = {}", default_timeout_mcycles);
        log_info(LogTest, "  --check_txrx_timeout: Check if timeout happens during tx & rx (if enabled, timeout_mcycles will also be used), default = {}", default_check_txrx_timeout);
        log_info(LogTest, "  --rx_disable_data_check: Disable data check on RX, default = {}", default_rx_disable_data_check);
        log_info(LogTest, "  --rx_disable_header_check: Disable header check on RX, default = {}", default_rx_disable_header_check);
        log_info(LogTest, "  --tx_skip_pkt_content_gen: Skip packet content generation during tx, default = {}", false);
        log_info(LogTest, "  --tx_pkt_dest_size_choice: choice for how packet destination and packet size are generated, default = {}", default_tx_pkt_dest_size_choice); // pkt_dest_size_choices_t
        log_info(LogTest, "  --tx_data_sent_per_iter_low: the criteria to determine the amount of tx data sent per iter is low (unit: words); if both 0, then disable counting it in tx kernel, default = {}", default_tx_data_sent_per_iter_low);
        log_info(LogTest, "  --tx_data_sent_per_iter_high: the criteria to determine the amount of tx data sent per iter is high (unit: words); if both 0, then disable counting it in tx kernel, default = {}", default_tx_data_sent_per_iter_high);
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
    uint32_t data_kb_per_tx = test_args::get_command_option_uint32(input_args, "--data_kb_per_tx", default_data_kb_per_tx);
    uint32_t max_packet_size_words = test_args::get_command_option_uint32(input_args, "--max_packet_size_words", default_max_packet_size_words);
    uint32_t routing_table_start_addr = test_args::get_command_option_uint32(
        input_args, "--routing_table_start_addr", default_routing_table_start_addr);
    uint32_t tx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--tx_queue_start_addr", default_tx_queue_start_addr);
    uint32_t tx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tx_queue_size_bytes", default_tx_queue_size_bytes);
    uint32_t rx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--rx_queue_start_addr", default_rx_queue_start_addr);
    uint32_t rx_queue_size_bytes =
        test_args::get_command_option_uint32(input_args, "--rx_queue_size_bytes", default_rx_queue_size_bytes);
    uint32_t tunneler_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tunneler_queue_size_bytes", default_tunneler_queue_size_bytes);
    uint32_t test_results_addr = test_args::get_command_option_uint32(input_args, "--test_results_addr", default_test_results_addr);
    uint32_t test_results_size = test_args::get_command_option_uint32(input_args, "--test_results_size", default_test_results_size);
    uint32_t tunneler_test_results_addr = test_args::get_command_option_uint32(input_args, "--tunneler_test_results_addr", default_tunneler_test_results_addr);
    uint32_t tunneler_test_results_size = test_args::get_command_option_uint32(input_args, "--tunneler_test_results_size", default_tunneler_test_results_size);
    uint32_t timeout_mcycles = test_args::get_command_option_uint32(input_args, "--timeout_mcycles", default_timeout_mcycles);
    uint32_t rx_disable_data_check = test_args::get_command_option_uint32(input_args, "--rx_disable_data_check", default_rx_disable_data_check);
    uint32_t rx_disable_header_check = test_args::get_command_option_uint32(input_args, "--rx_disable_header_check", default_rx_disable_header_check);
    bool tx_skip_pkt_content_gen = test_args::has_command_option(input_args, "--tx_skip_pkt_content_gen");
    uint32_t dump_stat_json = test_args::get_command_option_uint32(input_args, "--dump_stat_json", default_dump_stat_json);
    std::string output_dir = test_args::get_command_option(input_args, "--output_dir", std::string(default_output_dir));
    uint32_t check_txrx_timeout = test_args::get_command_option_uint32(input_args, "--check_txrx_timeout", default_check_txrx_timeout);
    uint32_t tx_pkt_dest_size_choice = (uint8_t)test_args::get_command_option_uint32(
        input_args, "--tx_pkt_dest_size_choice", default_tx_pkt_dest_size_choice);
    uint32_t tx_data_sent_per_iter_low = test_args::get_command_option_uint32(input_args, "--tx_data_sent_per_iter_low", default_tx_data_sent_per_iter_low);
    uint32_t tx_data_sent_per_iter_high = test_args::get_command_option_uint32(input_args, "--tx_data_sent_per_iter_high", default_tx_data_sent_per_iter_high);
    uint32_t fabric_command =
        test_args::get_command_option_uint32(input_args, "--fabric_command", default_fabric_command);
    uint32_t target_address =
        test_args::get_command_option_uint32(input_args, "--target_address", default_target_address);
    uint32_t atomic_increment =
        test_args::get_command_option_uint32(input_args, "--atomic_increment", default_atomic_increment);

    // assert((pkt_dest_size_choices_t)tx_pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE
    // && rx_disable_header_check || (pkt_dest_size_choices_t)tx_pkt_dest_size_choice ==
    // pkt_dest_size_choices_t::RANDOM);

    uint32_t test_device_id_l =
        test_args::get_command_option_uint32(input_args, "--device_id", default_test_device_id_l);
    uint32_t test_device_id_r =
        test_args::get_command_option_uint32(input_args, "--device_id_r", default_test_device_id_r);
    uint32_t num_src_endpoints =
        test_args::get_command_option_uint32(input_args, "--num_src_endpoints", default_num_src_endpoints);
    uint32_t num_dest_endpoints =
        test_args::get_command_option_uint32(input_args, "--num_dest_endpoints", default_num_dest_endpoints);
    uint32_t num_hops = test_args::get_command_option_uint32(input_args, "--num_hops", default_num_hops);
    allow_1st_noc_hop = test_args::has_command_option(input_args, "--allow_1st_noc_hop");
    bool bidirectional_traffic = test_args::has_command_option(input_args, "--bidirectional");

    uint32_t tx_signal_address = default_tx_signal_address;

    std::string board_type = test_args::get_command_option(input_args, "--board_type", std::string(default_board_type));

    uint32_t num_traffic_devices =
        test_args::get_command_option_uint32(input_args, "--num_devices", default_num_traffic_devices);

    uint32_t num_packets = test_args::get_command_option_uint32(input_args, "--num_packets", default_num_packets);

    uint32_t num_links = test_args::get_command_option_uint32(input_args, "--num_links", default_num_links);

    bool pass = true;
    uint32_t num_available_devices, num_allocated_devices = 0;

    std::map<string, string> defines = {
        {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
    };

    if (default_prng_seed == prng_seed) {
        std::random_device rd;
        prng_seed = rd();
    }

    global_rng.seed(prng_seed);

    // if using fixed packet sizes and num packets is specified, get the test data size
    if ((1 == tx_pkt_dest_size_choice) && (default_num_packets != num_packets)) {
        data_kb_per_tx = num_packets * max_packet_size_words * PACKET_WORD_SIZE_BYTES / 1024;
    }

    try {
        test_board_t test_board(board_type);
        num_available_devices = test_board.get_num_available_devices();

        // keep the number of test devices even
        // TODO: handle differently for multicast
        if (num_traffic_devices % 2) {
            num_traffic_devices++;
        }

        if (num_traffic_devices > num_available_devices) {
            throw std::runtime_error("Insufficient number of devices available");
        } else if (default_num_traffic_devices == num_traffic_devices) {
            num_traffic_devices = num_available_devices;
        }

        // validate left and right device IDs
        if ((default_test_device_id_l != test_device_id_l) && !test_board.is_valid_chip_id(test_device_id_l)) {
            throw std::runtime_error("Invalid left chip id");
        }
        if ((default_test_device_id_r != test_device_id_r) && !test_board.is_valid_chip_id(test_device_id_r)) {
            throw std::runtime_error("Invalid right chip id");
        }

        // if both left and right device IDs are specified, launch traffic only b/w them
        if ((default_test_device_id_l != test_device_id_l) && (default_test_device_id_r != test_device_id_r)) {
            if (test_device_id_l == test_device_id_r) {
                throw std::runtime_error("Left and right chips should be different");
            }
            test_board.unicast_map.push_back({test_device_id_l, test_device_id_r});
        } else {
            test_board.generate_unicast_map(num_hops);
        }

        std::unordered_map<chip_id_t, std::shared_ptr<test_device_t>> test_devices;
        std::vector<test_traffic_t> fabric_traffic;

        // init test devices from the list of chips
        for (auto& chip_id : test_board.physical_chip_ids) {
            test_devices[chip_id] = std::make_shared<test_device_t>(chip_id, &test_board);
        }

        // init traffic
        chip_id_t tx_chip_id, rx_chip_id;
        for (auto& [tx_chip_id, rx_chip_id] : test_board.unicast_map) {
            if (num_allocated_devices >= num_traffic_devices) {
                break;
            }

            test_traffic_t traffic(
                test_devices[tx_chip_id],
                test_devices[rx_chip_id],
                num_src_endpoints,
                num_dest_endpoints,
                target_address,
                num_hops,
                num_links);
            fabric_traffic.push_back(traffic);

            if (bidirectional_traffic) {
                test_traffic_t traffic_r(
                    test_devices[rx_chip_id],
                    test_devices[tx_chip_id],
                    num_src_endpoints,
                    num_dest_endpoints,
                    target_address,
                    num_hops,
                    num_links);
                fabric_traffic.push_back(traffic_r);
            }

            num_allocated_devices += 2;
        }

        // TODO: check this in a loop for all the devices involved in the traffic
        /*
        auto const& device_active_eth_cores =
            test_devices[test_device_id_l]->device_handle->get_active_ethernet_cores();
        if (device_active_eth_cores.size() == 0) {
            log_info(
                LogTest,
                "Device {} does not have enough active cores. Need 1 active ethernet core for this test.",
                test_device_id_l);
            for (auto& test_device : test_devices) {
                tt_metal::CloseDevice(test_device.second->device_handle);
            }
            throw std::runtime_error("Test cannot run on specified device.");
        } */

        // launch router kernels
        std::vector<uint32_t> router_compile_args = {
            (tunneler_queue_size_bytes >> 4),  // 0: rx_queue_size_words
            tunneler_test_results_addr,        // 1: test_results_addr
            tunneler_test_results_size,        // 2: test_results_size
            0,                                 // timeout_mcycles * 1000 * 1000 * 4, // 3: timeout_cycles
        };
        for (auto& [chip_id, test_device] : test_devices) {
            test_device->launch_router_kernels(router_compile_args, defines);
        }

        if (check_txrx_timeout) {
            defines["CHECK_TIMEOUT"] = "";
        }

        std::vector<uint32_t> tx_compile_args = {
            0,                           //(device->id() << 8) + src_endpoint_start_id + i,  // 0: src_endpoint_id
            num_dest_endpoints,          // 1: num_dest_endpoints
            dest_endpoint_start_id,      // 2:
            tx_queue_start_addr,         // 3: queue_start_addr_words
            (tx_queue_size_bytes >> 4),  // 4: queue_size_words
            routing_table_start_addr,    // 5: routeing table
            0,                           // tunneler_phys_core.x,        // 6: router_x
            0,                           // tunneler_phys_core.y,        // 7: router_y
            test_results_addr,           // 8: test_results_addr
            test_results_size,           // 9: test_results_size
            prng_seed,                   // 10: prng_seed
            data_kb_per_tx,              // 11: total_data_kb
            max_packet_size_words,       // 12: max_packet_size_words
            timeout_mcycles * 1000 * 1000 * 4,  // 13: timeout_cycles
            tx_skip_pkt_content_gen,            // 14: skip_pkt_content_gen
            tx_pkt_dest_size_choice,            // 15: pkt_dest_size_choice
            tx_data_sent_per_iter_low,          // 16: data_sent_per_iter_low
            tx_data_sent_per_iter_high,         // 17: data_sent_per_iter_high
            fabric_command,                     // 18: fabric_command
            target_address,                     // 19: target_address
            atomic_increment,                   // 20: atomic_increment
            tx_signal_address                   // 21: tx_signal_address
        };

        std::vector<uint32_t> rx_compile_args = {
            prng_seed,                // 0: prng seed
            data_kb_per_tx,           // 1: total data kb
            max_packet_size_words,    // 2: max packet size (in words)
            fabric_command,           // 3: fabric command
            target_address,           // 4: target address
            atomic_increment,         // 5: atomic increment
            test_results_addr,        // 6: test results addr
            test_results_size,        // 7: test results size in bytes
            tx_pkt_dest_size_choice,  // 8: pkt dest and size choice
            tx_skip_pkt_content_gen,  // 9: skip packet validation
        };

        // TODO: launch traffic kernels
        for (auto& traffic : fabric_traffic) {
            traffic.launch_kernels(
                tx_compile_args, rx_compile_args, defines, fabric_command, tx_signal_address, test_results_addr);
        }

        if (check_txrx_timeout) {
            defines.erase("CHECK_TIMEOUT");
        }

        auto start = std::chrono::system_clock::now();

        // launch programs
        for (auto& [chip_id, test_device] : test_devices) {
            tt_metal::detail::LaunchProgram(test_device->device_handle, test_device->program_handle, false);
        }

        // notify tx kernels to start transmitting
        for (auto& traffic : fabric_traffic) {
            traffic.notify_tx_workers(tx_signal_address);
        }

        // wait for rx kernels to finish
        for (auto& traffic : fabric_traffic) {
            traffic.wait_for_rx_workers_to_finish();
        }

        // terminate fabric routers
        for (auto& [chip_id, test_device] : test_devices) {
            test_device->terminate_router_kernels();
        }

        // wait for programs to exit
        for (auto& [chip_id, test_device] : test_devices) {
            tt_metal::detail::WaitProgramDone(test_device->device_handle, test_device->program_handle);
        }

        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        // collect traffic results
        for (auto& traffic : fabric_traffic) {
            pass &= traffic.collect_results(test_results_addr);
        }

        /*      TODO: Need to add these once control plane api is available to
                          get the first and last hop fabric router core.
                vector<uint32_t> router_results =
                    tt::llrt::read_hex_vec_from_core(
                        device->id(), tunneler_phys_core, tunneler_test_results_addr, 128);
                log_info(LogTest, "L Router status = {}",
           packet_queue_test_status_to_string(router_results[PQ_TEST_STATUS_INDEX])); pass &=
           (router_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

                vector<uint32_t> r_router_results =
                    tt::llrt::read_hex_vec_from_core(
                        device_r->id(), r_tunneler_phys_core, tunneler_test_results_addr, 128);
                log_info(LogTest, "R Router status = {}",
           packet_queue_test_status_to_string(r_router_results[PQ_TEST_STATUS_INDEX])); pass &=
           (r_router_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        */

        // close devices
        for (auto& [chip_id, test_device] : test_devices) {
            tt_metal::CloseDevice(test_device->device_handle);
        }

        // tally-up the packets and words from tx/rx kernels
        for (auto& traffic : fabric_traffic) {
            pass &= traffic.validate_results();
        }

        // print results
        if (pass) {
            for (auto& traffic : fabric_traffic) {
                traffic.print_result_summary();
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
