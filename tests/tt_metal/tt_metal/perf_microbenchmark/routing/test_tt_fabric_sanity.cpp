// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <nlohmann/json_fwd.hpp>
#include <stdint.h>
#include <tt_stl/span.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <tt-metalium/program.hpp>
#include "routing_test_common.hpp"
#include <tt-metalium/allocator.hpp>
#include "test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/hw/inc/fabric_routing_mode.h"

using std::vector;
using namespace tt;
using namespace tt::tt_fabric;
using json = nlohmann::json;

#define DEFAULT_NUM_HOPS (0xFFFFFFFF)

std::mt19937 global_rng;

// time based seed
uint32_t time_seed;

// decides if the tx puts the data directly on eth or if a noc hop is allowed as well
bool allow_1st_noc_hop = false;

uint32_t routing_table_addr;

// if the traffic b/w any pair of chips is bi-directional
bool bidirectional_traffic;

// benchmark test mode
bool benchmark_mode;

// push/pull buffer model
bool push_mode;

uint32_t tx_signal_address;
uint32_t host_signal_address;

// kernels
const std::string router_kernel_src = "tt_metal/fabric/impl/kernels/tt_fabric_router.cpp";
const std::string traffic_controller_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_controller.cpp";
const std::string rx_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen_rx.cpp";
std::string tx_kernel_src;

uint32_t get_noc_distance(uint32_t noc_idx, CoreCoord src, CoreCoord dest, uint32_t grid_size_x, uint32_t grid_size_y) {
    uint32_t x_dist = 0;
    uint32_t y_dist = 0;

    if (0 == noc_idx) {
        x_dist = src.x > dest.x ? (grid_size_x - src.x + dest.x) : (dest.x - src.x);
        y_dist = src.y > dest.y ? (grid_size_y - src.y + dest.y) : (dest.y - src.y);
    } else if (1 == noc_idx) {
        x_dist = src.x < dest.x ? (grid_size_x - dest.x + src.x) : (src.x - dest.x);
        y_dist = src.y < dest.y ? (grid_size_y - dest.y + src.y) : (src.y - dest.y);
    }

    return x_dist + y_dist;
}

inline std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate the range
    std::iota(range.begin(), range.end(), start);

    // shuffle the range
    std::shuffle(range.begin(), range.end(), global_rng);

    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

struct test_board_t {
    std::vector<chip_id_t> available_chip_ids;
    std::vector<chip_id_t> physical_chip_ids;
    std::vector<std::pair<chip_id_t, std::vector<chip_id_t>>> tx_rx_map;
    std::map<chip_id_t, IDevice*> device_handle_map;
    tt::tt_fabric::ControlPlane* control_plane;
    std::unique_ptr<tt::tt_fabric::ControlPlane> cp_owning_ptr;
    uint32_t num_chips_to_use;
    std::string mesh_graph_descriptor;

    test_board_t(std::string& board_type_) {
        if ("n300" == board_type_) {
            mesh_graph_descriptor = "n300_mesh_graph_descriptor.yaml";
            num_chips_to_use = 2;

            if (num_chips_to_use != tt_metal::GetNumAvailableDevices()) {
                throw std::runtime_error("Not found the expected 2 chips for n300");
            }

            for (auto i = 0; i < num_chips_to_use; i++) {
                available_chip_ids.push_back(i);
            }
        } else if ("t3k" == board_type_) {
            mesh_graph_descriptor = "t3k_mesh_graph_descriptor.yaml";
            num_chips_to_use = 8;

            if (num_chips_to_use != tt_metal::GetNumAvailableDevices()) {
                throw std::runtime_error("Not found the expected 8 chips for t3k");
            }

            for (auto i = 0; i < num_chips_to_use; i++) {
                available_chip_ids.push_back(i);
            }
        } else if ("glx8" == board_type_) {
            _init_galaxy_board(8);
        } else if ("glx16" == board_type_) {
            _init_galaxy_board(16);
        } else if ("glx32" == board_type_) {
            _init_galaxy_board(32);
        } else if ("ubb" == board_type_) {
            _init_galaxy_board(32, true);
        } else {
            throw std::runtime_error("Unsupported board");
        }

        // TODO: should we support odd number of chips (for unicast)?
        if ((available_chip_ids.size() % 2) != 0) {
            throw std::runtime_error("Odd number of chips detected, not supported currently");
        }

        tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::CUSTOM);

        device_handle_map = tt::tt_metal::detail::CreateDevices(available_chip_ids);
        control_plane = &tt::tt_metal::MetalContext::instance().get_control_plane();
        control_plane->write_routing_tables_to_all_chips();

        if (num_chips_to_use != available_chip_ids.size()) {
            // initialize partial board to get the set of physical chip IDs for fabric kernels
            if ("tg_mesh_graph_descriptor.yaml" == mesh_graph_descriptor) {
                _init_partial_galaxy_board(num_chips_to_use);
            }
        } else {
            physical_chip_ids = available_chip_ids;
        }
    }

    void _init_galaxy_board(uint32_t num_chips, bool all_pcie = false) {
        // TODO: add support for quanta galaxy variant
        if (all_pcie) {
            mesh_graph_descriptor = "quanta_galaxy_mesh_graph_descriptor.yaml";
        } else {
            mesh_graph_descriptor = "tg_mesh_graph_descriptor.yaml";
        }
        num_chips_to_use = num_chips;

        // do run time check for number of available chips
        if (tt_metal::GetNumAvailableDevices() < 32) {
            throw std::runtime_error("Not a valid galaxy board since it has less than 32 chips");
        }

        if (all_pcie) {
            for (auto i = 0; i < tt_metal::GetNumAvailableDevices(); i++) {
                available_chip_ids.push_back(i);
            }
        } else {
            for (auto i = 4; i < tt_metal::GetNumAvailableDevices() + 4; i++) {
                available_chip_ids.push_back(i);
            }
        }
    }

    void _init_control_plane(const std::string& mesh_graph_descriptor) {
        try {
            const std::filesystem::path mesh_graph_desc_path =
                std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                "tt_metal/fabric/mesh_graph_descriptors" / mesh_graph_descriptor;
            cp_owning_ptr = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
            control_plane = cp_owning_ptr.get();
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "{}", e.what());
        }
    }

    void _init_partial_galaxy_board(uint32_t num_chips) {
        uint32_t mesh_id = 4;
        uint32_t start_row_idx, start_row_max_idx, num_rows;
        chip_id_t physical_chip_id;

        // Init valid and available chip ids
        // consecutive rows of galaxy chips are chosen at random
        // following is the arrangement with virtual fabric_node_id
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
                physical_chip_id = control_plane->get_physical_chip_id_from_fabric_node_id(FabricNodeId(MeshId{mesh_id}, j));
                physical_chip_ids.push_back(physical_chip_id);
            }
        }
    }

    // TODO: This only supports 1d mcast right now, needs to be updated to support 2D mcast
    // Note that this currently only considers intra-mesh mcast
    // physical_start_chip_id here refers to the sender, not the mcast origin due to how we count depth
    std::vector<chip_id_t> get_physical_mcast_chip_ids(
        chip_id_t physical_start_chip_id, const std::unordered_map<RoutingDirection, uint32_t>& mcast_depth) {
        std::vector<chip_id_t> physical_dsts;
        // APIs use mesh chip id, so convert physical chip id to mesh chip id
        auto fabric_node_id = this->get_fabric_node_id(physical_start_chip_id);
        bool valid = true;
        for (const auto& [routing_direction, num_hops_in_direction] : mcast_depth) {
            for (auto j = 0; j < num_hops_in_direction; j++) {
                auto neighbors = this->get_intra_chip_neighbors(fabric_node_id, routing_direction);
                if (neighbors.empty()) {
                    valid = false;
                    break;
                }
                // Assumes all neighbors are the same chip
                fabric_node_id.chip_id = neighbors[0];
                // convert mesh chip id to physical chip id
                physical_dsts.push_back(this->control_plane->get_physical_chip_id_from_fabric_node_id(fabric_node_id));
            }
            if (!valid) {
                break;
            }
        }
        if (valid) {
            return physical_dsts;
        } else {
            return {};
        }
    }

    void generate_tx_rx_map(
        uint32_t num_hops, bool mcast, const std::unordered_map<RoutingDirection, uint32_t>& mcast_depth) {
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
            if (mcast) {
                for (auto i = 0; i < physical_chip_ids.size(); i++) {
                    auto physical_mcast_chip_ids = this->get_physical_mcast_chip_ids(physical_chip_ids[i], mcast_depth);
                    if (!physical_mcast_chip_ids.empty()) {
                        tx_rx_map.push_back({physical_chip_ids[i], std::move(physical_mcast_chip_ids)});
                        // Generate only one mcast for now to avoid overlapping cores
                        break;
                    }
                }
                TT_FATAL(!tx_rx_map.empty(), "Failed to generate multicast map");
                return;
            }
            for (auto i = 0; i < physical_chip_ids.size(); i += 2) {
                tx_rx_map.push_back({physical_chip_ids[i], {physical_chip_ids[i + 1]}});
            }
            return;
        }

        // for each physical chip id, store the neighbors
        // TDOD: update the logic to find inter-mesh neighbors
        for (auto chip_id : physical_chip_ids) {
            auto neighbors =
                tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                    chip_id);
            for (const auto& [neighbor, cores] : neighbors) {
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
            chip_id_t selected_chip_id{};

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

            if (UINT32_MAX == temp_neighbor_cnt) {
                continue;
                // TODO: re-enable once the algo is fixed
                // throw std::runtime_error("No neighbor found for this chip");
            }

            if (mcast) {
                // TODO: This assumes line mcast from neighbor with 1 hop
                auto physical_mcast_chip_ids = this->get_physical_mcast_chip_ids(chip_id, mcast_depth);
                if (!physical_mcast_chip_ids.empty() && (physical_mcast_chip_ids[0] == selected_chip_id)) {
                    tx_rx_map.push_back({chip_id, std::move(physical_mcast_chip_ids)});
                } else {
                    continue;
                }
            } else {
                tx_rx_map.push_back({chip_id, {selected_chip_id}});
            }

            // remove selected chip as it should not be picked again
            chip_n_hop_neighbors.erase(selected_chip_id);

            // remove the entry for current chip as it should not be picked again
            chip_n_hop_neighbors.erase(chip_id);
        }

        // error out if no valid tx rx mapping was found
        // We should only be able to hit this assertion when looking for mcast destinations
        if (!tx_rx_map.size()) {
            throw std::runtime_error("No valid tx rx mapping found");
        }
    }

    inline uint32_t get_num_available_devices() { return physical_chip_ids.size(); }

    inline bool is_valid_chip_id(chip_id_t physical_chip_id) {
        auto it = std::find(physical_chip_ids.begin(), physical_chip_ids.end(), physical_chip_id);
        return (it != physical_chip_ids.end());
    }

    inline tt_metal::IDevice* get_device_handle(chip_id_t physical_chip_id) {
        if (is_valid_chip_id(physical_chip_id)) {
            return device_handle_map[physical_chip_id];
        } else {
            throw std::runtime_error("Invalid physical chip id");
        }
    }

    inline FabricNodeId get_fabric_node_id(chip_id_t physical_chip_id) {
        return control_plane->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    }

    inline std::vector<std::pair<chip_id_t, chan_id_t>> get_route_to_chip(
        FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) {
        return control_plane->get_fabric_route(src_fabric_node_id, dst_fabric_node_id, src_chan_id);
    }

    inline stl::Span<const chip_id_t> get_intra_chip_neighbors(
        FabricNodeId fabric_node_id, RoutingDirection routing_direction) {
        return control_plane->get_intra_chip_neighbors(fabric_node_id, routing_direction);
    }

    inline routing_plane_id_t get_routing_plane_from_chan(chip_id_t physical_chip_id, chan_id_t eth_chan) {
        const auto fabric_node_id = this->get_fabric_node_id(physical_chip_id);
        return control_plane->get_routing_plane_id(fabric_node_id, eth_chan);
    }

    inline eth_chan_directions get_eth_chan_direction(FabricNodeId fabric_node_id, chan_id_t eth_chan) {
        auto active_eth_chans = control_plane->get_active_fabric_eth_channels(fabric_node_id);
        for (const auto& [eth_chan_, direction] : active_eth_chans) {
            if (eth_chan_ == eth_chan) {
                return direction;
            }
        }
        TT_THROW("Cannot find ethernet channel direction");
    }

    inline void close_devices() { tt::tt_metal::detail::CloseDevices(device_handle_map); }
};

struct test_device_t {
    chip_id_t physical_chip_id;
    test_board_t* board_handle;
    tt_metal::IDevice* device_handle;
    tt_metal::Program program_handle;
    std::vector<CoreCoord> worker_logical_cores;
    std::vector<CoreCoord> router_logical_cores;
    std::vector<CoreCoord> router_virtual_cores;
    CoreCoord core_range_start_virtual;
    CoreCoord core_range_end_virtual;
    MeshId mesh_id;
    chip_id_t logical_chip_id;
    uint32_t master_router_idx;
    uint32_t mesh_chip_id = 0;
    uint32_t router_mask = 0;
    metal_SocDescriptor soc_desc;
    std::unordered_map<chan_id_t, std::vector<std::pair<uint32_t, CoreCoord>>>
        router_worker_map;  // router chan to worker logical cores

    test_device_t(chip_id_t chip_id_, test_board_t* board_handle_) {
        physical_chip_id = chip_id_;
        board_handle = board_handle_;

        device_handle = board_handle->get_device_handle(physical_chip_id);
        program_handle = tt_metal::CreateProgram();
        auto fabric_node_id = board_handle->get_fabric_node_id(physical_chip_id);
        mesh_id = fabric_node_id.mesh_id;
        logical_chip_id = fabric_node_id.chip_id;
        mesh_chip_id = (*mesh_id << 16 | logical_chip_id);
        soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(physical_chip_id);

        // initalize list of worker cores in 8X8 grid
        // TODO: remove hard-coding
        for (auto i = 0; i < 8; i++) {
            for (auto j = 0; j < 8; j++) {
                worker_logical_cores.push_back(CoreCoord({i, j}));
            }
        }

        core_range_start_virtual = device_handle->worker_core_from_logical_core(CoreCoord(0, 0));
        core_range_end_virtual = device_handle->worker_core_from_logical_core(CoreCoord(7, 7));

        // populate router cores
        auto neighbors =
            tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                device_handle->id());
        for (const auto& [neighbor_chip, connected_logical_cores] : neighbors) {
            if (!(board_handle->is_valid_chip_id(neighbor_chip))) {
                continue;
            }

            for (auto logical_core : connected_logical_cores) {
                router_logical_cores.push_back(logical_core);
                router_virtual_cores.push_back(device_handle->ethernet_core_from_logical_core(logical_core));
                router_mask += 0x1 << logical_core.y;
            }
        }

        if (router_virtual_cores.empty()) {
            log_fatal(LogTest, "Couldnt find any ethernet core for device: {}", physical_chip_id);
            throw std::runtime_error("Need atleast 1 ethernet core to run fabric");
        }

        if (benchmark_mode) {
            _generate_router_worker_map();
        }

        master_router_idx = 0;
    }

    void create_router_kernels(std::vector<uint32_t>& compile_args, std::map<string, string>& defines) {
        uint32_t num_routers = router_logical_cores.size();
        std::vector<uint32_t> zero_buf(1, 0);

        for (auto i = 0; i < num_routers; i++) {
            std::vector<uint32_t> router_compile_args = compile_args;
            // setup run time args
            std::vector<uint32_t> runtime_args = {
                num_routers,                               // 0: number of active fabric routers
                router_mask,                               // 1: active fabric router mask
                router_logical_cores[master_router_idx].y  // 2: master router eth chan
            };

            // pass is_master flag as compile arg, index 0 is master
            if (master_router_idx == i) {
                router_compile_args.push_back(1);
            } else {
                router_compile_args.push_back(0);
            }

            uint32_t direction = board_handle->get_eth_chan_direction(
                FabricNodeId(mesh_id, logical_chip_id),
                soc_desc.logical_eth_core_to_chan_map.at(router_logical_cores[i]));
            router_compile_args.push_back(direction);

            // initialize the semaphore
            auto fabric_router_sync_sem_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
            tt::llrt::write_hex_vec_to_core(
                device_handle->id(), router_virtual_cores[i], zero_buf, fabric_router_sync_sem_addr);

            auto kernel = tt_metal::CreateKernel(
                program_handle,
                router_kernel_src,
                router_logical_cores[i],
                tt_metal::EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0, .compile_args = router_compile_args, .defines = defines});

            tt_metal::SetRuntimeArgs(program_handle, kernel, router_logical_cores[i], runtime_args);
        }
    }

    void wait_for_router_sync() {
        uint32_t master_router_status = 0;
        uint32_t expected_val = router_logical_cores.size();
        auto fabric_router_sync_sem_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
        while (expected_val != master_router_status) {
            master_router_status = tt::llrt::read_hex_vec_from_core(
                device_handle->id(), router_virtual_cores[master_router_idx], fabric_router_sync_sem_addr, 4)[0];
        }
    }

    void terminate_router_kernels() {
        std::vector<uint32_t> zero_buf(1, 0);
        auto fabric_router_sync_sem_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
        tt::llrt::write_hex_vec_to_core(
            device_handle->id(), router_virtual_cores[master_router_idx], zero_buf, fabric_router_sync_sem_addr);
    }

    std::vector<CoreCoord> select_random_worker_cores(uint32_t count) {
        std::vector<CoreCoord> result;

        // shuffle the list of cores
        std::shuffle(worker_logical_cores.begin(), worker_logical_cores.end(), global_rng);

        // return and delete the selected cores
        for (auto i = 0; i < count; i++) {
            result.push_back(worker_logical_cores.back());
            worker_logical_cores.pop_back();
        }

        return result;
    }

    inline uint32_t get_endpoint_id(CoreCoord& logical_core) {
        return ((device_handle->id()) << 8) | ((logical_core.x) << 4) | (logical_core.y);
    }

    inline uint32_t get_noc_offset(CoreCoord& logical_core) {
        CoreCoord phys_core = device_handle->worker_core_from_logical_core(logical_core);
        return tt_metal::MetalContext::instance().hal().noc_xy_encoding(phys_core.x, phys_core.y);
    }

    void get_available_router_cores(
        uint32_t num_hops,
        std::shared_ptr<test_device_t>& rx_device,
        std::vector<chan_id_t>& src_routers,
        std::vector<chan_id_t>& dest_routers) {
        // shortest route possible with least number of internal noc hops
        uint32_t shortest_route_length = 2 * num_hops - 1;
        bool select_router = false;

        // get the potential routers based on the fabric path
        for (auto i = 0; i < router_logical_cores.size(); i++) {
            std::vector<std::pair<chip_id_t, chan_id_t>> route;
            std::set<chip_id_t> chips_in_route;
            chan_id_t src_eth_chan = soc_desc.logical_eth_core_to_chan_map.at(router_logical_cores[i]);
            chips_in_route.insert(physical_chip_id);
            route = _get_route_to_chip(rx_device->mesh_id, rx_device->logical_chip_id, src_eth_chan);
            if (route.empty()) {
                continue;
            }

            auto dest_eth_chan = route.back().second;

            if (DEFAULT_NUM_HOPS == num_hops) {
                // no need to check for path length for default case, all routers can be used
                select_router = true;
            } else {
                for (auto& [chip_, chan_] : route) {
                    chips_in_route.insert(chip_);
                }

                // including the origin chip, the distinct number of chips should be num_hops + 1
                // if 1st noc hop at tx is allowed, the path will be longer
                if ((chips_in_route.size() == num_hops + 1) &&
                    (allow_1st_noc_hop || route.size() == shortest_route_length)) {
                    select_router = true;
                }
            }

            if (select_router) {
                src_routers.push_back(src_eth_chan);
                dest_routers.push_back(dest_eth_chan);
            }
        }

        // throw error if no potential router core found
        if (src_routers.size() == 0) {
            log_fatal(LogTest, "No router cores found for num hops: {}, on device: {}", num_hops, physical_chip_id);
            throw std::runtime_error("No router cores found for specified num hops");
        }
    }

    std::vector<std::tuple<chan_id_t, uint32_t, CoreCoord>> select_worker_cores(
        const std::vector<chan_id_t>& router_cores,
        uint32_t num_links,
        uint32_t count,
        uint32_t skip_first_n_workers = 0) {
        std::vector<std::tuple<chan_id_t, uint32_t, CoreCoord>> result;
        uint32_t link_idx = 0;
        if (benchmark_mode) {
            // temp map to keep a track of indices to start lookup from
            std::unordered_map<chan_id_t, uint32_t> router_worker_idx;
            for (auto i = 0; i < count; i++) {
                if (link_idx == num_links) {
                    link_idx = 0;
                }
                auto router = router_cores[link_idx++];
                auto worker_list = router_worker_map[router];

                if (router_worker_idx.count(router) == 0) {
                    // the highest priority workers are reserved for tx kernels
                    // skip them for choosing cores for rx kernels in bi-directional traffic mode
                    router_worker_idx[router] = skip_first_n_workers;
                }

                uint32_t j;
                for (j = router_worker_idx[router]; j < worker_list.size(); j++) {
                    // need to check if the next high priority worker for this router has already been picked by another
                    // router
                    auto it =
                        std::find(worker_logical_cores.begin(), worker_logical_cores.end(), worker_list[j].second);
                    if (it != worker_logical_cores.end()) {
                        result.emplace_back(std::make_tuple(router, worker_list[j].first, worker_list[j].second));
                        worker_logical_cores.erase(it);
                        break;
                    }
                }
                router_worker_idx[router] = j;
            }
        } else {
            auto worker_cores = select_random_worker_cores(count);
            for (auto& core : worker_cores) {
                if (link_idx == num_links) {
                    link_idx = 0;
                }
                // default to noc 0
                result.emplace_back(std::make_tuple(router_cores[link_idx++], 0, core));
            }
        }

        return result;
    }

    inline std::vector<std::pair<chip_id_t, chan_id_t>> _get_route_to_chip(
        MeshId dst_mesh_id, chip_id_t dst_chip_id, chan_id_t src_chan_id) {
        return board_handle->get_route_to_chip(
            FabricNodeId(mesh_id, logical_chip_id), FabricNodeId(dst_mesh_id, dst_chip_id), src_chan_id);
    }

    // generates a map fo preferred worker cores for a given router based on the physical distance
    // the worker cores physically close to a router core get higher priority
    void _generate_router_worker_map() {
        std::vector<CoreCoord> router_phys_cores;
        std::vector<CoreCoord> worker_phys_cores;
        CoreCoord grid_size = soc_desc.grid_size;
        uint32_t grid_size_x = grid_size.x;
        uint32_t grid_size_y = grid_size.y;

        for (auto& core : router_logical_cores) {
            router_phys_cores.push_back(soc_desc.get_physical_core_from_logical_core(core, CoreType::ETH));
        }

        for (auto& core : worker_logical_cores) {
            worker_phys_cores.push_back(soc_desc.get_physical_core_from_logical_core(core, CoreType::WORKER));
        }

        CoreCoord router_phys_core, worker_phys_core;
        uint32_t noc_dist, noc_index, noc0_dist, noc1_dist;
        for (auto i = 0; i < router_logical_cores.size(); i++) {
            router_phys_core = router_phys_cores[i];
            chan_id_t eth_chan = soc_desc.logical_eth_core_to_chan_map.at(router_logical_cores[i]);
            std::vector<std::pair<uint32_t, std::pair<uint32_t, CoreCoord>>> temp_map;
            for (auto j = 0; j < worker_logical_cores.size(); j++) {
                worker_phys_core = worker_phys_cores[j];
                noc0_dist = get_noc_distance(0, worker_phys_core, router_phys_core, grid_size_x, grid_size_y);
                noc1_dist = get_noc_distance(1, worker_phys_core, router_phys_core, grid_size_x, grid_size_y);
                if (noc0_dist <= noc1_dist) {
                    noc_dist = noc0_dist;
                    noc_index = 0;
                } else {
                    noc_dist = noc1_dist;
                    noc_index = 1;
                }

                temp_map.emplace_back(std::make_pair(noc_dist, std::make_pair(noc_index, worker_logical_cores[j])));
            }
            std::sort(temp_map.begin(), temp_map.end());

            for (auto& [noc_dist, pair] : temp_map) {
                router_worker_map[eth_chan].push_back(pair);
            }
        }
    }

    inline stl::Span<const chip_id_t> get_intra_chip_neighbors(RoutingDirection routing_direction) {
        return board_handle->get_intra_chip_neighbors(FabricNodeId(mesh_id, logical_chip_id), routing_direction);
    }
};

struct test_traffic_t {
    std::shared_ptr<test_device_t> tx_device;
    std::vector<std::shared_ptr<test_device_t>> rx_devices;
    uint32_t num_tx_workers;
    uint32_t num_rx_workers;
    uint32_t target_address;
    std::vector<std::tuple<chan_id_t, uint32_t, CoreCoord>> tx_workers;
    std::vector<std::tuple<chan_id_t, uint32_t, CoreCoord>> rx_workers;
    std::vector<CoreCoord> tx_virtual_cores;
    std::vector<CoreCoord> rx_virtual_cores;
    CoreCoord controller_logical_core;
    CoreCoord controller_virtual_core;
    std::vector<uint32_t> tx_to_rx_map;
    std::vector<std::vector<uint32_t>> rx_to_tx_map;
    std::vector<uint32_t> tx_to_rx_address_map;
    std::vector<std::vector<uint32_t>> rx_to_tx_address_map;
    std::vector<std::vector<uint32_t>> tx_results;
    std::vector<std::vector<std::vector<uint32_t>>> rx_results;
    uint32_t test_results_address;
    uint32_t rx_buf_size;
    uint32_t num_links_to_use;
    uint32_t link_idx = 0;
    bool sync_with_remote_controller_kernel = false;
    std::optional<chan_id_t> controller_outbound_eth_chan;
    std::optional<uint32_t> remote_controller_noc_encoding;
    std::optional<uint32_t> remote_controller_mesh_chip_id;

    test_traffic_t(
        std::shared_ptr<test_device_t>& tx_device_,
        std::vector<std::shared_ptr<test_device_t>>& rx_devices_,
        uint32_t num_src_endpoints,
        uint32_t num_dest_endpoints,
        uint32_t target_address_,
        uint32_t num_hops,
        uint32_t num_links_) {
        tx_device = tx_device_;
        rx_devices = rx_devices_;
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

        std::vector<chan_id_t> src_routers;
        std::vector<chan_id_t> dest_routers;
        // For Unicast there is only one rx device
        // For mcast, this only supports line mcast, we pass the last device as the rx device
        tx_device->get_available_router_cores(num_hops, *rx_devices.rbegin(), src_routers, dest_routers);
        num_links_to_use = std::min(num_links_, (uint32_t)src_routers.size());

        _generate_tx_to_rx_mapping();
        _generate_target_addresses();

        tx_workers = tx_device->select_worker_cores(src_routers, num_links_to_use, num_tx_workers);
        uint32_t num_cores_to_skip = 0;
        if (bidirectional_traffic) {
            // for bi-directional traffic leave the higher priority cores on the rx chip for tx kernels
            num_cores_to_skip = (num_rx_workers + num_links_to_use - 1) / num_links_to_use;
        }
        // Assumes uniform worker grid across receiver chips
        rx_workers =
            rx_devices[0]->select_worker_cores(dest_routers, num_links_to_use, num_rx_workers, num_cores_to_skip);

        // TODO: not the most optimum selection, might impact somewhat in bidirectional mode
        controller_logical_core = tx_device->select_random_worker_cores(1)[0];
        controller_virtual_core = tx_device->device_handle->worker_core_from_logical_core(controller_logical_core);

        for (auto& [router, noc, worker] : tx_workers) {
            tx_virtual_cores.push_back(tx_device->device_handle->worker_core_from_logical_core(worker));
        }

        for (auto& [router, noc, worker] : rx_workers) {
            rx_virtual_cores.push_back(rx_devices[0]->device_handle->worker_core_from_logical_core(worker));
        }
    }

    void set_remote_controller(test_traffic_t& reverse_traffic) {
        sync_with_remote_controller_kernel = true;
        auto remote_tx_device = reverse_traffic.tx_device;
        controller_outbound_eth_chan = std::get<0>(tx_workers[0]);
        remote_controller_noc_encoding = remote_tx_device->get_noc_offset(reverse_traffic.controller_logical_core);
        remote_controller_mesh_chip_id = remote_tx_device->mesh_chip_id;
    }

    void create_kernels(
        std::vector<uint32_t>& tx_compile_args,
        std::vector<uint32_t>& rx_compile_args,
        std::map<string, string>& defines,
        uint32_t fabric_command,
        uint32_t test_results_address_) {
        CoreCoord tx_core, rx_core;
        tt_metal::NOC noc_id;
        std::vector<uint32_t> zero_buf(2, 0);
        chan_id_t eth_chan;
        uint32_t mesh_chip_id = rx_devices[0]->mesh_chip_id;

        // update the test results address, which will be used later for polling, collecting results
        test_results_address = test_results_address_;

        {
            uint32_t mcast_encoding = tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
                tx_device->core_range_start_virtual.x,
                tx_device->core_range_start_virtual.y,
                tx_device->core_range_end_virtual.x,
                tx_device->core_range_end_virtual.y);

            // launch controller kernel
            // TODO: remove hardcoding
            std::vector<uint32_t> runtime_args = {
                time_seed,                          // 0: time based seed
                num_tx_workers,                     // 1: number of workers for mcast
                tx_signal_address,                  // 2: address to send signal on to workers
                host_signal_address,                // 3: address to receive signal from host
                64,                                 // 4: num mcast dest
                mcast_encoding,                     // 5: mcast dest noc encoding
                sync_with_remote_controller_kernel  // 6: if need to sync with the remote controller kernel
            };

            // if need to sync with remote controller
            if (sync_with_remote_controller_kernel) {
                runtime_args.push_back(remote_controller_mesh_chip_id.value());
                runtime_args.push_back(remote_controller_noc_encoding.value());
                runtime_args.push_back(controller_outbound_eth_chan.value());
            }

            // zero out the signal address
            tt::llrt::write_hex_vec_to_core(
                tx_device->physical_chip_id, controller_virtual_core, zero_buf, tx_signal_address);

            // zero out host sync address
            tt::llrt::write_hex_vec_to_core(
                tx_device->physical_chip_id, controller_virtual_core, zero_buf, host_signal_address);

            log_info(
                LogTest,
                "[Device: Phys: {}, Logical: {}] Controller running on: logical: x={},y={}; virtual: x={},y={}",
                tx_device->physical_chip_id,
                (uint32_t)tx_device->logical_chip_id,
                controller_logical_core.x,
                controller_logical_core.y,
                controller_virtual_core.x,
                controller_virtual_core.y);

            auto kernel = tt_metal::CreateKernel(
                tx_device->program_handle,
                traffic_controller_src,
                {controller_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = defines});

            tt_metal::SetRuntimeArgs(tx_device->program_handle, kernel, controller_logical_core, runtime_args);
        }

        // launch tx kernels
        for (auto i = 0; i < num_tx_workers; i++) {
            eth_chan = std::get<0>(tx_workers[i]);
            noc_id = (std::get<1>(tx_workers[i]) == 0) ? tt_metal::NOC::NOC_0 : tt_metal::NOC::NOC_1;
            tx_core = std::get<2>(tx_workers[i]);
            rx_core = std::get<2>(rx_workers[tx_to_rx_map[i]]);

            auto routing_plane =
                tx_device->board_handle->get_routing_plane_from_chan(tx_device->physical_chip_id, eth_chan);

            // setup runtime args
            std::vector<uint32_t> runtime_args = {
                time_seed,                                           // 0: time based seed
                tx_device->get_endpoint_id(tx_core),                 // 1: src_endpoint_id
                rx_devices[0]->get_noc_offset(rx_core),              // 2: dest_noc_offset
                tx_device->get_noc_offset(controller_logical_core),  // 3: controller noc offset
                eth_chan,                                            // 4: outbound eth chan
                mesh_chip_id,                                        // 5: mesh and chip id
                rx_buf_size,                                         // 6: space in rx's L1
            };

            if (ASYNC_WR & fabric_command) {
                runtime_args.push_back(tx_to_rx_address_map[i]);
            }

            // zero out the signal address
            tt::llrt::write_hex_vec_to_core(
                tx_device->physical_chip_id, tx_virtual_cores[i], zero_buf, tx_signal_address);

            log_info(
                LogTest,
                "[Device: Phys: {}, Logical: {}] TX running on: logical: x={},y={}; virtual: x={},y={}, Eth chan: {}",
                tx_device->physical_chip_id,
                (uint32_t)tx_device->logical_chip_id,
                tx_core.x,
                tx_core.y,
                tx_virtual_cores[i].x,
                tx_virtual_cores[i].y,
                (uint32_t)eth_chan);
            auto kernel = tt_metal::CreateKernel(
                tx_device->program_handle,
                tx_kernel_src,
                {tx_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = noc_id,
                    .compile_args = tx_compile_args,
                    .defines = defines});

            tt_metal::SetRuntimeArgs(tx_device->program_handle, kernel, tx_core, runtime_args);
        }

        // launch rx kernels
        for (auto i = 0; i < num_rx_workers; i++) {
            noc_id = (std::get<1>(rx_workers[i]) == 0) ? tt_metal::NOC::NOC_0 : tt_metal::NOC::NOC_1;
            rx_core = std::get<2>(rx_workers[i]);

            // setup runtime args
            std::vector<uint32_t> runtime_args = {
                time_seed,               // 0: time based seed
                rx_to_tx_map[i].size(),  // 1: num tx workers
                rx_buf_size              // 2: space available in L1
            };

            if (ASYNC_WR & fabric_command) {
                // push the src endpoint IDs
                for (auto j : rx_to_tx_map[i]) {
                    runtime_args.push_back(tx_device->get_endpoint_id(std::get<2>(tx_workers[j])));
                }

                // push target address per src
                for (auto address : rx_to_tx_address_map[i]) {
                    runtime_args.push_back(address);
                }
            } else if (ATOMIC_INC == fabric_command) {
                for (const auto& rx_device : rx_devices) {
                    tt::llrt::write_hex_vec_to_core(
                        rx_device->physical_chip_id, rx_virtual_cores[i], zero_buf, target_address);
                }
            }

            for (const auto& rx_device : rx_devices) {
                // zero out the test results address, which will be used for polling
                tt::llrt::write_hex_vec_to_core(
                    rx_device->physical_chip_id, rx_virtual_cores[i], zero_buf, test_results_address);

                log_info(
                    LogTest,
                    "[Device: Phys: {}, Logical: {}] RX kernel running on: logical: x={},y={}; virtual: x={},y={}",
                    rx_device->physical_chip_id,
                    (uint32_t)rx_device->logical_chip_id,
                    rx_core.x,
                    rx_core.y,
                    rx_virtual_cores[i].x,
                    rx_virtual_cores[i].y);
                auto kernel = tt_metal::CreateKernel(
                    rx_device->program_handle,
                    rx_kernel_src,
                    {rx_core},
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = noc_id,
                        .compile_args = rx_compile_args,
                        .defines = defines});

                tt_metal::SetRuntimeArgs(rx_device->program_handle, kernel, rx_core, runtime_args);
            }
        }
    }

    void notify_tx_controller() {
        std::vector<uint32_t> start_signal(1, 1);
        tt::llrt::write_hex_vec_to_core(
            tx_device->physical_chip_id, controller_virtual_core, start_signal, host_signal_address);
    }

    void notify_tx_workers(uint32_t address) {
        std::vector<uint32_t> start_signal(1, 1);
        for (auto core : tx_virtual_cores) {
            tt::llrt::write_hex_vec_to_core(tx_device->physical_chip_id, core, start_signal, address);
        }
    }

    void wait_for_tx_workers_to_finish() {
        for (auto& tx_core : tx_virtual_cores) {
            while (true) {
                auto tx_status =
                    tt::llrt::read_hex_vec_from_core(tx_device->physical_chip_id, tx_core, test_results_address, 4);
                if ((tx_status[0] & 0xFFFF) != 0) {
                    break;
                }
            }
        }
    }

    void wait_for_rx_workers_to_finish() {
        for (const auto& rx_device : rx_devices) {
            for (auto& rx_core : rx_virtual_cores) {
                while (true) {
                    auto rx_status =
                        tt::llrt::read_hex_vec_from_core(rx_device->physical_chip_id, rx_core, test_results_address, 4);
                    if ((rx_status[0] & 0xFFFF) != 0) {
                        break;
                    }
                }
            }
        }
    }

    bool collect_results(uint32_t test_results_address) {
        bool pass = true;

        // collect tx results
        for (uint32_t i = 0; i < num_tx_workers; i++) {
            tx_results.push_back(tt::llrt::read_hex_vec_from_core(
                tx_device->physical_chip_id, tx_virtual_cores[i], test_results_address, 128));
            log_info(
                LogTest,
                "[Device: Phys: {}, Logical: {}] TX{} status = {}",
                tx_device->physical_chip_id,
                (uint32_t)tx_device->logical_chip_id,
                i,
                tt_fabric_status_to_string(tx_results[i][TT_FABRIC_STATUS_INDEX]));
            pass &= (tx_results[i][TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);
        }

        // collect rx results
        rx_results.resize(rx_devices.size());
        for (uint32_t d = 0; d < rx_devices.size(); d++) {
            for (uint32_t i = 0; i < num_rx_workers; i++) {
                rx_results[d].push_back(tt::llrt::read_hex_vec_from_core(
                    rx_devices[d]->physical_chip_id, rx_virtual_cores[i], test_results_address, 128));
                log_info(
                    LogTest,
                    "[Device: Phys: {}, Logical: {}] RX{} status = {}",
                    rx_devices[d]->physical_chip_id,
                    (uint32_t)rx_devices[d]->logical_chip_id,
                    i,
                    tt_fabric_status_to_string(rx_results[d][i][TT_FABRIC_STATUS_INDEX]));
                pass &= (rx_results[d][i][TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);
            }
        }

        return pass;
    }

    bool validate_results() {
        bool pass = true;
        uint64_t num_tx_words, num_tx_packets;

        // tally-up data words and number of packets from rx and tx
        for (uint32_t d = 0; d < rx_devices.size(); d++) {
            for (uint32_t i = 0; i < num_rx_workers; i++) {
                num_tx_words = 0;
                num_tx_packets = 0;

                for (auto j : rx_to_tx_map[i]) {
                    num_tx_words += get_64b_result(tx_results[j], TT_FABRIC_WORD_CNT_INDEX);
                    num_tx_packets += get_64b_result(tx_results[j], TX_TEST_IDX_NPKT);
                }
                pass &= (get_64b_result(rx_results[d][i], TT_FABRIC_WORD_CNT_INDEX) == num_tx_words);
                pass &= (get_64b_result(rx_results[d][i], TX_TEST_IDX_NPKT) == num_tx_packets);

                if (!pass) {
                    log_fatal(
                        LogTest,
                        "Result validation failed b/w TX [Device: Phys: {}, Logical: {}] and RX [Device: Phys: {}, "
                        "Logical: {}]",
                        tx_device->physical_chip_id,
                        (uint32_t)tx_device->logical_chip_id,
                        rx_devices[d]->physical_chip_id,
                        (uint32_t)rx_devices[d]->logical_chip_id);
                    break;
                }
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
            uint64_t tx_words_sent = get_64b_result(tx_results[i], TT_FABRIC_WORD_CNT_INDEX);
            total_tx_words_sent += tx_words_sent;
            uint64_t tx_elapsed_cycles = get_64b_result(tx_results[i], TT_FABRIC_CYCLES_INDEX);
            double tx_bw = ((double)tx_words_sent) * PACKET_WORD_SIZE_BYTES / tx_elapsed_cycles;
            total_tx_bw += tx_bw;
            uint64_t iter = get_64b_result(tx_results[i], TT_FABRIC_ITER_INDEX);
            max_tx_elapsed_cycles = std::max(max_tx_elapsed_cycles, tx_elapsed_cycles);
            // uint64_t zero_data_sent_iter = get_64b_result(tx_results[i], TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER);
            // uint64_t few_data_sent_iter = get_64b_result(tx_results[i], TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER);
            // uint64_t many_data_sent_iter = get_64b_result(tx_results[i], TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER);
            // uint64_t num_packets = get_64b_result(tx_results[i], TX_TEST_IDX_NPKT);
            // double bytes_per_pkt = static_cast<double>(tx_words_sent) * PACKET_WORD_SIZE_BYTES /
            // static_cast<double>(num_packets);

            log_info(
                LogTest,
                "[Device: Phys: {}, Logical: {}] TX {} words sent: {}, elapsed cycles: {} -> BW: {:.2f} B/cycle",
                tx_device->physical_chip_id,
                tx_device->logical_chip_id,
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

        if (push_mode) {
            // for push mode, report average b/w
            total_tx_bw_2 = total_tx_bw / num_tx_workers;
        } else {
            total_tx_bw_2 = ((double)total_tx_words_sent) * PACKET_WORD_SIZE_BYTES / max_tx_elapsed_cycles;
        }

        for (uint32_t d = 0; d < rx_devices.size(); d++) {
            for (uint32_t i = 0; i < num_rx_workers; i++) {
                uint64_t words_received = get_64b_result(rx_results[d][i], TT_FABRIC_WORD_CNT_INDEX);
                uint32_t num_tx = rx_to_tx_map[i].size();
                log_info(
                    LogTest,
                    "[Device: Phys: {}, Logical: {}] RX {}, num producers = {}, words received = {}",
                    rx_devices[d]->physical_chip_id,
                    (uint32_t)rx_devices[d]->logical_chip_id,
                    i,
                    num_tx,
                    words_received);
            }
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
};

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
    constexpr uint32_t default_data_kb_per_tx = 1024 * 1024;

    constexpr uint32_t default_routing_table_start_addr = 0x7EC00;
    constexpr uint32_t default_tx_queue_start_addr = 0x80000;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = 0xa0000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;

    // if this is used for multicast on all workers, carefully set it to a value that
    // doesnt interfere with rx payload checking
    constexpr uint32_t default_tx_signal_address = 0x28000;

    constexpr uint32_t default_test_results_addr = 0x100000;
    constexpr uint32_t default_test_results_size = 0x40000;

    // TODO: use eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE which should be 0x19900, set test results size back
    // to 0x7000
    constexpr uint32_t default_tunneler_queue_size_bytes = 0x8000;   // maximum queue (power of 2)
    constexpr uint32_t default_tunneler_test_results_addr = 0x39000; // 0x8000 * 4 + 0x19000; 0x10000 * 4 + 0x19000 = 0x59000 > 0x40000 (256kB)
    constexpr uint32_t default_tunneler_test_results_size = 0x6000;  // 256kB total L1 in ethernet core - 0x39000

    constexpr uint32_t default_timeout_mcycles = 100000000;
    constexpr uint32_t default_rx_disable_data_check = 0;
    constexpr uint32_t default_rx_disable_header_check = 0;

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

    constexpr uint32_t default_multicast = 0;

    constexpr const char* default_board_type = "glx32";

    constexpr uint32_t default_num_traffic_devices = 0;

    constexpr uint32_t default_num_hops = 0xFFFFFFFF;

    constexpr uint32_t default_num_packets = 0xFFFFFFFF;

    constexpr uint32_t default_num_links = 1;

    constexpr uint32_t default_packet_size_kb = 4;

    constexpr uint32_t default_host_signal_address = 0x60000;

    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  --prng_seed: PRNG seed, default = 0x{:x}", default_prng_seed);
        log_info(LogTest, "  --data_kb_per_tx: Total data in KB per TX endpoint, default = {}", default_data_kb_per_tx);
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
        log_info(LogTest, "  --disable_txrx_timeout: Disable timeout during tx & rx (enabled by default)");
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
            LogTest, "  --device_id_r: DDevice on which the test will be run, default = {}", default_test_device_id_r);

        return 0;
    }

    uint32_t tx_x = test_args::get_command_option_uint32(input_args, "--tx_x", default_tx_x);
    uint32_t tx_y = test_args::get_command_option_uint32(input_args, "--tx_y", default_tx_y);
    uint32_t rx_x = test_args::get_command_option_uint32(input_args, "--rx_x", default_rx_x);
    uint32_t rx_y = test_args::get_command_option_uint32(input_args, "--rx_y", default_rx_y);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    uint32_t data_kb_per_tx =
        test_args::get_command_option_uint32(input_args, "--data_kb_per_tx", default_data_kb_per_tx);
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
    bool disable_txrx_timeout = test_args::has_command_option(input_args, "--disable_txrx_timeout");
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
    uint32_t data_mode = test_args::has_command_option(input_args, "--raw_data");

    // Note here that currently mcast_depth considers the mcast origin as a hop, and not the distance from the origin
    // This has side effects that specifying a depth of 0 or 1 will result in the same behavior
    std::unordered_map<RoutingDirection, uint32_t> mcast_depth;
    mcast_depth[RoutingDirection::E] = test_args::get_command_option_uint32(input_args, "--e_depth", default_multicast);
    mcast_depth[RoutingDirection::W] = test_args::get_command_option_uint32(input_args, "--w_depth", default_multicast);
    mcast_depth[RoutingDirection::N] = test_args::get_command_option_uint32(input_args, "--n_depth", default_multicast);
    mcast_depth[RoutingDirection::S] = test_args::get_command_option_uint32(input_args, "--s_depth", default_multicast);
    bool mcast = false;
    for (const auto& [dir, depth] : mcast_depth) {
        if (depth) {
            // TODO: Remove once generic mcast is supported
            if (mcast) {
                throw std::runtime_error("Only 1 mcast direction is supported right now");
            }
            mcast = true;
        }
    }

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
    bidirectional_traffic = test_args::has_command_option(input_args, "--bidirectional");

    tx_signal_address = default_tx_signal_address;
    host_signal_address = default_host_signal_address;

    std::string board_type = test_args::get_command_option(input_args, "--board_type", std::string(default_board_type));

    uint32_t num_traffic_devices =
        test_args::get_command_option_uint32(input_args, "--num_devices", default_num_traffic_devices);

    uint32_t num_packets = test_args::get_command_option_uint32(input_args, "--num_packets", default_num_packets);

    uint32_t num_links = test_args::get_command_option_uint32(input_args, "--num_links", default_num_links);

    bool fixed_async_wr_notif_addr = test_args::has_command_option(input_args, "--fixed_async_wr_notif_addr");

    benchmark_mode = test_args::has_command_option(input_args, "--benchmark");
    push_mode = test_args::has_command_option(input_args, "--push_router");
    uint32_t packet_size_kb =
        test_args::get_command_option_uint32(input_args, "--packet_size_kb", default_packet_size_kb);
    uint32_t max_packet_size_words = packet_size_kb * 1024 / PACKET_WORD_SIZE_BYTES;

    uint32_t timeout_cycles = timeout_mcycles * 1000;

    // Only supports line mcast from neighbour
    if (mcast && num_hops != default_num_hops && num_hops != 1) {
        throw std::runtime_error("Only line mcast is supported right now");
    }

    if (mcast && bidirectional_traffic) {
        throw std::runtime_error("Bidirectional traffic is not supported for mcast");
    }

    bool pass = true;
    uint32_t num_available_devices, num_allocated_devices = 0;

    std::map<string, string> defines;
    uint16_t routing_mode;
    if (!push_mode) {
        defines["FVC_MODE_PULL"] = "";
        routing_mode = (ROUTING_MODE_MESH | ROUTING_MODE_2D | ROUTING_MODE_PULL);
    } else {
        routing_mode = (ROUTING_MODE_MESH | ROUTING_MODE_2D | ROUTING_MODE_PUSH | ROUTING_MODE_LOW_LATENCY);
    }
    tt::tt_fabric::set_routing_mode(routing_mode);
    defines["ROUTING_MODE"] = std::to_string(static_cast<int>(routing_mode));

    if (benchmark_mode) {
        prng_seed = 100;
        tx_pkt_dest_size_choice = 1;
        tx_skip_pkt_content_gen = true;

        if (num_src_endpoints != num_dest_endpoints) {
            throw std::runtime_error("Currently for benchmark mode, num src should be same as num dest");
        }

        tx_kernel_src = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_tx_ubench.cpp";
    } else {
        tx_kernel_src = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen_tx.cpp";
    }

    // if using fixed packet sizes and num packets is specified, get the test data size
    if ((1 == tx_pkt_dest_size_choice) && (default_num_packets != num_packets)) {
        data_kb_per_tx = num_packets * packet_size_kb;
    }

    if (default_prng_seed == prng_seed) {
        std::random_device rd;
        prng_seed = rd();
    }

    global_rng.seed(prng_seed);
    time_seed = std::chrono::system_clock::now().time_since_epoch().count();

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
            if (mcast) {
                // TODO: We require mcast origin to be the neighbor for now
                // So get the path from test_device_id_l and verify the next chip is test_device_id_r
                auto physical_mcast_chip_ids = test_board.get_physical_mcast_chip_ids(test_device_id_l, mcast_depth);
                if (physical_mcast_chip_ids.empty() || physical_mcast_chip_ids[0] != test_device_id_r) {
                    throw std::runtime_error("No multicast path found");
                }
                test_board.tx_rx_map.push_back({test_device_id_l, std::move(physical_mcast_chip_ids)});
            } else {
                if (test_device_id_l == test_device_id_r) {
                    throw std::runtime_error("Left and right chips should be different");
                }
                test_board.tx_rx_map.push_back({test_device_id_l, {test_device_id_r}});
            }
        } else {
            test_board.generate_tx_rx_map(num_hops, mcast, mcast_depth);
        }

        std::unordered_map<chip_id_t, std::shared_ptr<test_device_t>> test_devices;
        std::vector<test_traffic_t> fabric_traffic;

        // init test devices from the list of chips
        for (auto& chip_id : test_board.physical_chip_ids) {
            test_devices[chip_id] = std::make_shared<test_device_t>(chip_id, &test_board);
        }

        // init traffic
        chip_id_t tx_chip_id, rx_chip_id;
        for (auto& [tx_chip_id, rx_chip_ids] : test_board.tx_rx_map) {
            if (num_allocated_devices >= num_traffic_devices) {
                break;
            }

            std::vector<std::shared_ptr<test_device_t>> rx_devices;
            rx_devices.reserve(rx_chip_ids.size());
            for (auto& rx_chip_id : rx_chip_ids) {
                rx_devices.push_back(test_devices[rx_chip_id]);
            }

            test_traffic_t traffic(
                test_devices[tx_chip_id],
                rx_devices,
                num_src_endpoints,
                num_dest_endpoints,
                target_address,
                num_hops,
                num_links);
            fabric_traffic.push_back(std::move(traffic));

            if (bidirectional_traffic) {
                std::vector<std::shared_ptr<test_device_t>> rx_devices = {test_devices[tx_chip_id]};
                test_traffic_t traffic_r(
                    test_devices[rx_chip_ids[0]],
                    rx_devices,
                    num_src_endpoints,
                    num_dest_endpoints,
                    target_address,
                    num_hops,
                    num_links);
                fabric_traffic.push_back(std::move(traffic_r));
            }

            if (bidirectional_traffic && benchmark_mode) {
                auto& traffic_fwd = fabric_traffic.at(fabric_traffic.size() - 2);
                auto& traffic_bwd = fabric_traffic.at(fabric_traffic.size() - 1);
                traffic_fwd.set_remote_controller(traffic_bwd);
                traffic_bwd.set_remote_controller(traffic_fwd);
            }

            num_allocated_devices += 1 + rx_chip_ids.size();
        }

        uint32_t worker_unreserved_base_addr =
            test_devices.begin()->second->device_handle->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

        // manual init fabric
        // create router kernels
        std::vector<uint32_t> router_compile_args = {
            (tunneler_queue_size_bytes >> 4),  // 0: rx_queue_size_words
            tunneler_test_results_addr,        // 1: test_results_addr
            tunneler_test_results_size,        // 2: test_results_size
            0,                                 // timeout_mcycles * 1000 * 1000 * 4, // 3: timeout_cycles
        };
        for (auto& [chip_id, test_device] : test_devices) {
            test_device->create_router_kernels(router_compile_args, defines);
        }
        if (!disable_txrx_timeout) {
            defines["CHECK_TIMEOUT"] = "";
        }

        uint32_t client_interface_addr = worker_unreserved_base_addr;
        uint32_t client_pull_req_buf_addr =
            client_interface_addr + PULL_CLIENT_INTERFACE_SIZE + sizeof(fabric_router_l1_config_t) * 4;

        std::vector<uint32_t> tx_compile_args = {
            data_mode,                         // 0: Data mode. 0 - Packetized Data. 1 Raw Data.
            num_dest_endpoints,                // 1: num_dest_endpoints
            dest_endpoint_start_id,            // 2:
            tx_queue_start_addr,               // 3: queue_start_addr_words
            (tx_queue_size_bytes >> 4),        // 4: queue_size_words
            routing_table_start_addr,          // 5: routeing table
            test_results_addr,                 // 6: test_results_addr
            test_results_size,                 // 7: test_results_size
            prng_seed,                         // 8: prng_seed
            data_kb_per_tx,                    // 9: total_data_kb
            max_packet_size_words,             // 10: max_packet_size_words
            timeout_cycles,                    // 11: timeout_cycles
            tx_skip_pkt_content_gen,           // 12: skip_pkt_content_gen
            tx_pkt_dest_size_choice,           // 13: pkt_dest_size_choice
            tx_data_sent_per_iter_low,         // 14: data_sent_per_iter_low
            tx_data_sent_per_iter_high,        // 15: data_sent_per_iter_high
            fabric_command,                    // 16: fabric_command
            target_address,                    // 17: target_address
            atomic_increment,                  // 18: atomic_increment
            tx_signal_address,                 // 19: tx_signal_address
            client_interface_addr,             // 20:
            client_pull_req_buf_addr,          // 21:
            fixed_async_wr_notif_addr,         // 22: use fixed addr for async wr atomic inc
            mcast,                             // 23: mcast
            mcast_depth[RoutingDirection::E],  // 24: mcast_e
            mcast_depth[RoutingDirection::W],  // 25: mcast_w
            mcast_depth[RoutingDirection::N],  // 26: mcast_n
            mcast_depth[RoutingDirection::S],  // 27: mcast_s
            push_mode                          // 28: Router mode. 0 - Pull, 1 - Push
        };

        std::vector<uint32_t> rx_compile_args = {
            prng_seed,                  // 0: prng seed
            data_kb_per_tx,             // 1: total data kb
            max_packet_size_words,      // 2: max packet size (in words)
            fabric_command,             // 3: fabric command
            target_address,             // 4: target address
            atomic_increment,           // 5: atomic increment
            test_results_addr,          // 6: test results addr
            test_results_size,          // 7: test results size in bytes
            tx_pkt_dest_size_choice,    // 8: pkt dest and size choice
            tx_skip_pkt_content_gen,    // 9: skip packet validation
            fixed_async_wr_notif_addr,  // 10: use fixed addr for async wr atomic inc
            timeout_cycles,             // 11: idle timeout cycles
        };

        // TODO: launch traffic kernels
        for (auto& traffic : fabric_traffic) {
            traffic.create_kernels(tx_compile_args, rx_compile_args, defines, fabric_command, test_results_addr);
        }

        if (!disable_txrx_timeout) {
            defines.erase("CHECK_TIMEOUT");
        }

        auto start = std::chrono::system_clock::now();

        // launch programs
        for (auto& [chip_id, test_device] : test_devices) {
            tt_metal::detail::LaunchProgram(test_device->device_handle, test_device->program_handle, false);
        }

        log_info(LogTest, "Programs launched, waiting for router sync");

        // wait for all routers to handshake with master router
        for (auto& [chip_id, test_device] : test_devices) {
            test_device->wait_for_router_sync();
        }

        log_info(LogTest, "Routers sync done, notifying tx controllers");

        // notify tx controller to signal the tx workers
        for (auto& traffic : fabric_traffic) {
            traffic.notify_tx_controller();
        }

        log_info(LogTest, "Notified TX controllers, waiting for TX workers to finish");

        // wait for rx kernels to finish
        for (auto& traffic : fabric_traffic) {
            traffic.wait_for_tx_workers_to_finish();
        }

        log_info(LogTest, "TX workers done, waiting for RX workers to finish");

        // wait for rx kernels to finish
        for (auto& traffic : fabric_traffic) {
            traffic.wait_for_rx_workers_to_finish();
        }

        log_info(LogTest, "RX workers done, terminating routers");

        // terminate fabric routers if control plane is not managed by DevicePool
        for (auto& [chip_id, test_device] : test_devices) {
            test_device->terminate_router_kernels();
        }

        log_info(LogTest, "Terminated routers, waiting for program done");

        // wait for programs to exit
        for (auto& [chip_id, test_device] : test_devices) {
            tt_metal::detail::WaitProgramDone(test_device->device_handle, test_device->program_handle);
        }
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        log_info(LogTest, "Programs done, collecting results");

        // collect traffic results
        for (auto& traffic : fabric_traffic) {
            pass &= traffic.collect_results(test_results_addr);
            if (!pass) {
                log_fatal(LogTest, "Result collection failed, skipping any further collection/validation");
                break;
            }
        }

        // close devices
        test_board.close_devices();

        // tally-up the packets and words from tx/rx kernels
        if (pass) {
            for (auto& traffic : fabric_traffic) {
                pass &= traffic.validate_results();
                if (!pass) {
                    log_fatal(LogTest, "Result validation failed, skipping any further validation");
                    break;
                }
            }
        }

        // print results
        if (pass) {
            for (auto& traffic : fabric_traffic) {
                traffic.print_result_summary();
            }
        }

    } catch (const std::exception& e) {
        pass = false;
        log_fatal(tt::LogTest, "{}", e.what());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
