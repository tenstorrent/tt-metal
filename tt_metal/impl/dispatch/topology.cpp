// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology.hpp"

#include <device_pool.hpp>
#include <host_api.hpp>
#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <map>
#include <typeinfo>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "assert.hpp"
#include "command_queue_common.hpp"
#include "control_plane.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch_core_common.hpp"
#include "fabric_host_interface.h"
#include "kernel_config/demux.hpp"
#include "kernel_config/eth_router.hpp"
#include "kernel_config/eth_tunneler.hpp"
#include "kernel_config/fd_kernel.hpp"
#include "kernel_types.hpp"
#include "metal_soc_descriptor.h"
#include "persistent_kernel_cache.hpp"
#include "program/program_impl.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include <tt-metalium/fabric.hpp>
#include "system_memory_manager.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_metal {

// For readablity, unset = x = -1
constexpr int x = -1;

void increment_node_ids(DispatchKernelNode& node, uint32_t inc) {
    node.id += inc;
    for (int& id : node.upstream_ids) {
        if (id != x) {
            id += inc;
        }
    }
    for (int& id : node.downstream_ids) {
        if (id != x) {
            id += inc;
        }
    }
}

//
// Prefetcher NOC selections
//
// Non Dispatch NOC: acquire pages on local semaphore
//
// Upstream: sync sem with tunnel and/or prefetch_h variant
//
// Downstream: send data to tunnel and/or prefetch_d variant and/or dispatch_d
//
constexpr noc_selection_t k_prefetcher_noc = {
    .non_dispatch_noc = tt::tt_metal::NOC::NOC_0,
    .upstream_noc = tt::tt_metal::NOC::NOC_0,
    .downstream_noc = tt::tt_metal::NOC::NOC_0,
};

//
// Dispatcher NOC selections. NOTE: Upstream and downstream NOCs cannot be the same.
//
// Non Dispatch NOC: acquire pages on local semaphore and send go/done
//
// Upstream: sync sem with tunnel and/or dispatch_d variant and/or prefetch_d
//
// Downstream: relay data from dispatch_d to dispatch_h (return to host) and/or dispatch_s
//
constexpr noc_selection_t k_dispatcher_noc = {
    .non_dispatch_noc = tt::tt_metal::NOC::NOC_0,
    .upstream_noc = tt::tt_metal::NOC::NOC_1,
    .downstream_noc = k_dispatch_downstream_noc,
};

//
// Dispatch S NOC selections.
//
// Non Dispatch NOC: acquire pages on local semaphore
//
// Upstream: sync sem with prefetcher_d and dispatcher_d
//
// Downstream: relay data from dispatch_d to dispatch_h (return to host) and/or dispatch_s
//
constexpr noc_selection_t k_dispatcher_s_noc = {
    .non_dispatch_noc = tt::tt_metal::NOC::NOC_1,
    .upstream_noc = tt::tt_metal::NOC::NOC_1,
    .downstream_noc = tt::tt_metal::NOC::NOC_1,
};

// Must be on different NOCs because Dispatch+S may be running on the same
// core. They are using stateful APIs. Running on the same NOC will mess up
// requests sent/to free count.
static_assert(k_dispatcher_noc.non_dispatch_noc != k_dispatcher_s_noc.non_dispatch_noc);

//
// Packet Queue NOC selections
//
// Non Dispatch NOC: Sync semaphore and relaying data between upstream and downstream components
//
// Upstream: UNUSED
//
// Downstream: UNUSED
//
constexpr noc_selection_t k_packet_queue_noc = {
    .non_dispatch_noc = tt::tt_metal::NOC::NOC_0,
    .upstream_noc = tt::tt_metal::NOC::NOC_0,
    .downstream_noc = tt::tt_metal::NOC::NOC_0,
};

//
// Fabric MUX NOC selections
//
// Must be NoC0
//
constexpr noc_selection_t k_fabric_mux_noc = {
    .non_dispatch_noc = tt::tt_metal::NOC::NOC_0,
    .upstream_noc = tt::tt_metal::NOC::NOC_0,
    .downstream_noc = tt::tt_metal::NOC::NOC_0,
};

// clang-format off
static const std::vector<DispatchKernelNode> single_chip_arch_1cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, k_prefetcher_noc},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, k_dispatcher_noc},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, k_dispatcher_s_noc},
};

static const std::vector<DispatchKernelNode> single_chip_arch_2cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, x, x, x}, k_prefetcher_noc},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, x, x, x}, k_prefetcher_noc},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, k_dispatcher_noc},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {x, x, x, x}, k_dispatcher_noc},
};

static const std::vector<DispatchKernelNode> single_chip_arch_2cq_dispatch_s = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 4, x, x}, k_prefetcher_noc},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {4, x, x, x}, k_dispatcher_noc},
    {2, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, 5, x, x}, k_prefetcher_noc},
    {3, 0, 0, 1, DISPATCH_HD, {2, x, x, x}, {5, x, x, x}, k_dispatcher_noc},
    {4, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, k_dispatcher_s_noc},
    {5, 0, 0, 1, DISPATCH_S, {2, x, x, x}, {3, x, x, x}, k_dispatcher_s_noc},
};

static const std::vector<DispatchKernelNode> two_chip_arch_1cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, k_prefetcher_noc},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, k_dispatcher_noc},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, k_dispatcher_s_noc},

    {3, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {5, x, x, x}, k_prefetcher_noc},
    {4, 0, 1, 0, DISPATCH_H, {6, x, x, x}, {3, x, x, x}, k_dispatcher_noc},

    {5, 0, 1, 0, PACKET_ROUTER_MUX, {3, x, x, x}, {7, x, x, x}, k_packet_queue_noc},
    {6, 0, 1, 0, DEMUX, {7, x, x, x}, {4, x, x, x}, k_packet_queue_noc},
    {7, 0, 1, 0, US_TUNNELER_REMOTE, {11, 5, x, x}, {11, 6, x, x}, k_packet_queue_noc},

    {8, 1, x, 0, PREFETCH_D, {13, x, x, x}, {9, 10, x, x}, k_prefetcher_noc},
    {9, 1, x, 0, DISPATCH_D, {8, x, x, x}, {10, 12, x, x}, k_dispatcher_noc},
    {10, 1, x, 0, DISPATCH_S, {8, x, x, x}, {9, x, x, x}, k_dispatcher_s_noc},

    {11, 1, x, 0, US_TUNNELER_LOCAL, {7, 12, x, x}, {7, 13, x, x}, k_packet_queue_noc},
    {12, 1, x, 0, MUX_D, {9, x, x, x}, {11, x, x, x}, k_packet_queue_noc},
    {13, 1, x, 0, PACKET_ROUTER_DEMUX, {11, x, x, x}, {8, x, x, x}, k_packet_queue_noc},
};

static const std::vector<DispatchKernelNode> two_chip_arch_2cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, x, x, x}, k_prefetcher_noc},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, x, x, x}, k_prefetcher_noc},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, k_dispatcher_noc},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {x, x, x, x}, k_dispatcher_noc},

    {4, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {8, x, x, x}, k_prefetcher_noc},
    {5, 0, 1, 1, PREFETCH_H, {x, x, x, x}, {8, x, x, x}, k_prefetcher_noc},
    {6, 0, 1, 0, DISPATCH_H, {9, x, x, x}, {4, x, x, x}, k_dispatcher_noc},
    {7, 0, 1, 1, DISPATCH_H, {9, x, x, x}, {5, x, x, x}, k_dispatcher_noc},

    {8, 0, 1, 0, PACKET_ROUTER_MUX, {4, 5, x, x}, {10, x, x, x}, k_packet_queue_noc},
    {9, 0, 1, 0, DEMUX, {10, x, x, x}, {6, 7, x, x}, k_packet_queue_noc},
    {10, 0, 1, 0, US_TUNNELER_REMOTE, {15, 8, x, x}, {15, 9, x, x}, k_packet_queue_noc},

    {11, 1, x, 0, PREFETCH_D, {17, x, x, x}, {13, x, x, x}, k_prefetcher_noc},
    {12, 1, x, 1, PREFETCH_D, {17, x, x, x}, {14, x, x, x}, k_prefetcher_noc},
    {13, 1, x, 0, DISPATCH_D, {11, x, x, x}, {16, x, x, x}, k_dispatcher_noc},
    {14, 1, x, 1, DISPATCH_D, {12, x, x, x}, {16, x, x, x}, k_dispatcher_noc},

    {15, 1, x, 0, US_TUNNELER_LOCAL, {10, 16, x, x}, {10, 17, x, x}, k_packet_queue_noc},
    {16, 1, x, 0, MUX_D, {13, 14, x, x}, {15, x, x, x}, k_packet_queue_noc},
    {17, 1, x, 0, PACKET_ROUTER_DEMUX, {15, x, x, x}, {11, 12, x, x}, k_packet_queue_noc},

};

static const std::vector<DispatchKernelNode> two_chip_arch_1cq_fabric = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, k_prefetcher_noc},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, k_dispatcher_noc},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, k_dispatcher_s_noc},

    {3, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {6, 5, x, x}, k_prefetcher_noc},
    {4, 0, 1, 0, DISPATCH_H, {7, x, x, x}, {3, 5, x, x}, k_dispatcher_noc},

    // H2D via MUX
    {5, 0, 1, 0, FABRIC_MUX, /*Full size*/ {3}, /*Header Only*/ {4}, k_fabric_mux_noc, 0},

    {6, 1, x, 0, PREFETCH_D, {3, x, x, x}, {7, 8, 9, x}, k_prefetcher_noc},
    {7, 1, x, 0, DISPATCH_D, {6, x, x, x}, {8, 4, 9, x}, k_dispatcher_noc},
    {8, 1, x, 0, DISPATCH_S, {6, x, x, x}, {7, x, x, x}, k_dispatcher_s_noc},

    // D2H via MUX
    {9, 1, 0, 0, RETURN_FABRIC_MUX, /*Full size*/ {7}, /*Header Only*/ {6}, k_fabric_mux_noc, 0},
};

static const std::vector<DispatchKernelNode> two_chip_arch_2cq_fabric = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, x, x, x}, k_prefetcher_noc},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, x, x, x}, k_prefetcher_noc},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, k_dispatcher_noc},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {x, x, x, x}, k_dispatcher_noc},

    {4, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {9, 8, x, x}, k_prefetcher_noc},
    {5, 0, 1, 1, PREFETCH_H, {x, x, x, x}, {10, 8, x, x}, k_prefetcher_noc},
    {6, 0, 1, 0, DISPATCH_H, {11, x, x, x}, {4, 8, x, x}, k_dispatcher_noc},
    {7, 0, 1, 1, DISPATCH_H, {12, x, x, x}, {5, 8, x, x}, k_dispatcher_noc},

    // H2D via MUX
    {8, 0, 1, 0, FABRIC_MUX, /*Full size*/ {4, 5}, /*Header Only*/ {6, 7}, k_fabric_mux_noc, 0},

    {9, 1, x, 0, PREFETCH_D, {4, x, x, x}, {11, 13, x, x}, k_prefetcher_noc},
    {10, 1, x, 1, PREFETCH_D, {5, x, x, x}, {12, 13, x, x}, k_prefetcher_noc},
    {11, 1, x, 0, DISPATCH_D, {9, x, x, x}, {6, 13, x, x}, k_dispatcher_noc},
    {12, 1, x, 1, DISPATCH_D, {10, x, x, x}, {7, 13, x, x}, k_dispatcher_noc},

    // D2H via MUX
    {13, 1, 0, 0, RETURN_FABRIC_MUX, /*Full size*/ {11, 12}, /*Header Only*/ {9, 10}, k_fabric_mux_noc, 0},
};

static const std::vector<DispatchKernelNode> galaxy_nine_chip_arch_1cq_fabric = {
    // Servicing remote chips 1-4
    // { id, device_id, servicing_device_id, cq, fd kernel, upstream ids, downstream ids, noc selection}
    {0, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {18, 8, x, x}, k_prefetcher_noc},
    {1, 0, 1, 0, DISPATCH_H, {19, x, x, x}, {0, 8, x, x}, k_dispatcher_noc},

    {2, 0, 2, 0, PREFETCH_H, {x, x, x, x}, {22, 8, x, x}, k_prefetcher_noc},
    {3, 0, 2, 0, DISPATCH_H, {23, x, x, x}, {2, 8, x, x}, k_dispatcher_noc},

    {4, 0, 3, 0, PREFETCH_H, {x, x, x, x}, {26, 8, x, x}, k_prefetcher_noc},
    {5, 0, 3, 0, DISPATCH_H, {27, x, x, x}, {4, 8, x, x}, k_dispatcher_noc},

    {6, 0, 4, 0, PREFETCH_H, {x, x, x, x}, {30, 8, x, x}, k_prefetcher_noc},
    {7, 0, 4, 0, DISPATCH_H, {31, x, x, x}, {6, 8, x, x}, k_dispatcher_noc},

    {8, 0, x, 0, FABRIC_MUX, /*full size*/ {0, 2, 4, 6}, /*header only*/ {1, 3, 5, 7}, k_fabric_mux_noc, 0},

    // Servicing remote chips 5-8
    {9, 0, 5, 0, PREFETCH_H, {x, x, x, x}, {34, 17, x, x}, k_prefetcher_noc},
    {10, 0, 5, 0, DISPATCH_H, {35, x, x, x}, {9, 17, x, x}, k_dispatcher_noc},

    {11, 0, 6, 0, PREFETCH_H, {x, x, x, x}, {38, 17, x, x}, k_prefetcher_noc},
    {12, 0, 6, 0, DISPATCH_H, {39, x, x, x}, {11, 17, x, x}, k_dispatcher_noc},

    {13, 0, 7, 0, PREFETCH_H, {x, x, x, x}, {42, 17, x, x}, k_prefetcher_noc},
    {14, 0, 7, 0, DISPATCH_H, {43, x, x, x}, {13, 17, x, x}, k_dispatcher_noc},

    {15, 0, 8, 0, PREFETCH_H, {x, x, x, x}, {46, 17, x, x}, k_prefetcher_noc},
    {16, 0, 8, 0, DISPATCH_H, {47, x, x, x}, {15, 17, x, x}, k_dispatcher_noc},

    {17, 0, x, 0, FABRIC_MUX, /*full size*/ {9, 11, 13, 15}, /*header only*/ {10, 12, 14, 16}, k_fabric_mux_noc, 1},

    // Remote chip 1
    {18, 1, x, 0, PREFETCH_D, {0, x, x, x}, {19, 20, 21, x}, k_prefetcher_noc},
    {19, 1, x, 0, DISPATCH_D, {18, x, x, x}, {1, 20, 21, x}, k_dispatcher_noc},
    {20, 1, x, 0, DISPATCH_S, {18, x, x, x}, {19, x, x, x}, k_dispatcher_s_noc},
    {21, 1, x, 0, RETURN_FABRIC_MUX, /*full size*/ {19}, /*header only*/ {18}, k_fabric_mux_noc, 0},

    // Remote chip 2
    {22, 2, x, 0, PREFETCH_D, {2, x, x, x}, {23, 24, 25, x}, k_prefetcher_noc},
    {23, 2, x, 0, DISPATCH_D, {22, x, x, x}, {3, 24, 25, x}, k_dispatcher_noc},
    {24, 2, x, 0, DISPATCH_S, {22, x, x, x}, {23, x, x, x}, k_dispatcher_s_noc},
    {25, 2, x, 0, RETURN_FABRIC_MUX, /*full size*/ {23}, /*header only*/ {22}, k_fabric_mux_noc, 0},

    // Remote chip 3
    {26, 3, x, 0, PREFETCH_D, {4, x, x, x}, {27, 28, 29, x}, k_prefetcher_noc},
    {27, 3, x, 0, DISPATCH_D, {26, x, x, x}, {5, 28, 29, x}, k_dispatcher_noc},
    {28, 3, x, 0, DISPATCH_S, {26, x, x, x}, {27, x, x, x}, k_dispatcher_s_noc},
    {29, 3, x, 0, RETURN_FABRIC_MUX, /*full size*/ {27}, /*header only*/ {26}, k_fabric_mux_noc, 0},

    // Remote chip 4
    {30, 4, x, 0, PREFETCH_D, {6, x, x, x}, {31, 32, 33, x}, k_prefetcher_noc},
    {31, 4, x, 0, DISPATCH_D, {30, x, x, x}, {7, 32, 33, x}, k_dispatcher_noc},
    {32, 4, x, 0, DISPATCH_S, {30, x, x, x}, {31, x, x, x}, k_dispatcher_s_noc},
    {33, 4, x, 0, RETURN_FABRIC_MUX, /*full size*/ {31}, /*header only*/ {30}, k_fabric_mux_noc, 0},

    // Remote chip 5
    {34, 5, x, 0, PREFETCH_D, {9, x, x, x}, {35, 36, 37, x}, k_prefetcher_noc},
    {35, 5, x, 0, DISPATCH_D, {34, x, x, x}, {10, 36, 37, x}, k_dispatcher_noc},
    {36, 5, x, 0, DISPATCH_S, {34, x, x, x}, {35, x, x, x}, k_dispatcher_s_noc},
    {37, 5, x, 0, RETURN_FABRIC_MUX, /*full size*/ {35}, /*header only*/ {34}, k_fabric_mux_noc, 1},

    // Remote chip 6
    {38, 6, x, 0, PREFETCH_D, {11, x, x, x}, {39, 40, 41, x}, k_prefetcher_noc},
    {39, 6, x, 0, DISPATCH_D, {38, x, x, x}, {12, 40, 41, x}, k_dispatcher_noc},
    {40, 6, x, 0, DISPATCH_S, {38, x, x, x}, {39, x, x, x}, k_dispatcher_s_noc},
    {41, 6, x, 0, RETURN_FABRIC_MUX, /*full size*/ {39}, /*header only*/ {38}, k_fabric_mux_noc, 1},

    // Remote chip 7
    {42, 7, x, 0, PREFETCH_D, {13, x, x, x}, {43, 44, 45, x}, k_prefetcher_noc},
    {43, 7, x, 0, DISPATCH_D, {42, x, x, x}, {14, 44, 45, x}, k_dispatcher_noc},
    {44, 7, x, 0, DISPATCH_S, {42, x, x, x}, {43, x, x, x}, k_dispatcher_s_noc},
    {45, 7, x, 0, RETURN_FABRIC_MUX, /*full size*/ {43}, /*header only*/ {42}, k_fabric_mux_noc, 1},

    // Remote chip 8
    {46, 8, x, 0, PREFETCH_D, {15, x, x, x}, {47, 48, 49, x}, k_prefetcher_noc},
    {47, 8, x, 0, DISPATCH_D, {46, x, x, x}, {16, 48, 49, x}, k_dispatcher_noc},
    {48, 8, x, 0, DISPATCH_S, {46, x, x, x}, {47, x, x, x}, k_dispatcher_s_noc},
    {49, 8, x, 0, RETURN_FABRIC_MUX, /*full size*/ {47}, /*header only*/ {46}, k_fabric_mux_noc, 1},
};

static const std::vector<DispatchKernelNode> galaxy_nine_chip_arch_2cq_fabric = {
    // Servicing remote chips 1-4
    // { id, device_id, servicing_device_id, cq, fd kernel, upstream ids, downstream ids, noc selection}
    {0, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {16, 34, x, x}, k_prefetcher_noc},
    {1, 0, 1, 1, PREFETCH_H, {x, x, x, x}, {16, 35, x, x}, k_prefetcher_noc},

    {2, 0, 1, 0, DISPATCH_H, {36, x, x, x}, {16, 0, x, x}, k_dispatcher_noc},
    {3, 0, 1, 1, DISPATCH_H, {37, x, x, x}, {16, 1, x, x}, k_dispatcher_noc},

    {4, 0, 2, 0, PREFETCH_H, {x, x, x, x}, {16, 41, x, x}, k_prefetcher_noc},
    {5, 0, 2, 1, PREFETCH_H, {x, x, x, x}, {16, 42, x, x}, k_prefetcher_noc},

    {6, 0, 2, 0, DISPATCH_H, {43, x, x, x}, {16, 4, x, x}, k_dispatcher_noc},
    {7, 0, 2, 1, DISPATCH_H, {44, x, x, x}, {16, 5, x, x}, k_dispatcher_noc},

    {8, 0, 3, 0, PREFETCH_H, {x, x, x, x}, {16, 48, x, x}, k_prefetcher_noc},
    {9, 0, 3, 1, PREFETCH_H, {x, x, x, x}, {16, 49, x, x}, k_prefetcher_noc},

    {10, 0, 3, 0, DISPATCH_H, {50, x, x, x}, {16, 8, x, x}, k_dispatcher_noc},
    {11, 0, 3, 1, DISPATCH_H, {51, x, x, x}, {16, 9, x, x}, k_dispatcher_noc},

    {12, 0, 4, 0, PREFETCH_H, {x, x, x, x}, {16, 55, x, x}, k_prefetcher_noc},
    {13, 0, 4, 1, PREFETCH_H, {x, x, x, x}, {16, 56, x, x}, k_prefetcher_noc},

    {14, 0, 4, 0, DISPATCH_H, {57, x, x, x}, {16, 12, x, x}, k_dispatcher_noc},
    {15, 0, 4, 1, DISPATCH_H, {58, x, x, x}, {16, 13, x, x}, k_dispatcher_noc},

    {16, 0, x, 0, FABRIC_MUX, /*full size*/ {0, 1, 4, 5, 8, 9, 12, 13}, /*header only*/ {2, 3, 6, 7, 10, 11, 14, 15}, k_fabric_mux_noc, 0},

    // Servicing remote chips 5-8
    {17, 0, 5, 0, PREFETCH_H, {x, x, x, x}, {33, 62, x, x}, k_prefetcher_noc},
    {18, 0, 5, 1, PREFETCH_H, {x, x, x, x}, {33, 63, x, x}, k_prefetcher_noc},

    {19, 0, 5, 0, DISPATCH_H, {64, x, x, x}, {33, 17, x, x}, k_dispatcher_noc},
    {20, 0, 5, 1, DISPATCH_H, {65, x, x, x}, {33, 18, x, x}, k_dispatcher_noc},

    {21, 0, 6, 0, PREFETCH_H, {x, x, x, x}, {33, 69, x, x}, k_prefetcher_noc},
    {22, 0, 6, 1, PREFETCH_H, {x, x, x, x}, {33, 70, x, x}, k_prefetcher_noc},

    {23, 0, 6, 0, DISPATCH_H, {71, x, x, x}, {33, 21, x, x}, k_dispatcher_noc},
    {24, 0, 6, 1, DISPATCH_H, {72, x, x, x}, {33, 22, x, x}, k_dispatcher_noc},

    {25, 0, 7, 0, PREFETCH_H, {x, x, x, x}, {33, 76, x, x}, k_prefetcher_noc},
    {26, 0, 7, 1, PREFETCH_H, {x, x, x, x}, {33, 77, x, x}, k_prefetcher_noc},

    {27, 0, 7, 0, DISPATCH_H, {78, x, x, x}, {33, 25, x, x}, k_dispatcher_noc},
    {28, 0, 7, 1, DISPATCH_H, {79, x, x, x}, {33, 26, x, x}, k_dispatcher_noc},

    {29, 0, 8, 0, PREFETCH_H, {x, x, x, x}, {33, 83, x, x}, k_prefetcher_noc},
    {30, 0, 8, 1, PREFETCH_H, {x, x, x, x}, {33, 84, x, x}, k_prefetcher_noc},

    {31, 0, 8, 0, DISPATCH_H, {85, x, x, x}, {33, 29, x, x}, k_dispatcher_noc},
    {32, 0, 8, 1, DISPATCH_H, {86, x, x, x}, {33, 30, x, x}, k_dispatcher_noc},

    {33, 0, x, 0, FABRIC_MUX, /*full size*/ {17, 18, 21, 22, 25, 26, 29, 30}, /*header only*/ {19, 20, 23, 24, 27, 28, 31, 32}, k_fabric_mux_noc, 1},

    // Remote chip 1
    {34, 1, x, 0, PREFETCH_D, {0, x, x, x}, {40, 36, 38, x}, k_prefetcher_noc},
    {35, 1, x, 1, PREFETCH_D, {1, x, x, x}, {40, 37, 39, x}, k_prefetcher_noc},
    {36, 1, x, 0, DISPATCH_D, {34, x, x, x}, {40, 38, 2, x}, k_dispatcher_noc},
    {37, 1, x, 1, DISPATCH_D, {35, x, x, x}, {40, 39, 3, x}, k_dispatcher_noc},
    {38, 1, x, 0, DISPATCH_S, {34, x, x, x}, {36, x, x, x}, k_dispatcher_s_noc},
    {39, 1, x, 1, DISPATCH_S, {35, x, x, x}, {37, x, x, x}, k_dispatcher_s_noc},
    {40, 1, x, 0, RETURN_FABRIC_MUX, /*full size*/ {36, 37}, /*header only*/ {34, 35}, k_fabric_mux_noc, 0},

    // Remote chip 2
    {41, 2, x, 0, PREFETCH_D, {4, x, x, x}, {47, 43, 45, x}, k_prefetcher_noc},
    {42, 2, x, 1, PREFETCH_D, {5, x, x, x}, {47, 44, 46, x}, k_prefetcher_noc},
    {43, 2, x, 0, DISPATCH_D, {41, x, x, x}, {47, 45, 6, x}, k_dispatcher_noc},
    {44, 2, x, 1, DISPATCH_D, {42, x, x, x}, {47, 46, 7, x}, k_dispatcher_noc},
    {45, 2, x, 0, DISPATCH_S, {41, x, x, x}, {43, x, x, x}, k_dispatcher_s_noc},
    {46, 2, x, 1, DISPATCH_S, {42, x, x, x}, {44, x, x, x}, k_dispatcher_s_noc},
    {47, 2, x, 0, RETURN_FABRIC_MUX, /*full size*/ {43, 44}, /*header only*/ {41, 42}, k_fabric_mux_noc, 0},

    // Remote chip 3
    {48, 3, x, 0, PREFETCH_D, {8, x, x, x}, {54, 50, 52, x}, k_prefetcher_noc},
    {49, 3, x, 1, PREFETCH_D, {9, x, x, x}, {54, 51, 53, x}, k_prefetcher_noc},
    {50, 3, x, 0, DISPATCH_D, {48, x, x, x}, {54, 52, 10, x}, k_dispatcher_noc},
    {51, 3, x, 1, DISPATCH_D, {49, x, x, x}, {54, 53, 11, x}, k_dispatcher_noc},
    {52, 3, x, 0, DISPATCH_S, {48, x, x, x}, {50, x, x, x}, k_dispatcher_s_noc},
    {53, 3, x, 1, DISPATCH_S, {49, x, x, x}, {51, x, x, x}, k_dispatcher_s_noc},
    {54, 3, x, 0, RETURN_FABRIC_MUX, /*full size*/ {50, 51}, /*header only*/ {48, 49}, k_fabric_mux_noc, 0},

    // Remote chip 4
    {55, 4, x, 0, PREFETCH_D, {12, x, x, x}, {61, 57, 59, x}, k_prefetcher_noc},
    {56, 4, x, 1, PREFETCH_D, {13, x, x, x}, {61, 58, 60, x}, k_prefetcher_noc},
    {57, 4, x, 0, DISPATCH_D, {55, x, x, x}, {61, 59, 14, x}, k_dispatcher_noc},
    {58, 4, x, 1, DISPATCH_D, {56, x, x, x}, {61, 60, 15, x}, k_dispatcher_noc},
    {59, 4, x, 0, DISPATCH_S, {55, x, x, x}, {57, x, x, x}, k_dispatcher_s_noc},
    {60, 4, x, 1, DISPATCH_S, {56, x, x, x}, {58, x, x, x}, k_dispatcher_s_noc},
    {61, 4, x, 0, RETURN_FABRIC_MUX, /*full size*/ {57, 58}, /*header only*/ {55, 56}, k_fabric_mux_noc, 0},

    // Remote chip 5
    {62, 5, x, 0, PREFETCH_D, {17, x, x, x}, {68, 64, 66, x}, k_prefetcher_noc},
    {63, 5, x, 1, PREFETCH_D, {18, x, x, x}, {68, 65, 67, x}, k_prefetcher_noc},
    {64, 5, x, 0, DISPATCH_D, {62, x, x, x}, {68, 66, 19, x}, k_dispatcher_noc},
    {65, 5, x, 1, DISPATCH_D, {63, x, x, x}, {68, 67, 20, x}, k_dispatcher_noc},
    {66, 5, x, 0, DISPATCH_S, {62, x, x, x}, {64, x, x, x}, k_dispatcher_s_noc},
    {67, 5, x, 1, DISPATCH_S, {63, x, x, x}, {65, x, x, x}, k_dispatcher_s_noc},
    {68, 5, x, 0, RETURN_FABRIC_MUX, /*full size*/ {64, 65}, /*header only*/ {62, 63}, k_fabric_mux_noc, 1},

    // Remote chip 6
    {69, 6, x, 0, PREFETCH_D, {21, x, x, x}, {75, 71, 73, x}, k_prefetcher_noc},
    {70, 6, x, 1, PREFETCH_D, {22, x, x, x}, {75, 72, 74, x}, k_prefetcher_noc},
    {71, 6, x, 0, DISPATCH_D, {69, x, x, x}, {75, 73, 23, x}, k_dispatcher_noc},
    {72, 6, x, 1, DISPATCH_D, {70, x, x, x}, {75, 74, 24, x}, k_dispatcher_noc},
    {73, 6, x, 0, DISPATCH_S, {69, x, x, x}, {71, x, x, x}, k_dispatcher_s_noc},
    {74, 6, x, 1, DISPATCH_S, {70, x, x, x}, {72, x, x, x}, k_dispatcher_s_noc},
    {75, 6, x, 0, RETURN_FABRIC_MUX, /*full size*/ {71, 72}, /*header only*/ {69, 70}, k_fabric_mux_noc, 1},

    // Remote chip 7
    {76, 7, x, 0, PREFETCH_D, {25, x, x, x}, {82, 78, 80, x}, k_prefetcher_noc},
    {77, 7, x, 1, PREFETCH_D, {26, x, x, x}, {82, 79, 81, x}, k_prefetcher_noc},
    {78, 7, x, 0, DISPATCH_D, {76, x, x, x}, {82, 80, 27, x}, k_dispatcher_noc},
    {79, 7, x, 1, DISPATCH_D, {77, x, x, x}, {82, 81, 28, x}, k_dispatcher_noc},
    {80, 7, x, 0, DISPATCH_S, {76, x, x, x}, {78, x, x, x}, k_dispatcher_s_noc},
    {81, 7, x, 1, DISPATCH_S, {77, x, x, x}, {79, x, x, x}, k_dispatcher_s_noc},
    {82, 7, x, 0, RETURN_FABRIC_MUX, /*full size*/ {78, 79}, /*header only*/ {76, 77}, k_fabric_mux_noc, 1},

    // Remote chip 8
    {83, 8, x, 0, PREFETCH_D, {29, x, x, x}, {89, 85, 87, x}, k_prefetcher_noc},
    {84, 8, x, 1, PREFETCH_D, {30, x, x, x}, {89, 86, 88, x}, k_prefetcher_noc},
    {85, 8, x, 0, DISPATCH_D, {83, x, x, x}, {89, 87, 31, x}, k_dispatcher_noc},
    {86, 8, x, 1, DISPATCH_D, {84, x, x, x}, {89, 88, 32, x}, k_dispatcher_noc},
    {87, 8, x, 0, DISPATCH_S, {83, x, x, x}, {85, x, x, x}, k_dispatcher_s_noc},
    {88, 8, x, 1, DISPATCH_S, {84, x, x, x}, {86, x, x, x}, k_dispatcher_s_noc},
    {89, 8, x, 0, RETURN_FABRIC_MUX, /*full size*/ {85, 86}, /*header only*/ {83, 84}, k_fabric_mux_noc, 1},
};

static const std::vector<DispatchKernelNode> galaxy_nine_chip_arch_1cq = {
    // For MMIO chip, TODO: investigate removing these, they aren't needed
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, k_prefetcher_noc},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, k_dispatcher_noc},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, k_dispatcher_s_noc},

    // Sevicing remote chips 1-4
    {3, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, k_prefetcher_noc},
    {4, 0, 1, 0, DISPATCH_H, {13, x, x, x}, {3, x, x, x}, k_dispatcher_noc},
    {5, 0, 2, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, k_prefetcher_noc},
    {6, 0, 2, 0, DISPATCH_H, {13, x, x, x}, {5, x, x, x}, k_dispatcher_noc},
    {7, 0, 3, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, k_prefetcher_noc},
    {8, 0, 3, 0, DISPATCH_H, {14, x, x, x}, {7, x, x, x}, k_dispatcher_noc},
    {9, 0, 4, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, k_prefetcher_noc},
    {10, 0, 4, 0, DISPATCH_H, {14, x, x, x}, {9, x, x, x}, k_dispatcher_noc},
    {11, 0, 1, 0, PACKET_ROUTER_MUX, {3, 5, 7, 9}, {15, x, x, x}, k_packet_queue_noc},
    {12, 0, 1, 0, DEMUX, {15, x, x, x}, {13, 14, x, x}, k_packet_queue_noc},
    {13, 0, 1, 0, DEMUX, {12, x, x, x}, {4, 6, x, x}, k_packet_queue_noc},
    {14, 0, 1, 0, DEMUX, {12, x, x, x}, {8, 10, x, x}, k_packet_queue_noc},
    {15, 0, 1, 0, US_TUNNELER_REMOTE, {29, 11, x, x}, {29, 12, x, x}, k_packet_queue_noc},

    // Servicing remote chips 5-8
    {16, 0, 5, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, k_prefetcher_noc},
    {17, 0, 5, 0, DISPATCH_H, {26, x, x, x}, {16, x, x, x}, k_dispatcher_noc},
    {18, 0, 6, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, k_prefetcher_noc},
    {19, 0, 6, 0, DISPATCH_H, {26, x, x, x}, {18, x, x, x}, k_dispatcher_noc},
    {20, 0, 7, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, k_prefetcher_noc},
    {21, 0, 7, 0, DISPATCH_H, {27, x, x, x}, {20, x, x, x}, k_dispatcher_noc},
    {22, 0, 8, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, k_prefetcher_noc},
    {23, 0, 8, 0, DISPATCH_H, {27, x, x, x}, {22, x, x, x}, k_dispatcher_noc},
    {24, 0, 5, 0, PACKET_ROUTER_MUX, {16, 18, 20, 22}, {28, x, x, x}, k_packet_queue_noc},
    {25, 0, 5, 0, DEMUX, {28, x, x, x}, {26, 27, x, x}, k_packet_queue_noc},
    {26, 0, 5, 0, DEMUX, {25, x, x, x}, {17, 19, x, x}, k_packet_queue_noc},
    {27, 0, 5, 0, DEMUX, {25, x, x, x}, {21, 23, x, x}, k_packet_queue_noc},
    {28, 0, 5, 0, US_TUNNELER_REMOTE, {59, 24, x, x}, {59, 25, x, x}, k_packet_queue_noc},

    // Remote chip 1
    {29, 1, x, 0, US_TUNNELER_LOCAL, {15, 30, x, x}, {15, 31, 32, x}, k_packet_queue_noc},
    {30, 1, x, 0, MUX_D, {34, 36, x, x}, {29, x, x, x}, k_packet_queue_noc},
    {31, 1, x, 0, PACKET_ROUTER_DEMUX, {29, x, x, x}, {33, 36, x, x}, k_packet_queue_noc},
    {32, 1, x, 0, PACKET_ROUTER_DEMUX, {29, x, x, x}, {36, x, x, x}, k_packet_queue_noc},
    {33, 1, x, 0, PREFETCH_D, {31, x, x, x}, {34, 35, x, x}, k_prefetcher_noc},
    {34, 1, x, 0, DISPATCH_D, {33, x, x, x}, {35, 30, x, x}, k_dispatcher_noc},
    {35, 1, x, 0, DISPATCH_S, {33, x, x, x}, {34, x, x, x}, k_dispatcher_s_noc},
    {36, 1, x, 0, US_TUNNELER_REMOTE, {37, 31, 32, x}, {37, 30, x, x}, k_packet_queue_noc},

    // Remote chip 2
    {37, 2, x, 0, US_TUNNELER_LOCAL, {36, 38, x, x}, {36, 39, 40, x}, k_packet_queue_noc},
    {38, 2, x, 0, MUX_D, {42, 44, x, x}, {37, x, x, x}, k_packet_queue_noc},
    {39, 2, x, 0, PACKET_ROUTER_DEMUX, {37, x, x, x}, {41, 44, x, x}, k_packet_queue_noc},
    {40, 2, x, 0, PACKET_ROUTER_DEMUX, {37, x, x, x}, {44, x, x, x}, k_packet_queue_noc},
    {41, 2, x, 0, PREFETCH_D, {39, x, x, x}, {42, 43, x, x}, k_prefetcher_noc},
    {42, 2, x, 0, DISPATCH_D, {41, x, x, x}, {43, 38, x, x}, k_dispatcher_noc},
    {43, 2, x, 0, DISPATCH_S, {41, x, x, x}, {42, x, x, x}, k_dispatcher_s_noc},
    {44, 2, x, 0, US_TUNNELER_REMOTE, {45, 39, 40, x}, {45, 38, x, x}, k_packet_queue_noc},

    // Remote chip 3
    {45, 3, x, 0, US_TUNNELER_LOCAL, {44, 46, x, x}, {44, 47, 48, x}, k_packet_queue_noc},
    {46, 3, x, 0, MUX_D, {50, 52, x, x}, {45, x, x, x}, k_packet_queue_noc},
    {47, 3, x, 0, PACKET_ROUTER_DEMUX, {45, x, x, x}, {49, 52, x, x}, k_packet_queue_noc},
    {48, 3, x, 0, PACKET_ROUTER_DEMUX, {45, x, x, x}, {52, x, x, x}, k_packet_queue_noc},
    {49, 3, x, 0, PREFETCH_D, {47, x, x, x}, {50, 51, x, x}, k_prefetcher_noc},
    {50, 3, x, 0, DISPATCH_D, {49, x, x, x}, {51, 46, x, x}, k_dispatcher_noc},
    {51, 3, x, 0, DISPATCH_S, {49, x, x, x}, {50, x, x, x}, k_dispatcher_s_noc},
    {52, 3, x, 0, US_TUNNELER_REMOTE, {53, 47, 48, x}, {53, 46, x, x}, k_packet_queue_noc},

    // Remote chip 4
    {53, 4, x, 0, US_TUNNELER_LOCAL, {52, 54, x, x}, {52, 55, x, x}, k_packet_queue_noc},
    {54, 4, x, 0, MUX_D, {57, x, x, x}, {53, x, x, x}, k_packet_queue_noc},
    {55, 4, x, 0, PACKET_ROUTER_DEMUX, {53, x, x, x}, {56, x, x, x}, k_packet_queue_noc},
    {56, 4, x, 0, PREFETCH_D, {55, x, x, x}, {57, 58, x, x}, k_prefetcher_noc},
    {57, 4, x, 0, DISPATCH_D, {56, x, x, x}, {58, 54, x, x}, k_dispatcher_noc},
    {58, 4, x, 0, DISPATCH_S, {56, x, x, x}, {57, x, x, x}, k_dispatcher_s_noc},

    // Remote chip 5
    {59, 5, x, 0, US_TUNNELER_LOCAL, {28, 60, x, x}, {28, 61, 62, x}, k_packet_queue_noc},
    {60, 5, x, 0, MUX_D, {64, 66, x, x}, {59, x, x, x}, k_packet_queue_noc},
    {61, 5, x, 0, PACKET_ROUTER_DEMUX, {59, x, x, x}, {63, 66, x, x}, k_packet_queue_noc},
    {62, 5, x, 0, PACKET_ROUTER_DEMUX, {59, x, x, x}, {66, x, x, x}, k_packet_queue_noc},
    {63, 5, x, 0, PREFETCH_D, {61, x, x, x}, {64, 65, x, x}, k_prefetcher_noc},
    {64, 5, x, 0, DISPATCH_D, {63, x, x, x}, {65, 60, x, x}, k_dispatcher_noc},
    {65, 5, x, 0, DISPATCH_S, {63, x, x, x}, {64, x, x, x}, k_dispatcher_s_noc},
    {66, 5, x, 0, US_TUNNELER_REMOTE, {67, 61, 62, x}, {67, 60, x, x}, k_packet_queue_noc},

    // Remote chip 6
    {67, 6, x, 0, US_TUNNELER_LOCAL, {66, 68, x, x}, {66, 69, 70, x}, k_packet_queue_noc},
    {68, 6, x, 0, MUX_D, {72, 74, x, x}, {67, x, x, x}, k_packet_queue_noc},
    {69, 6, x, 0, PACKET_ROUTER_DEMUX, {67, x, x, x}, {71, 74, x, x}, k_packet_queue_noc},
    {70, 6, x, 0, PACKET_ROUTER_DEMUX, {67, x, x, x}, {74, x, x, x}, k_packet_queue_noc},
    {71, 6, x, 0, PREFETCH_D, {69, x, x, x}, {72, 73, x, x}, k_prefetcher_noc},
    {72, 6, x, 0, DISPATCH_D, {71, x, x, x}, {73, 68, x, x}, k_dispatcher_noc},
    {73, 6, x, 0, DISPATCH_S, {71, x, x, x}, {72, x, x, x}, k_dispatcher_s_noc},
    {74, 6, x, 0, US_TUNNELER_REMOTE, {75, 69, 70, x}, {75, 68, x, x}, k_packet_queue_noc},

    // Remote chip 7
    {75, 7, x, 0, US_TUNNELER_LOCAL, {74, 76, x, x}, {74, 77, 78, x}, k_packet_queue_noc},
    {76, 7, x, 0, MUX_D, {80, 82, x, x}, {75, x, x, x}, k_packet_queue_noc},
    {77, 7, x, 0, PACKET_ROUTER_DEMUX, {75, x, x, x}, {79, 82, x, x}, k_packet_queue_noc},
    {78, 7, x, 0, PACKET_ROUTER_DEMUX, {75, x, x, x}, {82, x, x, x}, k_packet_queue_noc},
    {79, 7, x, 0, PREFETCH_D, {77, x, x, x}, {80, 81, x, x}, k_prefetcher_noc},
    {80, 7, x, 0, DISPATCH_D, {79, x, x, x}, {81, 76, x, x}, k_dispatcher_noc},
    {81, 7, x, 0, DISPATCH_S, {79, x, x, x}, {80, x, x, x}, k_dispatcher_s_noc},
    {82, 7, x, 0, US_TUNNELER_REMOTE, {83, 77, 78, x}, {83, 76, x, x}, k_packet_queue_noc},

    // Remote chip 8
    {83, 8, x, 0, US_TUNNELER_LOCAL, {82, 84, x, x}, {82, 85, x, x}, k_packet_queue_noc},
    {84, 8, x, 0, MUX_D, {87, x, x, x}, {83, x, x, x}, k_packet_queue_noc},
    {85, 8, x, 0, PACKET_ROUTER_DEMUX, {83, x, x, x}, {86, x, x, x}, k_packet_queue_noc},
    {86, 8, x, 0, PREFETCH_D, {85, x, x, x}, {87, 88, x, x}, k_prefetcher_noc},
    {87, 8, x, 0, DISPATCH_D, {86, x, x, x}, {88, 84, x, x}, k_dispatcher_noc},
    {88, 8, x, 0, DISPATCH_S, {86, x, x, x}, {87, x, x, x}, k_dispatcher_s_noc},
};

static const std::vector<DispatchKernelNode> galaxy_nine_chip_arch_2cq = {
    // For MMIO chip
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, 4, x, x}, k_prefetcher_noc},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, 5, x, x}, k_prefetcher_noc},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {4, x, x, x}, k_dispatcher_noc},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {5, x, x, x}, k_dispatcher_noc},
    {4, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {2, x, x, x}, k_dispatcher_s_noc},
    {5, 0, 0, 1, DISPATCH_S, {1, x, x, x}, {3, x, x, x}, k_dispatcher_s_noc},

    // Servicing remote chips 1-4
    {6, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, k_prefetcher_noc},
    {7, 0, 1, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, k_prefetcher_noc},
    {8, 0, 1, 0, DISPATCH_H, {25, x, x, x}, {6, x, x, x}, k_dispatcher_noc},
    {9, 0, 1, 1, DISPATCH_H, {25, x, x, x}, {7, x, x, x}, k_dispatcher_noc},
    {10, 0, 2, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, k_prefetcher_noc},
    {11, 0, 2, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, k_prefetcher_noc},
    {12, 0, 2, 0, DISPATCH_H, {25, x, x, x}, {10, x, x, x}, k_dispatcher_noc},
    {13, 0, 2, 1, DISPATCH_H, {25, x, x, x}, {11, x, x, x}, k_dispatcher_noc},
    {14, 0, 3, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, k_prefetcher_noc},
    {15, 0, 3, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, k_prefetcher_noc},
    {16, 0, 3, 0, DISPATCH_H, {26, x, x, x}, {14, x, x, x}, k_dispatcher_noc},
    {17, 0, 3, 1, DISPATCH_H, {26, x, x, x}, {15, x, x, x}, k_dispatcher_noc},
    {18, 0, 4, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, k_prefetcher_noc},
    {19, 0, 4, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, k_prefetcher_noc},
    {20, 0, 4, 0, DISPATCH_H, {26, x, x, x}, {18, x, x, x}, k_dispatcher_noc},
    {21, 0, 4, 1, DISPATCH_H, {26, x, x, x}, {19, x, x, x}, k_dispatcher_noc},
    {22, 0, 1, 0, PACKET_ROUTER_MUX, {6, 10, 14, 18}, {27, x, x, x}, k_packet_queue_noc},
    {23, 0, 1, 0, PACKET_ROUTER_MUX, {7, 11, 15, 19}, {27, x, x, x}, k_packet_queue_noc},
    {24, 0, 1, 0, DEMUX, {27, x, x, x}, {25, 26, x, x}, k_packet_queue_noc},
    {25, 0, 1, 0, DEMUX, {24, x, x, x}, {8, 9, 12, 13}, k_packet_queue_noc},
    {26, 0, 1, 0, DEMUX, {24, x, x, x}, {16, 17, 20, 21}, k_packet_queue_noc},
    {27, 0, 1, 0, US_TUNNELER_REMOTE, {50, 22, 23, x}, {50, 24, x, x}, k_packet_queue_noc},

    // Servicing remote chips 5-8
    {28, 0, 5, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, k_prefetcher_noc},
    {29, 0, 5, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, k_prefetcher_noc},
    {30, 0, 5, 0, DISPATCH_H, {47, x, x, x}, {28, x, x, x}, k_dispatcher_noc},
    {31, 0, 5, 1, DISPATCH_H, {47, x, x, x}, {29, x, x, x}, k_dispatcher_noc},
    {32, 0, 6, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, k_prefetcher_noc},
    {33, 0, 6, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, k_prefetcher_noc},
    {34, 0, 6, 0, DISPATCH_H, {47, x, x, x}, {32, x, x, x}, k_dispatcher_noc},
    {35, 0, 6, 1, DISPATCH_H, {47, x, x, x}, {33, x, x, x}, k_dispatcher_noc},
    {36, 0, 7, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, k_prefetcher_noc},
    {37, 0, 7, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, k_prefetcher_noc},
    {38, 0, 7, 0, DISPATCH_H, {48, x, x, x}, {36, x, x, x}, k_dispatcher_noc},
    {39, 0, 7, 1, DISPATCH_H, {48, x, x, x}, {37, x, x, x}, k_dispatcher_noc},
    {40, 0, 8, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, k_prefetcher_noc},
    {41, 0, 8, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, k_prefetcher_noc},
    {42, 0, 8, 0, DISPATCH_H, {48, x, x, x}, {40, x, x, x}, k_dispatcher_noc},
    {43, 0, 8, 1, DISPATCH_H, {48, x, x, x}, {41, x, x, x}, k_dispatcher_noc},
    {44, 0, 5, 0, PACKET_ROUTER_MUX, {28, 32, 36, 40}, {49, x, x, x}, k_packet_queue_noc},
    {45, 0, 5, 0, PACKET_ROUTER_MUX, {29, 33, 37, 41}, {49, x, x, x}, k_packet_queue_noc},
    {46, 0, 5, 0, DEMUX, {49, x, x, x}, {47, 48, x, x}, k_packet_queue_noc},
    {47, 0, 5, 0, DEMUX, {46, x, x, x}, {30, 31, 34, 35}, k_packet_queue_noc},
    {48, 0, 5, 0, DEMUX, {46, x, x, x}, {38, 39, 42, 43}, k_packet_queue_noc},
    {49, 0, 5, 0, US_TUNNELER_REMOTE, {93, 44, 45, x}, {93, 46, x, x}, k_packet_queue_noc},

    // Remote chip 1
    {50, 1, x, 0, US_TUNNELER_LOCAL, {27, 51, x, x}, {27, 52, 53, x}, k_packet_queue_noc},
    {51, 1, x, 0, MUX_D, {56, 57, 60, x}, {50, x, x, x}, k_packet_queue_noc},
    {52, 1, x, 0, PACKET_ROUTER_DEMUX, {50, x, x, x}, {54, 60, x, x}, k_packet_queue_noc},
    {53, 1, x, 0, PACKET_ROUTER_DEMUX, {50, x, x, x}, {55, 60, x, x}, k_packet_queue_noc},
    {54, 1, x, 0, PREFETCH_D, {52, x, x, x}, {56, 58, x, x}, k_prefetcher_noc},
    {55, 1, x, 1, PREFETCH_D, {53, x, x, x}, {57, 59, x, x}, k_prefetcher_noc},
    {56, 1, x, 0, DISPATCH_D, {54, x, x, x}, {58, 51, x, x}, k_dispatcher_noc},
    {57, 1, x, 1, DISPATCH_D, {55, x, x, x}, {59, 51, x, x}, k_dispatcher_noc},
    {58, 1, x, 0, DISPATCH_S, {54, x, x, x}, {56, x, x, x}, k_dispatcher_s_noc},
    // TODO: Why does the second dispatch S connect to the first dispatch D? Keep same as previous implementation for
    // now
    {59, 1, x, 1, DISPATCH_S, {54, x, x, x}, {56, x, x, x}, k_dispatcher_s_noc},
    {60, 1, x, 0, US_TUNNELER_REMOTE, {61, 52, 53, x}, {61, 51, x, x}, k_packet_queue_noc},

    // Remote chip 2
    {61, 2, x, 0, US_TUNNELER_LOCAL, {60, 62, x, x}, {60, 63, 64, x}, k_packet_queue_noc},
    {62, 2, x, 0, MUX_D, {67, 68, 71, x}, {61, x, x, x}, k_packet_queue_noc},
    {63, 2, x, 0, PACKET_ROUTER_DEMUX, {61, x, x, x}, {65, 71, x, x}, k_packet_queue_noc},
    {64, 2, x, 0, PACKET_ROUTER_DEMUX, {61, x, x, x}, {66, 71, x, x}, k_packet_queue_noc},
    {65, 2, x, 0, PREFETCH_D, {63, x, x, x}, {67, 69, x, x}, k_prefetcher_noc},
    {66, 2, x, 1, PREFETCH_D, {64, x, x, x}, {68, 70, x, x}, k_prefetcher_noc},
    {67, 2, x, 0, DISPATCH_D, {65, x, x, x}, {69, 62, x, x}, k_dispatcher_noc},
    {68, 2, x, 1, DISPATCH_D, {66, x, x, x}, {70, 62, x, x}, k_dispatcher_noc},
    {69, 2, x, 0, DISPATCH_S, {65, x, x, x}, {67, x, x, x}, k_dispatcher_s_noc},
    {70, 2, x, 1, DISPATCH_S, {65, x, x, x}, {67, x, x, x}, k_dispatcher_s_noc},
    {71, 2, x, 0, US_TUNNELER_REMOTE, {72, 63, 64, x}, {72, 62, x, x}, k_packet_queue_noc},

    // Remote chip 3
    {72, 3, x, 0, US_TUNNELER_LOCAL, {71, 73, x, x}, {71, 74, 75, x}, k_packet_queue_noc},
    {73, 3, x, 0, MUX_D, {78, 79, 82, x}, {72, x, x, x}, k_packet_queue_noc},
    {74, 3, x, 0, PACKET_ROUTER_DEMUX, {72, x, x, x}, {76, 82, x, x}, k_packet_queue_noc},
    {75, 3, x, 0, PACKET_ROUTER_DEMUX, {72, x, x, x}, {77, 82, x, x}, k_packet_queue_noc},
    {76, 3, x, 0, PREFETCH_D, {74, x, x, x}, {78, 80, x, x}, k_prefetcher_noc},
    {77, 3, x, 1, PREFETCH_D, {75, x, x, x}, {79, 81, x, x}, k_prefetcher_noc},
    {78, 3, x, 0, DISPATCH_D, {76, x, x, x}, {80, 73, x, x}, k_dispatcher_noc},
    {79, 3, x, 1, DISPATCH_D, {77, x, x, x}, {81, 73, x, x}, k_dispatcher_noc},
    {80, 3, x, 0, DISPATCH_S, {76, x, x, x}, {78, x, x, x}, k_dispatcher_s_noc},
    {81, 3, x, 1, DISPATCH_S, {76, x, x, x}, {78, x, x, x}, k_dispatcher_s_noc},
    {82, 3, x, 0, US_TUNNELER_REMOTE, {83, 74, 75, x}, {83, 73, x, x}, k_packet_queue_noc},

    // Remote chip 4
    {83, 4, x, 0, US_TUNNELER_LOCAL, {82, 84, x, x}, {82, 85, 86, x}, k_packet_queue_noc},
    {84, 4, x, 0, MUX_D, {89, 90, x, x}, {83, x, x, x}, k_packet_queue_noc},
    {85, 4, x, 0, PACKET_ROUTER_DEMUX, {83, x, x, x}, {87, x, x, x}, k_packet_queue_noc},
    {86, 4, x, 0, PACKET_ROUTER_DEMUX, {83, x, x, x}, {88, x, x, x}, k_packet_queue_noc},
    {87, 4, x, 0, PREFETCH_D, {85, x, x, x}, {89, 91, x, x}, k_prefetcher_noc},
    {88, 4, x, 1, PREFETCH_D, {86, x, x, x}, {90, 92, x, x}, k_prefetcher_noc},
    {89, 4, x, 0, DISPATCH_D, {87, x, x, x}, {91, 84, x, x}, k_dispatcher_noc},
    {90, 4, x, 1, DISPATCH_D, {88, x, x, x}, {92, 84, x, x}, k_dispatcher_noc},
    {91, 4, x, 0, DISPATCH_S, {87, x, x, x}, {89, x, x, x}, k_dispatcher_s_noc},
    {92, 4, x, 1, DISPATCH_S, {87, x, x, x}, {89, x, x, x}, k_dispatcher_s_noc},

    // Remote chip 5
    {93, 5, x, 0, US_TUNNELER_LOCAL, {49, 94, x, x}, {49, 95, 96, x}, k_packet_queue_noc},
    {94, 5, x, 0, MUX_D, {99, 100, 103, x}, {93, x, x, x}, k_packet_queue_noc},
    {95, 5, x, 0, PACKET_ROUTER_DEMUX, {93, x, x, x}, {97, 103, x, x}, k_packet_queue_noc},
    {96, 5, x, 0, PACKET_ROUTER_DEMUX, {93, x, x, x}, {98, 103, x, x}, k_packet_queue_noc},
    {97, 5, x, 0, PREFETCH_D, {95, x, x, x}, {99, 101, x, x}, k_prefetcher_noc},
    {98, 5, x, 1, PREFETCH_D, {96, x, x, x}, {100, 102, x, x}, k_prefetcher_noc},
    {99, 5, x, 0, DISPATCH_D, {97, x, x, x}, {101, 94, x, x}, k_dispatcher_noc},
    {100, 5, x, 1, DISPATCH_D, {98, x, x, x}, {102, 94, x, x}, k_dispatcher_noc},
    {101, 5, x, 0, DISPATCH_S, {97, x, x, x}, {99, x, x, x}, k_dispatcher_s_noc},
    {102, 5, x, 1, DISPATCH_S, {97, x, x, x}, {99, x, x, x}, k_dispatcher_s_noc},
    {103, 5, x, 0, US_TUNNELER_REMOTE, {104, 95, 96, x}, {104, 94, x, x}, k_packet_queue_noc},

    // Remote chip 6
    {104, 6, x, 0, US_TUNNELER_LOCAL, {103, 105, x, x}, {103, 106, 107, x}, k_packet_queue_noc},
    {105, 6, x, 0, MUX_D, {110, 111, 114, x}, {104, x, x, x}, k_packet_queue_noc},
    {106, 6, x, 0, PACKET_ROUTER_DEMUX, {104, x, x, x}, {108, 114, x, x}, k_packet_queue_noc},
    {107, 6, x, 0, PACKET_ROUTER_DEMUX, {104, x, x, x}, {109, 114, x, x}, k_packet_queue_noc},
    {108, 6, x, 0, PREFETCH_D, {106, x, x, x}, {110, 112, x, x}, k_prefetcher_noc},
    {109, 6, x, 1, PREFETCH_D, {107, x, x, x}, {111, 113, x, x}, k_prefetcher_noc},
    {110, 6, x, 0, DISPATCH_D, {108, x, x, x}, {112, 105, x, x}, k_dispatcher_noc},
    {111, 6, x, 1, DISPATCH_D, {109, x, x, x}, {113, 105, x, x}, k_dispatcher_noc},
    {112, 6, x, 0, DISPATCH_S, {108, x, x, x}, {110, x, x, x}, k_dispatcher_s_noc},
    {113, 6, x, 1, DISPATCH_S, {108, x, x, x}, {110, x, x, x}, k_dispatcher_s_noc},
    {114, 6, x, 0, US_TUNNELER_REMOTE, {115, 106, 107, x}, {115, 105, x, x}, k_packet_queue_noc},

    // Remote chip 7
    {115, 7, x, 0, US_TUNNELER_LOCAL, {114, 116, x, x}, {114, 117, 118, x}, k_packet_queue_noc},
    {116, 7, x, 0, MUX_D, {121, 122, 125, x}, {115, x, x, x}, k_packet_queue_noc},
    {117, 7, x, 0, PACKET_ROUTER_DEMUX, {115, x, x, x}, {119, 125, x, x}, k_packet_queue_noc},
    {118, 7, x, 0, PACKET_ROUTER_DEMUX, {115, x, x, x}, {120, 125, x, x}, k_packet_queue_noc},
    {119, 7, x, 0, PREFETCH_D, {117, x, x, x}, {121, 123, x, x}, k_prefetcher_noc},
    {120, 7, x, 1, PREFETCH_D, {118, x, x, x}, {122, 124, x, x}, k_prefetcher_noc},
    {121, 7, x, 0, DISPATCH_D, {119, x, x, x}, {123, 116, x, x}, k_dispatcher_noc},
    {122, 7, x, 1, DISPATCH_D, {120, x, x, x}, {124, 116, x, x}, k_dispatcher_noc},
    {123, 7, x, 0, DISPATCH_S, {119, x, x, x}, {121, x, x, x}, k_dispatcher_s_noc},
    {124, 7, x, 1, DISPATCH_S, {119, x, x, x}, {121, x, x, x}, k_dispatcher_s_noc},
    {125, 7, x, 0, US_TUNNELER_REMOTE, {126, 117, 118, x}, {126, 116, x, x}, k_packet_queue_noc},

    // Remote chip 8
    {126, 8, x, 0, US_TUNNELER_LOCAL, {125, 127, x, x}, {125, 128, 129, x}, k_packet_queue_noc},
    {127, 8, x, 0, MUX_D, {132, 133, x, x}, {126, x, x, x}, k_packet_queue_noc},
    {128, 8, x, 0, PACKET_ROUTER_DEMUX, {126, x, x, x}, {130, x, x, x}, k_packet_queue_noc},
    {129, 8, x, 0, PACKET_ROUTER_DEMUX, {126, x, x, x}, {131, x, x, x}, k_packet_queue_noc},
    {130, 8, x, 0, PREFETCH_D, {128, x, x, x}, {132, 134, x, x}, k_prefetcher_noc},
    {131, 8, x, 1, PREFETCH_D, {129, x, x, x}, {133, 135, x, x}, k_prefetcher_noc},
    {132, 8, x, 0, DISPATCH_D, {130, x, x, x}, {134, 127, x, x}, k_dispatcher_noc},
    {133, 8, x, 1, DISPATCH_D, {131, x, x, x}, {135, 127, x, x}, k_dispatcher_noc},
    {134, 8, x, 0, DISPATCH_S, {130, x, x, x}, {132, x, x, x}, k_dispatcher_s_noc},
    {135, 8, x, 1, DISPATCH_S, {130, x, x, x}, {132, x, x, x}, k_dispatcher_s_noc},
};
// clang-format on

std::vector<FDKernel*> node_id_to_kernel;
tt::tt_metal::detail::ProgramCompileGroup command_queue_compile_group;
std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> dispatch_cores;
std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> routing_cores;
std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> empty_cores;
std::unordered_map<chip_id_t, std::unordered_set<TerminationInfo>> termination_info;

// Helper function to automatically generate dispatch nodes given devices + num hw CQs + detection of card type.
std::vector<DispatchKernelNode> generate_nodes(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs) {
    // Select/generate the right input table, depends on (1) board [detected from total # of devices], and (2) number
    // of active devices. TODO: read this out of YAML instead of the structs above?
    uint32_t total_devices = tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices();
    TT_ASSERT(
        total_devices == 1 or total_devices == 2 or total_devices == 4 or total_devices == 8 or total_devices == 32 or
            total_devices == 36,
        "Unexpected target.");
    uint32_t num_devices = device_ids.size();
    TT_ASSERT(num_devices > 0, "Can't determine dispatch architecture with no active devices.");
    TT_ASSERT(num_devices <= total_devices);
    std::vector<DispatchKernelNode> nodes;

    std::set<chip_id_t> mmio_devices;
    std::set<chip_id_t> remote_devices;
    for (auto id : device_ids) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(id) == id) {
            mmio_devices.insert(id);
        } else {
            remote_devices.insert(id);
        }
    }

    // Helper function to get nodes for single device
    auto populate_single_device = [&]() {
        if (num_hw_cqs == 1) {
            return single_chip_arch_1cq;
        } else {
            // TODO: determine whether dispatch_s is inserted at this level, instead of inside
            // Device::dispatch_s_enabled().
            if (MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type() == CoreType::WORKER) {
                return single_chip_arch_2cq_dispatch_s;
            } else {
                return single_chip_arch_2cq;
            }
        }
    };

    if (remote_devices.empty()) {
        // MMIO devices only, just replicate a single chip arch for each
        std::vector<DispatchKernelNode> nodes_for_one_mmio = populate_single_device();
        uint32_t index_offset = 0;
        for (auto id : mmio_devices) {
            for (auto node : nodes_for_one_mmio) {
                node.device_id = id;
                node.servicing_device_id = id;
                increment_node_ids(node, index_offset);
                nodes.push_back(node);
            }
            index_offset += nodes_for_one_mmio.size();
        }
    } else {
        // Need to handle N300/T3000 separately from TG/TGG since they have different templates/tunnel depths
        // If using fabric, upstream would have already initalized to the proper config for dispatch
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
            // For Galaxy, we always init all remote devices associated with an mmio device.
            std::vector<DispatchKernelNode> nodes_for_one_mmio;
            if (rtoptions.get_fd_fabric()) {
                nodes_for_one_mmio =
                    (num_hw_cqs == 1) ? galaxy_nine_chip_arch_1cq_fabric : galaxy_nine_chip_arch_2cq_fabric;
            } else {
                nodes_for_one_mmio = (num_hw_cqs == 1) ? galaxy_nine_chip_arch_1cq : galaxy_nine_chip_arch_2cq;
            }
            uint32_t index_offset = 0;
            for (auto mmio_device_id : mmio_devices) {
                // Need a mapping from templated device id (1-8) to actual device id (from the tunnel)
                std::vector<chip_id_t> template_id_to_device_id;
                template_id_to_device_id.push_back(mmio_device_id);
                for (const auto& tunnel :
                     tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(
                         mmio_device_id)) {
                    TT_ASSERT(tunnel.size() == 5, "Galaxy expected 4-deep tunnels.");
                    for (auto remote_device_id : tunnel) {
                        if (remote_device_id != mmio_device_id) {
                            template_id_to_device_id.push_back(remote_device_id);
                        }
                    }
                }

                // Pull nodes from the template, updating their index and device id
                for (DispatchKernelNode node : nodes_for_one_mmio) {
                    int32_t num_devices = template_id_to_device_id.size();
                    TT_ASSERT(
                        node.device_id < num_devices,
                        "Device id {} out of bounds (max = {})",
                        node.device_id,
                        num_devices);
                    TT_ASSERT(
                        node.servicing_device_id < num_devices,
                        "Servicing device id {} out of bounds (max = {})",
                        node.servicing_device_id,
                        num_devices);
                    node.device_id = template_id_to_device_id[node.device_id];
                    node.servicing_device_id = template_id_to_device_id[node.servicing_device_id];
                    increment_node_ids(node, index_offset);
                    nodes.push_back(node);
                }
                index_offset += nodes_for_one_mmio.size();
            }
        } else {
            // Should be paired mmio/remote devices
            TT_ASSERT(
                mmio_devices.size() == remote_devices.size() or remote_devices.empty(),
                "N300/T3K expects devices in mmio/remote pairs.");
            std::vector<DispatchKernelNode> nodes_for_one_mmio;
            if (rtoptions.get_fd_fabric()) {
                nodes_for_one_mmio = (num_hw_cqs == 1) ? two_chip_arch_1cq_fabric : two_chip_arch_2cq_fabric;
            } else {
                nodes_for_one_mmio = (num_hw_cqs == 1) ? two_chip_arch_1cq : two_chip_arch_2cq;
            }

            uint32_t index_offset = 0;
            for (auto mmio_device_id : mmio_devices) {
                // Find the corresponding remote chip
                chip_id_t remote_device_id{};
                bool found_remote = false;
                for (auto id : remote_devices) {
                    if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(id) ==
                        mmio_device_id) {
                        remote_device_id = id;
                        found_remote = true;
                        break;
                    }
                }
                TT_ASSERT(found_remote, "Couldn't find paired remote chip for device {}", mmio_device_id);

                // Add dispatch kernels for the mmio/remote pair
                for (DispatchKernelNode node : nodes_for_one_mmio) {
                    constexpr uint32_t k_MMIO = 0;
                    constexpr uint32_t k_Remote = 1;
                    TT_ASSERT(node.device_id == k_MMIO || node.device_id == k_Remote);
                    TT_ASSERT(
                        node.servicing_device_id == k_MMIO || node.servicing_device_id == k_Remote ||
                        node.servicing_device_id == x);

                    if (node.device_id == k_MMIO) {
                        node.device_id = mmio_device_id;
                    } else {
                        // node.device_id == k_Remote
                        node.device_id = remote_device_id;
                    }

                    if (node.servicing_device_id == k_MMIO) {
                        node.servicing_device_id = mmio_device_id;
                    } else if (node.servicing_device_id == k_Remote) {
                        node.servicing_device_id = remote_device_id;
                    }
                    increment_node_ids(node, index_offset);
                    nodes.push_back(node);
                }
                index_offset += nodes_for_one_mmio.size();
            }
        }
    }

    return nodes;
}

// Populate node_id_to_kernel and set up kernel objects. Do this once at the beginning since they (1) don't need a valid
// Device until fields are populated, (2) need to be connected to kernel objects for devices that aren't created yet,
// and (3) the table to choose depends on total number of devices, not know at Device creation.
void populate_fd_kernels(const std::vector<IDevice*>& devices, uint32_t num_hw_cqs) {
    std::set<chip_id_t> device_ids;
    for (const auto& device : devices) {
        device_ids.insert(device->id());
    }
    populate_fd_kernels(generate_nodes(device_ids, num_hw_cqs));
}

void populate_fd_kernels(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs) {
    populate_fd_kernels(generate_nodes(device_ids, num_hw_cqs));
}

void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes) {
    // If we already had nodes from a previous run, clear them (since we could have a different # of devices or CQs).
    if (!node_id_to_kernel.empty()) {
        for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
            delete node_id_to_kernel[idx];
        }
        node_id_to_kernel.clear();
        command_queue_compile_group.clear();
    }

    // Read the input table, create configs for each node + track mmio devices and number of cqs.
    std::unordered_set<chip_id_t> mmio_device_ids;
    std::unordered_set<uint8_t> hw_cq_ids;
    for (const auto& node : nodes) {
        TT_ASSERT(node_id_to_kernel.size() == node.id);
        node_id_to_kernel.push_back(FDKernel::Generate(
            node.id,
            node.device_id,
            node.servicing_device_id,
            node.cq_id,
            node.noc_selection,
            node.kernel_type,
            node.tunnel_index));
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(node.device_id) ==
            node.device_id) {
            mmio_device_ids.insert(node.device_id);
        }
        hw_cq_ids.insert(node.cq_id);
    }
    uint32_t num_hw_cqs = hw_cq_ids.size();

    // Connect the graph with upstream/downstream kernels
    for (const auto& node : nodes) {
        for (int idx = 0; idx < node.upstream_ids.size(); idx++) {
            if (node.upstream_ids[idx] >= 0) {
                TT_ASSERT(
                    node.upstream_ids[idx] < node_id_to_kernel.size(),
                    "Upstream kernel id {} out of bounds (max = {})",
                    node.upstream_ids[idx],
                    node_id_to_kernel.size());
                node_id_to_kernel.at(node.id)->AddUpstreamKernel(node_id_to_kernel.at(node.upstream_ids[idx]));
            }
        }
        for (int idx = 0; idx < node.downstream_ids.size(); idx++) {
            if (node.downstream_ids[idx] >= 0) {
                TT_ASSERT(
                    node.downstream_ids[idx] < node_id_to_kernel.size(),
                    "Downstream kernel id {} out of bounds (max = {})",
                    node.downstream_ids[idx],
                    node_id_to_kernel.size());
                node_id_to_kernel.at(node.id)->AddDownstreamKernel(node_id_to_kernel.at(node.downstream_ids[idx]));
            }
        }
    }

    // For kernels on mmio chip, need to confirm which remote device each is servicing
    std::map<chip_id_t, uint32_t> device_id_to_tunnel_stop;
    std::map<chip_id_t, std::vector<chip_id_t>> mmio_device_id_to_serviced_devices;
    uint32_t tunnel_depth{};
    for (auto mmio_device_id : mmio_device_ids) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(mmio_device_id) !=
            mmio_device_id) {
            continue;
        }

        // Get a list of remote devices serviced by this mmio chip
        for (int idx = 0; idx < num_hw_cqs; idx++) {
            mmio_device_id_to_serviced_devices[mmio_device_id].push_back(mmio_device_id);
        }
        std::vector<chip_id_t> remote_devices;
        for (auto tunnel :
             tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
            tunnel_depth = tunnel.size();
            for (uint32_t tunnel_stop = 0; tunnel_stop < tunnel.size(); tunnel_stop++) {
                chip_id_t remote_device_id = tunnel[tunnel_stop];
                device_id_to_tunnel_stop[remote_device_id] = tunnel_stop;
                if (remote_device_id != mmio_device_id) {
                    for (int idx = 0; idx < num_hw_cqs; idx++) {
                        remote_devices.push_back(remote_device_id);
                    }
                }
            }
        }

        mmio_device_id_to_serviced_devices[mmio_device_id].insert(
            mmio_device_id_to_serviced_devices[mmio_device_id].end(), remote_devices.begin(), remote_devices.end());
    }

    // Go through each mmio device, and set placement cq_ids to ensure that we stamp out the correct # of demux/routers
    // per tunnel. TODO: We can fix dispatch_core_manager so we don't hard-code separate channels for this.
    std::map<chip_id_t, std::vector<FDKernel*>> mmio_device_id_to_kernels;
    for (auto fd_kernel : node_id_to_kernel) {
        chip_id_t mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(fd_kernel->GetDeviceId());
        if (fd_kernel->GetDeviceId() == mmio_device_id) {
            mmio_device_id_to_kernels[mmio_device_id].push_back(fd_kernel);
        }
    }
    for (auto& mmio_device_id_and_kernels : mmio_device_id_to_kernels) {
        int demux_id = 0, router_id = 0;
        for (auto fd_kernel : mmio_device_id_and_kernels.second) {
            if (auto demux_kernel = dynamic_cast<DemuxKernel*>(fd_kernel)) {
                demux_kernel->SetPlacementCQID((demux_id++) % 3);
            } else if (auto router_kernel = dynamic_cast<EthRouterKernel*>(fd_kernel)) {
                router_kernel->SetPlacementCQID((router_id++) % num_hw_cqs);
            }
        }
    }

    // Write VC count to all tunnelers
    std::map<chip_id_t, uint32_t> device_id_to_num_routers;  // Need to build this first. TODO: in the future walk the
                                                             // graph to populate VC counts
    std::map<chip_id_t, uint32_t> device_id_to_remaining_routers;
    for (auto fd_kernel : node_id_to_kernel) {
        if (auto router_kernel = dynamic_cast<EthRouterKernel*>(fd_kernel)) {
            chip_id_t router_device_id = router_kernel->GetDeviceId();
            chip_id_t mmio_device_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(router_device_id);

            // Router placement CQID already set for mmio device above, just need to do this for remotes
            if (router_device_id != mmio_device_id) {
                router_kernel->SetPlacementCQID(device_id_to_num_routers[router_device_id]);
            }
            device_id_to_remaining_routers[router_device_id]++;
            device_id_to_num_routers[router_device_id]++;
        }
    }
    for (auto fd_kernel : node_id_to_kernel) {
        uint32_t tunnel_stop = device_id_to_tunnel_stop[fd_kernel->GetDeviceId()];
        if (auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(fd_kernel)) {
            // Local tunneler needs to match remote tunneler on previous chip in the tunnel
            if (!tunneler_kernel->IsRemote()) {
                TT_ASSERT(tunnel_stop != 0);
                tunnel_stop--;
            }
            // # of VCs is return VC + total VCs for tunnel (num_hw_cqs per remote) - num_hw_cqs VCs per remote stop
            tunneler_kernel->SetVCCount(1 + (tunnel_depth - 1) * num_hw_cqs - tunnel_stop * num_hw_cqs);
        } else if (auto router_kernel = dynamic_cast<EthRouterKernel*>(fd_kernel)) {
            // Router has the same VCs as the remote tunneler for MMIO, same as local tunneler for remote
            if (tunnel_stop != 0) {
                tunnel_stop--;
            }
            uint32_t router_vcs_for_device = (tunnel_depth - 1) * num_hw_cqs - tunnel_stop * num_hw_cqs;
            // Divide the VCs between routers on one device
            uint32_t remaining_routers = device_id_to_remaining_routers[router_kernel->GetDeviceId()];
            uint32_t num_routers = device_id_to_num_routers[router_kernel->GetDeviceId()];

            // Special case for MMIO chips, can have routers servicing different tunnels
            chip_id_t mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(
                router_kernel->GetDeviceId());
            if (router_kernel->GetDeviceId() == mmio_device_id) {
                uint32_t num_tunnels = tt::tt_metal::MetalContext::instance()
                                           .get_cluster()
                                           .get_tunnels_from_mmio_device(mmio_device_id)
                                           .size();
                num_routers /= num_tunnels;
                uint32_t router_vcs = (router_vcs_for_device + num_routers - 1) / num_routers;
                router_kernel->SetVCCount(router_vcs);
            } else {
                uint32_t router_vcs = (router_vcs_for_device + remaining_routers - 1) / num_routers;
                router_kernel->SetVCCount(router_vcs);
            }
            device_id_to_remaining_routers[router_kernel->GetDeviceId()]--;
        }
    }
}

void populate_cq_static_args(IDevice* device) {
    TT_ASSERT(
        node_id_to_kernel.size() > 0,
        "Tried to populate static args on nodes without the nodes populated (need to run populate_fd_kernels()");
    // First pass, add device/program to all kernels for this device and generate static configs.
    auto cq_program_ptr = std::make_unique<Program>();
    for (auto node_and_kernel : node_id_to_kernel) {
        // GetDeviceId() uses Id from topology as IDevice* is not present yet
        if (node_and_kernel->GetDeviceId() == device->id()) {
            node_and_kernel->AddDevice(device);
            // TODO: Be careful downstream. Using get() on a smart pointer defeats the purpose of using them
            // Memory could be changed at that location later.
            node_and_kernel->AddProgram(cq_program_ptr.get());
            node_and_kernel->GenerateStaticConfigs();
        }
    }

    // Move program into the storage for later steps
    command_queue_compile_group.add_program(device, std::move(cq_program_ptr));
}

void create_cq_program(IDevice* device) {
    TT_FATAL(
        command_queue_compile_group.contains(device),
        "Tried to create and compile CQ program on device {} without static args populated (need to run "
        "populate_cq_static_args())",
        device->id());
    empty_cores.clear();
    // Third pass, populate dependent configs and create kernels for each node
    for (auto node_and_kernel : node_id_to_kernel) {
        if (node_and_kernel->GetDeviceId() == device->id()) {
            node_and_kernel->GenerateDependentConfigs();
        }
    }

    for (auto node_and_kernel : node_id_to_kernel) {
        if (node_and_kernel->GetDeviceId() == device->id()) {
            node_and_kernel->CreateKernel();
        }
    }

    // Register core coordinates for this device
    for (auto node_and_kernel : node_id_to_kernel) {
        if (node_and_kernel->GetDeviceId() != device->id()) {
            continue;
        }

        switch (node_and_kernel->GetKernelType()) {
            case FDKernelType::DISPATCH: dispatch_cores[device->id()].insert(node_and_kernel->GetVirtualCore()); break;
            case FDKernelType::ROUTING: routing_cores[device->id()].insert(node_and_kernel->GetVirtualCore()); break;
            case FDKernelType::VIRTUAL:
                // Not a real kernel
                break;
            case FDKernelType::UNSET:
                TT_THROW(
                    "Unknown kernel type {} {} on Device {}",
                    magic_enum::enum_name(node_and_kernel->GetKernelType()),
                    typeid(*node_and_kernel).name(),
                    device->id());
                break;
        }
    }

    // Register termination info
    for (auto node_and_kernel : node_id_to_kernel) {
        if (node_and_kernel->GetDeviceId() != device->id()) {
            continue;
        }

        const auto& info = node_and_kernel->GetTerminationInfo();
        if (info.has_value()) {
            termination_info[device->id()].insert(info.value());
        }
    }
}

void compile_cq_programs() {
    if (tt_metal::MetalContext::instance().rtoptions().get_skip_loading_fw()) {
        detail::EnablePersistentKernelCache();
    }

    command_queue_compile_group.compile_all(/*force_slow_dispatch=*/true);

    // Write runtime args to device
    command_queue_compile_group.write_runtime_args(/*force_slow_dispatch=*/true);

    if (tt_metal::MetalContext::instance().rtoptions().get_skip_loading_fw()) {
        detail::DisablePersistentKernelCache();
    }
}

std::unique_ptr<tt::tt_metal::Program> get_compiled_cq_program(tt::tt_metal::IDevice* device) {
    return command_queue_compile_group.remove_program(device);
}

void configure_dispatch_cores(IDevice* device) {
    // Set up completion_queue_writer core. This doesn't actually have a kernel so keep it out of the struct and config
    // it here. TODO: should this be in the struct?
    CoreType dispatch_core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map();
    uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    uint32_t cq_size = device->sysmem_manager().get_cq_size();
    std::vector<uint32_t> zero = {0x0};

    // Need to set up for all devices serviced by an mmio chip
    if (device->is_mmio_capable()) {
        for (chip_id_t serviced_device_id :
             tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(device->id())) {
            uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
                serviced_device_id);
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
                tt_cxy_pair completion_q_writer_location =
                    MetalContext::instance().get_dispatch_core_manager().completion_queue_writer_core(
                        serviced_device_id, channel, cq_id);
                IDevice* mmio_device = tt::DevicePool::instance().get_active_device(completion_q_writer_location.chip);
                uint32_t completion_q_wr_ptr =
                    my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
                uint32_t completion_q_rd_ptr =
                    my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
                uint32_t completion_q0_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(
                    CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT);
                uint32_t completion_q1_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(
                    CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT);
                // Initialize completion queue write pointer and read pointer copy
                uint32_t issue_queue_size = device->sysmem_manager().get_issue_queue_size(cq_id);
                uint32_t completion_queue_start_addr =
                    cq_start + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
                std::vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
                detail::WriteToDeviceL1(
                    mmio_device,
                    completion_q_writer_location,
                    completion_q_rd_ptr,
                    completion_queue_wr_ptr,
                    dispatch_core_type);
                detail::WriteToDeviceL1(
                    mmio_device,
                    completion_q_writer_location,
                    completion_q_wr_ptr,
                    completion_queue_wr_ptr,
                    dispatch_core_type);
                detail::WriteToDeviceL1(
                    mmio_device, completion_q_writer_location, completion_q0_last_event_ptr, zero, dispatch_core_type);
                detail::WriteToDeviceL1(
                    mmio_device, completion_q_writer_location, completion_q1_last_event_ptr, zero, dispatch_core_type);
            }
        }
    }
    // Configure cores for all nodes corresponding to this device
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        if (node_id_to_kernel[idx]->GetDeviceId() == device->id()) {
            node_id_to_kernel[idx]->ConfigureCore();
        }
    }
}

std::pair<tt::tt_fabric::FabricEriscDatamoverType, tt::tt_fabric::FabricEriscDatamoverAxis> get_fabric_edm_type(
    const tt::tt_fabric::ControlPlane& control_plane,
    tt_fabric::Topology topology,
    tt::tt_fabric::MeshId mesh_id,
    chip_id_t chip0,
    chip_id_t chip1,
    bool wrap_around_mesh) {
    if (topology != tt_fabric::Topology::Ring) {
        return {tt::tt_fabric::FabricEriscDatamoverType::Default, tt::tt_fabric::FabricEriscDatamoverAxis::Short};
    }

    auto physical_mesh_shape = control_plane.get_physical_mesh_shape(mesh_id);
    TT_FATAL(physical_mesh_shape.dims() == 2, "Dateline routing only supported for 2D mesh");

    auto mesh_num_rows = physical_mesh_shape[0];
    auto mesh_num_columns = physical_mesh_shape[1];
    auto fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Default;
    auto fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Short;

    auto smaller_chip_id = std::min(chip0, chip1);
    auto larger_chip_id = std::max(chip0, chip1);

    // Refactor this once mesh_id has row/col control
    // wrap_around_mesh is used to fold the edm connections on the corner chips of a 2D mesh to form an outer ring of
    // devices on the mesh.
    if (wrap_around_mesh) {
        // Wrap around dateline
        if (smaller_chip_id == 0 && larger_chip_id == mesh_num_columns) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
        } else if ((chip0 == 0 || chip0 == mesh_num_columns) && chip1 == chip0 + 1) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
        } else if ((chip1 == 0 || chip1 == mesh_num_columns) && chip0 == chip1 + 1) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
        } else if ((chip0 == 1 || chip0 == mesh_num_columns + 1) && (chip1 == chip0 + 1)) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;
        }
        // check if edm is on the longer axis
        if ((mesh_num_rows * mesh_num_columns) >=
            tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD) {
            fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Long;
        }
    } else {
        bool is_dateline_edm_along_column =
            smaller_chip_id % mesh_num_columns == 0 && larger_chip_id == (smaller_chip_id + mesh_num_columns - 1);
        bool is_dateline_edm_along_row = smaller_chip_id < mesh_num_columns &&
                                         larger_chip_id >= (mesh_num_columns * (mesh_num_rows - 1)) &&
                                         smaller_chip_id == larger_chip_id % mesh_num_columns;
        bool is_dateline_upstream_edm_along_column =
            (chip0 % mesh_num_columns == 0 && chip1 == chip0 + 1) ||
            (chip0 % mesh_num_columns == mesh_num_columns - 1 && chip1 == chip0 - 1);
        bool is_dateline_upstream_edm_along_row =
            (chip0 < mesh_num_columns && chip1 == chip0 + mesh_num_columns) ||
            (chip0 >= (mesh_num_columns * (mesh_num_rows - 1)) && chip1 == chip0 - mesh_num_columns);
        bool is_dateline_upstream_adjacent_edm_along_column =
            (chip1 % mesh_num_columns == 0 && chip0 == chip1 + 1) ||
            (chip1 % mesh_num_columns == mesh_num_columns - 1 && chip0 == chip1 - 1);
        bool is_dateline_upstream_adjacent_edm_along_row =
            (chip1 < mesh_num_columns && chip0 == chip1 + mesh_num_columns) ||
            (chip1 >= (mesh_num_columns * (mesh_num_rows - 1)) && chip0 == chip1 - mesh_num_columns);
        bool is_dateline_upstream_adjacent_upstream_edm_along_column =
            (chip0 % mesh_num_columns == 1 && chip1 == chip0 + 1) ||
            (chip0 % mesh_num_columns == mesh_num_columns - 2 && chip1 == chip0 - 1);
        bool is_dateline_upstream_adjacent_upstream_edm_along_row =
            (chip0 >= mesh_num_columns && chip0 < (2 * mesh_num_columns) && chip1 == chip0 + mesh_num_columns) ||
            (chip0 >= (mesh_num_columns * (mesh_num_rows - 2)) && chip0 < (mesh_num_columns * (mesh_num_rows - 1)) &&
             chip1 == chip0 - mesh_num_columns);
        bool is_edm_along_row = ((larger_chip_id - smaller_chip_id) == mesh_num_columns) ||
                                (smaller_chip_id == larger_chip_id % mesh_num_columns);
        // Column dateline
        if (is_dateline_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
        }
        // Row dateline
        else if (is_dateline_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
        }
        // Column dateline upstream
        else if (is_dateline_upstream_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
        }
        // Row dateline upstream
        else if (is_dateline_upstream_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
        }
        // Column dateline upstream adjacent
        else if (is_dateline_upstream_adjacent_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
        }
        // Row dateline upstream adjacent
        else if (is_dateline_upstream_adjacent_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
        }
        // Column dateline upstream adjacent device upstream
        else if (is_dateline_upstream_adjacent_upstream_edm_along_column) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;
        }
        // Row dateline upstream adjacent device upstream
        else if (is_dateline_upstream_adjacent_upstream_edm_along_row) {
            fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;
        }

        // check if edm is on the longer axis
        if ((mesh_num_columns >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD &&
             !is_edm_along_row) ||
            (mesh_num_rows >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD &&
             is_edm_along_row)) {
            fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Long;
        }
    }

    return {fabric_edm_type, fabric_edm_axis};
}

void build_tt_fabric_program(
    IDevice* device,
    Program* fabric_program_ptr,
    std::unordered_map<tt::tt_fabric::chan_id_t, tt::tt_fabric::FabricEriscDatamoverBuilder>& edm_builders) {
    using namespace tt_fabric;
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const bool is_TG = (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG);
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& edm_config = fabric_context.get_fabric_router_config();
    const auto configure_edm_builder_for_dispatch = [&](tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder) {
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fd_fabric()) {
            return;
        }
        constexpr uint32_t k_DispatchFabricRouterContextSwitchInterval = 16;
        // Dispatch requires a higher context switching freq to service slow dispatch / UMD / debug tools
        edm_builder.set_firmware_context_switch_interval(k_DispatchFabricRouterContextSwitchInterval);
        edm_builder.set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
    };

    if (is_TG && device->is_mmio_capable()) {
        auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
        for (const auto& [eth_chan, eth_direction] : router_chans_and_direction) {
            // remote_fabric_node_id is only used to determine the handshake master, no functional impact
            // for now treat the mmio chips as the handshake master
            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                device,
                *fabric_program_ptr,
                eth_logical_core,
                fabric_node_id,
                FabricNodeId{fabric_node_id.mesh_id, fabric_node_id.chip_id + 1},
                edm_config,
                false, /* build_in_worker_connection_mode */
                false, /* is_dateline */
                eth_direction);
            // Both links used by dispatch on TG Gateway (mmio device)
            // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
            configure_edm_builder_for_dispatch(edm_builder);
            edm_builders.insert({eth_chan, edm_builder});
        }

        return;
    }

    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> active_fabric_eth_channels;
    std::unordered_map<RoutingDirection, FabricNodeId> chip_neighbors;
    uint32_t num_intra_chip_neighbors = 0;
    const auto topology = fabric_context.get_fabric_topology();
    const bool is_2D_routing = topology == Topology::Mesh;

    const auto device_has_dispatch_tunnel = [&]() -> bool {
        auto mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
        // results are inclusive of the mmio_device_id so they will never be zero
        TT_ASSERT(tunnels_from_mmio.size() > 0);
        return (tunnels_from_mmio.size() - 1) > 0;
    }();

    for (const auto& direction : tt::tt_fabric::FabricContext::routing_directions) {
        auto active_eth_chans =
            control_plane.get_active_fabric_eth_routing_planes_in_direction(fabric_node_id, direction);
        if (active_eth_chans.empty()) {
            continue;
        }
        auto neighbors = control_plane.get_chip_neighbors(fabric_node_id, direction);
        auto intra_chip_neighbors = neighbors.find(fabric_node_id.mesh_id);
        if (intra_chip_neighbors != neighbors.end()) {
            // only count the number of unique intra chip neighbors
            // we assume that all neighbors in a direction are the same
            num_intra_chip_neighbors++;
        }
        // assume same neighbor per direction
        TT_FATAL(neighbors.size() == 1, "Multiple neighbor meshes per direction is unsupported");
        TT_FATAL(
            std::set<chip_id_t>(neighbors.begin()->second.begin(), neighbors.begin()->second.end()).size() == 1,
            "Multiple neighbors per direction is currently unsupported");

        // 1D fabric only supports intramesh connections apart from TG gateways
        if (!is_2D_routing) {
            uint32_t has_inter_mesh_connections = intra_chip_neighbors == neighbors.end();
            if (is_TG && has_inter_mesh_connections) {
                // if active eth channels are found but no neighbor on the same mesh, then the neighbor should be the
                // gateway
                TT_FATAL(
                    active_eth_chans.size() == 1, "Found more than one active eth link b/w mmio and remote chip on TG");
            } else {
                TT_FATAL(!has_inter_mesh_connections, "1D routing does not support intermesh connections");
            }
        }

        FabricNodeId neighbor_fabric_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
        chip_neighbors.emplace(direction, neighbor_fabric_node_id);

        active_fabric_eth_channels.insert({direction, active_eth_chans});
        log_debug(
            tt::LogMetal,
            "Building fabric router -> device (phys): {}, (logical): {}, direction: {}, active_eth_chans: {}",
            device->id(),
            control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
            direction,
            active_eth_chans.size());
    }

    if (active_fabric_eth_channels.empty()) {
        // Need at least 1 active fabric eth channel in at least 1 direction with a neighbor
        return;
    }

    const bool wrap_around_mesh = fabric_context.is_wrap_around_mesh(fabric_node_id.mesh_id);

    for (const auto& [direction, remote_fabric_node_id] : chip_neighbors) {
        const auto& [fabric_edm_type, fabric_edm_axis] = get_fabric_edm_type(
            control_plane,
            topology,
            fabric_node_id.mesh_id,
            fabric_node_id.chip_id,
            remote_fabric_node_id.chip_id,
            wrap_around_mesh);

        bool is_dateline = remote_fabric_node_id.mesh_id == fabric_node_id.mesh_id &&
                           fabric_edm_type == tt::tt_fabric::FabricEriscDatamoverType::Dateline;

        const auto& curr_edm_config = fabric_context.get_fabric_router_config(fabric_edm_type, fabric_edm_axis);
        for (const auto& eth_chan : active_fabric_eth_channels[direction]) {
            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                device,
                *fabric_program_ptr,
                eth_logical_core,
                fabric_node_id,
                remote_fabric_node_id,
                curr_edm_config,
                false, /* build_in_worker_connection_mode */
                is_dateline,
                control_plane.routing_direction_to_eth_direction(direction));
            edm_builders.insert({eth_chan, edm_builder});
        }

        // Last link may be used by dispatch if there is tunneling
        // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
        if (!active_fabric_eth_channels[direction].empty() && device_has_dispatch_tunnel) {
            const auto dispatch_eth_chan = active_fabric_eth_channels[direction].back();
            configure_edm_builder_for_dispatch(edm_builders.at(dispatch_eth_chan));
        }
    }

    const bool is_galaxy =
        tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::GALAXY;

    auto connect_downstream_builders = [&](RoutingDirection dir1, RoutingDirection dir2) {
        bool can_connect =
            (chip_neighbors.find(dir1) != chip_neighbors.end()) && (chip_neighbors.find(dir2) != chip_neighbors.end());
        if (can_connect) {
            auto eth_chans_dir1 = active_fabric_eth_channels.at(dir1);
            auto eth_chans_dir2 = active_fabric_eth_channels.at(dir2);

            // Hack for TG to connect the last routing plane correctly for dispatch
            // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
            if (is_TG && (eth_chans_dir1.size() != eth_chans_dir2.size())) {
                log_trace(tt::LogMetal, "applying hack for chip: {}", device->id());
                std::reverse(eth_chans_dir1.begin(), eth_chans_dir1.end());
                std::reverse(eth_chans_dir2.begin(), eth_chans_dir2.end());
            }

            // since tunneling cores are not guaraneteed to be reserved on the same routing plane, iterate through
            // the ordered eth channels in both directions
            uint32_t num_links = std::min(eth_chans_dir1.size(), eth_chans_dir2.size());
            for (uint32_t link = 0; link < num_links; link++) {
                auto eth_chan_dir1 = eth_chans_dir1[link];
                auto eth_chan_dir2 = eth_chans_dir2[link];

                auto& edm_builder1 = edm_builders.at(eth_chan_dir1);
                auto& edm_builder2 = edm_builders.at(eth_chan_dir2);
                edm_builder1.connect_to_downstream_edm(edm_builder2);
                edm_builder2.connect_to_downstream_edm(edm_builder1);

                // select VC based on the current link
                auto edm_noc_vc = edm_builder1.config.DEFAULT_NOC_VC + (link % edm_builder1.config.NUM_EDM_NOC_VCS);
                edm_builder1.config.edm_noc_vc = edm_noc_vc;
                edm_builder2.config.edm_noc_vc = edm_noc_vc;

                if (is_galaxy) {
                    get_optimal_noc_for_edm(edm_builder1, edm_builder2, num_links, topology);
                }
            }
        }
    };

    if (is_2D_routing) {
        // 2D Routing
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::S);
        connect_downstream_builders(RoutingDirection::E, RoutingDirection::W);
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::E);
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::W);
        connect_downstream_builders(RoutingDirection::S, RoutingDirection::E);
        connect_downstream_builders(RoutingDirection::S, RoutingDirection::W);
    } else if (wrap_around_mesh && num_intra_chip_neighbors == 2) {
        // 1D Routing wrap the corner chips, fold the internal connections
        auto it = chip_neighbors.begin();
        auto dir1 = it->first;
        it++;
        auto dir2 = it->first;
        connect_downstream_builders(dir1, dir2);
    } else {
        // 1D Routing
        connect_downstream_builders(RoutingDirection::N, RoutingDirection::S);
        connect_downstream_builders(RoutingDirection::E, RoutingDirection::W);
    }

    return;
}

std::unique_ptr<Program> create_and_compile_tt_fabric_program(IDevice* device) {
    std::unique_ptr<Program> fabric_program_ptr = std::make_unique<Program>();
    std::unordered_map<tt::tt_fabric::chan_id_t, tt::tt_fabric::FabricEriscDatamoverBuilder> edm_builders;

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    build_tt_fabric_program(device, fabric_program_ptr.get(), edm_builders);
    fabric_context.set_num_fabric_initialized_routers(device->id(), edm_builders.size());
    if (edm_builders.empty()) {
        return nullptr;
    }

    // for now it doesnt matter which channel is the master, so just pick the 1st in the map
    auto master_router_chan = edm_builders.begin()->first;
    fabric_context.set_fabric_master_router_chan(device->id(), master_router_chan);

    uint32_t router_channels_mask = 0;
    for (const auto& [router_chan, _] : edm_builders) {
        router_channels_mask += 0x1 << (uint32_t)router_chan;
    }

    std::map<std::string, std::string> defines = {};
    const auto topology = fabric_context.get_fabric_topology();
    if (topology == tt::tt_fabric::Topology::Mesh) {
        defines["FABRIC_2D"] = "";
    }

    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto num_enabled_eth_cores = edm_builders.size();
    const auto num_enabled_risc_cores =
        edm_builders.begin()->second.get_configured_risc_count();  // same across all eth cores
    size_t num_local_fabric_routers = num_enabled_risc_cores * num_enabled_eth_cores;
    for (auto& [eth_chan, edm_builder] : edm_builders) {
        edm_builder.set_wait_for_host_signal(true);
        const std::vector<uint32_t> rt_args = edm_builder.get_runtime_args();
        for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
            std::vector<uint32_t> ct_args = edm_builder.get_compile_time_args(risc_id);

            const auto is_master_risc_core = eth_chan == master_router_chan && (risc_id == 0);
            ct_args.push_back(is_master_risc_core);
            ct_args.push_back(master_router_chan);
            ct_args.push_back(num_local_fabric_routers);
            ct_args.push_back(router_channels_mask);

            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto kernel = tt::tt_metal::CreateKernel(
                *fabric_program_ptr,
                "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_datamover.cpp",
                eth_logical_core,
                tt::tt_metal::EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0,
                    .processor = static_cast<DataMovementProcessor>(risc_id),
                    .compile_args = ct_args,
                    .defines = defines,
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

            tt::tt_metal::SetRuntimeArgs(*fabric_program_ptr, kernel, eth_logical_core, rt_args);
        }

        log_debug(
            tt::LogMetal,
            "Building fabric router -> device (phys): {}, (logical): {}, channel: {}, num_local_fabric_routers: {}",
            device->id(),
            control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
            eth_chan,
            num_local_fabric_routers);
    }

    detail::CompileProgram(device, *fabric_program_ptr, /*force_slow_dispatch=*/device->using_fast_dispatch());
    return fabric_program_ptr;
}

std::unique_ptr<Program> create_and_compile_fabric_program(IDevice* device) {
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        return create_and_compile_tt_fabric_program(device);
    }
    return nullptr;
}

void configure_fabric_cores(IDevice* device) {
    std::vector<uint32_t> router_zero_buf(1, 0);
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto addresses_to_clear = control_plane.get_fabric_context().get_fabric_router_addresses_to_clear();
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}

const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(chip_id_t dev_id) {
    if (!dispatch_cores.contains(dev_id)) {
        return empty_cores[dev_id];
    }
    return dispatch_cores[dev_id];
}

const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(chip_id_t dev_id) {
    if (!routing_cores.contains(dev_id)) {
        return empty_cores[dev_id];
    }
    return routing_cores[dev_id];
}

const std::unordered_set<TerminationInfo>& get_registered_termination_cores(chip_id_t dev_id) {
    if (!termination_info.contains(dev_id)) {
        termination_info[dev_id] = {};
    }
    return termination_info.at(dev_id);
}

void reset_topology_state() {
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24439
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        delete node_id_to_kernel[idx];
    }
    node_id_to_kernel.clear();
    command_queue_compile_group.clear();
    dispatch_cores.clear();
    routing_cores.clear();
    empty_cores.clear();
    termination_info.clear();
}

}  // namespace tt::tt_metal
