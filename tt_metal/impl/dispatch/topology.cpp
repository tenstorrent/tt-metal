// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology.hpp"

#include "device/device_manager.hpp"
#include <host_api.hpp>
#include <enchantum/enchantum.hpp>
#include <experimental/fabric/mesh_graph.hpp>
#include <tt_metal.hpp>
#include <cstdint>
#include <map>
#include <typeinfo>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <tt_stl/assert.hpp>
#include "command_queue_common.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "device.hpp"
#include "context/metal_context.hpp"
#include "dispatch_core_common.hpp"
#include "kernel_config/fd_kernel.hpp"
#include "program/program_impl.hpp"
#include "program.hpp"
#include <tt_stl/span.hpp>
#include <experimental/fabric/fabric.hpp>
#include "system_memory_manager.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "dispatch_mem_map.hpp"
#include <llrt/tt_cluster.hpp>
#include "dispatch_core_manager.hpp"

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
    .non_dispatch_noc = NOC::NOC_0,
    .upstream_noc = NOC::NOC_0,
    .downstream_noc = NOC::NOC_0,
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
    .non_dispatch_noc = NOC::NOC_0,
    .upstream_noc = NOC::NOC_1,
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
    .non_dispatch_noc = NOC::NOC_1,
    .upstream_noc = NOC::NOC_1,
    .downstream_noc = NOC::NOC_1,
};

// Must be on different NOCs because Dispatch+S may be running on the same
// core. They are using stateful APIs. Running on the same NOC will mess up
// requests sent/to free count.
static_assert(k_dispatcher_noc.non_dispatch_noc != k_dispatcher_s_noc.non_dispatch_noc);

//
// Fabric MUX NOC selections
//
// Must be NoC0
//
constexpr noc_selection_t k_fabric_mux_noc = {
    .non_dispatch_noc = NOC::NOC_0,
    .upstream_noc = NOC::NOC_0,
    .downstream_noc = NOC::NOC_0,
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
// clang-format on

std::vector<FDKernel*> node_id_to_kernel;
detail::ProgramCompileGroup command_queue_compile_group;
std::unordered_map<ChipId, std::unordered_set<CoreCoord>> dispatch_cores;
std::unordered_map<ChipId, std::unordered_set<CoreCoord>> routing_cores;
std::unordered_map<ChipId, std::unordered_set<CoreCoord>> empty_cores;
std::unordered_map<ChipId, std::unordered_set<TerminationInfo>> termination_info;

// Helper function to automatically generate dispatch nodes given devices + num hw CQs + detection of card type.
std::vector<DispatchKernelNode> generate_nodes(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs) {
    // Select/generate the right input table, depends on (1) board [detected from total # of devices], and (2) number
    // of active devices. TODO: read this out of YAML instead of the structs above?
    uint32_t total_devices = MetalContext::instance().get_cluster().number_of_devices();
    TT_ASSERT(
        total_devices == 1 or total_devices == 2 or total_devices == 4 or total_devices == 8 or total_devices == 32 or
            total_devices == 36,
        "Unexpected target.");
    uint32_t num_devices = device_ids.size();
    TT_ASSERT(num_devices > 0, "Can't determine dispatch architecture with no active devices.");
    TT_ASSERT(num_devices <= total_devices);
    std::vector<DispatchKernelNode> nodes;

    std::set<ChipId> mmio_devices;
    std::set<ChipId> remote_devices;
    for (auto id : device_ids) {
        if (MetalContext::instance().get_cluster().get_associated_mmio_device(id) == id) {
            mmio_devices.insert(id);
        } else {
            remote_devices.insert(id);
        }
    }

    // Helper function to get nodes for single device
    auto populate_single_device = [&]() {
        if (num_hw_cqs == 1) {
            return single_chip_arch_1cq;
        }  // TODO: determine whether dispatch_s is inserted at this level, instead of inside
           // Device::dispatch_s_enabled().
        if (MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type() == CoreType::WORKER) {
            return single_chip_arch_2cq_dispatch_s;
        }
        return single_chip_arch_2cq;
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
        if (MetalContext::instance().get_cluster().is_galaxy_cluster()) {
            // For Galaxy, we always init all remote devices associated with an mmio device.
            std::vector<DispatchKernelNode> nodes_for_one_mmio =
                (num_hw_cqs == 1) ? galaxy_nine_chip_arch_1cq_fabric : galaxy_nine_chip_arch_2cq_fabric;
            uint32_t index_offset = 0;
            for (auto mmio_device_id : mmio_devices) {
                // Need a mapping from templated device id (1-8) to actual device id (from the tunnel)
                std::vector<ChipId> template_id_to_device_id;
                template_id_to_device_id.push_back(mmio_device_id);
                for (const auto& tunnel :
                     MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
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
            std::vector<DispatchKernelNode> nodes_for_one_mmio =
                (num_hw_cqs == 1) ? two_chip_arch_1cq_fabric : two_chip_arch_2cq_fabric;

            uint32_t index_offset = 0;
            for (auto mmio_device_id : mmio_devices) {
                // Find the corresponding remote chip
                ChipId remote_device_id{};
                bool found_remote = false;
                for (auto id : remote_devices) {
                    if (MetalContext::instance().get_cluster().get_associated_mmio_device(id) == mmio_device_id) {
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
    std::set<ChipId> device_ids;
    for (const auto& device : devices) {
        device_ids.insert(device->id());
    }
    populate_fd_kernels(generate_nodes(device_ids, num_hw_cqs));
}

void populate_fd_kernels(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs) {
    populate_fd_kernels(generate_nodes(device_ids, num_hw_cqs));
}

void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes) {
    // If we already had nodes from a previous run, clear them (since we could have a different # of devices or CQs).
    if (!node_id_to_kernel.empty()) {
        for (auto& kernel : node_id_to_kernel) {
            delete kernel;
        }
        node_id_to_kernel.clear();
        command_queue_compile_group.clear();
    }

    // Read the input table, create configs for each node + track mmio devices and number of cqs.
    std::unordered_set<ChipId> mmio_device_ids;
    std::unordered_set<uint8_t> hw_cq_ids;
    node_id_to_kernel.reserve(nodes.size());
    for (const auto& node : nodes) {
        TT_ASSERT(node_id_to_kernel.size() == node.id);
        node_id_to_kernel.emplace_back(FDKernel::Generate(
            node.id,
            node.device_id,
            node.servicing_device_id,
            node.cq_id,
            node.noc_selection,
            node.kernel_type,
            node.tunnel_index));
        if (MetalContext::instance().get_cluster().get_associated_mmio_device(node.device_id) == node.device_id) {
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
    std::map<ChipId, uint32_t> device_id_to_tunnel_stop;
    std::map<ChipId, std::vector<ChipId>> mmio_device_id_to_serviced_devices;
    for (auto mmio_device_id : mmio_device_ids) {
        if (MetalContext::instance().get_cluster().get_associated_mmio_device(mmio_device_id) != mmio_device_id) {
            continue;
        }

        // Get a list of remote devices serviced by this mmio chip
        for (int idx = 0; idx < num_hw_cqs; idx++) {
            mmio_device_id_to_serviced_devices[mmio_device_id].push_back(mmio_device_id);
        }
        std::vector<ChipId> remote_devices;
        for (auto tunnel : MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
            for (uint32_t tunnel_stop = 0; tunnel_stop < tunnel.size(); tunnel_stop++) {
                ChipId remote_device_id = tunnel[tunnel_stop];
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
}

void populate_cq_static_args(IDevice* device) {
    TT_ASSERT(
        !node_id_to_kernel.empty(),
        "Tried to populate static args on nodes without the nodes populated (need to run populate_fd_kernels()");
    // First pass, add device/program to all kernels for this device and generate static configs.
    auto cq_program_ptr = std::make_unique<Program>();
    for (auto* node_and_kernel : node_id_to_kernel) {
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
    // Third pass, populate dependent configs, runtime configs, and create kernels for each node
    for (auto* node_and_kernel : node_id_to_kernel) {
        if (node_and_kernel->GetDeviceId() == device->id()) {
            node_and_kernel->GenerateDependentConfigs();
            node_and_kernel->InitializeRuntimeArgsValues();
            node_and_kernel->CreateKernel();
            node_and_kernel->SetRuntimeArgs();
        }
    }

    // Register core coordinates for this device
    for (auto* node_and_kernel : node_id_to_kernel) {
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
                    enchantum::to_string(node_and_kernel->GetKernelType()),
                    typeid(*node_and_kernel).name(),
                    device->id());
                break;
        }
    }

    // Register termination info
    for (auto* node_and_kernel : node_id_to_kernel) {
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
    command_queue_compile_group.compile_all(/*force_slow_dispatch=*/true);

    // Write runtime args to device
    command_queue_compile_group.write_runtime_args(/*force_slow_dispatch=*/true);
}

std::unique_ptr<Program> get_compiled_cq_program(IDevice* device) {
    return command_queue_compile_group.remove_program(device);
}

void configure_dispatch_cores(IDevice* device) {
    // Set up completion_queue_writer core. This doesn't actually have a kernel so keep it out of the struct and config
    // it here. TODO: should this be in the struct?
    CoreType dispatch_core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    const auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map();
    uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    uint32_t cq_size = device->sysmem_manager().get_cq_size();
    std::vector<uint32_t> zero = {0x0};

    // Need to set up for all devices serviced by an mmio chip
    if (device->is_mmio_capable()) {
        for (ChipId serviced_device_id :
             MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(device->id())) {
            uint16_t channel =
                MetalContext::instance().get_cluster().get_assigned_channel_for_device(serviced_device_id);
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
                tt_cxy_pair completion_q_writer_location =
                    MetalContext::instance().get_dispatch_core_manager().completion_queue_writer_core(
                        serviced_device_id, channel, cq_id);
                IDevice* mmio_device =
                    MetalContext::instance().device_manager()->get_active_device(completion_q_writer_location.chip);
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
    for (auto& kernel : node_id_to_kernel) {
        if (kernel->GetDeviceId() == device->id()) {
            kernel->ConfigureCore();
        }
    }
}

const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(ChipId dev_id) {
    if (!dispatch_cores.contains(dev_id)) {
        return empty_cores[dev_id];
    }
    return dispatch_cores[dev_id];
}

const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(ChipId dev_id) {
    if (!routing_cores.contains(dev_id)) {
        return empty_cores[dev_id];
    }
    return routing_cores[dev_id];
}

const std::unordered_set<TerminationInfo>& get_registered_termination_cores(ChipId dev_id) {
    if (!termination_info.contains(dev_id)) {
        termination_info[dev_id] = {};
    }
    return termination_info.at(dev_id);
}

void reset_topology_state() {
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24439
    for (auto& kernel : node_id_to_kernel) {
        delete kernel;
    }
    node_id_to_kernel.clear();
    command_queue_compile_group.clear();
    dispatch_cores.clear();
    routing_cores.clear();
    empty_cores.clear();
    termination_info.clear();
}

}  // namespace tt::tt_metal
