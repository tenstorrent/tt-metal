// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topology.hpp"
#include "kernel_config/fd_kernel.hpp"
#include <device_pool.hpp>
#include <tt_metal.hpp>
#include <host_api.hpp>
#include "kernel_config/fd_kernel.hpp"
#include "kernel_config/prefetch.hpp"
#include "kernel_config/dispatch.hpp"
#include "kernel_config/dispatch_s.hpp"
#include "kernel_config/mux.hpp"
#include "kernel_config/demux.hpp"
#include "kernel_config/eth_router.hpp"
#include "kernel_config/eth_tunneler.hpp"
#include "fabric_host_interface.h"

#include "tt_cluster.hpp"

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

static const std::vector<DispatchKernelNode> single_chip_arch_1cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
};

static const std::vector<DispatchKernelNode> single_chip_arch_2cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {x, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
};

static const std::vector<DispatchKernelNode> single_chip_arch_2cq_dispatch_s = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 4, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {4, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {2, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, 5, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {3, 0, 0, 1, DISPATCH_HD, {2, x, x, x}, {5, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {4, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {5, 0, 0, 1, DISPATCH_S, {2, x, x, x}, {3, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
};

static const std::vector<DispatchKernelNode> two_chip_arch_1cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    {3, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {5, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {4, 0, 1, 0, DISPATCH_H, {6, x, x, x}, {3, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},

    {5, 0, 1, 0, PACKET_ROUTER_MUX, {3, x, x, x}, {7, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {6, 0, 1, 0, DEMUX, {7, x, x, x}, {4, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {7, 0, 1, 0, US_TUNNELER_REMOTE, {11, 5, x, x}, {11, 6, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    {8, 1, x, 0, PREFETCH_D, {13, x, x, x}, {9, 10, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {9, 1, x, 0, DISPATCH_D, {8, x, x, x}, {10, 12, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {10, 1, x, 0, DISPATCH_S, {8, x, x, x}, {9, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    {11, 1, x, 0, US_TUNNELER_LOCAL, {7, 12, x, x}, {7, 13, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {12, 1, x, 0, MUX_D, {9, x, x, x}, {11, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {13, 1, x, 0, PACKET_ROUTER_DEMUX, {11, x, x, x}, {8, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
};

static const std::vector<DispatchKernelNode> two_chip_arch_2cq = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {x, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},

    {4, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {8, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {5, 0, 1, 1, PREFETCH_H, {x, x, x, x}, {8, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {6, 0, 1, 0, DISPATCH_H, {9, x, x, x}, {4, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {7, 0, 1, 1, DISPATCH_H, {9, x, x, x}, {5, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},

    {8, 0, 1, 0, PACKET_ROUTER_MUX, {4, 5, x, x}, {10, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {9, 0, 1, 0, DEMUX, {10, x, x, x}, {6, 7, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {10, 0, 1, 0, US_TUNNELER_REMOTE, {15, 8, x, x}, {15, 9, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    {11, 1, x, 0, PREFETCH_D, {17, x, x, x}, {13, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {12, 1, x, 1, PREFETCH_D, {17, x, x, x}, {14, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {13, 1, x, 0, DISPATCH_D, {11, x, x, x}, {16, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {14, 1, x, 1, DISPATCH_D, {12, x, x, x}, {16, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},

    {15, 1, x, 0, US_TUNNELER_LOCAL, {10, 16, x, x}, {10, 17, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {16, 1, x, 0, MUX_D, {13, 14, x, x}, {15, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {17, 1, x, 0, PACKET_ROUTER_DEMUX, {15, x, x, x}, {11, 12, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

};

static const std::vector<DispatchKernelNode> galaxy_nine_chip_arch_1cq = {
    // For MMIO chip, TODO: investigate removing these, they aren't needed
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, 2, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {2, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {2, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {1, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    // Sevicing remote chips 1-4
    {3, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {4, 0, 1, 0, DISPATCH_H, {13, x, x, x}, {3, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {5, 0, 2, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {6, 0, 2, 0, DISPATCH_H, {13, x, x, x}, {5, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {7, 0, 3, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {8, 0, 3, 0, DISPATCH_H, {14, x, x, x}, {7, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {9, 0, 4, 0, PREFETCH_H, {x, x, x, x}, {11, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {10, 0, 4, 0, DISPATCH_H, {14, x, x, x}, {9, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {11, 0, 1, 0, PACKET_ROUTER_MUX, {3, 5, 7, 9}, {15, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {12, 0, 1, 0, DEMUX, {15, x, x, x}, {13, 14, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {13, 0, 1, 0, DEMUX, {12, x, x, x}, {4, 6, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {14, 0, 1, 0, DEMUX, {12, x, x, x}, {8, 10, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {15, 0, 1, 0, US_TUNNELER_REMOTE, {29, 11, x, x}, {29, 12, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Servicing remote chips 5-8
    {16, 0, 5, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {17, 0, 5, 0, DISPATCH_H, {26, x, x, x}, {16, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {18, 0, 6, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {19, 0, 6, 0, DISPATCH_H, {26, x, x, x}, {18, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {20, 0, 7, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {21, 0, 7, 0, DISPATCH_H, {27, x, x, x}, {20, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {22, 0, 8, 0, PREFETCH_H, {x, x, x, x}, {24, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {23, 0, 8, 0, DISPATCH_H, {27, x, x, x}, {22, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {24, 0, 5, 0, PACKET_ROUTER_MUX, {16, 18, 20, 22}, {28, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {25, 0, 5, 0, DEMUX, {28, x, x, x}, {26, 27, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {26, 0, 5, 0, DEMUX, {25, x, x, x}, {17, 19, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {27, 0, 5, 0, DEMUX, {25, x, x, x}, {21, 23, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {28, 0, 5, 0, US_TUNNELER_REMOTE, {59, 24, x, x}, {59, 25, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 1
    {29, 1, x, 0, US_TUNNELER_LOCAL, {15, 30, x, x}, {15, 31, 32, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {30, 1, x, 0, MUX_D, {34, 36, x, x}, {29, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {31, 1, x, 0, PACKET_ROUTER_DEMUX, {29, x, x, x}, {33, 36, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {32, 1, x, 0, PACKET_ROUTER_DEMUX, {29, x, x, x}, {36, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {33, 1, x, 0, PREFETCH_D, {31, x, x, x}, {34, 35, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {34, 1, x, 0, DISPATCH_D, {33, x, x, x}, {35, 30, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {35, 1, x, 0, DISPATCH_S, {33, x, x, x}, {34, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {36, 1, x, 0, US_TUNNELER_REMOTE, {37, 31, 32, x}, {37, 30, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 2
    {37, 2, x, 0, US_TUNNELER_LOCAL, {36, 38, x, x}, {36, 39, 40, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {38, 2, x, 0, MUX_D, {42, 44, x, x}, {37, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {39, 2, x, 0, PACKET_ROUTER_DEMUX, {37, x, x, x}, {41, 44, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {40, 2, x, 0, PACKET_ROUTER_DEMUX, {37, x, x, x}, {44, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {41, 2, x, 0, PREFETCH_D, {39, x, x, x}, {42, 43, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {42, 2, x, 0, DISPATCH_D, {41, x, x, x}, {43, 38, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {43, 2, x, 0, DISPATCH_S, {41, x, x, x}, {42, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {44, 2, x, 0, US_TUNNELER_REMOTE, {45, 39, 40, x}, {45, 38, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 3
    {45, 3, x, 0, US_TUNNELER_LOCAL, {44, 46, x, x}, {44, 47, 48, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {46, 3, x, 0, MUX_D, {50, 52, x, x}, {45, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {47, 3, x, 0, PACKET_ROUTER_DEMUX, {45, x, x, x}, {49, 52, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {48, 3, x, 0, PACKET_ROUTER_DEMUX, {45, x, x, x}, {52, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {49, 3, x, 0, PREFETCH_D, {47, x, x, x}, {50, 51, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {50, 3, x, 0, DISPATCH_D, {49, x, x, x}, {51, 46, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {51, 3, x, 0, DISPATCH_S, {49, x, x, x}, {50, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {52, 3, x, 0, US_TUNNELER_REMOTE, {53, 47, 48, x}, {53, 46, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 4
    {53, 4, x, 0, US_TUNNELER_LOCAL, {52, 54, x, x}, {52, 55, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {54, 4, x, 0, MUX_D, {57, x, x, x}, {53, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {55, 4, x, 0, PACKET_ROUTER_DEMUX, {53, x, x, x}, {56, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {56, 4, x, 0, PREFETCH_D, {55, x, x, x}, {57, 58, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {57, 4, x, 0, DISPATCH_D, {56, x, x, x}, {58, 54, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {58, 4, x, 0, DISPATCH_S, {56, x, x, x}, {57, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    // Remote chip 5
    {59, 5, x, 0, US_TUNNELER_LOCAL, {28, 60, x, x}, {28, 61, 62, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {60, 5, x, 0, MUX_D, {64, 66, x, x}, {59, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {61, 5, x, 0, PACKET_ROUTER_DEMUX, {59, x, x, x}, {63, 66, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {62, 5, x, 0, PACKET_ROUTER_DEMUX, {59, x, x, x}, {66, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {63, 5, x, 0, PREFETCH_D, {61, x, x, x}, {64, 65, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {64, 5, x, 0, DISPATCH_D, {63, x, x, x}, {65, 60, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {65, 5, x, 0, DISPATCH_S, {63, x, x, x}, {64, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {66, 5, x, 0, US_TUNNELER_REMOTE, {67, 61, 62, x}, {67, 60, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 6
    {67, 6, x, 0, US_TUNNELER_LOCAL, {66, 68, x, x}, {66, 69, 70, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {68, 6, x, 0, MUX_D, {72, 74, x, x}, {67, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {69, 6, x, 0, PACKET_ROUTER_DEMUX, {67, x, x, x}, {71, 74, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {70, 6, x, 0, PACKET_ROUTER_DEMUX, {67, x, x, x}, {74, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {71, 6, x, 0, PREFETCH_D, {69, x, x, x}, {72, 73, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {72, 6, x, 0, DISPATCH_D, {71, x, x, x}, {73, 68, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {73, 6, x, 0, DISPATCH_S, {71, x, x, x}, {72, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {74, 6, x, 0, US_TUNNELER_REMOTE, {75, 69, 70, x}, {75, 68, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 7
    {75, 7, x, 0, US_TUNNELER_LOCAL, {74, 76, x, x}, {74, 77, 78, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {76, 7, x, 0, MUX_D, {80, 82, x, x}, {75, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {77, 7, x, 0, PACKET_ROUTER_DEMUX, {75, x, x, x}, {79, 82, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {78, 7, x, 0, PACKET_ROUTER_DEMUX, {75, x, x, x}, {82, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {79, 7, x, 0, PREFETCH_D, {77, x, x, x}, {80, 81, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {80, 7, x, 0, DISPATCH_D, {79, x, x, x}, {81, 76, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {81, 7, x, 0, DISPATCH_S, {79, x, x, x}, {80, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {82, 7, x, 0, US_TUNNELER_REMOTE, {83, 77, 78, x}, {83, 76, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 8
    {83, 8, x, 0, US_TUNNELER_LOCAL, {82, 84, x, x}, {82, 85, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {84, 8, x, 0, MUX_D, {87, x, x, x}, {83, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {85, 8, x, 0, PACKET_ROUTER_DEMUX, {83, x, x, x}, {86, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {86, 8, x, 0, PREFETCH_D, {85, x, x, x}, {87, 88, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {87, 8, x, 0, DISPATCH_D, {86, x, x, x}, {88, 84, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {88, 8, x, 0, DISPATCH_S, {86, x, x, x}, {87, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
};

static const std::vector<DispatchKernelNode> galaxy_nine_chip_arch_2cq = {
    // For MMIO chip
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {2, 4, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 1, PREFETCH_HD, {x, x, x, x}, {3, 5, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {2, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {4, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {3, 0, 0, 1, DISPATCH_HD, {1, x, x, x}, {5, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {4, 0, 0, 0, DISPATCH_S, {0, x, x, x}, {2, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {5, 0, 0, 1, DISPATCH_S, {1, x, x, x}, {3, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    // Servicing remote chips 1-4
    {6, 0, 1, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {7, 0, 1, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {8, 0, 1, 0, DISPATCH_H, {25, x, x, x}, {6, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {9, 0, 1, 1, DISPATCH_H, {25, x, x, x}, {7, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {10, 0, 2, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {11, 0, 2, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {12, 0, 2, 0, DISPATCH_H, {25, x, x, x}, {10, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {13, 0, 2, 1, DISPATCH_H, {25, x, x, x}, {11, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {14, 0, 3, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {15, 0, 3, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {16, 0, 3, 0, DISPATCH_H, {26, x, x, x}, {14, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {17, 0, 3, 1, DISPATCH_H, {26, x, x, x}, {15, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {18, 0, 4, 0, PREFETCH_H, {x, x, x, x}, {22, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {19, 0, 4, 1, PREFETCH_H, {x, x, x, x}, {23, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {20, 0, 4, 0, DISPATCH_H, {26, x, x, x}, {18, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {21, 0, 4, 1, DISPATCH_H, {26, x, x, x}, {19, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {22, 0, 1, 0, PACKET_ROUTER_MUX, {6, 10, 14, 18}, {27, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {23, 0, 1, 0, PACKET_ROUTER_MUX, {7, 11, 15, 19}, {27, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {24, 0, 1, 0, DEMUX, {27, x, x, x}, {25, 26, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {25, 0, 1, 0, DEMUX, {24, x, x, x}, {8, 9, 12, 13}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {26, 0, 1, 0, DEMUX, {24, x, x, x}, {16, 17, 20, 21}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {27, 0, 1, 0, US_TUNNELER_REMOTE, {50, 22, 23, x}, {50, 24, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Servicing remote chips 5-8
    {28, 0, 5, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {29, 0, 5, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {30, 0, 5, 0, DISPATCH_H, {47, x, x, x}, {28, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {31, 0, 5, 1, DISPATCH_H, {47, x, x, x}, {29, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {32, 0, 6, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {33, 0, 6, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {34, 0, 6, 0, DISPATCH_H, {47, x, x, x}, {32, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {35, 0, 6, 1, DISPATCH_H, {47, x, x, x}, {33, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {36, 0, 7, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {37, 0, 7, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {38, 0, 7, 0, DISPATCH_H, {48, x, x, x}, {36, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {39, 0, 7, 1, DISPATCH_H, {48, x, x, x}, {37, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {40, 0, 8, 0, PREFETCH_H, {x, x, x, x}, {44, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {41, 0, 8, 1, PREFETCH_H, {x, x, x, x}, {45, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {42, 0, 8, 0, DISPATCH_H, {48, x, x, x}, {40, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {43, 0, 8, 1, DISPATCH_H, {48, x, x, x}, {41, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {44, 0, 5, 0, PACKET_ROUTER_MUX, {28, 32, 36, 40}, {49, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {45, 0, 5, 0, PACKET_ROUTER_MUX, {29, 33, 37, 41}, {49, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {46, 0, 5, 0, DEMUX, {49, x, x, x}, {47, 48, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {47, 0, 5, 0, DEMUX, {46, x, x, x}, {30, 31, 34, 35}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {48, 0, 5, 0, DEMUX, {46, x, x, x}, {38, 39, 42, 43}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {49, 0, 5, 0, US_TUNNELER_REMOTE, {93, 44, 45, x}, {93, 46, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 1
    {50, 1, x, 0, US_TUNNELER_LOCAL, {27, 51, x, x}, {27, 52, 53, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {51, 1, x, 0, MUX_D, {56, 57, 60, x}, {50, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {52, 1, x, 0, PACKET_ROUTER_DEMUX, {50, x, x, x}, {54, 60, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {53, 1, x, 0, PACKET_ROUTER_DEMUX, {50, x, x, x}, {55, 60, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {54, 1, x, 0, PREFETCH_D, {52, x, x, x}, {56, 58, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {55, 1, x, 1, PREFETCH_D, {53, x, x, x}, {57, 59, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {56, 1, x, 0, DISPATCH_D, {54, x, x, x}, {58, 51, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {57, 1, x, 1, DISPATCH_D, {55, x, x, x}, {59, 51, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {58, 1, x, 0, DISPATCH_S, {54, x, x, x}, {56, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    // TODO: Why does the second dispatch S connect to the first dispatch D? Keep same as previous implementation for
    // now
    {59, 1, x, 1, DISPATCH_S, {54, x, x, x}, {56, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {60, 1, x, 0, US_TUNNELER_REMOTE, {61, 52, 53, x}, {61, 51, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 2
    {61, 2, x, 0, US_TUNNELER_LOCAL, {60, 62, x, x}, {60, 63, 64, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {62, 2, x, 0, MUX_D, {67, 68, 71, x}, {61, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {63, 2, x, 0, PACKET_ROUTER_DEMUX, {61, x, x, x}, {65, 71, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {64, 2, x, 0, PACKET_ROUTER_DEMUX, {61, x, x, x}, {66, 71, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {65, 2, x, 0, PREFETCH_D, {63, x, x, x}, {67, 69, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {66, 2, x, 1, PREFETCH_D, {64, x, x, x}, {68, 70, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {67, 2, x, 0, DISPATCH_D, {65, x, x, x}, {69, 62, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {68, 2, x, 1, DISPATCH_D, {66, x, x, x}, {70, 62, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {69, 2, x, 0, DISPATCH_S, {65, x, x, x}, {67, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {70, 2, x, 1, DISPATCH_S, {65, x, x, x}, {67, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {71, 2, x, 0, US_TUNNELER_REMOTE, {72, 63, 64, x}, {72, 62, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 3
    {72, 3, x, 0, US_TUNNELER_LOCAL, {71, 73, x, x}, {71, 74, 75, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {73, 3, x, 0, MUX_D, {78, 79, 82, x}, {72, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {74, 3, x, 0, PACKET_ROUTER_DEMUX, {72, x, x, x}, {76, 82, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {75, 3, x, 0, PACKET_ROUTER_DEMUX, {72, x, x, x}, {77, 82, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {76, 3, x, 0, PREFETCH_D, {74, x, x, x}, {78, 80, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {77, 3, x, 1, PREFETCH_D, {75, x, x, x}, {79, 81, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {78, 3, x, 0, DISPATCH_D, {76, x, x, x}, {80, 73, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {79, 3, x, 1, DISPATCH_D, {77, x, x, x}, {81, 73, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {80, 3, x, 0, DISPATCH_S, {76, x, x, x}, {78, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {81, 3, x, 1, DISPATCH_S, {76, x, x, x}, {78, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {82, 3, x, 0, US_TUNNELER_REMOTE, {83, 74, 75, x}, {83, 73, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 4
    {83, 4, x, 0, US_TUNNELER_LOCAL, {82, 84, x, x}, {82, 85, 86, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {84, 4, x, 0, MUX_D, {89, 90, x, x}, {83, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {85, 4, x, 0, PACKET_ROUTER_DEMUX, {83, x, x, x}, {87, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {86, 4, x, 0, PACKET_ROUTER_DEMUX, {83, x, x, x}, {88, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {87, 4, x, 0, PREFETCH_D, {85, x, x, x}, {89, 91, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {88, 4, x, 1, PREFETCH_D, {86, x, x, x}, {90, 92, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {89, 4, x, 0, DISPATCH_D, {87, x, x, x}, {91, 84, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {90, 4, x, 1, DISPATCH_D, {88, x, x, x}, {92, 84, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {91, 4, x, 0, DISPATCH_S, {87, x, x, x}, {89, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {92, 4, x, 1, DISPATCH_S, {87, x, x, x}, {89, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},

    // Remote chip 5
    {93, 5, x, 0, US_TUNNELER_LOCAL, {49, 94, x, x}, {49, 95, 96, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {94, 5, x, 0, MUX_D, {99, 100, 103, x}, {93, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {95, 5, x, 0, PACKET_ROUTER_DEMUX, {93, x, x, x}, {97, 103, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {96, 5, x, 0, PACKET_ROUTER_DEMUX, {93, x, x, x}, {98, 103, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {97, 5, x, 0, PREFETCH_D, {95, x, x, x}, {99, 101, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {98, 5, x, 1, PREFETCH_D, {96, x, x, x}, {100, 102, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {99, 5, x, 0, DISPATCH_D, {97, x, x, x}, {101, 94, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {100, 5, x, 1, DISPATCH_D, {98, x, x, x}, {102, 94, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {101, 5, x, 0, DISPATCH_S, {97, x, x, x}, {99, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {102, 5, x, 1, DISPATCH_S, {97, x, x, x}, {99, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {103, 5, x, 0, US_TUNNELER_REMOTE, {104, 95, 96, x}, {104, 94, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 6
    {104, 6, x, 0, US_TUNNELER_LOCAL, {103, 105, x, x}, {103, 106, 107, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {105, 6, x, 0, MUX_D, {110, 111, 114, x}, {104, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {106, 6, x, 0, PACKET_ROUTER_DEMUX, {104, x, x, x}, {108, 114, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {107, 6, x, 0, PACKET_ROUTER_DEMUX, {104, x, x, x}, {109, 114, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {108, 6, x, 0, PREFETCH_D, {106, x, x, x}, {110, 112, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {109, 6, x, 1, PREFETCH_D, {107, x, x, x}, {111, 113, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {110, 6, x, 0, DISPATCH_D, {108, x, x, x}, {112, 105, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {111, 6, x, 1, DISPATCH_D, {109, x, x, x}, {113, 105, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {112, 6, x, 0, DISPATCH_S, {108, x, x, x}, {110, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {113, 6, x, 1, DISPATCH_S, {108, x, x, x}, {110, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {114, 6, x, 0, US_TUNNELER_REMOTE, {115, 106, 107, x}, {115, 105, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 7
    {115, 7, x, 0, US_TUNNELER_LOCAL, {114, 116, x, x}, {114, 117, 118, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {116, 7, x, 0, MUX_D, {121, 122, 125, x}, {115, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {117, 7, x, 0, PACKET_ROUTER_DEMUX, {115, x, x, x}, {119, 125, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {118, 7, x, 0, PACKET_ROUTER_DEMUX, {115, x, x, x}, {120, 125, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {119, 7, x, 0, PREFETCH_D, {117, x, x, x}, {121, 123, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {120, 7, x, 1, PREFETCH_D, {118, x, x, x}, {122, 124, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {121, 7, x, 0, DISPATCH_D, {119, x, x, x}, {123, 116, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {122, 7, x, 1, DISPATCH_D, {120, x, x, x}, {124, 116, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {123, 7, x, 0, DISPATCH_S, {119, x, x, x}, {121, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {124, 7, x, 1, DISPATCH_S, {119, x, x, x}, {121, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {125, 7, x, 0, US_TUNNELER_REMOTE, {126, 117, 118, x}, {126, 116, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},

    // Remote chip 8
    {126, 8, x, 0, US_TUNNELER_LOCAL, {125, 127, x, x}, {125, 128, 129, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {127, 8, x, 0, MUX_D, {132, 133, x, x}, {126, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {128, 8, x, 0, PACKET_ROUTER_DEMUX, {126, x, x, x}, {130, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {129, 8, x, 0, PACKET_ROUTER_DEMUX, {126, x, x, x}, {131, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {130, 8, x, 0, PREFETCH_D, {128, x, x, x}, {132, 134, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {131, 8, x, 1, PREFETCH_D, {129, x, x, x}, {133, 135, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {132, 8, x, 0, DISPATCH_D, {130, x, x, x}, {134, 127, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {133, 8, x, 1, DISPATCH_D, {131, x, x, x}, {135, 127, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
    {134, 8, x, 0, DISPATCH_S, {130, x, x, x}, {132, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
    {135, 8, x, 1, DISPATCH_S, {130, x, x, x}, {132, x, x, x}, NOC::NOC_1, NOC::NOC_1, NOC::NOC_1},
};

std::vector<FDKernel*> node_id_to_kernel;

// Helper function to automatically generate dispatch nodes given devices + num hw CQs + detection of card type.
std::vector<DispatchKernelNode> generate_nodes(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs) {
    // Select/generate the right input table, depends on (1) board [detected from total # of devices], and (2) number
    // of active devices. TODO: read this out of YAML instead of the structs above?
    uint32_t total_devices = tt::Cluster::instance().number_of_devices();
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
        if (tt::Cluster::instance().get_associated_mmio_device(id) == id) {
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
            if (dispatch_core_manager::instance().get_dispatch_core_type(0) == CoreType::WORKER) {
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
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            // For Galaxy, we always init all remote devices associated with an mmio device.
            const std::vector<DispatchKernelNode>* nodes_for_one_mmio =
                (num_hw_cqs == 1) ? &galaxy_nine_chip_arch_1cq : &galaxy_nine_chip_arch_2cq;
            uint32_t index_offset = 0;
            for (auto mmio_device_id : mmio_devices) {
                // Need a mapping from templated device id (1-8) to actual device id (from the tunnel)
                std::vector<chip_id_t> template_id_to_device_id;
                template_id_to_device_id.push_back(mmio_device_id);
                for (const auto& tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
                    TT_ASSERT(tunnel.size() == 5, "Galaxy expected 4-deep tunnels.");
                    for (auto remote_device_id : tunnel) {
                        if (remote_device_id != mmio_device_id) {
                            template_id_to_device_id.push_back(remote_device_id);
                        }
                    }
                }

                // Pull nodes from the template, updating their index and device id
                for (DispatchKernelNode node : *nodes_for_one_mmio) {
                    node.device_id = template_id_to_device_id[node.device_id];
                    node.servicing_device_id = template_id_to_device_id[node.servicing_device_id];
                    increment_node_ids(node, index_offset);
                    nodes.push_back(node);
                }
                index_offset += nodes_for_one_mmio->size();
            }
        } else {
            // Should be paired mmio/remote devices
            TT_ASSERT(
                mmio_devices.size() == remote_devices.size() or remote_devices.empty(),
                "N300/T3K expects devices in mmio/remote pairs.");
            const std::vector<DispatchKernelNode>* nodes_for_one_mmio =
                (num_hw_cqs == 1) ? &two_chip_arch_1cq : &two_chip_arch_2cq;
            uint32_t index_offset = 0;
            for (auto mmio_device_id : mmio_devices) {
                // Find the corresponding remote chip
                chip_id_t remote_device_id;
                bool found_remote = false;
                for (auto id : remote_devices) {
                    if (tt::Cluster::instance().get_associated_mmio_device(id) == mmio_device_id) {
                        remote_device_id = id;
                        found_remote = true;
                        break;
                    }
                }
                TT_ASSERT(found_remote, "Couldn't find paired remote chip for device {}", mmio_device_id);

                // Add dispatch kernels for the mmio/remote pair
                for (DispatchKernelNode node : *nodes_for_one_mmio) {
                    TT_ASSERT(node.device_id == 0 || node.device_id == 1);
                    if (node.device_id == 0) {
                        node.device_id = mmio_device_id;
                        if (node.servicing_device_id == 0) {
                            node.servicing_device_id = mmio_device_id;
                        } else if (node.servicing_device_id == 1) {
                            node.servicing_device_id = remote_device_id;
                        }
                    } else {
                        node.device_id = remote_device_id;
                    }
                    increment_node_ids(node, index_offset);
                    nodes.push_back(node);
                }
                index_offset += nodes_for_one_mmio->size();
            }
        }
    }

    return nodes;
}

// Populate node_id_to_kernel and set up kernel objects. Do this once at the beginning since they (1) don't need a valid
// Device until fields are populated, (2) need to be connected to kernel objects for devices that aren't created yet,
// and (3) the table to choose depends on total number of devices, not know at Device creation.
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
            {node.my_noc, node.upstream_noc, node.downstream_noc},
            node.kernel_type));
        if (tt::Cluster::instance().get_associated_mmio_device(node.device_id) == node.device_id) {
            mmio_device_ids.insert(node.device_id);
        }
        hw_cq_ids.insert(node.cq_id);
    }
    uint32_t num_hw_cqs = hw_cq_ids.size();

    // Connect the graph with upstream/downstream kernels
    for (const auto& node : nodes) {
        for (int idx = 0; idx < DISPATCH_MAX_UPSTREAM_KERNELS; idx++) {
            if (node.upstream_ids[idx] >= 0) {
                node_id_to_kernel.at(node.id)->AddUpstreamKernel(node_id_to_kernel.at(node.upstream_ids[idx]));
            }
        }
        for (int idx = 0; idx < DISPATCH_MAX_DOWNSTREAM_KERNELS; idx++) {
            if (node.downstream_ids[idx] >= 0) {
                node_id_to_kernel.at(node.id)->AddDownstreamKernel(node_id_to_kernel.at(node.downstream_ids[idx]));
            }
        }
    }

    // For kernels on mmio chip, need to confirm which remote device each is servicing
    std::map<chip_id_t, uint32_t> device_id_to_tunnel_stop;
    std::map<chip_id_t, std::vector<chip_id_t>> mmio_device_id_to_serviced_devices;
    uint32_t tunnel_depth;
    for (auto mmio_device_id : mmio_device_ids) {
        if (tt::Cluster::instance().get_associated_mmio_device(mmio_device_id) != mmio_device_id) {
            continue;
        }

        // Get a list of remote devices serviced by this mmio chip
        for (int idx = 0; idx < num_hw_cqs; idx++) {
            mmio_device_id_to_serviced_devices[mmio_device_id].push_back(mmio_device_id);
        }
        std::vector<chip_id_t> remote_devices;
        for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
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
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(fd_kernel->GetDeviceId());
        if (fd_kernel->GetDeviceId() == mmio_device_id) {
            mmio_device_id_to_kernels[mmio_device_id].push_back(fd_kernel);
        }
    }
    for (auto& mmio_device_id_and_kernels : mmio_device_id_to_kernels) {
        chip_id_t mmio_device_id = mmio_device_id_and_kernels.first;
        int prefetch_h_id = 0, dispatch_h_id = 0;
        int demux_id = 0, router_id = 0, tunneler_id = 0;
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
            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(router_device_id);

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
            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(router_kernel->GetDeviceId());
            if (router_kernel->GetDeviceId() == mmio_device_id) {
                uint32_t num_tunnels = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id).size();
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

std::unique_ptr<Program> create_and_compile_cq_program(IDevice* device) {
    TT_ASSERT(
        node_id_to_kernel.size() > 0,
        "Tried to create CQ program without nodes populated (need to run populate_fd_kernels()");

    // First pass, add device/program to all kernels for this device and generate static configs.
    auto cq_program_ptr = std::make_unique<Program>();
    // for (auto &node_and_kernel : node_id_to_kernel) {
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        if (node_id_to_kernel[idx]->GetDeviceId() == device->id()) {
            node_id_to_kernel[idx]->AddDeviceAndProgram(device, cq_program_ptr.get());
            node_id_to_kernel[idx]->GenerateStaticConfigs();
        }
    }

    // Third pass, populate dependent configs and create kernels for each node
    // for (auto &node_and_kernel : node_id_to_kernel) {
    for (int idx = 0; idx < node_id_to_kernel.size(); idx++) {
        if (node_id_to_kernel[idx]->GetDeviceId() == device->id()) {
            node_id_to_kernel[idx]->GenerateDependentConfigs();
            node_id_to_kernel[idx]->CreateKernel();
        }
    }

    // Compile the program and return it so Device can register it
    detail::CompileProgram(device, *cq_program_ptr, /*fd_bootloader_mode=*/true);
    return cq_program_ptr;
}

void configure_dispatch_cores(IDevice* device) {
    // Set up completion_queue_writer core. This doesn't actually have a kernel so keep it out of the struct and config
    // it here. TODO: should this be in the struct?
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    auto& my_dispatch_constants = DispatchMemMap::get(dispatch_core_type);
    uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    uint32_t cq_size = device->sysmem_manager().get_cq_size();
    std::vector<uint32_t> zero = {0x0};

    // Need to set up for all devices serviced by an mmio chip
    if (device->is_mmio_capable()) {
        for (chip_id_t serviced_device_id :
             tt::Cluster::instance().get_devices_controlled_by_mmio_device(device->id())) {
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(serviced_device_id);
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
                tt_cxy_pair completion_q_writer_location =
                    dispatch_core_manager::instance().completion_queue_writer_core(serviced_device_id, channel, cq_id);
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

std::unique_ptr<Program> create_and_compile_fabric_program(IDevice* device) {
    auto fabric_program_ptr = std::make_unique<Program>();

    std::uint32_t router_mask = 0;
    auto router_chans = tt::Cluster::instance().get_fabric_ethernet_channels(device->id());
    auto fabric_config = tt::Cluster::instance().get_fabric_config();
    size_t num_routers = router_chans.size();
    for (const auto& router_chan : router_chans) {
        router_mask += 0x1 << (uint32_t)router_chan;
    }
    auto master_router_chan = (uint32_t)(*router_chans.begin());
    // setup runtime args
    std::vector<uint32_t> router_runtime_args = {
        num_routers,         // 0: number of active fabric routers
        router_mask,         // 1: active fabric router mask
        master_router_chan,  // 2: master router channel
    };

    std::vector<uint32_t> router_compile_args = {
        (tt::tt_fabric::DEFAULT_ROUTER_RX_QUEUE_SIZE_BYTES >> 4),  // 0: rx_queue_size_words
        0,                                                         // 1: test_results_addr
        0,                                                         // 2: test_results_size
        0,  // 3: timeout_mcycles * 1000 * 1000 * 4, // 3: timeout_cycles
        0,  // 4: is_master_router
    };

    std::map<string, string> router_defines = {};
    if (fabric_config == FabricConfig::FABRIC_2D) {
        router_defines["FVC_MODE_PULL"] = "";
    }

    // TODO: Manual clear of semaphore, move this to proper Metal sempahore apis
    std::vector<uint32_t> fabric_sem_zero_buf(1, 0);

    for (const auto& router_chan : router_chans) {
        CoreCoord virtual_eth_core =
            tt::Cluster::instance().get_virtual_eth_core_from_channel(device->id(), router_chan);
        auto router_logical_core = device->logical_core_from_ethernet_core(virtual_eth_core);
        if (master_router_chan == router_chan) {
            router_compile_args[4] = 1;
        } else {
            router_compile_args[4] = 0;
        }
        auto kernel = tt_metal::CreateKernel(
            *fabric_program_ptr,
            "tt_metal/fabric/impl/kernels/tt_fabric_router.cpp",
            router_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0, .compile_args = router_compile_args, .defines = router_defines});

        tt_metal::SetRuntimeArgs(*fabric_program_ptr, kernel, router_logical_core, router_runtime_args);
    }

    detail::CompileProgram(device, *fabric_program_ptr, /*fd_bootloader_mode=*/device->using_fast_dispatch());
    return fabric_program_ptr;
}

void configure_fabric_cores(IDevice* device) {
    std::vector<uint32_t> router_zero_buf(1, 0);

    auto router_chans = tt::Cluster::instance().get_fabric_ethernet_channels(device->id());
    for (const auto& router_chan : router_chans) {
        CoreCoord virtual_eth_core =
            tt::Cluster::instance().get_virtual_eth_core_from_channel(device->id(), router_chan);
        auto router_logical_core = device->logical_core_from_ethernet_core(virtual_eth_core);
        // initialize the semaphore
        auto fabric_router_sync_sem_addr =
            hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
        detail::WriteToDeviceL1(
            device, router_logical_core, fabric_router_sync_sem_addr, router_zero_buf, CoreType::ETH);
    }
}

}  // namespace tt::tt_metal
