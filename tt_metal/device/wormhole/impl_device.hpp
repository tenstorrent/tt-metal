/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "device/tt_silicon_driver_common.hpp"

// See src/t6ifc/t6py/packages/tenstorrent/data/wormhole/pci/tlb.yaml
// local_offset: [ 0, 15,  0,  "36-bit address prefix, prepended to the 20 LSBs of issued address to form a 56-bit NOC address. The 1MB TLB #n corresponds to the 1MB MMIO range starting at (0x0 + N*0x100000)."]
// x_end       : [ 0, 21, 16,  "" ]
// y_end       : [ 0, 27, 22,  "" ]
// x_start     : [ 0, 33, 28,  "" ]
// y_start     : [ 0, 39, 34,  "" ]
// noc_sel:      [ 0, 40, 40,  "NOC select (1 = NOC1, 0 = NOC0)"]
// mcast:        [ 0, 41, 41,  "1 = multicast, 0 = unicast"]
// ordering:     [ 0, 43, 42,  "ordering mode (01 = strict (full AXI ordering), 00 = relaxed (no RAW hazard), 10 = posted (may have RAW hazard)"]
// linked:       [ 0, 44, 44,  "linked"]

// local_offset: [ 0, 14,  0,  "35-bit address prefix, prepended to the 21 LSBs of issued address to form a 56-bit NOC address. The 2MB TLB #n corresponds to the 2MB MMIO range starting at (0x9C00000 + N*0x200000)."]
// x_end       : [ 0, 20, 15,  "" ]
// y_end       : [ 0, 26, 21,  "" ]
// x_start     : [ 0, 32, 27,  "" ]
// y_start     : [ 0, 38, 33,  "" ]
// noc_sel:      [ 0, 39, 39,  "NOC select (1 = NOC1, 0 = NOC0)"]
// mcast:        [ 0, 40, 40,  "1 = multicast, 0 = unicast"]
// ordering:     [ 0, 42, 41,  "ordering mode (01 = strict (full AXI ordering), 00 = relaxed (no RAW hazard), 10 = posted (may have RAW hazard)"]
// linked:       [ 0, 43, 43,  "linked"]

// local_offset: [ 0, 11,  0,  "32-bit address prefix, prepended to the 24 LSBs of issued address to form a 56-bit NOC address. The 16MB TLB #n corresponds to the 16MB MMIO range starting at (0xB000000 + N*0x1000000)."]
// x_end       : [ 0, 17, 12,  "" ]
// y_end       : [ 0, 23, 18,  "" ]
// x_start     : [ 0, 29, 24,  "" ]
// y_start     : [ 0, 35, 30,  "" ]
// noc_sel:      [ 0, 36, 36,  "NOC select (1 = NOC1, 0 = NOC0)"]
// mcast:        [ 0, 37, 37,  "1 = multicast, 0 = unicast"]
// ordering:     [ 0, 39, 38,  "ordering mode (01 = strict (full AXI ordering), 00 = relaxed (no RAW hazard), 10 = posted (may have RAW hazard)"]
// linked:       [ 0, 40, 40,  "linked"]

const auto TLB_1M_OFFSET = TLB_OFFSETS {
    .local_offset = 0,
    .x_end = 16,
    .y_end = 22,
    .x_start = 28,
    .y_start = 34,
    .noc_sel = 40,
    .mcast = 41,
    .ordering = 42,
    .linked = 44,
    .static_vc = 45,
    .static_vc_end = 46
};

const auto TLB_2M_OFFSET = TLB_OFFSETS {
    .local_offset = 0,
    .x_end = 15,
    .y_end = 21,
    .x_start = 27,
    .y_start = 33,
    .noc_sel = 39,
    .mcast = 40,
    .ordering = 41,
    .linked = 43,
    .static_vc = 44,
    .static_vc_end = 45
};

const auto TLB_16M_OFFSET = TLB_OFFSETS {
    .local_offset = 0,
    .x_end = 12,
    .y_end = 18,
    .x_start = 24,
    .y_start = 30,
    .noc_sel = 36,
    .mcast = 37,
    .ordering = 38,
    .linked = 40,
    .static_vc = 41,
    .static_vc_end = 42
};
