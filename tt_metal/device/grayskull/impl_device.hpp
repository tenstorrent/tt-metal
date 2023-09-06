/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "device/tt_silicon_driver_common.hpp"

// See src/t6ifc/t6py/packages/tenstorrent/data/grayskull/pci/tlb.yaml
// 1M
// local_offset: [ 0, 11,  0,  "36-bit address prefix, prepended to the 20 LSBs of issued address to form a 56-bit NOC address. The 1MB TLB #n corresponds to the 1MB MMIO range starting at (0x0 + N*0x100000)."]
// x_end       : [ 0, 17, 12,  "" ]
// y_end       : [ 0, 23, 18,  "" ]
// x_start     : [ 0, 29, 24,  "" ]
// y_start     : [ 0, 35, 30,  "" ]
// noc_sel:      [ 0, 36, 36,  "NOC select (1 = NOC1, 0 = NOC0)"]
// mcast:        [ 0, 37, 37,  "1 = multicast, 0 = unicast"]
// ordering:     [ 0, 39, 38,  "ordering mode (01 = strict (full AXI ordering), 00 = relaxed (no RAW hazard), 10 = posted (may have RAW hazard)"]
// linked:       [ 0, 40, 40,  "linked"]

// 2M
// local_offset: [ 0, 10,  0,  "35-bit address prefix, prepended to the 21 LSBs of issued address to form a 56-bit NOC address. The 2MB TLB #n corresponds to the 2MB MMIO range starting at (0x9C00000 + N*0x200000)."]
// x_end       : [ 0, 16, 11,  "" ]
// y_end       : [ 0, 22, 17,  "" ]
// x_start     : [ 0, 28, 23,  "" ]
// y_start     : [ 0, 34, 29,  "" ]
// noc_sel:      [ 0, 35, 35,  "NOC select (1 = NOC1, 0 = NOC0)"]
// mcast:        [ 0, 36, 36,  "1 = multicast, 0 = unicast"]
// ordering:     [ 0, 38, 37,  "ordering mode (01 = strict (full AXI ordering), 00 = relaxed (no RAW hazard), 10 = posted (may have RAW hazard)"]
// linked:       [ 0, 39, 39,  "linked"]

// 16M
// local_offset: [ 0, 7 ,  0,  "32-bit address prefix, prepended to the 24 LSBs of issued address to form a 56-bit NOC address. The 16MB TLB #n corresponds to the 16MB MMIO range starting at (0xB000000 + N*0x1000000)."]
// x_end       : [ 0, 13,  8,  "" ]
// y_end       : [ 0, 19, 14,  "" ]
// x_start     : [ 0, 25, 20,  "" ]
// y_start     : [ 0, 31, 26,  "" ]
// noc_sel:      [ 0, 32, 32,  "NOC select (1 = NOC1, 0 = NOC0)"]
// mcast:        [ 0, 33, 33,  "1 = multicast, 0 = unicast"]
// ordering:     [ 0, 35, 34,  "ordering mode (01 = strict (full AXI ordering), 00 = relaxed (no RAW hazard), 10 = posted (may have RAW hazard)"]
// linked:       [ 0, 36, 36,  "linked"]

const auto TLB_1M_OFFSET = TLB_OFFSETS {
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

const auto TLB_2M_OFFSET = TLB_OFFSETS {
    .local_offset = 0,
    .x_end = 11,
    .y_end = 17,
    .x_start = 23,
    .y_start = 29,
    .noc_sel = 35,
    .mcast = 36,
    .ordering = 37,
    .linked = 39,
    .static_vc = 40,
    .static_vc_end = 41
};

const auto TLB_16M_OFFSET = TLB_OFFSETS {
    .local_offset = 0,
    .x_end = 8,
    .y_end = 14,
    .x_start = 20,
    .y_start = 26,
    .noc_sel = 32,
    .mcast = 33,
    .ordering = 34,
    .linked = 36,
    .static_vc = 37,
    .static_vc_end = 38
};
