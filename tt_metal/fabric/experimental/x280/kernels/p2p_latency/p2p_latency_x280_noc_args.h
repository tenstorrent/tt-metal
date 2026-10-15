// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * X280-side NoC mailbox arguments shared by x280_peer_tile, tensix_peer_tile,
 * erisc_peer_tile and drisc_peer_tile.
 * See docs/benchmarks/design.md, "Roles and mailbox model".
 */
#ifndef X280_P2P_LATENCY_X280_NOC_ARGS_H
#define X280_P2P_LATENCY_X280_NOC_ARGS_H

#include <stddef.h>
#include <stdint.h>

#include "../../include/x280_noc_coord.h"
#include "../../include/x280_noc_ordering.h"
#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "ubench_section_load.h"

/* The ordering the X280 opens its window onto the peer with, which its records
 * name: Strict AXI, because WAIT_FOR_COMPLETION needs the store's BRESP to wait
 * for the NoC acknowledgement. */
#define P2P_LATENCY_X280_NOC_ORDERING NOC_ORDER_STRICT

/* Addresses are absolute in this X280's address space. peer_inbound_addr must
 * lie in the 2 MiB window configured for peer_noc. */
typedef struct {
    uint64_t own_inbound_addr;
    uint64_t own_result_addr;
    uint64_t peer_inbound_addr;
    x280_noc_coord_t peer_noc; /* virtual (UMD TRANSLATED) destination */
    uint32_t peer_tlb_index;   /* X280_TLB_PAIRED_BASE + n */
    uint32_t reserved;
} p2p_latency_x280_noc_backend_args_t;

/* The section carries no layout version: the cookies name its sections, and a
 * peer built against another layout is not detected. */
typedef struct {
    uint32_t cookie_common;
    uint32_t pad0;
    p2p_latency_common_args_t common;
    uint32_t cookie_backend;
    uint32_t pad1;
    p2p_latency_x280_noc_backend_args_t backend;
} p2p_latency_x280_noc_section_t;

UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_x280_noc_backend_args_t) == 40u, "X280 NOC backend args: 3 x u64 + 4 x u32 = 24 + 16 = 40");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_backend_args_t, own_inbound_addr) == 0u,
    "own_inbound_addr is the first backend field");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_backend_args_t, own_result_addr) == 8u,
    "own_result_addr remains naturally aligned after own_inbound_addr");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_backend_args_t, peer_inbound_addr) == 16u,
    "peer_inbound_addr remains naturally aligned after own_result_addr");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_backend_args_t, peer_noc) == 24u, "the u32 tail begins after three u64 addresses");
UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_x280_noc_section_t) == 88u,
    "X280 NOC section = 4 cookie + 4 pad + 32 common + 4 cookie + 4 pad + 40 backend");
UBENCH_LAYOUT_ASSERT(offsetof(p2p_latency_x280_noc_section_t, cookie_common) == 0u, "common cookie is the first field");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_section_t, pad0) == 4u, "pad0 preserves natural alignment for the common u64 fields");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_section_t, common) == 8u, "common args begin on their natural 8-byte alignment");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_section_t, cookie_backend) == 40u,
    "cookie_backend is right after the 32-byte common args");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_section_t, pad1) == 44u,
    "pad1 preserves natural alignment for the backend u64 fields");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_noc_section_t, backend) == 48u, "backend args begin on their natural 8-byte alignment");

#endif /* X280_P2P_LATENCY_X280_NOC_ARGS_H */
