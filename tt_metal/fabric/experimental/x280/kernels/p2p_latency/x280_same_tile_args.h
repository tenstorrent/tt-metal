// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/p2p_latency/x280_same_tile_args.h -- x280_same_tile backend argument layout.
 *
 * x280_same_tile is X280 hart <-> X280 hart, same tile. Both participants live in the
 * same L2CPU's shared LIM, so their inbound mailboxes, staging regions, and
 * result regions are all LIM addresses in the same address space. No NOC
 * window is consumed; no address translation is required. The medium selector
 * says whether transmit goes through the coherent cache or an uncached alias.
 *
 * The runtime tells every hart the same args_addr. x280_same_tile packs TWO
 * participant sections back to back so hart 0 (participant A) and hart 1
 * (participant B) each read their own section by fixed hartid-scaled offset.
 * See docs/benchmarks/design.md, §Roles and mailbox model.
 *
 * Compiled by the host case and by the X280 image.
 */
#ifndef X280_P2P_LATENCY_X280_SAME_TILE_ARGS_H
#define X280_P2P_LATENCY_X280_SAME_TILE_ARGS_H

#include <stddef.h>
#include <stdint.h>

#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "ubench_section_load.h"

/* --------------------------------------------------------------------------
 * Medium selector
 * --------------------------------------------------------------------------
 * x280_same_tile is the only member with a medium choice (see design §Participant rows).
 * The backend admits coherent cached LIM and refuses uncached LIM, whose
 * visibility operation is not established.
 */
typedef enum {
    P2P_LATENCY_X280_SAME_TILE_MEDIUM_COHERENT = 0u,
    P2P_LATENCY_X280_SAME_TILE_MEDIUM_UNCACHED = 1u
} p2p_latency_x280_same_tile_medium_t;

/* --------------------------------------------------------------------------
 * x280_same_tile backend argument section
 * --------------------------------------------------------------------------
 * All four *_addr fields are LIM absolute addresses (i.e. what
 * X280Tile::available_lim() yielded plus an offset). The image writes
 * peer_inbound and own_result and reads own_inbound through a plain pointer.
 *
 * Fields carry every address the backend needs; the frontend never touches
 * them. Keeping them here rather than deriving them lets a host case place
 * mailboxes wherever available LIM allows and lets the image validate the
 * placement it was told rather than the one it recomputed.
 */
typedef struct {
    uint64_t own_inbound_addr;  /* my inbound mailbox, size = P2P_LATENCY_MAILBOX_BYTES */
    uint64_t peer_inbound_addr; /* peer's inbound mailbox */
    uint64_t own_staging_addr;  /* my staging, host-seeded */
    uint64_t own_result_addr;   /* my p2p_latency_result_t */
    uint32_t medium;            /* p2p_latency_x280_same_tile_medium_t */
    uint32_t reserved;
} p2p_latency_x280_same_tile_backend_args_t;

/* --------------------------------------------------------------------------
 * Per-participant argument section carried in the args block.
 *
 * cookie_common precedes the common args; cookie_backend separates common
 * from backend args. Both cookies are checked by the frontend and the
 * backend respectively, so a host/device layout skew fails by name at INIT.
 *
 * pad0 and pad1 are explicit because both `common` and `backend` contain u64
 * fields and so demand 8-byte alignment. Explicit padding fields are stable
 * across host and device compilers with -O0..O3, whereas implicit padding is
 * a compiler decision the layout header can only assert after the fact. Zero
 * them on the host so a device that ever inspects them sees a defined value.
 * -------------------------------------------------------------------------- */
typedef struct {
    uint32_t cookie_common;
    uint32_t pad0;
    p2p_latency_common_args_t common;
    uint32_t cookie_backend;
    uint32_t pad1;
    p2p_latency_x280_same_tile_backend_args_t backend;
} p2p_latency_x280_same_tile_section_t;

/* Args block: exactly two participant sections. The image at hart k reads
 * sections[k] (k in {0, 1}); harts 2 and 3 do not participate. */
typedef struct {
    p2p_latency_x280_same_tile_section_t sections[2];
} p2p_latency_x280_same_tile_args_block_t;

/* --------------------------------------------------------------------------
 * Static assertions
 * -------------------------------------------------------------------------- */

UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_x280_same_tile_backend_args_t) == 40u, "x280_same_tile backend args are 40 bytes: 4 x u64 + 2 x u32");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_same_tile_backend_args_t, own_inbound_addr) == 0u,
    "own_inbound_addr is the first backend field");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_same_tile_backend_args_t, medium) == 32u, "medium follows the four addresses");
UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_x280_same_tile_section_t) == 88u,
    "x280_same_tile section = 4 cookie + 4 pad + 32 common + 4 cookie + 4 pad + 40 backend");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_same_tile_section_t, common) == 8u,
    "common args begin on their natural 8-byte alignment");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_same_tile_section_t, cookie_backend) == 40u,
    "cookie_backend is right after the 32-byte common args");
UBENCH_LAYOUT_ASSERT(
    offsetof(p2p_latency_x280_same_tile_section_t, backend) == 48u,
    "backend args begin on their natural 8-byte alignment");
UBENCH_LAYOUT_ASSERT(
    sizeof(p2p_latency_x280_same_tile_args_block_t) == 2u * sizeof(p2p_latency_x280_same_tile_section_t),
    "the args block is exactly two participant sections");
/* The args block as a whole is what the host writes into cache-line aligned
 * available LIM. Round-up padding is the host's responsibility: the runtime
 * checks the block's placement, not its layout. */

#endif /* X280_P2P_LATENCY_X280_SAME_TILE_ARGS_H */
