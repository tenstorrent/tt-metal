// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/transport_bandwidth/tensix_to_x280.h -- the Tensix -> X280 member
 * of the transport_bandwidth suite.
 *
 * Tensix is the initiator; X280 is a passive addressable target. The only
 * X280-side prerequisite is that the tile is in resident-idle with LIM ECC
 * primed and the destination region is a whole cache-line-aligned span of
 * host-primed LIM. The design's "no generic remote drain kernel" rule holds
 * here: no active X280 image is submitted.
 *
 * The kernel receives its arguments through Metal's runtime-args interface.
 * The slot numbering below is fixed by this header so the host packer and
 * the kernel unpacker agree without a shared struct; slot 0 carries
 * TB_TX2X_LAYOUT_COOKIE (ubench_layout_cookie.h).
 *
 * See x280/CHARACTERIZATION_UBENCH_DESIGN.md § Transport bandwidth for the
 * frontend/backend contract this member implements.
 */
#ifndef X280_TRANSPORT_BW_TENSIX_TO_X280_H
#define X280_TRANSPORT_BW_TENSIX_TO_X280_H

#include <stdint.h>

#include "../shared/ubench_layout_cookie.h"
#include "transport_bandwidth_common.h"
#include "ubench_section_load.h"

/* --------------------------------------------------------------------------
 * Runtime-arg schema
 * --------------------------------------------------------------------------
 * The order matters; add new slots only at the end.
 */
#define TB_TX2X_LAYOUT_COOKIE UBENCH_LAYOUT_COOKIE('T', 'B', 'M', 1)

enum transport_bw_tensix_to_x280_arg_slot {
    TB_TX2X_ARG_LAYOUT_COOKIE = UBENCH_LAYOUT_COOKIE_SLOT,
    TB_TX2X_ARG_TRANSFER_SIZE = 1,
    TB_TX2X_ARG_WARMUP_COUNT = 2,
    TB_TX2X_ARG_TRANSFER_COUNT = 3,
    TB_TX2X_ARG_BOUNDARY = 4, /* transport_bw_boundary_t */
    TB_TX2X_ARG_RUN_NONCE = 5,
    TB_TX2X_ARG_OWN_STAGING_ADDR = 6, /* L1: source of every issue */
    TB_TX2X_ARG_OWN_SCRATCH_ADDR = 7, /* L1: 4B slot for the verify read-back */
    TB_TX2X_ARG_OWN_RESULT_ADDR = 8,  /* L1: transport_bw_result_t destination */
    TB_TX2X_ARG_PEER_DST_ADDR_LO = 9, /* X280 LIM (low 32 bits) */
    TB_TX2X_ARG_PEER_DST_ADDR_HI = 10,
    TB_TX2X_ARG_PEER_VERIFY_ADDR_LO = 11, /* X280 LIM sentinel word */
    TB_TX2X_ARG_PEER_VERIFY_ADDR_HI = 12,
    TB_TX2X_ARG_PEER_NOC_X = 13, /* TRANSLATED for Tensix NOC */
    TB_TX2X_ARG_PEER_NOC_Y = 14,
    TB_TX2X_ARG_COUNT = 15
};

/* --------------------------------------------------------------------------
 * Byte-size envelope
 * --------------------------------------------------------------------------
 * transfer_size is one of the backend's listed sizes. The staging region on
 * L1 must be at least transfer_size bytes; the host allocates and zeroes
 * it to that upper bound.
 */
#define TB_TX2X_STAGING_MAX_BYTES 16384u /* the largest listed transfer_size */

/* --------------------------------------------------------------------------
 * Static assertions
 * -------------------------------------------------------------------------- */
UBENCH_LAYOUT_ASSERT((int)TB_TX2X_ARG_COUNT == 15, "Tensix->X280 runtime-arg schema has 15 slots");

#endif /* X280_TRANSPORT_BW_TENSIX_TO_X280_H */
