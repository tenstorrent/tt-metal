// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Decode DS matmul y[M <= 32, N] = x[M, K] @ W[K, N] (Stage 7a megakernel building block, ds_mm.py is the host side):
// W DRAM width-sharded over the 8 banks (DSLinear's tensor, read in place), 16 bank-reader cores (2 per bank, the DS
// placement), reader (b, r) owning output tiles [b * SHARD_W + r * NR, + NR). Per K block of KB tile rows:
//   BRISC / NoC0: the reader's weight block (KB rows x NR tiles, contiguous per row in its bank) into a 3-deep ring,
//                 one transaction id per block (2 blocks in flight);
//   NCRISC / NoC1: the activation block (KB tiles) pulled from the x tensor; after the last block, the NR output tiles
//                  written to y;
//   TRISC: matmul_block subblocks of SUBW tiles, every block packed onto the same fp32 partial tiles with packer L1
//          accumulation; after the last block the partials are copied (exact fp32 unpack) and packed to the output
//          dtype.
#pragma once

#include <cstdint>

#ifndef DSMM_CB_BASE
#define DSMM_CB_BASE 0  // megakernel: CB indices shifted past the other phases' (e.g. 24)
#endif

namespace dsmm {

constexpr uint32_t CB_IN0 = DSMM_CB_BASE + 0;   // activation block (x dtype), 2 x KB tiles
constexpr uint32_t CB_W = DSMM_CB_BASE + 1;     // weight ring (W dtype), 3 x KB x NR tiles
constexpr uint32_t CB_PART = DSMM_CB_BASE + 2;  // fp32 partials, NR tiles (L1-accumulated in place)
constexpr uint32_t CB_OUT = DSMM_CB_BASE + 3;   // output tiles (y dtype), NR tiles

}  // namespace dsmm
