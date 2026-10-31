// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QWEN38_RR_SLICE=1 (two links, SPLIT 1): every multicast packet carries only the half slices (OUT_TILES chunks of
// this link) of the receivers in the packet's range, in hop order, instead of the whole half row.  The forward
// packet (BRISC) holds the receivers rank + 1 .. rank + fwd, the backward packet (NCRISC) the receivers rank - 1 ..
// rank - bwd.  Staging slot and landing region of one link: block h - 1 of the forward packet at (h - 1) * H, block
// h - 1 of the backward packet at (fwd + h - 1) * H, H = OUT_TILES * PITCH; own slice (link 0, 2 H bytes) at
// (RING - 1) * H of the staging slot; link L lands at L * (RING - 1) * H of the sender's landing slot.

#pragma once

// the forward multicast range (BRISC) of rank s; the backward range is RING - 1 - this
template <uint32_t RING>
FORCE_INLINE uint32_t rr_fwd_range(uint32_t s) {
#if defined(QWEN38_RING_ROUTE)
    constexpr uint32_t half = RING / 2;
    return (s & 1) ? half : RING - 1 - half;
#else
    return RING - 1 - s;
#endif
}
