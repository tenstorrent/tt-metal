// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The live bound of index_score_decode (custom_ops/index_score_decode/op.py, ``bound`` given; the host planner is
// custom_ops/sparse_bound.py): the reader reads the per-chip ``[1, 1]`` uint32 bound (pools in use this step) and
// turns it into the number of KC-tile chunks every user's row is scored over; the compute receives that count through
// a mailbox CB, the writer derives it from the same tensor. The (user, band) work items then split the LIVE chunks
// evenly over the same ``n_bands`` bands instead of the capacity's static split, so all three kernels agree on every
// item's chunk range (``index_score_item_range``) and every CB push is matched by a pop.

#pragma once

#include <cstdint>

// Chunks of 32*KC pools covering ``min(bound_pools, Tt * 32)`` (whole chunks, at least one).
inline uint32_t index_score_live_chunks(uint32_t bound_pools, uint32_t Tt, uint32_t KC) {
    const uint32_t cap = Tt * 32u;
    const uint32_t live = bound_pools < cap ? bound_pools : cap;
    const uint32_t chunk = 32u * KC;
    const uint32_t n = (live + chunk - 1u) / chunk;
    return n == 0u ? 1u : n;
}

// Chunks per band when ``live`` chunks are split over ``n_bands`` bands (the static split is ``band_chunks`` of the
// capacity: the caller passes ``live = total_chunks`` and keeps ``band_chunks`` without a bound).
inline uint32_t index_score_band_chunks(uint32_t live, uint32_t n_bands) { return (live + n_bands - 1u) / n_bands; }

// The chunk range [c_begin, c_end) of band ``band`` (empty when the band starts past the live chunks).
inline void index_score_item_range(
    uint32_t band, uint32_t per_band, uint32_t live, uint32_t& c_begin, uint32_t& c_end) {
    c_begin = band * per_band;
    c_end = (c_begin + per_band < live) ? (c_begin + per_band) : live;
}
