// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * NoC tile coordinates as host and device code exchange them.
 */
#ifndef X280_NOC_COORD_H
#define X280_NOC_COORD_H

#include <stdbool.h>
#include <stdint.h>

/** Largest NoC coordinate on either axis: a NoC address carries six bits per axis. */
#define X280_NOC_COORD_MAX 63u

/**
 * One tile's NoC coordinate. Every carrier states which system it holds; the
 * host produces translated (virtual) values through the X280 runtime resolver.
 */
typedef struct {
    uint32_t x;
    uint32_t y;
} x280_noc_coord_t;

/**
 * A multicast rectangle, from its start corner to its end corner on each axis
 * (start <= end). A NoC1 sender swaps the corners when it issues the
 * multicast; the rectangle itself is NoC-independent.
 */
typedef struct {
    x280_noc_coord_t start;
    x280_noc_coord_t end;
} x280_noc_rect_t;

/** The range check every coordinate unpack applies: both axes within six bits. */
static inline bool x280_noc_coord_valid(x280_noc_coord_t c) {
    return c.x <= X280_NOC_COORD_MAX && c.y <= X280_NOC_COORD_MAX;
}

/**
 * Whether a consumer on NoC `noc` (0 or 1) reaches the tile translated
 * coordinate `c` names. Firmware leaves X untranslated where Y < 2 and X != 8,
 * so there the same pair names a mirrored tile on NoC 1
 * (BlackholeA0/NoC/Coordinates.md, "Coordinate Translation").
 */
static inline bool x280_noc_translated_addressable(x280_noc_coord_t c, uint32_t noc) {
    return x280_noc_coord_valid(c) && !(noc == 1u && c.y < 2u && c.x != 8u);
}

/** Both corners pass x280_noc_coord_valid and start <= end on each axis. */
static inline bool x280_noc_rect_valid(x280_noc_rect_t r) {
    return x280_noc_coord_valid(r.start) && x280_noc_coord_valid(r.end) && r.start.x <= r.end.x && r.start.y <= r.end.y;
}

/** Whether `c` lies inside `r`, edges included. `r` must be valid. */
static inline bool x280_noc_rect_contains(x280_noc_rect_t r, x280_noc_coord_t c) {
    return c.x >= r.start.x && c.x <= r.end.x && c.y >= r.start.y && c.y <= r.end.y;
}

/**
 * Packs `c` as (y << 6) | x, the form an externally fixed table or a dense
 * argument word carries. Returns false and leaves `*enc` unwritten when `c`
 * fails x280_noc_coord_valid; a value is refused, never masked.
 */
static inline bool x280_noc_xy_pack(x280_noc_coord_t c, uint32_t* enc) {
    if (!x280_noc_coord_valid(c)) {
        return false;
    }
    *enc = (c.y << 6) | c.x;
    return true;
}

/**
 * Unpacks a (y << 6) | x word. Returns false and leaves `*c` unwritten when
 * any bit above the twelve coordinate bits is set.
 */
static inline bool x280_noc_xy_unpack(uint32_t enc, x280_noc_coord_t* c) {
    if ((enc >> 12) != 0u) {
        return false;
    }
    c->x = enc & X280_NOC_COORD_MAX;
    c->y = (enc >> 6) & X280_NOC_COORD_MAX;
    return true;
}

#endif /* X280_NOC_COORD_H */
