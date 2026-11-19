// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The NoC ordering of an X280 TLB window, as device code programs it
 * (noc.h) and host code and records name it.
 */
#ifndef X280_NOC_ORDERING_H
#define X280_NOC_ORDERING_H

#include <stdint.h>

/**
 * A unicast window's NoC ordering: the `ordering` field of
 * BlackholeA0/L2CPUTile/TLBWindows.md, "Configuration". Each value is the
 * field's encoding.
 *
 * Counted Writes (3) is not a member. One store through such a window leaves
 * the L2CPU tile's later non-posted stores returning before their NoC
 * acknowledgement until an ASIC reset; noc.h's
 * raw `posted`/`strict_order` entry points still reach it.
 */
typedef enum {
    NOC_ORDER_DEFAULT = 0, /* Default */
    NOC_ORDER_STRICT = 1,  /* Strict AXI */
    NOC_ORDER_POSTED = 2   /* Posted Writes */
} noc_ordering_t;

/** Whether `ordering` is a noc_ordering_t member; the checked entry points and the
 * host encoder refuse any other value. */
static inline int noc_ordering_valid(noc_ordering_t ordering) {
    return (uint32_t)ordering <= (uint32_t)NOC_ORDER_POSTED;
}

#endif /* X280_NOC_ORDERING_H */
