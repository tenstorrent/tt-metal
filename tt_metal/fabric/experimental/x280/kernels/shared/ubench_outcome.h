// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Suite-local phases map to shared outcomes without changing wire values. */
#ifndef X280_UBENCH_OUTCOME_H
#define X280_UBENCH_OUTCOME_H

#include <stdint.h>

/* Only OK admits a measurement row; IN_PROGRESS is non-terminal. */
typedef enum {
    UBENCH_OUTCOME_OK = 0u,
    UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH = 1u,
    UBENCH_OUTCOME_TIMEOUT = 2u,
    UBENCH_OUTCOME_TRAP = 3u,
    UBENCH_OUTCOME_PATH_MISMATCH = 4u,
    UBENCH_OUTCOME_IN_PROGRESS = 5u
} ubench_outcome_t;

#if defined(__cplusplus)
constexpr int ubench_outcome_is_measurement(ubench_outcome_t o) { return o == UBENCH_OUTCOME_OK; }
#else
static inline int ubench_outcome_is_measurement(ubench_outcome_t o) { return o == UBENCH_OUTCOME_OK; }
#endif

#endif /* X280_UBENCH_OUTCOME_H */
