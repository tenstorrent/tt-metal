// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"

#ifdef TRISC_PACK

#include "circular_buffer.h"

using namespace tt;

/*
 * The CB holds 64 pages (2 steps).
 */

static constexpr auto CB_ID = tt::CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

namespace NAMESPACE {
void MAIN {
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb_reserve_back(CB_ID, CB_STEP_SIZE);
        cb_push_back(CB_ID, CB_STEP_SIZE);
    }
}
}  // namespace NAMESPACE
#else
namespace NAMESPACE {
void MAIN {}
}  // namespace NAMESPACE
#endif
