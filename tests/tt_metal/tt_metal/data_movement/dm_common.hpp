// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DM_COMMON_HPP
#define DM_COMMON_HPP

#include <cstdint>

namespace tt::tt_metal::unit_tests::dm {
// Unique id for each test run
extern uint32_t runtime_host_id;
}  // namespace tt::tt_metal::unit_tests::dm

#endif  // DM_COMMON_HPP
