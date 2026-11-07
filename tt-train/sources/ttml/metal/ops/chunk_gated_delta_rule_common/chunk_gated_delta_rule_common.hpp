// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constants shared by the forward and backward chunk_gated_delta_rule ops.  Included by
// the host factories and by the kernels, so it has no dependencies beyond <cstdint>.

#pragma once

#include <cstdint>

namespace chunk_gated_delta_rule_common {

// L1 below the circular-buffer region that get_max_worker_l1_unreserved_size() still counts: the
// kernel-config ring buffer the kernel binaries live in, plus slack.  Every L1 fit check subtracts it.
constexpr uint32_t kL1KernelConfigReserve = 80U * 1024U;

}  // namespace chunk_gated_delta_rule_common
