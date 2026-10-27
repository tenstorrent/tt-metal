// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "clock_rollover_body.hpp"
void kernel_main() { clock_rollover::run(); }
