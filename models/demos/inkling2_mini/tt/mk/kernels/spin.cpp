// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Test stand-in for a DRAM-light program (a single-core residual kernel): busy-waits `cycles` cycles.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() { riscv_wait(get_arg_val<uint32_t>(0)); }
