// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Step-0 producer, data-movement variant (BRISC / NCRISC). The host launches this on RISCV_0 (PROC_IDX=0)
// and RISCV_1 (PROC_IDX=1); PROC_IDX comes from the kernel -D defines. See producer_common.h.
#include "producer_common.h"

void kernel_main() { producer_run(); }
