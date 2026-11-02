// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prints Quasar's per-core L1 budget from dev_mem_map.h: the end of the firmware map, the allocator's unreserved
// base (MEM_MAP_END + the kernel-config ring, aligned to 64 B, as create_tensix_mem_map does) and the usable bytes.
// Build: gcc -I tt_metal/hw/inc/internal/tt-2xx/quasar -o /tmp/quasar_usable_l1 \
//            tests/ttnn/unit_tests/operations/debug/probes/quasar_usable_l1.c && /tmp/quasar_usable_l1
// Result (2026-11-01): MEM_MAP_END 230976 (0x38640), unreserved base 333376 (0x51640), usable 3860928 B (3.68 MiB).
#include <stdint.h>
#include <stdio.h>
#define MaxDMProcessorsPerCoreType 8
#include "dev_mem_map.h"

int main() {
    const unsigned align = 64;
    const unsigned end = MEM_MAP_END;
    const unsigned unreserved = ((end + MEM_KERNEL_CONFIG_SIZE - 1) | (align - 1)) + 1;
    printf(
        "MEM_MAP_END=%u (0x%x)\nUNRESERVED=%u (0x%x)\nusable=%u (%.4f MiB)\n",
        end,
        end,
        unreserved,
        unreserved,
        MEM_L1_SIZE - unreserved,
        (MEM_L1_SIZE - unreserved) / 1048576.0);
    return 0;
}
