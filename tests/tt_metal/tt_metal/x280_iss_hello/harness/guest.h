// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared guest addresses for harness tests. Must sit in Spike LIM
// (0x08000000 + 0x1E0000) and above the linked image (ends 0x08120000).

#ifndef X280_HARNESS_GUEST_H_
#define X280_HARNESS_GUEST_H_

#define HARNESS_DATA_BASE 0x08140000UL
#define HARNESS_DATA_SIZE 0x00001000UL
#define HARNESS_MAGIC 0x48524E53UL /* 'HRNS' */

#endif
