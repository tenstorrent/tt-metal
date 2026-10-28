// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's call-site ids. An id is the address of its call site's handle in .device_publish_ids
// (internal/debug/device_publish_impl.h); nothing may branch on its value except TT_ZONE_STALL_ID.
#pragma once

#define TT_ZONE_ID_COUNT (1u << 16)
// The profiler's own stall zone: recognized by value, so it has no ELF record and no source location.
#define TT_ZONE_STALL_ID (TT_ZONE_ID_COUNT - 1u)
