// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's call-site ids. An id is the address of its call site's handle in .device_publish_ids
// (internal/debug/device_publish_impl.h); nothing may branch on its value except TT_STALL_CALL_SITE_ID.
#pragma once

#define TT_CALL_SITE_ID_COUNT (1u << 16)
// The stall call site, the producer's own, which every image shares: recognized by value, so it has no ELF record and
// no source location (llrt/call_sites.cpp describes it).
#define TT_STALL_CALL_SITE_ID (TT_CALL_SITE_ID_COUNT - 1u)
