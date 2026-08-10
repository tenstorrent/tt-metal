// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/profiler_types.hpp>

namespace tt::tt_metal {
namespace detail {

/**
 * Sync TT devices with host
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range               |
 * Required |
 * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
 * */
void ProfilerSync(ProfilerSyncState state);

}  // namespace detail
}  // namespace tt::tt_metal
