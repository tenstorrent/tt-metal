// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <sys/types.h>

// Owner identity of a /dev/shm resource: the pid embedded in a NamedShm name
// and that process's start time. This is the liveness rule the
// ShmResourceTracker stale scan applies, shared with the connector side so a
// descriptor or segment left behind by a dead owner reads as "not published
// yet". Linux only (/proc), runtime-internal: not part of the public API.

namespace tt::tt_metal::distributed {

// Pid encoded in a NamedShm name, [/]tt_{prefix}_{pid}_{random}_{counter}; 0 when the name carries none.
pid_t pid_from_shm_name(const std::string& shm_name);

// Start time of `pid` in clock ticks since boot (/proc/<pid>/stat field 22); 0 when unreadable.
uint64_t process_start_time(pid_t pid);

// True when `pid` is a running process (not a zombie) and, if `start_time` is non-zero, it is the
// instance that started then. A pid handed to another process after the owner died is "dead".
bool is_process_alive(pid_t pid, uint64_t start_time);

}  // namespace tt::tt_metal::distributed
