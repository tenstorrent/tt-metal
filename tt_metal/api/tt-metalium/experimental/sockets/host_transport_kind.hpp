// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::distributed::host_transport {

enum class TransportKind : uint8_t {
    // One-sided verbs. Saturates the link; needs a RoCE device on both hosts.
    Rdma,
    // Two-sided point-to-point through DistributedContext. Runs anywhere MPI
    // does, including builds with no RDMA support at all. Much slower, and how
    // much depends entirely on what the MPI build can use underneath -- see the
    // tech report; this backend is for reach, not for speed.
    Mpi,
};

}  // namespace tt::tt_metal::distributed::host_transport
