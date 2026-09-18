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
    // does, but far slower, by however much the MPI build's own transport is.
    // For reach, not for speed; see the tech report.
    Mpi,
};

}  // namespace tt::tt_metal::distributed::host_transport
