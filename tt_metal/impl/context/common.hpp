// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_metal {

std::shared_ptr<distributed::multihost::DistributedContext> construct_compute_only_distributed_context(
    const tt::tt_fabric::ControlPlane& control_plane);

}  // namespace tt::tt_metal
