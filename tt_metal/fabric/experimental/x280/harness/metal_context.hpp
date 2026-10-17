// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/metal_context.hpp -- the process's Metal context, as the suites and
// fixtures reach it.

#ifndef X280_HARNESS_METAL_CONTEXT_HPP
#define X280_HARNESS_METAL_CONTEXT_HPP

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace x280::harness {

/** The process's tt::Cluster, owned by MetalContext and valid while it lives. */
inline const tt::Cluster& cluster() { return tt::tt_metal::MetalContext::instance().get_cluster(); }

/** Whether Metal's watcher is enabled; its host NoC sanitizer admits no L2CPU coordinate. */
inline bool watcher_enabled() { return tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled(); }

}  // namespace x280::harness

#endif  // X280_HARNESS_METAL_CONTEXT_HPP
