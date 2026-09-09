// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

/**
 * @warning INTERNAL. Everything declared in this header lives under
 * @c api/internal: it exists to serve tt-metal's own tooling and
 * bindings, is not part of the supported user-facing API, and may change or be
 * removed without a deprecation period.
 */

/**
 * Resolves the sub-devices a workload occupies, using the same implementation mesh
 * dispatch uses, so callers cannot drift from dispatch's own placement decision.
 *
 * Returns a reference to state cached on the workload; it is valid until the workload
 * is next modified or destroyed. @p mesh_workload is non-const because resolving
 * populates that cache.
 */
const std::unordered_set<SubDeviceId>& get_mesh_workload_sub_device_ids(
    distributed::MeshWorkload& mesh_workload, distributed::MeshDevice* mesh_device);

}  // namespace tt::tt_metal::internal
