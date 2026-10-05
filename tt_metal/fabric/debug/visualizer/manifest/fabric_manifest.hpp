// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_fabric {

class ControlPlane;

// Version of the fabric manifest schema that will be emitted.
constexpr int FABRIC_MANIFEST_VERSION = 1;

// Standard per-rank path for the fabric manifest.
std::filesystem::path fabric_manifest_path(const tt::llrt::RunTimeOptions& rtoptions);

// Removes this rank's manifest from an earlier run.
void remove_stale_fabric_manifest(const tt::llrt::RunTimeOptions& rtoptions);

// Writes this rank's fabric manifest, which includes facts about the fabric that are fixed for the run.
// This requires the fabric routers to be compiled.
void write_fabric_manifest(const ControlPlane& control_plane, const tt::llrt::RunTimeOptions& rtoptions);

}  // namespace tt::tt_fabric
