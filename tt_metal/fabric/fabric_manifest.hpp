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

// Version of the fabric manifest schema emitted below.
constexpr int FABRIC_MANIFEST_VERSION = 1;

// Standard per-rank path for the fabric manifest.
std::filesystem::path fabric_manifest_path(const tt::llrt::RunTimeOptions& rtoptions);

// Serialize this fabric instance's topology to the fabric manifest JSON file.
//
// This captures state that is frozen for the run, including fabric config, meshes, chip coordinates, and the set of
// ethernet cores actually running fabric routers.
void serialize_fabric_manifest_to_file(
    const ControlPlane& control_plane, const std::filesystem::path& output_file_path);

}  // namespace tt::tt_fabric
