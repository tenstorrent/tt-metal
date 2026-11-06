// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_fabric {

class ControlPlane;

namespace manifest {
struct ManifestFacts;
}  // namespace manifest

// The manifest's JSON for `facts`. It reads nothing but `facts`: the types and vocabulary blocks are derived from them.
nlohmann::ordered_json make_manifest_json(const manifest::ManifestFacts& facts);

// Writes this rank's fabric manifest, which includes facts about the fabric that are fixed for the run, into this run's
// folder: <logs_dir>/generated/fabric/runs/<run_id>/. Rank 0 makes the run id and broadcasts it, so every rank of the
// fabric must call this. This requires the fabric routers to be compiled.
void write_fabric_manifest(const ControlPlane& control_plane, const tt::llrt::RunTimeOptions& rtoptions);

}  // namespace tt::tt_fabric
