// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <nlohmann/json_fwd.hpp>

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_fabric {

class ControlPlane;

namespace manifest {
struct ManifestFacts;
}  // namespace manifest

// How make_manifest_json writes each fields block.
enum class FieldsLayout : uint8_t {
    // Each unique block once, in the top-level field_sets, with its id (such as "F0") where the block was. Useful for
    // when blocks repeat across routers, so we can use one copy as the reference.
    SHARED,
    // Each block in full where it is, with no field_sets. Needed for the decoded file as its structure uses the
    // manifest and adds on decoded values. The shared layouts don't work then because each field entry holds its own
    // decoded value.
    PER_ROUTER,
};

// The manifest's JSON for `facts`. It reads nothing but `facts`: the types and vocabulary blocks are derived from them.
nlohmann::ordered_json make_manifest_json(
    const manifest::ManifestFacts& facts, FieldsLayout fields_layout = FieldsLayout::SHARED);

// Writes this rank's fabric manifest, which includes facts about the fabric that are fixed for the run, into this run's
// folder: <logs_dir>/generated/fabric_debug/runs/<run_id>/. Rank 0 makes the run id and broadcasts it, so every rank of
// the fabric must call this. This requires the fabric routers to be compiled.
void write_fabric_manifest(const ControlPlane& control_plane, const tt::llrt::RunTimeOptions& rtoptions);

}  // namespace tt::tt_fabric
