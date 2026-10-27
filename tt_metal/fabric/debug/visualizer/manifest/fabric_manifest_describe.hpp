// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {
class ControlPlane;
class FabricBuilderContext;
}  // namespace tt::tt_fabric

// The top-level manifest blocks' descriptions.

namespace tt::tt_fabric::manifest {

// Describes this process and the build it runs.
RunInfo describe_run(const ControlPlane& control_plane);

// Describes the fabric settings this run asked for, and what FabricContext derived from them.
FabricInfo describe_fabric(const ControlPlane& control_plane);

// Describes how each VC's credits travel on a mesh on this host. VCs no router has senders on are left out.
std::vector<VcCreditTransport> describe_credit_transport(const FabricBuilderContext& builder_context, MeshId mesh_id);

// Describes the L1 areas and registers the architecture fixes on every router core.
Arch describe_arch(const tt::tt_metal::Hal& hal);

// Describes every struct and enum a schema names.
TypeCatalog describe_types();

}  // namespace tt::tt_fabric::manifest
