// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal::internal {

// EXPERIMENTAL / INTERNAL API. Not part of the stable public surface; signature may change.
//
// Resolve a FabricNodeId to the HARDWARE-STABLE 64-bit ASIC unique id of the chip it names.
//
// This is the chip's physical, host-global-unique identity (the same value the UMD reports as
// ``UmdDevice::unique_id()`` and the same id fabric sockets route by). It is resolved live from
// the FabricNodeId through the fabric control plane (``ControlPlane::get_asic_id_from_fabric_node_id``).
//
// It is explicitly NOT the process-local UMD ChipId (``MeshDevice::get_device_id``, 0..n-1), which
// collides across the multiple meshes on a single host. Callers that need to key per-chip state that
// is shared across meshes/processes (e.g. KV-cache migration device maps) must use this ASIC unique
// id, not the logical device id.
AsicID get_chip_unique_id_from_fabric_node_id(const tt::tt_fabric::FabricNodeId& fabric_node_id);

// Convenience overload mirroring the (mesh_id, chip_id) form a FabricNodeId is built from.
AsicID get_chip_unique_id_from_fabric_node_id(uint32_t mesh_id, uint32_t chip_id);

}  // namespace tt::tt_metal::internal
