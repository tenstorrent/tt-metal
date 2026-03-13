// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt_stl/strong_type.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal {

// Due to the number of existing MetalContext instance calls in the codebase,
// we use a ContextId for objects to access the corresponding MetalContext instance to make the
// migration easier.
// TODO: Remove the ContextId in favor of directly passing around the MetalContext reference.
using ContextId = ttsl::StrongType<int, struct ContextIdTag>;

// The default context is implicitly created the first time MetalContext::instance(DEFAULT_CONTEXT_ID) is called.
constexpr ContextId DEFAULT_CONTEXT_ID = ContextId{0};

// Limit the number of context IDs so they can be stored in an array
constexpr size_t MAX_CONTEXT_COUNT = 32;

// Helper function to extract the context ID from the public API device type by casting it to the impl type.
// This function may perform a dynamic cast on the IDevice so should not be used in performance critical paths.
// TODO: Remove these extract_context_id functions once all child objects have MetalEnv and MetalContext references
// passed in
ContextId extract_context_id(const IDevice* device);

ContextId extract_context_id(const distributed::MeshDevice* mesh_device);

// Extract context ID from MeshDevice or Device, using whichever one is not nullptr
// This is used specifically by the profiler as some functions accept both a MeshDevice xor a IDevice pointer
// TOOD: Remove this and make profiler aware of which MetalEnv the device is on
// https://github.com/tenstorrent/tt-metal/issues/39741
ContextId extract_context_id(const distributed::MeshDevice* mesh_device, const IDevice* device);

}  // namespace tt::tt_metal
