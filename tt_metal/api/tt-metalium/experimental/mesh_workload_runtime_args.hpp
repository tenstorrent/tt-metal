// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt_stl/span.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::experimental {

// EXPERIMENTAL: these declarations may change or be removed without notice and carry no stability guarantee.
//
// One device's values of a per-device runtime args field of a legacy program in a MeshWorkload. Fields follow the
// rules of distributed::SetRuntimeArgsPerDevice (which sets every device of the range in one call); these set or drop
// the values of one `device`, which must lie in `program_range`. A device without a value of a field uses the
// program's own args.

// Words [first_word, first_word + values.size()) of the kernel's unique runtime args on `core`.
void SetRuntimeArgs(
    distributed::MeshWorkload& mesh_workload,
    const distributed::MeshCoordinateRange& program_range,
    const distributed::MeshCoordinate& device,
    KernelHandle kernel,
    const CoreCoord& core,
    uint32_t first_word,
    ttsl::Span<const uint32_t> values);
// Words [first_word, first_word + values.size()) of the kernel's common runtime args.
void SetCommonRuntimeArgs(
    distributed::MeshWorkload& mesh_workload,
    const distributed::MeshCoordinateRange& program_range,
    const distributed::MeshCoordinate& device,
    KernelHandle kernel,
    uint32_t first_word,
    ttsl::Span<const uint32_t> values);
// Drops one device's value of a field (`core` nullopt: common runtime args): it follows the program's own args again.
void ClearRuntimeArgs(
    distributed::MeshWorkload& mesh_workload,
    const distributed::MeshCoordinateRange& program_range,
    const distributed::MeshCoordinate& device,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word);

}  // namespace tt::tt_metal::experimental
