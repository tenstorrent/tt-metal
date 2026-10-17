// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <vector>
#include <tt-metalium/experimental/info.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/maybe_remote.hpp>
#include <tt_stl/assert.hpp>

// Experimental queries may change without notice and may evolve into MeshDevice methods.
namespace tt::tt_metal::experimental::mesh_device {

namespace detail {
std::vector<distributed::MeshCoordinate> local_coordinates(const distributed::MeshDevice& mesh);
[[noreturn]] void throw_non_uniform_info(const distributed::MeshCoordinate& coord, std::string_view property_name);
}  // namespace detail

/**
 * @brief Queries a property at one mesh coordinate.
 * @tparam InfoType Property tag from <tt-metalium/experimental/info.hpp>.
 * @throws std::runtime_error If the coordinate is out of bounds or belongs to another host.
 */
// Reject unsupported tags at compile time instead of leaving an undefined symbol at link time.
template <class InfoType>
typename InfoType::return_type get_info(const distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord) =
    delete;

// Supported properties; definitions live in mesh_device.cpp.
template <>
std::uint32_t get_info<info::l1_alignment>(
    const distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord);
template <>
std::uint32_t get_info<info::dram_alignment>(
    const distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord);
template <>
tt::ARCH get_info<info::architecture>(const distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord);
template <>
std::string get_info<info::architecture_name>(
    const distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord);

/**
 * @brief Queries a property of the local devices of a mesh.
 *
 * The property must have the same value on every local device. Use get_info_per_device<InfoType>(mesh) or
 * get_info<InfoType>(mesh, coord) for properties that can differ between devices.
 *
 * @tparam InfoType Property tag from `<tt-metalium/experimental/info.hpp>`, e.g. `info::l1_alignment`.
 * @return The value shared by all local devices, of type `InfoType::return_type`.
 * @throws std::runtime_error If the mesh has no local devices, or if the local devices report different values.
 */
template <class InfoType>
typename InfoType::return_type get_info(const distributed::MeshDevice& mesh) {
    const std::vector<distributed::MeshCoordinate> coords = detail::local_coordinates(mesh);
    TT_FATAL(!coords.empty(), "Cannot query {}: mesh device has no local devices", InfoType::name);
    typename InfoType::return_type value = get_info<InfoType>(mesh, coords.front());
    for (const distributed::MeshCoordinate& coord : coords) {
        if (!(get_info<InfoType>(mesh, coord) == value)) {
            detail::throw_non_uniform_info(coord, InfoType::name);
        }
    }
    return value;
}

/**
 * @brief Queries a property of every device in a mesh.
 *
 * @tparam InfoType Property tag from `<tt-metalium/experimental/info.hpp>`, e.g. `info::l1_alignment`.
 * @return A container shaped like the mesh. Entries for local devices hold the property value, of type
 * `InfoType::return_type`; entries for remote devices (owned by another host) are marked remote.
 */
template <class InfoType>
distributed::DistributedMeshContainer<typename InfoType::return_type> get_info_per_device(
    const distributed::MeshDevice& mesh) {
    distributed::DistributedMeshContainer<typename InfoType::return_type> result(mesh.shape());
    for (const distributed::MeshCoordinate& coord : detail::local_coordinates(mesh)) {
        result.at(coord) =
            distributed::MaybeRemote<typename InfoType::return_type>::local(get_info<InfoType>(mesh, coord));
    }
    return result;
}

}  // namespace tt::tt_metal::experimental::mesh_device
