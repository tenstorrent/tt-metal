// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/unit_mesh_group.hpp>

#include <tt_stl/assert.hpp>

#include "mesh_device_impl.hpp"

namespace tt::tt_metal::distributed {

UnitMeshGroup UnitMeshGroup::open(ttsl::Span<const ChipId> chip_ids, const MeshDeviceOptions& options) {
    return MeshDeviceImpl::create_unit_mesh_group(
        DEFAULT_CONTEXT_ID, std::vector<ChipId>(chip_ids.begin(), chip_ids.end()), options);
}

UnitMeshGroup::UnitMeshGroup(UnitMeshGroup&& other) noexcept : members_(std::move(other.members_)) {
    other.members_.clear();
}

UnitMeshGroup& UnitMeshGroup::operator=(UnitMeshGroup&& other) noexcept {
    if (this != &other) {
        close();
        members_ = std::move(other.members_);
        other.members_.clear();
    }
    return *this;
}

UnitMeshGroup::~UnitMeshGroup() {
    // Same teardown as dropping the members one by one. Not close(): a destructor must not throw.
    members_.clear();
}

void UnitMeshGroup::close() {
    for (auto& [chip_id, mesh] : members_) {
        mesh->close();
    }
    members_.clear();
}

MeshDevice& UnitMeshGroup::at(ChipId chip_id) const {
    auto it = members_.find(chip_id);
    TT_FATAL(it != members_.end(), "Chip {} is not a member of this UnitMeshGroup", chip_id);
    return *it->second;
}

std::vector<ChipId> UnitMeshGroup::chip_ids() const {
    std::vector<ChipId> ids;
    ids.reserve(members_.size());
    for (const auto& [chip_id, mesh] : members_) {
        ids.push_back(chip_id);
    }
    return ids;
}

}  // namespace tt::tt_metal::distributed
