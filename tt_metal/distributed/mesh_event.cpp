// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_event.hpp>

#include <tt_stl/assert.hpp>
#include "mesh_device.hpp"
#include "mesh_event_impl.hpp"

namespace tt::tt_metal::distributed {

MeshEventImpl::MeshEventImpl(
    uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) :
    id_(id), device_(device), mesh_cq_id_(mesh_cq_id), device_range_(device_range) {}

MeshEvent::MeshEvent(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) :
    MeshEvent(MeshEventImpl(id, device, mesh_cq_id, device_range)) {}

MeshEvent::MeshEvent(MeshEventImpl impl) : impl_(std::make_unique<MeshEventImpl>(std::move(impl))) {}

MeshEvent::MeshEvent(const MeshEvent& other) :
    impl_(other.impl_ ? std::make_unique<MeshEventImpl>(*other.impl_) : nullptr) {}

MeshEvent& MeshEvent::operator=(const MeshEvent& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<MeshEventImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

MeshEvent::MeshEvent(MeshEvent&& other) noexcept = default;
MeshEvent& MeshEvent::operator=(MeshEvent&& other) noexcept = default;
MeshEvent::~MeshEvent() = default;

MeshDevice* MeshEvent::device() const { return impl().device(); }

MeshEventImpl& MeshEvent::impl() {
    TT_FATAL(impl_ != nullptr, "MeshEvent is in a moved-from state.");
    return *impl_;
}
const MeshEventImpl& MeshEvent::impl() const {
    TT_FATAL(impl_ != nullptr, "MeshEvent is in a moved-from state.");
    return *impl_;
}

std::ostream& operator<<(std::ostream& os, const MeshEvent& event) {
    os << "MeshEvent(id=" << event.impl().id() << ", device_id=" << event.device()->id()
       << ", mesh_cq_id=" << event.impl().mesh_cq_id() << ", device_range=" << event.impl().device_range() << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
