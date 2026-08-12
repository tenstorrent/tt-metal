// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>

namespace tt::tt_metal::distributed {

class MeshDevice;
class MeshEventImpl;
class MeshCoordinateRange;

class MeshEvent {
public:
    MeshEvent(const MeshEvent& other);
    MeshEvent& operator=(const MeshEvent& other);
    MeshEvent(MeshEvent&& other) noexcept;
    MeshEvent& operator=(MeshEvent&& other) noexcept;
    ~MeshEvent();

    MeshDevice* device() const;

    // debug/test/internal usage.
    MeshEventImpl& impl();
    const MeshEventImpl& impl() const;

    friend std::ostream& operator<<(std::ostream& os, const MeshEvent& event);

private:
    explicit MeshEvent(std::unique_ptr<MeshEventImpl> impl);

    std::unique_ptr<MeshEventImpl> pimpl_;

    friend MeshEvent make_mesh_event(
        uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);
};

}  // namespace tt::tt_metal::distributed
