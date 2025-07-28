// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt {
namespace tt_metal {
namespace distributed {
class MeshDevice;
class MeshEventImpl;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {

class MeshEvent {
public:
    MeshEvent(std::unique_ptr<MeshEventImpl> impl);
    ~MeshEvent();

    MeshEvent(const MeshEvent& other);
    MeshEvent& operator=(const MeshEvent& other);

    MeshEvent(MeshEvent&& other) noexcept;
    MeshEvent& operator=(MeshEvent&& other) noexcept;

    uint32_t id() const;
    MeshDevice* device() const;
    uint32_t mesh_cq_id() const;
    const MeshCoordinateRange& device_range() const;

    friend std::ostream& operator<<(std::ostream& os, const MeshEvent& event);

private:
    std::unique_ptr<MeshEventImpl> pimpl_;
};

}  // namespace tt::tt_metal::distributed
