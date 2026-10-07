// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

class MeshCommandQueue;
class MeshDevice;
class MeshEventImpl;

class MeshEvent {
public:
    MeshEvent(uint32_t id, MeshCommandQueue& cq, const MeshCoordinateRange& device_range);
    [[deprecated(
        "Use MeshEvent(uint32_t, MeshCommandQueue&, const MeshCoordinateRange&) instead. MeshEvent(uint32_t, "
        "MeshDevice*, uint32_t, const MeshCoordinateRange&) will be removed after September 9th, 2026.")]]
    MeshEvent(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);
    explicit MeshEvent(MeshEventImpl impl);

    MeshEvent(const MeshEvent& other);
    MeshEvent& operator=(const MeshEvent& other);
    MeshEvent(MeshEvent&& other) noexcept;
    MeshEvent& operator=(MeshEvent&& other) noexcept;
    ~MeshEvent();

    MeshDevice* device() const;

    MeshEventImpl& impl();
    const MeshEventImpl& impl() const;

    friend std::ostream& operator<<(std::ostream& os, const MeshEvent& event);

private:
    std::unique_ptr<MeshEventImpl> impl_;
};

}  // namespace tt::tt_metal::distributed
