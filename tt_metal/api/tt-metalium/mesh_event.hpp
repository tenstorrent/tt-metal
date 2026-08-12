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
