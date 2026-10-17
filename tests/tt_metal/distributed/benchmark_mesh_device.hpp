// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::distributed::test {

// The whole mesh, opened on first use and shared by every benchmark in the binary, since it can only be open once.
inline MeshDevice& benchmark_mesh_device() {
    static std::shared_ptr<MeshDevice> device = MeshDevice::create(MeshDeviceConfig(std::nullopt));
    return *device;
}

}  // namespace tt::tt_metal::distributed::test
