// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Host half of this rank's virtual cables (experimental): E2H -> H2H -> H2E and credits back.
#pragma once

#include <memory>
#include <string>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental {

class VirtualCableBridge {
public:
    // Collective, after fabric is up; same verdict on every rank. Null with `err` set on failure.
    static std::unique_ptr<VirtualCableBridge> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, std::string& err);

    // Collective: every rank destroys it together (mesh close).
    ~VirtualCableBridge();
    VirtualCableBridge(const VirtualCableBridge&) = delete;
    VirtualCableBridge& operator=(const VirtualCableBridge&) = delete;

    // Empty while healthy.
    std::string first_error() const;
    // Per-end counters, for logs.
    std::string describe() const;

private:
    VirtualCableBridge();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
