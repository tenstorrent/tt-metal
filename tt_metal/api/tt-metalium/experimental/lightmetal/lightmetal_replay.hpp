// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace tt::tt_metal {

class IDevice;

namespace experimental::lightmetal {

class LightMetalBinary;
namespace detail {
class LightMetalReplayImpl;
}

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob and transfers ownership of the blob.
    explicit LightMetalReplay(LightMetalBinary&& binary, IDevice* device = nullptr);
    LightMetalReplay(LightMetalReplay&&) noexcept;
    ~LightMetalReplay();

    LightMetalReplay(const LightMetalReplay&) = delete;
    LightMetalReplay& operator=(const LightMetalReplay&) = delete;

    LightMetalReplay& operator=(LightMetalReplay&&) noexcept;

    // Run the stored LightMetal binary by looping over all commands, and executing them.
    // Returns true if passed.
    bool run();

private:
    std::unique_ptr<detail::LightMetalReplayImpl> pimpl_;
};

}  // namespace experimental::lightmetal

}  // namespace tt::tt_metal
