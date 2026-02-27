// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "metalium_env_impl.hpp"

namespace tt::tt_metal {

// Friend class of MetaliumEnv. For internal use to fully access the MetaliumEnv object.
class MetaliumEnvAccessor {
public:
    MetaliumEnvAccessor(MetaliumEnv& metalium_env) noexcept : metalium_env_(&metalium_env) {}
    // Careful. Const will be casted away.
    MetaliumEnvAccessor(const MetaliumEnv& metalium_env) noexcept :
        metalium_env_(const_cast<MetaliumEnv*>(&metalium_env)) {}
    // Takes ownership of the MetaliumEnv. Useful for testing.
    MetaliumEnvAccessor(std::unique_ptr<MetaliumEnv> metalium_env) noexcept :
        owned_env_(std::move(metalium_env)), metalium_env_(owned_env_.get()) {}
    ~MetaliumEnvAccessor() = default;

    MetaliumEnv& get_metalium_env() { return *metalium_env_; }

    llrt::RunTimeOptions& get_rtoptions() { return metalium_env_->impl().get_rtoptions(); }

    tt::Cluster& get_cluster() { return metalium_env_->impl().get_cluster(); }

    const tt::tt_metal::Hal& get_hal() const { return metalium_env_->impl().get_hal(); }

private:
    std::unique_ptr<MetaliumEnv> owned_env_;
    MetaliumEnv* metalium_env_ = nullptr;
};

}  // namespace tt::tt_metal
