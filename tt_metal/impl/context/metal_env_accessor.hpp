// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal_env_impl.hpp"

namespace tt::tt_metal {

// Friend class of MetalEnv for internal use.
class MetalEnvAccessor {
public:
    explicit MetalEnvAccessor(MetalEnv& env) noexcept : env_(env) {}

    MetalEnvImpl& impl() { return env_.impl(); }
    const MetalEnvImpl& impl() const { return env_.impl(); }

private:
    MetalEnv& env_;
};

}  // namespace tt::tt_metal
