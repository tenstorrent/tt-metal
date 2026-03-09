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

    llrt::RunTimeOptions& get_rtoptions() { return env_.impl().get_rtoptions(); }
    tt::Cluster& get_cluster() { return env_.impl().get_cluster(); }
    const Hal& get_hal() const { return env_.impl().get_hal(); }
    void acquire() { env_.impl().acquire(); }
    void release() { env_.impl().release(); }
    bool check_use_count_zero() const { return env_.impl().check_use_count_zero(); }

private:
    MetalEnv& env_;
};

}  // namespace tt::tt_metal
