// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <tt-umd/types/arch.hpp>

namespace tt::tt_metal {

class MetalEnvImpl;

// Hardware properties of the devices in a MetalEnv, e.g. architecture, memory sizes and alignments.
//
// This is a lightweight, copyable, non-owning view. It is valid only while the MetalEnv it was obtained from is
// alive. Values are read from the environment on each call.
class MetalDeviceProperties {
public:
    /// @return Architecture of the devices.
    tt::ARCH arch() const;

    /// @return Human-readable name of the architecture of the devices.
    std::string arch_name() const;

    /// @return Size in bytes of each Tensix core's node local mem.
    uint32_t node_local_mem_size() const;

    /// @return Required address alignment in bytes for DRAM allocations.
    uint32_t dram_alignment() const;

    /// @return Required address alignment in bytes for node local mem allocations.
    uint32_t node_local_mem_alignment() const;

    /// @return Maximum number of dataflow buffers per core.
    uint32_t num_dataflow_buffers() const;

    /// @return Maximum usable node local mem size in bytes when the ring-buffer size is 0.
    uint32_t max_worker_node_local_mem_unreserved_size() const;

    /// @return Representable SFPU epsilon value.
    float sfpu_eps() const;

    /// @return Representable SFPU NaN value.
    float sfpu_nan() const;

    /// @return Representable SFPU Infinity value.
    float sfpu_inf() const;

private:
    const MetalEnvImpl* impl_;

public:
    // Internal use: MetalEnvImpl is not part of the public API, so this can only be called from within tt_metal.
    // Obtain an instance through MetalEnv::get_device_properties().
    explicit MetalDeviceProperties(const MetalEnvImpl& impl) : impl_(&impl) {}
};

}  // namespace tt::tt_metal
