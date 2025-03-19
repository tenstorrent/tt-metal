// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dispatch_core_common.hpp>

#include <unordered_set>

namespace tt::tt_metal {

class MetalContext {
public:
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext& operator=(MetalContext&& other) noexcept = delete;
    MetalContext(const MetalContext&) = delete;
    MetalContext(MetalContext&& other) noexcept = delete;

    static void initialize(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) noexcept;
    static MetalContext& instance();

private:
    MetalContext(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs);
    ~MetalContext();

    uint8_t num_hw_cqs_;
    DispatchCoreConfig dispatch_core_config_;

    // Used to track which FW has been build already
    std::unordered_set<uint32_t> firmware_built_keys_;

    static MetalContext* _inst;
};

}  // namespace tt::tt_metal
