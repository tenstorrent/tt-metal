// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/experimental/context/metalium_env.hpp>

namespace tt::tt_metal {

class MetaliumEnv::MetaliumEnvImpl {
public:
    explicit MetaliumEnvImpl(MetaliumEnvDescriptor descriptor);
    ~MetaliumEnvImpl();

    llrt::RunTimeOptions& get_rtoptions();
    const Hal& get_hal();
    Cluster& get_cluster();
    const MetaliumEnvDescriptor& get_descriptor() const;
    void destroy_early() noexcept;
    bool is_initialized() const;

private:
    bool initialized_ = false;
    MetaliumEnvDescriptor descriptor_;

    std::unique_ptr<llrt::RunTimeOptions> rtoptions_;
    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<Hal> hal_;

    void initialize_base_objects();
    void verify_fw_capabilities();
};

}  // namespace tt::tt_metal
