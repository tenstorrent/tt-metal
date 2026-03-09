// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <set>
#include <atomic>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

namespace tt::tt_metal {

class MetalEnv::MetalEnvImpl {
public:
    explicit MetalEnvImpl(MetalEnvDescriptor descriptor);
    ~MetalEnvImpl();

    llrt::RunTimeOptions& get_rtoptions();
    const Hal& get_hal();
    Cluster& get_cluster();
    const MetalEnvDescriptor& get_descriptor() const;

    bool check_use_count_zero() const;

    void acquire();
    void release();

private:
    MetalEnvDescriptor descriptor_;

    std::unique_ptr<llrt::RunTimeOptions> rtoptions_;
    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<Hal> hal_;

    std::atomic<int> use_count_{0};

    void initialize_base_objects();
    void verify_fw_capabilities();

    static std::mutex s_registry_mutex_;
    static std::set<MetalEnvImpl*> s_registry_;
    static std::once_flag s_atfork_registered_;
    static void prefork_check_all();
};

}  // namespace tt::tt_metal
