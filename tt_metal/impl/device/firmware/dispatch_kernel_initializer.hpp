// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "firmware_initializer.hpp"

namespace tt::tt_metal {

class DispatchKernelInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key() { return InitializerKey::Dispatch; }

    using FirmwareInitializer::FirmwareInitializer;

    void init(const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& init_done) override;
    void configure() override;
    void teardown() override;
    // Returns true if fast dispatch is enabled and has been configured
    bool is_initialized() const override;

private:
    void compile_dispatch_kernels();

    void init_device_command_queues();

    void terminate_command_queues();

    void wait_for_dispatch_cores() const;

    void process_termination_signals() const;

    bool using_fast_dispatch() const;

    std::vector<Device*> devices_;
    bool initialized_ = false;
};

}  // namespace tt::tt_metal
