// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "firmware_initializer.hpp"

namespace tt::tt_metal {

struct ProfilerStateManager;

class ProfilerInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key = InitializerKey::Profiler;

    ProfilerInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor,
        bool skip_remote_devices,
        ProfilerStateManager* profiler_state_manager);

    void init(const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& init_done) override;
    void configure() override;
    void teardown() override;
    void post_teardown() override;
    bool is_initialized() const override;

private:
    [[maybe_unused]] bool skip_remote_devices_;
    [[maybe_unused]] ProfilerStateManager* profiler_state_manager_;
    [[maybe_unused]] std::vector<Device*> devices_;
    bool initialized_ = false;
};

}  // namespace tt::tt_metal
