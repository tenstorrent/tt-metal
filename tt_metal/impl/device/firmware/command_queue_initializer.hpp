// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "firmware_initializer.hpp"

namespace tt::tt_metal {

class CommandQueueInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key() { return InitializerKey::CommandQueue; }

    CommandQueueInitializer(std::shared_ptr<const ContextDescriptor> descriptor, bool skip_remote_devices);

    void init(const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& init_done) override;
    void configure() override;
    void teardown() override;
    bool is_initialized() const override;

private:
    void initialize_host(Device* dev) const;

    bool using_fast_dispatch() const;

    bool skip_remote_devices_;
    std::vector<Device*> devices_;
    bool initialized_ = false;
};

}  // namespace tt::tt_metal
