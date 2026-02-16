// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include <llrt/rtoptions.hpp>

#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

class ContextDescriptor;
class Device;

enum class InitializerKey {
    Profiler,
    CommandQueue,
    Fabric,
    Dispatch,
};

class FirmwareInitializer {
public:
    explicit FirmwareInitializer(std::shared_ptr<const ContextDescriptor> descriptor);

    virtual ~FirmwareInitializer() = default;

    FirmwareInitializer(const FirmwareInitializer&) = delete;
    FirmwareInitializer& operator=(const FirmwareInitializer&) = delete;
    FirmwareInitializer(FirmwareInitializer&&) = delete;
    FirmwareInitializer& operator=(FirmwareInitializer&&) = delete;

    virtual void init(const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& init_done) = 0;

    // This is called after all init calls have completed
    virtual void configure() = 0;

    virtual void teardown() = 0;

    // This is called after all teardown calls have completed and devices have been closed
    virtual void post_teardown();

    virtual bool is_initialized() const = 0;

protected:
    const Hal& hal_;
    Cluster& cluster_;
    const llrt::RunTimeOptions& rtoptions_;
    std::shared_ptr<const ContextDescriptor> descriptor_;
};

}  // namespace tt::tt_metal
