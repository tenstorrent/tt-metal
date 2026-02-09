// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "context/experimental/context.hpp"
#include <memory>

namespace tt {
class Cluster;
}

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal {
class Hal;
}

namespace tt::tt_metal::experimental {

class MetaliumObject;

// Base class for runtime backends.
class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;
    virtual void initialize(const ContextDescriptor& descriptor) = 0;
    virtual void teardown() = 0;
};

// Silicon backend - uses shared MetaliumObject
class SiliconRuntimeBackend : public RuntimeBackend {
public:
    explicit SiliconRuntimeBackend(std::shared_ptr<MetaliumObject> metalium_object);
    ~SiliconRuntimeBackend() override = default;

    void initialize(const ContextDescriptor& descriptor) override;
    void teardown() override;

    std::shared_ptr<MetaliumObject> get_metalium_object() { return metalium_object_; }

private:
    std::shared_ptr<MetaliumObject> metalium_object_;
};

// Mock backend - creates and owns its own Cluster/HAL/RuntimeOptions configured for mock
class MockDeviceRuntimeBackend : public RuntimeBackend {
public:
    MockDeviceRuntimeBackend() = default;
    ~MockDeviceRuntimeBackend() override = default;

    void initialize(const ContextDescriptor& descriptor) override;
    void teardown() override;

    // Access to mock-specific objects
    tt::Cluster& cluster() { return *cluster_; }
    tt::tt_metal::Hal& hal() { return *hal_; }
    tt::llrt::RunTimeOptions& rtoptions() { return *rtoptions_; }

private:
    std::unique_ptr<tt::llrt::RunTimeOptions> rtoptions_;
    std::shared_ptr<tt::tt_metal::Hal> hal_;
    std::shared_ptr<tt::Cluster> cluster_;
};

}  // namespace tt::tt_metal::experimental
