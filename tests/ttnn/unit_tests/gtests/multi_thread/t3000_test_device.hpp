#pragma once

#include "ttnn/async_runtime.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/mesh_device.hpp>

class T3000TestDevice {
public:
    T3000TestDevice();
    ~T3000TestDevice();
    void TearDown();

    tt::ARCH arch_;
    size_t num_devices_;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;

private:
    bool device_open;
};
