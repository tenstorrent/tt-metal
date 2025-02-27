#include "t3000_test_device.hpp"

using namespace tt;
using namespace tt_metal;

using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshShape;

T3000TestDevice::T3000TestDevice() : device_open(false) {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
    }
    arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

    num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    if (arch_ == tt::ARCH::WORMHOLE_B0 and num_devices_ == 8 and tt::tt_metal::GetNumPCIeDevices() == 4) {
        auto config = MeshDeviceConfig{.mesh_shape = MeshShape{2, 4}};
        // creates a mesh device with two command queues
        mesh_device_ = MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 2, DispatchCoreConfig{DispatchCoreType::ETH});
        std::vector<chip_id_t> ids(num_devices_, 0);
        std::iota(ids.begin(), ids.end(), 0);

    } else {
        TT_THROW("This suite can only be run on T3000 Wormhole devices");
    }
    device_open = true;
}

T3000TestDevice::~T3000TestDevice() {
    if (device_open) {
        TearDown();
    }
}

void T3000TestDevice::TearDown() {
    device_open = false;
    mesh_device_->close();
}
