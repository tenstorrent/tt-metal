#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>
#include "context/metal_context.hpp"
#include "data_types.hpp"
#include "kernel_types.hpp"
#include "tt_metal.hpp"

TEST(Tunneling, LiteFabric) {
    auto devices = tt::tt_metal::detail::CreateDevices({0, 1});
    auto device = devices[0];
    auto pgm = tt::tt_metal::CreateProgram();

    auto& mc = tt::tt_metal::MetalContext::instance();

    for (auto core : mc.get_control_plane().get_active_ethernet_cores(device->id())) {
        auto kernel = tt::tt_metal::CreateKernel(
            pgm,
            "tests/tt_metal/tt_metal/tunneling/tunnel.cpp",
            core,
            tt::tt_metal::EthernetConfig{
                .eth_mode = tt::tt_metal::SENDER,
                .compile_args = {},
                .defines = {},
            });
        log_info(tt::LogTest, "Core {}", core.str());
        break;
    }

    tt::tt_metal::detail::CloseDevices(devices);
}
