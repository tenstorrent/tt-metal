// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"

////////////////////////////////////////////////////////////////////////////////
// DEVICE_PRINT tests for DRAM programmable cores (DRISC, Blackhole only). What DRISC prints is checked with every
// other RISC type by the *OnEveryRisc tests in test_print_output.cpp.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

class DevicePrintDramFixture : public DevicePrintFixture {};

#define DRAM_SKIP_GUARDS()                                                                                  \
    do {                                                                                                    \
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();                                     \
        if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {                               \
            log_info(tt::LogTest, "Skipping: DRAM programmable cores not available on this architecture."); \
            GTEST_SKIP();                                                                                   \
        }                                                                                                   \
    } while (0)

// Each DRAM view's NOC0 worker endpoint runs the syseng firmware, which sits where a DPRINT buffer would go, so the
// print server must neither poll it nor write its magic there.
TEST_F(DevicePrintDramFixture, DramPrintSkipsSysengEndpoints) {
    DRAM_SKIP_GUARDS();
    for (auto& mesh_device : this->devices_) {
        auto& context = mesh_device->impl().metal_context();
        const auto& cluster = mesh_device->impl().metal_env().get_cluster();
        for (IDevice* device : mesh_device->get_devices()) {
            const auto& soc_desc = cluster.get_soc_desc(device->id());
            std::unordered_set<CoreCoord> noc0_endpoints;
            for (int dram_view = 0; dram_view < static_cast<int>(soc_desc.get_num_dram_views()); dram_view++) {
                noc0_endpoints.insert(soc_desc.get_preferred_worker_core_for_dram_view(dram_view, /*noc=*/0));
            }

            size_t num_dram_print_cores = 0;
            for (const auto& print_core : context.dprint_server()->get_print_cores(device->id())) {
                if (print_core.type != CoreType::DRAM) {
                    continue;
                }
                num_dram_print_cores++;
                const CoreCoord virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
                    device->id(), print_core.coord, CoreType::DRAM);
                EXPECT_FALSE(noc0_endpoints.contains(virtual_core))
                    << "Device " << device->id() << " DRAM print core " << print_core.coord.str() << " (virtual "
                    << virtual_core.str() << ") is a syseng-owned NOC0 endpoint";
            }
            EXPECT_GT(num_dram_print_cores, 0u);
        }
    }
}

#undef DRAM_SKIP_GUARDS
