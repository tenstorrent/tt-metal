// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "llrt/core_descriptor.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// Regression test for the ETH dispatch core descriptor on chips whose logical ethernet grid is smaller than the
// descriptor's `dispatch_cores` list.
//
// UMD numbers a chip's logical ETH cores over its unharvested ethernet channels only. Blackhole always harvests 2
// of its 14 ethernet channels, so every Blackhole exposes the logical ETH cores (0,0)..(0,11), while
// blackhole_140_arch_eth_dispatch.yaml lists (0,0)..(0,13). Without the existence check in
// MetalEnvImpl::get_core_descriptor_config, the two missing cores reach L1BankingAllocator::generate_config, whose
// logical-to-virtual translation throws
//     "No core coordinate found at location: (0, 12, ETH, LOGICAL)"
// and DispatchCoreType::ETH cannot open the device (the same throw is reported in GitHub #49701).
//
// The invariant checked here: every logical dispatch core the ETH core descriptor yields for a chip is one of that
// chip's logical ETH cores (a key of metal_SocDescriptor::logical_eth_core_to_chan_map). This is a host-side query
// on the default MetalContext. It opens no device and does not use ETH dispatch itself (that is what the fix
// enables), so it fails on an unpatched harvested Blackhole whatever dispatch core type the test run uses.
class EthDispatchCoreDescriptorTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            GTEST_SKIP() << "The core descriptor reserves no dispatch cores in slow dispatch mode";
        }
        env_ = &MetalEnvAccessor(MetalContext::instance().get_env()).impl();
        const tt::ARCH arch = env_->get_cluster().arch();
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "No ETH dispatch core descriptor for arch " << tt::arch_to_str(arch);
        }
        if (env_->get_rtoptions().get_simulator_enabled()) {
            GTEST_SKIP() << "Simulator core descriptors have no ETH dispatch cores";
        }
    }

    MetalEnvImpl* env_ = nullptr;
};

TEST_F(EthDispatchCoreDescriptorTest, EthDispatchCoresAreLogicalEthCoresOfTheChip) {
    Cluster& cluster = env_->get_cluster();
    const DispatchCoreConfig eth_dispatch_core_config(DispatchCoreType::ETH);
    bool checked_a_chip = false;
    for (const ChipId device_id : cluster.all_chip_ids()) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device_id);
        const auto& logical_eth_cores = soc_desc.logical_eth_core_to_chan_map;
        if (logical_eth_cores.empty()) {
            // A chip without ethernet cores (e.g. Blackhole p100) cannot use ETH dispatch at all.
            continue;
        }
        checked_a_chip = true;
        for (const uint8_t num_hw_cqs : {uint8_t{1}, uint8_t{2}}) {
            const std::vector<CoreCoord>& dispatch_cores =
                tt::get_logical_dispatch_cores(*env_, device_id, num_hw_cqs, eth_dispatch_core_config);
            EXPECT_FALSE(dispatch_cores.empty())
                << "The ETH core descriptor yields no dispatch cores for device " << device_id << " with "
                << static_cast<int>(num_hw_cqs) << " HW CQ(s)";
            for (const CoreCoord& core : dispatch_cores) {
                EXPECT_TRUE(logical_eth_cores.contains(core))
                    << "ETH dispatch core " << core.str() << " from the core descriptor (device " << device_id << ", "
                    << static_cast<int>(num_hw_cqs) << " HW CQ(s)) is not a logical ETH core of the chip, "
                    << "which has " << logical_eth_cores.size() << " logical ETH cores (eth harvesting mask 0x"
                    << std::hex << soc_desc.harvesting_masks.eth_harvesting_mask << std::dec << ")";
            }
        }
    }
    if (!checked_a_chip) {
        GTEST_SKIP() << "No chip in the cluster has ethernet cores";
    }
}

}  // namespace tt::tt_metal
