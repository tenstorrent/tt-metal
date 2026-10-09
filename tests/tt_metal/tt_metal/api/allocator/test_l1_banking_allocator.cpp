// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <tt-metalium/allocator.hpp>
#include "llrt/core_descriptor.hpp"
#include <tt-metalium/host_api.hpp>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include "llrt/metal_soc_descriptor.hpp"
#include "impl/context/metal_context.hpp"
#include "internal/tt-2xx/quasar/noc/att/att_address.h"
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"

using namespace tt::tt_metal;
namespace unit_tests::test_l1_banking_allocator {

uint64_t get_alloc_limit(distributed::MeshDevice& mesh_device) {
    const metal_SocDescriptor& soc_desc =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(mesh_device.get_device_ids()[0]);
    uint32_t l1_unreserved_base = mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t interleaved_l1_bank_size = soc_desc.worker_l1_size - l1_unreserved_base;
    return interleaved_l1_bank_size;
}

}  // namespace unit_tests::test_l1_banking_allocator

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, TestL1BuffersAllocatedTopDown) {
    std::vector<uint32_t> alloc_sizes = {32 * 1024, 64 * 1024, 128 * 1024};
    size_t total_size_bytes = 0;

    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(this->device());

    std::vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
    int alloc_size_idx = 0;
    uint32_t total_buffer_size = 0;
    while (total_size_bytes < alloc_limit) {
        uint32_t buffer_size = alloc_sizes.at(alloc_size_idx);
        alloc_size_idx = (alloc_size_idx + 1) % alloc_sizes.size();
        if (total_buffer_size + buffer_size >= alloc_limit) {
            break;
        }
        distributed::DeviceLocalBufferConfig local_config{.page_size = buffer_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        std::shared_ptr<distributed::MeshBuffer> buffer =
            distributed::MeshBuffer::create(buffer_config, local_config, &this->device());
        buffers.emplace_back(std::move(buffer));
        total_buffer_size += buffer_size;
        EXPECT_EQ(buffers.back()->address(), this->device().l1_size_per_core() - total_buffer_size);
    }
    buffers.clear();
}

TEST_F(UnitMeshFixture, TestL1BuffersDoNotGrowBeyondBankSize) {
    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(this->device());
    distributed::DeviceLocalBufferConfig local_config{.page_size = alloc_limit + 64, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = alloc_limit + 64};
    EXPECT_ANY_THROW(auto buffer = distributed::MeshBuffer::create(buffer_config, local_config, &this->device()));
}

// ---- L1 bank order with NoC address translation tables -------------------------------------------------------------

// With the ATT enabled (Quasar, TT_METAL_NOC_ATT) the allocator numbers L1 banks in plain row-major core order instead
// of shuffling them, which is the tables' own order: the map lists the workers by logical row-major position, so bank i
// is the i-th endpoint. Hardware that walks banks (the address generator's banking loop) relies on this.
TEST_F(QuasarMeshDeviceSingleCardFixture, L1BanksAreInRowMajorOrderWithAtt) {
    if (std::getenv("TT_METAL_NOC_ATT") == nullptr) {
        GTEST_SKIP() << "NoC address translation tables are not enabled (TT_METAL_NOC_ATT)";
    }
    auto& device = *devices_.at(0);
    const auto& allocator = *device.allocator();
    const uint32_t num_banks = allocator.get_num_banks(BufferType::L1);
    ASSERT_GT(num_banks, 0u);

    std::vector<CoreCoord> cores;
    cores.reserve(num_banks);
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        cores.push_back(allocator.get_logical_core_from_bank_id(bank));
    }
    // Row-major: rows (y) outermost, columns (x) within a row.
    for (uint32_t bank = 1; bank < num_banks; ++bank) {
        const CoreCoord& prev = cores[bank - 1];
        const CoreCoord& cur = cores[bank];
        EXPECT_TRUE(cur.y > prev.y || (cur.y == prev.y && cur.x > prev.x))
            << "bank " << bank << " (core " << cur.str() << ") does not follow bank " << bank - 1 << " (core "
            << prev.str() << ") in row-major order";
    }

    // End to end against the map itself: the ATT selectors of the banks' cores are consecutive and ascending.
    const std::string_view map_name = std::getenv("TT_METAL_NOC_ATT");
    const noc_att::MapData* map = nullptr;
    if (map_name == "grendel_qsr1") {
        map = &grendel_qsr1_att_config::MAP;
    } else if (map_name == "quasar_aether_2x3") {
        map = &quasar_aether_2x3_att_config::MAP;
    }
    ASSERT_NE(map, nullptr) << "unknown ATT map " << map_name;
    std::vector<uint32_t> selectors;
    selectors.reserve(num_banks);
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        const CoreCoord virtual_core = device.worker_core_from_logical_core(cores[bank]);
        const noc_att::ResolvedTile tile =
            noc_att::resolve(*map, noc_att::Address::worker(virtual_core.x, virtual_core.y, 0));
        ASSERT_TRUE(tile.valid) << "bank " << bank << " core " << virtual_core.str() << " is not in the ATT map";
        selectors.push_back(tile.selector);
    }
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        EXPECT_EQ(selectors[bank], selectors[0] + bank) << "bank " << bank;
    }
}

}  // namespace tt::tt_metal
