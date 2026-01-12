// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
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

using namespace tt::tt_metal;
namespace unit_tests::test_l1_banking_allocator {

uint64_t get_alloc_limit(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices()[0];
    const metal_SocDescriptor& soc_desc =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    uint32_t l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t interleaved_l1_bank_size = soc_desc.worker_l1_size - l1_unreserved_base;
    return interleaved_l1_bank_size;
}

}  // namespace unit_tests::test_l1_banking_allocator

namespace tt::tt_metal {

TEST_F(MeshDeviceSingleCardBufferFixture, TestL1BuffersAllocatedTopDown) {
    std::vector<uint32_t> alloc_sizes = {32 * 1024, 64 * 1024, 128 * 1024};
    size_t total_size_bytes = 0;

    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(this->devices_[0]);

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
            distributed::MeshBuffer::create(buffer_config, local_config, this->devices_[0].get());
        buffers.emplace_back(std::move(buffer));
        total_buffer_size += buffer_size;
        EXPECT_EQ(buffers.back()->address(), this->devices_[0]->l1_size_per_core() - total_buffer_size);
    }
    buffers.clear();
}

TEST_F(MeshDeviceSingleCardBufferFixture, TestL1BuffersDoNotGrowBeyondBankSize) {
    uint64_t alloc_limit = unit_tests::test_l1_banking_allocator::get_alloc_limit(this->devices_[0]);
    distributed::DeviceLocalBufferConfig local_config{.page_size = alloc_limit + 64, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = alloc_limit + 64};
    EXPECT_ANY_THROW(
        auto buffer = distributed::MeshBuffer::create(buffer_config, local_config, this->devices_[0].get()));
}

}  // namespace tt::tt_metal
