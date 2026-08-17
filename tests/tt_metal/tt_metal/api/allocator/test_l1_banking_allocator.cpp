// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
#include <tt-metalium/graph_tracking.hpp>
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

// Hooks that block only deallocation. Models the asymmetry a real graph capture produces:
// ProcessorHooks::hook_allocate/hook_deallocate both return the same `do_block` flag, and
// GraphProcessor::begin_capture(RunMode::NO_DISPATCH) flips it mid-run, so a buffer allocated
// before the capture is registered with the allocator but its free is suppressed inside it.
class DeallocateBlockingHooks : public tt::tt_metal::IGraphHooks {
public:
    bool hook_allocate(const tt::tt_metal::Buffer*) override { return false; }
    bool hook_deallocate(tt::tt_metal::Buffer*) override { return true; }
    bool hook_program(tt::tt_metal::Program*) override { return false; }
    bool hook_write_to_device(const tt::tt_metal::Buffer*) override { return false; }
    bool hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer*) override { return false; }
    bool hook_read_from_device(tt::tt_metal::Buffer*) override { return false; }
    bool hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer*) override { return false; }
};

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

// A buffer whose allocation was real is registered in the allocator's tracking set. If a
// graph-capture hook then suppresses the free path, nothing else ever removes it, and once the
// Buffer is destroyed the set holds a dangling pointer -- which ~AllocatorImpl dereferences when
// it detaches buffers that outlive it. Deregistration must therefore happen regardless of the
// hook. (This intentionally leaks the buffer's banks until device close; suppressing the free is
// the hook's business. GraphTracker also logs "Can't hook deallocation of a buffer which
// allocation wasn't hooked" here -- that warning is the asymmetry being exercised.)
TEST_F(MeshDeviceSingleCardBufferFixture, HookSuppressedDeallocateUntracksBuffer) {
    auto* device = this->devices_[0]->get_devices()[0];
    const auto& allocator = device->allocator();

    const size_t baseline = allocator->get_allocated_buffers().size();

    constexpr DeviceAddr kBufferSize = 2048;
    auto buffer = Buffer::create(device, kBufferSize, kBufferSize, BufferType::DRAM);
    ASSERT_EQ(allocator->get_allocated_buffers().size(), baseline + 1)
        << "buffer was not registered with the allocator; the rest of this test proves nothing";

    auto hooks = std::make_shared<unit_tests::test_l1_banking_allocator::DeallocateBlockingHooks>();
    ASSERT_TRUE(GraphTracker::instance().add_hook(hooks));
    buffer.reset();
    GraphTracker::instance().clear_hook();

    EXPECT_EQ(allocator->get_allocated_buffers().size(), baseline)
        << "hook-suppressed deallocation left a dangling Buffer* in the allocator's tracking set";
}

}  // namespace tt::tt_metal
