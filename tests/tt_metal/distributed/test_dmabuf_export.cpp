// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <span>
#include <unordered_map>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dmabuf_export.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/impl/dispatch/host_device_transfer.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using namespace tt::tt_metal::experimental::dmabuf_export;

using DmabufExportTest = GenericMeshDeviceFixture;

// Not a multiple of the DRAM alignment on any architecture, so pages carry padding in the bank.
constexpr uint32_t kPageSize = 1056;

std::shared_ptr<MeshBuffer> create_interleaved_dram_buffer(MeshDevice& mesh_device, uint32_t num_pages) {
    const DeviceLocalBufferConfig local_config{.page_size = kPageSize, .buffer_type = BufferType::DRAM};
    const ReplicatedBufferConfig global_config{.size = static_cast<DeviceAddr>(num_pages) * kPageSize};
    return MeshBuffer::create(global_config, local_config, &mesh_device);
}

TEST_F(DmabufExportTest, DramBankExtentsMatchDeviceLayout) {
    const uint32_t num_banks = mesh_device_->allocator()->get_num_banks(BufferType::DRAM);
    const uint32_t num_pages = (num_banks * 3) + 5;
    auto buffer = create_interleaved_dram_buffer(*mesh_device_, num_pages);

    std::vector<uint32_t> src(num_pages * kPageSize / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 0);
    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), buffer, src, /*blocking=*/true);
    const auto* src_bytes = reinterpret_cast<const uint8_t*>(src.data());

    const std::vector<DramBankExtent> extents = get_dram_bank_extents(*buffer);
    ASSERT_FALSE(extents.empty());

    std::unordered_map<MeshCoordinate, uint32_t> pages_per_device;
    for (const auto& extent : extents) {
        EXPECT_EQ(extent.page_stride, num_banks);
        EXPECT_EQ(extent.address, buffer->address());
        pages_per_device[extent.device_coord] += extent.num_pages;

        IDevice* device = buffer->get_device_buffer(extent.device_coord)->device();
        std::vector<uint8_t> bank_bytes(extent.size());
        slow_dispatch::ReadFromDeviceDRAMChannel(
            *device, static_cast<int>(extent.bank_id), static_cast<uint32_t>(extent.address), std::span(bank_bytes));

        for (uint32_t k = 0; k < extent.num_pages; ++k) {
            const uint32_t buffer_page = extent.first_page + (k * extent.page_stride);
            EXPECT_EQ(
                std::memcmp(
                    bank_bytes.data() + (k * extent.aligned_page_size),
                    src_bytes + (static_cast<size_t>(buffer_page) * kPageSize),
                    kPageSize),
                0)
                << "device " << extent.device_coord << ", bank " << extent.bank_id << ", page " << buffer_page;
        }
    }
    for (const auto& [coord, count] : pages_per_device) {
        EXPECT_EQ(count, num_pages) << "device " << coord;
    }
}

TEST_F(DmabufExportTest, MaxExportSizeStaysWithinExtent) {
    auto buffer = create_interleaved_dram_buffer(*mesh_device_, /*num_pages=*/64);
    for (const auto& extent : get_dram_bank_extents(*buffer)) {
        const DeviceAddr max_size = max_export_size(*buffer, extent, 0);
        EXPECT_GT(max_size, 0u);
        EXPECT_LE(max_size, extent.size());
    }
}

TEST_F(DmabufExportTest, PageCoveringExportReportsRequestedRange) {
    if (!is_supported()) {
        GTEST_SKIP() << "KMD or kernel does not support dma-buf export";
    }
    // UMD refuses to export unless an RDMA NIC is present to consume the dma-buf.
    const std::filesystem::path infiniband = "/sys/class/infiniband";
    if (!std::filesystem::exists(infiniband) || std::filesystem::is_empty(infiniband)) {
        GTEST_SKIP() << "No RDMA device under " << infiniband;
    }

    auto buffer = create_interleaved_dram_buffer(*mesh_device_, /*num_pages=*/64);
    const DramBankExtent extent = get_dram_bank_extents(*buffer).front();
    const DeviceAddr size = max_export_size(*buffer, extent, 0);

    DmabufExport exported = export_dram_bank_range(buffer, extent, 0, size, ExportBounds::PageCovering);

    const auto host_page = static_cast<size_t>(getpagesize());
    EXPECT_GE(exported.fd(), 0);
    EXPECT_EQ(exported.size() % host_page, 0u);
    EXPECT_LT(exported.data_offset(), host_page);
    EXPECT_EQ(exported.data_size(), size);
    EXPECT_LE(exported.data_offset() + exported.data_size(), exported.size());
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
