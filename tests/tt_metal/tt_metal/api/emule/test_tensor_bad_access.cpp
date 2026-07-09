// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Host_UAF_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/core_subset_write/buffer_write.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// These tests verify the host-side Use-After-Free sanitizer in tt_metal.cpp
// fires for every public Buffer-access entry point. Each test allocates a
// Buffer, deallocates it (keeping the Buffer object alive — Buffer's address_
// member still points at memory the allocator has now reclaimed), and then
// invokes the entry point, expecting an abort with the [ASAN ERROR]
// Use-After-Free message naming that specific entry point.

TEST_F(MeshDeviceFixture, Host_UAF_WriteToBuffer_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];

    auto buffer = Buffer::create(device, 1024, 1024, BufferType::L1);
    DeallocateBuffer(*buffer);

    std::vector<uint32_t> data(256, 0xABCDEFu);
    EXPECT_DEATH(detail::WriteToBuffer(*buffer, data), ".*Use-After-Free.*WriteToBuffer.*");
}

TEST_F(MeshDeviceFixture, Host_UAF_ReadFromBuffer_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];

    auto buffer = Buffer::create(device, 1024, 1024, BufferType::L1);
    DeallocateBuffer(*buffer);

    std::vector<uint32_t> out;
    EXPECT_DEATH(detail::ReadFromBuffer(*buffer, out), ".*Use-After-Free.*ReadFromBuffer.*");
}

TEST_F(MeshDeviceFixture, Host_UAF_ReadShard_SanityCheck) {
    auto* device = this->devices_.at(0)->get_devices()[0];

    // ReadShard's sanitizer check runs before its is_sharded() assertion, so a
    // plain interleaved buffer still drives the UAF path under test here.
    auto buffer = Buffer::create(device, 1024, 1024, BufferType::L1);
    DeallocateBuffer(*buffer);

    std::vector<uint8_t> out(1024);
    EXPECT_DEATH(detail::ReadShard(*buffer, out.data(), /*core_id=*/0), ".*Use-After-Free.*ReadShard.*");
}

TEST_F(MeshDeviceFixture, Host_UAF_CoreSubsetWriteToBuffer_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];

    auto buffer = Buffer::create(device, 1024, 1024, BufferType::L1);
    DeallocateBuffer(*buffer);

    std::vector<uint32_t> data(256, 0xABCDEFu);
    CoreRangeSet logical_core_filter{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}};
    EXPECT_DEATH(
        experimental::core_subset_write::WriteToBuffer(
            *buffer,
            ttsl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(uint32_t)),
            logical_core_filter),
        ".*Use-After-Free.*core_subset_write.*");
}

// Catches use *through* a shared_ptr<Buffer>: the templated shared_ptr
// overloads in tt_metal.hpp dereference and forward to the Buffer& entry
// points, so this confirms the sanitizer fires through the smart-pointer path
// too (the path most user code actually takes).
TEST_F(MeshDeviceFixture, Host_UAF_WriteToBuffer_SharedPtrOverload_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];

    auto buffer = Buffer::create(device, 1024, 1024, BufferType::L1);
    DeallocateBuffer(*buffer);

    std::vector<uint32_t> data(256, 0xABCDEFu);
    EXPECT_DEATH(detail::WriteToBuffer(buffer, data), ".*Use-After-Free.*WriteToBuffer.*");
}

// Positive control: a write/read round-trip on a LIVE (still-allocated) buffer
// must NOT abort and must round-trip the data. Guards the check from regressing
// to flag healthy buffers (e.g. an inverted is_allocated() test).
TEST_F(MeshDeviceFixture, Host_UAF_Allocated_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];

    auto buffer = Buffer::create(device, 1024, 1024, BufferType::L1);  // left allocated

    std::vector<uint32_t> data(256, 0xABCDEFu);
    detail::WriteToBuffer(*buffer, data);  // must NOT abort

    std::vector<uint32_t> out;
    detail::ReadFromBuffer(*buffer, out);  // must NOT abort
    EXPECT_EQ(out, data);

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
