// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::distributed::test {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::StorageType;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::H2DMode;
using ::tt::tt_metal::distributed::H2DSocket;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoreCoord;

using MultiDeviceTensorCreationTest = GenericMeshDeviceFixture;

// Helper for sweeping copy_tensor_over_socket over (shape, scratch CB budget, FIFO size)
// combinations. Mirrors the test_h2d_socket / test_d2h_socket pattern in
// tests/tt_metal/distributed/test_hd_sockets.cpp.
//
// Hardcoded to ROW_MAJOR UINT32 DRAM today; templatize on dtype if/when we need it.
void test_copy_tensor_over_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const ttnn::Shape& logical_shape,
    uint32_t scratch_cb_size_bytes,
    uint32_t fifo_size_bytes,
    H2DMode mode = H2DMode::DEVICE_PULL,
    const MeshCoreCoord& recv_core = MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0))) {
    auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    auto spec = TensorSpec(logical_shape, tensor_layout);

    std::vector<uint32_t> src(logical_shape.volume());
    std::iota(src.begin(), src.end(), 0u);
    Tensor host_tensor = Tensor::from_vector<uint32_t>(src, spec);

    Tensor device_tensor = tt::tt_metal::create_device_tensor(spec, mesh_device.get());
    ASSERT_NE(device_tensor.buffer(), nullptr);
    ASSERT_EQ(device_tensor.dtype(), DataType::UINT32);
    ASSERT_EQ(device_tensor.layout(), Layout::ROW_MAJOR);
    ASSERT_EQ(device_tensor.memory_config().buffer_type(), BufferType::DRAM);
    ASSERT_EQ(device_tensor.buffer()->num_pages() * device_tensor.buffer()->page_size(), src.size() * sizeof(uint32_t));

    H2DSocket socket(mesh_device, recv_core, BufferType::L1, fifo_size_bytes, mode);

    tt::tt_metal::copy_tensor_over_socket(host_tensor, device_tensor, {&socket}, scratch_cb_size_bytes);

    socket.barrier();

    auto readback = device_tensor.to_vector<uint32_t>();
    ASSERT_THAT(readback, SizeIs(src.size()));
    EXPECT_EQ(readback, src) << "shape=" << logical_shape << " scratch_cb_size_bytes=" << scratch_cb_size_bytes
                             << " fifo_size_bytes=" << fifo_size_bytes;
}

TEST_F(MultiDeviceTensorCreationTest, Empty) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    const Tensor mesh_replicated_tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_F(MultiDeviceTensorCreationTest, EmptyLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    ASSERT_FALSE(mesh_device->get_devices().empty());

    const Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);

    const Tensor mesh_replicated_tensor = ttnn::empty_like(
        tensor,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        *mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_F(MultiDeviceTensorCreationTest, Full) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    const Tensor mesh_replicated_tensor = ttnn::full(
        ttnn::Shape({32, 32}),
        /*fill_value=*/42,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        std::ref(*mesh_device),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(mesh_replicated_tensor.logical_shape(), ttnn::Shape({32, 32}));
    EXPECT_EQ(mesh_replicated_tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(mesh_replicated_tensor.layout(), Layout::ROW_MAJOR);

    auto device_tensors = get_device_tensors(mesh_replicated_tensor);
    EXPECT_THAT(device_tensors, SizeIs(mesh_device->num_devices()));
    for (const auto& device_tensor : device_tensors) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(32 * 32));
        EXPECT_THAT(values, Each(Eq(42.0f)));
    }
}

TEST_F(MultiDeviceTensorCreationTest, FullLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    Tensor mesh_replicated_tensor = ttnn::full_like(
        tensor,
        /*fill_value=*/42,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        std::ref(*mesh_device));

    EXPECT_EQ(mesh_replicated_tensor.logical_shape(), tensor.logical_shape());
    EXPECT_EQ(mesh_replicated_tensor.padded_shape(), tensor.padded_shape());
    EXPECT_EQ(mesh_replicated_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(mesh_replicated_tensor.layout(), tensor.layout());

    auto device_tensors = get_device_tensors(mesh_replicated_tensor);
    EXPECT_THAT(device_tensors, SizeIs(mesh_device->num_devices()));
    for (const auto& device_tensor : device_tensors) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(32 * 32));
        EXPECT_THAT(values, Each(Eq(42.0f)));
    }
}

TEST_F(MultiDeviceTensorCreationTest, FullLikeWithOptTensor) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);

    Tensor opt_output = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    Tensor mesh_replicated_tensor = ttnn::full_like(
        tensor,
        /*fill_value=*/42,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        opt_output);

    EXPECT_EQ(mesh_replicated_tensor.logical_shape(), tensor.logical_shape());
    EXPECT_EQ(mesh_replicated_tensor.padded_shape(), tensor.padded_shape());
    EXPECT_EQ(mesh_replicated_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(mesh_replicated_tensor.layout(), tensor.layout());
    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_F(MultiDeviceTensorCreationTest, Arange) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    Tensor tensor = ttnn::arange(
        /*start=*/0,
        /*stop=*/1024,
        /*step=*/1,
        ttnn::DataType::FLOAT32,
        std::ref(*mesh_device));

    EXPECT_EQ(tensor.logical_shape(), ttnn::Shape({1024}));
    EXPECT_THAT(get_device_tensors(tensor), SizeIs(mesh_device->num_devices()));

    std::vector<float> expected(1024);
    std::iota(expected.begin(), expected.end(), 0.0f);
    for (const auto& device_tensor : get_device_tensors(tensor)) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(1024));
        EXPECT_THAT(values, Pointwise(FloatEq(), expected));
    }
}

TEST_F(MultiDeviceTensorCreationTest, CopyTensorOverH2DSocket_Uint32_RowMajor_Dram) {
    // Per-row physical size for ROW_MAJOR UINT32 with innermost dim 640: 640 * 4 = 2560 B.
    // Each (shape, scratch_cb_size, fifo) tuple below exercises a different code path in the
    // chunk picker. The FIFO is sized generously (>= 2 * socket_page_size) so the host can
    // overlap fills with the kernel's NoC writes after the early-release.

    // Single chunk: budget >= total bytes. Baseline regression for the original test.
    test_copy_tensor_over_socket(this->mesh_device_, ttnn::Shape({1, 1, 1, 640}), /*scratch_cb=*/2560, /*fifo=*/2560);

    // Multi-chunk, even split: 16 tensor pages, budget for 4 -> 4 chunks of 4 pages each.
    test_copy_tensor_over_socket(
        this->mesh_device_, ttnn::Shape({1, 1, 16, 640}), /*scratch_cb=*/4 * 2560, /*fifo=*/16 * 2560);

    // Multi-chunk, page-at-a-time: budget == tensor_page_size -> pages_per_chunk = 1.
    // Stresses iteration count and the per-iteration socket/CB overhead.
    test_copy_tensor_over_socket(
        this->mesh_device_, ttnn::Shape({1, 1, 32, 640}), /*scratch_cb=*/2560, /*fifo=*/8 * 2560);

    // Divisor fallback: 7 tensor pages (prime). Budget-max would give 4, but 7 % 4 != 0,
    // 7 % 3 != 0, 7 % 2 != 0 -> the loop falls back to pages_per_chunk = 1, num_socket_pages = 7.
    test_copy_tensor_over_socket(
        this->mesh_device_, ttnn::Shape({1, 1, 7, 640}), /*scratch_cb=*/4 * 2560, /*fifo=*/8 * 2560);

    // Oversized budget: budget far exceeds total bytes -> still a single chunk.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 8, 640}),
        /*scratch_cb=*/1024 * 1024,
        /*fifo=*/1024 * 1024);

    // Large multi-chunk: 256 tensor pages (640 KB total). Budget for 16 pages -> 16 chunks of
    // 16 pages each. Exercises early-release + writes_flushed at non-trivial iteration count.
    test_copy_tensor_over_socket(
        this->mesh_device_, ttnn::Shape({1, 1, 256, 640}), /*scratch_cb=*/16 * 2560, /*fifo=*/64 * 2560);
}

TEST_F(MultiDeviceTensorCreationTest, CopyTensorOverH2DSocket_Uint32_RowMajor_Dram_Stress) {
    // Move MB-scale data through every interesting (volume, chunk-size, page-size) combination.
    // Each case asserts byte-exact round-trip, so this also catches drift in the chunking math
    // and early-release ordering at scale.
    //
    // L1 budget note: even in DEVICE_PULL mode H2DSocket::init_data_buffer allocates a
    // (fifo_size + pcie_alignment) L1 buffer per worker core (see h2d_socket.cpp:124).
    // On Blackhole the usable L1 is ~1.3 MB, so we cap `scratch_cb + fifo_size` per case
    // to ~768 KB to leave room for dispatch overhead. Without this cap, large FIFOs
    // collide with the scratch CB on the recv core.
    //
    // For ROW_MAJOR UINT32:
    //   tensor_page_size = innermost_dim * 4
    //   tensor_num_pages = product(outer dims)
    //   total bytes      = num_pages * page_size

    // 1 MB / 2 chunks.    page=1 KB,  1024 pages. budget 512 KB   -> 2 chunks of 512 pages each.
    // Large fat chunk: the most data we can push per PCIe read without blowing the L1 budget.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 1024, 256}),
        /*scratch_cb=*/512 * 1024,
        /*fifo=*/512 * 1024);

    // 1 MB / 256 chunks.  page=1 KB,  1024 pages. budget 4 KB     -> 256 chunks of 4 pages each.
    // FIFO holds only 4 chunks, so host write loop has to block on reserve_bytes repeatedly.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 1024, 256}),
        /*scratch_cb=*/4 * 1024,
        /*fifo=*/16 * 1024);

    // 4 MB / 32 chunks.   page=4 KB,  1024 pages. budget 128 KB   -> 32 chunks of 32 pages each.
    // Balanced "fat chunk" config: each chunk fans out 32 NoC writes after one PCIe read.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 1024, 1024}),
        /*scratch_cb=*/128 * 1024,
        /*fifo=*/512 * 1024);

    // 4 MB / 1024 chunks. page=4 KB,  1024 pages. budget 4 KB     -> 1024 chunks of 1 page each.
    // Same total bytes as the row above but maximum chunk count; stresses per-iter overhead.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 1024, 1024}),
        /*scratch_cb=*/4 * 1024,
        /*fifo=*/16 * 1024);

    // 16 MB / 128 chunks. page=4 KB,  4096 pages. budget 128 KB   -> 128 chunks of 32 pages each.
    // Largest single-test volume; useful for catching anything that degrades super-linearly.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 4096, 1024}),
        /*scratch_cb=*/128 * 1024,
        /*fifo=*/512 * 1024);

    // 256 KB / 4 chunks.  page=16 KB, 16 pages.   budget 64 KB    -> 4 chunks of 4 pages each.
    // Wide-innermost-dim regime: big tensor pages, few of them, large per-NoC-write payload.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 16, 4096}),
        /*scratch_cb=*/64 * 1024,
        /*fifo=*/256 * 1024);

    // 1 MB / 2048 chunks. page=512 B, 2048 pages. budget 512 B    -> 2048 chunks of 1 page each.
    // Smallest-page x highest-chunk-count combo. Worst case for per-iteration overhead.
    test_copy_tensor_over_socket(
        this->mesh_device_,
        ttnn::Shape({1, 1, 2048, 128}),
        /*scratch_cb=*/512,
        /*fifo=*/4 * 1024);
}

}  // namespace
}  // namespace ttnn::distributed::test
