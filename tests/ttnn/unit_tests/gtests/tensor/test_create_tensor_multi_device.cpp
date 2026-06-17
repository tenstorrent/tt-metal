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
#include "ttnn/tensor/socket_services.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
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
using ::tt::tt_metal::TensorTopology;
using ::tt::tt_metal::distributed::H2DMode;
using ::tt::tt_metal::distributed::H2DSocket;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::MeshCoreCoord;
using ::tt::tt_metal::distributed::MeshMapperConfig;

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

// Multi-device helper: full mesh, replicated tensor, one H2D socket per device coord
// (all pinned to core (0,0)). Asserts that every device's read-back matches `src`.
//
// ROW_MAJOR UINT32 DRAM, like the single-device helper above. The host tensor is built via
// `replicate_tensor_to_mesh_mapper` so its `DistributedHostBuffer` has a populated shard
// at every mesh coord, and copy_tensor_over_socket fans the bytes out via N sockets.
void test_copy_tensor_over_socket_replicated(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const ttnn::Shape& logical_shape,
    uint32_t scratch_cb_size_bytes,
    uint32_t fifo_size_bytes,
    H2DMode mode = H2DMode::DEVICE_PULL) {
    auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    auto spec = TensorSpec(logical_shape, tensor_layout);

    std::vector<uint32_t> src(logical_shape.volume());
    std::iota(src.begin(), src.end(), 0u);

    // Build a replicated distributed host tensor: every mesh coord gets the same shard.
    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device);
    Tensor host_tensor = distribute_tensor(Tensor::from_vector<uint32_t>(src, spec), *mapper);
    ASSERT_EQ(host_tensor.storage_type(), StorageType::HOST);

    // Allocate a matching device tensor with the same fully-replicated topology so the
    // host shards' coords line up 1:1 with the device buffers.
    Tensor device_tensor = tt::tt_metal::create_device_tensor(spec, mesh_device.get(), host_tensor.tensor_topology());
    ASSERT_NE(device_tensor.buffer(), nullptr);
    ASSERT_EQ(device_tensor.dtype(), DataType::UINT32);
    ASSERT_EQ(device_tensor.layout(), Layout::ROW_MAJOR);
    ASSERT_EQ(device_tensor.memory_config().buffer_type(), BufferType::DRAM);

    // One socket per mesh coord, all on core (0,0). H2DSocket is non-copyable, so own them
    // via unique_ptr and hand raw pointers to copy_tensor_over_socket.
    std::vector<std::unique_ptr<H2DSocket>> owned_sockets;
    std::vector<H2DSocket*> raw_sockets;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        owned_sockets.push_back(std::make_unique<H2DSocket>(
            mesh_device, MeshCoreCoord(coord, CoreCoord(0, 0)), BufferType::L1, fifo_size_bytes, mode));
        raw_sockets.push_back(owned_sockets.back().get());
    }

    tt::tt_metal::copy_tensor_over_socket(host_tensor, device_tensor, raw_sockets, scratch_cb_size_bytes);

    for (auto& s : owned_sockets) {
        s->barrier();
    }

    // Per-shard verification: every device should hold the full source vector.
    auto device_tensors = get_device_tensors(device_tensor);
    ASSERT_EQ(device_tensors.size(), mesh_device->num_devices());
    for (size_t i = 0; i < device_tensors.size(); ++i) {
        auto readback = device_tensors[i].to_vector<uint32_t>();
        ASSERT_THAT(readback, SizeIs(src.size()));
        EXPECT_EQ(readback, src) << "device_idx=" << i << " shape=" << logical_shape
                                 << " scratch_cb_size_bytes=" << scratch_cb_size_bytes
                                 << " fifo_size_bytes=" << fifo_size_bytes;
    }
}

// Multi-device helper: 2D mesh, flat 1D tensor sharded along the innermost tensor dim
// across mesh dim 0 (rows) and replicated across mesh dim 1 (cols).
//
// Global tensor shape: [1, 1, 1, num_rows * per_row_size] UINT32 ROW_MAJOR DRAM.
// Per-device shard shape: [1, 1, 1, per_row_size]. Devices in mesh row `r` all hold the
// same shard (src[r * per_row_size : (r+1) * per_row_size]); cols replicate.
//
// Per-device tensor page size = per_row_size * sizeof(uint32_t) bytes (innermost dim).
// Pick per_row_size so that page size is PCIe-aligned (>=16 B in practice). 640 elements
// of uint32 = 2560 B, comfortably aligned.
void test_copy_tensor_over_socket_shard_rows_replicate_cols(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t per_row_size,
    uint32_t scratch_cb_size_bytes,
    uint32_t fifo_size_bytes,
    H2DMode mode = H2DMode::DEVICE_PULL) {
    const auto mesh_shape = mesh_device->shape();
    ASSERT_EQ(mesh_shape.dims(), 2u) << "This helper requires a 2D mesh; got " << mesh_shape;
    const uint32_t num_rows = mesh_shape[0];

    const ttnn::Shape global_shape({1, 1, 1, num_rows * per_row_size});
    const ttnn::Shape shard_shape({1, 1, 1, per_row_size});

    auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    auto global_spec = TensorSpec(global_shape, tensor_layout);
    auto shard_spec = TensorSpec(shard_shape, tensor_layout);

    // Distinct values so a per-device readback uniquely identifies which slice landed where.
    std::vector<uint32_t> src(global_shape.volume());
    std::iota(src.begin(), src.end(), 0u);

    // 2D mapper: shard along tensor dim 3 (innermost) across mesh dim 0, replicate across
    // mesh dim 1. Single shard dim + all preceding tensor dims = 1 -> mapper takes the
    // zero-copy borrow path for each per-device HostBuffer.
    ttsl::SmallVector<MeshMapperConfig::Placement> placements = {
        MeshMapperConfig::Shard{3},
        MeshMapperConfig::Replicate{},
    };
    auto mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = placements});

    Tensor host_tensor = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *mapper);
    ASSERT_EQ(host_tensor.storage_type(), StorageType::HOST);
    ASSERT_EQ(host_tensor.tensor_spec().logical_shape(), shard_shape) << "mapper output shard shape mismatch";

    Tensor device_tensor =
        tt::tt_metal::create_device_tensor(shard_spec, mesh_device.get(), host_tensor.tensor_topology());
    ASSERT_NE(device_tensor.buffer(), nullptr);
    ASSERT_EQ(device_tensor.dtype(), DataType::UINT32);

    // One socket per mesh coord, all pinned to core (0,0).
    std::vector<std::unique_ptr<H2DSocket>> owned_sockets;
    std::vector<H2DSocket*> raw_sockets;
    for (const auto& coord : MeshCoordinateRange(mesh_shape)) {
        owned_sockets.push_back(std::make_unique<H2DSocket>(
            mesh_device, MeshCoreCoord(coord, CoreCoord(0, 0)), BufferType::L1, fifo_size_bytes, mode));
        raw_sockets.push_back(owned_sockets.back().get());
    }

    tt::tt_metal::copy_tensor_over_socket(host_tensor, device_tensor, raw_sockets, scratch_cb_size_bytes);

    for (auto& s : owned_sockets) {
        s->barrier();
    }

    // Per-shard verification: each device's readback must equal the slice of `src` for its
    // mesh row, regardless of mesh column (those are replicated copies of the same shard).
    auto device_tensors = get_device_tensors(device_tensor);
    ASSERT_EQ(device_tensors.size(), mesh_device->num_devices());
    for (const auto& sub : device_tensors) {
        const auto coords = sub.device_storage().get_coords();
        ASSERT_EQ(coords.size(), 1u);
        const auto& coord = coords[0];
        const uint32_t row = coord[0];

        auto readback = sub.to_vector<uint32_t>();
        ASSERT_EQ(readback.size(), per_row_size);

        std::vector<uint32_t> expected(src.begin() + row * per_row_size, src.begin() + (row + 1) * per_row_size);
        EXPECT_EQ(readback, expected) << "coord=" << coord << " scratch_cb=" << scratch_cb_size_bytes
                                      << " fifo=" << fifo_size_bytes;
    }
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

TEST_F(MultiDeviceTensorCreationTest, CopyTensorOverH2DSocket_MultiDevice_Replicated) {
    // Replicate the same bytes to every device coord in the mesh. Sweeps the same chunking
    // regimes as the single-device test but with one socket per device. Each case asserts
    // byte-exact per-device readback.

    // Single chunk per device: baseline.
    test_copy_tensor_over_socket_replicated(
        this->mesh_device_, ttnn::Shape({1, 1, 1, 640}), /*scratch_cb=*/2560, /*fifo=*/2560);

    // Multi-chunk per device, even split: 16 pages, 4 chunks of 4.
    test_copy_tensor_over_socket_replicated(
        this->mesh_device_, ttnn::Shape({1, 1, 16, 640}), /*scratch_cb=*/4 * 2560, /*fifo=*/16 * 2560);

    // Multi-chunk per device, page-at-a-time: exercises max iteration count.
    test_copy_tensor_over_socket_replicated(
        this->mesh_device_, ttnn::Shape({1, 1, 32, 640}), /*scratch_cb=*/2560, /*fifo=*/8 * 2560);

    // Divisor fallback (prime page count).
    test_copy_tensor_over_socket_replicated(
        this->mesh_device_, ttnn::Shape({1, 1, 7, 640}), /*scratch_cb=*/4 * 2560, /*fifo=*/8 * 2560);

    // Larger transfer to verify per-device pipelining at scale.
    test_copy_tensor_over_socket_replicated(
        this->mesh_device_, ttnn::Shape({1, 1, 256, 640}), /*scratch_cb=*/16 * 2560, /*fifo=*/64 * 2560);
}

TEST_F(MultiDeviceTensorCreationTest, CopyTensorOverH2DSocket_MultiDevice_ShardRowsReplicateCols) {
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2 || mesh_shape[0] < 2 || mesh_shape[1] < 1) {
        GTEST_SKIP() << "This test requires a 2D mesh with shape (>=2) x (>=1); got " << mesh_shape;
    }

    // Global tensor: [1, 1, 1, num_rows * 640] UINT32 ROW_MAJOR DRAM.
    // Per-device shard: [1, 1, 1, 640]. All devices in mesh row r get src[r*640:(r+1)*640];
    // the cols in each row hold the same 640 elements.
    //
    // For the 8x4 mesh case: global = [1, 1, 1, 5120], each of 32 devices gets 640 elements.
    //
    // Per-device chunking math:
    //   tensor_page_size = innermost_dim * 4 = 640 * 4 = 2560 B   (PCIe-aligned)
    //   tensor_num_pages = 1 * 1 * 1            = 1
    //   total per-device = 2560 B
    //
    // num_pages == 1 means there's only one chunk per device by construction; multi-chunk
    // sweeps would need a per-device shape with num_pages > 1 (e.g. [1, 1, N, 640]).
    test_copy_tensor_over_socket_shard_rows_replicate_cols(
        this->mesh_device_,
        /*per_row_size=*/640,
        /*scratch_cb=*/2560,
        /*fifo=*/2560);
}

// =====================================================================================
// H2DStreamService tests
// =====================================================================================

// Smallest possible H2DStreamService correctness test:
//   * fully replicated mapper (every device gets the full tensor),
//   * single chunk per transfer (scratch CB sized to hold the whole tensor),
//   * fixed shape [1, 1, 1, 64K] UINT32 ROW_MAJOR DRAM.
//
// What this validates that nothing else in the suite covers:
//   1. Ctor builds + enqueues the persistent workload non-blocking.
//   2. The persistent kernel actually loops (write B sees DIFFERENT data than write A;
//      if the kernel exited after the first transfer this would fail).
//   3. forward_to_tensor(const Tensor&) end-to-end.
//   4. barrier() drains in-flight host->socket writes.
//   5. Dtor sequence (barrier -> signal_termination -> Finish) doesn't deadlock.
//
// Per-shard chunking math (uniform across every device, replicated):
//   tensor_page_size  = 65536 * 4 = 262144 B   (PCIe-aligned: 64 | 262144)
//   tensor_num_pages  = 1*1*1     = 1
//   -> with scratch_cb_size_bytes == tensor_page_size:
//      socket_page_size  = 262144 B
//      num_socket_pages  = 1
//      pages_per_chunk   = 1
//
// L1 budget on the recv core: 256 KiB scratch CB + 256 KiB H2DSocket L1 data buffer
// (DEVICE_PULL still allocates one — known limitation noted in the prior session) =
// 512 KiB, comfortably under the ~768 KiB envelope we've been holding to.
TEST_F(MultiDeviceTensorCreationTest, H2DStreamService_Replicated_SingleChunk_64K_Reuse) {
    const ttnn::Shape global_shape({1, 1, 1, 65536});
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(global_shape, tensor_layout);
    const auto bytes_per_shard = static_cast<uint32_t>(global_spec.compute_packed_buffer_size_bytes());
    ASSERT_EQ(bytes_per_shard, 65536u * sizeof(uint32_t));

    // Build a fully-replicated MeshMapperConfig matching the mesh's dimensionality.
    // (Same outcome as ttnn::distributed::replicate_tensor_to_mesh_mapper, but we need
    // the raw config to hand to the service.)
    ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(
        this->mesh_device_->shape().dims(), MeshMapperConfig::Replicate{});

    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*this->mesh_device_, MeshMapperConfig{.placements = replicate_all}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = bytes_per_shard,
        .scratch_cb_size_bytes = bytes_per_shard,
        .socket_mode = H2DMode::DEVICE_PULL,
    };

    tt::tt_metal::H2DStreamService service(this->mesh_device_, std::move(cfg));

    // For fully-replicated mapping the per-shard spec equals the global spec — assert here
    // so the spec-equality check inside forward_to_tensor(host_tensor) can't possibly fire
    // without obvious diagnostics on the test side.
    ASSERT_EQ(service.get_per_shard_spec(), global_spec);
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);
    EXPECT_EQ(service.get_sockets().size(), this->mesh_device_->num_devices());

    // External mapper instance used only to produce well-formed distributed host tensors.
    // Same placements as the one moved into the service -> same per-shard spec & topology.
    auto external_mapper =
        create_mesh_mapper(*this->mesh_device_, MeshMapperConfig{.placements = replicate_all});

    auto build_host_tensor = [&](uint32_t seed) {
        std::vector<uint32_t> data(global_shape.volume());
        std::iota(data.begin(), data.end(), seed);
        return distribute_tensor(Tensor::from_vector<uint32_t>(data, global_spec), *external_mapper);
    };

    auto readback_per_device = [&]() {
        auto subs = get_device_tensors(service.get_backing_tensor());
        EXPECT_EQ(subs.size(), this->mesh_device_->num_devices());
        std::vector<std::vector<uint32_t>> out;
        out.reserve(subs.size());
        for (auto& sub : subs) {
            out.push_back(sub.to_vector<uint32_t>());
        }
        return out;
    };

    // --- Write A ---------------------------------------------------------------------
    auto host_a = build_host_tensor(/*seed=*/0u);
    ASSERT_EQ(host_a.tensor_spec(), service.get_per_shard_spec());

    service.forward_to_tensor(host_a);
    service.barrier();

    {
        std::vector<uint32_t> expected_a(global_shape.volume());
        std::iota(expected_a.begin(), expected_a.end(), 0u);
        auto results = readback_per_device();
        for (size_t i = 0; i < results.size(); ++i) {
            ASSERT_EQ(results[i].size(), expected_a.size()) << "device " << i << " size after write A";
            EXPECT_EQ(results[i], expected_a) << "device " << i << " contents after write A";
        }
    }

    // --- Write B (reuse check) -------------------------------------------------------
    // Different seed -> wildly different bytes. If the persistent kernel exited after
    // write A, this readback would still equal expected_a and the EXPECT below would
    // fire. This is the load-bearing assertion that the service is actually persistent.
    auto host_b = build_host_tensor(/*seed=*/0x12345678u);

    service.forward_to_tensor(host_b);
    service.barrier();

    {
        std::vector<uint32_t> expected_b(global_shape.volume());
        std::iota(expected_b.begin(), expected_b.end(), 0x12345678u);
        auto results = readback_per_device();
        for (size_t i = 0; i < results.size(); ++i) {
            ASSERT_EQ(results[i].size(), expected_b.size()) << "device " << i << " size after write B";
            EXPECT_EQ(results[i], expected_b) << "device " << i << " contents after write B";
        }
    }
    // `service` going out of scope exercises the dtor:
    //   barrier() -> signal_termination() -> Finish(mesh CQ).
    // If anything in that sequence deadlocks the test hangs here.
}

// Mirror of the test above, but driving the service via the raw-bytes path
// `forward_to_tensor(ttsl::Span<const std::byte>)` instead of a pre-distributed
// host tensor. The service runs its internal mapper on the borrowed bytes,
// distributes, and streams; this test validates that whole chain end-to-end.
//
// What this validates that the host-tensor test doesn't:
//   1. make_borrowed_host_tensor wraps the caller's bytes without breaking the
//      mapper's zero-copy fast path (for replicated ROW_MAJOR data we expect
//      no host-side normalization copy, only the per-FIFO socket writes).
//   2. The internal mapper produces a distributed tensor whose per-shard spec
//      matches `device_tensor_.tensor_spec()` (i.e. the bytes-path spec
//      assertion never fires).
//   3. The persistent kernel still loops across multiple bytes-path transfers
//      (write B is the load-bearing reuse check, same as in the Tensor test).
TEST_F(MultiDeviceTensorCreationTest, H2DStreamService_Replicated_SingleChunk_64K_Reuse_BytesPath) {
    const ttnn::Shape global_shape({1, 1, 1, 65536});
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(global_shape, tensor_layout);
    const auto bytes_per_shard = static_cast<uint32_t>(global_spec.compute_packed_buffer_size_bytes());
    ASSERT_EQ(bytes_per_shard, 65536u * sizeof(uint32_t));

    ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(
        this->mesh_device_->shape().dims(), MeshMapperConfig::Replicate{});

    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*this->mesh_device_, MeshMapperConfig{.placements = replicate_all}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = bytes_per_shard,
        .scratch_cb_size_bytes = bytes_per_shard,
        .socket_mode = H2DMode::DEVICE_PULL,
    };

    tt::tt_metal::H2DStreamService service(this->mesh_device_, std::move(cfg));

    ASSERT_EQ(service.get_per_shard_spec(), global_spec);
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);
    EXPECT_EQ(service.get_sockets().size(), this->mesh_device_->num_devices());

    auto readback_per_device = [&]() {
        auto subs = get_device_tensors(service.get_backing_tensor());
        EXPECT_EQ(subs.size(), this->mesh_device_->num_devices());
        std::vector<std::vector<uint32_t>> out;
        out.reserve(subs.size());
        for (auto& sub : subs) {
            out.push_back(sub.to_vector<uint32_t>());
        }
        return out;
    };

    // The caller owns the source bytes; reinterpret as `Span<const std::byte>`.
    // The service's borrowed wrap doesn't outlive forward_to_tensor, and
    // H2DSocket::write is synchronous, so this storage lifetime is sufficient.
    auto as_byte_span = [](const std::vector<uint32_t>& v) {
        return ttsl::Span<const std::byte>(reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(uint32_t));
    };

    // --- Write A ---------------------------------------------------------------------
    std::vector<uint32_t> data_a(global_shape.volume());
    std::iota(data_a.begin(), data_a.end(), 0u);

    service.forward_to_tensor(as_byte_span(data_a));
    service.barrier();

    {
        auto results = readback_per_device();
        for (size_t i = 0; i < results.size(); ++i) {
            ASSERT_EQ(results[i].size(), data_a.size()) << "device " << i << " size after bytes-path write A";
            EXPECT_EQ(results[i], data_a) << "device " << i << " contents after bytes-path write A";
        }
    }

    // --- Write B (reuse check) -------------------------------------------------------
    std::vector<uint32_t> data_b(global_shape.volume());
    std::iota(data_b.begin(), data_b.end(), 0x12345678u);

    service.forward_to_tensor(as_byte_span(data_b));
    service.barrier();

    {
        auto results = readback_per_device();
        for (size_t i = 0; i < results.size(); ++i) {
            ASSERT_EQ(results[i].size(), data_b.size()) << "device " << i << " size after bytes-path write B";
            EXPECT_EQ(results[i], data_b) << "device " << i << " contents after bytes-path write B";
        }
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
