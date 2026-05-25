// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// H2DStreamService end-to-end correctness sweeps.
//
// Each test invokes `run_h2d_stream_service_case` with a (global_shape, placements,
// chunking) tuple. The helper:
//   * builds the service (which itself runs the mapper once to derive a per-shard
//     spec + topology, allocates the device tensor, and launches the persistent
//     receiver kernels),
//   * writes a fresh iota source through the chosen input path (`Tensor` or `Bytes`),
//   * barriers, reads back each per-device sub-tensor, and compares to per-coord
//     expected shards computed by re-running the same mapper on the host side,
//   * repeats with a different seed to prove the persistent kernel actually loops
//     (the "reuse" check — if the kernel exited after the first transfer the second
//     readback would still match the first source).
//
// All cases use UINT32 ROW_MAJOR DRAM-interleaved because that's the only regime
// the bytes path supports today. Block-float / TILE / non-default memory configs
// are deliberately out of scope for this file.

#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/socket_services.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::distributed::test {
namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::Tensor;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::H2DMode;
using ::tt::tt_metal::distributed::MeshMapperConfig;

// Two ways to push data into the service. The bytes path runs the service's
// internal mapper on the borrowed input; the Tensor path expects the caller to
// have already distributed via an equivalent mapper.
enum class InputPath {
    Tensor,
    Bytes,
};

inline const char* input_path_name(InputPath p) { return p == InputPath::Bytes ? "Bytes" : "Tensor"; }

// One sweep case. Pinned to UINT32 ROW_MAJOR DRAM-interleaved; varying axes are
// global shape, placement pattern, and chunking budget.
struct H2DServiceCase {
    ttnn::Shape global_shape;
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    uint32_t scratch_cb_size_bytes = 0;
    uint32_t fifo_size_bytes = 0;
    H2DMode mode = H2DMode::DEVICE_PULL;
};

// Drives the service through two full transfers (Write A + Write B reuse check)
// for a given case + input path, verifying every per-device readback against
// the per-coord shard produced by an external mapper with the same config.
//
// The Write B step is load-bearing: if the persistent kernel had exited after
// the first transfer, the second readback would still match data_a and the
// inner EXPECT_EQ would catch it.
void run_h2d_stream_service_case(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const H2DServiceCase& cs,
    InputPath input_path) {
    SCOPED_TRACE(
        ::testing::Message() << "global_shape=" << cs.global_shape << " scratch_cb=" << cs.scratch_cb_size_bytes
                             << " fifo=" << cs.fifo_size_bytes << " input_path=" << input_path_name(input_path));

    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(cs.global_shape, tensor_layout);

    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = cs.fifo_size_bytes,
        .scratch_cb_size_bytes = cs.scratch_cb_size_bytes,
        .socket_mode = cs.mode,
    };

    tt::tt_metal::H2DStreamService service(mesh_device, std::move(cfg));
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);
    ASSERT_EQ(service.get_sockets().size(), mesh_device->num_devices());

    // Push `src` through the selected input path and block until the service
    // has drained it. Tensor-path constructs a fresh external mapper each call;
    // bytes path lets the service's internal mapper handle distribution.
    auto do_write = [&](const std::vector<uint32_t>& src) {
        if (input_path == InputPath::Bytes) {
            auto bytes = ttsl::Span<const std::byte>(
                reinterpret_cast<const std::byte*>(src.data()), src.size() * sizeof(uint32_t));
            service.forward_to_tensor(bytes);
        } else {
            auto mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
            auto host = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *mapper);
            service.forward_to_tensor(host);
        }
        service.barrier();
    };

    // Read every per-device sub-tensor of the backing tensor and compare against
    // the per-coord shard from an external mapper applied to `src`. Using the
    // mapper here (instead of hand-coding per-placement slicing) keeps the
    // verification logic independent of the placement pattern under test.
    auto verify = [&](const std::vector<uint32_t>& src) {
        auto verify_mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
        auto distributed_host = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *verify_mapper);
        const auto& dhb = distributed_host.host_storage().host_tensor().buffer();

        auto subs = get_device_tensors(service.get_backing_tensor());
        ASSERT_EQ(subs.size(), mesh_device->num_devices());
        for (auto& sub : subs) {
            const auto coords = sub.device_storage().get_coords();
            ASSERT_EQ(coords.size(), 1u) << "expected one coord per device sub-tensor";
            const auto& coord = coords[0];

            ASSERT_TRUE(dhb.is_local(coord)) << "external mapper has no shard for coord " << coord;
            auto shard_opt = dhb.get_shard(coord);
            ASSERT_TRUE(shard_opt.has_value()) << "external mapper shard not populated at coord " << coord;
            auto exp_bytes = shard_opt->view_bytes();
            const auto* exp_begin = reinterpret_cast<const uint32_t*>(exp_bytes.data());
            std::vector<uint32_t> expected(exp_begin, exp_begin + exp_bytes.size() / sizeof(uint32_t));

            auto readback = sub.to_vector<uint32_t>();
            ASSERT_EQ(readback.size(), expected.size()) << "size mismatch at coord " << coord;
            EXPECT_EQ(readback, expected) << "contents mismatch at coord " << coord;
        }
    };

    // --- Write A ----------------------------------------------------------------
    std::vector<uint32_t> data_a(cs.global_shape.volume());
    std::iota(data_a.begin(), data_a.end(), 0u);
    do_write(data_a);
    verify(data_a);

    // --- Write B (persistent-kernel reuse check) --------------------------------
    // Wildly different seed so a stale buffer (e.g. kernel exited after A) shows
    // up in the readback compare as A's contents still landing.
    std::vector<uint32_t> data_b(cs.global_shape.volume());
    std::iota(data_b.begin(), data_b.end(), 0x12345678u);
    do_write(data_b);
    verify(data_b);
}

using H2DStreamServiceTest = ::tt::tt_metal::GenericMeshDeviceFixture;

// Build a fully-replicated placements vector sized to this mesh's dimensionality.
// Replicate on every mesh dim => same shard at every coord.
ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(
    const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    return ttsl::SmallVector<MeshMapperConfig::Placement>(
        mesh_device.shape().dims(), MeshMapperConfig::Replicate{});
}

// A — Replicated sweep. Mirrors the row structure used by the sharded sweep
// below: each row varies (per_row_size, N, scratch_cb, fifo) so the matrix
// covers both axes simultaneously:
//
//   axis 1 — page size (per_row_size * 4 bytes):
//     * 512 B  (per_row=128)   stresses per-iter overhead at high page count
//     * 2560 B (per_row=640)   baseline, matches the one-shot copy reference
//     * 4 KB   (per_row=1024)  typical real-world value
//     * 16 KB  (per_row=4096)  max NoC fan-out per PCIe read
//
//   axis 2 — chunk-picker branches:
//     * single chunk per device
//     * multi-chunk even split
//     * page-at-a-time (max iteration count)
//     * divisor fallback (prime N -> pages_per_chunk == 1)
//
// Per-device shape is [1, 1, N, per_row_size]. Under fully-replicated placements
// the global shape equals the per-device shape (every device gets the same
// shard), so the bytes path stays on the mapper's zero-copy replicate
// fast-path for every row.
TEST_F(H2DStreamServiceTest, Replicated_Sweep) {
    // `scratch_cb` and `fifo` are expressed as multiples of one tensor page
    // (per_row_size * sizeof(uint32_t)). Same shape as the sharded sweep's
    // Row struct so the two tests can be compared side-by-side.
    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        uint32_t scratch_cb_pages;
        uint32_t fifo_pages;
    };
    const Row rows[] = {
        // --- page_size = 2560 B (per_row_size = 640) ---------------------------
        // Single chunk: budget == total bytes. Baseline.
        Row{/*per_row=*/640, /*N=*/1, /*cb=*/1, /*fifo=*/1},
        // Multi-chunk even: 16 pages, budget for 4 -> 4 chunks of 4.
        Row{/*per_row=*/640, /*N=*/16, /*cb=*/4, /*fifo=*/16},
        // Page-at-a-time: 32 pages, budget == one page -> 32 chunks of 1.
        Row{/*per_row=*/640, /*N=*/32, /*cb=*/1, /*fifo=*/8},
        // Divisor fallback: 7 (prime) pages -> pages_per_chunk falls back to 1.
        Row{/*per_row=*/640, /*N=*/7, /*cb=*/4, /*fifo=*/8},

        // --- page_size = 512 B (per_row_size = 128) ----------------------------
        // Tiny page x many of them. Stresses per-iter overhead at small page size.
        Row{/*per_row=*/128, /*N=*/64, /*cb=*/1, /*fifo=*/8},

        // --- page_size = 4 KB (per_row_size = 1024) ----------------------------
        // Medium page, multi-chunk even split.
        Row{/*per_row=*/1024, /*N=*/16, /*cb=*/4, /*fifo=*/16},

        // --- page_size = 16 KB (per_row_size = 4096) ---------------------------
        // Large page; few of them, biggest fan-out per PCIe read.
        Row{/*per_row=*/4096, /*N=*/4, /*cb=*/2, /*fifo=*/4},
    };

    for (const auto& row : rows) {
        const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
        H2DServiceCase cs{
            // Fully replicated: global shape == per-device shape.
            .global_shape = ttnn::Shape({1, 1, row.N, row.per_row_size}),
            .placements = replicate_all(*this->mesh_device_),
            .scratch_cb_size_bytes = row.scratch_cb_pages * per_row_bytes,
            .fifo_size_bytes = row.fifo_pages * per_row_bytes,
        };
        run_h2d_stream_service_case(this->mesh_device_, cs, InputPath::Tensor);
        run_h2d_stream_service_case(this->mesh_device_, cs, InputPath::Bytes);
    }
}

// B/C/D/E — Sharded sweep across three placement patterns × four page sizes ×
// chunk-picker branches. Single 2D-mesh skip-guard; each placement pattern
// has its own additional applicability check (rows/cols >= 2 as appropriate).
//
// Placement patterns swept:
//   * ShardRowsReplicateCols: {Shard{3}, Replicate{}}     (needs rows >= 2)
//   * ReplicateRowsShardCols: {Replicate{}, Shard{3}}     (needs cols >= 2)
//   * FullShard2D:            {Shard{2}, Shard{3}}        (needs both >= 2)
//
// All three patterns yield the same per-device shape `[1, 1, N, per_row_size]`,
// so the chunking math and PCIe-page behavior are invariant across patterns —
// the differences live entirely in the mapper's distribution logic and the
// global shape multiplication.
//
// Row axes (per pattern):
//   page size (per_row_size * 4 bytes):
//     * 512 B (per_row=128)   stresses per-iter overhead at high page count
//     * 2560 B (per_row=640)  baseline, matches the one-shot copy reference
//     * 4 KB (per_row=1024)   typical real-world value
//     * 16 KB (per_row=4096)  max NoC fan-out per PCIe read
//   chunk-picker branches (driven by N + scratch_cb):
//     * single chunk per device
//     * multi-chunk even split
//     * page-at-a-time (max iteration count)
//     * divisor fallback (prime N -> pages_per_chunk == 1)
//
// L1 budget per recv core: scratch_cb + fifo + ~256 KB socket data buffer
// (DEVICE_PULL still allocates one). All rows below stay under the ~768 KB
// envelope.
//
// What this validates that A doesn't:
//   * The service threads non-trivial placements into the per-shard spec
//     correctly at every page size.
//   * The Bytes input path drives the mapper's xtensor sharding code (the
//     replicated fast-path no longer applies once any Shard placement is
//     present), AND co-exercises it with the kernel's chunked inner loop
//     in the multi-chunk rows.
//   * Mesh-dim ordering is honoured: ReplicateRowsShardCols catches anywhere
//     that hardcodes mesh-dim 0 as the shard axis; FullShard2D catches
//     anywhere that assumes shards overlap.
//   * The kernel's per-page PCIe burst-chunking (noc_read_page_chunked at
//     NOC_MAX_BURST_SIZE) is hit at multiple page sizes, not just one.
TEST_F(H2DStreamServiceTest, Sharded_Sweep) {
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "This test requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    // `scratch_cb` and `fifo` are expressed as multiples of one tensor page
    // (per_row_size * sizeof(uint32_t)) so each row is read as
    // (page_size_class, num_pages, pages_per_chunk_budget, fifo_pages).
    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        uint32_t scratch_cb_pages;
        uint32_t fifo_pages;
    };
    const Row rows[] = {
        // --- page_size = 2560 B (per_row_size = 640) ---------------------------
        // Single chunk: baseline parity with the one-shot copy test.
        Row{/*per_row=*/640, /*N=*/1, /*cb=*/1, /*fifo=*/1},
        // Multi-chunk even split: 16 pages, budget 4 -> 4 chunks of 4.
        Row{/*per_row=*/640, /*N=*/16, /*cb=*/4, /*fifo=*/16},
        // Page-at-a-time: 32 pages, budget 1 -> 32 chunks of 1.
        Row{/*per_row=*/640, /*N=*/32, /*cb=*/1, /*fifo=*/8},
        // Divisor fallback: 7 prime pages -> pages_per_chunk falls back to 1.
        Row{/*per_row=*/640, /*N=*/7, /*cb=*/4, /*fifo=*/8},

        // --- page_size = 512 B (per_row_size = 128) ----------------------------
        // Smallest interesting page x many of them. Stresses per-iter overhead.
        Row{/*per_row=*/128, /*N=*/64, /*cb=*/1, /*fifo=*/8},

        // --- page_size = 4 KB (per_row_size = 1024) ----------------------------
        // Medium page, multi-chunk even split: 16 pages, budget 4 -> 4 chunks of 4.
        Row{/*per_row=*/1024, /*N=*/16, /*cb=*/4, /*fifo=*/16},

        // --- page_size = 16 KB (per_row_size = 4096) ---------------------------
        // Large page; few of them, biggest fan-out per PCIe read.
        Row{/*per_row=*/4096, /*N=*/4, /*cb=*/2, /*fifo=*/4},
    };

    // One placement pattern's worth of work. Per-device shape is always
    // [1, 1, N, per_row_size]; `make_global_shape` describes how the global
    // tensor scales the per-device dims under this pattern's placements.
    auto run_pattern = [&](const char* label,
                           const ttsl::SmallVector<MeshMapperConfig::Placement>& placements,
                           const std::function<ttnn::Shape(uint32_t /*N*/, uint32_t /*per_row_size*/)>&
                               make_global_shape) {
        SCOPED_TRACE(::testing::Message() << "placement_pattern=" << label);
        for (const auto& row : rows) {
            const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
            H2DServiceCase cs{
                .global_shape = make_global_shape(row.N, row.per_row_size),
                .placements = placements,
                .scratch_cb_size_bytes = row.scratch_cb_pages * per_row_bytes,
                .fifo_size_bytes = row.fifo_pages * per_row_bytes,
            };
            run_h2d_stream_service_case(this->mesh_device_, cs, InputPath::Tensor);
            run_h2d_stream_service_case(this->mesh_device_, cs, InputPath::Bytes);
        }
    };

    // B/C — ShardRows × ReplicateCols.
    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols",
            {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_rows * per_row_size});
            });
    }

    // E — ReplicateRows × ShardCols. Mirror of B with mesh dims flipped:
    // shards across mesh-dim 1 instead of 0. Catches code that hardcodes
    // mesh-dim 0 as the shard axis.
    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols",
            {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_cols * per_row_size});
            });
    }

    // D — Full 2D shard. No replication: every device gets a distinct slice
    // along both tensor dim 2 (mesh-dim 0) and tensor dim 3 (mesh-dim 1).
    // Per-device shape stays [1, 1, N, per_row_size], so chunk math matches
    // B/C/E; only the global shape multiplier changes.
    if (num_rows >= 2 && num_cols >= 2) {
        run_pattern(
            "FullShard2D",
            {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, num_rows * N, num_cols * per_row_size});
            });
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
