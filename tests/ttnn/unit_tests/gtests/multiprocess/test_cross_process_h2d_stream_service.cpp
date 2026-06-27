// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-process H2DStreamService integration test (runs under MPI as 2 ranks: owner + connector).

#include <array>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::tt_metal::distributed {
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

int g_world_rank = -1;
int g_world_size = -1;
// Original 2-rank world, held for SetUp-time cross-rank coordination (mesh-shape broadcast).
std::shared_ptr<::tt::tt_metal::distributed::multihost::DistributedContext> g_cross_rank_world;

// Deterministic per-iter source data; identical formula on both ranks so no
// IPC is needed for seeds.
std::vector<uint32_t> make_iter_data(uint32_t iter, size_t volume) {
    std::vector<uint32_t> v(volume);
    std::iota(v.begin(), v.end(), iter * 0x12345678u);
    return v;
}

// Build the per-coord consumer MeshWorkload.
tt::tt_metal::distributed::MeshWorkload build_worker_workload(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::H2DStreamService& service,
    const tt::tt_metal::Tensor& output_tensor,
    const CoreRange& worker_cores) {
    const tt::tt_metal::Tensor& input_tensor = service.get_backing_tensor();
    auto* input_buf = input_tensor.buffer();
    auto* output_buf = output_tensor.buffer();
    TT_FATAL(input_buf != nullptr, "build_worker_workload: input tensor has no buffer");
    TT_FATAL(output_buf != nullptr, "build_worker_workload: output tensor has no buffer");

    const uint32_t page_size = input_buf->page_size();
    const uint32_t num_pages = input_buf->num_pages();

    const uint32_t num_workers = (worker_cores.end_coord.x - worker_cores.start_coord.x + 1) *
                                 (worker_cores.end_coord.y - worker_cores.start_coord.y + 1);
    TT_FATAL(num_workers > 0, "build_worker_workload: worker_cores must contain at least one core");
    TT_FATAL(
        num_pages % num_workers == 0,
        "build_worker_workload: tensor page count ({}) must be divisible by num_workers ({})",
        num_pages,
        num_workers);
    const uint32_t pages_per_worker = num_pages / num_workers;

    const uint32_t data_ready_sem_addr = static_cast<uint32_t>(service.get_data_ready_sem_addr());
    const uint32_t input_tensor_addr = static_cast<uint32_t>(input_buf->address());
    const uint32_t output_tensor_addr = static_cast<uint32_t>(output_buf->address());

    const auto& topology = input_tensor.tensor_topology();
    const auto& coords = topology.mesh_coords();
    TT_FATAL(!coords.empty(), "build_worker_workload: tensor topology has no coords");
    const tt::tt_metal::Buffer* sample_dbuf = input_tensor.mesh_buffer().get_device_buffer(coords.front());
    auto accessor_args = tt::tt_metal::TensorAccessorArgs(*sample_dbuf);
    auto accessor_compile_args = accessor_args.get_compile_time_args();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::distributed::MeshWorkload worker_workload;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);

        const CoreCoord service_logical = service.get_service_core(coord);
        const CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t consumed_counter_addr = static_cast<uint32_t>(service.get_consumed_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();

        auto cb_cfg = tt::tt_metal::CircularBufferConfig(page_size, {{scratch_cb_index, tt::DataFormat::UInt32}})
                          .set_page_size(scratch_cb_index, page_size);
        tt::tt_metal::CreateCircularBuffer(program, worker_cores, cb_cfg);

        std::vector<uint32_t> ct_args = {
            data_ready_sem_addr,
            input_tensor_addr,
            output_tensor_addr,
            page_size,
            static_cast<uint32_t>(scratch_cb_index),
            // Metadata copy block (indices 5..8). Disabled in this test.
            0u,
            0u,
            0u,
            0u,
        };
        ct_args.insert(ct_args.end(), accessor_compile_args.begin(), accessor_compile_args.end());

        auto kernel_handle = tt::tt_metal::CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/persistent_h2d_worker_test.cpp",
            worker_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = ct_args,
            });

        uint32_t worker_idx = 0;
        for (uint32_t y = worker_cores.start_coord.y; y <= worker_cores.end_coord.y; ++y) {
            for (uint32_t x = worker_cores.start_coord.x; x <= worker_cores.end_coord.x; ++x) {
                const CoreCoord core{x, y};
                const uint32_t start_page = worker_idx * pages_per_worker;
                const uint32_t end_page = start_page + pages_per_worker;
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    kernel_handle,
                    core,
                    {
                        start_page,
                        end_page,
                        consumed_counter_addr,
                        static_cast<uint32_t>(service_phys.x),
                        static_cast<uint32_t>(service_phys.y),
                    });
                ++worker_idx;
            }
        }
        worker_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return worker_workload;
}

// One case to run end-to-end across processes.
struct CrossProcessCase {
    ttnn::Shape global_shape;
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    uint32_t scratch_cb_size_bytes;
    uint32_t fifo_size_bytes;
    CoreRange worker_cores;
    uint32_t num_iterations;
};

// Rank-0 (owner) body.
void run_owner(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const CrossProcessCase& cs,
    const std::string& service_id) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const TensorSpec global_spec(cs.global_shape, tensor_layout);

    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = cs.fifo_size_bytes,
        .scratch_cb_size_bytes = cs.scratch_cb_size_bytes,
        .socket_mode = H2DMode::DEVICE_PULL,
        .worker_cores = cs.worker_cores,
        .metadata_size_bytes = 0,
    };
    tt::tt_metal::H2DStreamService service(mesh_device, std::move(cfg));

    const auto& backing = service.get_backing_tensor();
    auto output_tensor =
        tt::tt_metal::create_device_tensor(backing.tensor_spec(), mesh_device.get(), backing.tensor_topology());

    auto worker_workload = build_worker_workload(mesh_device, service, output_tensor, cs.worker_cores);

    const auto descriptor_path = service.export_descriptor(service_id);
    ASSERT_FALSE(descriptor_path.empty());

    auto verify_mapper =
        ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});

    const size_t volume = cs.global_shape.volume();
    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        SCOPED_TRACE(::testing::Message() << "iter=" << iter);

        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), worker_workload, /*blocking=*/false);
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());

        auto data = make_iter_data(iter, volume);
        auto distributed_host =
            ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(data, global_spec), *verify_mapper);
        const auto& dhb = distributed_host.host_storage().host_tensor().buffer();

        auto subs = ttnn::distributed::get_device_tensors(output_tensor);
        ASSERT_EQ(subs.size(), mesh_device->num_devices());
        for (auto& sub : subs) {
            const auto coords = sub.device_storage().get_coords();
            ASSERT_EQ(coords.size(), 1u) << "expected one coord per device sub-tensor";
            const auto& coord = coords[0];

            ASSERT_TRUE(dhb.is_local(coord)) << "verify mapper has no shard for coord " << coord;
            auto shard_opt = dhb.get_shard(coord);
            ASSERT_TRUE(shard_opt.has_value()) << "verify mapper shard not populated at coord " << coord;
            auto exp_bytes = shard_opt->view_bytes();
            const auto* exp_begin = reinterpret_cast<const uint32_t*>(exp_bytes.data());
            std::vector<uint32_t> expected(exp_begin, exp_begin + exp_bytes.size() / sizeof(uint32_t));

            auto readback = sub.to_vector<uint32_t>();
            ASSERT_EQ(readback.size(), expected.size()) << "size mismatch at iter=" << iter << " coord=" << coord;
            EXPECT_EQ(readback, expected) << "contents mismatch at iter=" << iter << " coord=" << coord;
        }
    }
}

// Rank-1 (connector) body. Holds no MeshDevice — only PCIe writes through
// each connected H2DSocket.
void run_connector(const std::string& service_id, uint32_t num_iterations, size_t volume) {
    auto service = tt::tt_metal::H2DStreamService::connect(service_id, /*timeout_ms=*/30000);
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        auto data = make_iter_data(iter, volume);
        auto bytes = ttsl::Span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size() * sizeof(uint32_t));
        service->forward_to_tensor(bytes);
    }
    service->barrier();
}

// MPI fixture: rank 0 opens the MeshDevice and broadcasts its shape to rank 1 (no device).
class CrossProcessH2DStreamServiceFixture : public ::tt::tt_metal::MeshDeviceFixtureBase {
protected:
    CrossProcessH2DStreamServiceFixture() :
        ::tt::tt_metal::MeshDeviceFixtureBase(::tt::tt_metal::MeshDeviceFixtureBase::Config{}) {}

    void SetUp() override {
        ASSERT_EQ(g_world_size, 2) << "This test requires exactly 2 MPI ranks";
        rank_ = g_world_rank;
        if (rank_ == 0) {
            ::tt::tt_metal::MeshDeviceFixtureBase::SetUp();
        }

        // Broadcast the mesh shape from rank 0 to rank 1, encoded as [dims, dim_0, dim_1, ...].
        const auto& world = g_cross_rank_world;
        ASSERT_NE(world, nullptr) << "g_cross_rank_world not initialized in main";
        constexpr size_t kMaxDims = 4;
        std::array<uint32_t, kMaxDims + 1> shape_msg{};
        if (rank_ == 0) {
            const auto& shape = mesh_device_->shape();
            shape_msg[0] = static_cast<uint32_t>(shape.dims());
            ASSERT_LE(shape.dims(), kMaxDims) << "MeshShape has more dims than wire format supports";
            for (size_t i = 0; i < shape.dims(); ++i) {
                shape_msg[i + 1] = shape[i];
            }
        }
        world->broadcast(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(shape_msg.data()), shape_msg.size() * sizeof(uint32_t)),
            ::tt::tt_metal::distributed::multihost::Rank{0});
        const uint32_t dims = shape_msg[0];
        ttsl::SmallVector<uint32_t> shape_dims(dims);
        for (uint32_t i = 0; i < dims; ++i) {
            shape_dims[i] = shape_msg[i + 1];
        }
        mesh_shape_ = ::tt::tt_metal::distributed::MeshShape(shape_dims);
    }

    void TearDown() override {
        if (rank_ == 0) {
            ::tt::tt_metal::MeshDeviceFixtureBase::TearDown();
        }
    }

    int rank_ = -1;
    ::tt::tt_metal::distributed::MeshShape mesh_shape_{1, 1};
};

TEST_F(CrossProcessH2DStreamServiceFixture, Sweep) {
    constexpr uint32_t kNumIterations = 100;  // per case
    const CoreRange worker_cores{CoreCoord{0, 0}, CoreCoord{3, 0}};

    // Mesh shape was negotiated in SetUp via MPI broadcast — both ranks
    // enumerate the same cases regardless of who holds the device handle.
    const uint32_t mesh_dims = mesh_shape_.dims();
    const uint32_t mesh_rows = mesh_dims >= 1 ? mesh_shape_[0] : 1;
    const uint32_t mesh_cols = mesh_dims >= 2 ? mesh_shape_[1] : 1;

    // Tensor footprint tier. `per_row` is the innermost-dim length (== one socket page);
    // `per_device_pages` is pages per shard, divisible by num_workers (= 4).
    struct Size {
        const char* label;
        uint32_t per_row;
        uint32_t per_device_pages;
    };
    const Size sizes[] = {
        {"small", 640, 16},    // 2.5 KB page, 40 KB / device
        {"medium", 2048, 64},  // 8 KB page,  512 KB / device
        {"large", 4096, 128},  // 16 KB page, 2 MB / device
    };

    // Chunk plan: cb_pages bounds the on-core L1 scratch CB; fifo_pages bounds
    // the host-side ring. Both are expressed in units of the size tier's page
    // bytes (== per_row * 4).
    struct Chunking {
        uint32_t cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {1, 8, "cb1_fifo8"},
        {2, 8, "cb2_fifo8"},
        {4, 16, "cb4_fifo16"},
        {4, 32, "cb4_fifo32"},
        {8, 64, "cb8_fifo64"},
    };

    struct Pattern {
        const char* label;
        ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    };

    std::vector<Pattern> patterns;
    if (mesh_dims == 2 && mesh_rows >= 2) {
        patterns.push_back(
            Pattern{"ShardRowsReplicateCols", {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}}});
    }
    if (mesh_dims == 2 && mesh_cols >= 2) {
        patterns.push_back(
            Pattern{"ReplicateRowsShardCols", {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}}});
    }
    if (mesh_dims == 2 && mesh_rows >= 2 && mesh_cols >= 2) {
        patterns.push_back(Pattern{"FullShard2D", {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}}});
    }

    // Resolve a pattern's global shape against the active size tier. Each
    // sharded mesh dim multiplies the corresponding tensor dim; replicated
    // mesh dims leave it untouched.
    auto resolve_shape = [&](const Pattern& p, const Size& sz) -> ttnn::Shape {
        uint32_t height = sz.per_device_pages;
        uint32_t width = sz.per_row;
        if (mesh_dims == 2 && p.placements.size() == 2) {
            if (std::holds_alternative<MeshMapperConfig::Shard>(p.placements[0])) {
                const auto& sh = std::get<MeshMapperConfig::Shard>(p.placements[0]);
                if (sh.dim == 2) {
                    height *= mesh_rows;
                } else if (sh.dim == 3) {
                    width *= mesh_rows;
                }
            }
            if (std::holds_alternative<MeshMapperConfig::Shard>(p.placements[1])) {
                const auto& sh = std::get<MeshMapperConfig::Shard>(p.placements[1]);
                if (sh.dim == 2) {
                    height *= mesh_cols;
                } else if (sh.dim == 3) {
                    width *= mesh_cols;
                }
            }
        }
        return ttnn::Shape({1, 1, height, width});
    };

    int case_counter = 0;
    for (const auto& sz : sizes) {
        const uint32_t per_row_bytes = sz.per_row * sizeof(uint32_t);
        for (const auto& pattern : patterns) {
            const ttnn::Shape global_shape = resolve_shape(pattern, sz);
            for (const auto& ch : chunkings) {
                SCOPED_TRACE(
                    ::testing::Message() << "rank=" << rank_ << " size=" << sz.label << " placement=" << pattern.label
                                         << " chunk=" << ch.label);

                // Service IDs must agree across ranks; use a deterministic counter.
                const std::string service_id = "xproc_h2d_stream_" + std::to_string(case_counter++);

                CrossProcessCase cs{
                    .global_shape = global_shape,
                    .placements = pattern.placements,
                    .scratch_cb_size_bytes = ch.cb_pages * per_row_bytes,
                    .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
                    .worker_cores = worker_cores,
                    .num_iterations = kNumIterations,
                };

                if (rank_ == 0) {
                    run_owner(mesh_device_, cs, service_id);
                } else {
                    run_connector(service_id, kNumIterations, cs.global_shape.volume());
                }
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed

int main(int argc, char** argv) {
    using namespace tt::tt_metal::distributed::multihost;

    DistributedContext::create(argc, argv);
    const auto& world = DistributedContext::get_current_world();
    tt::tt_metal::distributed::g_world_rank = *world->rank();
    tt::tt_metal::distributed::g_world_size = *world->size();

    // Stash the cross-rank world before the per-rank split below — the
    // fixture's SetUp broadcasts the mesh shape from rank 0 to rank 1 over
    // this 2-rank communicator.
    tt::tt_metal::distributed::g_cross_rank_world = world;

    // Split per rank so each gets its own MetalContext sub-world.
    auto local_ctx = world->split(Color(tt::tt_metal::distributed::g_world_rank), Key(0));
    DistributedContext::set_current_world(local_ctx);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
