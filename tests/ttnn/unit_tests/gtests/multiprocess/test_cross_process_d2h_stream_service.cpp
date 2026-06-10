// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Cross-process D2HStreamService integration tests (MPI, 2 ranks: owner + connector).
//
// Launch:
//   mpirun -np 2 ./build_Release/test/tt_metal/distributed/cross_process_d2h_stream_service_test

#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/d2h_socket_service.hpp"
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
std::shared_ptr<::tt::tt_metal::distributed::multihost::DistributedContext> g_cross_rank_world;

std::vector<uint32_t> make_iter_data(uint32_t iter, size_t volume) {
    std::vector<uint32_t> v(volume);
    std::iota(v.begin(), v.end(), iter * 0x12345678u);
    return v;
}

std::vector<uint32_t> make_worker_fill_pattern(uint32_t fill_seed, uint32_t page_size, uint32_t num_pages) {
    std::vector<uint32_t> out(num_pages * (page_size / sizeof(uint32_t)));
    size_t idx = 0;
    for (uint32_t p = 0; p < num_pages; ++p) {
        for (uint32_t i = 0; i < page_size / sizeof(uint32_t); ++i) {
            out[idx++] = fill_seed + p + i;
        }
    }
    return out;
}

std::vector<uint8_t> make_metadata_pattern(uint32_t iter, uint32_t metadata_size_bytes) {
    std::vector<uint8_t> out(metadata_size_bytes);
    for (uint32_t i = 0; i < metadata_size_bytes; ++i) {
        out[i] = static_cast<uint8_t>((iter * 17 + i) & 0xff);
    }
    return out;
}

uint32_t worker_idx_from_xy(const tt::tt_metal::CoreRange& worker_cores, uint32_t x, uint32_t y) {
    return (y - worker_cores.start_coord.y) * (worker_cores.end_coord.x - worker_cores.start_coord.x + 1) +
           (x - worker_cores.start_coord.x);
}

void write_worker_metadata(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::D2HStreamService& service,
    const tt::tt_metal::CoreRange& worker_cores,
    const std::vector<uint8_t>& metadata) {
    const uint32_t worker_metadata_addr = static_cast<uint32_t>(service.get_worker_metadata_addr());
    for (const auto& coord : service.get_backing_tensor().tensor_topology().mesh_coords()) {
        auto* device = mesh_device->get_device(coord);
        for (uint32_t y = worker_cores.start_coord.y; y <= worker_cores.end_coord.y; ++y) {
            for (uint32_t x = worker_cores.start_coord.x; x <= worker_cores.end_coord.x; ++x) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    device, tt::tt_metal::CoreCoord{x, y}, worker_metadata_addr, metadata, tt::CoreType::WORKER);
            }
        }
    }
}

tt::tt_metal::distributed::MeshWorkload build_d2h_metadata_worker_workload(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::D2HStreamService& service,
    const tt::tt_metal::CoreRange& worker_cores,
    uint32_t fill_seed) {
    const tt::tt_metal::Tensor& backing = service.get_backing_tensor();
    auto* backing_buf = backing.buffer();
    TT_FATAL(backing_buf != nullptr, "build_d2h_metadata_worker_workload: backing tensor has no buffer");

    const uint32_t page_size = backing_buf->page_size();
    const uint32_t num_pages = backing_buf->num_pages();
    const uint32_t num_workers = (worker_cores.end_coord.x - worker_cores.start_coord.x + 1) *
                                 (worker_cores.end_coord.y - worker_cores.start_coord.y + 1);
    TT_FATAL(num_pages % num_workers == 0, "tensor page count must divide num_workers");
    TT_FATAL(num_workers >= 1, "metadata forwarding requires at least one worker core");
    const uint32_t pages_per_worker = num_pages / num_workers;

    const uint32_t transfer_done_sem_addr = static_cast<uint32_t>(service.get_transfer_done_sem_addr());
    const uint32_t backing_tensor_addr = static_cast<uint32_t>(backing_buf->address());

    const auto& coords = backing.tensor_topology().mesh_coords();
    const tt::tt_metal::Buffer* sample_dbuf = backing.mesh_buffer().get_device_buffer(coords.front());
    auto accessor_compile_args = tt::tt_metal::TensorAccessorArgs(*sample_dbuf).get_compile_time_args();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::distributed::MeshWorkload workloads;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);
        const tt::tt_metal::CoreCoord service_logical = service.get_service_core(coord);
        const tt::tt_metal::CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t write_ack_counter_addr = static_cast<uint32_t>(service.get_write_ack_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();
        auto cb_cfg = tt::tt_metal::CircularBufferConfig(page_size, {{scratch_cb_index, tt::DataFormat::UInt32}})
                          .set_page_size(scratch_cb_index, page_size);
        tt::tt_metal::CreateCircularBuffer(program, tt::tt_metal::CoreRangeSet(worker_cores), cb_cfg);

        const tt::tt_metal::CoreCoord metadata_master = service.get_metadata_master_core();
        const uint32_t worker_metadata_l1_addr = static_cast<uint32_t>(service.get_worker_metadata_addr());
        const uint32_t metadata_input_addr = static_cast<uint32_t>(service.get_metadata_input_addr(coord));

        auto worker_kernel = tt::tt_metal::CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/persistent_d2h_worker.cpp",
            worker_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = [&] {
                    std::vector<uint32_t> ct_args = {
                        transfer_done_sem_addr,
                        backing_tensor_addr,
                        page_size,
                        static_cast<uint32_t>(scratch_cb_index),
                        static_cast<uint32_t>(service.metadata_size_bytes()),
                        worker_metadata_l1_addr,
                    };
                    ct_args.insert(ct_args.end(), accessor_compile_args.begin(), accessor_compile_args.end());
                    return ct_args;
                }()});
        for (uint32_t y = worker_cores.start_coord.y; y <= worker_cores.end_coord.y; ++y) {
            for (uint32_t x = worker_cores.start_coord.x; x <= worker_cores.end_coord.x; ++x) {
                const tt::tt_metal::CoreCoord core{x, y};
                const uint32_t grid_idx = worker_idx_from_xy(worker_cores, x, y);
                const uint32_t start_page = grid_idx * pages_per_worker;
                const bool is_master = core.x == metadata_master.x && core.y == metadata_master.y;
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    worker_kernel,
                    core,
                    {start_page,
                     start_page + pages_per_worker,
                     fill_seed,
                     service_phys.x,
                     service_phys.y,
                     write_ack_counter_addr,
                     is_master ? 1u : 0u,
                     is_master ? metadata_input_addr : 0u});
            }
        }

        workloads.add_program(tt::tt_metal::distributed::MeshCoordinateRange(coord), std::move(program));
    }
    return workloads;
}

struct CrossProcessCase {
    ttnn::Shape global_shape;
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    uint32_t scratch_cb_size_bytes;
    uint32_t fifo_size_bytes;
    uint32_t metadata_size_bytes;
    uint32_t num_iterations;
};

void run_owner(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const CrossProcessCase& cs,
    const std::string& service_id) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const TensorSpec global_spec(cs.global_shape, tensor_layout);

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .fifo_size_bytes = cs.fifo_size_bytes,
        .scratch_cb_size_bytes = cs.scratch_cb_size_bytes,
    };

    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));
    service.export_descriptor(service_id);

    g_cross_rank_world->barrier();
    std::vector<std::byte> discard(service.payload_size_bytes());
    service.read_from_tensor(discard);
    service.barrier();
    g_cross_rank_world->barrier();

    auto mapper = ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
    const size_t volume = cs.global_shape.volume();

    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        auto data = make_iter_data(iter, volume);
        auto host_src = ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(data, global_spec), *mapper);
        auto& backing = const_cast<Tensor&>(service.get_backing_tensor());
        copy_to_device(host_src, backing);
        Finish(mesh_device->mesh_command_queue());
        service.notify_backing_ready();
        g_cross_rank_world->barrier();
        g_cross_rank_world->barrier();
    }
    g_cross_rank_world->barrier();
}

TensorSpec make_global_spec(const CrossProcessCase& cs) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    return TensorSpec(cs.global_shape, tensor_layout);
}

void verify_sharded_readback(
    const CrossProcessCase& cs,
    const ::tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::vector<uint32_t>& src,
    const Tensor& read_host) {
    const TensorSpec global_spec = make_global_spec(cs);
    auto verify_mapper =
        ttnn::distributed::create_mesh_mapper(mesh_shape, MeshMapperConfig{.placements = cs.placements});
    auto expected_host =
        ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *verify_mapper);
    const auto& exp_dhb = expected_host.host_storage().host_tensor().buffer();
    const auto& act_dhb = read_host.host_storage().host_tensor().buffer();
    for (const auto& coord : expected_host.tensor_topology().mesh_coords()) {
        auto exp_shard = exp_dhb.get_shard(coord);
        auto act_shard = act_dhb.get_shard(coord);
        ASSERT_TRUE(exp_shard.has_value()) << "expected shard missing at coord " << coord;
        ASSERT_TRUE(act_shard.has_value()) << "actual shard missing at coord " << coord;
        const auto* exp_ptr = reinterpret_cast<const uint32_t*>(exp_shard->view_bytes().data());
        const auto* act_ptr = reinterpret_cast<const uint32_t*>(act_shard->view_bytes().data());
        std::vector<uint32_t> expected(exp_ptr, exp_ptr + exp_shard->view_bytes().size() / sizeof(uint32_t));
        std::vector<uint32_t> actual(act_ptr, act_ptr + act_shard->view_bytes().size() / sizeof(uint32_t));
        EXPECT_EQ(actual, expected) << "shard mismatch at coord " << coord;
    }
}

// Sharded cases use the Tensor read path; the bytes path needs a composer the connector lacks.
void run_owner_sharded(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const CrossProcessCase& cs,
    const std::string& service_id) {
    const TensorSpec global_spec = make_global_spec(cs);

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .fifo_size_bytes = cs.fifo_size_bytes,
        .scratch_cb_size_bytes = cs.scratch_cb_size_bytes,
    };

    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));
    service.export_descriptor(service_id);

    g_cross_rank_world->barrier();
    auto drain_mapper =
        ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
    auto drain_host = ttnn::distributed::distribute_tensor(
        Tensor::from_vector<uint32_t>(std::vector<uint32_t>(cs.global_shape.volume(), 0), global_spec), *drain_mapper);
    service.read_from_tensor(drain_host);
    service.barrier();
    g_cross_rank_world->barrier();

    auto mapper = ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
    const size_t volume = cs.global_shape.volume();

    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        auto data = make_iter_data(iter, volume);
        auto host_src = ttnn::distributed::distribute_tensor(Tensor::from_vector<uint32_t>(data, global_spec), *mapper);
        auto& backing = const_cast<Tensor&>(service.get_backing_tensor());
        copy_to_device(host_src, backing);
        Finish(mesh_device->mesh_command_queue());
        service.notify_backing_ready();
        g_cross_rank_world->barrier();
        g_cross_rank_world->barrier();
    }
    g_cross_rank_world->barrier();
}

void run_connector_sharded(
    const CrossProcessCase& cs,
    const ::tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::string& service_id) {
    auto service = tt::tt_metal::D2HStreamService::connect(service_id, /*timeout_ms=*/30000);
    const TensorSpec global_spec = make_global_spec(cs);
    const size_t volume = cs.global_shape.volume();

    g_cross_rank_world->barrier();
    auto drain_mapper =
        ttnn::distributed::create_mesh_mapper(mesh_shape, MeshMapperConfig{.placements = cs.placements});
    auto drain_host = ttnn::distributed::distribute_tensor(
        Tensor::from_vector<uint32_t>(std::vector<uint32_t>(volume, 0), global_spec), *drain_mapper);
    service->read_from_tensor(drain_host);
    service->barrier();
    g_cross_rank_world->barrier();

    auto read_mapper = ttnn::distributed::create_mesh_mapper(mesh_shape, MeshMapperConfig{.placements = cs.placements});
    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        g_cross_rank_world->barrier();
        auto read_host = ttnn::distributed::distribute_tensor(
            Tensor::from_vector<uint32_t>(std::vector<uint32_t>(volume, 0), global_spec), *read_mapper);
        service->read_from_tensor(read_host);
        service->barrier();
        verify_sharded_readback(cs, mesh_shape, make_iter_data(iter, volume), read_host);
        g_cross_rank_world->barrier();
    }
    g_cross_rank_world->barrier();
}

void run_connector(const CrossProcessCase& cs, const std::string& service_id) {
    auto service = tt::tt_metal::D2HStreamService::connect(service_id, /*timeout_ms=*/30000);
    const size_t volume = cs.global_shape.volume();
    const size_t payload = service->payload_size_bytes();

    g_cross_rank_world->barrier();
    std::vector<std::byte> discard(payload);
    service->read_from_tensor(discard);
    service->barrier();
    g_cross_rank_world->barrier();

    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        g_cross_rank_world->barrier();
        std::vector<std::byte> out(payload);
        service->read_from_tensor(out);
        service->barrier();
        auto expected = make_iter_data(iter, volume);
        std::vector<uint32_t> read_vec(out.size() / sizeof(uint32_t));
        std::memcpy(read_vec.data(), out.data(), out.size());
        EXPECT_EQ(read_vec, expected) << "connector read mismatch at iter " << iter;
        g_cross_rank_world->barrier();
    }
    g_cross_rank_world->barrier();
}

void run_owner_metadata(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const CrossProcessCase& cs,
    const std::string& service_id,
    const tt::tt_metal::CoreRange& worker_cores,
    const tt::tt_metal::CoreCoord& metadata_master) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const TensorSpec global_spec(cs.global_shape, tensor_layout);

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = ttnn::distributed::create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .fifo_size_bytes = cs.fifo_size_bytes,
        .scratch_cb_size_bytes = cs.scratch_cb_size_bytes,
        .worker_cores = worker_cores,
        .metadata_master_core = metadata_master,
        .metadata_size_bytes = cs.metadata_size_bytes,
    };

    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));
    service.export_descriptor(service_id);

    constexpr uint32_t kFillSeed = 0;
    auto worker_workload = build_d2h_metadata_worker_workload(mesh_device, service, worker_cores, kFillSeed);

    g_cross_rank_world->barrier();
    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        const auto metadata = make_metadata_pattern(iter, cs.metadata_size_bytes);
        write_worker_metadata(mesh_device, service, worker_cores, metadata);
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), worker_workload, /*blocking=*/false);
        g_cross_rank_world->barrier();
        g_cross_rank_world->barrier();
        Finish(mesh_device->mesh_command_queue());
    }
    g_cross_rank_world->barrier();
}

void run_connector_metadata(const CrossProcessCase& cs, const std::string& service_id) {
    auto service = tt::tt_metal::D2HStreamService::connect(service_id, /*timeout_ms=*/30000);
    const size_t payload = service->payload_size_bytes();
    ASSERT_EQ(service->metadata_size_bytes(), cs.metadata_size_bytes);

    const uint32_t per_row = cs.global_shape[3];
    const uint32_t page_size = per_row * sizeof(uint32_t);
    const uint32_t num_pages = static_cast<uint32_t>((payload / sizeof(uint32_t)) / per_row);
    const auto expected_payload = make_worker_fill_pattern(/*fill_seed=*/0, page_size, num_pages);

    g_cross_rank_world->barrier();
    for (uint32_t iter = 0; iter < cs.num_iterations; ++iter) {
        g_cross_rank_world->barrier();
        std::vector<std::byte> out(payload);
        std::vector<std::byte> metadata_out(cs.metadata_size_bytes);
        service->read_from_tensor(out, metadata_out);
        service->barrier();

        std::vector<uint32_t> read_vec(out.size() / sizeof(uint32_t));
        std::memcpy(read_vec.data(), out.data(), out.size());
        EXPECT_EQ(read_vec, expected_payload) << "connector payload mismatch at iter " << iter;

        const auto expected_metadata = make_metadata_pattern(iter, cs.metadata_size_bytes);
        std::vector<uint8_t> read_metadata(metadata_out.size());
        std::memcpy(read_metadata.data(), metadata_out.data(), metadata_out.size());
        EXPECT_EQ(read_metadata, expected_metadata) << "connector metadata mismatch at iter " << iter;

        g_cross_rank_world->barrier();
    }
    g_cross_rank_world->barrier();
}

class CrossProcessD2HStreamServiceFixture : public ::tt::tt_metal::MeshDeviceFixtureBase {
protected:
    CrossProcessD2HStreamServiceFixture() :
        ::tt::tt_metal::MeshDeviceFixtureBase(::tt::tt_metal::MeshDeviceFixtureBase::Config{}) {}

    void SetUp() override {
        ASSERT_EQ(g_world_size, 2) << "This test requires exactly 2 MPI ranks";
        rank_ = g_world_rank;
        if (rank_ == 0) {
            ::tt::tt_metal::MeshDeviceFixtureBase::SetUp();
        }

        // Broadcast over g_cross_rank_world; the per-rank split in main is size-1.
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

TEST_F(CrossProcessD2HStreamServiceFixture, ReplicatedMinimal) {
    constexpr uint32_t kNumIterations = 10;
    const std::string service_id = "cross_process_d2h_minimal";
    ttsl::SmallVector<MeshMapperConfig::Placement> placements(mesh_shape_.dims(), MeshMapperConfig::Replicate{});

    const uint32_t per_row_bytes = 640 * sizeof(uint32_t);
    CrossProcessCase cs{
        .global_shape = ttnn::Shape({1, 1, 16, 640}),
        .placements = placements,
        .scratch_cb_size_bytes = 4 * per_row_bytes,
        .fifo_size_bytes = 16 * per_row_bytes,
        .metadata_size_bytes = 0,
        .num_iterations = kNumIterations,
    };
    if (rank_ == 0) {
        run_owner(mesh_device_, cs, service_id);
    } else {
        run_connector(cs, service_id);
    }
}

TEST_F(CrossProcessD2HStreamServiceFixture, ReplicatedMetadata) {
    constexpr uint32_t kNumIterations = 10;
    const std::string service_id = "cross_process_d2h_metadata";
    ttsl::SmallVector<MeshMapperConfig::Placement> placements(mesh_shape_.dims(), MeshMapperConfig::Replicate{});

    const tt::tt_metal::CoreRange worker_cores({0, 0}, {3, 0});  // 4 workers; 16 % 4 == 0
    const tt::tt_metal::CoreCoord metadata_master{0, 0};

    const uint32_t per_row_bytes = 640 * sizeof(uint32_t);
    CrossProcessCase cs{
        .global_shape = ttnn::Shape({1, 1, 16, 640}),
        .placements = placements,
        .scratch_cb_size_bytes = 4 * per_row_bytes,
        .fifo_size_bytes = 16 * per_row_bytes,
        .metadata_size_bytes = 64,
        .num_iterations = kNumIterations,
    };
    if (rank_ == 0) {
        run_owner_metadata(mesh_device_, cs, service_id, worker_cores, metadata_master);
    } else {
        run_connector_metadata(cs, service_id);
    }
}

TEST_F(CrossProcessD2HStreamServiceFixture, Sweep) {
    constexpr uint32_t kNumIterations = 10;  // per case

    struct Size {
        const char* label;
        uint32_t per_row;
        uint32_t per_device_pages;
    };
    const Size sizes[] = {
        {"small", 640, 8},     // 2.5 KB page, 20 KB / device
        {"medium", 1024, 32},  // 4 KB page, 128 KB / device
    };

    struct Chunking {
        uint32_t cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {1, 8, "cb1_fifo8"},
        {4, 16, "cb4_fifo16"},
    };

    ttsl::SmallVector<MeshMapperConfig::Placement> all_replicate(mesh_shape_.dims(), MeshMapperConfig::Replicate{});

    int case_counter = 0;
    for (const auto& sz : sizes) {
        const uint32_t per_row_bytes = sz.per_row * sizeof(uint32_t);
        const ttnn::Shape global_shape({1, 1, sz.per_device_pages, sz.per_row});
        for (const auto& ch : chunkings) {
            SCOPED_TRACE(::testing::Message() << "rank=" << rank_ << " size=" << sz.label << " chunk=" << ch.label);

            const std::string service_id = "xproc_d2h_stream_" + std::to_string(case_counter++);

            std::cout << "[xproc-d2h] rank=" << rank_ << " case=" << service_id << " size=" << sz.label
                      << " chunk=" << ch.label << " shape=" << global_shape << std::endl;

            CrossProcessCase cs{
                .global_shape = global_shape,
                .placements = all_replicate,
                .scratch_cb_size_bytes = ch.cb_pages * per_row_bytes,
                .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
                .metadata_size_bytes = 0,
                .num_iterations = kNumIterations,
            };

            if (rank_ == 0) {
                run_owner(mesh_device_, cs, service_id);
            } else {
                run_connector(cs, service_id);
            }
        }
    }
}

TEST_F(CrossProcessD2HStreamServiceFixture, Sharded_Sweep) {
    if (mesh_shape_.dims() != 2) {
        GTEST_SKIP() << "Sharded_Sweep requires a 2D mesh; got " << mesh_shape_;
    }
    if (mesh_shape_[0] < 2 && mesh_shape_[1] < 2) {
        GTEST_SKIP() << "no shardable mesh axis on " << mesh_shape_;
    }

    constexpr uint32_t kNumIterations = 10;
    const uint32_t num_rows = mesh_shape_[0];
    const uint32_t num_cols = mesh_shape_[1];
    const uint32_t per_row = 640;
    const uint32_t per_row_bytes = per_row * sizeof(uint32_t);

    struct Row {
        uint32_t N;
        uint32_t scratch_cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        {1, 1, 1, "N1_cb1_fifo1"},
        {16, 4, 16, "N16_cb4_fifo16"},
        {7, 4, 8, "N7_cb4_fifo8_prime_pages"},
    };

    int case_counter = 0;
    const auto run_pattern = [&](const char* label,
                                 const ttsl::SmallVector<MeshMapperConfig::Placement>& placements,
                                 auto make_global_shape) {
        for (const auto& row : rows) {
            SCOPED_TRACE(::testing::Message() << "rank=" << rank_ << " pattern=" << label << " row=" << row.label);

            const ttnn::Shape global_shape = make_global_shape(row.N);

            const std::string service_id = "xproc_d2h_sharded_" + std::to_string(case_counter++);
            std::cout << "[xproc-d2h-sharded] rank=" << rank_ << " case=" << service_id << " pattern=" << label
                      << " row=" << row.label << " shape=" << global_shape << std::endl;

            CrossProcessCase cs{
                .global_shape = global_shape,
                .placements = placements,
                .scratch_cb_size_bytes = row.scratch_cb_pages * per_row_bytes,
                .fifo_size_bytes = row.fifo_pages * per_row_bytes,
                .metadata_size_bytes = 0,
                .num_iterations = kNumIterations,
            };

            if (rank_ == 0) {
                run_owner_sharded(mesh_device_, cs, service_id);
            } else {
                run_connector_sharded(cs, mesh_shape_, service_id);
            }
        }
    };

    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols", {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}}, [&](uint32_t N) {
                return ttnn::Shape({1, 1, N, num_rows * per_row});
            });
    }
    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols", {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}}, [&](uint32_t N) {
                return ttnn::Shape({1, 1, N, num_cols * per_row});
            });
    }
    if (num_rows >= 2 && num_cols >= 2) {
        run_pattern("FullShard2D", {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}}, [&](uint32_t N) {
            return ttnn::Shape({1, 1, num_rows * N, num_cols * per_row});
        });
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

    tt::tt_metal::distributed::g_cross_rank_world = world;

    auto local_ctx = world->split(Color(tt::tt_metal::distributed::g_world_rank), Key(0));
    DistributedContext::set_current_world(local_ctx);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
