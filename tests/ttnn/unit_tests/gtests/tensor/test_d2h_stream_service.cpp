// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <vector>

#include "gtest/gtest.h"

#include <tt_stl/assert.hpp>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <ttnn/distributed/distributed_configs.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/d2h_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/outbound_socket_service_sync/outbound_socket_service_sync.hpp"

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
using ::tt::tt_metal::distributed::MeshMapperConfig;

enum class OutputPath {
    Tensor,
    Bytes,
};

struct D2HServiceCase {
    ttnn::Shape global_shape;
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    uint32_t max_socket_page_size_bytes = 0;
    uint32_t fifo_size_bytes = 0;
};

ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    return ttsl::SmallVector<MeshMapperConfig::Placement>(mesh_device.shape().dims(), MeshMapperConfig::Replicate{});
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

uint32_t worker_idx_from_xy(const tt::tt_metal::CoreRange& worker_cores, uint32_t x, uint32_t y) {
    return (y - worker_cores.start_coord.y) * (worker_cores.end_coord.x - worker_cores.start_coord.x + 1) +
           (x - worker_cores.start_coord.x);
}

tt::tt_metal::distributed::MeshWorkload build_d2h_worker_workload(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::D2HStreamService& service,
    const tt::tt_metal::CoreRange& worker_cores,
    uint32_t fill_seed) {
    const tt::tt_metal::Tensor& backing = service.get_backing_tensor();
    auto* backing_buf = backing.buffer();
    TT_FATAL(backing_buf != nullptr, "build_d2h_worker_workload: backing tensor has no buffer");

    const uint32_t page_size = backing_buf->page_size();
    const uint32_t num_pages = backing_buf->num_pages();
    const uint32_t num_workers = (worker_cores.end_coord.x - worker_cores.start_coord.x + 1) *
                                 (worker_cores.end_coord.y - worker_cores.start_coord.y + 1);
    TT_FATAL(num_pages % num_workers == 0, "tensor page count must divide num_workers");
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

        const bool metadata_enabled = service.metadata_size_bytes() > 0;
        const uint32_t metadata_size = metadata_enabled ? service.metadata_size_bytes() : 0u;
        const uint32_t worker_metadata_l1_addr =
            metadata_enabled ? static_cast<uint32_t>(service.get_worker_metadata_addr()) : 0u;
        const tt::tt_metal::CoreCoord metadata_master =
            metadata_enabled ? service.get_metadata_master_core() : tt::tt_metal::CoreCoord{0, 0};
        const uint32_t metadata_input_addr =
            metadata_enabled ? static_cast<uint32_t>(service.get_metadata_input_addr(coord)) : 0u;

        auto worker_kernel = tt::tt_metal::CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/tensor/kernels/persistent_d2h_worker.cpp",
            worker_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args =
                    [&] {
                        std::vector<uint32_t> ct_args = {
                            transfer_done_sem_addr,
                            backing_tensor_addr,
                            page_size,
                            static_cast<uint32_t>(scratch_cb_index),
                            metadata_size,
                            worker_metadata_l1_addr,
                        };
                        ct_args.insert(ct_args.end(), accessor_compile_args.begin(), accessor_compile_args.end());
                        return ct_args;
                    }(),
            });

        for (uint32_t y = worker_cores.start_coord.y; y <= worker_cores.end_coord.y; ++y) {
            for (uint32_t x = worker_cores.start_coord.x; x <= worker_cores.end_coord.x; ++x) {
                const tt::tt_metal::CoreCoord core{x, y};
                const uint32_t grid_idx = worker_idx_from_xy(worker_cores, x, y);
                const uint32_t start_page = grid_idx * pages_per_worker;
                const bool is_master = metadata_enabled && core.x == metadata_master.x && core.y == metadata_master.y;
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

void run_d2h_worker_sync_case(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const D2HServiceCase& cs,
    const tt::tt_metal::CoreRange& worker_cores,
    std::optional<tt::tt_metal::CoreCoord> metadata_master = std::nullopt,
    uint32_t num_iterations = 5,
    uint32_t metadata_size_bytes = 0) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(cs.global_shape, tensor_layout);

    if (metadata_size_bytes > 0) {
        TT_FATAL(
            metadata_master.has_value(), "run_d2h_worker_sync_case: metadata_master required when metadata enabled");
    }

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .fifo_size_bytes = cs.fifo_size_bytes,
        .max_socket_page_size_bytes = cs.max_socket_page_size_bytes,
        .worker_cores = worker_cores,
        .metadata_master_core = metadata_master,
        .metadata_size_bytes = metadata_size_bytes,
    };

    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));
    constexpr uint32_t kFillSeed = 0;
    auto worker_workload = build_d2h_worker_workload(mesh_device, service, worker_cores, kFillSeed);

    const uint32_t page_size = service.get_backing_tensor().buffer()->page_size();
    const uint32_t num_pages = service.get_backing_tensor().buffer()->num_pages();
    const auto expected = make_worker_fill_pattern(kFillSeed, page_size, num_pages);

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        const auto expected_metadata = make_metadata_pattern(iter, metadata_size_bytes);
        if (metadata_size_bytes > 0) {
            write_worker_metadata(mesh_device, service, worker_cores, expected_metadata);
        }

        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), worker_workload, /*blocking=*/false);
        std::vector<std::byte> out(service.payload_size_bytes());
        std::vector<std::byte> metadata_out(metadata_size_bytes);
        service.read_from_tensor(out, metadata_out);
        service.barrier();
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());
        std::vector<uint32_t> read_vec(out.size() / sizeof(uint32_t));
        std::memcpy(read_vec.data(), out.data(), out.size());
        EXPECT_EQ(read_vec, expected) << "worker-sync read mismatch";
        if (metadata_size_bytes > 0) {
            std::vector<uint8_t> read_metadata(metadata_out.size());
            std::memcpy(read_metadata.data(), metadata_out.data(), metadata_out.size());
            EXPECT_EQ(read_metadata, expected_metadata) << "worker-sync metadata mismatch";
        }
    }
}

void run_d2h_worker_sync_case_per_shard(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const D2HServiceCase& cs,
    const tt::tt_metal::CoreRange& worker_cores,
    std::optional<tt::tt_metal::CoreCoord> metadata_master = std::nullopt,
    uint32_t num_iterations = 5,
    uint32_t metadata_size_bytes = 0) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(cs.global_shape, tensor_layout);

    if (metadata_size_bytes > 0) {
        TT_FATAL(
            metadata_master.has_value(),
            "run_d2h_worker_sync_case_per_shard: metadata_master required when metadata enabled");
    }

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .fifo_size_bytes = cs.fifo_size_bytes,
        .max_socket_page_size_bytes = cs.max_socket_page_size_bytes,
        .worker_cores = worker_cores,
        .metadata_master_core = metadata_master,
        .metadata_size_bytes = metadata_size_bytes,
    };

    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));
    constexpr uint32_t kFillSeed = 0;
    auto worker_workload = build_d2h_worker_workload(mesh_device, service, worker_cores, kFillSeed);

    const uint32_t page_size = service.get_backing_tensor().buffer()->page_size();
    const uint32_t per_device_pages = service.get_backing_tensor().buffer()->num_pages();
    const auto expected_per_device = make_worker_fill_pattern(kFillSeed, page_size, per_device_pages);

    auto read_mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
    const size_t global_volume = cs.global_shape.volume();

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        SCOPED_TRACE(::testing::Message() << "iter=" << iter);
        const auto expected_metadata = make_metadata_pattern(iter, metadata_size_bytes);
        if (metadata_size_bytes > 0) {
            write_worker_metadata(mesh_device, service, worker_cores, expected_metadata);
        }

        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), worker_workload, /*blocking=*/false);

        auto read_host = distribute_tensor(
            Tensor::from_vector<uint32_t>(std::vector<uint32_t>(global_volume, 0), global_spec), *read_mapper);
        std::vector<std::byte> metadata_out(metadata_size_bytes);
        service.read_from_tensor(read_host, metadata_out);
        service.barrier();
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());

        const auto& read_buf = read_host.host_storage().host_tensor().buffer();
        for (const auto& coord : service.get_backing_tensor().tensor_topology().mesh_coords()) {
            auto shard_opt = read_buf.get_shard(coord);
            ASSERT_TRUE(shard_opt.has_value()) << "no shard at coord " << coord;
            const auto* ptr = reinterpret_cast<const uint32_t*>(shard_opt->view_bytes().data());
            std::vector<uint32_t> actual(ptr, ptr + shard_opt->view_bytes().size() / sizeof(uint32_t));
            EXPECT_EQ(actual, expected_per_device) << "per-shard mismatch at coord " << coord;
        }

        if (metadata_size_bytes > 0) {
            std::vector<uint8_t> read_metadata(metadata_out.size());
            std::memcpy(read_metadata.data(), metadata_out.data(), metadata_out.size());
            EXPECT_EQ(read_metadata, expected_metadata) << "worker-sync metadata mismatch";
        }
    }
}

void run_d2h_stream_service_case(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const D2HServiceCase& cs,
    OutputPath output_path,
    uint32_t num_iterations = 10) {
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(cs.global_shape, tensor_layout);

    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements}),
        .fifo_size_bytes = cs.fifo_size_bytes,
        .max_socket_page_size_bytes = cs.max_socket_page_size_bytes,
    };

    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);
    ASSERT_EQ(service.get_sockets().size(), mesh_device->num_devices());

    // Drain the kernel's initial auto-iteration; use Tensor read for sharded cases.
    {
        auto drain_mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
        auto drain_host = distribute_tensor(
            Tensor::from_vector<uint32_t>(std::vector<uint32_t>(cs.global_shape.volume(), 0), global_spec),
            *drain_mapper);
        service.read_from_tensor(drain_host);
        service.barrier();
    }

    auto make_iter_data = [&](uint32_t iter) {
        std::vector<uint32_t> v(cs.global_shape.volume());
        std::iota(v.begin(), v.end(), iter * 0x12345678u);
        return v;
    };

    auto verify = [&](const std::vector<uint32_t>& src, const Tensor& read_host) {
        auto verify_mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
        auto expected_host = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *verify_mapper);
        const auto& exp_dhb = expected_host.host_storage().host_tensor().buffer();
        const auto& act_dhb = read_host.host_storage().host_tensor().buffer();
        for (const auto& coord : expected_host.tensor_topology().mesh_coords()) {
            auto exp_shard = exp_dhb.get_shard(coord);
            auto act_shard = act_dhb.get_shard(coord);
            ASSERT_TRUE(exp_shard.has_value());
            ASSERT_TRUE(act_shard.has_value());
            const auto* exp_ptr = reinterpret_cast<const uint32_t*>(exp_shard->view_bytes().data());
            const auto* act_ptr = reinterpret_cast<const uint32_t*>(act_shard->view_bytes().data());
            std::vector<uint32_t> expected(exp_ptr, exp_ptr + exp_shard->view_bytes().size() / sizeof(uint32_t));
            std::vector<uint32_t> actual(act_ptr, act_ptr + act_shard->view_bytes().size() / sizeof(uint32_t));
            EXPECT_EQ(actual, expected) << "shard mismatch at coord " << coord;
        }
    };

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        SCOPED_TRACE(::testing::Message() << "iteration=" << iter);
        auto src = make_iter_data(iter);

        auto mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
        auto host_src = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *mapper);
        auto& backing = const_cast<Tensor&>(service.get_backing_tensor());
        tt::tt_metal::copy_to_device(host_src, backing);
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());

        if (output_path == OutputPath::Tensor) {
            auto read_host = distribute_tensor(
                Tensor::from_vector<uint32_t>(std::vector<uint32_t>(cs.global_shape.volume(), 0), global_spec),
                *mapper);
            service.read_from_tensor(read_host);
            service.barrier();
            verify(src, read_host);
        } else {
            std::vector<std::byte> out(service.payload_size_bytes());
            service.read_from_tensor(out);
            service.barrier();
            std::vector<uint32_t> read_vec(out.size() / sizeof(uint32_t));
            std::memcpy(read_vec.data(), out.data(), out.size());
            EXPECT_EQ(read_vec, src);
        }
    }
}

// Single-mesh end-to-end exercise of the metadata-only D2H path: build a service with no
// DRAM payload, then per iteration push a distinct record through the worker op and read it
// back on the host, checking the value and cross-chip equality.
void run_d2h_metadata_only_case(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::CoreRange& worker_cores,  // single core -> num_workers == 1
    uint32_t metadata_size_bytes,
    uint32_t fifo_size_bytes,
    uint32_t num_iterations = 5) {
    TT_FATAL(metadata_size_bytes % sizeof(uint32_t) == 0, "metadata must be uint32-aligned");

    // Metadata-only service: global_spec omitted, single worker, metadata master defaults.
    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = std::nullopt,
        .fifo_size_bytes = fifo_size_bytes,
        .worker_cores = worker_cores,
        .metadata_size_bytes = metadata_size_bytes,
    };
    tt::tt_metal::D2HStreamService service(mesh_device, std::move(cfg));

    // No DRAM backing tensor in metadata-only mode.
    EXPECT_EQ(service.payload_size_bytes(), 0u) << "metadata-only mode must not allocate a DRAM payload";
    EXPECT_EQ(service.metadata_size_bytes(), metadata_size_bytes);

    // Record: [1,1,1,N] uint32, replicated across the mesh, allocated once and refilled per iter.
    const uint32_t n = metadata_size_bytes / sizeof(uint32_t);
    const auto record_spec = TensorSpec(
        ttnn::Shape({1, 1, 1, n}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}));
    auto rep = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = replicate_all(*mesh_device)});
    Tensor record_dev = distribute_tensor(Tensor::from_vector<uint32_t>(std::vector<uint32_t>(n, 0), record_spec), *rep)
                            .to_device(mesh_device.get());

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        SCOPED_TRACE(::testing::Message() << "iter=" << iter);

        const auto expected = make_metadata_pattern(iter, metadata_size_bytes);
        std::vector<uint32_t> words(n);
        std::memcpy(words.data(), expected.data(), metadata_size_bytes);
        // Stage the record on device (same CQ as the op -> ordered), then drive the worker op.
        auto host_record = distribute_tensor(Tensor::from_vector<uint32_t>(words, record_spec), *rep);
        tt::tt_metal::copy_to_device(host_record, record_dev);

        ttnn::experimental::outbound_socket_service_sync(service, /*input=*/std::nullopt, /*metadata=*/record_dev);

        // Host pulls the record off each chip's socket; read_metadata asserts cross-chip equality.
        std::vector<std::byte> out(metadata_size_bytes);
        service.read_metadata(out);
        service.barrier();  // drain the socket before flushing the CQ / reusing record_dev
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());

        std::vector<uint8_t> got(metadata_size_bytes);
        std::memcpy(got.data(), out.data(), metadata_size_bytes);
        EXPECT_EQ(got, expected) << "metadata-only readback mismatch";
    }
}

// Service cores (ServiceCoreManager::claim) are only supported on Blackhole or UBB Galaxy
// clusters; skip the whole suite on any other configuration so unsupported runners skip
// cleanly instead of hitting the claim TT_FATAL.
class D2HStreamServiceTest : public ::tt::tt_metal::GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        if (!(cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE)) {
            GTEST_SKIP() << "D2HStreamService service cores require Blackhole or UBB Galaxy";
        }
        ::tt::tt_metal::GenericMeshDeviceFixture::SetUp();
    }
};

TEST_F(D2HStreamServiceTest, Replicated_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        uint32_t scratch_cb_pages;
        uint32_t fifo_pages;
    };
    const Row rows[] = {
        Row{640, 1, 1, 1},
        Row{640, 16, 4, 16},
        Row{640, 32, 1, 8},
        Row{640, 7, 4, 8},
        Row{128, 64, 1, 8},
        Row{1024, 16, 4, 16},
        Row{4096, 4, 2, 4},
    };

    for (const auto& row : rows) {
        const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
        D2HServiceCase cs{
            .global_shape = ttnn::Shape({1, 1, row.N, row.per_row_size}),
            .placements = replicate_all(*this->mesh_device_),
            .max_socket_page_size_bytes = row.scratch_cb_pages * per_row_bytes,
            .fifo_size_bytes = row.fifo_pages * per_row_bytes,
        };
        run_d2h_stream_service_case(this->mesh_device_, cs, OutputPath::Tensor);
        run_d2h_stream_service_case(this->mesh_device_, cs, OutputPath::Bytes);
    }
}

TEST_F(D2HStreamServiceTest, Replicated_WorkerSync) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const tt::tt_metal::CoreRange worker_cores({0, 0}, {1, 0});
    const uint32_t per_row_bytes = 640 * sizeof(uint32_t);
    D2HServiceCase cs{
        .global_shape = ttnn::Shape({1, 1, 16, 640}),
        .placements = replicate_all(*this->mesh_device_),
        .max_socket_page_size_bytes = 4 * per_row_bytes,
        .fifo_size_bytes = 16 * per_row_bytes,
    };
    run_d2h_worker_sync_case(this->mesh_device_, cs, worker_cores);
}

TEST_F(D2HStreamServiceTest, Replicated_WorkerSync_Metadata) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const tt::tt_metal::CoreRange worker_cores({0, 0}, {1, 0});
    const tt::tt_metal::CoreCoord metadata_master{0, 0};
    const uint32_t per_row_bytes = 640 * sizeof(uint32_t);
    D2HServiceCase cs{
        .global_shape = ttnn::Shape({1, 1, 16, 640}),
        .placements = replicate_all(*this->mesh_device_),
        .max_socket_page_size_bytes = 4 * per_row_bytes,
        .fifo_size_bytes = 16 * per_row_bytes,
    };
    run_d2h_worker_sync_case(
        this->mesh_device_, cs, worker_cores, metadata_master, /*num_iterations=*/5, /*metadata_size_bytes=*/64);
}

TEST_F(D2HStreamServiceTest, Replicated_WorkerSync_Metadata_SingleWorker) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const tt::tt_metal::CoreRange worker_cores({0, 0}, {0, 0});
    const tt::tt_metal::CoreCoord metadata_master{0, 0};
    const uint32_t per_row_bytes = 640 * sizeof(uint32_t);
    D2HServiceCase cs{
        .global_shape = ttnn::Shape({1, 1, 16, 640}),
        .placements = replicate_all(*this->mesh_device_),
        .max_socket_page_size_bytes = 4 * per_row_bytes,
        .fifo_size_bytes = 16 * per_row_bytes,
    };
    run_d2h_worker_sync_case(
        this->mesh_device_, cs, worker_cores, metadata_master, /*num_iterations=*/5, /*metadata_size_bytes=*/64);
}

TEST_F(D2HStreamServiceTest, Sharded_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "Sharded_Sweep requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        uint32_t scratch_cb_pages;
        uint32_t fifo_pages;
    };
    const Row rows[] = {
        Row{640, 1, 1, 1},
        Row{640, 16, 4, 16},
        Row{640, 32, 1, 8},
        Row{640, 7, 4, 8},
        Row{128, 64, 1, 8},
        Row{1024, 8, 2, 8},
        Row{4096, 4, 2, 4},
    };

    // Sharded placements: Tensor read path only (bytes path needs a composer).
    auto run_pattern = [&](const char* label,
                           const ttsl::SmallVector<MeshMapperConfig::Placement>& placements,
                           const std::function<ttnn::Shape(uint32_t, uint32_t)>& make_global_shape) {
        SCOPED_TRACE(::testing::Message() << "placement_pattern=" << label);
        for (const auto& row : rows) {
            SCOPED_TRACE(
                ::testing::Message() << "per_row=" << row.per_row_size << " N=" << row.N
                                     << " cb_pages=" << row.scratch_cb_pages << " fifo_pages=" << row.fifo_pages);
            const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
            D2HServiceCase cs{
                .global_shape = make_global_shape(row.N, row.per_row_size),
                .placements = placements,
                .max_socket_page_size_bytes = row.scratch_cb_pages * per_row_bytes,
                .fifo_size_bytes = row.fifo_pages * per_row_bytes,
            };
            run_d2h_stream_service_case(this->mesh_device_, cs, OutputPath::Tensor, /*num_iterations=*/10);
        }
    };

    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols",
            {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}},
            [&](uint32_t N, uint32_t per_row_size) { return ttnn::Shape({1, 1, N, num_rows * per_row_size}); });
    }
    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols",
            {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) { return ttnn::Shape({1, 1, N, num_cols * per_row_size}); });
    }
    if (num_rows >= 2 && num_cols >= 2) {
        run_pattern(
            "FullShard2D",
            {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, num_rows * N, num_cols * per_row_size});
            });
    }
}

TEST_F(D2HStreamServiceTest, Replicated_WorkerSync_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        tt::tt_metal::CoreRange worker_cores;
        tt::tt_metal::CoreCoord metadata_master;
        uint32_t num_iterations;
        uint32_t metadata_size_bytes;
        const char* label;
    };
    struct Chunking {
        uint32_t cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {0, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 0, "1_worker_p640_N16"},
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {1, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 0, "2_workers_p640_N16"},
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 0, "4_workers_p640_N16"},
        {640,
         16,
         tt::tt_metal::CoreRange{{0, 0}, {1, 1}},
         tt::tt_metal::CoreCoord{0, 0},
         20,
         0,
         "4_workers_2x2_p640_N16"},
        {640, 32, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 0, "4_workers_p640_N32"},
        {128, 64, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 0, "4_workers_p128_N64"},
        {1024,
         16,
         tt::tt_metal::CoreRange{{0, 0}, {3, 0}},
         tt::tt_metal::CoreCoord{0, 0},
         20,
         0,
         "4_workers_p1024_N16"},
        {4096, 8, tt::tt_metal::CoreRange{{0, 0}, {1, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 0, "2_workers_p4096_N8"},
        {640,
         120,
         tt::tt_metal::CoreRange{{0, 0}, {11, 9}},
         tt::tt_metal::CoreCoord{0, 0},
         100,
         0,
         "120_workers_full_grid"},
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 20, 16, "4_workers_meta_16B"},
        {640,
         16,
         tt::tt_metal::CoreRange{{0, 0}, {3, 0}},
         tt::tt_metal::CoreCoord{0, 0},
         20,
         256,
         "4_workers_meta_256B"},
        {640,
         16,
         tt::tt_metal::CoreRange{{0, 0}, {3, 0}},
         tt::tt_metal::CoreCoord{0, 0},
         20,
         2544,
         "4_workers_meta_near_page"},
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {1, 8, "cb1_fifo8"},
        {4, 16, "cb4_fifo16"},
        {1, 32, "cb1_fifo32"},
    };

    for (const auto& row : rows) {
        for (const auto& ch : chunkings) {
            SCOPED_TRACE(
                ::testing::Message() << "case=" << row.label << " chunk=" << ch.label
                                     << " metadata_size=" << row.metadata_size_bytes);
            const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
            D2HServiceCase cs{
                .global_shape = ttnn::Shape({1, 1, row.N, row.per_row_size}),
                .placements = replicate_all(*this->mesh_device_),
                .max_socket_page_size_bytes = ch.cb_pages * per_row_bytes,
                .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
            };
            run_d2h_worker_sync_case(
                this->mesh_device_,
                cs,
                row.worker_cores,
                row.metadata_size_bytes > 0 ? std::make_optional(row.metadata_master) : std::nullopt,
                row.num_iterations,
                row.metadata_size_bytes);
        }
    }
}

TEST_F(D2HStreamServiceTest, Sharded_WorkerSync_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "Sharded_WorkerSync_Sweep requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    struct Row {
        uint32_t per_row_size;
        uint32_t N;  // per-device page count (must satisfy N % num_workers == 0)
        tt::tt_metal::CoreRange worker_cores;
        tt::tt_metal::CoreCoord metadata_master;
        uint32_t metadata_size_bytes;
        const char* label;
    };
    struct Chunking {
        uint32_t cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {1, 0}}, tt::tt_metal::CoreCoord{0, 0}, 0, "2_workers_row"},
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 0, "4_workers_row"},
        {128, 64, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 0, "4_workers_p128_N64"},
        {1024, 16, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 0, "4_workers_p1024_N16"},
        {4096, 8, tt::tt_metal::CoreRange{{0, 0}, {1, 0}}, tt::tt_metal::CoreCoord{0, 0}, 0, "2_workers_p4096_N8"},
        {640, 16, tt::tt_metal::CoreRange{{0, 0}, {3, 0}}, tt::tt_metal::CoreCoord{0, 0}, 128, "4_workers_meta_128B"},
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {4, 16, "cb4_fifo16"},
    };

    auto run_pattern = [&](const char* pattern_label,
                           const ttsl::SmallVector<MeshMapperConfig::Placement>& placements,
                           const std::function<ttnn::Shape(uint32_t, uint32_t)>& make_global_shape) {
        SCOPED_TRACE(::testing::Message() << "placement_pattern=" << pattern_label);
        for (const auto& row : rows) {
            for (const auto& ch : chunkings) {
                SCOPED_TRACE(
                    ::testing::Message() << "case=" << row.label << " chunk=" << ch.label
                                         << " metadata_size=" << row.metadata_size_bytes);
                const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
                D2HServiceCase cs{
                    .global_shape = make_global_shape(row.N, row.per_row_size),
                    .placements = placements,
                    .max_socket_page_size_bytes = ch.cb_pages * per_row_bytes,
                    .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
                };
                run_d2h_worker_sync_case_per_shard(
                    this->mesh_device_,
                    cs,
                    row.worker_cores,
                    row.metadata_size_bytes > 0 ? std::make_optional(row.metadata_master) : std::nullopt,
                    /*num_iterations=*/20,
                    row.metadata_size_bytes);
            }
        }
    };

    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols",
            {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}},
            [&](uint32_t N, uint32_t per_row_size) { return ttnn::Shape({1, 1, N, num_rows * per_row_size}); });
    }
    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols",
            {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) { return ttnn::Shape({1, 1, N, num_cols * per_row_size}); });
    }
    if (num_rows >= 2 && num_cols >= 2) {
        run_pattern(
            "FullShard2D",
            {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, num_rows * N, num_cols * per_row_size});
            });
    }
}

// One worker, one 16B record, default fifo -- verifies the metadata-only path works.
TEST_F(D2HStreamServiceTest, MetadataOnly_WorkerOp) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const tt::tt_metal::CoreRange worker_cores({0, 0}, {0, 0});  // single designated worker
    run_d2h_metadata_only_case(this->mesh_device_, worker_cores, /*metadata_size_bytes=*/16, /*fifo_size_bytes=*/4096);
}

// Same metadata-only path across a few record sizes to catch page-size / alignment issues.
TEST_F(D2HStreamServiceTest, MetadataOnly_WorkerOp_Sizes) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const tt::tt_metal::CoreRange worker_cores({0, 0}, {0, 0});
    for (uint32_t md : {16u, 64u, 256u}) {
        SCOPED_TRACE(::testing::Message() << "metadata_size=" << md);
        run_d2h_metadata_only_case(this->mesh_device_, worker_cores, md, /*fifo_size_bytes=*/4096);
    }
}

// Perf check (disabled by default; needs hardware): confirms the device-issued completion
// path (worker op -> read_metadata) is cheaper per record than the old full-device
// Synchronize barrier it replaces. Path A times the new path, Path B the old one.
TEST_F(D2HStreamServiceTest, DISABLED_MetadataOnly_Microbench) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "D2HStreamService kernels are only available on UBB Galaxy systems";
    }
    const tt::tt_metal::CoreRange worker_cores({0, 0}, {0, 0});
    const uint32_t md = 16, iters = 1000;
    tt::tt_metal::D2HStreamService::Config cfg{
        .global_spec = std::nullopt, .fifo_size_bytes = 4096, .worker_cores = worker_cores, .metadata_size_bytes = md};
    tt::tt_metal::D2HStreamService service(this->mesh_device_, std::move(cfg));

    // Build the record device tensor (same shape/layout the unit helper uses).
    const uint32_t n = md / sizeof(uint32_t);
    const auto record_spec = TensorSpec(
        ttnn::Shape({1, 1, 1, n}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}));
    auto rep =
        create_mesh_mapper(*this->mesh_device_, MeshMapperConfig{.placements = replicate_all(*this->mesh_device_)});
    Tensor record_dev = distribute_tensor(Tensor::from_vector<uint32_t>(std::vector<uint32_t>(n, 0), record_spec), *rep)
                            .to_device(this->mesh_device_.get());

    // Warmup: absorb the first-call program-cache miss so it doesn't skew Path A.
    ttnn::experimental::outbound_socket_service_sync(service, /*input=*/std::nullopt, /*metadata=*/record_dev);
    {
        std::vector<std::byte> out(md);
        service.read_metadata(out);
    }

    // Path A: completion via read_metadata only (no barrier).
    double a_ns = 0.0;
    for (uint32_t i = 0; i < iters; ++i) {
        // Setup (Untimed): seed a distinct per-iter pattern, stage it, and forward it.
        const auto expected = make_metadata_pattern(i, md);
        std::vector<uint32_t> words(n);
        std::memcpy(words.data(), expected.data(), md);
        auto host_record = distribute_tensor(Tensor::from_vector<uint32_t>(words, record_spec), *rep);
        tt::tt_metal::copy_to_device(host_record, record_dev);
        ttnn::experimental::outbound_socket_service_sync(service, /*input=*/std::nullopt, /*metadata=*/record_dev);

        // Timed: just the metadata completion.
        std::vector<std::byte> out(md);
        auto t0 = std::chrono::high_resolution_clock::now();
        service.read_metadata(out);
        auto t1 = std::chrono::high_resolution_clock::now();
        a_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();

        // Verify on host.
        std::vector<uint8_t> got(md);
        std::memcpy(got.data(), out.data(), md);
        EXPECT_EQ(got, expected) << "Path A readback mismatch at iter " << i;
    }
    // Path B: completion via full device sync + read_metadata (the barrier we're replacing).
    double b_ns = 0.0;
    for (uint32_t i = 0; i < iters; ++i) {
        // Setup (Untimed): same as Path A.
        const auto expected = make_metadata_pattern(i, md);
        std::vector<uint32_t> words(n);
        std::memcpy(words.data(), expected.data(), md);
        auto host_record = distribute_tensor(Tensor::from_vector<uint32_t>(words, record_spec), *rep);
        tt::tt_metal::copy_to_device(host_record, record_dev);
        ttnn::experimental::outbound_socket_service_sync(service, /*input=*/std::nullopt, /*metadata=*/record_dev);

        // Timed: full mesh barrier + the same metadata read.
        std::vector<std::byte> out(md);
        auto t2 = std::chrono::high_resolution_clock::now();
        tt::tt_metal::distributed::Synchronize(this->mesh_device_.get(), /*cq_id=*/std::nullopt);
        service.read_metadata(out);
        auto t3 = std::chrono::high_resolution_clock::now();
        b_ns += std::chrono::duration<double, std::nano>(t3 - t2).count();

        // Verify on host.
        std::vector<uint8_t> got(md);
        std::memcpy(got.data(), out.data(), md);
        EXPECT_EQ(got, expected) << "Path B readback mismatch at iter " << i;
    }

    const double a_us = a_ns / 1000.0 / iters;
    const double b_us = b_ns / 1000.0 / iters;
    std::cout << "[d2h-md] read_metadata=" << a_us << "us  read+Synchronize=" << b_us << "us\n";
    EXPECT_LT(a_us, b_us) << "metadata path should be cheaper than a full device sync";
}

}  // namespace
}  // namespace ttnn::distributed::test
