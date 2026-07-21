// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <future>
#include <numeric>
#include <span>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <ttnn/distributed/distributed_configs.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/small_vector.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::distributed::test {
namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ttnn::Tensor;

// The Bytes path runs the service's internal mapper on the borrowed input; the
// Tensor path expects the caller to have already distributed via an equivalent mapper.
enum class InputPath {
    Tensor,
    Bytes,
};

inline const char* input_path_name(InputPath p) { return p == InputPath::Bytes ? "Bytes" : "Tensor"; }

// One sweep case, pinned to UINT32 ROW_MAJOR DRAM-interleaved.
struct H2DServiceCase {
    ttnn::Shape global_shape;
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    uint32_t max_socket_page_size_bytes = 0;
    uint32_t fifo_size_bytes = 0;
    uint32_t metadata_size_bytes = 0;  // optional inline metadata multicast, 0 = disabled
    bool parallel_host_push = false;   // fan each transfer's per-socket writes across host threads
    uint32_t host_push_thread_count = 0;
};

tt::tt_metal::distributed::MeshWorkload build_worker_workload(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::H2DStreamService& service,
    const ttnn::Tensor& output_tensor,
    const CoreRange& worker_cores,
    uint32_t metadata_size_bytes,
    uint32_t metadata_input_addr,
    uint32_t metadata_output_addr) {
    const ttnn::Tensor& input_tensor = service.get_backing_tensor();
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
        "build_worker_workload: tensor page count ({}) must be divisible by num_workers ({}). "
        "Pick a (shape, worker_cores) pair where the innermost-most-significant page count is a "
        "multiple of the worker count, or adjust the test parameters.",
        num_pages,
        num_workers);
    const uint32_t pages_per_worker = num_pages / num_workers;

    const uint32_t data_ready_sem_addr = static_cast<uint32_t>(service.get_data_ready_sem_addr());
    const uint32_t input_tensor_addr = static_cast<uint32_t>(input_buf->address());
    const uint32_t output_tensor_addr = static_cast<uint32_t>(output_buf->address());

    const auto& topology = input_tensor.tensor_topology();
    const auto& coords = topology.mesh_coords();
    TT_FATAL(!coords.empty(), "build_worker_workload: tensor topology has no coords");
    const tt::tt_metal::Buffer* sample_dbuf =
        input_tensor.mesh_buffer().get_device_buffer(coords.front());
    auto accessor_args = tt::tt_metal::TensorAccessorArgs(*sample_dbuf);
    auto accessor_compile_args = accessor_args.get_compile_time_args();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::distributed::MeshWorkload worker_workload;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);

        // Service-core physical NoC coords and consumed-counter address both vary per device.
        const CoreCoord service_logical = service.get_service_core(coord);
        const CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t consumed_counter_addr =
            static_cast<uint32_t>(service.get_consumed_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();

        auto cb_cfg = tt::tt_metal::CircularBufferConfig(
                          page_size, {{scratch_cb_index, tt::DataFormat::UInt32}})
                          .set_page_size(scratch_cb_index, page_size);
        tt::tt_metal::CreateCircularBuffer(program, worker_cores, cb_cfg);

        std::vector<uint32_t> ct_args = {
            data_ready_sem_addr,
            input_tensor_addr,
            output_tensor_addr,
            page_size,
            static_cast<uint32_t>(scratch_cb_index),
            // Metadata copy block (indices 5..8); all zero disables the kernel's copy loop.
            metadata_size_bytes > 0 ? 1u : 0u,
            metadata_size_bytes,
            metadata_input_addr,
            metadata_output_addr,
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

        // Row-major iteration so worker index matches the page-slice assignment.
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

        worker_workload.add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }

    return worker_workload;
}

void run_h2d_stream_service_case(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const H2DServiceCase& cs,
    InputPath input_path,
    std::optional<CoreRange> worker_cores = std::nullopt,
    uint32_t num_iterations = 2) {
    SCOPED_TRACE(
        ::testing::Message() << "global_shape=" << cs.global_shape
                             << " max_socket_page=" << cs.max_socket_page_size_bytes << " fifo=" << cs.fifo_size_bytes
                             << " input_path=" << input_path_name(input_path) << " parallel_host_push="
                             << cs.parallel_host_push << " host_push_thread_count=" << cs.host_push_thread_count);

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
        .max_socket_page_size_bytes = cs.max_socket_page_size_bytes,
        .worker_cores = worker_cores,
        .metadata_size_bytes = cs.metadata_size_bytes,
        .parallel_host_push = cs.parallel_host_push,
        .host_push_thread_count = cs.host_push_thread_count,
    };

    tt::tt_metal::H2DStreamService service(mesh_device, std::move(cfg));
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);
    ASSERT_EQ(service.get_sockets().size(), mesh_device->num_devices());

    std::optional<ttnn::Tensor> output_tensor;
    tt::tt_metal::distributed::MeshWorkload worker_workload;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> metadata_output_buffer;
    tt::tt_metal::DeviceAddr metadata_output_addr = 0;
    if (worker_cores.has_value()) {
        const auto& backing = service.get_backing_tensor();
        output_tensor.emplace(
            ttnn::create_device_tensor(backing.tensor_spec(), mesh_device.get(), backing.tensor_topology()));

        if (cs.metadata_size_bytes > 0) {
            const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
            const tt::tt_metal::DeviceAddr aligned_shard_size = tt::align(
                static_cast<tt::tt_metal::DeviceAddr>(cs.metadata_size_bytes),
                static_cast<tt::tt_metal::DeviceAddr>(l1_align));
            const uint32_t num_workers = (worker_cores->end_coord.x - worker_cores->start_coord.x + 1) *
                                         (worker_cores->end_coord.y - worker_cores->start_coord.y + 1);
            const tt::tt_metal::CoreRangeSet shard_grid(*worker_cores);

            tt::tt_metal::distributed::DeviceLocalBufferConfig device_local = {
                .page_size = aligned_shard_size,
                .buffer_type = tt::tt_metal::BufferType::L1,
                .sharding_args = tt::tt_metal::BufferShardingArgs(
                    tt::tt_metal::ShardSpecBuffer(
                        shard_grid,
                        {1, 1},
                        tt::tt_metal::ShardOrientation::ROW_MAJOR,
                        {1, 1},
                        {num_workers, 1}),
                    tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = std::nullopt,
                .sub_device_id = std::nullopt,
            };
            tt::tt_metal::distributed::MeshBufferConfig mesh_config =
                tt::tt_metal::distributed::ReplicatedBufferConfig{
                    .size = aligned_shard_size * static_cast<tt::tt_metal::DeviceAddr>(num_workers),
                };
            metadata_output_buffer = tt::tt_metal::distributed::MeshBuffer::create(
                mesh_config, device_local, mesh_device.get());
            metadata_output_addr = metadata_output_buffer->address();
        }

        worker_workload = build_worker_workload(
            mesh_device,
            service,
            *output_tensor,
            *worker_cores,
            cs.metadata_size_bytes,
            cs.metadata_size_bytes > 0 ? static_cast<uint32_t>(service.get_metadata_addr()) : 0u,
            static_cast<uint32_t>(metadata_output_addr));
    }

    auto push = [&](const std::vector<uint32_t>& src, const std::vector<std::byte>& meta) {
        const auto meta_span = ttsl::Span<const std::byte>(meta.data(), meta.size());
        if (input_path == InputPath::Bytes) {
            auto bytes = ttsl::Span<const std::byte>(
                reinterpret_cast<const std::byte*>(src.data()), src.size() * sizeof(uint32_t));
            service.forward_to_tensor(bytes, meta_span);
        } else {
            auto mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
            auto host = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *mapper);
            service.forward_to_tensor(host, meta_span);
        }
    };

    auto consume_one = [&]() {
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), worker_workload, /*blocking=*/false);
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());
    };

    auto verify = [&](const std::vector<uint32_t>& src, const std::vector<std::byte>& expected_meta) {
        auto verify_mapper = create_mesh_mapper(*mesh_device, MeshMapperConfig{.placements = cs.placements});
        auto distributed_host = distribute_tensor(Tensor::from_vector<uint32_t>(src, global_spec), *verify_mapper);
        const auto& dhb = distributed_host.host_storage().host_tensor().buffer();

        const auto& tensor_under_test = worker_cores.has_value() ? *output_tensor : service.get_backing_tensor();
        auto subs = get_device_tensors(tensor_under_test);
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

            if (cs.metadata_size_bytes > 0) {
                auto* d = mesh_device->get_device(coord);
                const auto* exp_meta_u8 = reinterpret_cast<const uint8_t*>(expected_meta.data());
                const std::vector<uint8_t> expected_meta_u8(exp_meta_u8, exp_meta_u8 + expected_meta.size());

                // Read from the worker-owned output region, not the service-owned input region.
                for (uint32_t y = worker_cores->start_coord.y; y <= worker_cores->end_coord.y; ++y) {
                    for (uint32_t x = worker_cores->start_coord.x; x <= worker_cores->end_coord.x; ++x) {
                        const CoreCoord worker_logical{x, y};
                        std::vector<uint8_t> meta_readback(cs.metadata_size_bytes);
                        tt::tt_metal::detail::ReadFromDeviceL1(
                            d,
                            worker_logical,
                            static_cast<uint32_t>(metadata_output_addr),
                            std::span<uint8_t>(meta_readback.data(), meta_readback.size()));
                        EXPECT_EQ(meta_readback, expected_meta_u8)
                            << "metadata mismatch at coord " << coord << " (worker logical=(" << worker_logical.x
                            << "," << worker_logical.y << "))";
                    }
                }
            }
        }
    };

    auto make_iter_data = [&](uint32_t iter) {
        std::vector<uint32_t> v(cs.global_shape.volume());
        std::iota(v.begin(), v.end(), iter * 0x12345678u);
        return v;
    };

    auto make_iter_metadata = [&](uint32_t iter) {
        std::vector<std::byte> v(cs.metadata_size_bytes);
        for (uint32_t i = 0; i < cs.metadata_size_bytes; ++i) {
            v[i] = std::byte{static_cast<uint8_t>((iter * 0x9E37u + i) & 0xFFu)};
        }
        return v;
    };

    if (worker_cores.has_value()) {
        auto writer = std::async(std::launch::async, [&] {
            for (uint32_t iter = 0; iter < num_iterations; ++iter) {
                SCOPED_TRACE(::testing::Message() << "[writer] iteration=" << iter);
                push(make_iter_data(iter), make_iter_metadata(iter));
            }
        });
        auto consumer = std::async(std::launch::async, [&] {
            for (uint32_t iter = 0; iter < num_iterations; ++iter) {
                SCOPED_TRACE(::testing::Message() << "[consumer] iteration=" << iter);
                consume_one();
                verify(make_iter_data(iter), make_iter_metadata(iter));
            }
        });
        // Consumer first so a verify failure surfaces before we wait on the writer.
        consumer.get();
        writer.get();
    } else {
        for (uint32_t iter = 0; iter < num_iterations; ++iter) {
            SCOPED_TRACE(::testing::Message() << "iteration=" << iter);
            auto data = make_iter_data(iter);
            auto meta = make_iter_metadata(iter);
            push(data, meta);
            service.barrier();
            verify(data, meta);
        }
    }
}

// Service cores (ServiceCoreManager::claim) are only supported on Blackhole or UBB Galaxy
// clusters; skip the whole suite on any other configuration so unsupported runners skip
// cleanly instead of hitting the claim TT_FATAL.
class H2DStreamServiceTest : public ::tt::tt_metal::GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        if (!(cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE)) {
            GTEST_SKIP() << "H2DStreamService service cores require Blackhole or UBB Galaxy";
        }
        ::tt::tt_metal::GenericMeshDeviceFixture::SetUp();
    }
};

// Fully-replicated placements sized to this mesh's dimensionality.
ttsl::SmallVector<MeshMapperConfig::Placement> replicate_all(
    const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    return ttsl::SmallVector<MeshMapperConfig::Placement>(
        mesh_device.shape().dims(), MeshMapperConfig::Replicate{});
}

TEST_F(H2DStreamServiceTest, Replicated_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }
    // max_coalesce_pages caps the socket page at that many tensor pages (read-coalescing granularity,
    // NOT a scratch-CB total -- slot depth auto-fills L1); fifo_pages sizes the host FIFO. Both are in
    // units of one tensor page (per_row_size * sizeof(uint32_t)).
    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        uint32_t max_coalesce_pages;
        uint32_t fifo_pages;
    };
    const Row rows[] = {
        Row{/*per_row=*/640, /*N=*/1, /*max_coalesce=*/1, /*fifo=*/1},
        Row{/*per_row=*/640, /*N=*/16, /*max_coalesce=*/4, /*fifo=*/16},
        Row{/*per_row=*/640, /*N=*/32, /*max_coalesce=*/1, /*fifo=*/8},
        // Prime page count exercises the pages_per_chunk divisor fallback to 1.
        Row{/*per_row=*/640, /*N=*/7, /*max_coalesce=*/4, /*fifo=*/8},
        Row{/*per_row=*/128, /*N=*/64, /*max_coalesce=*/1, /*fifo=*/8},
        Row{/*per_row=*/1024, /*N=*/16, /*max_coalesce=*/4, /*fifo=*/16},
        Row{/*per_row=*/4096, /*N=*/4, /*max_coalesce=*/2, /*fifo=*/4},
        // Fully auto-sized geometry: cb=0 -> burst-derived socket page, fifo=0 -> service-sized host
        // FIFO. Exercises the max_socket_page_size_bytes=0 / fifo_size_bytes=0 default paths.
        Row{/*per_row=*/640, /*N=*/16, /*max_coalesce=*/0, /*fifo=*/0},
    };

    for (const auto& row : rows) {
        const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
        H2DServiceCase cs{
            .global_shape = ttnn::Shape({1, 1, row.N, row.per_row_size}),
            .placements = replicate_all(*this->mesh_device_),
            .max_socket_page_size_bytes = row.max_coalesce_pages * per_row_bytes,
            .fifo_size_bytes = row.fifo_pages * per_row_bytes,
        };
        run_h2d_stream_service_case(
            this->mesh_device_, cs, InputPath::Tensor, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
        run_h2d_stream_service_case(
            this->mesh_device_, cs, InputPath::Bytes, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
    }
}

TEST_F(H2DStreamServiceTest, Sharded_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "This test requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    // max_coalesce_pages caps the socket page at that many tensor pages (read-coalescing granularity,
    // NOT a scratch-CB total -- slot depth auto-fills L1); fifo_pages sizes the host FIFO. Both are in
    // units of one tensor page (per_row_size * sizeof(uint32_t)).
    struct Row {
        uint32_t per_row_size;
        uint32_t N;
        uint32_t max_coalesce_pages;
        uint32_t fifo_pages;
    };
    const Row rows[] = {
        Row{/*per_row=*/640, /*N=*/1, /*max_coalesce=*/1, /*fifo=*/1},
        Row{/*per_row=*/640, /*N=*/16, /*max_coalesce=*/4, /*fifo=*/16},
        Row{/*per_row=*/640, /*N=*/32, /*max_coalesce=*/1, /*fifo=*/8},
        // Prime page count exercises the pages_per_chunk divisor fallback to 1.
        Row{/*per_row=*/640, /*N=*/7, /*max_coalesce=*/4, /*fifo=*/8},
        Row{/*per_row=*/128, /*N=*/64, /*max_coalesce=*/1, /*fifo=*/8},
        Row{/*per_row=*/1024, /*N=*/16, /*max_coalesce=*/4, /*fifo=*/16},
        Row{/*per_row=*/4096, /*N=*/4, /*max_coalesce=*/2, /*fifo=*/4},
    };

    // Per-device shape is always [1, 1, N, per_row_size]; make_global_shape scales it per pattern.
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
                .max_socket_page_size_bytes = row.max_coalesce_pages * per_row_bytes,
                .fifo_size_bytes = row.fifo_pages * per_row_bytes,
            };
            run_h2d_stream_service_case(
                this->mesh_device_, cs, InputPath::Tensor, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
            run_h2d_stream_service_case(
                this->mesh_device_, cs, InputPath::Bytes, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
        }
    };

    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols",
            {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_rows * per_row_size});
            });
    }

    // Mirror of the above with shard on mesh-dim 1; catches code that hardcodes mesh-dim 0 as the shard axis.
    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols",
            {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_cols * per_row_size});
            });
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

TEST_F(H2DStreamServiceTest, Replicated_WorkerSync_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }
    struct Row {
        uint32_t per_row_size;
        uint32_t N;  // tensor pages per device; must satisfy N % num_workers == 0
        CoreRange worker_cores;
        uint32_t num_iterations;
        uint32_t metadata_size_bytes;  // 0 = disabled; must be <= smallest socket_page_size in chunkings
        const char* label;
    };
    struct Chunking {
        uint32_t max_coalesce_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 0, "4_workers_row"},
        // Single worker exercises the num_workers==1 degenerate-multicast path.
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}, 20, 0, "1_worker"},
        // Full 12x10 grid = 120 cores; N bumped to 120 to keep divisibility.
        {640, 120, CoreRange{CoreCoord{0, 0}, CoreCoord{11, 9}}, 100, 0, "120_workers_full_grid"},

        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 16, "4_workers_meta_16B"},
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 256, "4_workers_meta_256B"},
        // Just under socket_page_size=2560 in max_coalesce_pages=1: host pads only 16 B of zeros.
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 2544, "4_workers_meta_near_page"},
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {1, 8, "cb1_fifo8"},
        {4, 16, "cb4_fifo16"},
        // 320KB CB + 320KB FIFO; fifo >= pages_per_chunk=min(N,128) holds for N in {16, 120}.
        {128, 128, "cb128_fifo128"},
        // 10KB CB + ~1MB FIFO; CB+FIFO total fits under the ~1MB usable L1 per service core.
        {4, 400, "cb4_fifo1MB"},
    };

    for (const auto& row : rows) {
        const uint32_t num_workers = (row.worker_cores.end_coord.x - row.worker_cores.start_coord.x + 1) *
                                     (row.worker_cores.end_coord.y - row.worker_cores.start_coord.y + 1);
        for (const auto& ch : chunkings) {
            SCOPED_TRACE(
                ::testing::Message() << "case=" << row.label << " chunk=" << ch.label << " N=" << row.N
                                     << " per_row=" << row.per_row_size << " num_workers=" << num_workers
                                     << " num_iterations=" << row.num_iterations
                                     << " metadata_size_bytes=" << row.metadata_size_bytes);

            const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
            H2DServiceCase cs{
                .global_shape = ttnn::Shape({1, 1, row.N, row.per_row_size}),
                .placements = replicate_all(*this->mesh_device_),
                .max_socket_page_size_bytes = ch.max_coalesce_pages * per_row_bytes,
                .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
                .metadata_size_bytes = row.metadata_size_bytes,
            };
            for (auto path : {InputPath::Tensor, InputPath::Bytes}) {
                SCOPED_TRACE(::testing::Message() << "path=" << input_path_name(path));
                run_h2d_stream_service_case(this->mesh_device_, cs, path, row.worker_cores, row.num_iterations);
            }
        }
    }
}

TEST_F(H2DStreamServiceTest, Sharded_WorkerSync_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "This test requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    struct Row {
        uint32_t per_row_size;
        uint32_t N;  // per-device page count; must satisfy N % num_workers == 0
        CoreRange worker_cores;
        uint32_t metadata_size_bytes;  // 0 = disabled; must be <= smallest socket_page_size in chunkings
        const char* label;
    };
    struct Chunking {
        uint32_t max_coalesce_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 0, "4_workers_row"},
        // Single worker exercises the num_workers==1 degenerate-multicast path.
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}, 0, "1_worker"},
        // Full 12x10 grid = 120 cores; N bumped to 120 to keep divisibility.
        Row{640, 120, CoreRange{CoreCoord{0, 0}, CoreCoord{11, 9}}, 0, "120_workers_full_grid"},

        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 16, "4_workers_meta_16B"},
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 256, "4_workers_meta_256B"},
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 2544, "4_workers_meta_near_page"},
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {1, 8, "cb1_fifo8"},
        {4, 16, "cb4_fifo16"},
        {128, 128, "cb128_fifo128"},
        {4, 400, "cb4_fifo1MB"},
    };

    constexpr uint32_t kNumIterations = 20;

    auto run_pattern = [&](const char* pattern_label,
                           const ttsl::SmallVector<MeshMapperConfig::Placement>& placements,
                           const std::function<ttnn::Shape(uint32_t /*N*/, uint32_t /*per_row_size*/)>&
                               make_global_shape) {
        SCOPED_TRACE(::testing::Message() << "placement_pattern=" << pattern_label);
        for (const auto& row : rows) {
            const uint32_t num_workers = (row.worker_cores.end_coord.x - row.worker_cores.start_coord.x + 1) *
                                         (row.worker_cores.end_coord.y - row.worker_cores.start_coord.y + 1);
            for (const auto& ch : chunkings) {
                const uint32_t per_row_bytes = row.per_row_size * sizeof(uint32_t);
                H2DServiceCase cs{
                    .global_shape = make_global_shape(row.N, row.per_row_size),
                    .placements = placements,
                    .max_socket_page_size_bytes = ch.max_coalesce_pages * per_row_bytes,
                    .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
                    .metadata_size_bytes = row.metadata_size_bytes,
                };
                for (auto path : {InputPath::Tensor, InputPath::Bytes}) {
                    SCOPED_TRACE(
                        ::testing::Message() << "case=" << row.label << " chunk=" << ch.label
                                             << " path=" << input_path_name(path) << " per_row=" << row.per_row_size
                                             << " N=" << row.N << " num_workers=" << num_workers
                                             << " metadata_size_bytes=" << row.metadata_size_bytes);
                    run_h2d_stream_service_case(this->mesh_device_, cs, path, row.worker_cores, kNumIterations);
                }
            }
        }
    };

    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols",
            {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_rows * per_row_size});
            });
    }

    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols",
            {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_cols * per_row_size});
            });
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

// Multi-threaded host-push coverage. parallel_host_push is orthogonal to chunk geometry -- it only
// changes how the host fans the per-socket writes across threads; the data and the device side are
// identical -- so this test is deliberately narrow on geometry (two chunkings).
TEST_F(H2DStreamServiceTest, MultiThreadedHostPush_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }
    const auto mesh_shape = this->mesh_device_->shape();
    const bool mesh_2d = mesh_shape.dims() == 2;
    const uint32_t num_cols = mesh_2d ? static_cast<uint32_t>(mesh_shape[1]) : 0u;

    constexpr uint32_t per_row_size = 640;
    constexpr uint32_t N = 16;  // per-device page count; divisible by the 4-worker grid below
    const uint32_t per_row_bytes = per_row_size * sizeof(uint32_t);

    struct Chunking {
        uint32_t max_coalesce_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Chunking chunkings[] = {
        {1, 8, "cb1_fifo8"},    // socket page == one tensor page -> deep slot fill under parallel push
        {4, 16, "cb4_fifo16"},  // coalesced socket page
    };

    struct PlacementCase {
        const char* label;
        ttsl::SmallVector<MeshMapperConfig::Placement> placements;
        ttnn::Shape global_shape;
    };
    std::vector<PlacementCase> placement_cases;
    placement_cases.push_back({"replicated", replicate_all(*this->mesh_device_), ttnn::Shape({1, 1, N, per_row_size})});
    // Sharded (distinct data per socket) on a 2D mesh; per-device shape stays [1,1,N,per_row_size].
    if (mesh_2d && num_cols >= 2) {
        placement_cases.push_back(
            {"shard_cols",
             {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
             ttnn::Shape({1, 1, N, num_cols * per_row_size})});
    }

    // 4 workers; N=16 is divisible by 4.
    const CoreRange worker_row{CoreCoord{0, 0}, CoreCoord{3, 0}};
    struct Scenario {
        std::optional<CoreRange> workers;
        uint32_t metadata_size_bytes;
        const char* label;
    };
    const Scenario scenarios[] = {
        {std::nullopt, 0, "plain_barrier"},
        {worker_row, 0, "worker_sync"},
        {worker_row, 256, "worker_sync_meta256"},
    };

    for (const auto& pc : placement_cases) {
        SCOPED_TRACE(::testing::Message() << "placement=" << pc.label);
        for (const auto& ch : chunkings) {
            for (const auto& sc : scenarios) {
                H2DServiceCase cs{
                    .global_shape = pc.global_shape,
                    .placements = pc.placements,
                    .max_socket_page_size_bytes = ch.max_coalesce_pages * per_row_bytes,
                    .fifo_size_bytes = ch.fifo_pages * per_row_bytes,
                    .metadata_size_bytes = sc.metadata_size_bytes,
                    .parallel_host_push = true,
                };
                const uint32_t iters = sc.workers.has_value() ? 20u : 10u;
                for (auto path : {InputPath::Tensor, InputPath::Bytes}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << "chunk=" << ch.label << " scenario=" << sc.label << " path=" << input_path_name(path));
                    run_h2d_stream_service_case(this->mesh_device_, cs, path, sc.workers, iters);
                }
            }
        }
    }
}

TEST_F(H2DStreamServiceTest, HostPushThreadCount_Sweep) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }

    constexpr uint32_t per_row_size = 640;
    constexpr uint32_t N = 16;
    const uint32_t per_row_bytes = per_row_size * sizeof(uint32_t);
    const uint32_t oversized_thread_count = static_cast<uint32_t>(this->mesh_device_->num_devices()) +
                                            tt::tt_metal::H2DStreamService::kAutoHostPushThreadCount;

    struct ThreadCountCase {
        bool parallel_host_push;
        uint32_t host_push_thread_count;
        const char* label;
    };
    const ThreadCountCase thread_count_cases[] = {
        {false, oversized_thread_count, "disabled_ignores_large_count"},
        {true, 0, "auto_tuned_default"},
        {true, 1, "explicit_serial"},
        {true, 2, "two_threads"},
        {true, 3, "odd_thread_count"},
        {true, tt::tt_metal::H2DStreamService::kAutoHostPushThreadCount, "tuned_default_explicit"},
        {true, oversized_thread_count, "oversized_count_clamps"},
    };

    for (const auto& tc : thread_count_cases) {
        SCOPED_TRACE(::testing::Message() << "thread_count_case=" << tc.label);
        H2DServiceCase cs{
            .global_shape = ttnn::Shape({1, 1, N, per_row_size}),
            .placements = replicate_all(*this->mesh_device_),
            .max_socket_page_size_bytes = 4 * per_row_bytes,
            .fifo_size_bytes = 16 * per_row_bytes,
            .parallel_host_push = tc.parallel_host_push,
            .host_push_thread_count = tc.host_push_thread_count,
        };
        run_h2d_stream_service_case(this->mesh_device_, cs, InputPath::Bytes, /*worker_cores=*/std::nullopt, 3);
    }
}

namespace {
// Volume-preserving column rotation of an N_C*W-element uint32 array (ring-SDPA prefill reshuffle).
void ring_sdpa_reshuffle(
    ttsl::Span<std::byte> bytes, uint32_t c_start, uint32_t intra, uint32_t N_C, uint32_t W) {
    const size_t volume = N_C * W;
    TT_FATAL(bytes.size() == volume * sizeof(uint32_t), "reshuffle: size mismatch");
    const auto* in = reinterpret_cast<const uint32_t*>(bytes.data());
    std::vector<uint32_t> out(volume);

    // Column c_start takes [0, W - intra) from the head and the trailing intra elements from the tail.
    for (uint32_t i = 0; i < W - intra; ++i) {
        out[c_start * W + i] = in[i];
    }
    for (uint32_t i = 0; i < intra; ++i) {
        out[c_start * W + (W - intra) + i] = in[volume - intra + i];
    }

    for (uint32_t k = 1; k < N_C; ++k) {
        const uint32_t col = (c_start + k) % N_C;
        const uint32_t s = (W - intra) + (k - 1) * W;
        for (uint32_t i = 0; i < W; ++i) {
            out[col * W + i] = in[s + i];
        }
    }

    std::memcpy(bytes.data(), out.data(), bytes.size());
}
}  // namespace

TEST_F(H2DStreamServiceTest, Preprocessor_RingSDPAReshuffle) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        GTEST_SKIP() << "H2DStreamService kernels are only available on UBB Galaxy systems";
    }
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "Preprocessor reshuffle test requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_cols = mesh_shape[1];
    if (num_cols < 2) {
        GTEST_SKIP() << "Need num_cols >= 2 for sharded-along-cols reshuffle; got " << num_cols;
    }

    const uint32_t N_C = num_cols;
    // W * sizeof(uint32_t) = 256 B per-shard page is a multiple of the 64 B PCIe alignment on Blackhole.
    constexpr uint32_t W = 64;

    ttsl::SmallVector<MeshMapperConfig::Placement> placements{
        MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}};

    const auto global_shape = ttnn::Shape({1, 1, 1, N_C * W});
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(global_shape, tensor_layout);

    uint32_t current_chunk_P_aligned = 0;
    auto preprocessor = [N_C, &current_chunk_P_aligned](
                            ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata) {
        (void)metadata;
        const uint32_t c_start = (current_chunk_P_aligned / W) % N_C;
        const uint32_t intra = current_chunk_P_aligned % W;
        ring_sdpa_reshuffle(bytes, c_start, intra, N_C, W);
    };

    const uint32_t tensor_page_size_bytes = W * sizeof(uint32_t);
    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*this->mesh_device_, MeshMapperConfig{.placements = placements}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = tensor_page_size_bytes,
        .max_socket_page_size_bytes = tensor_page_size_bytes,
        .worker_cores = std::nullopt,
        .metadata_size_bytes = 0,
        .preprocessor = preprocessor,
    };
    tt::tt_metal::H2DStreamService service(this->mesh_device_, std::move(cfg));
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);

    std::vector<uint32_t> src(N_C * W);
    std::iota(src.begin(), src.end(), 0);

    // Covers identity, pure column rotation, pure intra rotation, and combined.
    struct Case {
        uint32_t chunk_P_aligned;
        const char* label;
    };
    const Case cases[] = {
        {0,                                   "identity"},
        {W,                                   "c_start=1, intra=0"},
        {W / 2,                               "c_start=0, intra=W/2"},
        {W * (N_C / 2) + W / 2,               "c_start=N_C/2, intra=W/2"},
    };

    for (const auto& tc : cases) {
        SCOPED_TRACE(
            ::testing::Message() << "chunk_P_aligned=" << tc.chunk_P_aligned << " label=" << tc.label);
        current_chunk_P_aligned = tc.chunk_P_aligned;  // picked up by preprocessor closure
        auto bytes_span = ttsl::Span<const std::byte>(
            reinterpret_cast<const std::byte*>(src.data()), src.size() * sizeof(uint32_t));

        service.forward_to_tensor(bytes_span, /*metadata=*/{});
        service.barrier();

        std::vector<uint32_t> expected_global = src;
        const uint32_t c_start = (tc.chunk_P_aligned / W) % N_C;
        const uint32_t intra = tc.chunk_P_aligned % W;
        ring_sdpa_reshuffle(
            ttsl::Span<std::byte>(
                reinterpret_cast<std::byte*>(expected_global.data()),
                expected_global.size() * sizeof(uint32_t)),
            c_start, intra, N_C, W);

        auto subs = ttnn::distributed::get_device_tensors(service.get_backing_tensor());
        ASSERT_EQ(subs.size(), this->mesh_device_->num_devices());
        for (auto& sub : subs) {
            const auto coords = sub.device_storage().get_coords();
            ASSERT_EQ(coords.size(), 1u);
            const auto& coord = coords[0];
            const uint32_t col = coord[1];
            const uint32_t row = coord[0];
            (void)row;
            std::vector<uint32_t> expected_shard(
                expected_global.begin() + col * W,
                expected_global.begin() + (col + 1) * W);
            auto readback = sub.to_vector<uint32_t>();
            ASSERT_EQ(readback.size(), expected_shard.size())
                << "size mismatch at coord=" << coord;
            EXPECT_EQ(readback, expected_shard)
                << "contents mismatch at coord=" << coord
                << " chunk_P_aligned=" << tc.chunk_P_aligned;
        }
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
