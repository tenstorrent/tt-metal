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
#include "ttnn/tensor/socket_services.hpp"
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
    // Optional inline metadata multicast (0 = disabled). When non-zero, every
    // forward_to_tensor call ships this many trailing bytes; the helper
    // generates a deterministic per-iter pattern and the kernel multicasts it
    // to every worker core's local L1 copy at service.get_metadata_addr().
    uint32_t metadata_size_bytes = 0;
};

// Builds the per-coord worker MeshWorkload for the worker-sync handshake test.
//
// One Program is constructed per participating mesh coord (per-coord because
// the consumed-counter address and the service core's physical NoC coords are
// per-device). All Programs share identical CT args (uniform across the mesh
// by design — data_ready_sem_addr, input/output tensor addresses, page size,
// and TensorAccessorArgs) but receive different per-coord runtime args.
//
// Each worker core in `worker_cores` is assigned a contiguous slice of the
// flat tensor page array; total pages must divide evenly by the number of
// workers (asserted).
tt::tt_metal::distributed::MeshWorkload build_worker_workload(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::H2DStreamService& service,
    const tt::tt_metal::Tensor& output_tensor,
    const CoreRange& worker_cores,
    uint32_t metadata_size_bytes,
    uint32_t metadata_input_addr,
    uint32_t metadata_output_addr) {
    const tt::tt_metal::Tensor& input_tensor = service.get_backing_tensor();
    auto* input_buf = input_tensor.buffer();
    auto* output_buf = output_tensor.buffer();
    TT_FATAL(input_buf != nullptr, "build_worker_workload: input tensor has no buffer");
    TT_FATAL(output_buf != nullptr, "build_worker_workload: output tensor has no buffer");

    const uint32_t page_size = input_buf->page_size();
    const uint32_t num_pages = input_buf->num_pages();

    // Count workers in row-major order over the CoreRange (consistent with the
    // RT-arg assignment loop below).
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

    // CT args uniform across mesh + workers. Single TensorAccessorArgs set is
    // reused with the input and output base addresses inside the kernel.
    const uint32_t data_ready_sem_addr = static_cast<uint32_t>(service.get_data_ready_sem_addr());
    const uint32_t input_tensor_addr = static_cast<uint32_t>(input_buf->address());
    const uint32_t output_tensor_addr = static_cast<uint32_t>(output_buf->address());

    // Need a per-coord device buffer pointer to feed TensorAccessorArgs(); the
    // resulting CT args describe the tensor layout, which is uniform across
    // coords for the same TensorSpec — picking any coord's per-device buffer
    // yields the same accessor args.
    const auto& topology = input_tensor.tensor_topology();
    const auto& coords = topology.mesh_coords();
    TT_FATAL(!coords.empty(), "build_worker_workload: tensor topology has no coords");
    const tt::tt_metal::Buffer* sample_dbuf =
        input_tensor.mesh_buffer().get_device_buffer(coords.front());
    auto accessor_args = tt::tt_metal::TensorAccessorArgs(*sample_dbuf);
    auto accessor_compile_args = accessor_args.get_compile_time_args();

    // Single scratch CB sized to one tensor page on every worker core.
    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::distributed::MeshWorkload worker_workload;
    for (const auto& coord : coords) {
        auto* device = mesh_device->get_device(coord);

        // Resolve service-core physical NoC coords for this device. Worker
        // kernels NoC-write atomic-incs to (service_noc_x, service_noc_y,
        // consumed_counter_addr); both pieces vary per device.
        const CoreCoord service_logical = service.get_service_core(coord);
        const CoreCoord service_phys = device->worker_core_from_logical_core(service_logical);
        const uint32_t consumed_counter_addr =
            static_cast<uint32_t>(service.get_consumed_counter_addr(coord));

        auto program = tt::tt_metal::CreateProgram();

        // Scratch CB: one page, BFLOAT16 data-format slot is a no-op for raw
        // L1 staging; we just need the L1 region for the read-then-write copy.
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
            // Metadata copy block (indices 5..8). All zero when metadata is
            // disabled — the kernel's `if constexpr (metadata_enabled)` guard
            // drops the copy loop entirely.
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

        // Per-worker runtime args: contiguous page slice + service-core handle.
        // Iterate row-major over the CoreRange so worker index matches the
        // page-slice assignment.
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

// Drives the service through `num_iterations` full transfers for a given case +
// input path. The service is spawned once outside the loop; the worker workload
// (when worker-sync is enabled) is built once and re-enqueued per iteration.
// Each iteration writes a fresh iota source with a distinct seed and verifies
// every per-device readback against the per-coord shard produced by an external
// mapper with the same config. The persistent-kernel "reuse check" is implicit:
// any iter after the first produces wrong contents if the kernel had exited.
//
// When `worker_cores` is set, the helper additionally:
//   * passes the CoreRange to Config::worker_cores so the service kernel
//     enables its multicast / consumed-counter handshake,
//   * allocates a second device tensor (`output_tensor`) with the same per-
//     shard spec + topology as the service's backing tensor,
//   * builds a worker MeshWorkload (one Program per coord) using the kernel
//     at tests/.../kernels/persistent_h2d_worker_test.cpp,
//   * enqueues the worker workload after each forward_to_tensor + barrier and
//     drains via Finish, so the output tensor reflects this iteration's data,
//   * verifies against `output_tensor` instead of the service's backing tensor.
void run_h2d_stream_service_case(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const H2DServiceCase& cs,
    InputPath input_path,
    std::optional<CoreRange> worker_cores = std::nullopt,
    uint32_t num_iterations = 2) {
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
        .worker_cores = worker_cores,
        .metadata_size_bytes = cs.metadata_size_bytes,
    };

    tt::tt_metal::H2DStreamService service(mesh_device, std::move(cfg));
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);
    ASSERT_EQ(service.get_sockets().size(), mesh_device->num_devices());

    // Worker-sync path: allocate an output tensor with the same per-shard spec
    // + topology as the service's backing tensor, build the worker MeshWorkload
    // once (Programs + per-worker RT args persist across enqueues).
    //
    // When metadata is enabled we also allocate a *second* L1-sharded buffer
    // mirroring the service's metadata input buffer (one shard per worker, same
    // size, same shard layout, REPLICATED across the mesh). The worker kernel
    // copies metadata_input -> metadata_output before atomic-incing the
    // consumed counter, so the host's per-iter `ReadFromDeviceL1` can read a
    // worker-owned region that the service kernel never touches — eliminating
    // the race where the next iter's multicast overwrites the input region
    // before the host has finished verifying the current iter.
    std::optional<tt::tt_metal::Tensor> output_tensor;
    tt::tt_metal::distributed::MeshWorkload worker_workload;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> metadata_output_buffer;
    tt::tt_metal::DeviceAddr metadata_output_addr = 0;
    if (worker_cores.has_value()) {
        const auto& backing = service.get_backing_tensor();
        output_tensor.emplace(tt::tt_metal::create_device_tensor(
            backing.tensor_spec(), mesh_device.get(), backing.tensor_topology()));

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

    // Push `src` through the selected input path, optionally appending an
    // inline metadata payload (empty span when metadata is disabled). NO
    // host-side synchronisation — returns as soon as the bytes are in the
    // socket FIFO. Subsequent flow control (waiting for the kernel to drain)
    // happens either via service.barrier() in the serial path, or via the
    // device-side worker handshake in the threaded path.
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

    // Dispatch one worker iteration and wait for it to complete on-device. The
    // service kernel's data_ready multicast (which only fires after the host's
    // matching push has been fully drained into the backing tensor) gates the
    // worker kernel; this Finish therefore transitively waits for the host
    // push too. Only called from the worker-sync path.
    auto consume_one = [&]() {
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), worker_workload, /*blocking=*/false);
        tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());
    };

    // Read every per-device sub-tensor of the tensor under test and compare
    // against the per-coord shard from an external mapper applied to `src`.
    // Worker-sync path: read from `output_tensor` (where the workers copy
    // input->output). No-worker path: read from the service's backing tensor.
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

            // Metadata verification: read service.get_metadata_addr() from a
            // representative worker core on this device and compare against
            // the host-side expected payload. The metadata buffer is
            // REPLICATED across the mesh and HEIGHT_SHARDED across worker_cores,
            // so any single worker core's slice is a valid spot-check — the
            // multicast hit the whole bbox atomically. Sample worker_cores
            // start_coord per device. EXPECT_* (not ASSERT_*) so a consumer
            // thread keeps draining remaining iters on mismatch instead of
            // wedging the writer in `forward_to_tensor`.
            if (cs.metadata_size_bytes > 0) {
                auto* d = mesh_device->get_device(coord);
                const auto* exp_meta_u8 = reinterpret_cast<const uint8_t*>(expected_meta.data());
                const std::vector<uint8_t> expected_meta_u8(exp_meta_u8, exp_meta_u8 + expected_meta.size());

                // Verify every worker core in worker_cores received the
                // multicast and copied the bytes correctly. Reading from the
                // WORKER-OWNED output region (populated by the worker kernel
                // before its atomic-inc), not the service-owned input region —
                // the latter is overwritten by the next iter's multicast and
                // would race the consumer thread's readback.
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

    // Deterministic per-iter source data so both threads can independently
    // compute the same expected payload without sharing memory. Iter 0 has
    // seed 0; later iters use seed = iter * 0x12345678u so a kernel that
    // silently exited after iter 0 would surface as a contents mismatch on
    // iter 1+ (subsumes the old "Write A / Write B" reuse check).
    auto make_iter_data = [&](uint32_t iter) {
        std::vector<uint32_t> v(cs.global_shape.volume());
        std::iota(v.begin(), v.end(), iter * 0x12345678u);
        return v;
    };

    // Deterministic per-iter inline metadata payload, byte-indexed so any
    // metadata_size_bytes (aligned or not) gets a unique pattern per iter and
    // per byte offset. Returns an empty vector when metadata is disabled —
    // the empty span passes the service's exact-size validation against
    // Config::metadata_size_bytes=0. The 0x9E37u multiplier is the upper half
    // of the fractional golden ratio (Knuth's multiplicative hash), giving
    // good spread across iters while keeping the byte-level math trivial.
    auto make_iter_metadata = [&](uint32_t iter) {
        std::vector<std::byte> v(cs.metadata_size_bytes);
        for (uint32_t i = 0; i < cs.metadata_size_bytes; ++i) {
            v[i] = std::byte{static_cast<uint8_t>((iter * 0x9E37u + i) & 0xFFu)};
        }
        return v;
    };

    // --- Iteration loop ---------------------------------------------------------
    // Two execution modes:
    //   * Threaded (worker-sync ON): writer + consumer threads, ZERO host sync.
    //     The metadata L1 readback in `verify` reads from a *worker-owned*
    //     output buffer (the worker kernel copies metadata input -> output
    //     before atomic-incing consumed_counter), so the next iter's service-
    //     side multicast can overwrite the metadata input region without
    //     racing the consumer's readback.
    //   * Serial (no worker-sync): push -> service.barrier() -> verify on the
    //     main thread, one iter at a time.
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
        // Order matters: consumer first so that if a verify failure throws
        // (or propagates exception via the future), we surface it before
        // waiting on the writer. Both must complete before `service` goes out
        // of scope — its dtor signals termination + wait_done, which would
        // race with an in-flight writer.
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
        run_h2d_stream_service_case(
            this->mesh_device_, cs, InputPath::Tensor, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
        run_h2d_stream_service_case(
            this->mesh_device_, cs, InputPath::Bytes, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
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
            run_h2d_stream_service_case(
                this->mesh_device_, cs, InputPath::Tensor, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
            run_h2d_stream_service_case(
                this->mesh_device_, cs, InputPath::Bytes, /*worker_cores=*/std::nullopt, /*num_iterations=*/10);
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

// W — Worker-sync handshake sweep across worker grid sizes.
//
// Wires up the optional `Config::worker_cores` path end-to-end:
//   * Service kernel multicasts data_ready_sem after each transfer.
//   * Worker kernels poll data_ready_sem locally, copy their assigned page slice
//     from the backing tensor to a separately-allocated output tensor, and
//     atomic-inc the per-device consumed_counter on the service core.
//   * Service kernel waits for num_workers more incs and proceeds.
//   * Host reads back `output_tensor` (not the backing tensor) and verifies.
//
// Per-device shape under fully-replicated mapping == global shape. The worker
// grid on this arch is 12 columns x 10 rows = 120 cores; rows exercise the
// extremes (single worker, mid-range, full grid) plus the original 4-worker
// baseline. `N` (page count) must be divisible by `num_workers` in every row.
TEST_F(H2DStreamServiceTest, Replicated_WorkerSync_Sweep) {
    struct Row {
        uint32_t per_row_size;
        uint32_t N;  // tensor pages per device (must satisfy N % num_workers == 0)
        CoreRange worker_cores;
        uint32_t num_iterations;
        // Inline metadata size in bytes. 0 = metadata path disabled. Must be
        // <= the smallest socket_page_size across the chunking sweep below
        // (single-metadata-page constraint enforced by H2DStreamService).
        uint32_t metadata_size_bytes;
        const char* label;
    };
    // Chunking variants explored for every (shape, worker_cores) row. Values
    // are in tensor-page units; per-row byte sizes are `cb_pages * per_row_bytes`
    // and `fifo_pages * per_row_bytes`. `derive_chunk_plan` will pick the
    // largest pages_per_chunk <= cb_pages that divides N, so cb_pages > N is
    // safe (clamps to N).
    struct Chunking {
        uint32_t cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        // Baseline: 4 workers in a single row, 4 pages each.
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 0, "4_workers_row"},
        // Single worker covering all 16 pages — exercises the num_workers==1 path
        // and the degenerate multicast (single destination).
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}, 20, 0, "1_worker"},
        // Full 12x10 worker grid = 120 cores, one page per worker. Page count is
        // bumped to 120 so the divisibility constraint holds.
        {640, 120, CoreRange{CoreCoord{0, 0}, CoreCoord{11, 9}}, 100, 0, "120_workers_full_grid"},

        // --- Metadata sweep ----------------------------------------------------
        // Same 4-worker setup as the baseline above; only the trailing inline
        // metadata size varies. All sizes are <= 2560 B (smallest socket_page_size
        // in the chunking sweep below: cb_pages=1, per_row_bytes=2560), so the
        // service's single-metadata-page constraint holds for every chunking.
        //
        // 16 B: sub-L1-alignment payload; exercises the page-pad path with a
        // tiny metadata head.
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 16, "4_workers_meta_16B"},
        // 256 B: common "small struct" size; multiple cache lines.
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 256, "4_workers_meta_256B"},
        // 2544 B: just under socket_page_size=2560 in the cb_pages=1 chunking;
        // upper-bound case where the kernel multicasts almost the entire
        // trailing page and the host pads only 16 B of zeros.
        {640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 20, 2544, "4_workers_meta_near_page"},
    };
    const Chunking chunkings[] = {
        // Smallest possible: one tensor page per socket page, FIFO holds one page.
        {1, 1, "cb1_fifo1"},
        // Page-at-a-time but deeper FIFO so host can fill ahead of kernel.
        {1, 8, "cb1_fifo8"},
        // 4-page chunks fanned out from a deeper FIFO. Exercises pages_per_chunk
        // > 1 path and the corresponding multi-page socket FIFO accounting.
        {4, 16, "cb4_fifo16"},
        // Mid-large allocation on both axes: 128*2560B = 320KB CB + 320KB FIFO.
        // fifo >= pages_per_chunk=min(N,128) holds for N ∈ {16, 120}.
        {128, 128, "cb128_fifo128"},
        // Near-1MB FIFO with small CB: 4*2560B = 10KB CB, 400*2560B ≈ 1MB FIFO.
        // Stresses the service-core L1 allocator's top-down headroom; CB stays
        // small so CB+FIFO total fits under the ~1MB usable L1 per service core.
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
                .scratch_cb_size_bytes = ch.cb_pages * per_row_bytes,
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

// Worker-sync handshake across multi-device sharding topologies.
//
// Mirrors `Sharded_Sweep`'s placement patterns (ShardRowsReplicateCols,
// ReplicateRowsShardCols, FullShard2D) AND mirrors Replicated_WorkerSync_Sweep's
// worker-grid axis (1 / 4 / 120 workers). The cross-product catches any
// regression where sharding interacts with worker-count: e.g., a multicast
// bbox bug that only surfaces on a full-grid run combined with a sharded
// per-device shape.
//
// Per-device shape is `[1, 1, N, per_row_size]` under every placement pattern
// (the mappers slice the global shape down to this); page count per device == N,
// and `N % num_workers == 0` must hold per row.
TEST_F(H2DStreamServiceTest, Sharded_WorkerSync_Sweep) {
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
        // Inline metadata size in bytes. 0 = metadata path disabled. Must be
        // <= the smallest socket_page_size across the chunking sweep below
        // (single-metadata-page constraint enforced by H2DStreamService).
        uint32_t metadata_size_bytes;
        const char* label;
    };
    // Chunking sweep — see comment in Replicated_WorkerSync_Sweep for rationale.
    struct Chunking {
        uint32_t cb_pages;
        uint32_t fifo_pages;
        const char* label;
    };
    const Row rows[] = {
        // 4-worker row. 16 / 4 = 4 pages per worker.
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 0, "4_workers_row"},
        // Single worker covers all per-device pages. Exercises the
        // num_workers==1 degenerate-multicast path with sharded per-device shapes.
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}, 0, "1_worker"},
        // Full 12x10 worker grid = 120 cores, one page per worker. N bumped
        // to 120 to keep divisibility; per-device tensor grows to [1,1,120,640].
        Row{640, 120, CoreRange{CoreCoord{0, 0}, CoreCoord{11, 9}}, 0, "120_workers_full_grid"},

        // --- Metadata sweep ----------------------------------------------------
        // Same 4-worker setup; only the trailing inline metadata size varies.
        // All sizes are <= 2560 B (smallest socket_page_size in the chunking
        // sweep), so the single-metadata-page constraint holds for every
        // chunking. The metadata cross-product with sharded placements catches
        // any interaction between the L1 multicast bbox and per-device sharded
        // shapes.
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 16, "4_workers_meta_16B"},
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 256, "4_workers_meta_256B"},
        Row{640, 16, CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}, 2544, "4_workers_meta_near_page"},
    };
    const Chunking chunkings[] = {
        {1, 1, "cb1_fifo1"},
        {1, 8, "cb1_fifo8"},
        {4, 16, "cb4_fifo16"},
        // Mid-large on both axes (320KB CB + 320KB FIFO).
        {128, 128, "cb128_fifo128"},
        // Near-1MB FIFO with small CB (10KB CB, ~1MB FIFO).
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
                    .scratch_cb_size_bytes = ch.cb_pages * per_row_bytes,
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

    // Shard on tensor dim 3 across mesh-dim 0, replicate across mesh-dim 1.
    if (num_rows >= 2) {
        run_pattern(
            "ShardRowsReplicateCols",
            {MeshMapperConfig::Shard{3}, MeshMapperConfig::Replicate{}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_rows * per_row_size});
            });
    }

    // Mirror: replicate across mesh-dim 0, shard tensor dim 3 across mesh-dim 1.
    if (num_cols >= 2) {
        run_pattern(
            "ReplicateRowsShardCols",
            {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, N, num_cols * per_row_size});
            });
    }

    // Full 2D shard: tensor dim 2 across mesh-dim 0, tensor dim 3 across mesh-dim 1.
    if (num_rows >= 2 && num_cols >= 2) {
        run_pattern(
            "FullShard2D",
            {MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{3}},
            [&](uint32_t N, uint32_t per_row_size) {
                return ttnn::Shape({1, 1, num_rows * N, num_cols * per_row_size});
            });
    }
}

// F — Preprocessor hook. Validates that `Config::preprocessor` runs on a
// service-owned scratch BEFORE the mapper distributes shards and that the
// scratch is correctly reused across calls.
//
// We model the ring-SDPA prefill reshuffle (see
// models/demos/deepseek_v3_b1/docs/ring_sdpa_prefill_reshuffling.md). For
// each test case `chunk_P_aligned` is set on a test-local variable that the
// preprocessor closure captures by reference; the preprocessor derives
// (c_start, intra) and rotates the input array so each column receives the
// slot positions it owns. Because the mapper shards dim 3 into `num_cols`
// contiguous slices of `W` elements each, the reshuffle works by reordering
// the input so the mapper's natural distribution lines up with the
// analytical expected.
//
// We DON'T exercise the device-side metadata multicast path here — a
// production caller would route `chunk_P_aligned` through the existing
// per-call metadata buffer (which also reaches the worker kernel), but
// that path is covered by the worker-sync sweep tests. Here `metadata` is
// empty and `metadata_size_bytes == 0`, isolating the preprocessor wiring.
//
// Per-coord expected (from doc section 6): after the reshuffle, column
// `k` receives `W` consecutive logical positions starting at
//
//   start_pos[k] = chunk_P_aligned                                       if k == c_start
//                  chunk_P_aligned + ((k - c_start) mod N_C) * W - intra otherwise
//
// modulo the full-iota wraparound at N_C*W (since the input is just
// `[0, 1, ..., N_C*W - 1]`).

namespace {
// Reshuffle helper — direct port of the Python `reshuffle` in section 5
// step 4 of the ring-SDPA doc. Volume-preserving column rotation of an
// N_C*W-element uint32 array.
void ring_sdpa_reshuffle(
    ttsl::Span<std::byte> bytes, uint32_t c_start, uint32_t intra, uint32_t N_C, uint32_t W) {
    const size_t volume = N_C * W;
    TT_FATAL(bytes.size() == volume * sizeof(uint32_t), "reshuffle: size mismatch");
    auto* in = reinterpret_cast<const uint32_t*>(bytes.data());
    std::vector<uint32_t> out(volume);

    // Column c_start gets two pieces: `[0, W - intra)` from the head + `[N_C*W - intra, N_C*W)`
    // from the tail (the wraparound across two stripe wraps).
    for (uint32_t i = 0; i < W - intra; ++i) {
        out[c_start * W + i] = in[i];
    }
    for (uint32_t i = 0; i < intra; ++i) {
        out[c_start * W + (W - intra) + i] = in[volume - intra + i];
    }

    // Every other column gets one contiguous slice.
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
    const auto mesh_shape = this->mesh_device_->shape();
    if (mesh_shape.dims() != 2) {
        GTEST_SKIP() << "Preprocessor reshuffle test requires a 2D mesh; got " << mesh_shape;
    }
    const uint32_t num_cols = mesh_shape[1];
    if (num_cols < 2) {
        GTEST_SKIP() << "Need num_cols >= 2 for sharded-along-cols reshuffle; got " << num_cols;
    }

    const uint32_t N_C = num_cols;
    // W chosen so the per-shard tensor page = W * sizeof(uint32_t) is a
    // multiple of the PCIe alignment (64 B on Blackhole). 64 uint32 = 256 B.
    constexpr uint32_t W = 64;

    // Shard dim 3 across cols, replicate across rows — same axis structure as
    // ring-SDPA prefill (`{Replicate{}, Shard{3}}`).
    ttsl::SmallVector<MeshMapperConfig::Placement> placements{
        MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}};

    const auto global_shape = ttnn::Shape({1, 1, 1, N_C * W});
    const auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    const auto global_spec = TensorSpec(global_shape, tensor_layout);

    // Preprocessor — pulls the current chunk's `chunk_P_aligned` out of a
    // test-local variable captured by reference, derives (c_start, intra), and
    // mutates `bytes` in place via the column rotation. The metadata span is
    // intentionally unused here: `metadata_size_bytes == 0` below means the
    // service's per-call metadata contract is "empty span only" and this
    // test must honour it. A production caller would tie the preprocessor's
    // params to the metadata buffer that ALSO flows to the worker kernels
    // (see ring_sdpa_prefill_reshuffling.md) — that path is exercised by the
    // worker-sync sweep tests, not this one.
    uint32_t current_chunk_P_aligned = 0;
    auto preprocessor = [N_C, &current_chunk_P_aligned](
                            ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata) {
        (void)metadata;
        const uint32_t c_start = (current_chunk_P_aligned / W) % N_C;
        const uint32_t intra = current_chunk_P_aligned % W;
        ring_sdpa_reshuffle(bytes, c_start, intra, N_C, W);
    };

    // Tensor page size = bytes per write into a per-shard DRAM bank. With the
    // mapper sharding dim 3 across N_C columns, the per-shard tensor is
    // [1, 1, 1, W], so its row (and only page, since num_pages == 1) is
    // `W * sizeof(uint32_t)` bytes — NOT N_C * W * sizeof(uint32_t).
    const uint32_t tensor_page_size_bytes = W * sizeof(uint32_t);
    tt::tt_metal::H2DStreamService::Config cfg{
        .global_spec = global_spec,
        .mapper = create_mesh_mapper(*this->mesh_device_, MeshMapperConfig{.placements = placements}),
        .socket_buffer_type = BufferType::L1,
        .fifo_size_bytes = tensor_page_size_bytes,
        .scratch_cb_size_bytes = tensor_page_size_bytes,
        .socket_mode = H2DMode::DEVICE_PULL,
        .worker_cores = std::nullopt,
        .metadata_size_bytes = 0,  // device-side metadata path is independent of this test
        .preprocessor = preprocessor,
    };
    tt::tt_metal::H2DStreamService service(this->mesh_device_, std::move(cfg));
    ASSERT_NE(service.get_backing_tensor().buffer(), nullptr);

    // Source: simple iota so per-coord expected values are predictable.
    std::vector<uint32_t> src(N_C * W);
    std::iota(src.begin(), src.end(), 0);

    // Sweep a handful of (chunk_P_aligned) values covering: identity, pure
    // column rotation (intra=0, c_start>0), pure intra rotation (c_start=0,
    // intra>0), and combined.
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

        // Empty metadata to honour `metadata_size_bytes == 0` contract.
        service.forward_to_tensor(bytes_span, /*metadata=*/{});
        service.barrier();

        // Compute the analytical expected per-coord array — re-run the same
        // reshuffle on a host-local copy, then split into N_C contiguous
        // slices of W (the mapper's natural distribution along dim 3).
        std::vector<uint32_t> expected_global = src;
        const uint32_t c_start = (tc.chunk_P_aligned / W) % N_C;
        const uint32_t intra = tc.chunk_P_aligned % W;
        ring_sdpa_reshuffle(
            ttsl::Span<std::byte>(
                reinterpret_cast<std::byte*>(expected_global.data()),
                expected_global.size() * sizeof(uint32_t)),
            c_start, intra, N_C, W);

        // Verify every per-device sub-tensor.
        auto subs = ttnn::distributed::get_device_tensors(service.get_backing_tensor());
        ASSERT_EQ(subs.size(), this->mesh_device_->num_devices());
        for (auto& sub : subs) {
            const auto coords = sub.device_storage().get_coords();
            ASSERT_EQ(coords.size(), 1u);
            const auto& coord = coords[0];
            // Replicate-on-rows: every row sees the same shard for a given column.
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
