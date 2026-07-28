// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// MPI-based cross-process loopback socket test.
// Rank 0 (launcher): opens device, creates H2D+D2H sockets, launches loopback kernel.
// Rank 1 (connector): connects to sockets via descriptors exported by the launcher, drives data.
//
// Run with: mpirun -np 2 ./multiproc_loopback_test
//

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gtest/gtest.h"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "impl/context/metal_context.hpp"

static int g_world_rank = -1;
static int g_world_size = -1;
static int test_counter = 0;

namespace tt::tt_metal::distributed {
namespace {

using namespace tt::tt_metal::distributed::multihost;

constexpr uint32_t NUM_ITERATIONS = 1000;

struct LoopbackConfig {
    std::size_t fifo_size;
    std::size_t page_size;
    std::size_t data_size;
};

constexpr LoopbackConfig loopback_configs[] = {
    {1024, 64, 1024},       // No wrap
    {1024, 64, 32768},      // Even wrap
    {4096, 1088, 78336},    // Uneven wrap
    {16512, 1088, 156672},  // Uneven wrap, multiple host pages
};

const MeshCoreCoord SOCKET_CORE = {MeshCoordinate(0, 0), CoreCoord(0, 0)};

void run_launcher(
    const std::shared_ptr<MeshDevice>& mesh_device,
    H2DMode h2d_mode,
    const LoopbackConfig& cfg,
    uint32_t num_iterations,
    const std::string& socket_id_suffix) {
    const auto& socket_core = SOCKET_CORE;
    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", socket_id_suffix);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", socket_id_suffix);

    auto h2d_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, cfg.fifo_size, h2d_mode);
    h2d_socket.export_descriptor(h2d_socket_id);

    auto d2h_socket = D2HSocket(mesh_device, socket_core, cfg.fifo_size);
    d2h_socket.export_descriptor(d2h_socket_id);

    // L1 landing slot for DEVICE_PULL: the H2D FIFO lives in pinned host memory, so
    // the kernel needs a page of local L1 to pull into before looping back to D2H.
    auto scratch_shard_params =
        ShardSpecBuffer(CoreRangeSet(socket_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig scratch_local_config{
        .page_size = cfg.page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(scratch_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto scratch_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = cfg.page_size}, scratch_local_config, mesh_device.get());

    auto program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(h2d_socket.get_config_buffer_address()),
                static_cast<uint32_t>(d2h_socket.get_config_buffer_address()),
                static_cast<uint32_t>(cfg.page_size),
                static_cast<uint32_t>(cfg.data_size),
                num_iterations,
                static_cast<uint32_t>(h2d_mode == H2DMode::DEVICE_PULL),
                static_cast<uint32_t>(scratch_buffer->address()),
            }});

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(socket_core.device_coord), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device->mesh_command_queue());
}

void run_launcher(const std::shared_ptr<MeshDevice>& mesh_device, H2DMode h2d_mode, const LoopbackConfig& cfg) {
    run_launcher(mesh_device, h2d_mode, cfg, NUM_ITERATIONS, fmt::to_string(test_counter));
}

// Drive a contiguous range of iterations [start_iter, end_iter) against an
// already-attached pair of sockets. Takes raw pointers so it can be used both
// with std::vector buffers (pass .data()) and with mmap-shared buffers in the
// crash-safe test. Does NOT call barrier() — the caller decides whether to
// synchronize (e.g. the crash-safe child intentionally skips the barriers).
void drive_iter_range(
    H2DSocket& h2d_socket,
    D2HSocket& d2h_socket,
    const uint32_t* src,
    uint32_t* dst,
    const LoopbackConfig& cfg,
    uint32_t start_iter,
    uint32_t end_iter) {
    uint32_t page_size_words = cfg.page_size / sizeof(uint32_t);
    uint32_t data_size_words = cfg.data_size / sizeof(uint32_t);
    uint32_t num_txns = cfg.data_size / cfg.page_size;

    std::thread write_thread([&]() {
        for (uint32_t i = start_iter; i < end_iter; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                h2d_socket.write(const_cast<uint32_t*>(src) + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    std::thread read_thread([&]() {
        for (uint32_t i = start_iter; i < end_iter; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                d2h_socket.read(dst + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    write_thread.join();
    read_thread.join();
}

void run_connector(const LoopbackConfig& cfg) {
    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", test_counter);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", test_counter);

    auto h2d_socket = H2DSocket::connect(h2d_socket_id, 30000);
    auto d2h_socket = D2HSocket::connect(d2h_socket_id, 30000);

    h2d_socket->set_page_size(cfg.page_size);
    d2h_socket->set_page_size(cfg.page_size);

    uint32_t data_size_words = cfg.data_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(data_size_words * NUM_ITERATIONS);
    std::vector<uint32_t> dst_vec(data_size_words * NUM_ITERATIONS, 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    drive_iter_range(*h2d_socket, *d2h_socket, src_vec.data(), dst_vec.data(), cfg, 0, NUM_ITERATIONS);
    h2d_socket->barrier();
    d2h_socket->barrier();

    EXPECT_EQ(src_vec, dst_vec) << "Loopback verification FAILED (fifo=" << cfg.fifo_size << " page=" << cfg.page_size
                                << " data=" << cfg.data_size << ")";
}

// Drive consecutive H2D/D2H socket attach/detach cycles against a single
// launcher kernel. Connector k handles iters_per_connector[k] iterations and
// then destructs (close SHM, no unlink). The next connector calls
// H2DSocket::connect / D2HSocket::connect anew and must pick up the prior
// connector's state (bytes_sent / bytes_acked / write_ptr / read_ptr /
// page_size) from SHM. Verifies the connector-state persistence path.
void run_sequential_connectors(
    const LoopbackConfig& cfg, const std::vector<uint32_t>& iters_per_connector, const std::string& socket_id_suffix) {
    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", socket_id_suffix);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", socket_id_suffix);

    const uint32_t total_iters = std::accumulate(iters_per_connector.begin(), iters_per_connector.end(), 0u);
    const uint32_t data_size_words = cfg.data_size / sizeof(uint32_t);

    std::vector<uint32_t> src_vec(data_size_words * total_iters);
    std::vector<uint32_t> dst_vec(data_size_words * total_iters, 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    uint32_t start = 0;
    for (uint32_t iters : iters_per_connector) {
        auto h2d_socket = H2DSocket::connect(h2d_socket_id, 30000);
        auto d2h_socket = D2HSocket::connect(d2h_socket_id, 30000);

        // Every prior connector destructed cleanly (owner stamps clean_shutdown=1
        // on construct; each iteration's destructor reasserts it), so each
        // attach here must observe clean_shutdown == 1.
        EXPECT_TRUE(h2d_socket->had_clean_prior_shutdown())
            << "H2D connector saw unclean prior shutdown after clean destructor";
        EXPECT_TRUE(d2h_socket->had_clean_prior_shutdown())
            << "D2H connector saw unclean prior shutdown after clean destructor";

        // set_page_size is idempotent and exercises the page_size_/fifo_curr_size_
        // flush path on every attach.
        h2d_socket->set_page_size(cfg.page_size);
        d2h_socket->set_page_size(cfg.page_size);

        uint32_t end = start + iters;
        drive_iter_range(*h2d_socket, *d2h_socket, src_vec.data(), dst_vec.data(), cfg, start, end);
        h2d_socket->barrier();
        d2h_socket->barrier();
        // h2d_socket / d2h_socket destroyed here: SHM is unmapped (not unlinked).
        // Persistent state is left behind in the SHM region for the next connector
        // iteration.
        start = end;
    }

    EXPECT_EQ(src_vec, dst_vec) << "Sequential connector loopback FAILED (fifo=" << cfg.fifo_size
                                << " page=" << cfg.page_size << " data=" << cfg.data_size
                                << " num_connectors=" << iters_per_connector.size() << ")";
}

void run_sequential_connectors(
    const LoopbackConfig& cfg,
    uint32_t num_connectors,
    uint32_t iters_per_connector,
    const std::string& socket_id_suffix) {
    run_sequential_connectors(cfg, std::vector<uint32_t>(num_connectors, iters_per_connector), socket_id_suffix);
}

// Verify that the per-op flush in push_bytes / pop_bytes / set_page_size leaves
// SHM connector-state in a consistent shape even when the driving process is
// killed without running destructors or barriers. A fork()'d child drives
// `child_iters` iterations and then _exit(0)s — no h2d.barrier(), no D2H
// barrier, no socket destruction. The parent then attaches a fresh connector
// and drives the remaining iterations to completion. Without per-op flushing,
// the parent would see all-zero connector state and desync from the device.
void run_crash_safe_connector(
    const LoopbackConfig& cfg, uint32_t total_iters, uint32_t child_iters, const std::string& socket_id_suffix) {
    ASSERT_LT(child_iters, total_iters);

    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", socket_id_suffix);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", socket_id_suffix);

    const uint32_t data_size_words = cfg.data_size / sizeof(uint32_t);
    const size_t total_words = static_cast<size_t>(data_size_words) * total_iters;
    const size_t total_bytes = total_words * sizeof(uint32_t);

    // mmap with MAP_SHARED | MAP_ANONYMOUS so child writes to dst_vec are
    // visible to the parent post-_exit. Layout: [src_vec][dst_vec].
    void* shared = mmap(nullptr, 2 * total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(shared, MAP_FAILED) << "mmap failed: " << std::strerror(errno);
    auto* src_vec = static_cast<uint32_t*>(shared);
    auto* dst_vec = src_vec + total_words;
    std::iota(src_vec, src_vec + total_words, 0u);
    std::memset(dst_vec, 0, total_bytes);

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork failed: " << std::strerror(errno);

    if (pid == 0) {
        // Child: attach, drive child_iters of loopback, then _exit(0) WITHOUT
        // running destructors or barriers. Per-op SHM flushes are the only
        // mechanism leaving consistent state behind.
        auto h2d_socket = H2DSocket::connect(h2d_socket_id, 30000);
        auto d2h_socket = D2HSocket::connect(d2h_socket_id, 30000);
        h2d_socket->set_page_size(cfg.page_size);
        d2h_socket->set_page_size(cfg.page_size);
        drive_iter_range(*h2d_socket, *d2h_socket, src_vec, dst_vec, cfg, 0, child_iters);
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status)) << "Child did not exit cleanly (signal "
                                   << (WIFSIGNALED(status) ? WTERMSIG(status) : -1) << ")";
    ASSERT_EQ(WEXITSTATUS(status), 0);

    {
        auto h2d_socket = H2DSocket::connect(h2d_socket_id, 30000);
        auto d2h_socket = D2HSocket::connect(d2h_socket_id, 30000);

        // Child _exit()'d without running destructors, so the SHM clean_shutdown
        // flag must still be 0 here — the resumed connectors should report the
        // prior shutdown as unclean.
        EXPECT_FALSE(h2d_socket->had_clean_prior_shutdown())
            << "H2D connector failed to detect crashed child's unclean shutdown";
        EXPECT_FALSE(d2h_socket->had_clean_prior_shutdown())
            << "D2H connector failed to detect crashed child's unclean shutdown";

        // The parent intentionally does NOT call set_page_size — it must be
        // inherited from the child's flushed SHM state, alongside the byte and
        // pointer counters.
        EXPECT_EQ(h2d_socket->get_page_size(), cfg.page_size) << "page_size not inherited from crashed child connector";
        EXPECT_EQ(d2h_socket->get_page_size(), cfg.page_size) << "page_size not inherited from crashed child connector";

        drive_iter_range(*h2d_socket, *d2h_socket, src_vec, dst_vec, cfg, child_iters, total_iters);
        h2d_socket->barrier();
        d2h_socket->barrier();
    }

    bool ok = std::equal(src_vec, src_vec + total_words, dst_vec);
    EXPECT_TRUE(ok) << "Crash-safe loopback verification FAILED (fifo=" << cfg.fifo_size << " page=" << cfg.page_size
                    << " data=" << cfg.data_size << " child_iters=" << child_iters << " total_iters=" << total_iters
                    << ")";

    ASSERT_EQ(munmap(shared, 2 * total_bytes), 0);
}

class MultiProcLoopbackFixture : public MeshDeviceFixtureBase {
protected:
    MultiProcLoopbackFixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 1}}) {}

    void SetUp() override {
        ASSERT_EQ(g_world_size, 2) << "This test requires exactly 2 MPI ranks";
        rank_ = g_world_rank;
        if (rank_ == 0) {
            MeshDeviceFixtureBase::SetUp();
        }
    }

    void TearDown() override {
        if (rank_ == 0) {
            MeshDeviceFixtureBase::TearDown();
        }
    }

    int rank_ = -1;
};

TEST_F(MultiProcLoopbackFixture, CrossProcessLoopback) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        for (const auto& cfg : loopback_configs) {
            if (rank_ == 0) {
                run_launcher(mesh_device_, h2d_mode, cfg);
            } else {
                run_connector(cfg);
            }
            test_counter++;
        }
    }
}

// Verify connector-state persistence across sequential connector attach/detach
// cycles. The launcher kernel runs once for total_iters; rank 1 drives that
// kernel from `num_connectors` consecutive H2DSocket/D2HSocket instances.
// Without state persistence, the second connector's counters would reset to
// zero, desyncing it from the device-resident pointers and hanging.
TEST_F(MultiProcLoopbackFixture, BackToBackConnectorPersistence) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    // Uneven-wrap config is the most stressing for connector handoff: the FIFO
    // pointer is at an arbitrary offset when the prior connector destructs.
    const LoopbackConfig cfg = {4096, 1088, 78336};
    constexpr uint32_t NUM_CONNECTORS = 8;
    constexpr uint32_t ITERS_PER_CONNECTOR = 200;
    constexpr uint32_t TOTAL_ITERS = NUM_CONNECTORS * ITERS_PER_CONNECTOR;

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        std::string suffix = fmt::format("seq_{}", test_counter);
        if (rank_ == 0) {
            run_launcher(mesh_device_, h2d_mode, cfg, TOTAL_ITERS, suffix);
        } else {
            run_sequential_connectors(cfg, NUM_CONNECTORS, ITERS_PER_CONNECTOR, suffix);
        }
        test_counter++;
    }
}

// Verify that the per-op SHM flushes in push_bytes / pop_bytes / set_page_size
// are sufficient to recover when a driver process dies without running
// destructors. rank 1 fork()s a child that drives part of the kernel's
// iterations and _exit(0)s; the parent then attaches a fresh connector and
// completes the remaining iterations using only the state the child flushed
// per-op.
TEST_F(MultiProcLoopbackFixture, CrashSafeConnectorPersistence) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    const LoopbackConfig cfg = {4096, 1088, 78336};
    constexpr uint32_t CHILD_ITERS = 100;
    constexpr uint32_t TOTAL_ITERS = 200;

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        std::string suffix = fmt::format("crash_{}", test_counter);
        if (rank_ == 0) {
            run_launcher(mesh_device_, h2d_mode, cfg, TOTAL_ITERS, suffix);
        } else {
            run_crash_safe_connector(cfg, TOTAL_ITERS, CHILD_ITERS, suffix);
        }
        test_counter++;
    }
}

// Connector handoff at non-uniform iteration boundaries. The connector handoff
// happens at arbitrary FIFO offsets (not iter-aligned multiples), confirming
// the persisted write_ptr / read_ptr / bytes_sent / bytes_acked are sufficient
// regardless of where the previous connector stopped.
TEST_F(MultiProcLoopbackFixture, UnevenIterDistribution) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    const LoopbackConfig cfg = {4096, 1088, 78336};
    const std::vector<uint32_t> iters_per_connector = {1, 47, 391, 161};  // 600 total
    const uint32_t total_iters = std::accumulate(iters_per_connector.begin(), iters_per_connector.end(), 0u);

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        std::string suffix = fmt::format("uneven_{}", test_counter);
        if (rank_ == 0) {
            run_launcher(mesh_device_, h2d_mode, cfg, total_iters, suffix);
        } else {
            run_sequential_connectors(cfg, iters_per_connector, suffix);
        }
        test_counter++;
    }
}

// True multi-process connector reuse: rank 1 fork()s `num_connectors` children
// sequentially, each one a complete attach → drive → clean-destruct → exit
// cycle in its OWN OS process. Each child has its own PID, address space, fd
// table, and PCIeCoreWriter cluster cache — matching the production scenario
// the persistent state was added to support ("process 1 drives the socket and
// exits, process 2 launches and connects").
//
// Only the first child calls set_page_size, exercising the cross-process
// page_size_/fifo_curr_size_ inheritance in addition to the counter/pointer
// inheritance.
void run_multi_process_connectors(
    const LoopbackConfig& cfg,
    uint32_t num_connectors,
    uint32_t iters_per_connector,
    const std::string& socket_id_suffix) {
    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", socket_id_suffix);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", socket_id_suffix);

    const uint32_t total_iters = num_connectors * iters_per_connector;
    const uint32_t data_size_words = cfg.data_size / sizeof(uint32_t);
    const size_t total_words = static_cast<size_t>(data_size_words) * total_iters;
    const size_t total_bytes = total_words * sizeof(uint32_t);

    // mmap with MAP_SHARED | MAP_ANONYMOUS so each child's writes to dst_vec
    // are visible to the parent for verification. Layout: [src_vec][dst_vec].
    void* shared = mmap(nullptr, 2 * total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(shared, MAP_FAILED) << "mmap failed: " << std::strerror(errno);
    auto* src_vec = static_cast<uint32_t*>(shared);
    auto* dst_vec = src_vec + total_words;
    std::iota(src_vec, src_vec + total_words, 0u);
    std::memset(dst_vec, 0, total_bytes);

    for (uint32_t k = 0; k < num_connectors; ++k) {
        pid_t pid = fork();
        ASSERT_NE(pid, -1) << "fork failed: " << std::strerror(errno);

        if (pid == 0) {
            // Child scope: socket destructors run on scope exit, then _exit(0)
            // skips global teardown (atexit handlers, MPI dtors) that belongs
            // to the parent.
            {
                auto h2d_socket = H2DSocket::connect(h2d_socket_id, 30000);
                auto d2h_socket = D2HSocket::connect(d2h_socket_id, 30000);

                if (k == 0) {
                    h2d_socket->set_page_size(cfg.page_size);
                    d2h_socket->set_page_size(cfg.page_size);
                } else {
                    // Subsequent children inherit page_size from prior child via SHM.
                    if (h2d_socket->get_page_size() != cfg.page_size || d2h_socket->get_page_size() != cfg.page_size) {
                        _exit(2);  // signal inheritance failure to the parent
                    }
                }

                uint32_t start = k * iters_per_connector;
                uint32_t end = start + iters_per_connector;
                drive_iter_range(*h2d_socket, *d2h_socket, src_vec, dst_vec, cfg, start, end);
                h2d_socket->barrier();
                d2h_socket->barrier();
            }
            _exit(0);
        }

        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFEXITED(status)) << "Connector child " << k << " did not exit cleanly (signal "
                                       << (WIFSIGNALED(status) ? WTERMSIG(status) : -1) << ")";
        ASSERT_EQ(WEXITSTATUS(status), 0) << "Connector child " << k << " exited with status " << WEXITSTATUS(status)
                                          << (WEXITSTATUS(status) == 2 ? " (page_size inheritance failed)" : "");
    }

    bool ok = std::equal(src_vec, src_vec + total_words, dst_vec);
    EXPECT_TRUE(ok) << "Multi-process connector reuse FAILED (fifo=" << cfg.fifo_size << " page=" << cfg.page_size
                    << " data=" << cfg.data_size << " num_connectors=" << num_connectors << ")";

    ASSERT_EQ(munmap(shared, 2 * total_bytes), 0);
}

// Stress repeated attach/detach to catch accumulated drift or per-attach
// leakage. Uses many short-lived connectors over the same launcher kernel.
TEST_F(MultiProcLoopbackFixture, ManyShortConnectors) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    const LoopbackConfig cfg = {4096, 1088, 78336};
    constexpr uint32_t NUM_CONNECTORS = 30;
    constexpr uint32_t ITERS_PER_CONNECTOR = 10;
    constexpr uint32_t TOTAL_ITERS = NUM_CONNECTORS * ITERS_PER_CONNECTOR;

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        std::string suffix = fmt::format("many_{}", test_counter);
        if (rank_ == 0) {
            run_launcher(mesh_device_, h2d_mode, cfg, TOTAL_ITERS, suffix);
        } else {
            run_sequential_connectors(cfg, NUM_CONNECTORS, ITERS_PER_CONNECTOR, suffix);
        }
        test_counter++;
    }
}

// End-to-end test that connectors are truly cross-process reusable: rank 1
// fork()s NUM_CONNECTORS distinct children sequentially against a single
// launcher kernel. Each child is a fresh OS process with its own PID, fd
// table, and PCIeCoreWriter cluster cache. This mirrors the production
// scenario the persistent state mechanism was added for ("process 1 drives
// the socket, exits, process 2 launches and connects, etc.").
TEST_F(MultiProcLoopbackFixture, MultiProcessConnectorReuse) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    const LoopbackConfig cfg = {4096, 1088, 78336};
    constexpr uint32_t NUM_CONNECTORS = 4;
    constexpr uint32_t ITERS_PER_CONNECTOR = 50;
    constexpr uint32_t TOTAL_ITERS = NUM_CONNECTORS * ITERS_PER_CONNECTOR;

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        std::string suffix = fmt::format("mpr_{}", test_counter);
        if (rank_ == 0) {
            run_launcher(mesh_device_, h2d_mode, cfg, TOTAL_ITERS, suffix);
        } else {
            run_multi_process_connectors(cfg, NUM_CONNECTORS, ITERS_PER_CONNECTOR, suffix);
        }
        test_counter++;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed

int main(int argc, char** argv) {
    using namespace tt::tt_metal::distributed::multihost;

    DistributedContext::create(argc, argv);
    const auto& world = DistributedContext::get_current_world();
    g_world_rank = *world->rank();
    g_world_size = *world->size();

    auto local_ctx = world->split(Color(g_world_rank), Key(0));
    DistributedContext::set_current_world(local_ctx);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
