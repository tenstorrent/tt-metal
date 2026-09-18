// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::distributed::host_socket_test {

// Sender and receiver rank for every test in this suite.
constexpr multihost::Rank kSenderRank{0};
constexpr multihost::Rank kReceiverRank{1};

// SOCKET_MODE values understood by host_socket_{sender,receiver}.cpp.
constexpr uint32_t kModeHostTransport = 1;
constexpr uint32_t kModeD2D = 2;

inline const char* kSenderKernel = "tests/tt_metal/tt_metal/test_kernels/misc/socket/host_socket_sender.cpp";
inline const char* kReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/misc/socket/host_socket_receiver.cpp";
inline const char* kPingPongKernel = "tests/tt_metal/tt_metal/test_kernels/misc/socket/host_socket_pingpong.cpp";
inline const char* kEchoKernel = "tests/tt_metal/tt_metal/test_kernels/misc/socket/host_socket_echo.cpp";

// Warmup laps the latency kernels run untimed; must match WARMUP_ITERS in
// tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_noc_utils.h.
constexpr uint32_t kWarmupIters = 5;

/// min / p50 / avg / p99 / max over a sample set, in the samples' own unit.
struct Percentiles {
    double min = 0;
    double p50 = 0;
    double avg = 0;
    double p99 = 0;
    double max = 0;
    size_t count = 0;
};
Percentiles summarize(std::vector<double> samples);

/// Appends one latency row to $TT_HOST_SOCKET_CSV_LATENCY, if set.
void record_latency(const std::string& label, uint32_t page_size, const Percentiles& us);

struct Params {
    uint32_t page_size = 14336;  // 14 KiB: the target packet size. Not a power of two.
    uint32_t fifo_pages = 16;    // ring depth; must cover bandwidth x round-trip
    uint32_t num_cores = 1;
    uint64_t bytes_per_core = 0;  // 0 => one FIFO's worth
    uint32_t device_id = 0;
    uint32_t iterations = 1;   // transfers per device session
    double min_seconds = 0.0;  // keep iterating until this much time has passed

    uint32_t fifo_size() const { return page_size * fifo_pages; }
    uint64_t data_size() const { return bytes_per_core != 0 ? bytes_per_core : fifo_size(); }
};

/// Reads overrides from the environment so a launcher can sweep without a rebuild.
Params params_from_env(Params defaults = {});

/// Environment lookups with a fallback, for knobs that are not part of Params.
uint64_t env_or(const char* name, uint64_t fallback);
double env_or_double(const char* name, double fallback);

/// AI clock in MHz, which is also cycles per microsecond.
double get_cycles_per_us(const MeshDevice& mesh_device);

/// An L1 buffer sharded to a single core.
std::shared_ptr<MeshBuffer> make_core_l1_buffer(MeshDevice* device, const CoreCoord& core, uint32_t size);

/// True when pinned host memory can be mapped to the NOC (vIOMMU enabled). Every
/// test skips rather than fails when this is false.
bool host_sockets_supported(const std::shared_ptr<MeshDevice>& mesh_device);

/// Deterministic payload for a core, computed identically on both ranks.
std::vector<uint32_t> payload_for_core(uint32_t core_index, uint64_t size_bytes);

/// Runs `params.iterations` transfers (at least `params.min_seconds` worth) over one
/// HostMeshSocket, verifying every byte on the receiving rank when `verify` is set.
/// Writes achieved throughput in GB/s to `gbps_out` on the sender rank (0 on the
/// receiver). Returns void so gtest's ASSERT_* and GTEST_SKIP can be used inside.
/// `ack_latency_ns_out`, when given, collects the relay's forward-to-credit
/// samples (the streaming ack round trip) and enables that sampling.
void run_transfer(
    const Params& params, bool verify, double* gbps_out = nullptr, std::vector<uint64_t>* ack_latency_ns_out = nullptr);

/// Appends one benchmark row to $TT_HOST_SOCKET_CSV, if that is set. Header is
/// written when the file is created, so a sweep can concatenate runs.
void record_result(const Params& params, double gbps);

}  // namespace tt::tt_metal::distributed::host_socket_test
