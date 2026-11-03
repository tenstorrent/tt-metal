// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental API for user kernels that stay resident on the programmable DRAM cores (Blackhole DRISCs) while
// fast dispatch keeps running on the worker grid: e.g. weight senders that serve every launch of an op.
//
// The flow:
//   1. AreDramCoreKernelsSupported(mesh) and GetDramCores(mesh, coord, bank) pick the cores (a bank's DRAM cores,
//      their NoC coordinates, which NIUs firmware left in stream mode, their GDDR MPFE port).
//   2. WriteDramCoreL1 seeds the kernels' control words (in GetDramCoreKernelL1Region) before the launch.
//   3. LaunchDramCoreKernels builds one program per device of the mesh and launches it by slow dispatch next to the
//      active fast-dispatch session. It returns at once; the kernels run until they exit on their own.
//   4. The caller stops them through its own protocol (e.g. a stop word written with WriteDramCoreL1, an
//      acknowledgement polled with WaitForDramCoreL1Word), then DramCoreKernels::wait_done(). Kernels still running
//      when the mesh device closes are stopped there through the same protocol (LaunchDramCoreKernels' on_close).
//
// The kernels own the DRAM cores they run on: they cannot share a core with the tensor prefetcher
// (StartTensorPrefetcher) or the streaming profiler's relays, which use the same L1 region.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCoordinate;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

// True if `mesh_device` has programmable DRAM cores (Blackhole with firmware >= 19.12.0.0) and the streaming
// profiler, which keeps resident relays on the same cores, is off. LaunchDramCoreKernels TT_FATALs otherwise.
bool AreDramCoreKernelsSupported(const distributed::MeshDevice& mesh_device);

// The DRAM-core L1 range a resident kernel may use, the same on every DRAM core: above the zone the runtime keeps
// for DRAM-sender PrefetcherPipe state. Addresses are as the kernel sees them (not NoC-offset).
struct DramCoreL1Region {
    uint32_t base = 0;
    uint32_t size = 0;
};
DramCoreL1Region GetDramCoreKernelL1Region(const distributed::MeshDevice& mesh_device);

// One programmable DRAM core of a bank, on the device at a given mesh coordinate.
struct DramCoreInfo {
    // {bank, index}: what DramCoreKernelConfig::core and WriteDramCoreL1 take. Index 0 is the bank's NOC0 worker
    // endpoint, 1 its NOC1 worker endpoint, then the bank's other DRAM cores.
    CoreCoord logical_core;
    // NoC address of the core (what another core's kernel targets it with).
    CoreCoord virtual_core;
    // Position on the NOC0 grid (physical), for hop distances.
    CoreCoord noc0_core;
    // Bit n set: the core is a DRAM view's NOC n endpoint, so that NIU forwards DRAM accesses (NOC2AXI) and a
    // kernel on the core cannot initiate transactions on NOC n. 0: both NIUs are in stream mode.
    uint8_t endpoint_noc_mask = 0;
    // The core's port on the bank's GDDR memory controller multi-port front end (P1..P3).
    uint32_t mpfe_port = 0;
    // False for a core no kernel can run on (on Blackhole, the NOC0 worker endpoint, owned by system firmware).
    bool usable = false;
};
// Every DRAM core of `bank` (0 .. dram_grid_size().x - 1), in logical-index order. `mesh_coord` must name a device
// this rank drives.
std::vector<DramCoreInfo> GetDramCores(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord, uint32_t bank);

// The NOC0 grid (physical) position of a logical worker core, on the device at `mesh_coord` (a device this rank
// drives): pairs with DramCoreInfo::noc0_core for hop distances, and is what a worker reads as its own NoC
// coordinates.
CoreCoord GetWorkerNoc0Core(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord, const CoreCoord& logical_core);

// One resident kernel on one DRAM core.
struct DramCoreKernelConfig {
    std::string kernel_file;
    // Logical DRAM core (DramCoreInfo::logical_core), usable.
    CoreCoord core;
    // The NoC the kernel's default calls use; the NIU must be in stream mode (DramCoreInfo::endpoint_noc_mask).
    NOC noc = NOC::NOC_0;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
    std::vector<uint32_t> runtime_args;
};

struct DramCoreKernelsState;

// Kernels launched by LaunchDramCoreKernels. Destroying the handle neither stops nor waits for them (the mesh device
// still stops them at close).
class DramCoreKernels {
public:
    explicit DramCoreKernels(std::shared_ptr<DramCoreKernelsState> state);
    ~DramCoreKernels();
    DramCoreKernels(const DramCoreKernels&) = delete;
    DramCoreKernels& operator=(const DramCoreKernels&) = delete;

    // False once wait_done() returned or the mesh device closed (after which the device is gone: the caller's stop
    // protocol must not touch it).
    bool running() const;
    // Blocks until every kernel has exited. Call it once the caller's own protocol has told them to exit.
    void wait_done();

private:
    std::shared_ptr<DramCoreKernelsState> state_;
};

// Launches the kernels `kernels(coord)` returns (at most one per DRAM core) on every device of the mesh this rank
// drives, one program a device, by slow dispatch beside the active fast dispatch. Returns once they are launched.
// `on_close`: the caller's stop protocol, called if the kernels still run when the mesh device closes (after its
// command queues have finished); the close then waits for them to exit.
std::unique_ptr<DramCoreKernels> LaunchDramCoreKernels(
    distributed::MeshDevice& mesh_device,
    const std::function<std::vector<DramCoreKernelConfig>(const distributed::MeshCoordinate&)>& kernels,
    std::function<void()> on_close = {});

// Host access to DRAM-core L1 (`address` as the kernel sees it, 4 B aligned).
// Writes `words` to the core on every device of the mesh this rank drives, flushed before it returns.
void WriteDramCoreL1(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<const uint32_t> words);
// Reads the core's words on the device at `mesh_coord` (a device this rank drives).
std::vector<uint32_t> ReadDramCoreL1(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t num_words);
// Polls the core's L1 word at `address` on every device of the mesh this rank drives until it equals `expected`, or
// until `timeout` has passed; returns whether every device's did. Without hardware behind the devices (a mock
// cluster) nothing runs, and it returns true at once.
bool WaitForDramCoreL1Word(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t expected,
    std::chrono::milliseconds timeout);

}  // namespace tt::tt_metal::experimental
