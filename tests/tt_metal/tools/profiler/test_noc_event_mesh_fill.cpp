// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// tmp reproducer for issues/metal-profiler-noc-spill-reader without tt-blaze: one persistent program on every worker
// core of every chip of a mesh, BRISC + NCRISC + TRISC, with many named zones (NUM_OPS micro-ops x NEST sub-zones,
// ITERS iterations) and a steady mix of NoC traffic (DRAM page reads, writes and semaphore increments to the right
// neighbour, linked multicasts along the row), captured with the device profiler and NoC-event tracing. Then one
// strict host read at mesh close, which is where the blaze capture aborts.
//
// Knobs (env, prefix MESH_FILL_): MESH=RxC (default: a unit mesh on device 0; 4x2 and 8x4 on a galaxy), CORES=WxH
// (default: the whole compute grid), ITERS (2), OPS (125), NEST (4), READS (8), WRITES (2), SEMS (1), MCAST_EVERY (1),
// MCAST_LINKED (4), DELAY_NOPS (0), PAGE_SIZE (2048), NUM_PAGES (1024), PROGRAMS (1: launches of the same program),
// BLOCKING (1), SKIP_READ (0), TRISC (1: compute kernel with zones), WAIT_SPINS (200000), BLOAT_BSS / BLOAT_DATA (words
// added to the DM kernels' .bss / .data) and BLOAT_CODE (unrolled nops): kernel-size knobs, see the DM kernel.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {
uint32_t env_u32(const char* name, uint32_t default_value) {
    const char* value = std::getenv(name);
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : default_value;
}
std::pair<uint32_t, uint32_t> env_pair(const char* name, uint32_t a, uint32_t b) {
    const char* value = std::getenv(name);
    if (!value) {
        return {a, b};
    }
    const char* x = std::strchr(value, 'x');
    if (!x) {
        return {a, b};
    }
    return {static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), static_cast<uint32_t>(std::strtoul(x + 1, nullptr, 10))};
}
}  // namespace

int main() {
    bool pass = true;
    try {
        const auto [mesh_rows, mesh_cols] = env_pair("MESH_FILL_MESH", 0, 0);
        const auto [grid_w_req, grid_h_req] = env_pair("MESH_FILL_CORES", 0, 0);
        const uint32_t iters = env_u32("MESH_FILL_ITERS", 2);
        const uint32_t ops = env_u32("MESH_FILL_OPS", 125);
        const uint32_t nest = env_u32("MESH_FILL_NEST", 4);
        const uint32_t reads = env_u32("MESH_FILL_READS", 8);
        const uint32_t writes = env_u32("MESH_FILL_WRITES", 2);
        const uint32_t sems = env_u32("MESH_FILL_SEMS", 1);
        const uint32_t mcast_every = env_u32("MESH_FILL_MCAST_EVERY", 1);
        const uint32_t mcast_linked = env_u32("MESH_FILL_MCAST_LINKED", 4);
        const uint32_t delay_nops = env_u32("MESH_FILL_DELAY_NOPS", 0);
        const uint32_t page_size = env_u32("MESH_FILL_PAGE_SIZE", 2048);
        const uint32_t num_pages = env_u32("MESH_FILL_NUM_PAGES", 1024);
        const uint32_t programs = env_u32("MESH_FILL_PROGRAMS", 1);
        const bool blocking = env_u32("MESH_FILL_BLOCKING", 1) != 0;
        const bool skip_read = env_u32("MESH_FILL_SKIP_READ", 0) != 0;
        const bool with_trisc = env_u32("MESH_FILL_TRISC", 1) != 0;
        const uint32_t wait_spins = env_u32("MESH_FILL_WAIT_SPINS", 200000);
        const uint32_t bloat_bss = env_u32("MESH_FILL_BLOAT_BSS", 0);
        const uint32_t bloat_data = env_u32("MESH_FILL_BLOAT_DATA", 0);
        const uint32_t bloat_code = env_u32("MESH_FILL_BLOAT_CODE", 0);

        std::shared_ptr<distributed::MeshDevice> mesh_device =
            (mesh_rows && mesh_cols)
                ? distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(mesh_rows, mesh_cols)))
                : distributed::MeshDevice::create_unit_mesh(0);
        const distributed::MeshCoordinateRange device_range(mesh_device->shape());

        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t grid_w = grid_w_req ? std::min<uint32_t>(grid_w_req, grid.x) : grid.x;
        const uint32_t grid_h = grid_h_req ? std::min<uint32_t>(grid_h_req, grid.y) : grid.y;
        const CoreRange cores({0, 0}, {grid_w - 1, grid_h - 1});
        fmt::print(
            "noc_event_mesh_fill: mesh {}x{} ({} devices) cores {}x{} iters={} ops={} nest={} reads={} writes={} sems={} "
            "mcast_every={} mcast_linked={} delay_nops={} page_size={} num_pages={} programs={} blocking={} skip_read={} "
            "trisc={} bloat_bss={} bloat_data={} bloat_code={}\n",
            mesh_device->num_rows(), mesh_device->num_cols(), mesh_device->num_devices(), grid_w, grid_h, iters, ops,
            nest, reads, writes, sems, mcast_every, mcast_linked, delay_nops, page_size, num_pages, programs, blocking,
            skip_read, with_trisc, bloat_bss, bloat_data, bloat_code);

        // One DRAM page buffer per device (replicated), read by every core.
        const distributed::DeviceLocalBufferConfig local_config{.page_size = page_size, .buffer_type = BufferType::DRAM};
        const distributed::ReplicatedBufferConfig replicated_config{.size = static_cast<DeviceAddr>(page_size) * num_pages};
        auto dram_buffer = distributed::MeshBuffer::create(replicated_config, local_config, mesh_device.get());

        Program program = CreateProgram();
        constexpr uint32_t cb_id = 0;
        CircularBufferConfig cb_config =
            CircularBufferConfig(4 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
        CreateCircularBuffer(program, cores, cb_config);
        const uint32_t sem_brisc = CreateSemaphore(program, cores, 0);
        const uint32_t sem_ncrisc = CreateSemaphore(program, cores, 0);

        std::map<std::string, std::string> defines = {
            {"NUM_OPS", std::to_string(std::min<uint32_t>(ops, 128))},
            {"NEST", std::to_string(nest)},
            {"READS", std::to_string(reads)},
            {"WRITES", std::to_string(writes)},
            {"SEMS", std::to_string(sems)},
            {"MCAST_EVERY", std::to_string(mcast_every)},
            {"MCAST_LINKED", std::to_string(mcast_linked)},
            {"DELAY_NOPS", std::to_string(delay_nops)},
            {"ITERS", std::to_string(iters)},
            {"WAIT_SPINS", std::to_string(wait_spins)},
            {"BLOAT_BSS_WORDS", std::to_string(bloat_bss)},
            {"BLOAT_DATA_WORDS", std::to_string(bloat_data)},
            {"BLOAT_CODE", std::to_string(bloat_code)}};
        std::vector<uint32_t> compile_args;
        TensorAccessorArgs(*dram_buffer->get_reference_buffer()).append_to(compile_args);

        const std::string dm_kernel = "tests/tt_metal/tools/profiler/kernels/noc_event_mesh_fill_dm.cpp";
        KernelHandle brisc = CreateKernel(
            program, dm_kernel, cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args,
                .defines = defines});
        KernelHandle ncrisc = CreateKernel(
            program, dm_kernel, cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = compile_args,
                .defines = defines});
        if (with_trisc) {
            CreateKernel(
                program, "tests/tt_metal/tools/profiler/kernels/noc_event_mesh_fill_compute.cpp", cores,
                ComputeConfig{.defines = defines});
        }

        // Per-core runtime args: right neighbour (NoC coords), multicast row (column 1 .. W-1), semaphore id.
        for (uint32_t y = 0; y < grid_h; y++) {
            const CoreCoord row_first = mesh_device->worker_core_from_logical_core({1, y});
            const CoreCoord row_last = mesh_device->worker_core_from_logical_core({grid_w - 1, y});
            for (uint32_t x = 0; x < grid_w; x++) {
                const CoreCoord logical{x, y};
                const CoreCoord self = mesh_device->worker_core_from_logical_core(logical);
                const bool has_right = x + 1 < grid_w;
                const CoreCoord right = has_right ? mesh_device->worker_core_from_logical_core({x + 1, y}) : self;
                const bool is_sender = (x == 0) && grid_w > 1;
                std::vector<uint32_t> args = {
                    page_size,
                    num_pages,
                    static_cast<uint32_t>(dram_buffer->address()),
                    y * grid_w + x,
                    has_right ? 1u : 0u,
                    static_cast<uint32_t>(right.x),
                    static_cast<uint32_t>(right.y),
                    x > 0 ? 1u : 0u,
                    sem_brisc,
                    is_sender ? 1u : 0u,
                    static_cast<uint32_t>(row_first.x),
                    static_cast<uint32_t>(row_last.x),
                    static_cast<uint32_t>(self.y),
                    grid_w - 1};
                SetRuntimeArgs(program, brisc, logical, args);
                args[8] = sem_ncrisc;
                SetRuntimeArgs(program, ncrisc, logical, args);
            }
        }

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        for (uint32_t p = 0; p < programs; p++) {
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, blocking || p + 1 == programs);
        }
        if (!skip_read) {
            ReadMeshDeviceProfilerResults(*mesh_device);
        }
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }
    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }
    return 0;
}
