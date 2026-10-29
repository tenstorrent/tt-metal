// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Device profiler with NoC-event tracing (TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1): one
// worker core, BRISC fills its 512-word L1 profiler ring to exactly argv[1] words (510 or 512, default 512) with
// 2-word event records and then issues one DRAM page read; NCRISC issues one read. The NoC event's quick_push then
// writes its START/END pair at the ring index without a room check, so the pair overruns BRISC's ring into the NCRISC
// ring that follows it in L1 and overwrites the NCRISC chunk header (see the kernel).
//
// Expected failure modes: 510 -> ReadMeshDeviceProfilerResults aborts (BRISC START without END); 512 -> the NCRISC
// header carries the pair instead of the (0, 0) run-start sentinel, the host drops the whole NCRISC lane, and
// profile_log_device.csv has no NCRISC rows at all (the test fails on that). Run from the repo root.
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
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

constexpr uint32_t PAGE_SIZE = 2048;
constexpr uint32_t RING_WORDS = 512;     // PROFILER_L1_VECTOR_SIZE
constexpr uint32_t RING_FIRST_USER = 12;  // CUSTOM_MARKERS

std::string device_csv_path() {
    const char* profiler_dir = std::getenv("TT_METAL_PROFILER_DIR");
    if (profiler_dir) {
        return std::string(profiler_dir) + "/.logs/profile_log_device.csv";
    }
    const char* metal_home = std::getenv("TT_METAL_HOME");
    const std::string prefix = metal_home ? std::string(metal_home) + "/" : std::string();
    return prefix + "generated/profiler/.logs/profile_log_device.csv";
}

// Per-RISC row counts of the device CSV and its rows with timer id 0 (a chunk header read as a marker).
// Columns: PCIe slot, core_x, core_y, RISC processor type, timer_id, ...
void scan_device_csv(const std::string& path, std::map<std::string, uint32_t>& rows_per_risc, std::vector<std::string>& id0) {
    std::ifstream csv(path);
    if (!csv) {
        throw std::runtime_error("device profiler CSV not found: " + path);
    }
    std::string line;
    std::getline(csv, line);  // ARCH / CHIP_FREQ line
    std::getline(csv, line);  // column names
    while (std::getline(csv, line)) {
        std::stringstream fields(line);
        std::string risc;
        std::string timer_id;
        std::string field;
        for (int i = 0; i < 5 && std::getline(fields, field, ','); i++) {
            if (i == 3) {
                risc = field;
            } else if (i == 4) {
                timer_id = field;
            }
        }
        rows_per_risc[risc]++;
        if (timer_id == "0") {
            id0.push_back(line);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t ring_index = argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : RING_WORDS;
    if (ring_index < RING_FIRST_USER || ring_index > RING_WORDS || ring_index % 2) {
        fmt::print(stderr, "usage: {} [brisc ring index, even, {}..{}]\n", argv[0], RING_FIRST_USER, RING_WORDS);
        return 1;
    }
    const uint32_t brisc_events = (ring_index - RING_FIRST_USER) / 2;
    fmt::print("noc_event_ring_overrun: BRISC ring index {} ({} event records), then one NoC read\n", ring_index, brisc_events);

    bool pass = true;
    try {
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        const distributed::MeshCoordinateRange device_range(mesh_device->shape());
        const CoreCoord core{0, 0};

        const distributed::DeviceLocalBufferConfig local_config{.page_size = PAGE_SIZE, .buffer_type = BufferType::DRAM};
        const distributed::ReplicatedBufferConfig replicated_config{.size = PAGE_SIZE};
        auto dram_buffer = distributed::MeshBuffer::create(replicated_config, local_config, mesh_device.get());

        Program program = CreateProgram();
        CircularBufferConfig cb_config =
            CircularBufferConfig(PAGE_SIZE, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, PAGE_SIZE);
        CreateCircularBuffer(program, core, cb_config);

        const std::map<std::string, std::string> defines = {{"BRISC_EVENTS", std::to_string(brisc_events)}};
        std::vector<uint32_t> compile_args;
        TensorAccessorArgs(*dram_buffer->get_reference_buffer()).append_to(compile_args);
        const std::vector<uint32_t> runtime_args = {PAGE_SIZE, static_cast<uint32_t>(dram_buffer->address())};

        const std::string kernel = "tests/tt_metal/tools/profiler/kernels/noc_event_ring_overrun.cpp";
        KernelHandle brisc = CreateKernel(
            program,
            kernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});
        KernelHandle ncrisc = CreateKernel(
            program,
            kernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = compile_args,
                .defines = defines});
        SetRuntimeArgs(program, brisc, core, runtime_args);
        SetRuntimeArgs(program, ncrisc, core, runtime_args);

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

        ReadMeshDeviceProfilerResults(*mesh_device);
        pass &= mesh_device->close();

        const std::string csv = device_csv_path();
        std::map<std::string, uint32_t> rows_per_risc;
        std::vector<std::string> id0;
        scan_device_csv(csv, rows_per_risc, id0);
        fmt::print("device CSV {}: BRISC rows {}, NCRISC rows {}\n", csv, rows_per_risc["BRISC"], rows_per_risc["NCRISC"]);
        if (rows_per_risc["NCRISC"] == 0) {
            pass = false;
            fmt::print(stderr, "no NCRISC rows in the device CSV: the NCRISC lane was dropped (its chunk header lost the run-start sentinel)\n");
        }
        if (!id0.empty()) {
            pass = false;
            fmt::print(stderr, "{} marker(s) with timer id 0 (a chunk header read as a marker), first: {}\n", id0.size(), id0.front());
        }
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
