// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Device profiler with NoC-event tracing (TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1): one
// worker core, BRISC fills its 512-word L1 profiler ring to exactly argv[1] words (510 or 512, default 512) with
// 2-word event records, with argv[2] == "zone" opens and closes an empty zone, and then issues one DRAM page read;
// NCRISC issues one read. The NoC event's quick_push then pushes BRISC's ring (see the kernel).
//
// Guards against profiler markers written past the end of the ring, into the NCRISC ring that follows it in L1: the
// quick-push START/END pair when the ring stands at 510 or 512, and a zone END when the zone was opened at 510. Such
// an overrun overwrites the NCRISC chunk header, which the host then drops (no NCRISC rows in profile_log_device.csv),
// parses as a zone marker with timer id 0 when the lane already holds earlier chunks, or aborts on at mesh close
// (BRISC START without END). The test fails on a missing NCRISC lane or on a zone marker with timer id 0. Run from
// the repo root.
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

// Per-RISC row counts of the device CSV and its zone-marker rows with timer id 0 (a chunk header read as a marker:
// the id words of the header decode as ZONE_START/ZONE_END with id 0; a TS_EVENT with id 0 would be a legitimate
// DeviceRecordEvent(0)).
// Columns: PCIe slot, core_x, core_y, RISC processor type, timer_id, time, data, run host ID, trace id, trace id
// counter, zone name, type, ...
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
        std::string type;
        std::string field;
        for (int i = 0; i < 12 && std::getline(fields, field, ','); i++) {
            if (i == 3) {
                risc = field;
            } else if (i == 4) {
                timer_id = field;
            } else if (i == 11) {
                type = field;
            }
        }
        rows_per_risc[risc]++;
        if (timer_id == "0" && (type == "ZONE_START" || type == "ZONE_END")) {
            id0.push_back(line);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t ring_index = argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : RING_WORDS;
    const bool tail_zone = argc > 2 && std::string(argv[2]) == "zone";
    if (ring_index < RING_FIRST_USER || ring_index > RING_WORDS || ring_index % 2 || (argc > 2 && !tail_zone)) {
        fmt::print(
            stderr, "usage: {} [brisc ring index, even, {}..{}] [zone]\n", argv[0], RING_FIRST_USER, RING_WORDS);
        return 1;
    }
    const uint32_t brisc_events = (ring_index - RING_FIRST_USER) / 2;
    fmt::print(
        "noc_event_ring_overrun: BRISC ring index {} ({} event records), {}then one NoC read\n",
        ring_index,
        brisc_events,
        tail_zone ? "an empty zone, " : "");

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

        std::map<std::string, std::string> defines = {{"BRISC_EVENTS", std::to_string(brisc_events)}};
        if (tail_zone) {
            defines["BRISC_TAIL_ZONE"] = "1";
        }
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
            fmt::print(
                stderr,
                "{} zone marker(s) with timer id 0 (a chunk header read as a marker), first: {}\n",
                id0.size(),
                id0.front());
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
