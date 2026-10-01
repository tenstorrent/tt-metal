// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Programming example: run 100 programs with assigned IDs and attach a real-time
// profiler callback that writes each program's timing data to a CSV file.
// Similar to test_multi_op but uses the real-time profiler (D2H socket) and
// RegisterProgramRealtimeProfilerCallback to stream records to a CSV.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include <fmt/compile.h>
#include <fmt/format.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NUM_PROGRAMS = 100;
const char* const DEFAULT_CSV_PATH = "realtime_profiler_records.csv";

static std::ofstream g_csv_file;
static uint64_t g_dropped_records = 0;

static void WriteRealtimeRecordsToCsv(const tt::tt_metal::experimental::ProgramRealtimeRecordBatch& batch) {
    static thread_local fmt::memory_buffer csv_buffer;
    csv_buffer.clear();
    auto out = std::back_inserter(csv_buffer);

    for (const auto& record : batch.records) {
        uint64_t duration_cycles =
            (record.end_timestamp >= record.start_timestamp) ? (record.end_timestamp - record.start_timestamp) : 0;
        double duration_ns = (record.frequency > 0.0) ? (static_cast<double>(duration_cycles) / record.frequency) : 0.0;

        fmt::format_to(
            out,
            FMT_COMPILE("{},{},{},{},{},{:.6g},{:.6g},\""),
            record.runtime_id,
            record.chip_id,
            record.start_timestamp,
            record.end_timestamp,
            duration_cycles,
            duration_ns,
            record.frequency);

        for (size_t i = 0; i < record.kernel_sources.size(); i++) {
            const std::string_view source = record.kernel_sources[i];
            if (i > 0) {
                csv_buffer.push_back(';');
            }
            if (source.find('"') == std::string_view::npos) {
                csv_buffer.append(source);
            } else {
                for (char c : source) {
                    if (c == '"') {
                        csv_buffer.push_back('"');
                    }
                    csv_buffer.push_back(c);
                }
            }
        }
        csv_buffer.push_back('"');
        csv_buffer.push_back('\n');
    }

    g_dropped_records += batch.dropped;

    if (csv_buffer.size() != 0) {
        g_csv_file.write(csv_buffer.data(), static_cast<std::streamsize>(csv_buffer.size()));
    }
}

static void RunPrograms(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t num_programs) {
    CoreCoord compute_with_storage_size = mesh_device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores(start_core, end_core);

    for (uint32_t i = 0; i < num_programs; i++) {
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        Program program = CreateProgram();

        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op_compute.cpp",
            all_cores,
            ComputeConfig{.compile_args = std::vector<uint32_t>{}});

        program.set_runtime_id(static_cast<uint64_t>(i) + 1);

        workload.add_program(device_range, std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
}

int main(int argc, char** argv) {
    const uint32_t num_programs = (argc > 1) ? static_cast<uint32_t>(std::stoi(argv[1])) : NUM_PROGRAMS;
    const std::string csv_path = (argc > 2) ? argv[2] : DEFAULT_CSV_PATH;

    bool pass = true;

    try {
        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        g_csv_file.open(csv_path);
        if (!g_csv_file.is_open()) {
            fmt::print(stderr, "Failed to open CSV file: {}\n", csv_path);
            return 1;
        }
        g_csv_file << "runtime_id,chip_id,start_timestamp,end_timestamp,duration_cycles,duration_ns,frequency_ghz,"
                   << "kernel_sources\n";

        tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle callback_handle =
            tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback(WriteRealtimeRecordsToCsv);

        RunPrograms(mesh_device, num_programs);
        mesh_device->quiesce_devices();

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback(callback_handle);

        g_csv_file.close();

        if (g_dropped_records != 0) {
            fmt::print(
                stderr,
                "Warning: dropped {} record(s); callback could not keep up with incoming records\n",
                g_dropped_records);
        }
        fmt::print("Wrote real-time profiler records to {} ({} programs)\n", csv_path, num_programs);
        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
