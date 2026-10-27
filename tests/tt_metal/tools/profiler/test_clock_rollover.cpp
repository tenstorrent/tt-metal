// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Blackhole-only, one device at a time. Raw artifacts must be outside any Git checkout.
// --expect-tear returns 3 (inconclusive) if the genuine producer does not tear.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "kernels/clock_rollover_layout.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace fs = std::filesystem;
namespace cr = clock_rollover;

static void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
static uint64_t reference_stamp(uint32_t hi, uint32_t lo) { return (uint64_t(hi) << 32) | lo; }
static uint64_t marker_stamp(uint32_t hi, uint32_t lo) { return reference_stamp(hi & 0xfffu, lo); }

int main(int argc, char** argv) {
    try {
        uint32_t device_id = 0, rounds = 1, cores = 8;
        bool expect_tear = false;
        fs::path output;
        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "--expect-tear") {
                expect_tear = true;
                continue;
            }
            require(i + 1 < argc, "Missing value for " + option);
            std::string value = argv[++i];
            if (option == "--device") {
                device_id = std::stoul(value);
            } else if (option == "--rounds") {
                rounds = std::stoul(value);
            } else if (option == "--cores") {
                cores = std::stoul(value);
            } else if (option == "--output-dir") {
                output = value;
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
        }
        require(rounds > 0 && rounds <= 64, "--rounds must be 1..64");
        require(cores > 0, "--cores must be positive");
        require(output.is_absolute(), "Required --output-dir must be absolute and outside Git");
        output = fs::weakly_canonical(output);
        for (auto parent = output; !parent.empty(); parent = parent.parent_path()) {
            require(!fs::exists(parent / ".git"), "Raw output must be outside Git: " + output.string());
            if (parent == parent.root_path()) {
                break;
            }
        }
        fs::create_directories(output);
        require(std::getenv("TT_METAL_DEVICE_PROFILER") != nullptr, "Set TT_METAL_DEVICE_PROFILER=1");
        auto mesh = distributed::MeshDevice::create_unit_mesh(device_id);
        auto* device = mesh->get_device(distributed::MeshCoordinate(0, 0));
        require(device->arch() == ARCH::BLACKHOLE, "This regression supports Blackhole only");
        auto grid = mesh->compute_with_storage_grid_size();
        require(cores <= grid.x * grid.y, "--cores exceeds the worker grid");
        std::vector<CoreRange> ranges;
        for (uint32_t i = 0; i < cores; ++i) {
            CoreCoord core{i % grid.x, i / grid.x};
            ranges.emplace_back(core, core);
        }
        CoreRangeSet core_set(ranges);
        // Reserve one complete segment in every L1 bank; each core uses the same local address.
        const auto banks = device->allocator()->get_num_banks(BufferType::L1);
        auto raw = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = uint64_t(cr::core_bytes) * banks},
            distributed::DeviceLocalBufferConfig{.page_size = cr::core_bytes, .buffer_type = BufferType::L1},
            mesh.get());
        Program program = CreateProgram();
        std::map<std::string, std::string> defines{{"CLOCK_ROLLOVER_ADDRESS", std::to_string(raw->address())}};
        const std::string base = "tests/tt_metal/tools/profiler/kernels/";
        CreateKernel(
            program,
            base + "clock_rollover.cpp",
            core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
        CreateKernel(
            program,
            base + "clock_rollover.cpp",
            core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defines});
        CreateKernel(program, base + "clock_rollover_compute.cpp", core_set, ComputeConfig{.defines = defines});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        uint64_t samples = 0, failures = 0, covered = 0;
        for (uint32_t round = 0; round < rounds; ++round) {
            for (uint32_t core = 0; core < cores; ++core) {
                auto path =
                    output / ("device" + std::to_string(device_id) + "_round" + std::to_string(round) + "_core" +
                              std::to_string(core % grid.x) + "_" + std::to_string(core / grid.x) + ".bin");
                require(!fs::exists(path), "Refusing to overwrite raw capture: " + path.string());
            }
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
            // Save every lane before the normal parser can assert on a torn firmware/kernel marker.
            std::vector<std::vector<uint32_t>> snapshots(cores);
            for (uint32_t core_index = 0; core_index < cores; ++core_index) {
                CoreCoord core{core_index % grid.x, core_index / grid.x};
                auto& words = snapshots[core_index];
                require(
                    detail::ReadFromDeviceL1(device, core, raw->address(), cr::core_bytes, words), "L1 read failed");
                require(words.size() == cr::core_bytes / 4, "Short raw L1 read");
                auto path = output / ("device" + std::to_string(device_id) + "_round" + std::to_string(round) +
                                      "_core" + std::to_string(core.x) + "_" + std::to_string(core.y) + ".bin");
                require(!fs::exists(path), "Refusing to overwrite raw capture: " + path.string());
                std::ofstream file(path, std::ios::binary);
                file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint32_t));
                file.close();
                require(bool(file), "Cannot save raw capture: " + path.string());
            }
            bool structurally_valid = true;
            for (uint32_t core = 0; core < cores; ++core) {
                for (uint32_t lane = 0; lane < cr::lanes; ++lane) {
                    const auto* h = snapshots[core].data() + lane * cr::lane_words;
                    bool valid = h[0] == cr::magic && h[1] == cr::version && h[2] == lane && h[3] == cr::capacity &&
                                 h[10] == 1 && h[4] > 0 && h[4] < 1000000 && h[5] == h[4] % cr::capacity &&
                                 h[7] == h[6] + 1 && h[8] < h[4];
                    uint32_t count = std::min(h[4], cr::capacity);
                    uint32_t first = h[4] - count;
                    valid &= first < h[8] && h[8] + 63 < h[4];
                    // Avoid conflating the separate 44-bit profiler timestamp wrap with LOW rollover.
                    valid &= (h[6] & 0xfffu) != 0xfffu;
                    if (!valid) {
                        structurally_valid = false;
                        std::cerr << "Uncovered/invalid boundary core=" << core << " lane=" << lane
                                  << " completion=" << h[10] << " total=" << h[4] << '\n';
                        continue;
                    }
                    bool saw_before = false, saw_after = false;
                    uint64_t previous_end = 0;
                    for (uint32_t seq = first; seq < h[4]; ++seq) {
                        uint32_t slot = seq % cr::capacity;
                        const auto* r = h + cr::header_words + slot * cr::record_words;
                        if (r[0] != seq || r[1] != slot) {
                            structurally_valid = false;
                            break;
                        }
                        saw_before |= r[2] == h[6];
                        saw_after |= r[8] == h[7];
                        uint64_t before = marker_stamp(r[2], r[3]);
                        uint64_t start = marker_stamp(r[4], r[5]);
                        uint64_t end = marker_stamp(r[6], r[7]);
                        uint64_t after = marker_stamp(r[8], r[9]);
                        bool coherent =
                            before <= start && start <= end && end <= after && (seq == first || previous_end <= start);
                        coherent &= (r[4] & 0x80000000u) && (r[6] & 0x80000000u);
                        if (!coherent) {
                            ++failures;
                            std::cerr << "Producer timestamp violation device=" << device_id << " round=" << round
                                      << " core=" << core << " lane=" << lane << " sequence=" << seq
                                      << " before=" << before << " start=" << start << " end=" << end
                                      << " after=" << after << '\n';
                        }
                        previous_end = end;
                        ++samples;
                    }
                    structurally_valid &= saw_before && saw_after;
                    if (saw_before && saw_after) {
                        ++covered;
                    }
                }
            }
            std::cout << "round=" << round << " covered_lane_rollovers=" << covered << " retained_samples=" << samples
                      << " producer_violations=" << failures << " structurally_valid=" << structurally_valid
                      << std::endl;
            // Preserve production validation. No host timestamp repair or assertion relaxation.
            ReadMeshDeviceProfilerResults(*mesh);
            require(structurally_valid, "Raw capture did not cover a valid natural LOW rollover on every lane");
        }
        require(mesh->close(), "Device close failed");
        std::cout << "covered_lane_rollovers=" << covered << " retained_samples=" << samples
                  << " producer_violations=" << failures << " injected_control=none\n";
        if (expect_tear && failures == 0) {
            std::cout << "INCONCLUSIVE: no genuine producer tear observed; increase cores/rounds\n";
            return 3;
        }
        if (expect_tear) {
            return 0;
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Clock rollover regression failed: " << error.what() << '\n';
        return 2;
    }
}
