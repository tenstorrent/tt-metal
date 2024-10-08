// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "common/core_coord.h"
#include "common/logger.hpp"
#include "device/tt_cluster_descriptor_types.h"
#include "dispatch/command_queue.hpp"
#include "kernels/kernel.hpp"
#include "kernels/kernel_types.hpp"
#include "program/program.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt_metal;

constexpr uint32_t DEFAULT_NUM_KERNELS = 1;
constexpr uint32_t DEFAULT_MIN_KERNEL_SIZE_BYTES = 17;
constexpr uint32_t DEFAULT_MAX_KERNEL_SIZE_BYTES = 2097152;
constexpr uint32_t DEFAULT_MIN_KERNEL_RUNTIME_SECONDS = 0;
constexpr uint32_t DEFAULT_MAX_KERNEL_RUNTIME_SECONDS = 60;

uint32_t num_kernels_g;
uint32_t min_kernel_size_bytes_g;
uint32_t max_kernel_size_bytes_g;
uint32_t min_kernel_runtime_seconds_g;
uint32_t max_kernel_runtime_seconds_g;

const string kernel_file_path = "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/dispatcher_kernel_size_and_runtime.cpp";

void init(int argc, char **argv) {
    const std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
            log_info(LogTest, "Usage:");
            log_info(LogTest, "  -n: Number of kernels (default: {})", DEFAULT_NUM_KERNELS);
            log_info(LogTest, "  -smin: Minimum kernel size in bytes (default: {})", DEFAULT_MIN_KERNEL_SIZE_BYTES);
            log_info(LogTest, "  -smax: Maximum kernel size in bytes (default: {})", DEFAULT_MAX_KERNEL_SIZE_BYTES);
            log_info(LogTest, "  -rmin: Minimum kernel runtime in seconds (default: {})", DEFAULT_MIN_KERNEL_RUNTIME_SECONDS);
            log_info(LogTest, "  -rmax: Maximum kernel runtime in seconds (default: {})", DEFAULT_MAX_KERNEL_RUNTIME_SECONDS);
            // log_info(LogTest, "  -t: Dispatch to Tensix cores");
            // log_info(LogTest, "  -e: Dispatch to ethernet cores");
            exit(0);
    }

    num_kernels_g = test_args::get_command_option_uint32(input_args, "-n", DEFAULT_NUM_KERNELS);
}

uint32_t generate_random_num(const uint32_t min, const uint32_t max) {
    return min + rand() % (max - min + 1);
}

KernelHandle initialize_kernel(Program& program) {
    const uint32_t kernel_size_bytes = generate_random_num(min_kernel_size_bytes_g, max_kernel_size_bytes_g);
    const uint32_t kernel_runtime_seconds = generate_random_num(min_kernel_runtime_seconds_g, max_kernel_runtime_seconds_g);
    const std::map<string, string> defines = {
        {"KERNEL_SIZE_BYTES", std::to_string(kernel_size_bytes)},
        {"KERNEL_RUNTIME_SECONDS", std::to_string(kernel_runtime_seconds)}
    };
    KernelHandle kernel_id = CreateKernel(program, kernel_file_path, CoreCoord(0, 0), const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config);
    return kernel_id;
}

Program initialize_program() {
    Program program = CreateProgram();
    for (uint32_t i = 0; i < num_kernels_g; i++) {
        initialize_kernel(program);
    }
    return program;
}

int main(int argc, char **argv) {
    init(argc, argv);

    bool pass = true;
    try {
        const chip_id_t device_id = 0;
        Device *device = CreateDevice(device_id);
        CommandQueue& cq = device->command_queue();

        Program program = initialize_program();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        // RunProgram(cq, program, false);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    int return_code;
    if (pass) {
        log_info(LogTest, "Test Passed");
        return_code = 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return_code = 1;
    }
    return return_code;
}
