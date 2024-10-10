// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include "assert.hpp"
#include "common/core_coord.h"
#include "impl/kernels/data_types.hpp"
#include "llrt/tt_cluster.hpp"
#include "logger.hpp"
#include "device/tt_cluster_descriptor_types.h"
#include "test_common.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/tt_metal/unit_tests_common/common/test_utils.hpp"
#include "tt_soc_descriptor.h"

using namespace tt;
using namespace tt_metal;

constexpr uint32_t DEFAULT_NUM_KERNELS = 1;
constexpr uint32_t DEFAULT_MIN_KERNEL_SIZE_BYTES = 17;
constexpr uint32_t DEFAULT_MAX_KERNEL_SIZE_BYTES = 4096;
constexpr uint32_t DEFAULT_MIN_KERNEL_RUNTIME_CYCLES = 0;
constexpr uint32_t DEFAULT_MAX_KERNEL_RUNTIME_CYCLES = 100000;
constexpr bool DEFAULT_ONLY_DISPATCH_TO_TENSIX_CORES = false;
constexpr bool DEFAULT_ONLY_DISPATCH_TO_ACTIVE_ETH_CORES = false;

uint32_t num_kernels_g;
uint32_t min_kernel_size_bytes_g;
uint32_t max_kernel_size_bytes_g;
uint32_t min_kernel_runtime_cycles_g;
uint32_t max_kernel_runtime_cycles_g;
bool only_dispatch_to_tensix_cores_g;
bool only_dispatch_to_active_eth_cores_g;

const string kernel_file_path = "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/dispatcher_kernel_size_and_runtime.cpp";

void init(int argc, char **argv) {
    const std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
            log_info(LogTest, "Usage:");
            log_info(LogTest, "  -n: Number of kernels (default: {})", DEFAULT_NUM_KERNELS);
            log_info(LogTest, "  -smin: Minimum kernel size in bytes (default: {})", DEFAULT_MIN_KERNEL_SIZE_BYTES);
            log_info(LogTest, "  -smax: Maximum kernel size in bytes (default: {})", DEFAULT_MAX_KERNEL_SIZE_BYTES);
            log_info(LogTest, "  -rmin: Minimum kernel runtime in cycles (default: {})", DEFAULT_MIN_KERNEL_RUNTIME_CYCLES);
            log_info(LogTest, "  -rmax: Maximum kernel runtime in cycles (default: {})", DEFAULT_MAX_KERNEL_RUNTIME_CYCLES);
            log_info(LogTest, "  -t: Only dispatch to Tensix cores (default: {})", DEFAULT_ONLY_DISPATCH_TO_TENSIX_CORES);
            log_info(LogTest, "  -e: Only dispatch to active ethernet cores (default: {})", DEFAULT_ONLY_DISPATCH_TO_ACTIVE_ETH_CORES);
            exit(0);
    }

    num_kernels_g = test_args::get_command_option_uint32(input_args, "-n", DEFAULT_NUM_KERNELS);
    min_kernel_size_bytes_g = test_args::get_command_option_uint32(input_args, "-smin", DEFAULT_MIN_KERNEL_SIZE_BYTES);
    max_kernel_size_bytes_g = test_args::get_command_option_uint32(input_args, "-smax", DEFAULT_MAX_KERNEL_SIZE_BYTES);
    min_kernel_runtime_cycles_g = test_args::get_command_option_uint32(input_args, "-rmin", DEFAULT_MIN_KERNEL_RUNTIME_CYCLES);
    max_kernel_runtime_cycles_g = test_args::get_command_option_uint32(input_args, "-rmax", DEFAULT_MAX_KERNEL_RUNTIME_CYCLES);
    only_dispatch_to_tensix_cores_g = test_args::has_command_option(input_args, "-t");
    only_dispatch_to_active_eth_cores_g = test_args::has_command_option(input_args, "-e");

    // if (num_kernels_g < DEFAULT_NUM_KERNELS) {
    //     log_fatal("Number of kernels must be >= {}", DEFAULT_NUM_KERNELS);
    //     exit(0);
    // }
    if (min_kernel_size_bytes_g < DEFAULT_MIN_KERNEL_SIZE_BYTES) {
        log_fatal("Minimum kernel size must be >= {} bytes", DEFAULT_MIN_KERNEL_SIZE_BYTES);
        exit(0);
    }
    if (max_kernel_size_bytes_g > DEFAULT_MAX_KERNEL_SIZE_BYTES) {
        log_fatal("Maximum kernel size must be <= {} bytes", DEFAULT_MAX_KERNEL_SIZE_BYTES);
        exit(0);
    }
    if (min_kernel_runtime_cycles_g < DEFAULT_MIN_KERNEL_RUNTIME_CYCLES) {
        log_fatal("Minimum kernel runtime must be >= {} cycles", DEFAULT_MIN_KERNEL_RUNTIME_CYCLES);
        exit(0);
    }
    if (max_kernel_runtime_cycles_g > DEFAULT_MAX_KERNEL_RUNTIME_CYCLES) {
        log_fatal("Maximum kernel size must be <= {} cycles", DEFAULT_MAX_KERNEL_RUNTIME_CYCLES);
        exit(0);
    }
    if (only_dispatch_to_tensix_cores_g && only_dispatch_to_active_eth_cores_g) {
        log_fatal("Flags {-t, -e} are mutually exclusive");
        exit(0);
    }
}

uint32_t generate_random_num(const uint32_t min, const uint32_t max) {
    return min + rand() % (max - min + 1);
}

CoreRangeSet get_kernel_core_range_set(const Device* device, const CoreType& core_type) {
    std::set<CoreRange> cores;
    if (core_type == CoreType::WORKER) {
        CoreCoord start_core(0, 0);
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        CoreCoord end_core(worker_grid_size.x - 1, worker_grid_size.y - 1);
        cores.emplace(start_core, end_core);
    }
    else {
        TT_FATAL(core_type == CoreType::ETH, "Unsupported core type");
        for (CoreCoord core : device->get_active_ethernet_cores()) {
            cores.emplace(core);
        }
    }
    CoreRangeSet crs(cores);
    return crs;
}

KernelHandle initialize_kernel(Program& program, const Device* device) {
    const uint32_t kernel_size_bytes = generate_random_num(min_kernel_size_bytes_g, max_kernel_size_bytes_g);
    const uint32_t kernel_runtime_cycles = generate_random_num(min_kernel_runtime_cycles_g, max_kernel_runtime_cycles_g);
    const std::map<string, string> defines = {
        {"KERNEL_SIZE_BYTES", std::to_string(kernel_size_bytes)},
        {"KERNEL_RUNTIME_SECONDS", std::to_string(kernel_runtime_cycles)}
    };

    log_info(LogTest, "Size: {}", kernel_size_bytes);
    log_info(LogTest, "Runtime: {}", kernel_runtime_cycles);

    KernelHandle kernel_id;
    if (only_dispatch_to_tensix_cores_g) {
        CoreRangeSet cores = get_kernel_core_range_set(device, CoreType::WORKER);
        // CoreCoord start_core(0, 0);
        // CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        // CoreCoord end_core(worker_grid_size.x - 1, worker_grid_size.y - 1);
        // CoreRange cores(start_core, end_core);
        log_info(LogTest, "Cores: {}", cores.str());
        kernel_id = CreateKernel(program, kernel_file_path, cores, DataMovementConfig{.defines = defines});
    } else if (only_dispatch_to_active_eth_cores_g) {
        CoreRangeSet cores = get_kernel_core_range_set(device, CoreType::ETH);
        kernel_id = CreateKernel(program, kernel_file_path, cores, EthernetConfig{.defines = defines});
    } else {
        if (rand() % 2 == 0) {
            CoreRangeSet cores = get_kernel_core_range_set(device, CoreType::WORKER);
            kernel_id = CreateKernel(program, kernel_file_path, cores, DataMovementConfig{.defines = defines});
        } else {
            CoreRangeSet cores = get_kernel_core_range_set(device, CoreType::ETH);
            kernel_id = CreateKernel(program, kernel_file_path, cores, EthernetConfig{.defines = defines});
        }
    }

    return kernel_id;
}

Program initialize_program(const Device* device) {
    Program program = CreateProgram();
    for (uint32_t i = 0; i < num_kernels_g; i++) {
        initialize_kernel(program, device);
    }
    return program;
}

int main(int argc, char **argv) {
    init(argc, argv);
    srand(time(nullptr));
    const bool use_slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != NULL;

    bool pass = true;
    try {
        const chip_id_t device_id = 0;
        Device *device = CreateDevice(device_id);
        if (only_dispatch_to_active_eth_cores_g && device->get_active_ethernet_cores().empty()) {
            log_fatal("Device {} does not have any active ethernet cores", device_id);
            CloseDevice(device);
            exit(0);
        }

        CommandQueue& cq = device->command_queue();

        Program program = initialize_program(device);
        RunProgram(device, program, use_slow_dispatch);

        pass &= CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    int return_code;
    if (pass) {
        log_info(LogTest, "Test Passed");
        return_code = 0;
    } else {
        log_fatal(LogTest, "Test Failed");
        return_code = 1;
    }
    return return_code;
}
