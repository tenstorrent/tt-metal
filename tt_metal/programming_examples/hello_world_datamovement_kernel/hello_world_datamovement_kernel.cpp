// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/ez.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // Create a single-device context and run two data movement kernels on core {0, 0}.
    // There are 2 Data Movement cores per Tensix. In applications one is usually used for reading data from DRAM and
    // the other for writing data. However for demonstration purposes, we will create 2 Data Movement kernels that
    // simply print a message to show them running at the same time.
    DeviceContext ctx(0);

    auto program =
        ProgramBuilder(CoreCoord{0, 0})
            .reader(OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp")
            .done()
            .writer(OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp")
            .done()
            .build();

    fmt::print("Hello, Core (0, 0) on Device 0, Please start execution. I will standby for your communication.\n");
    ctx.run(std::move(program));

    // The program should print the following (with TT_METAL_DPRINT_CORES=0,0):
    // NC and BR are Data Movement core 1 and 0 respectively.
    //
    // 0:(x=0,y=0):NC: My logical coordinates are 0,0
    // 0:(x=0,y=0):NC: Hello, host, I am running a void data movement kernel on Data Movement core 1.
    // 0:(x=0,y=0):BR: My logical coordinates are 0,0
    // 0:(x=0,y=0):BR: Hello, host, I am running a void data movement kernel on Data Movement core 0.
    //
    // Deconstructing the output:
    // 0: - Device ID
    // (x=0,y=0): - Tensix core coordinates (so both Data Movement cores are on the same Tensix core)
    // NC: - Data Movement core 1
    // BR: - Data Movement core 0

    fmt::print("Thank you, Core (0, 0) on Device 0, for the completed task.\n");
    return 0;
}
