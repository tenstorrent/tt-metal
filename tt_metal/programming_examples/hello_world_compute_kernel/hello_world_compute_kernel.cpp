// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/ez.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

// A bit of a hack to handle packaged examples but also work inside the Metalium git repo.
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

    // Create a single-device context and run a trivial compute kernel on core {0, 0}.
    // Within a Tensix, 3 RISC-V cores run collaboratively to perform compute tasks. These are the UNPACK, MATH,
    // and PACK cores.
    // Please view the respective documentation for more information:
    // https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md#tenstorrent-architecture-overview
    DeviceContext ctx(0);

    auto program = ProgramBuilder(CoreCoord{0, 0})
                       .compute(OVERRIDE_KERNEL_PREFIX "hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp")
                       .build();

    fmt::print("Hello, Core (0, 0) on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");
    ctx.run(std::move(program));

    // The kernel will print the following messages (with TT_METAL_DPRINT_CORES=0,0):
    // 0:(x=0,y=0):TR0: Hello, I am the UNPACK core running the compute kernel
    // 0:(x=0,y=0):TR1: Hello, I am the MATH core running the compute kernel
    // 0:(x=0,y=0):TR2: Hello, I am the PACK core running the compute kernel
    //
    // Deconstructing the output:
    // 0: - Device ID
    // (x=0,y=0): - Tensix core coordinates (so the 3 compute cores are all on the same core {0, 0})
    // TR0: Compute core 0 (UNPACK)
    // TR1: Compute core 1 (MATH)
    // TR2: Compute core 2 (PACK)
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    return 0;
}
