// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal UMD-level reproduction for the device bring-up hang (issue #47866).
//
// Opens all local chips on THIS host and brings them up via tt::umd::Cluster::start_device().
// No MPI, no mesh-graph-descriptor, no tt-run, no tt-metal MetalContext — just the pure UMD
// device bring-up path that hangs in the multi-host run.
//
// On a healthy host this prints "all chips brought up OK" and exits 0. On a host with a wedged
// ASIC it hangs inside start_device; with the temporary [PSD-DEBUG] per-chip logging in UMD
// (device/cluster.cpp + device/chip/local_chip.cpp) the last "bringing up chip N" / "step k/4"
// line names the exact chip and sub-step that hung.
//
// Build:  cmake --build build --target umd_start_device_probe
// Run:    TT_METAL_LOGGER_TYPES=UMD TT_METAL_LOGGER_LEVEL=debug ./build/tools/scaleout/umd_start_device_probe

#include <cstdio>

#include <umd/device/cluster.hpp>

int main() {
    using namespace tt::umd;

    std::printf("[umd_start_device_probe] constructing Cluster (enumerating local chips) ...\n");
    std::fflush(stdout);
    Cluster cluster;

    std::printf("[umd_start_device_probe] start_device({.init_device = true}) — bringing up all local chips ...\n");
    std::fflush(stdout);
    cluster.start_device({.init_device = true});

    std::printf("[umd_start_device_probe] all chips brought up OK; closing.\n");
    std::fflush(stdout);
    cluster.close_device();
    return 0;
}
