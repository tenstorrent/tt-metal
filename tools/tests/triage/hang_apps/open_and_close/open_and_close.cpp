// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Process B in the SafeDeviceGuard race scenario:
// Opens device 0, waits briefly, then closes cleanly.
// Intended to be run concurrently with triage_hang_app_add_2_integers_hang.

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    printf("open_and_close: acquiring device\n");
    fflush(stdout);

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    printf("open_and_close: device acquired, closing\n");
    fflush(stdout);

    mesh_device->close();

    printf("open_and_close: done\n");
    fflush(stdout);

    return 0;
}
