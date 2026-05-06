// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Opens device 0 and sleeps until killed — for SIGKILL recovery test.

#include <unistd.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    printf("hold_device: acquiring device\n");
    fflush(stdout);

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    printf("hold_device: device acquired, sleeping until killed\n");
    fflush(stdout);

    while (true) {
        sleep(1);
    }
}
