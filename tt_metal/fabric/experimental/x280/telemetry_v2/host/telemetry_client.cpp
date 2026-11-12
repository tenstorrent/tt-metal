// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "x280_metal_transport.hpp"

#include "x280_address_map.h"

#include <tt-metalium/mesh_device.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
    }

    uint64_t expected_magic = static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC);


    return 0;    
}
