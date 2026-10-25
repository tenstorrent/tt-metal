// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Power microbenchmark of compute threads that wait (test_layer_energy.py::test_wait_power).
//
// Each mode runs for `cycles` wall-clock cycles on all three TRISCs, then returns:
//   1 = spin on the wall-clock register,
//   2 = the unpack thread polls circular buffer 0's tiles-received counter as cb_wait_front does (the CB is never
//       pushed); the math and pack threads spin on the wall clock.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"

namespace {
inline uint32_t wall_clock() { return static_cast<uint32_t>(ckernel::read_wall_clock()); }

template <uint32_t cycles>
inline void spin(uint32_t start) {
    while (wall_clock() - start < cycles) {
    }
}

#ifdef TRISC_UNPACK
template <uint32_t cycles>
inline void poll_cb(uint32_t start) {
    volatile tt_l1_ptr std::uint32_t* tiles_received = get_cb_tiles_received_ptr(0);
    const uint16_t acked = get_local_cb_interface(0).tiles_acked;
    do {
        for (uint32_t i = 0; i < 16; i++) {
            const uint16_t received = static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(tiles_received)));
            if (static_cast<uint16_t>(received - acked) >= 1) {
                return;
            }
        }
    } while (wall_clock() - start < cycles);
}
#endif
}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t cycles = get_compile_time_arg_val(1);
    const uint32_t start = wall_clock();
    if constexpr (mode == 1) {
        spin<cycles>(start);
    } else if constexpr (mode == 2) {
        UNPACK((poll_cb<cycles>(start)));
        MATH((spin<cycles>(start)));
        PACK((spin<cycles>(start)));
    }
}
