// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel of test_noc_event_mesh_fill: the same zone structure as the data-movement kernel (ITERS x NUM_OPS
// micro-op zones with NEST sub-zones), no NoC traffic, DELAY_NOPS nops per sub-zone so TRISC zones overlap the
// data-movement ones in time.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/debug/kernel_profiler.hpp"

FORCE_INLINE void micro_op() {
    for (uint32_t k = 0; k < NEST; k++) {
        DeviceZoneScopedN("SUBOP");
        for (uint32_t d = 0; d < DELAY_NOPS + 64; d++) {
            asm volatile("nop");
        }
    }
}
#define OP(n)                                      \
    if constexpr ((n) < NUM_OPS) {                 \
        DeviceZoneScopedN("MICROOP_" #n);          \
        micro_op();                                \
    }

void kernel_main() {
    for (uint32_t it = 0; it < ITERS; it++) {
        DeviceZoneScopedN("ITERATION");
        OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7) OP(8) OP(9) OP(10) OP(11) OP(12) OP(13) OP(14) OP(15)
        OP(16) OP(17) OP(18) OP(19) OP(20) OP(21) OP(22) OP(23) OP(24) OP(25) OP(26) OP(27) OP(28) OP(29) OP(30) OP(31)
        OP(32) OP(33) OP(34) OP(35) OP(36) OP(37) OP(38) OP(39) OP(40) OP(41) OP(42) OP(43) OP(44) OP(45) OP(46) OP(47)
        OP(48) OP(49) OP(50) OP(51) OP(52) OP(53) OP(54) OP(55) OP(56) OP(57) OP(58) OP(59) OP(60) OP(61) OP(62) OP(63)
        OP(64) OP(65) OP(66) OP(67) OP(68) OP(69) OP(70) OP(71) OP(72) OP(73) OP(74) OP(75) OP(76) OP(77) OP(78) OP(79)
        OP(80) OP(81) OP(82) OP(83) OP(84) OP(85) OP(86) OP(87) OP(88) OP(89) OP(90) OP(91) OP(92) OP(93) OP(94) OP(95)
        OP(96) OP(97) OP(98) OP(99) OP(100) OP(101) OP(102) OP(103) OP(104) OP(105) OP(106) OP(107) OP(108) OP(109) OP(110) OP(111)
        OP(112) OP(113) OP(114) OP(115) OP(116) OP(117) OP(118) OP(119) OP(120) OP(121) OP(122) OP(123) OP(124) OP(125) OP(126) OP(127)
    }
}
