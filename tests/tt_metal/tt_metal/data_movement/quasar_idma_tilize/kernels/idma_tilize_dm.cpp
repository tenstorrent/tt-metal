// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar iDMA tilize, fp32: rewrites a row-major tensor in L1 into tilized face order.
//
// A 32x32 tile is four 16x16 faces, each row-major. A face row is 16 contiguous datums in both
// layouts, so the tilize is one iDMA copy per face row: 64 copies of 64 B per tile.
//
// Address generator 0 walks one tile column of the source, row pitch P:
//
//   for (base = col; ; base += 16P)                 // face pair: top faces, then bottom faces
//     for (outer = 0; outer < 128; outer += 64)     // left face, then right face
//       for (inner = 0; inner < 16P; inner += P)    // 16 rows of the face
//         yield base + outer + inner
//
// and the destination one tile at a time, a whole tile row apart. One push + one issue per copy.
//
// Timing record at result_addr, 32-bit words:
//   [0] wall:     NEO0 wall clock (the TRISC timeline), cmdbuf setup through the last ack
//   [1] dm_setup: DM cycles, cmdbuf setup
//   [2] dm_issue: DM cycles, addrgen setup + every push + issue
//   [3] dm_drain: DM cycles, ack wait after the last issue
//
// Measured on emu-quasar-1x3, 4 x 8 tiles (32 tiles, 2048 copies), output bit-exact:
//
//   wall clock   12796 cycles | 399.9 per tile | 6.25 per copy | 10.2 B/cycle
//   DM cycles    setup 52 | issue 12622 | drain 36
//
// The cost is issue: the DM core starts one 64 B copy about every 6 cycles (6.16 DM cycles per
// push + issue), whatever the VC count. One tile is 16 face rows x 4 faces = 64 copies, so
// 6 x 64 = ~384 cycles, against the ~400 measured. The copies themselves move only ~10 B/cycle.
//
// To run (host test: ../test_idma_tilize.cpp, gtest QuasarIdmaTilize.Fp32), from the repo root:
//
//   cmake -B build_Release -DTT_METAL_BUILD_TESTS=ON
//   cmake --build build_Release --target unit_tests_data_movement
//
//   export TT_METAL_HOME=$PWD ARCH_NAME=quasar TT_METAL_SLOW_DISPATCH_MODE=1
//   export TT_METAL_SIMULATOR=<emulator build, e.g. .../tt-umd-simulators/build/emu-quasar-1x3>
//   export TT_UMD_SIMULATOR_PATH=$TT_METAL_SIMULATOR
//   export NNG_SOCKET_ADDR=<emulator host, e.g. tcp://tensix-l-04:49879> NNG_SOCKET_LOCAL_PORT=5555
//   ./build_Release/test/tt_metal/unit_tests_data_movement --gtest_filter="QuasarIdmaTilize.Fp32"
//
// The kernel is JIT-compiled at run time, so kernel edits need no rebuild. The results print
// as one "iDMA tilize, fp32, ..." log line block.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/noc_nonblocking_api.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "tensix_neo_reg.h"

using namespace overlay;

namespace {

constexpr std::uint32_t DATUM_BYTES = 4;
constexpr std::uint32_t FACE_DIM = 16;
constexpr std::uint32_t TILE_DIM = 32;
constexpr std::uint32_t FACE_ROW_BYTES = FACE_DIM * DATUM_BYTES;
constexpr std::uint32_t TILE_BYTES = TILE_DIM * TILE_DIM * DATUM_BYTES;
constexpr std::uint32_t COPIES_PER_TILE = 4 * FACE_DIM;

// The clock the TRISCs stamp on. Reading WALL_CLOCK_0 latches the high word.
inline std::uint64_t wall_clock() {
    const std::uint32_t lo =
        *reinterpret_cast<volatile std::uint32_t*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_0_REG_ADDR);
    const std::uint32_t hi =
        *reinterpret_cast<volatile std::uint32_t*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_1_AT_REG_ADDR);
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
}

// This DM hart's cycle count, a different clock domain from the wall clock.
inline std::uint64_t dm_cycles() {
    std::uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

}  // namespace

void kernel_main() {
    const std::uint32_t src_addr = get_arg(args::src_addr);  // row-major, L1
    const std::uint32_t dst_addr = get_arg(args::dst_addr);  // tilized, L1
    const std::uint32_t rt_dim = get_arg(args::rt_dim);
    const std::uint32_t ct_dim = get_arg(args::ct_dim);
    const std::uint32_t result_addr = get_arg(args::result_addr);

    const std::uint32_t row_pitch = ct_dim * TILE_DIM * DATUM_BYTES;

    const std::uint64_t wall_start = wall_clock();
    const std::uint64_t t0 = dm_cycles();

    // iDMA copy of one face row per issue, request VCs round-robined over every iDMA VC.
    // MAX_BYTES_IN_PACKET must be set after the reset or the VC fan-out is inert.
    reset_cmdbuf_0();
    idma_setup_as_copy_cmdbuf_0(/*wrapping=*/false);
    setup_ongoing_cmdbuf_0({.req_vc = true});
    setup_wrapping_vcs_cmdbuf_0(
        /*req=*/{.start = CMDBUF_FIRST_IDMA_VC, .end = CMDBUF_FIRST_IDMA_VC + CMDBUF_NUM_IDMA_VCS - 1},
        /*resp=*/{.start = CMDBUF_WR_RESP_VC, .end = CMDBUF_WR_RESP_VC});
    setup_max_bytes_in_packet_cmdbuf_0(FACE_ROW_BYTES);
    setup_trids_cmdbuf_0(CMDBUF_DEF_TRID);
    set_len_cmdbuf_0(FACE_ROW_BYTES);
    const std::uint64_t t1 = dm_cycles();

    for (std::uint32_t c = 0; c < ct_dim; c++) {
        reset_addrgen_0();
        setup_src_2D_stride_addrgen_0(
            src_addr + c * TILE_DIM * DATUM_BYTES,
            /*face_size=*/FACE_DIM * row_pitch,
            /*inner=*/{.stride = row_pitch, .end = FACE_DIM * row_pitch},
            /*outer=*/{.stride = FACE_ROW_BYTES, .end = 2 * FACE_ROW_BYTES});
        // Single-iteration outer loop so the face step fires after every tile.
        setup_dest_2D_stride_addrgen_0(
            dst_addr + c * TILE_BYTES,
            /*face_size=*/ct_dim * TILE_BYTES,
            /*inner=*/{.stride = FACE_ROW_BYTES, .end = TILE_BYTES},
            /*outer=*/{.stride = TILE_BYTES, .end = TILE_BYTES});
        for (std::uint32_t i = 0; i < rt_dim * COPIES_PER_TILE; i++) {
            push_both_addrgen_0();
            issue_cmdbuf_0();
        }
    }
    const std::uint64_t t2 = dm_cycles();
    while (!idma_acked_cmdbuf_0()) {
    }
    const std::uint64_t t3 = dm_cycles();
    const std::uint64_t wall_end = wall_clock();

    // cmdbuf 0 is also the NOC write cmdbuf.
    init_wr_cmd_buf(noc_local_xy());

    volatile std::uint32_t* result = reinterpret_cast<volatile std::uint32_t*>(result_addr + MEM_L1_UNCACHED_BASE);
    result[0] = static_cast<std::uint32_t>(wall_end - wall_start);
    result[1] = static_cast<std::uint32_t>(t1 - t0);
    result[2] = static_cast<std::uint32_t>(t2 - t1);
    result[3] = static_cast<std::uint32_t>(t3 - t2);
}
