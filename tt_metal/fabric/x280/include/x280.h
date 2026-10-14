// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_H
#define X280_H

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <stdint.h>

/* --- Phase 3: Memory test constants --- */

/* DDR mailbox layout (valid in DDR mode only -- WayEnable=0xf).
 * LIM addresses become Zero Device in DDR mode; do NOT use LIM for mailbox. */
#define MAILBOX_READY_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x100000ULL))
#define MAILBOX_DONE_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x100008ULL))
#define MAILBOX_BUF_SIZE_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x100010ULL))
#define INPUT_BUF_BASE ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x200000ULL))
#define OUTPUT_BUF_BASE ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x300000ULL))
#define EVICT_BUF_BASE ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x800000ULL))

/* Flag values for DDR mailbox protocol */
#define READY_FLAG 0xCAFEBABE00000001ULL
#define DONE_FLAG 0xCAFEBABE00000002ULL

/* Increment constant for DDR compute test: output[i] = input[i] + INCREMENT_CONST */
#define INCREMENT_CONST 0x0000CAFE00000001ULL

/* Cache line size (L3 LIM stride for sweep) */
#define CACHE_LINE 64

/* LIM test result: firmware writes 0 for pass, nonzero for fail */
#define LIM_TEST_PASS 0x0000000000000000ULL

/* Linker symbols (defined in x280.ld, declared here for C access) */
extern uint64_t _bss_end;

/* Hart 0's probe word in the standalone trap block: 0=normal, 1=probe armed,
 * 2=fault occurred. Only hart 0 may arm through this symbol -- the other harts
 * have their own words, at _trap_block + hartid * _trap_block_size + 0x18.
 * Images on the per-hart protocol use x280_hart_probe() instead. */
extern uint64_t _probe_active;

/* --- Phase 4: NOC test constants --- */

/* Phase 3 used offsets: 0x100000 (READY), 0x100008 (DONE), 0x100010 (BUF_SIZE),
 *                       0x200000 (INPUT_BUF), 0x300000 (OUTPUT_BUF), 0x800000+ (EVICT).
 * Phase 4 NOC mailbox starts at 0x400000 -- clear of all Phase 3 regions. */

/* Host writes desired write count here before booting firmware (0 = use default of 1) */
#define NOC_WRITE_COUNT_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x400000ULL))
/* Firmware writes number of targets tested here (host reads for validation) */
#define NOC_NUM_TARGETS_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x400008ULL))
/* Firmware writes per-target noc_result structs here (indexed by target index) */
#define NOC_RESULT_BASE_ADDR ((volatile uint8_t*)(X280_LOCAL_DDR_TILE_BASE + 0x410000ULL))

/* --- Phase 5: NOC read + ordering test constants --- */

/* TLB slot indices live in rt/x280_noc_map.h. */

/* Phase 5 result region at DDR+0x420000 (clear of Phase 4 at 0x410000) */
#define NOC_PHASE5_RESULT_ADDR ((volatile uint8_t*)(X280_LOCAL_DDR_TILE_BASE + 0x420000ULL))

/* Phase 5 test mode selector: host writes before boot to choose test
 * 0 = Phase 4 write-only
 * 1 = Phase 5 read roundtrip + ordering tests */
#define NOC_TEST_MODE_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x400010ULL))
#define NOC_TEST_MODE_WRITES 0
#define NOC_TEST_MODE_PHASE5 1
#define NOC_TEST_MODE_NIU_REPLAY 2

/* --- Phase 5 NOC-04: DMA/NIU register probing region in DDR --- */
/* Layout: 16 x u32 DMA regs, 16 x u32 NIU0 regs, 16 x u32 NIU1 regs = 192 bytes */
#define NOC_DMA_PROBE_ADDR ((volatile uint32_t*)(X280_LOCAL_DDR_TILE_BASE + 0x430000ULL))
#define NOC_DMA_PROBE_COUNT 16 /* registers to read per region */
#define NOC_DMA_PROBE_DONE_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x430100ULL))
#define NOC_DMA_PROBE_DONE_FLAG 0xBE00000000000001ULL

/* --- Phase 6: Integration test constants --- */

/* Integration test base: clear of all Phase 3-5 regions.
 * Phase 5 highest offset: 0x430100 (NOC_DMA_PROBE_DONE_ADDR).
 * Integration starts at 0x440000 with 16 KiB total layout.
 *
 * Phase 3 uses: 0x100000-0x8FFFFF (mailbox + bufs + eviction)
 * Phase 4 uses: 0x400000-0x41FFFF
 * Phase 5 uses: 0x420000-0x430107
 */
#define INT_BASE_OFFSET 0x440000ULL
#define INT_INPUT_OFFSET 0x440000ULL         /* [+0x0000..+0x0FFF] host writes, X280 reads */
#define INT_NOC_RESULT_OFFSET 0x441000ULL    /* [+0x1000..+0x1FFF] X280 NOC writes/reads */
#define INT_FINAL_RESULT_OFFSET 0x442000ULL  /* [+0x2000..+0x2FFF] X280 Memory Port writes */
#define INT_SENTINEL_OFFSET 0x443000ULL      /* [+0x3000..+0x3007] X280 writes after eviction */
#define INT_MAILBOX_READY_OFFSET 0x443008ULL /* [+0x3008] READY_FLAG from host */

#define INT_INPUT_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + INT_INPUT_OFFSET))
#define INT_NOC_RESULT_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + INT_NOC_RESULT_OFFSET))
#define INT_FINAL_RESULT_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + INT_FINAL_RESULT_OFFSET))
#define INT_SENTINEL_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + INT_SENTINEL_OFFSET))
#define INT_MAILBOX_READY_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + INT_MAILBOX_READY_OFFSET))

#define INT_SENTINEL_DONE 0xCAFEF00D00000006ULL
#define INT_BUF_ELEMENTS 64 /* uint64_t elements per buffer (512 bytes) */

/* --- NIU replay experiment: write NOC command to L2CPU NIU #0 --- */
/* Source data region in DDR (X280 pre-fills with known pattern) */
#define NIU_REPLAY_SRC_OFFSET 0x450000ULL /* 64 bytes of known data */
#define NIU_REPLAY_SRC_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + NIU_REPLAY_SRC_OFFSET))
/* Destination: DRAM (9,3) at offset 0x300000 — clear of all other test regions */
#define NIU_REPLAY_DST_OFFSET 0x300000UL
#define NIU_REPLAY_DST_X 9
#define NIU_REPLAY_DST_Y 3
/* Result region: NIU reg dump before/after fire */
#define NIU_REPLAY_RESULT_ADDR ((volatile uint32_t*)(X280_LOCAL_DDR_TILE_BASE + 0x460000ULL))
#define NIU_REPLAY_DONE_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + 0x460200ULL))
#define NIU_REPLAY_DONE_FLAG 0xC0DE000000000007ULL

/* --- Phase 7: Vector + Multi-Hart test constants --- */

/* Phase 7 base offset: clear of all prior phases.
 * Highest prior offset: NIU_REPLAY_DONE_ADDR =
 * X280_LOCAL_DDR_TILE_BASE + 0x460200 + 8.
 * Phase 7 starts at clean 64KB boundary: 0x470000. */
#define P7_BASE_OFFSET 0x470000ULL

/* Multi-hart: each hart writes its mhartid here (4 x u64 = 32 bytes) */
#define HART_RESULT_OFFSET (P7_BASE_OFFSET + 0x000ULL)
#define HART_RESULT_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + HART_RESULT_OFFSET))

/* IPI ack: harts 1-3 write ack flags after receiving IPI (4 x u64 = 32 bytes) */
#define IPI_ACK_OFFSET X280_IPI_ACK_OFFSET
#define IPI_ACK_ADDR ((volatile uint64_t*)X280_IPI_ACK_ADDR)
#define IPI_ACK_FLAG X280_IPI_ACK_FLAG

/* Vector test inputs and output (64 x u32 = 256 bytes each) */
#define VEC_INPUT_A_OFFSET (P7_BASE_OFFSET + 0x040ULL)
#define VEC_INPUT_B_OFFSET (P7_BASE_OFFSET + 0x140ULL)
#define VEC_OUTPUT_OFFSET (P7_BASE_OFFSET + 0x240ULL)
#define VEC_INPUT_A_ADDR ((volatile uint32_t*)(X280_LOCAL_DDR_TILE_BASE + VEC_INPUT_A_OFFSET))
#define VEC_INPUT_B_ADDR ((volatile uint32_t*)(X280_LOCAL_DDR_TILE_BASE + VEC_INPUT_B_OFFSET))
#define VEC_OUTPUT_ADDR ((volatile uint32_t*)(X280_LOCAL_DDR_TILE_BASE + VEC_OUTPUT_OFFSET))
#define VEC_ELEMENTS 64 /* VLMAX=16 -> 4 iterations */

/* VLEN check: firmware writes the vl returned by first vsetvli (must be 16 for VLEN=512) */
#define VEC_VLEN_CHECK_OFFSET (P7_BASE_OFFSET + 0x340ULL)
#define VEC_VLEN_CHECK_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + VEC_VLEN_CHECK_OFFSET))
#define VEC_EXPECTED_VL 16 /* VLEN=512, SEW=32, LMUL=1 -> VLMAX=16 */

/* Phase 7 done sentinel */
#define P7_SENTINEL_OFFSET (P7_BASE_OFFSET + 0x348ULL)
#define P7_SENTINEL_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + P7_SENTINEL_OFFSET))
#define P7_SENTINEL_DONE 0xCAFEF00D00000007ULL

/* Per-hart stack size (must match entry.S and the linker scripts) */
#define HART_STACK_SIZE 0x8000UL /* 32 KiB */

/* Hart 0 boot-complete flag -- harts 1-3 spin on this before accessing globals */
#define HART0_BOOT_FLAG_OFFSET (P7_BASE_OFFSET + 0x350ULL)
#define HART0_BOOT_FLAG_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + HART0_BOOT_FLAG_OFFSET))
#define HART0_BOOT_DONE 0xB007000000000000ULL

/* --- Phase 8: Benchmark suite constants --- */

/* Phase 8 DDR layout: highest prior offset = HART0_BOOT_FLAG at 0x470358.
 * Phase 8 starts at clean 64KB boundary: 0x480000. */
#define BENCH_BASE_OFFSET 0x480000ULL

/* Result region: array of bench_result_t structs.
 * Each struct is 32 bytes: { id[4], pad[4], cycles[8], iterations[8], bytes[8] }.
 * Max 64 results = 2048 bytes at [0x480000..0x4807FF].
 * (28 cache-sweep benchmarks added in Phase 8 extension: 29 + 28 = 57 total) */
#define BENCH_RESULT_ADDR ((volatile uint8_t*)(X280_LOCAL_DDR_TILE_BASE + BENCH_BASE_OFFSET))
#define BENCH_MAX_RESULTS 64

/* Metadata region at +0x800: firmware writes PLL frequency, NUM_RESULTS */
#define BENCH_META_OFFSET (BENCH_BASE_OFFSET + 0x800ULL)
#define BENCH_META_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + BENCH_META_OFFSET))
/* Host writes PLL frequency here before boot; firmware reads it for metadata */
#define BENCH_PLL_INPUT_OFFSET (BENCH_BASE_OFFSET + 0x810ULL)
#define BENCH_PLL_INPUT_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + BENCH_PLL_INPUT_OFFSET))
/* META_ADDR[0] = pll_freq_mhz, META_ADDR[1] = num_results */

/* Sentinel at +0x1000: signals benchmark complete */
#define BENCH_SENTINEL_OFFSET (BENCH_BASE_OFFSET + 0x1000ULL)
#define BENCH_SENTINEL_ADDR ((volatile uint64_t*)(X280_LOCAL_DDR_TILE_BASE + BENCH_SENTINEL_OFFSET))
#define BENCH_SENTINEL_DONE 0xCAFEF00D00000008ULL

/* NOC DRAM target offset for benchmark writes/reads (clear of all other regions) */
#define BENCH_NOC_TARGET_OFFSET 0x500000UL

/* --- Phase 8 extension: Cache hierarchy sweep benchmarks ---
 *
 * The cache sweep uses the same DRAM region as BENCH_NOC_TARGET_OFFSET.
 * Cache sweep benchmarks run LAST in main(), after all NOC/DDR benchmarks,
 * so overwriting this region is safe.
 *
 * Layout:
 *   CACHE_SWEEP_BASE_ADDR  = X280_LOCAL_DDR_TILE_BASE + 0x500000  (4 MiB sweep buffer)
 *   EVICT_L3_BASE_ADDR     = X280_LOCAL_DDR_TILE_BASE + 0x900000  (2 MiB full-L3 eviction buffer)
 *
 * Both regions fit within DRAM_SIZE (12 MiB) and below the stack (0xBE0000).
 */
#define CACHE_SWEEP_BASE_OFFSET 0x500000ULL
#define CACHE_SWEEP_BASE_ADDR ((volatile uint8_t*)(X280_LOCAL_DDR_TILE_BASE + CACHE_SWEEP_BASE_OFFSET))
#define CACHE_SWEEP_MAX_SIZE (4 * 1024 * 1024) /* 4 MiB */

#define EVICT_L3_BASE_OFFSET 0x900000ULL /* 2 MiB full-L3 eviction buffer */
#define EVICT_L3_SIZE (2 * 1024 * 1024)  /* 2 MiB -- covers full 2 MiB L3 */

#endif /* X280_H */
