// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <overlay_addresses.h>
#include <riscv-pk/encoding.h>
#include <noc_parameters.h>
#include <rocc_instructions.hpp>

#ifdef METAL_LIB_INC
#ifdef __cplusplus
extern "C" {
#endif
#include <metal/cache.h>
#include <metal/cpu.h>
#ifdef __cplusplus
}
#endif
#endif

#define RESET_POSTCODE 0x2B000000

#define DATA_SIGNATURE 0xFEEDC0DE

#define REG32(val) (*((volatile uint32_t*)(((uint64_t)(val)))))
#define REG64(val) (*((volatile uint64_t*)(((uint64_t)(val)))))

#define NUM_TILES_SIZES 12
#define TILE_FREQ {5, 5, 5, 5, 5, 11, 12, 11, 5, 5, 15, 16}
#ifdef OVL_NIGHTLY
#define TILE_SIZES    \
    {160 * 1024,      \
     96 * 1024 + 128, \
     64 * 1024 + 16,  \
     64 * 1024,       \
     32 * 1204 + 192, \
     4 * 1024 + 16,   \
     2 * 1024 + 32,   \
     1 * 1024 + 64,   \
     256,             \
     128,             \
     64,              \
     16}
#else
#define TILE_SIZES    \
    {2 * 1024,        \
     4 * 1024 + 128,  \
     64 * 1024 + 16,  \
     64 * 1024,       \
     32 * 1204 + 192, \
     4 * 1024 + 16,   \
     2 * 1024 + 32,   \
     1 * 1024 + 64,   \
     256,             \
     128,             \
     64,              \
     16}
#endif
#define DEF_TILE_SIZE (1 * 1024 + 64)
#define PERF_TILE_SIZE (64 * 1024)
// Make sure fw never goes over 80KB
#define SRC_ADDR_START (80 * 1024)
#define DEST_ADDR_START (2 * 1024 * 1024)
#define MAX_SIZE_BUFFS (2 * 1024 * 1024 - SRC_ADDR_START)
#define BUFS_PER_CORE_SIZE (200 * 1024)

#define EP_SHIFT 26
#define MASK_SHIFT 36

#define NUM_CPUS 8

// ------------------------------------------------
//  Structures
// ------------------------------------------------

typedef struct {
    uint64_t reg_0;
    uint64_t reg_1;
    uint64_t reg_2;
    uint64_t reg_3;
    uint64_t reg_4;
    uint64_t reg_5;
    uint64_t reg_6;
    uint64_t reg_7;
} ACCELERATOR_VECTOR_t;

//
#ifdef __cplusplus
extern "C" {
#endif
void test_pass(size_t core_id);
void test_fail(size_t core_id);

// calling this function will output a new timestamp
// val must be different than val prev val or it will not output
void test_timestamp(size_t core_id, uint32_t val);

inline void flush_l2(uint64_t addr) {
    asm volatile("fence" : : : "memory");
    *((volatile uint64_t*)L2_FLUSH_ADDR) = addr;
    asm volatile("fence" : : : "memory");
}

inline void invalidate_l2(uint64_t addr) {
    asm volatile("fence" : : : "memory");
    *((volatile uint64_t*)L2_INVALIDATE_ADDR) = addr;
    asm volatile("fence" : : : "memory");
}

inline void noc_fence() {
    NOC_FENCE();
    asm volatile("fence" : : : "memory");
}

uint8_t mem_compare(size_t core_id, uint32_t addr1, uint32_t addr2, uint32_t size_bytes);
uint8_t mem_check_zero(size_t core_id, uint32_t addr, uint32_t size_bytes);
void local_mem_copy(uint32_t src, uint32_t dest, uint32_t bytes);

uint64_t xorshift64(uint64_t state);
uint64_t rnd64();
void mem_rnd64(uint32_t addr);
uint64_t rnd64_range(uint64_t min, uint64_t max);
void mem_rnd64_range(uint32_t addr, uint64_t min, uint64_t max);
uint64_t rnd64_dist(
    uint64_t range1_percent,
    uint64_t range1_min,
    uint64_t range1_max,
    uint64_t range2_percent,
    uint64_t range2_min,
    uint64_t range2_max,
    uint64_t range3_percent,
    uint64_t range3_min,
    uint64_t range3_max);
void mem_rnd64_dist(
    uint32_t addr,
    uint64_t range1_percent,
    uint64_t range1_min,
    uint64_t range1_max,
    uint64_t range2_percent,
    uint64_t range2_min,
    uint64_t range2_max,
    uint64_t range3_percent,
    uint64_t range3_min,
    uint64_t range3_max);
void set_rnd_seed();

void mem_create_data(size_t core_id, uint32_t start_addr, uint32_t num_bytes, uint32_t seq_id = 0);
uint8_t mem_check_data(size_t core_id, uint32_t start_addr, uint32_t seq_id = 0);

uint32_t rnd_tile_size(uint32_t max_size);
uint32_t perf_tile_size(uint32_t max_size);

void program_branch_prediction_mode_csr(uint32_t static_branch_predict);
void program_cpu_chicken_csr(
    uint32_t disableDCacheClockGate,
    uint32_t disableICacheClockGate,
    uint32_t disableCoreClockGate,
    uint32_t disableSpeculativeICacheRefill,
    uint32_t suppressCorruptOnGrantData,
    uint32_t disableICachePrefetch);

void enable_pc_capture();
uint64_t get_debug_pc(int cpu_id);

void testbench_reset(
    uint32_t core_resets, uint32_t uncore_reset, uint32_t aiclk_reset, uint32_t nocclk_reset, uint32_t debug_reset);
void testbench_release_reset();

void test_init();

uint32_t mx();
uint32_t my();
uint32_t mxsize();
uint32_t mysize();

uint64_t mx_smn();
uint64_t my_smn();
uint64_t mxsize_smn();
uint64_t mysize_smn();

uint32_t is_dispatch();

#ifdef __cplusplus
}
#endif
