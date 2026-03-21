// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This header is force-included when compiling kernel code for clang-tidy
// static analysis. It provides:
//
//  1. Stub values for content normally generated at JIT compile time
//     (by tt_metal/jit_build/genfiles.cpp).
//  2. Global variable definitions that the firmware wrappers
//     (ncrisck.cc / brisck.cc / trisck.cc) normally provide.
//  3. Includes of the *real* firmware headers so that clang-tidy analyses
//     as much real code as possible.
//
// The actual values don't matter for static analysis — we just need
// syntactically valid definitions so that clang-tidy can parse the kernel
// source code.

#pragma once

#include <cstdint>

// ===========================================================================
// 1) Stubs for JIT-generated content
// ===========================================================================

// -- Legacy compute kernel syntax support --
// Compute kernels use `void MAIN { ... }` where MAIN expands to a function
// name like math_main().  We provide a fallback here.
#ifndef MAIN
#define MAIN math_main()
#endif
#ifndef MATH
#define MATH(x) x
#endif
#ifndef PACK
#define PACK(x)
#endif
#ifndef UNPACK
#define UNPACK(x)
#endif

// -- chlkc_dst_accum_mode.h (JIT-generated) --
constexpr bool DST_ACCUM_MODE = false;

// -- chlkc_dst_sync_mode.h (JIT-generated) --
#ifndef DST_SYNC_MODE
#define DST_SYNC_MODE DstSync::SyncHalf
#endif

// -- chlkc_math_approx_mode.h (JIT-generated) --
constexpr bool APPROX = true;

// -- chlkc_math_fidelity.h (JIT-generated) --
constexpr std::int32_t MATH_FIDELITY = 255;

// -- chlkc_pack_data_format.h (JIT-generated) --
constexpr unsigned char pack_src_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr unsigned char pack_dst_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// -- chlkc_pack_tile_dims.h (JIT-generated) --
constexpr uint8_t pack_tile_num_faces[32] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
constexpr uint8_t pack_partial_face[32] = {};
constexpr uint8_t pack_tile_face_r_dim[32] = {
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
};
constexpr uint8_t pack_narrow_tile[32] = {};
constexpr uint8_t pack_tile_r_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint8_t pack_tile_c_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint16_t pack_tile_size[32] = {
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
};

// -- chlkc_unpack_data_format.h (JIT-generated) --
constexpr std::int32_t unpack_src_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr std::int32_t unpack_dst_format[32] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// -- chlkc_unpack_tile_dims.h (JIT-generated) --
constexpr uint8_t unpack_tile_num_faces[32] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
constexpr uint8_t unpack_partial_face[32] = {};
constexpr uint8_t unpack_tile_face_r_dim[32] = {
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
};
constexpr uint8_t unpack_narrow_tile[32] = {};
constexpr uint8_t unpack_tile_r_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint8_t unpack_tile_c_dim[32] = {
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
};
constexpr uint16_t unpack_tile_size[32] = {
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
    1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088,
};

// ===========================================================================
// 1b) Named compile-time argument support
// ===========================================================================
// In real JIT builds each kernel gets its own KERNEL_COMPILE_TIME_ARG_MAP
// define that maps string names → indices into the compile-time args array.
// The function get_named_compile_time_arg_val() in compile_time_args.h is
// only declared when that macro is defined (#ifdef guard).
//
// For clang-tidy analysis we do NOT define KERNEL_COMPILE_TIME_ARG_MAP
// (the string-map approach can't be made constexpr for arbitrary names).
// Instead, we provide get_named_compile_time_arg_val as a function-like
// macro that expands to a constexpr-compatible constant.  Because the
// real function definition in compile_time_args.h is guarded by
// #ifdef KERNEL_COMPILE_TIME_ARG_MAP, it is simply skipped — no conflict.
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
#undef KERNEL_COMPILE_TIME_ARG_MAP
#endif
#define get_named_compile_time_arg_val(name) (static_cast<uint32_t>(1))

// ===========================================================================
// 2) Include the real firmware headers
// ===========================================================================
// This mirrors what the firmware wrappers (ncrisck.cc / brisck.cc) do.
// The includes pull in the real device API headers so clang-tidy analyses
// real code, not stubs.

#include "risc_common.h"
#include "tensix.h"

// Workaround: In the TT RISC-V GCC toolchain, "uint short" is valid syntax
// (GCC extension for "unsigned short").  Clang doesn't support this.
// Since tensix.h already processed `using uint = std::uint32_t;`, we can
// safely redefine `uint` as `unsigned` so that `uint short` → `unsigned short`.
// On 32-bit RISC-V, unsigned int == uint32_t, so this is type-safe.
#define uint unsigned

#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"

// ===========================================================================
// 3) Global variable definitions
// ===========================================================================
// The firmware wrappers define these; headers only declare them as extern.
// We provide definitions here so that the translation units link (or at
// least parse) correctly.

// From ncrisck.cc / brisck.cc — NOC tracking counters
uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
uint32_t noc_posted_writes_num_issued[NUM_NOCS];

// From risc_common.h extern declarations
uint8_t my_x[NUM_NOCS];
uint8_t my_y[NUM_NOCS];

// From firmware_common.h — bank-to-NOC coordinate lookup tables
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];

// From firmware_common.h — virtual coordinate lookup tables
uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)];
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)];

// From firmware_common.h / dataflow_api_common.h — runtime arg pointers
// (noc_index is defined as constexpr in the headers when NOC_INDEX is set)
uint32_t* rta_l1_base;
uint32_t* crta_l1_base;
uint32_t* sem_l1_base[ProgrammableCoreType::COUNT];

// From trisck.cc / ckernel_globals.h — compute kernel globals
uint32_t cfg_state_id;
uint32_t unp_cfg_context;
volatile uint32_t l1_buffer[16];
uint32_t pack_sync_tile_dst_ptr;
uint32_t math_sync_tile_dst_index;
uint32_t gl_alu_format_spec_reg;
uint32_t op_info_offset;

// From brisck.cc / trisck.cc — ckernel namespace variables
// The exact types use `uint` (typedef for uint32_t) and `tt_reg_ptr`
// (RISC-V attribute).  We use the same types here so there's no mismatch
// when ckernel.h is later included by compute kernel code.
namespace ckernel {
volatile tt_reg_ptr uint* regfile = reinterpret_cast<volatile uint*>(REGFILE_BASE);
volatile tt_reg_ptr uint* pc_buf_base = reinterpret_cast<volatile uint*>(PC_BUF_BASE);
volatile tt_reg_ptr uint* mailbox_base[4] = {
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX0_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX2_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX3_BASE),
};
}  // namespace ckernel

// From circular_buffer_interface.h
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
