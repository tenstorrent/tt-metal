// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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
// MATH_FIDELITY is defined in the chlkc_descriptors.h stub (after llk_defs.h
// pulls in ckernel::MathFidelity).  Do NOT define it here as int32_t — the
// real generated file uses `constexpr ckernel::MathFidelity` and the LLK
// templates reject an integer literal as the template argument.

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
constexpr uint8_t unpack_num_faces_r_dim[32] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};
constexpr uint8_t unpack_num_faces_c_dim[32] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};

// -- chlkc_pack_tile_dims.h additions --
constexpr uint8_t pack_num_faces_r_dim[32] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};
constexpr uint8_t pack_num_faces_c_dim[32] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};

// -- PROCESSOR_INDEX (set by compiler flags per processor type) --
// BRISC=0, NCRISC=1, TRISC0/1/2=2/3/4 on tt-1xx.  Compute kernels supply
// this via -DPROCESSOR_INDEX=<n>; provide 0 as a safe default for others
// (e.g. BRISC dataflow kernels).
#ifndef PROCESSOR_INDEX
#define PROCESSOR_INDEX 0
#endif

// ===========================================================================
// 1b) Kernel-specific JIT macro fallbacks
// ===========================================================================
// Some kernels (e.g. eltwise_binary.cpp) use macros that are normally set by
// the JIT layer as kernel-specific compile-time arguments.  Provide sensible
// defaults here so clang-tidy can parse the translation unit.
//
// ELTWISE_OP_TYPE — template argument of type EltwiseBinaryType (unscoped enum
//   from llk_defs.h).  ELWMUL (= 0) is a safe default; the identifier will be
//   in scope when the kernel headers are included later.
#ifndef ELTWISE_OP_TYPE
#define ELTWISE_OP_TYPE ELWMUL
#endif
//
// ELTWISE_OP — function-like macro called as ELTWISE_OP(icb0,icb1,itile0,itile1,idst).
//   mul_tiles() is declared in tt_metal/hw/inc/api/compute/eltwise_binary.h
//   which is already included by the time ELTWISE_OP is called.
#ifndef ELTWISE_OP
#define ELTWISE_OP(icb0, icb1, itile0, itile1, idst) mul_tiles(icb0, icb1, itile0, itile1, idst)
#endif
//
// REDUCE_OP / REDUCE_DIM — used as default template arguments in
//   api/compute/reduce.h:  template <PoolType reduce_type = REDUCE_OP,
//                                    ReduceDim reduce_dim = REDUCE_DIM, ...>
// The real JIT build injects these from KERNEL_COMPILE_TIME_ARGS.
// Both PoolType and ReduceDim are unscoped enums in llk_defs.h (pulled in
// by compute headers before reduce.h is reached).  SUM / REDUCE_ROW are
// safe, widely-supported defaults.
#ifndef REDUCE_OP
#define REDUCE_OP SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM REDUCE_ROW
#endif
//
// FP32_DEST_ACC — bool template arg used in reduce_init / reduce_tile /
//   reduce_uninit (api/compute/reduce.h) as enforce_fp32_accumulation.
// The real JIT build injects this from KERNEL_COMPILE_TIME_ARGS.
// Default to false (FP32 accum disabled) — same as DST_ACCUM_MODE above.
#ifndef FP32_DEST_ACC
#define FP32_DEST_ACC false
#endif
//
// TILE_HW_VAL — tile height × width (pixels).  Injected by groupnorm program
//   factory as std::to_string(tile_hw) where tile_hw = tile.get_tile_hw().
//   For standard 32×32 tiles the value is 1024.
#ifndef TILE_HW_VAL
#define TILE_HW_VAL 1024
#endif
//
// ENABLE_FP32_DEST_ACC — same meaning as FP32_DEST_ACC but used by softmax
//   attention compute kernels (softmax.cpp, softmax_large_tensor.cpp,
//   softmax_sharded.cpp).  Injected by the program factory as "1" or "0".
//   Default to false to avoid static_assert(is_fp32_dest_acc_en) when
//   DST_ACCUM_MODE is false.
#ifndef ENABLE_FP32_DEST_ACC
#define ENABLE_FP32_DEST_ACC false
#endif
//
// EXP_APPROX — bool flag controlling approximate vs exact exp in softmax
//   attention compute kernels.  Injected by the program factory as "1" or "0"
//   (math_approx_mode ? "1" : "0").  Default to false (exact exp).
#ifndef EXP_APPROX
#define EXP_APPROX false
#endif
//
// EXP_APPROX_MODE — same semantics as EXP_APPROX but used by SDPA (scaled
//   dot-product attention) compute kernels.  Injected by the program factory.
#ifndef EXP_APPROX_MODE
#define EXP_APPROX_MODE false
#endif

// ===========================================================================
// 1d) Named compile-time argument support
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

#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"

// Force-process the ckernel_ops.h stub *before* any kernel TU is compiled.
//
// Problem: ckernel.h (in tt_llk_wormhole_b0/common/inc/) uses #include "ckernel_ops.h"
// which, under C/C++ rules, looks in the *directory of the including file first*.
// That means it finds the real ckernel_ops.h (same directory) before our stub in
// jit_stubs/ can intercept.  The real header defines INSTRUCTION_WORD as an asm
// statement that produces "unknown directive" / "invalid constraint 'i'" errors.
//
// Fix: Include "ckernel_ops.h" here (the prelude is force-included first, so
// jit_stubs/ is searched first).  Our stub runs, #include_next's the real file,
// and then redefines INSTRUCTION_WORD to a no-op.  Both the stub and the real file
// are now #pragma-once-locked.  Later when ckernel.h tries to include "ckernel_ops.h"
// from its own directory, both files are already in the "seen" set — skipped.
// INSTRUCTION_WORD stays as the harmless no-op.
#include "ckernel_ops.h"

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

// From firmware wrappers (brisc.cc / active_erisc.cc / etc.) — bank address offset tables
// declared as extern in firmware_common.h, used by dataflow_api_addrgen.h
int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
int32_t bank_to_l1_offset[NUM_L1_BANKS];

// From firmware_common.h — virtual coordinate lookup tables
uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)];
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)];

// Pull in dataflow_api_common.h early so that noc_index (constexpr uint8_t
// noc_index = NOC_INDEX) is defined before any tensor accessor headers that
// use it as a default argument (e.g. pages_address_iterator.h).  Kernels
// that include api/tensor/tensor_accessor.h without api/dataflow/dataflow_api.h
// would otherwise fail with "'noc_index' does not refer to a value".
// Because the header uses #pragma once, including it here and again from
// dataflow_api.h is a no-op for the second inclusion — no redefinition.
#include "internal/dataflow/dataflow_api_common.h"

// From firmware_common.h / dataflow_api_common.h — runtime arg pointers
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

// ===========================================================================
// 4) Dispatch / prefetch kernel JIT-constant defaults
// ===========================================================================
// cq_dispatch.cpp, cq_prefetch.cpp and cq_dispatch_subordinate.cpp are
// parameterised by dozens of JIT-generated #defines.  Real builds get them
// from fd_kernel.cpp via JitBuildOptions.  For clang-tidy we provide safe
// integer defaults (0 / 1) under #ifndef guards so the translation units parse
// without errors; the actual values are irrelevant for static analysis.

// -- Shared / cq_common.hpp --
#ifndef FD_CORE_TYPE
#define FD_CORE_TYPE 0   // ProgrammableCoreType::TENSIX
#endif

// -- cq_dispatch_subordinate.cpp & cq_dispatch.cpp --
#ifndef CB_BASE
#define CB_BASE 0
#endif
#ifndef CB_LOG_PAGE_SIZE
#define CB_LOG_PAGE_SIZE 10
#endif
#ifndef CB_SIZE
#define CB_SIZE 0
#endif
#ifndef MY_DISPATCH_CB_SEM_ID
#define MY_DISPATCH_CB_SEM_ID 0
#endif
#ifndef UPSTREAM_DISPATCH_CB_SEM_ID
#define UPSTREAM_DISPATCH_CB_SEM_ID 0
#endif
#ifndef DISPATCH_S_SYNC_SEM_BASE_ADDR
#define DISPATCH_S_SYNC_SEM_BASE_ADDR 0
#endif
#ifndef MCAST_GO_SIGNAL_ADDR
#define MCAST_GO_SIGNAL_ADDR 0
#endif
#ifndef UNICAST_GO_SIGNAL_ADDR
#define UNICAST_GO_SIGNAL_ADDR 0
#endif
#ifndef DISTRIBUTED_DISPATCHER
#define DISTRIBUTED_DISPATCHER 0
#endif
#ifndef FIRST_STREAM_USED
#define FIRST_STREAM_USED 0
#endif
#ifndef MAX_NUM_WORKER_SEMS
#define MAX_NUM_WORKER_SEMS 1
#endif
#ifndef MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES
#define MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES 1
#endif
#ifndef VIRTUALIZE_UNICAST_CORES
#define VIRTUALIZE_UNICAST_CORES 0
#endif
#ifndef NUM_VIRTUAL_UNICAST_CORES
#define NUM_VIRTUAL_UNICAST_CORES 0
#endif
#ifndef NUM_PHYSICAL_UNICAST_CORES
#define NUM_PHYSICAL_UNICAST_CORES 0
#endif
#ifndef WORKER_MCAST_GRID
#define WORKER_MCAST_GRID 0
#endif
#ifndef NUM_WORKER_CORES_TO_MCAST
#define NUM_WORKER_CORES_TO_MCAST 0
#endif
#ifndef UPSTREAM_NOC_X
#define UPSTREAM_NOC_X 0
#endif
#ifndef UPSTREAM_NOC_Y
#define UPSTREAM_NOC_Y 0
#endif
#ifndef DOWNSTREAM_NOC_X
#define DOWNSTREAM_NOC_X 0
#endif
#ifndef DOWNSTREAM_NOC_Y
#define DOWNSTREAM_NOC_Y 0
#endif
#ifndef MY_NOC_X
#define MY_NOC_X 0
#endif
#ifndef MY_NOC_Y
#define MY_NOC_Y 0
#endif
#ifndef DOWNSTREAM_SUBORDINATE_NOC_X
#define DOWNSTREAM_SUBORDINATE_NOC_X 0
#endif
#ifndef DOWNSTREAM_SUBORDINATE_NOC_Y
#define DOWNSTREAM_SUBORDINATE_NOC_Y 0
#endif
#ifndef UPSTREAM_NOC_INDEX
#define UPSTREAM_NOC_INDEX 1  // must differ from NOC_INDEX (0) per static_assert in cq_dispatch.cpp
#endif

// -- cq_dispatch.cpp specific --
#ifndef DISPATCH_CB_BASE
#define DISPATCH_CB_BASE 0
#endif
#ifndef DISPATCH_CB_LOG_PAGE_SIZE
#define DISPATCH_CB_LOG_PAGE_SIZE 10
#endif
#ifndef DISPATCH_CB_PAGES
#define DISPATCH_CB_PAGES 1
#endif
#ifndef DISPATCH_CB_BLOCKS
#define DISPATCH_CB_BLOCKS 1
#endif
#ifndef UPSTREAM_SYNC_SEM
#define UPSTREAM_SYNC_SEM 0
#endif
#ifndef COMMAND_QUEUE_BASE_ADDR
#define COMMAND_QUEUE_BASE_ADDR 0
#endif
#ifndef COMPLETION_QUEUE_BASE_ADDR
#define COMPLETION_QUEUE_BASE_ADDR 0
#endif
#ifndef COMPLETION_QUEUE_SIZE
#define COMPLETION_QUEUE_SIZE 0
#endif
#ifndef DOWNSTREAM_CB_BASE
#define DOWNSTREAM_CB_BASE 0
#endif
#ifndef DOWNSTREAM_CB_SIZE
#define DOWNSTREAM_CB_SIZE 0
#endif
#ifndef MY_DOWNSTREAM_CB_SEM_ID
#define MY_DOWNSTREAM_CB_SEM_ID 0
#endif
#ifndef DOWNSTREAM_CB_SEM_ID
#define DOWNSTREAM_CB_SEM_ID 0
#endif
#ifndef SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE
#define SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE 0
#endif
#ifndef SPLIT_PREFETCH
#define SPLIT_PREFETCH 0
#endif
#ifndef PREFETCH_H_NOC_XY
#define PREFETCH_H_NOC_XY 0
#endif
#ifndef PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR
#define PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR 0
#endif
#ifndef PREFETCH_H_MAX_CREDITS
#define PREFETCH_H_MAX_CREDITS 0
#endif
#ifndef PACKED_WRITE_MAX_UNICAST_SUB_CMDS
#define PACKED_WRITE_MAX_UNICAST_SUB_CMDS 1
#endif
#ifndef DEV_COMPLETION_Q_RD_PTR
#define DEV_COMPLETION_Q_RD_PTR 0
#endif
#ifndef DEV_COMPLETION_Q_WR_PTR
#define DEV_COMPLETION_Q_WR_PTR 0
#endif
#ifndef DEV_DISPATCH_PROGRESS_PTR
#define DEV_DISPATCH_PROGRESS_PTR 0
#endif
#ifndef HOST_COMPLETION_Q_WR_PTR
#define HOST_COMPLETION_Q_WR_PTR 0
#endif
#ifndef IS_D_VARIANT
#define IS_D_VARIANT 0
#endif
#ifndef IS_H_VARIANT
#define IS_H_VARIANT 0
#endif
#ifndef EW_DIM
#define EW_DIM 1
#endif
#ifndef WORKER_CREDITS_STREAM_ID
#define WORKER_CREDITS_STREAM_ID 0
#endif

// -- Fabric constants (used by cq_dispatch.cpp and cq_prefetch.cpp) --
#ifndef FABRIC_HEADER_RB_BASE
#define FABRIC_HEADER_RB_BASE 0
#endif
#ifndef FABRIC_HEADER_RB_ENTRIES
#define FABRIC_HEADER_RB_ENTRIES 0
#endif
#ifndef FABRIC_MUX_BUFFER_INDEX_ADDRESS
#define FABRIC_MUX_BUFFER_INDEX_ADDRESS 0
#endif
#ifndef FABRIC_MUX_CHANNEL_BASE_ADDRESS
#define FABRIC_MUX_CHANNEL_BASE_ADDRESS 0
#endif
#ifndef FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES
#define FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES 0
#endif
#ifndef FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS
#define FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS 0
#endif
#ifndef FABRIC_MUX_CONNECTION_INFO_ADDRESS
#define FABRIC_MUX_CONNECTION_INFO_ADDRESS 0
#endif
#ifndef FABRIC_MUX_FLOW_CONTROL_ADDRESS
#define FABRIC_MUX_FLOW_CONTROL_ADDRESS 0
#endif
#ifndef FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL
#define FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL 0
#endif
#ifndef FABRIC_MUX_STATUS_ADDRESS
#define FABRIC_MUX_STATUS_ADDRESS 0
#endif
#ifndef FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS
#define FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS 0
#endif
#ifndef FABRIC_MUX_X
#define FABRIC_MUX_X 0
#endif
#ifndef FABRIC_MUX_Y
#define FABRIC_MUX_Y 0
#endif
#ifndef FABRIC_WORKER_BUFFER_INDEX_SEM
#define FABRIC_WORKER_BUFFER_INDEX_SEM 0
#endif
#ifndef FABRIC_WORKER_FLOW_CONTROL_SEM
#define FABRIC_WORKER_FLOW_CONTROL_SEM 0
#endif
#ifndef FABRIC_WORKER_TEARDOWN_SEM
#define FABRIC_WORKER_TEARDOWN_SEM 0
#endif
#ifndef MY_FABRIC_SYNC_STATUS_ADDR
#define MY_FABRIC_SYNC_STATUS_ADDR 0
#endif
#ifndef NUM_HOPS
#define NUM_HOPS 0
#endif
#ifndef OFFSETOF_MY_DEV_ID
#define OFFSETOF_MY_DEV_ID 0
#endif
#ifndef OFFSETOF_ROUTER_DIRECTION
#define OFFSETOF_ROUTER_DIRECTION 0
#endif
#ifndef OFFSETOF_TO_DEV_ID
#define OFFSETOF_TO_DEV_ID 0
#endif
#ifndef TO_MESH_ID
#define TO_MESH_ID 0
#endif

// -- cq_prefetch.cpp specific --
#ifndef CMDDAT_Q_BASE
#define CMDDAT_Q_BASE 0
#endif
#ifndef CMDDAT_Q_BLOCKS
#define CMDDAT_Q_BLOCKS 1
#endif
#ifndef CMDDAT_Q_LOG_PAGE_SIZE
#define CMDDAT_Q_LOG_PAGE_SIZE 10
#endif
#ifndef CMDDAT_Q_PAGES
#define CMDDAT_Q_PAGES 1
#endif
#ifndef CMDDAT_Q_SIZE
#define CMDDAT_Q_SIZE 0
#endif
#ifndef DISPATCH_S_BUFFER_BASE
#define DISPATCH_S_BUFFER_BASE 0
#endif
#ifndef DISPATCH_S_BUFFER_SIZE
#define DISPATCH_S_BUFFER_SIZE 0
#endif
#ifndef DISPATCH_S_CB_LOG_PAGE_SIZE
#define DISPATCH_S_CB_LOG_PAGE_SIZE 10
#endif
#ifndef DOWNSTREAM_CB_LOG_PAGE_SIZE
#define DOWNSTREAM_CB_LOG_PAGE_SIZE 10
#endif
#ifndef DOWNSTREAM_CB_PAGES
#define DOWNSTREAM_CB_PAGES 1
#endif
#ifndef DOWNSTREAM_DISPATCH_S_CB_SEM_ID
#define DOWNSTREAM_DISPATCH_S_CB_SEM_ID 0
#endif
#ifndef DOWNSTREAM_SYNC_SEM_ID
#define DOWNSTREAM_SYNC_SEM_ID 0
#endif
#ifndef MY_DISPATCH_S_CB_SEM_ID
#define MY_DISPATCH_S_CB_SEM_ID 0
#endif
#ifndef MY_UPSTREAM_CB_SEM_ID
#define MY_UPSTREAM_CB_SEM_ID 0
#endif
#ifndef UPSTREAM_CB_SEM_ID
#define UPSTREAM_CB_SEM_ID 0
#endif
#ifndef PCIE_BASE
#define PCIE_BASE 0
#endif
#ifndef PCIE_SIZE
#define PCIE_SIZE 0
#endif
#ifndef PREFETCH_Q_BASE
#define PREFETCH_Q_BASE 0
#endif
#ifndef PREFETCH_Q_PCIE_RD_PTR_ADDR
#define PREFETCH_Q_PCIE_RD_PTR_ADDR 0
#endif
#ifndef PREFETCH_Q_RD_PTR_ADDR
#define PREFETCH_Q_RD_PTR_ADDR 0
#endif
#ifndef PREFETCH_Q_SIZE
#define PREFETCH_Q_SIZE 0
#endif
#ifndef RINGBUFFER_SIZE
#define RINGBUFFER_SIZE 0
#endif
#ifndef SCRATCH_DB_BASE
#define SCRATCH_DB_BASE 0
#endif
#ifndef SCRATCH_DB_SIZE
#define SCRATCH_DB_SIZE 0
#endif
