// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constants shared by the HPCG host code (the Python ops in tt/, xhpcg's C++ ops in xhpcg/tt/) and the kernels, each
// defined once here. Python reads this file (tt/constants.py), so it keeps to these forms:
//   constexpr <type> NAME = <expression>;       integers, + - * / ( ) and names defined above
//   constexpr <type> NAME[] = {a, b, ...};
//   #define HPCG_<OP>_CBS(X) X(CB_NAME, index) ...   circular-buffer indices of an op's kernels, lines continued by '\'
// Kernels include it with HPCG_KERNEL_SIDE defined and see only the first part: the host part's names (TILE, NGRP,
// ...) would collide with the kernels' defines.
#pragma once
#include <cstdint>

// ---- shared with the kernels ----

// sparse_rows: launch options, one runtime flags word (reader arg 10, writer arg 12, compute arg 3) rather than
// defines, so launches that differ only in options share one kernel binary (kernels/gather.h)
constexpr uint32_t F_WSPLIT = 1;  // the writer loads the odd window pieces (both NoCs load the window)
constexpr uint32_t F_ZEROX = 2;   // x = 0 on entry: the windows are zeroed on chip, x is not read
constexpr uint32_t F_RESTF = 4;   // the SpMV pass over block 0 outputs r - A z (the restriction's input)
constexpr uint32_t F_ADDB = 8;    // every output is b + A x (prolongation)

// stencil_matrix: a run's description, written by the data-movement RISC that owns the chunk (uint32 words): [0] word
// of the run's first row in the staging page [1] x index i0 of that row [2] rows [3] window offset of the first row
// itself, then per slot j: [4 + 3 j] window offset at x index 0 (K), [5 + 3 j], [6 + 3 j]: rows [a, e) of the run
// that have the slot's neighbour (the others take their own row's offset)
constexpr uint32_t RUN_WORDS = 4 + 3 * 27;
constexpr uint32_t MAX_RUNS = 96;  // runs described per chunk; the owning RISC writes all slots of any more itself
// the work mailbox (CB_W, uint32 words): message (h, s) at 16 (2 h + s), done (t, h, s) at MB_DONE + 16 (4 t + 2 h + s),
// end (h) at MB_END + 16 h, run list (h, s) at MB_RUNS + (2 h + s) MAX_RUNS RUN_WORDS (kernels/fill.h)
constexpr uint32_t MB_DONE = 64;
constexpr uint32_t MB_END = 256;
constexpr uint32_t MB_RUNS = 512;
constexpr uint32_t MB_WORDS = MB_RUNS + 4 * MAX_RUNS * RUN_WORDS;
constexpr uint32_t SET_PAGES = 55;  // one staging set per RISC: 27 offset pages, 27 value pages, one b page
constexpr uint32_t XCAP = 2048;     // value exceptions recorded per staging set (CB_X0 / CB_X1: 2 x 8 KB)

#ifndef HPCG_KERNEL_SIDE
// ---- host only ----
namespace hpcg_tt {

constexpr int64_t TILE = 1024;  // fp32 words per 32x32 tile (a 4 KB page)
constexpr int NLEV = 4;         // multigrid levels

// sparse_rows (tt/custom_ops/sparse_rows/op.py, xhpcg/tt/ops.cpp)
// circular buffers must end below 1,572,864 B of L1 and start 111,616 B in (measured: 1,500,480 B of CBs ended at
// 1,612,096 B), so at most 1,461,248 B of CBs per core
constexpr uint32_t L1_BUDGET = 1461248;
constexpr int NGATH = 4;  // gathering processors per core: reader, writer, pack thread, unpack thread
constexpr int NGRP = 3;   // slot groups per segment: a work unit is (segment, group); its reads are issued one unit ahead
constexpr int VDEP = 27;  // matrix-value pages in flight (reader)
constexpr int GDEP[] = {8, 6, 4, 3, 2};  // gathered pages in flight per gatherer, largest that fits L1 first
// NoC per core: reader on NoC 0 / writer on NoC 1, swapped inside this core rectangle (logical x0, x1, y0, y1).
// DRAM read responses congest some links; the swap measured fastest for every launch with an SpMV pass.
constexpr int NOC_FLIP_RECT[] = {7, 12, 0, 9};
// planner (tt/custom_ops/sparse_rows/plan.py, xhpcg/tt/plan.cpp)
constexpr int64_t ALIGN = 32;  // rows per ownership unit (128 B of fp32 values)
constexpr int MAX_NB = 14;     // 16 semaphores per core: 1 ready flag + 1 own progress slot + up to 14 neighbour slots
constexpr int32_t GRAN = 16;   // window granule (words) = 64 B: every window piece is NoC-aligned in DRAM and in L1
constexpr int32_t WGAP = 8;    // column runs at most this many granules apart share one window piece
constexpr int64_t MAX_PIECES[] = {256, 4096};  // window pieces per core first made room for, then retried with
#define HPCG_SPARSE_ROWS_CBS(X)                                                                                    \
    X(CB_V, 0) X(CB_G, 1) X(CB_B, 2) X(CB_G2, 5) X(CB_RDY, 6) X(CB_G3, 8) X(CB_OFF3, 9) X(CB_MB, 10) X(CB_G4, 11) \
        X(CB_OFF4, 12) X(CB_V2, 13) X(CB_OUT, 16) X(CB_WIN, 24) X(CB_OFF, 25) X(CB_META, 26) X(CB_META2, 27)       \
            X(CB_OFF2, 28)

// stencil_matrix (tt/custom_ops/stencil_matrix/op.py, xhpcg/tt/generate.cpp)
constexpr uint32_t DM_SLOTS = 6;  // offset slots each data-movement RISC writes itself; the three TRISCs write 21
constexpr uint32_t MB_BYTES = 4 * MB_WORDS;  // the work mailbox
// of every 8 chunks, the writer RISC's (NoC 1) share. Its NoC writes stall less: on the P150 the reader RISC's (NoC 0)
// writes from the top core rows wait on congested routes. Generation A + B (ms) for shares 4 / 5 / 6, median of 3:
// 192^3 8.60 / 6.84 / 7.41, 288^3 24.55 / 22.40 / 25.42, 104^3 2.39 / 2.66 / 2.81 (104^3 setup waits on the host)
constexpr uint32_t CHUNKS_RISC1_DEFAULT = 5;
constexpr uint32_t JOB_ARGS = 10;             // common arguments per job (kernels/generate.cpp)
constexpr uint32_t MIN_META_PAGES = 8;  // META pages per RISC at least (a program's buffer sizes are in its cache key)
#define HPCG_STENCIL_MATRIX_CBS(X) X(CB_S0, 0) X(CB_S1, 1) X(CB_M0, 2) X(CB_M1, 3) X(CB_Z, 4) X(CB_X0, 5) X(CB_X1, 6) X(CB_W, 7)

// cg_vector (tt/custom_ops/cg_vector/op.py, xhpcg/tt/ops.cpp)
constexpr uint32_t VB = 2;  // tiles per read batch (1-8 measured within 3%)
#define HPCG_CG_VECTOR_CBS(X) \
    X(CB_A, 0) X(CB_B, 1) X(CB_C, 2) X(CB_D, 3) X(CB_S, 4) X(CB_ACC, 5) X(CB_OUT, 16) X(CB_OUT2, 17) X(CB_OUT3, 18)
#define HPCG_DOT_FINISH_CBS(X) X(CB_A, 0) X(CB_OUT, 16) X(CB_T, 17)

// a CB table as C++ (name, index) pairs: {HPCG_SPARSE_ROWS_CBS(HPCG_CB_PAIR)}
#define HPCG_CB_PAIR(name, index) {#name, index},

}  // namespace hpcg_tt
#endif
