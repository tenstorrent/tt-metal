// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

// Kernels define these names at global scope, so DPRINT doesn't: generic type names, here the ones DPRINT's old stream
// API used, and names that kernels in this repository define for their own printing.
union F32 {
    float f;
    uint32_t u;
};
struct BF16 {};
using U32 = uint32_t;
struct ENDL {};
struct SETPRECISION {};
struct FIXED {};
struct DEFAULTFLOAT {};
struct HEX {};
struct OCT {};
struct DEC {};
struct SETW {};
struct U32_ARRAY {};
struct TYPED_U32_ARRAY {};
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#define DEBUG_PRINT 0

void dprint(F32 value) { DPRINT("dprint: {}\n", value.f); }

void print_tile_rows(U32 rows) { DPRINT("print_tile_rows: {}\n", rows); }

void kernel_main() {
    F32 value;
    value.u = 0x3fc00000;  // 1.5f
    dprint(value);
    print_tile_rows(2);
    DPRINT_DISPATCH("DPRINT_DISPATCH\n");
    DPRINT_COMBINE("DPRINT_COMBINE\n");
    DPRINT("DEBUG_PRINT: {}\n", DEBUG_PRINT);
}
