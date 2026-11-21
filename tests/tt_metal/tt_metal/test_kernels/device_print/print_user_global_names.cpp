// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

// Kernels define these names at global scope, so DPRINT doesn't: generic type names, here the ones DPRINT's old stream
// API and its tile code used, and names that kernels in this repository define for their own printing.
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
struct tile_info_t {};
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#define DEBUG_PRINT 0

// As the CCL op kernels declare it.
void dprint(const F32& value) { DPRINT("dprint: {}\n", value.f); }

// As the moe_compute op kernels declare it.
void print_tile_rows(
    U32 cb_idx,
    U32 tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT("print_tile_rows: {}\n", cb_idx);
}

// Uses each type name as a type here and with parentheses at the call, so a function or function-like macro of that
// name fails too.
void use_type_names(
    BF16, ENDL, SETPRECISION, FIXED, DEFAULTFLOAT, HEX, OCT, DEC, SETW, U32_ARRAY, TYPED_U32_ARRAY, tile_info_t) {}

void kernel_main() {
    use_type_names(
        BF16(),
        ENDL(),
        SETPRECISION(),
        FIXED(),
        DEFAULTFLOAT(),
        HEX(),
        OCT(),
        DEC(),
        SETW(),
        U32_ARRAY(),
        TYPED_U32_ARRAY(),
        tile_info_t());
    // Taking each function's address fails if any other function of that name is declared.
    auto print_value = &dprint;
    auto print_rows = &print_tile_rows;
    F32 value = F32();
    value.u = 0x3fc00000;  // 1.5f
    print_value(value);
    print_rows(U32(2), 0, false, 0, 32, 0, 32);
    DPRINT_DISPATCH("DPRINT_DISPATCH\n");
    DPRINT_COMBINE("DPRINT_COMBINE\n");
    DPRINT("DEBUG_PRINT: {}\n", DEBUG_PRINT);
}
