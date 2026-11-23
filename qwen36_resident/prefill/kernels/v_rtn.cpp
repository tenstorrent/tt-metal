// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Strata S0 (bfp4 V): rounds a bf16 tile tensor in place to nearest on the bfp4 grid it is about to be packed to, so
// the device packer's truncation toward zero (the typecast that follows) keeps every value exactly. Blackhole bfp4
// shares one exponent per face row (16 consecutive bf16 of a tile's face, contiguous in L1) and keeps 3 magnitude bits:
// the grid of a row is m * 2^(emax - 129), m = 0 .. 7, emax the row's largest biased exponent. Software, on a
// data-movement core: no device buffer is allocated (the ttnn-op version allocated intermediates inside the prefill
// trace and hung served, 2026-11-23). Ties round up (half away from zero); m is capped at 7 (the row's exponent is
// kept).
//
// Runtime args: 0 the tensor's DRAM address (interleaved bf16 tiles, 2048 B pages), 1 first tile, 2 tiles.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_buf = 0;
constexpr uint32_t kTileBytes = 2048;

FORCE_INLINE void round_row(volatile tt_l1_ptr uint16_t* r) {
    uint32_t emax = 0;
    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t e = (r[i] >> 7) & 0xFF;
        emax = e > emax ? e : emax;
    }
    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t b = r[i];
        const uint32_t e = (b >> 7) & 0xFF;
        const uint32_t sign = b & 0x8000;
        uint32_t m = 0;
        if (e != 0) {  // (denormals: below any representable step of a normal row, 0)
            // |x| / 2^(emax - 129) = (128 + mantissa) * 2^(e - emax - 5): shift right by emax - e + 5 (>= 5), rounded
            const uint32_t sh = emax - e + 5;
            if (sh < 16) {
                m = (128 + (b & 0x7F) + (1u << (sh - 1))) >> sh;
                m = m > 7 ? 7 : m;
            }
        }
        if (m == 0 || emax < 3) {
            r[i] = sign;  // (a signed zero: the packer stores the sign bit)
            continue;
        }
        const uint32_t p = m >= 4 ? 2 : (m >= 2 ? 1 : 0);  // the leading bit of m
        r[i] = sign | ((emax - 2 + p) << 7) | ((m << (7 - p)) & 0x7F);
    }
}

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> s{.bank_base_address = addr, .page_size = kTileBytes};
    const uint32_t buf = get_write_ptr(cb_buf);
    for (uint32_t t = t0; t < t0 + nt; t++) {
        const uint64_t src = s.get_noc_addr(t);
        noc_async_read(src, buf, kTileBytes);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
        for (uint32_t row = 0; row < kTileBytes / 32; row++) {
            round_row(v + row * 16);
        }
        noc_async_write(buf, src, kTileBytes);
        noc_async_write_barrier();
    }
}
