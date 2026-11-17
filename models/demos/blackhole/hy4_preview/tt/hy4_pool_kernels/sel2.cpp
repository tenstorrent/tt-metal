// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B36 (HY4_B36_SEL2=1): the stage-2 select of the indexer as ONE program, replaces the stock top-KL over the NB*KB
// block candidates (topk_large_indices) + cs_gather (ids and values in top-k order).
//   Input (cs_cand2 outputs): bvals bf16 / bgids u32 = NB sorted-descending candidate lists of KB entries per row s
//   (blocked layout: page s holds the NB lists back to back; non-blocked: page s*NB + b), -inf (0xFF80) / sentinel id
//   for invalid entries. Output: lvals bf16 [S,1,1,KL], lids u32 [S,1,1,KL]: the KL best entries in total order (value
//   desc, list b asc, position asc), exhausted -> (-inf, 0xFFFFFFFF). Sorted descending, as the stock chain's gathered
//   output (the stock xl top-k breaks ties arbitrarily). Merge path split: core (q, s) produces the output ranks [q*OQ,
//   (q+1)*OQ): one bisection (as sel3) finds how many entries of every list precede rank q*OQ (cursors), then a NB-way
//   merge emits OQ entries; the ids are fetched with 64 B reads.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline uint32_t key16(uint32_t v) {
    v &= 0xFFFFu;
    v = (v == 0x8000u) ? 0u : v;  // -0 == +0
    return (v & 0x8000u) ? (~v & 0xFFFFu) : (v | 0x8000u);
}

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t NQ = get_compile_time_arg_val(1);
    constexpr uint32_t KL = get_compile_time_arg_val(2);
    constexpr uint32_t NB = get_compile_time_arg_val(3);
    constexpr uint32_t KB = get_compile_time_arg_val(4);
    constexpr uint32_t BLOCKED = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id = get_compile_time_arg_val(6);
    constexpr auto v_args = TensorAccessorArgs<7>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto ov_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto og_args = TensorAccessorArgs<ov_args.next_compile_time_args_offset()>();
    constexpr uint32_t OQ = KL / NQ;
    static_assert(NB <= 16 && KL % NQ == 0);
    const uint32_t s = get_arg_val<uint32_t>(0), q = get_arg_val<uint32_t>(1);
    const uint32_t a_v = get_common_arg_val<uint32_t>(0), a_g = get_common_arg_val<uint32_t>(1),
                   a_ov = get_common_arg_val<uint32_t>(2), a_og = get_common_arg_val<uint32_t>(3);
    Noc noc;
    constexpr uint32_t VPB = BLOCKED ? NB * KB * 2 : KB * 2;  // page bytes of bvals
    constexpr uint32_t GPB = BLOCKED ? NB * KB * 4 : KB * 4;  // page bytes of bgids
    const auto v_acc = TensorAccessor(v_args, a_v, VPB);
    const auto g_acc = TensorAccessor(g_args, a_g, GPB);
    const auto ov_acc = TensorAccessor(ov_args, a_ov, KL * 2);
    const auto og_acc = TensorAccessor(og_args, a_og, KL * 4);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    // L1: [vals NB*KB*2][id staging OQ*64][out vals OQ*2 (padded to 64)][out ids OQ*4]
    const uint32_t vals_b = base;
    const uint32_t stg_b = vals_b + NB * KB * 2;
    const uint32_t ov_b = stg_b + OQ * 64;
    const uint32_t og_b = ov_b + ((OQ * 2 + 63) & ~63u);

    {
        DeviceZoneScopedN("S2_RD");
        if constexpr (BLOCKED) {
            noc.async_read(v_acc, cb, VPB, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = 0});
        } else {
            for (uint32_t b = 0; b < NB; ++b) {
                noc.async_read(
                    v_acc, cb, KB * 2, {.page_id = s * NB + b, .offset_bytes = 0}, {.offset_bytes = b * KB * 2});
            }
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
    }
    const uint16_t* __restrict vals = reinterpret_cast<const uint16_t*>(vals_b);

    uint32_t cur[16];
    {
        DeviceZoneScopedN("S2_SEL");
        const uint32_t R = q * OQ;  // number of entries preceding this core's first output rank
        uint32_t cl[16], ch[16], tmp[16];
        auto cnt_ge = [&](uint32_t c, uint32_t T, uint32_t a, uint32_t b) -> uint32_t {
            const uint16_t* __restrict l = vals + c * KB;
            if (T > 0x8000u) {
                const int32_t tb = (int32_t)(T - 0x8000u);
                while (a < b) {
                    const uint32_t m = (a + b) >> 1;
                    if ((int32_t)(int16_t)l[m] >= tb) {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
            } else if (T == 0x8000u) {
                while (a < b) {
                    const uint32_t m = (a + b) >> 1;
                    if ((uint32_t)l[m] <= 0x8000u) {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
            } else {
                const uint32_t nb = (~T) & 0xFFFFu;
                while (a < b) {
                    const uint32_t m = (a + b) >> 1;
                    if ((uint32_t)l[m] <= nb) {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
            }
            return a;
        };
        if (R == 0) {
            for (uint32_t b = 0; b < NB; ++b) {
                cur[b] = 0;
            }
        } else {
            const uint32_t PIV = (R + NB - 1) / NB - 1;
            uint32_t kmin = 0xFFFFu, kmax = 0;
            for (uint32_t c = 0; c < NB; ++c) {
                const uint32_t k = key16(vals[c * KB + PIV]);
                kmin = k < kmin ? k : kmin;
                kmax = k > kmax ? k : kmax;
            }
            uint32_t lo = kmin > 0x80u ? kmin : 0x80u;
            uint32_t tot = 0;
            for (uint32_t c = 0; c < NB; ++c) {
                cl[c] = cnt_ge(c, lo, 0, KB);
                tot += cl[c];
            }
            if (tot <= R && lo == 0x80u) {  // fewer valid entries than R: every list is consumed
                for (uint32_t c = 0; c < NB; ++c) {
                    cur[c] = cl[c];
                }
            } else {
                uint32_t hi = kmax + 1;  // count_ge(hi) < R by construction
                for (uint32_t c = 0; c < NB; ++c) {
                    ch[c] = cnt_ge(c, hi, 0, cl[c]);
                }
                while (hi - lo > 1) {
                    const uint32_t mid = (lo + hi) >> 1;
                    uint32_t t2 = 0;
                    for (uint32_t c = 0; c < NB; ++c) {
                        tmp[c] = cnt_ge(c, mid, ch[c], cl[c]);
                        t2 += tmp[c];
                    }
                    if (t2 >= R) {
                        lo = mid;
                        for (uint32_t c = 0; c < NB; ++c) {
                            cl[c] = tmp[c];
                        }
                    } else {
                        hi = mid;
                        for (uint32_t c = 0; c < NB; ++c) {
                            ch[c] = tmp[c];
                        }
                    }
                }
                uint32_t gt = 0;
                for (uint32_t c = 0; c < NB; ++c) {
                    gt += ch[c];
                }
                uint32_t rem = R - gt;  // tied entries to place before rank R: lower list first
                for (uint32_t c = 0; c < NB; ++c) {
                    const uint32_t g = cl[c] - ch[c];
                    const uint32_t take = g < rem ? g : rem;
                    rem -= take;
                    cur[c] = ch[c] + take;
                }
            }
        }
    }

    volatile uint16_t* ovp = reinterpret_cast<volatile uint16_t*>(ov_b);
    volatile uint32_t* ogp = reinterpret_cast<volatile uint32_t*>(og_b);
    uint32_t nsrc[OQ];
    {
        DeviceZoneScopedN("S2_MRG");
        uint32_t hk[16];
        for (uint32_t b = 0; b < NB; ++b) {
            hk[b] = cur[b] < KB ? key16(vals[b * KB + cur[b]]) : 0u;
        }
        for (uint32_t j = 0; j < OQ; ++j) {
            uint32_t bk = 0, bb = 0;
            for (uint32_t b = 0; b < NB; ++b) {
                if (hk[b] > bk) {
                    bk = hk[b];
                    bb = b;
                }
            }
            if (bk >= 0x80u) {
                const uint32_t p = cur[bb];
                ovp[j] = vals[bb * KB + p];
                const uint32_t e = (BLOCKED ? bb * KB : 0) + p;  // element index inside the id page
                const uint32_t pg = BLOCKED ? s : s * NB + bb;
                const uint32_t off = (e * 4u) & ~63u;
                noc.async_read(
                    g_acc, cb, 64, {.page_id = pg, .offset_bytes = off}, {.offset_bytes = stg_b - base + j * 64});
                nsrc[j] = (e * 4u) & 63u;
                cur[bb] = p + 1;
                hk[bb] = p + 1 < KB ? key16(vals[bb * KB + p + 1]) : 0u;
            } else {
                ovp[j] = 0xFF80;
                nsrc[j] = 0xFFFFFFFFu;
            }
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t j = 0; j < OQ; ++j) {
            ogp[j] =
                nsrc[j] == 0xFFFFFFFFu ? 0xFFFFFFFFu : *reinterpret_cast<volatile uint32_t*>(stg_b + j * 64 + nsrc[j]);
        }
    }
    {
        DeviceZoneScopedN("S2_WR");
        noc.async_write(cb, ov_acc, OQ * 2, {.offset_bytes = ov_b - base}, {.page_id = s, .offset_bytes = q * OQ * 2});
        noc.async_write(cb, og_acc, OQ * 4, {.offset_bytes = og_b - base}, {.page_id = s, .offset_bytes = q * OQ * 4});
        noc.async_write_barrier();
    }
}
