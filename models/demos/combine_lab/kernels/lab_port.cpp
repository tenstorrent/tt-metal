// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine lab port (one core a fabric connection: a direction and link of this chip) - the production combine port's
// two jobs, nothing else:
//  - rows out: ROWS rows to every (destination chip, accumulator) pair it serves (its destinations' accumulators on
//    its link), each one fabric unicast of PB bytes from this core's source row into the pair's landing ring - slot
//    (rows sent to the pair so far) % D of the ring (me, accumulator) on the destination's accumulator core - once the
//    ring has room (D + credits back - rows sent); pairs are served round robin, a full ring never blocks the others;
//  - credits back: for every source chip whose credits this port returns (the first hop toward it is this port's,
//    its accumulators on this link), a snapshot of the cumulative rows each accumulator took from it, written over the
//    source port's credit array for this chip, once CRED_THR rows are pending (or nothing else is left to do).
// Everything is cumulative over calls (rings drain every call), so no per-call zeroing.
//
// CT (named): see the constexprs. RT: 0 mesh id, 1 me (block index), 2 link, 3 NDST, NDST x (dst block index, dst
// fabric chip id), NSRC, NSRC x (src block index, src fabric chip id, src port core x | y << 8), A x acc core
// (x | y << 8), then the fabric connection's args.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#ifdef LAB_TIMING
#include "api/debug/kernel_profiler.hpp"
#endif
using namespace tt::tt_fabric;

constexpr uint32_t PB = get_named_compile_time_arg_val("PB");      // packet bytes (row + trailer)
constexpr uint32_t SLOT = get_named_compile_time_arg_val("SLOT");  // landing slot stride
constexpr uint32_t A = get_named_compile_time_arg_val("A");        // accumulators a chip
constexpr uint32_t AP = get_named_compile_time_arg_val("AP");      // A rounded up to 4
constexpr uint32_t D = get_named_compile_time_arg_val("D");        // slots a (source, accumulator) ring
constexpr uint32_t N = get_named_compile_time_arg_val("N");        // chips
constexpr uint32_t L = get_named_compile_time_arg_val("L");        // links a direction
constexpr uint32_t NH = get_named_compile_time_arg_val("NH");      // rotating headers a destination
constexpr uint32_t ROWS = get_named_compile_time_arg_val("ROWS");  // rows a (destination, accumulator) pair a call
constexpr uint32_t CRED_THR = get_named_compile_time_arg_val("CRED_THR");
constexpr uint32_t ST = get_named_compile_time_arg_val("STATE");  // the state block (same address on every core)
constexpr uint32_t RINGS = get_named_compile_time_arg_val("RINGS");
constexpr uint32_t CREDIT = get_named_compile_time_arg_val("CREDIT");  // 0: no flow control (slots overwritten)
// POOL > 0: the rings in a pool of POOL cores (production's landing on the flat expert cores): ring slot g = (o N + s) D
// + k on pool core g % NPOOL, at RINGS + (g / NPOOL) SLOT; 0: on the accumulator's own core
constexpr uint32_t POOL = get_named_compile_time_arg_val("POOL");
constexpr uint32_t NPOOL = POOL > 0 ? POOL : 1;  // (a divisor in branches compiled with POOL 0)
// ORDER: the round robin over the pairs - 0 destination-major (a destination's accumulators back to back: bursts to
// one chip), 1 accumulator-major (consecutive rows to different destinations, as the all-to-all)
constexpr uint32_t ORDER = get_named_compile_time_arg_val("ORDER");
// ORDER 2: bursts of BURST rows a destination (its accumulators in groups of BURST, destinations cycled)
constexpr uint32_t BURST = get_named_compile_time_arg_val("BURST");
// PRODUCER: the rows come from the writers (lab_writer.cpp) through this port's injection ring (IRING, ISLOTS slots):
// a row is sent once its reader (lab_preader.cpp) marks it READY, its home and accumulator read from its trailer (at
// ROWB); rows go out in any order within the ring (a row waits only for its own ring's room), slots are freed in
// order once their sends are flushed (FREED). Else synthetic rows (the pairs above).
constexpr uint32_t PRODUCER = get_named_compile_time_arg_val("PRODUCER");
// SL > 1 (SPLIT): a row's L parts go one a link of its direction - this port carries part lnk of every row its
// direction serves, landing in the home's ring (me, part); else a whole row a link (production's link hash)
constexpr uint32_t SL = get_named_compile_time_arg_val("SL");
constexpr uint32_t IRING = get_named_compile_time_arg_val("IRING");
constexpr uint32_t ISLOTS = get_named_compile_time_arg_val("ISLOTS");
constexpr uint32_t ROWB = get_named_compile_time_arg_val("ROWB");
constexpr uint32_t READY = ST + get_named_compile_time_arg_val("O_READY");
constexpr uint32_t FREED = ST + get_named_compile_time_arg_val("O_FREED");
// FUSED: this RISC also runs the hand-off (the reader's grants and ready rows, lab_preader.cpp) every pass, as
// production's port writer does with the accumulators on the port cores
constexpr uint32_t FUSED = get_named_compile_time_arg_val("FUSED");
constexpr uint32_t W = get_named_compile_time_arg_val("W");
constexpr uint32_t GB = get_named_compile_time_arg_val("GB");
constexpr uint32_t GRANT = ST + get_named_compile_time_arg_val("O_GRANT");
constexpr uint32_t LANDED = ST + get_named_compile_time_arg_val("O_LANDED");
constexpr uint32_t GRANTED = ST + get_named_compile_time_arg_val("O_GRANTED");
// the state block's offsets (lab.py StateLayout)
constexpr uint32_t CRED = ST + get_named_compile_time_arg_val("O_CRED");    // [N][AP] credits back (source side)
constexpr uint32_t USED = ST + get_named_compile_time_arg_val("O_USED");    // [N][AP] rows sent (source side)
constexpr uint32_t RET = ST + get_named_compile_time_arg_val("O_RET");      // [N][AP] rows taken (home side)
constexpr uint32_t RTOT = ST + get_named_compile_time_arg_val("O_RTOT");    // [N] rows taken a source (home side)
constexpr uint32_t RDONE = ST + get_named_compile_time_arg_val("O_RDONE");  // [N] covered by the last snapshot
constexpr uint32_t HDRS = ST + get_named_compile_time_arg_val("O_HDRS");    // [N][NH] x HSZ headers
constexpr uint32_t STAGE = ST + get_named_compile_time_arg_val("O_STAGE");  // [NH] x 4 AP snapshot staging
constexpr uint32_t SRC = ST + get_named_compile_time_arg_val("O_SRC");      // the source row (PB bytes)
constexpr uint32_t WORK = ST + get_named_compile_time_arg_val("O_WORK");    // [N A] pairs, [N A] rows left
constexpr uint32_t HSZ = 128;
static_assert(sizeof(PACKET_HEADER_TYPE) <= HSZ, "header stride");

void kernel_main() {
    size_t a = 0;
    const uint16_t mesh = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++), lnk = get_arg_val<uint32_t>(a++);
    const uint32_t n_prod = get_arg_val<uint32_t>(a++);  // (PRODUCER: this port's rows this call)
    const uint32_t pid = get_arg_val<uint32_t>(a++);     // (this port's id: direction index L + link)
    const uint32_t ndst = get_arg_val<uint32_t>(a++);
    uint8_t dst_idx[N];
    uint16_t dst_chip[N];
    for (uint32_t i = 0; i < ndst; ++i) {
        dst_idx[i] = get_arg_val<uint32_t>(a++);
        dst_chip[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t nsrc = get_arg_val<uint32_t>(a++);
    uint8_t src_idx[N];
    uint16_t src_chip[N];
    uint32_t src_xy[N];
    for (uint32_t i = 0; i < nsrc; ++i) {
        src_idx[i] = get_arg_val<uint32_t>(a++);
        src_chip[i] = get_arg_val<uint32_t>(a++);
        src_xy[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t acc_xy[A];
    for (uint32_t o = 0; o < A; ++o) {
        acc_xy[o] = get_arg_val<uint32_t>(a++);
    }
    uint32_t pool_xy[POOL > 0 ? POOL : 1];
    for (uint32_t i = 0; i < POOL; ++i) {
        pool_xy[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t nw = FUSED ? get_arg_val<uint32_t>(a++) : 0;  // (FUSED: the writers feeding this port)
    uint32_t wxy[FUSED ? W : 1];
    for (uint32_t i = 0; i < nw; ++i) {
        wxy[i] = get_arg_val<uint32_t>(a++);
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CRED);
    volatile tt_l1_ptr uint32_t* used = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(USED);
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RET);
    volatile tt_l1_ptr uint32_t* rtot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RTOT);
    volatile tt_l1_ptr uint32_t* rdone = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RDONE);
    auto hdr = [&](uint32_t chip_slot, uint32_t rot) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(HDRS + (chip_slot * NH + rot) * HSZ);
    };
    // headers: one route per destination / source chip, copied to its NH rotations (a send rewrites the NoC command)
    auto route = [&](uint32_t chip_slot, uint16_t chip) {
        auto* h0 = hdr(chip_slot, 0);
        (void)fabric_set_unicast_route(h0, chip, mesh);
        const volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(h0);
        for (uint32_t r = 1; r < NH; ++r) {
            volatile tt_l1_ptr uint32_t* hr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr(chip_slot, r));
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; ++w) {
                hr[w] = s0[w];
            }
        }
    };
    for (uint32_t i = 0; i < ndst; ++i) {
        route(dst_idx[i], dst_chip[i]);
    }
    for (uint32_t i = 0; i < nsrc; ++i) {
        route(src_idx[i], src_chip[i]);
    }
    // the source row's arrival flag (its last word)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SRC + PB - 4)[0] = 1;

    // the pairs this port serves: (dst slot, accumulator) with the accumulator's rows on my link
    uint32_t npair = 0;
    volatile tt_l1_ptr uint32_t* pair = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WORK);
    volatile tt_l1_ptr uint32_t* left = pair + N * A;
    if constexpr (ORDER == 0) {
        for (uint32_t i = 0; i < ndst; ++i) {
            for (uint32_t o = 0; o < A; ++o) {
                if (SL > 1 || (o + me + dst_idx[i]) % L == lnk) {
                    pair[npair++] = dst_idx[i] << 8 | o;
                }
            }
        }
    } else if constexpr (ORDER == 2) {
        for (uint32_t g0 = 0; g0 < A; g0 += BURST * L) {  // (a destination's accumulators on my link: every L-th)
            for (uint32_t i = 0; i < ndst; ++i) {
                for (uint32_t o = g0; o < g0 + BURST * L && o < A; ++o) {
                    if (SL > 1 || (o + me + dst_idx[i]) % L == lnk) {
                        pair[npair++] = dst_idx[i] << 8 | o;
                    }
                }
            }
        }
    } else {
        for (uint32_t o = 0; o < A; ++o) {
            for (uint32_t i = 0; i < ndst; ++i) {
                if (SL > 1 || (o + me + dst_idx[i]) % L == lnk) {
                    pair[npair++] = dst_idx[i] << 8 | o;
                }
            }
        }
    }
    for (uint32_t p = 0; p < npair; ++p) {
        left[p] = ROWS;
    }
    uint32_t rows_left = PRODUCER ? n_prod : npair * ROWS;
    // credits to return this call: every (source, accumulator on my link) pair's ROWS
    uint32_t ret_left = 0;
    for (uint32_t i = 0; i < nsrc; ++i) {
        for (uint32_t o = 0; o < A; ++o) {
            if (SL > 1 || (o + src_idx[i] + me) % L == lnk) {
                ret_left += ROWS;
            }
        }
    }
    if constexpr (!CREDIT) {
        ret_left = 0;
    }
    uint32_t sent = 0, unflushed = 0;
#ifdef LAB_TIMING
    uint32_t t_fab = 0, t_busy = 0, t_blk = 0, t_tail = 0;
    const uint32_t t0 = get_timestamp_32b();
    uint32_t t_last = 0;
#endif
    conn.open();
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(READY);
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FREED);
    uint32_t front = *freed;  // (PRODUCER: the ring's oldest unfreed row)
    uint64_t sentm = 0;       // (bit j: row front + j sent)
    const uint32_t part = SL > 1 ? lnk : 0;
    const uint32_t end = front + n_prod;
    auto flush = [&]() {
        noc_async_writes_flushed();
        unflushed = 0;
        if constexpr (PRODUCER) {
            while (sentm & 1ull) {
                sentm >>= 1;
                ++front;
            }
            *freed = front;
        }
    };
    auto give_back = [&](bool any) {
        for (uint32_t i = 0; i < nsrc; ++i) {
            const uint32_t s = src_idx[i];
            const uint32_t pend = rtot[s] - rdone[s];
            if (pend == 0 || (!any && pend < CRED_THR)) {
                continue;
            }
            if (unflushed >= NH) {
                flush();
            }
            const uint32_t rot = sent % NH;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(STAGE + rot * 4 * AP);
            uint32_t cov = 0;
            for (uint32_t o = 0; o < AP; ++o) {
                const uint32_t v = ret[s * AP + o];
                st[o] = v;
                cov += v;
            }
            auto* h = hdr(s, rot);
            h->to_noc_unicast_write(
                NocUnicastCommandHeader{get_noc_addr(src_xy[i] & 0xFF, src_xy[i] >> 8, CRED + 4 * me * AP)}, 4 * AP);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(STAGE + rot * 4 * AP, 4 * AP, reinterpret_cast<uint32_t>(h));
            ++sent;
            ++unflushed;
            const uint32_t newly = cov - rdone[s];
            rdone[s] = cov;
            ret_left = newly < ret_left ? ret_left - newly : 0;
        }
    };
    uint32_t p = 0;
    // (FUSED: the hand-off's state - lab_preader.cpp)
    volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LANDED);
    volatile tt_l1_ptr uint32_t* granted_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GRANTED);
    uint32_t granted = *granted_w;
    auto hand_off = [&]() {
        if (granted < end) {
            const uint32_t room = ISLOTS - (granted - front);
            const uint32_t g = room < end - granted ? room : end - granted;
            if (g >= GB || (g > 0 && granted + g == end)) {
                for (uint32_t i = 0; i < nw; ++i) {
                    noc_semaphore_inc(get_noc_addr(wxy[i] & 0xFF, wxy[i] >> 8, GRANT + 4 * pid), g);
                }
                granted += g;
            }
        }
        uint32_t m = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < nw; ++i) {
            m = landed[i] < m ? landed[i] : m;
        }
        if (m > *ready) {
            *ready = m;
        }
    };
    while (rows_left > 0 || ret_left > 0) {
        invalidate_l1_cache();
        bool did = false;
        if constexpr (FUSED) {
            hand_off();
        }
#ifdef LAB_TIMING
        const uint32_t tp = get_timestamp_32b();
        const bool had_rows = rows_left > 0;
#endif
        if constexpr (CREDIT) {
            give_back(false);
        }
        if constexpr (PRODUCER) {  // one pass over the ring's ready rows
            const uint32_t rdy = *ready;
            for (uint32_t j = 0; j < ISLOTS && unflushed < NH; ++j) {
                const uint32_t k = front + j;
                if (k >= rdy || k >= end) {
                    break;
                }
                if (sentm >> j & 1ull) {
                    continue;
                }
                const uint32_t slot = IRING + (k % ISLOTS) * SLOT;
                const volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(slot + ROWB);
                const uint32_t d = tr[0], o = tr[1], ci = d * AP + o;
                const uint32_t u = used[ci];
                if (CREDIT && D + cred[ci] - u < 1) {
                    continue;
                }
                uint64_t dn;
                if constexpr (POOL > 0) {
                    const uint32_t g = (o * N * SL + me * SL + part) * D + u % D, xy = pool_xy[g % NPOOL];
                    dn = get_noc_addr(xy & 0xFF, xy >> 8, RINGS + (g / NPOOL) * SLOT);
                } else {
                    dn = get_noc_addr(acc_xy[o] & 0xFF, acc_xy[o] >> 8, RINGS + ((me * SL + part) * D + u % D) * SLOT);
                }
                auto* h = hdr(d, sent % NH);
                h->to_noc_unicast_write(NocUnicastCommandHeader{dn}, PB);
#ifdef LAB_TIMING
                const uint32_t tf = get_timestamp_32b();
#endif
                conn.wait_for_empty_write_slot();
#ifdef LAB_TIMING
                t_fab += get_timestamp_32b() - tf;
#endif
                conn.send_current_slot_non_blocking(slot, PB, reinterpret_cast<uint32_t>(h));
                used[ci] = u + 1;
                sentm |= 1ull << j;
                ++sent;
                ++unflushed;
                --rows_left;
                did = true;
#ifdef LAB_TIMING
                t_last = get_timestamp_32b() - t0;
#endif
            }
        }
        // one pass over the pairs: a row to every pair with rows left and room
        for (uint32_t k = 0; !PRODUCER && k < npair && rows_left > 0 && unflushed < NH;
             ++k, p = p + 1 == npair ? 0 : p + 1) {
            if (left[p] == 0) {
                continue;
            }
            const uint32_t d = pair[p] >> 8, o = pair[p] & 0xFF, ci = d * AP + o;
            const uint32_t u = used[ci];
            if (CREDIT && D + cred[ci] - u < 1) {
                continue;
            }
            uint64_t dn;
            if constexpr (POOL > 0) {
                const uint32_t g = (o * N * SL + me * SL + part) * D + u % D, xy = pool_xy[g % NPOOL];
                dn = get_noc_addr(xy & 0xFF, xy >> 8, RINGS + (g / NPOOL) * SLOT);
            } else {
                dn = get_noc_addr(acc_xy[o] & 0xFF, acc_xy[o] >> 8, RINGS + ((me * SL + part) * D + u % D) * SLOT);
            }
            auto* h = hdr(d, sent % NH);
            h->to_noc_unicast_write(NocUnicastCommandHeader{dn}, PB);
#ifdef LAB_TIMING
            const uint32_t tf = get_timestamp_32b();
#endif
            conn.wait_for_empty_write_slot();
#ifdef LAB_TIMING
            t_fab += get_timestamp_32b() - tf;
#endif
            conn.send_current_slot_non_blocking(SRC, PB, reinterpret_cast<uint32_t>(h));
            used[ci] = u + 1;
            ++sent;
            ++unflushed;
            --left[p];
            --rows_left;
            did = true;
#ifdef LAB_TIMING
            t_last = get_timestamp_32b() - t0;
#endif
        }
        if (unflushed >= NH || (!did && unflushed)) {
            flush();
        }
        if (!did) {
            if constexpr (CREDIT) {
                give_back(true);
            }
        }
#ifdef LAB_TIMING
        {
            const uint32_t dt = get_timestamp_32b() - tp;
            if (did) {
                t_busy += dt;
            } else if (had_rows) {
                t_blk += dt;  // (rows left, every ring with rows full)
            } else {
                t_tail += dt;  // (all rows out: credits left to return)
            }
        }
#endif
    }
    flush();
    if constexpr (FUSED) {
        *granted_w = granted;
    }
    conn.close();
    noc_async_atomic_barrier();
#ifdef LAB_TIMING
    DeviceTimestampedData("LAB-PEND", get_timestamp_32b() - t0);
    DeviceTimestampedData("LAB-PLAST", t_last);
    DeviceTimestampedData("LAB-PFAB", t_fab);
    DeviceTimestampedData("LAB-PBUSY", t_busy);
    DeviceTimestampedData("LAB-PBLK", t_blk);
    DeviceTimestampedData("LAB-PTAIL", t_tail);
    DeviceTimestampedData("LAB-PROWS", npair * ROWS);
#endif
}
