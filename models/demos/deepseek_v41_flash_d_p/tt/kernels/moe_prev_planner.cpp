// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MOE_REPLICAS=prev (P.56): the replica planner of one MoE layer, run on every chip right after the layer's MoE call
// (tt/expert_prev.py has the terms, the host mirror and the data layouts). ROLE 1 (sender cores, one a first fabric hop
// direction): this chip's expert counts (its own slots of head.counts) to every other chip's landing row, a fused
// write + atomic increment of the receiver's arrival counter. ROLE 0 (the planner core): waits for the other chips'
// counts, plans the layer's replicas for its next call from them (the same integer arithmetic on every chip: every
// chip gets the same plan), writes the slot map into the layer's router bias, the copies of its own chip into the fill
// job pages of the coming MoE calls (the combine's REC_FX pages, the copied layer's buffers in words 6 / 7), and the
// state. ROLE 2 (reset, a request's first chunk): every job page header of this chip zeroed, the global state
// emptied, a new epoch (every layer's state then reads as empty).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t ROLE = get_compile_time_arg_val(0);
constexpr uint32_t NC = get_compile_time_arg_val(1);        // chips
constexpr uint32_t EPC = get_compile_time_arg_val(2);       // home slots a chip
constexpr uint32_t SPC = get_compile_time_arg_val(3);       // slots a chip (EPC + the spare)
constexpr uint32_t SPW = get_compile_time_arg_val(4);       // a landing row's words (SPC rounded up to 16)
constexpr uint32_t W = get_compile_time_arg_val(5);         // job page ring (MoE calls)
constexpr uint32_t NCALL = get_compile_time_arg_val(6);     // MoE calls a chunk (a layer's next call: this + NCALL)
constexpr uint32_t PCS = get_compile_time_arg_val(7);       // pieces a copy
constexpr uint32_t BUD = get_compile_time_arg_val(8);       // pieces a source port a call
constexpr uint32_t CAP = get_compile_time_arg_val(9);       // the pages' cap word (packets in the channel)
constexpr uint32_t MAXJ = get_compile_time_arg_val(10);     // jobs a page
constexpr uint32_t FIX = get_compile_time_arg_val(11);      // a replica's fixed cost (tokens)
constexpr uint32_t MINX = get_compile_time_arg_val(12);     // the least tokens a replica takes
constexpr uint32_t SLACK = get_compile_time_arg_val(13);    // a held replica is kept within this many tokens
constexpr uint32_t MAXNEW = get_compile_time_arg_val(14);   // new copies a plan
constexpr uint32_t LNK = get_compile_time_arg_val(15);      // links a direction (job page = dir x LNK + link 0)
constexpr uint32_t CPAGE = get_compile_time_arg_val(16);    // head.counts' page bytes (aligned)
constexpr uint32_t GW = get_compile_time_arg_val(17);       // global state words
constexpr uint32_t SW = get_compile_time_arg_val(18);       // a layer state's words
constexpr uint32_t MAPPAGE = get_compile_time_arg_val(19);  // the router bias's slot map tile (page)
constexpr uint32_t E = NC * EPC;
constexpr uint32_t NO_EXPERT = 0xFFFFFFFFu;
constexpr uint32_t PG = 1024;                   // a job page's bytes
constexpr uint32_t GSTAMP = 16, GBUD = 16 + W;  // global state: stamps [W], budgets [W][NC][4] (pieces | jobs << 16)
constexpr uint32_t SHOLD = 16, SREADY = 16 + NC, SRECV = 16 + 2 * NC, SRX = 16 + 3 * NC;

inline uint32_t al64(uint32_t x) { return (x + 63) & ~63u; }

inline uint32_t f32_bits(uint32_t v) {  // an integer below 2^24 as fp32 bits (exact)
    if (v == 0) {
        return 0;
    }
    uint32_t e = 0, w = v;  // (v's top bit in five steps: the RISCs have no clz)
    if (w >> 16) {
        w >>= 16;
        e += 16;
    }
    if (w >> 8) {
        w >>= 8;
        e += 8;
    }
    if (w >> 4) {
        w >>= 4;
        e += 4;
    }
    if (w >> 2) {
        w >>= 2;
        e += 2;
    }
    if (w >> 1) {
        e += 1;
    }
    return ((127u + e) << 23) | ((v << (23 - e)) & 0x7FFFFFu);
}

inline uint32_t tile_index(uint32_t i) {  // word i of a row-major 32 x 32 tile -> its place in the tile's faces
    const uint32_t r = i >> 5, c = i & 31;
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the replica planner needs a 2D fabric (agent rule 4)"
#endif
}

inline void read_page(uint32_t base, uint32_t page_bytes, uint32_t page, uint32_t l1, uint32_t bytes) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_bytes};
    noc_async_read(g.get_noc_addr(page), l1, bytes);
}

inline void write_page(uint32_t base, uint32_t page_bytes, uint32_t page, uint32_t off, uint32_t l1, uint32_t bytes) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_bytes};
    noc_async_write(l1, g.get_noc_addr(page, off), bytes);
}

void kernel_main() {
    const uint32_t base = al64(get_write_ptr(0));
    if constexpr (ROLE == 1) {
        // ---------------------------------------------------------------- sender: my counts to the other chips
        const uint32_t cnt = get_arg_val<uint32_t>(0), land = get_arg_val<uint32_t>(1), gst = get_arg_val<uint32_t>(2);
        const uint32_t me = get_arg_val<uint32_t>(3), sem = get_arg_val<uint32_t>(4);
        const uint32_t px = get_arg_val<uint32_t>(5), py = get_arg_val<uint32_t>(6), nd = get_arg_val<uint32_t>(7);
        size_t fa = 8 + 2 * nd;
        const uint32_t l_cnt = base, l_g = al64(l_cnt + CPAGE), l_pay = l_g + 64;
        read_page(gst, GW * 4, 0, l_g, 64);
        read_page(cnt, CPAGE, 0, l_cnt, CPAGE);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_cnt);
        volatile tt_l1_ptr uint32_t* pay = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_pay);
        for (uint32_t j = 0; j < SPW; ++j) {
            pay[j] = j < SPC ? cw[me * SPC + j] : 0u;
        }
        const uint32_t par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g)[0] & 1;
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fa);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        const InterleavedAddrGen<true> lg = {.bank_base_address = land, .page_size = SPW * 4};
        const uint64_t to = lg.get_noc_addr(par * NC + me);  // (my row on every receiver: the same address)
        const uint64_t arrival = get_noc_addr(px, py, sem);
        for (uint32_t j = 0; j < nd; ++j) {
            const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(8 + 2 * j));
            const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(9 + 2 * j));
            route_to(hdr, chip_id, mesh_id);
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{to, arrival, 1, true}, SPW * 4);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(l_pay, SPW * 4, reinterpret_cast<uint32_t>(hdr));
            noc_async_writes_flushed();  // (the payload and the header left L1 before the header is rewritten)
        }
        conn.close();
        return;
    }
    if constexpr (ROLE == 2) {
        // ---------------------------------------------------------------- reset: a request starts
        const uint32_t gst = get_arg_val<uint32_t>(0);
        const uint32_t l_g = base, l_z = al64(l_g + GW * 4);
        volatile tt_l1_ptr uint32_t* G = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_z);
        read_page(gst, GW * 4, 0, l_g, GW * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < 16; ++i) {
            z[i] = 0;
        }
        for (uint32_t s = 0; s < W; ++s) {
            const uint32_t ring = get_arg_val<uint32_t>(2 + s);
            for (uint32_t d = 0; d < 4; ++d) {
                write_page(ring, PG, d * LNK, 0, l_z, 64);
            }
        }
        G[1] = G[1] + 1;  // (a new epoch: every layer's state reads as empty)
        for (uint32_t s = 0; s < W; ++s) {
            G[GSTAMP + s] = NO_EXPERT;
        }
        for (uint32_t i = GBUD; i < GW; ++i) {
            G[i] = 0;
        }
        write_page(gst, GW * 4, 0, 0, l_g, GW * 4);
        noc_async_write_barrier();
        return;
    }
    // -------------------------------------------------------------------- planner
    const uint32_t cnt = get_arg_val<uint32_t>(0), land = get_arg_val<uint32_t>(1), gst = get_arg_val<uint32_t>(2);
    const uint32_t sta = get_arg_val<uint32_t>(3), rb = get_arg_val<uint32_t>(4);
    const uint32_t gu = get_arg_val<uint32_t>(5), dw = get_arg_val<uint32_t>(6), cst = get_arg_val<uint32_t>(7);
    const uint32_t me = get_arg_val<uint32_t>(8), sem = get_arg_val<uint32_t>(9);
    constexpr uint32_t RING0 = 10;
    // L1: counts page, landing rows, global state, layer state, directions, map tile, header / job staging, work
    const uint32_t l_cnt = base, l_land = al64(l_cnt + CPAGE), l_g = al64(l_land + NC * SPW * 4);
    const uint32_t l_st = al64(l_g + GW * 4), l_dir = al64(l_st + SW * 4), l_map = al64(l_dir + NC * NC * 4);
    const uint32_t l_hdr = l_map + 4096, l_job = l_hdr + 64, l_stg = l_job + PG, l_pgs = l_stg + al64(NCALL * 16);
    const uint32_t l_load = l_pgs + NCALL * PG;
    const uint32_t l_share = l_load + E * 4, l_T = l_share + E * 4, l_nr = l_T + NC * 4, l_used = l_nr + E * 4;
    const uint32_t l_fail = l_used + NC * 4;
    volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_cnt);
    volatile tt_l1_ptr uint32_t* lw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_land);
    volatile tt_l1_ptr uint32_t* G = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_st);
    volatile tt_l1_ptr uint32_t* dir = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_dir);
    volatile tt_l1_ptr uint32_t* map = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_map);
    volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_hdr);
    volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_stg);
    uint32_t* load = reinterpret_cast<uint32_t*>(l_load);    // (this core's own work array)
    uint32_t* share = reinterpret_cast<uint32_t*>(l_share);  // (this core's own work array)
    uint32_t* T = reinterpret_cast<uint32_t*>(l_T);          // (this core's own work array)
    uint32_t* nr = reinterpret_cast<uint32_t*>(l_nr);        // (this core's own work array)
    uint32_t* used = reinterpret_cast<uint32_t*>(l_used);    // (this core's own work array)
    uint32_t* fail = reinterpret_cast<uint32_t*>(l_fail);    // (this core's own work array)
    const uint32_t t0 = get_timestamp_32b();
    read_page(gst, GW * 4, 0, l_g, GW * 4);
    read_page(sta, SW * 4, 0, l_st, SW * 4);
    read_page(cst, NC * NC * 4 + 4096, 0, l_dir, NC * NC * 4);
    {  // (the empty map's tile, fp32 bits in tile order: the directions' page holds it after them)
        const InterleavedAddrGen<true> cg = {.bank_base_address = cst, .page_size = NC * NC * 4 + 4096};
        noc_async_read(cg.get_noc_addr(0, NC * NC * 4), l_map, 4096);
    }
    read_page(cnt, CPAGE, 0, l_cnt, CPAGE);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t g = G[0], par = g & 1, epoch = G[1];
    const uint32_t t1 = get_timestamp_32b();
    // the other chips' counts: n - 1 arrivals, then taken back
    if (NC > 1) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        noc_semaphore_wait_min(arrived, NC - 1);
        noc_semaphore_inc(get_noc_addr(sem), 0u - (NC - 1));
        noc_async_atomic_barrier();
    }
    for (uint32_t s = 0; s < NC; ++s) {
        if (s != me) {
            read_page(land, SPW * 4, par * NC + s, l_land + s * SPW * 4, SPW * 4);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t t2 = get_timestamp_32b();
    for (uint32_t j = 0; j < SPW; ++j) {
        lw[me * SPW + j] = j < SPC ? cw[me * SPC + j] : 0u;
    }
    // the job pages call g just read: their headers zeroed (that ring slot is next used at g + W)
    for (uint32_t i = 0; i < 16; ++i) {
        hw[i] = 0;
    }
    for (uint32_t d = 0; d < 4; ++d) {
        write_page(get_arg_val<uint32_t>(RING0 + g % W), PG, d * LNK, 0, l_hdr, 64);
    }
    noc_async_write_barrier();
    // the layer's state (another epoch: empty)
    if (st[0] != epoch) {
        st[0] = epoch;
        for (uint32_t k = 0; k < NC; ++k) {
            st[SHOLD + k] = NO_EXPERT;
            st[SREADY + k] = 0;
        }
    }
    for (uint32_t k = 0; k < NC; ++k) {
        st[SRECV + k] = NO_EXPERT;
        st[SRX + k] = 0;
        used[k] = 0;
        fail[k] = 0;
    }
    // each expert's tokens in the call that just ran (its home slot + the spare slots that held it)
    for (uint32_t b = 0; b < E; ++b) {
        load[b] = lw[(b / EPC) * SPW + b % EPC];
        nr[b] = 0;
    }
    for (uint32_t k = 0; k < NC; ++k) {
        if (st[SHOLD + k] != NO_EXPERT) {
            load[st[SHOLD + k]] += lw[k * SPW + EPC];
        }
    }
    uint32_t tsum = 0, tmax0 = 0;
    for (uint32_t c = 0; c < NC; ++c) {
        uint32_t t = 0;
        for (uint32_t j = 0; j < EPC; ++j) {
            t += load[c * EPC + j];
        }
        T[c] = t;
        tsum += t;
        tmax0 = t > tmax0 ? t : tmax0;
    }
    for (uint32_t b = 0; b < E; ++b) {
        share[b] = load[b];
    }
    uint32_t nrec = 0, nreuse = 0, nnew = 0, ndrop = 0, pieces = 0;
    // fill: copy (h -> k) of home slot `tab` in the coming calls g + 1 .. g + NCALL - 2 (the last one before the
    // layer's next call is a fence: its start barrier on the same routes lands the pieces); on success the last call
    // that carries a piece, else 0 (nothing changed)
    auto alloc = [&](uint32_t h, uint32_t k, uint32_t tab) -> uint32_t {
        const uint32_t p = dir[h * NC + k];
        uint32_t room = 0;
        for (uint32_t o = g + 1; o + 2 <= g + NCALL; ++o) {
            const uint32_t s = o % W;
            const uint32_t v = G[GSTAMP + s] == o ? G[GBUD + (s * NC + h) * 4 + p] : 0u;
            if ((v >> 16) < MAXJ && (v & 0xFFFF) < BUD) {
                room += BUD - (v & 0xFFFF);
            }
        }
        if (room < PCS) {
            return 0;
        }
        uint32_t rem = PCS, p0 = 0, last = 0, ns = 0;
        for (uint32_t o = g + 1; o + 2 <= g + NCALL && rem; ++o) {
            const uint32_t s = o % W;
            if (G[GSTAMP + s] != o) {
                G[GSTAMP + s] = o;
                for (uint32_t i = 0; i < NC * 4; ++i) {
                    G[GBUD + s * NC * 4 + i] = 0;
                }
            }
            const uint32_t v = G[GBUD + (s * NC + h) * 4 + p];
            const uint32_t u = v & 0xFFFF, j = v >> 16;
            if (j >= MAXJ || u >= BUD) {
                continue;
            }
            const uint32_t take = BUD - u < rem ? BUD - u : rem;
            G[GBUD + (s * NC + h) * 4 + p] = (u + take) | (j + 1) << 16;
            if (h == me) {  // (my job: page dir x LNK of ring slot s, job j - staged, written below)
                stg[4 * ns] = s;
                stg[4 * ns + 1] = j;
                stg[4 * ns + 2] = p0;
                stg[4 * ns + 3] = take;
                ++ns;
            }
            p0 += take;
            rem -= take;
            last = o;
        }
        if (ns) {  // (my jobs' pages read at once, changed, written at once: whole pages, DRAM stays 64 B aligned)
            for (uint32_t i = 0; i < ns; ++i) {
                read_page(get_arg_val<uint32_t>(RING0 + stg[4 * i]), PG, p * LNK, l_pgs + i * PG, PG);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t i = 0; i < ns; ++i) {
                volatile tt_l1_ptr uint32_t* pg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_pgs + i * PG);
                const uint32_t j = stg[4 * i + 1];
                pg[0] = j + 1;
                pg[1] = CAP;
                volatile tt_l1_ptr uint32_t* e = pg + 16 + 8 * j;
                e[0] = tab | k << 8;
                e[1] = stg[4 * i + 2];
                e[2] = stg[4 * i + 2] + stg[4 * i + 3];
                e[3] = 0;
                e[4] = 0;
                e[5] = 0;
                e[6] = gu;
                e[7] = dw;
                write_page(get_arg_val<uint32_t>(RING0 + stg[4 * i]), PG, p * LNK, 0, l_pgs + i * PG, PG);
            }
            noc_async_write_barrier();
        }
        pieces += PCS;
        return last;
    };
    for (uint32_t it = 0; it < 2 * NC; ++it) {
        uint32_t h = 0;
        for (uint32_t c = 1; c < NC; ++c) {
            h = T[c] > T[h] ? c : h;
        }
        uint32_t b = h * EPC;
        for (uint32_t j = 1; j < EPC; ++j) {
            b = share[h * EPC + j] > share[b] ? h * EPC + j : b;
        }
        if (share[b] == 0) {
            break;
        }
        uint32_t kr = NO_EXPERT, kn = NO_EXPERT;
        for (uint32_t k = 0; k < NC; ++k) {
            if (k == h || used[k]) {
                continue;
            }
            if (st[SHOLD + k] == b && st[SREADY + k] <= g + NCALL && (kr == NO_EXPERT || T[k] < T[kr])) {
                kr = k;
            }
            if (!fail[k] && st[SREADY + k] <= g && nnew < MAXNEW && (kn == NO_EXPERT || T[k] < T[kn])) {
                kn = k;
            }
        }
        uint32_t k = NO_EXPERT;
        bool fresh = false;
        if (kr != NO_EXPERT && (kn == NO_EXPERT || T[kr] <= T[kn] + SLACK)) {
            k = kr;
        } else if (kn != NO_EXPERT) {
            k = kn;
            fresh = st[SHOLD + kn] != b;
        } else {
            break;
        }
        const int32_t gap = static_cast<int32_t>(T[h]) - static_cast<int32_t>(T[k]) - static_cast<int32_t>(FIX);
        uint32_t x = gap > 0 ? static_cast<uint32_t>(gap) >> 1 : 0u;
        x = x < share[b] ? x : share[b];
        if (x < MINX) {
            break;
        }
        if (fresh) {
            const uint32_t last = alloc(h, k, b % EPC);
            if (last == 0) {
                fail[k] = 1;
                ++ndrop;
                continue;
            }
            st[SHOLD + k] = b;
            st[SREADY + k] = last + 2;
            ++nnew;
        } else {
            ++nreuse;
        }
        used[k] = 1;
        st[SRECV + k] = b;
        st[SRX + k] = x;
        ++nr[b];
        ++nrec;
        share[b] -= x;
        T[h] -= x;
        T[k] += x + FIX;
    }
    const uint32_t t3 = get_timestamp_32b();
    uint32_t tmax1 = 0;
    for (uint32_t c = 0; c < NC; ++c) {
        tmax1 = T[c] > tmax1 ? T[c] : tmax1;
    }
    // the slot map (router_rows.hpp slot_of, the uneven split): own slots, copy words, copies (receivers by chip)
    uint32_t first = 0;  // (l_map holds the empty map: own slots, no copies)
    for (uint32_t b = 0; b < E; ++b) {
        if (nr[b] == 0) {
            continue;
        }
        map[tile_index(E + b)] = f32_bits(nr[b] | first << 6);
        const uint32_t tot = load[b];
        uint32_t acc = share[b], i = 0;
        for (uint32_t k = 0; k < NC; ++k) {
            if (st[SRECV + k] != b) {
                continue;
            }
            uint32_t lb = (acc * 32768u + tot / 2) / tot;
            lb = lb < 1 ? 1 : (lb > 32767 ? 32767 : lb);
            map[tile_index(2 * E + first + i)] = f32_bits((k * SPC + EPC) | lb << 9);
            acc += st[SRX + k];
            ++i;
        }
        first += nr[b];
    }
    write_page(rb, 4096, MAPPAGE, 0, l_map, 4096);
    st[1] = g;
    st[2] = nrec;
    st[3] = nreuse;
    st[4] = nnew;
    st[5] = ndrop;
    st[6] = tmax0;
    st[7] = tmax1;
    st[8] = tsum / NC;
    st[9] = 1;
    st[10] = t1 - t0;  // (diagnostic cycle stamps: state reads, the counts' wait + rows, the plan, the map)
    st[11] = t2 - t1;
    st[12] = t3 - t2;
    st[13] = get_timestamp_32b() - t3;
    G[0] = g + 1;
    G[2] = G[2] + nnew;
    G[3] = G[3] + pieces;
    G[4] = G[4] + ndrop;
    write_page(sta, SW * 4, 0, 0, l_st, SW * 4);
    write_page(gst, GW * 4, 0, 0, l_g, GW * 4);
    noc_async_write_barrier();
}
