// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric ring microbench, RISC split (fr2_*): a core a lane, the sender RISC. The processor RISC (fr2_proc.cpp) reads
// every landed packet and marks it forward or retire (word READY: arrivals examined, act[] per slot); this RISC only
// sends: the marked forwards from their ring slots, its own packets (bubble rule: a forward needs 1 free downstream
// slot, an injection 2), and an explicit credit for the other lane when it has nothing else to send. It frees ring
// slots by the NoC's sent-write counter (word FREED, the freed prefix; the processor passes it to the partner core).
// A sent packet's last 16 B: forwards left, the other lane's freed prefix (piggy-backed credit), origin, sequence.
// CT: 0 S, 1 PB, 2 NO, 3 E, 4 N, 5 ring (L1), 6 own buffers, 7 headers (S + 1 x HS), 8 words, 9 HS, 10 MODE (bit 0
//     stateful send path, bit 1 posted), 11 BMAX (forwards a pass), 12 act (L1, S words)
// RT: 0 mesh id, 1 destination chip id, 2 destination core x | y << 8 (the next chip's core of this lane), 3 origin,
//     4 partner core x | y << 8 (the other lane's core), then the fabric connection args.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "fr2_words.hpp"
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), PB = get_compile_time_arg_val(1);
    constexpr uint32_t NO = get_compile_time_arg_val(2), E = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4), ring = get_compile_time_arg_val(5);
    constexpr uint32_t own = get_compile_time_arg_val(6), hdrs = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8), HS = get_compile_time_arg_val(9);
    constexpr uint32_t MODE = get_compile_time_arg_val(10), BMAX = get_compile_time_arg_val(11);
    constexpr uint32_t act = get_compile_time_arg_val(12);
    constexpr bool STATEFUL = MODE & 1, POSTED = MODE & 2;
    constexpr uint32_t TOT = (N - 1) * E, QN = 2 * S;
    static_assert(sizeof(PACKET_HEADER_TYPE) <= HS);
    using namespace fr2;
    auto w = [](uint32_t i) -> volatile tt_l1_ptr uint32_t& { return word(W, i); };

    const uint16_t mesh = get_arg_val<uint32_t>(0);
    const uint8_t dchip = get_arg_val<uint32_t>(1);
    const uint32_t dxy = get_arg_val<uint32_t>(2), origin = get_arg_val<uint32_t>(3), pxy = get_arg_val<uint32_t>(4);
    size_t a = 5;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    conn.open();
    auto hdr = [](uint32_t s) { return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdrs + s * HS); };
    for (uint32_t s = 0; s < S; ++s) {
        (void)fabric_set_unicast_route(hdr(s), dchip, mesh);
        hdr(s)->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                get_noc_addr(dxy & 0xFF, dxy >> 8, ring + s * PB),
                get_noc_addr(dxy & 0xFF, dxy >> 8, W + 16 * ARR),
                1,
                false},
            PB);
    }
    // (the explicit credit: to the next chip's other-lane core, the same logical core as this chip's partner)
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hc = hdr(S);
    (void)fabric_set_unicast_route(hc, dchip, mesh);
    const uint64_t cx_dst = get_noc_addr(pxy & 0xFF, pxy >> 8, W + 16 * CX);

    uint32_t q_snap[QN], q_jb[QN], q_own[QN];
    uint32_t qh = 0, qt = 0, j = 0, k = 0, own_sent = 0, own_ret = 0, last_x = 0, t_first = 0, last_freed = 0;
    uint32_t t_router = 0, t_cred = 0, t_idle = 0, n_x = 0;
    auto retire = [&](bool all) {
        if (all) {
            noc_async_writes_flushed();
        }
        const uint32_t sent =
            NOC_STATUS_READ_REG(noc_index, POSTED ? NIU_MST_POSTED_WR_REQ_SENT : NIU_MST_NONPOSTED_WR_REQ_SENT);
        while (qh != qt && (all || static_cast<int32_t>(sent - q_snap[qh % QN]) >= 0)) {
            own_ret += q_own[qh % QN];
            ++qh;
        }
        const uint32_t fr = qh == qt ? j : q_jb[qh % QN];
        if (fr != last_freed) {
            w(FREED) = fr;
            last_freed = fr;
        }
    };
    auto send = [&](uint32_t src, uint32_t jb, uint32_t is_own) {
        if (!conn.edm_has_space_for_packet()) {
            const uint32_t s0 = wall();
            conn.wait_for_empty_write_slot();
            t_router += wall() - s0;
        }
        if constexpr (STATEFUL) {
            conn.send_current_slot_stateful_non_blocking<POSTED>(src, PB, reinterpret_cast<uint32_t>(hdr(k % S)));
        } else {
            conn.send_current_slot_non_blocking<POSTED>(src, PB, reinterpret_cast<uint32_t>(hdr(k % S)));
        }
        q_snap[qt % QN] = POSTED ? noc_posted_writes_num_issued[noc_index] : noc_nonposted_writes_num_issued[noc_index];
        q_jb[qt % QN] = jb;
        q_own[qt % QN] = is_own;
        ++qt;
        if (qt - qh == QN) {
            retire(true);
        }
        if (k == 0) {
            t_first = wall();
        }
        ++k;
    };
    if constexpr (STATEFUL) {
        conn.setup_stateful_send_cmd_bufs<POSTED>();
    }
    uint32_t tl = wall();
    while (true) {
        invalidate_l1_cache();
        bool prog = false, blocked = false;
        const uint32_t cpb = w(CPB), cx = w(CX);
        uint32_t free_ds = S - (k - (cpb > cx ? cpb : cx));
        const uint32_t ready = w(READY), fo = w(FREED_O);
        for (uint32_t n = 0; j < ready && n < BMAX; ++n) {
            if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act)[j % S] == ACT_RETIRE) {
                ++j;
                prog = true;
            } else if (free_ds >= 1) {
                const uint32_t slot = ring + (j % S) * PB;
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + PB - 16)[1] = fo;
                send(slot, j, 0);
                --free_ds;
                ++j;
                prog = true;
            } else {
                blocked = true;
                break;
            }
        }
        if (j == ready && own_sent < E && own_sent - own_ret < NO) {
            if (free_ds >= 2) {
                const uint32_t src = own + (own_sent % NO) * PB;
                volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + PB - 16);
                m[0] = N - 2;
                m[1] = fo;
                m[2] = origin;
                m[3] = own_sent;
                send(src, j, 1);
                ++own_sent;
                prog = true;
            } else {
                blocked = true;
            }
        }
        retire(false);
        const uint32_t now = wall();
        if (!prog) {
            (blocked ? t_cred : t_idle) += now - tl;
            if (fo != last_x && conn.edm_has_space_for_packet()) {  // (the other lane's credit, nothing to carry it)
                noc_async_writes_flushed();
                hc->to_noc_unicast_inline_write(tt::tt_fabric::NocUnicastInlineWriteCommandHeader{cx_dst, fo});
                conn.send_payload_flush_non_blocking_from_address(
                    reinterpret_cast<uint32_t>(hc), sizeof(PACKET_HEADER_TYPE));
                last_x = fo;
                ++n_x;
                if constexpr (STATEFUL) {
                    noc_async_writes_flushed();
                    conn.setup_stateful_send_cmd_bufs<POSTED>();
                }
            }
            if (k == TOT && j == TOT && qh == qt) {
                if (w(DONE) == 0) {
                    w(ST_T + 1) = wall();
                    w(DONE) = 1;
                }
                if (w(DONE_O) && w(FREED_O) == last_x) {
                    break;
                }
            }
        }
        tl = now;
    }
    w(ST_T + 0) = t_first;
    w(ST_T + 2) = k;
    w(ST_T + 3) = j;
    w(ST_T + 4) = t_router;
    w(ST_T + 5) = t_cred;
    w(ST_T + 6) = t_idle;
    w(ST_T + 7) = n_x;
    w(TDONE) = 1;
    noc_async_full_barrier();
    conn.close();
}
