// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric ring microbench (weight-streaming shape), one RISC = one one-way lane D of a link: lane 0 (BRISC) sends to the
// next chip of the ring, lane 1 (NCRISC) to the previous one; both lanes of a link share the core. Every chip injects E
// packets a lane (its own weights), every packet travels N - 1 hops (forwarded N - 2 times), so every link direction
// carries (N - 1) E packets. A lane's packets land in an S-slot L1 ring on the next chip (the k-th packet a sender
// sends into slot k % S, a fused write + atomic inc of the receiver's arrival word); the receiver forwards or retires
// them in arrival order. Ring flow control: the sender's view of the receiver's freed slots. A lane's freed count goes
// upstream on the other lane of the same core (which sends upstream): written into the last 16 B of every packet that
// lane sends (piggy-backed; the receiving core copies it into a word for its sender), and as an explicit inline write
// when that lane has nothing else to send. Bubble rule: a forward needs 1 free downstream slot, an injection 2, so the
// ring cannot fill up and deadlock. A packet's last 16 B: forwards left, piggy-backed credit, origin, sequence.
// CT: 0 D, 1 S, 2 PB, 3 NO (own buffers), 4 E, 5 N, 6 FE (0: retire by the NoC sent counter; > 0 flush every FE), 7
// ring of lane 0 (L1), 8 ring
//     of lane 1, 9 own buffers of this lane, 10 headers of this lane (S + 1 x HS), 11 words (L1), 12 HS
// CT 13 MODE (bit 0 the connection's stateful send path, bit 1 posted NoC writes).
// RT: 0 mesh id, 1 destination chip id, 2 destination core x | y << 8, 3 origin id, 4 the other lane's core x | y << 8
//     (0: this core), then the fabric connection args.
// Words (16 B apart): 0 / 1 arrivals of lane 0 / 1, 2 / 3 piggy-backed credit for lane 0 / 1's sender, 4 / 5 explicit
// credit, 6 / 7 freed slots of lane 0 / 1, 8 / 9 done, 10 + 8 D .. stamps (first send, last send, sends, arrivals,
// cycles waiting for router space / downstream credit / idle, explicit credits sent).
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t D = get_compile_time_arg_val(0), S = get_compile_time_arg_val(1);
    constexpr uint32_t PB = get_compile_time_arg_val(2), NO = get_compile_time_arg_val(3);
    constexpr uint32_t E = get_compile_time_arg_val(4), N = get_compile_time_arg_val(5);
    constexpr uint32_t FE = get_compile_time_arg_val(6);
    constexpr uint32_t ring0 = get_compile_time_arg_val(7), ring1 = get_compile_time_arg_val(8);
    constexpr uint32_t own = get_compile_time_arg_val(9), hdrs = get_compile_time_arg_val(10);
    constexpr uint32_t W = get_compile_time_arg_val(11), HS = get_compile_time_arg_val(12);
    constexpr uint32_t MODE = get_compile_time_arg_val(13);  // (bit 0 stateful send path, bit 1 posted writes)
    constexpr bool STATEFUL = MODE & 1, POSTED = MODE & 2;
    constexpr uint32_t BMAX = get_compile_time_arg_val(14);  // (packets handled a pass at most)
    constexpr uint32_t myring = D ? ring1 : ring0, O = 1 - D;
    constexpr uint32_t ARR = 0, CPB = 2, CX = 4, FREED = 6, DONE = 8, ST = 10 + 8 * D;
    constexpr uint32_t TOT = (N - 1) * E;
    static_assert(FE <= S && sizeof(PACKET_HEADER_TYPE) <= HS);
    auto w = [](uint32_t i) -> volatile tt_l1_ptr uint32_t& {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(W + 16 * i);
    };

    const uint16_t mesh = get_arg_val<uint32_t>(0);
    const uint8_t dchip = get_arg_val<uint32_t>(1);
    const uint32_t dxy = get_arg_val<uint32_t>(2), origin = get_arg_val<uint32_t>(3);
    const uint32_t pxy = get_arg_val<uint32_t>(4);  // (the other lane's core: itself unless the lanes are split)
    const bool split = pxy != 0;
    size_t a = 5;
    uint32_t last_freed = 0;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    conn.open();
    // (a word the other lane's RISC reads: local, or written into the partner core when the lanes are split)
    auto put = [&](uint32_t i, uint32_t v) {
        w(i) = v;
        if (split) {
            noc_inline_dw_write(get_noc_addr(pxy & 0xFF, pxy >> 8, W + 16 * i), v);
            if constexpr (STATEFUL) {  // (the inline write used a command buffer the stateful send path keeps state in)
                conn.setup_stateful_send_cmd_bufs<POSTED>();
            }
        }
    };
    auto hdr = [](uint32_t s) { return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdrs + s * HS); };
    for (uint32_t s = 0; s < S; ++s) {
        (void)fabric_set_unicast_route(hdr(s), dchip, mesh);
        hdr(s)->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                get_noc_addr(dxy & 0xFF, dxy >> 8, myring + s * PB),
                get_noc_addr(dxy & 0xFF, dxy >> 8, W + 16 * (ARR + D)),
                1,
                false},
            PB);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hc = hdr(S);
    (void)fabric_set_unicast_route(hc, dchip, mesh);
    // (the explicit credit goes to the downstream chip's other-lane core: the same logical core as this chip's partner)
    const uint32_t cxy = split ? pxy : dxy;
    const uint64_t cx_dst = get_noc_addr(cxy & 0xFF, cxy >> 8, W + 16 * (CX + O));

    // Outstanding sends (FIFO of S): the NoC's issued-write count right after the send, the arrival prefix that is
    // free once it has left L1, whether it was an own buffer. FE 0: retire by the NoC's sent counter (no blocking
    // flush); FE > 0: a blocking flush every FE sends.
    constexpr uint32_t QN = 2 * S;
    uint32_t q_snap[QN], q_jb[QN], q_own[QN];
    uint32_t qh = 0, qt = 0;
    uint32_t j = 0, k = 0, own_sent = 0, own_ret = 0, last_x = 0, t_first = 0, cmax = 0;
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
            put(FREED + D, fr);
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
        if (qt - qh == QN) {  // (the queue is full: wait for its oldest send to leave L1)
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
        const uint32_t cpb = w(CPB + D), cx = w(CX + D);
        uint32_t free_ds = S - (k - (cpb > cx ? cpb : cx));
        const uint32_t avail_all = w(ARR + D), avail = avail_all - j > BMAX ? j + BMAX : avail_all;
        const uint32_t fo_now = w(FREED + O);
        if (j < avail) {  // (the newest landed packet carries the freshest credit for the other lane's sender)
            const uint32_t c =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(myring + ((avail - 1) % S) * PB + PB - 16)[1];
            if (c > cmax) {  // (monotonic: packets carry non-decreasing counts)
                cmax = c;
                put(CPB + O, c);
            }
        }
        while (j < avail) {  // (every landed packet in one pass)
            const uint32_t slot = myring + (j % S) * PB;
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + PB - 16);
            const uint32_t fl = m[0];
            if (fl == 0) {  // (its last hop: retire)
                ++j;
                prog = true;
            } else if (free_ds >= 1) {
                m[0] = fl - 1;
                m[1] = fo_now;
                send(slot, j, 0);
                --free_ds;
                ++j;
                prog = true;
            } else {
                blocked = true;
                break;
            }
        }
        if (j == avail_all && own_sent < E && own_sent - own_ret < NO) {
            if (free_ds >= 2) {
                const uint32_t src = own + (own_sent % NO) * PB;
                volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + PB - 16);
                m[0] = N - 2;
                m[1] = fo_now;
                m[2] = origin;
                m[3] = own_sent;
                send(src, j, 1);
                ++own_sent;
                prog = true;
            } else {
                blocked = true;
            }
        }
        if constexpr (FE > 0) {
            if (qt - qh >= FE || (!prog && qh != qt)) {
                retire(true);
            } else if (qh == qt && j != last_freed) {
                put(FREED + D, j);
                last_freed = j;
            }
        } else {
            retire(false);
        }
        const uint32_t now = wall();
        if (!prog) {
            (blocked ? t_cred : t_idle) += now - tl;
            const uint32_t fo = w(FREED + O);
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
                if (w(DONE + D) == 0) {
                    w(ST + 1) = wall();
                    put(DONE + D, 1);
                }
                if (w(DONE + O) && w(FREED + O) == last_x) {
                    break;
                }
            }
        }
        tl = now;
    }
    w(ST + 0) = t_first;
    w(ST + 2) = k;
    w(ST + 3) = j;
    w(ST + 4) = t_router;
    w(ST + 5) = t_cred;
    w(ST + 6) = t_idle;
    w(ST + 7) = n_x;
    noc_async_full_barrier();
    conn.close();
}
