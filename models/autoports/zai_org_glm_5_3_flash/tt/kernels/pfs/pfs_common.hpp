// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream (tt/pf_stream.py): the PMK PfStreamPhase body (prefill_megakernel.md §4c) — weights streamed once
// by bank streamers and multicast to the activation-stationary members. Persistent sync words on each STREAMER (one
// 16-B word each, zeroed once by the host, MONOTONIC across launches unless noted).
// no include guard: the PMK composite includes each phase body inside its own namespace (one copy per phase)
#include <cstdint>

#if !defined(MK_PHASED)  // standalone program (the unit); the PMK composite defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;  // every RISC (the member's TRISC reads RT_PROBE under PFS_PROBE)
#endif

namespace pfs {
// ONE compile-time arg list per phase, shared by every kernel of the phase (the PMK composite gives a phase's CT list
// to every RISC of every role that runs it): get_compile_time_arg_val(PH_CT0 + CT_x).
constexpr uint32_t CT_NBK = 0;    // K-blocks per N-block (even)
constexpr uint32_t CT_KB = 1;     // K rows per block
constexpr uint32_t CT_NB = 2;     // N-block columns (<= 8: one full-sync fp32 DST)
constexpr uint32_t CT_NW = 3;     // stream columns (NW / NB N-blocks per stream)
constexpr uint32_t CT_TB = 4;     // weight tile bytes
constexpr uint32_t CT_P = 5;      // the members' PARK slots (<= 4: READY parity words)
constexpr uint32_t CT_NDEST = 6;  // members (multicast receivers)
constexpr uint32_t CT_RING = 7;   // the streamers' ring CB
constexpr uint32_t CT_KT = 8;     // K tiles (the stationary in0 row)
constexpr uint32_t CT_X = 9;      // the members' stationary in0 CB (XS 35 / IN0 40)
constexpr uint32_t CT_PARK = 10;  // the members' PARK CB (29)
constexpr uint32_t CT_V0 = 11;    // the members' VALID semaphores (lane 0 / 1)
constexpr uint32_t CT_V1 = 12;
constexpr uint32_t CT_NBLK = 13;   // N-blocks of the phase (N / 32 / NB)
constexpr uint32_t CT_OUT = 14;    // the members' output CB (OUT 42)
constexpr uint32_t CT_QBASE = 15;  // the phase's first global block in the launch (even): PARK slot and VALID continue
constexpr uint32_t CT_SLOT = 16;   // the phase's sync slot on the streamers (SYNC_WORDS words each)
constexpr uint32_t CT_WAITSEM = 17;  // H2 outside a PMK composite: the reader waits semaphore WAITSEM >= WAITN before
constexpr uint32_t CT_WAITN = 18;    //   handing the stationary block to the compute (WAITN 0 = no wait)
constexpr uint32_t CT_EPI =
    19;  // the member epilogue: 0 = bf16 tiles to CB_OUT (the writer stores them); 1 = SwiGLU
         //   (N-block = (gate, up) tile pairs: SiLU(min(g, L)) * clamp(u, -L, L) in fp32 DST) ->
         //   NB / 2 act tiles into CB_OUT = the NEXT phase's in0 CB (no writer work); 2 = the
         //   N-block packed UNTILIZED (32 rows x NB tiles ROW_MAJOR) -> the writer stores rows
constexpr uint32_t CT_LIMIT = 20;  // EPI 1: the SwiGLU limit L, fp32 bits
constexpr uint32_t CT_XSRC =
    21;                          // 1: the stationary in0 was pushed by the previous phase's compute (EPI 1): the reader
                                 //   neither waits nor pushes it
constexpr uint32_t CT_SPC = 22;  // streams per streamer core (pmk/proj: a core serves its rectangle's SPC streams in
                                 //   order; 1 = 7a's one stream per core with the TOKEN chain across cores)
constexpr uint32_t CT_RT = 23;   // row tiles per member (pmk/proj lever 1: 2 on 96 members; RT x NB fp32 tiles in DST)
constexpr uint32_t CT_NTT = 24;  // the output store's total N tiles (the rectangle writes its columns COL0 .. of it)
constexpr uint32_t CT_ACC = 25;  // TensorAccessorArgs of the output store, then (PFS_UNIT) of x
// ONE runtime arg list per core per phase (the composite appends it for every RISC of the core): ph_rt0 + RT_x.
constexpr uint32_t RT_SYNC = 0;  // the streamers' sync base (every core)
// streamer cores
constexpr uint32_t RT_ACTIVE = 1;  // 0 = this streamer has no stream in the phase (the body returns at once)
constexpr uint32_t RT_ST = 2, RT_L = 3, RT_NST = 4, RT_W = 5, RT_BANK = 6, RT_M0 = 7, RT_VC = 8;
constexpr uint32_t RT_LX = 9, RT_LY = 10, RT_HX = 11, RT_HY = 12, RT_NX = 13, RT_NY = 14;
constexpr uint32_t RT_NOC = 15;  // the NoC this lane multicasts on (7a: lane L on NoC L; pmk/proj placements may put
                                 //   both lanes on the NoC whose dimension order reaches the rectangle without a wrap)
constexpr uint32_t RT_SB = 16;   // SPC x (bank, m0) of the core's streams (stream RT_ST + s, in order); RT_BANK /
                                 //   RT_M0 hold the first pair too
// member cores
constexpr uint32_t RT_FIRST = 1;    // member 0: publishes the PARK / VALID addresses
constexpr uint32_t RT_NS = 2;       // streamers of the phase
constexpr uint32_t RT_PROBE = 3;    // the member's probe shard (0 = none)
constexpr uint32_t RT_OUTADDR = 4;  // the output store's address
constexpr uint32_t RT_ROW = 5;      // the member's row tile
constexpr uint32_t RT_XADDR = 6;    // (PFS_UNIT) x's address
constexpr uint32_t RT_SIGX = 7;     // after the member's whole output row is stored: +1 at NoC (SIGX, SIGY) L1 SIGA
constexpr uint32_t RT_SIGY = 8;     //   (SIGA 0 = none; e.g. the RS lane's LREADY + k / the reducer's OWNRDY + i: the
constexpr uint32_t RT_SIGA = 9;     //   H5 producer contract of tt/fabric_stream.FabricStreamRSR ready=True)
constexpr uint32_t RT_COL0 = 10;    // the member's first output column tile (its rectangle's column group)
constexpr uint32_t RT_SXY = 11;     // NS x streamer NoC (x, y), streamer 2 (st / SPC) + lane
constexpr uint32_t WORD = 16;
constexpr uint32_t READY = 0;     // +1 per member per PARK slot reserved for one of THIS streamer's EVEN blocks i
constexpr uint32_t READY1 = 13;   //   ... ODD blocks: with credits ahead (<= P blocks past the member's next landing,
                                  //   P <= 4) a member can credit block i + 1 but never i + 2 before block i was sent,
                                  //   so each parity word only holds credits of blocks <= the awaited one (v2b's single
                                  //   counter mixed blocks 0 and 2: d0 10:57Z watcher stop)
constexpr uint32_t TOKEN = 1;     // launches in which the previous streamer of the chain finished its multicasts
constexpr uint32_t GEN = 2;       // launches done by this streamer
constexpr uint32_t PARKADDR = 3;  // the members' PARK (CB 29) base (member 0 writes it at launch start)
constexpr uint32_t VALIDADDR0 = 4;  // the members' VALID semaphore of NoC lane 0 / 1 (member 0 writes them)
constexpr uint32_t VALIDADDR1 = 5;
constexpr uint32_t VAL0 = 6;  // this streamer's multicast source words for the VALID sets (lane 0 / 1)
constexpr uint32_t VAL1 = 7;
// PFS_PROBE: this streamer's cycles (last launch) waiting for the TOKEN / its DRAM reads / READY / NoC flushes; span
constexpr uint32_t P_TOKEN = 8;
constexpr uint32_t P_READ = 9;
constexpr uint32_t P_READY = 10;
constexpr uint32_t P_FLUSH = 11;
constexpr uint32_t P_SPAN = 12;
// the members' probe shard (PFS_PROBE): cycles waiting for a free PARK slot (compute-bound) / for VALID (stream-bound)
constexpr uint32_t M_RESERVE = 0;
constexpr uint32_t M_VALID = 1;
constexpr uint32_t M_SPAN = 2;
constexpr uint32_t M_XFILL = 3;     // reader: cycles of the unit's x fill (PFS_UNIT) / the H2 wait
constexpr uint32_t M_WSTALL = 4;    // writer: cycles in noc_async_writes_flushed + the final write barrier (the store)
constexpr uint32_t M_THROTTLE = 5;  // compute (MATH): weight blocks whose matmul saw the Blackhole FW throttle flag odd
constexpr uint32_t M_CSPAN = 6;     // compute (MATH): cycles from the first PARK wait to the last block popped
constexpr uint32_t M_WORDS = 8;     // the member probe shard: M_WORDS x WORD bytes
constexpr uint32_t SYNC_WORDS = 16;

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
inline volatile tt_l1_ptr uint32_t* word(uint32_t base, uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * WORD);
}
inline void wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    WAYPOINT("PFSW");
    do {
        invalidate_l1_cache();
    } while (static_cast<int32_t>(*p - v) < 0);
    WAYPOINT("PFSD");
}
#endif
}  // namespace pfs
