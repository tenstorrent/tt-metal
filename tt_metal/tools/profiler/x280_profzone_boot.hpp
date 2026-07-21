// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host-side bring-up for the X280 `profzone` drainer firmware.
//
// This is the SINGLE SOURCE OF TRUTH for the host<->profzone LIM contract (mailbox/SRCLUT/STAGECTL
// addresses + the P_NONCE mode-bit layout) and the boot sequence. Both the standalone benchmark example
// (test_x280_realprof) and the production RealtimeProfilerManager call boot_profzone() so the two can never
// drift apart -- the drift (manager still writing the old 4-word param layout with no SRCLUT while profzone
// moved to the 2-word linearized stream) is exactly what this module exists to prevent.
//
// profzone is an ACTIVE firmware: the X280 L2CPU leaves reset ONCE (-> resident idle FW), then an indirect
// JUMP hands off to profzone, which OWNS the X280 and drains continuously until P_STOP. For production it is
// booted once at MeshDevice bring-up and stays resident for the whole session (P_STOP only at teardown), so
// the idle FW is never re-entered.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tools/profiler/x280_driver.hpp"
#include "prof_packet.h"  // PP_STICKY_SRC, pp_word0 (2-word + split-sticky format contract)

namespace tt::tt_metal::profiler {

// ---- host<->profzone LIM contract (MUST match tools/x280_bm/src/profzone.c) ----
inline constexpr uint64_t kProfzoneMboxParams = 0x08011000ULL;   // params (P_* at offsets below)
inline constexpr uint64_t kProfzoneMboxResults = 0x08011040ULL;  // per-hart result slot (h*0x40)
inline constexpr uint64_t kProfzoneMboxCoords = 0x08011200ULL;   // num_cores x {u32 noc_x, u32 noc_y}
inline constexpr uint64_t kProfzoneSrcLutBase = 0x08014000ULL;   // num_cores*NRISC x 8B STICKY-SRC packets
inline constexpr uint64_t kProfzoneHackedBase = 0x08017200ULL;   // per-hart host-writes ack (stride 0x40)
inline constexpr uint64_t kProfzoneStageCtl = 0x08018000ULL;     // per-reader LIM SPSC prod/cons pointers
inline constexpr uint64_t kProfzoneNRisc = 5;

// params (byte offsets into kProfzoneMboxParams)
inline constexpr uint64_t kPOffPcieEnc = 0x00;
inline constexpr uint64_t kPOffHostBase = 0x08;
inline constexpr uint64_t kPOffProfL1 = 0x10;
inline constexpr uint64_t kPOffNumCores = 0x18;
inline constexpr uint64_t kPOffHringWords = 0x20;
inline constexpr uint64_t kPOffStop = 0x28;   // P_STOP: host sets 1 to end the drain (teardown only)
inline constexpr uint64_t kPOffNonce = 0x30;  // mode bits (below)
inline constexpr uint64_t kPOffNRead = 0x38;  // reader count (split) / drain-hart count (direct)

// P_NONCE mode-bit layout
inline constexpr uint64_t kNonceReadNoc = 0x1ULL;       // bit0: read over NoC1 (else NoC0)
inline constexpr uint64_t kNonceDirect = 0x100ULL;      // bit8: direct drain (no reader/relay split)
inline constexpr uint64_t kNonceSplitNoc = 0x200ULL;    // bit9: split reads across NoC0/NoC1 per hart
inline constexpr uint64_t kNonceWnoc1 = 0x800ULL;       // bit11: posted PCIe write over NoC1
inline constexpr uint64_t kNonceNodrain = 0x1000ULL;    // bit12
inline constexpr uint64_t kNonceFullread = 0x2000ULL;   // bit13: reader always drains a full buffer (bench)
inline constexpr uint64_t kNonceBulkcore = 0x4000ULL;   // bit14: one bulk NoC read per core
inline constexpr uint64_t kNonceDualrelay = 0x8000ULL;  // bit15: one relay hart per reader
inline constexpr uint64_t kNonceAdaptive = 0x10000ULL;  // bit16: per-core adaptive bulk-vs-per-risc switch

// X280 boot-phase word (from x280_boot.h) -- lets the host confirm profzone is a RESIDENT active FW.
inline constexpr uint64_t kX280BootPhaseAddr = 0x080160C0ULL;       // X280_BOOT_HANDSHAKE_BASE(0x08016000)+0xC0
inline constexpr uint64_t kX280PhaseRunningActive = 0x7E570001ULL;  // X280_BOOT_PHASE_RUNNING_ACTIVE_FW
inline constexpr uint64_t kX280PhaseReturnedIdle = 0x1D1E0002ULL;   // X280_BOOT_PHASE_RETURNED_TO_IDLE

inline uint64_t profzone_hacked_addr(uint64_t h) { return kProfzoneHackedBase + h * 0x40; }
inline uint64_t profzone_harthb_addr(uint64_t h) { return kProfzoneMboxResults + 0x100 + h * 8; }

// All inputs the host computes from the cluster/mesh and hands to the boot. Coords is the pre-built
// MBOX_COORDS payload (num_cores x {u32 noc_x, u32 noc_y}, TRANSLATED to the X280's NoC view).
struct ProfzoneBootCfg {
    std::vector<uint8_t> idle_fw;    // resident idle FW binary (bytes, e.g. read_file(path))
    std::vector<uint8_t> active_fw;  // profzone.bin binary (bytes)
    int pll_mhz = 1000;

    uint64_t pcie_enc = 0;     // PCIe tile NoC-enc for the posted D2H write
    uint64_t host_base = 0;    // host sysmem base the relay writes into
    uint64_t prof_l1 = 0;      // worker profiler-ring L1 base (per-core NoC window offset)
    uint64_t num_cores = 0;    // worker cores drained
    uint32_t hring_words = 0;  // per host-ring depth (words)
    uint64_t ndh = 0;          // # host rings (== reader count for dual-relay split)
    uint64_t nread = 2;        // reader harts (split) / unused (direct)
    uint64_t ndrain = 1;       // drain harts (direct mode)

    const uint8_t* coords = nullptr;  // MBOX_COORDS payload
    uint32_t coords_bytes = 0;

    // mode
    uint64_t read_noc = 0;
    bool direct = false;
    bool split_noc = false;
    bool wnoc1 = false;
    bool nodrain = false;
    bool fullread = false;
    bool bulkcore = false;
    bool dualrelay = true;
    bool adaptive = true;
};

namespace detail {
template <typename T>
inline void pz_pack(std::vector<uint8_t>& buf, size_t off, T val) {
    std::memcpy(buf.data() + off, &val, sizeof(T));
}
}  // namespace detail

// Boot profzone as a resident active FW. Does: ensure_idle -> zero HACKED/STAGECTL -> write coords -> write
// the STICKY-SRC lookup table -> write params -> zero heartbeats -> JUMP handoff -> verify every hart reached
// its work loop (heartbeat == 3). Returns true iff all harts are up; on false, `half_broken` says whether the
// idle FW itself failed to come up (needs `tt-smi -r`). Does NOT set P_STOP -> profzone stays resident.
inline bool boot_profzone(X280Driver& drv, const ProfzoneBootCfg& cfg, uint64_t& nharts_out, bool& half_broken) {
    using namespace std::chrono;
    half_broken = false;
    if (!drv.ensure_idle(cfg.idle_fw, cfg.pll_mhz, milliseconds(3000), half_broken)) {
        return false;
    }
    // zero HACKED (per host ring) + STAGECTL BEFORE boot -- no init race.
    {
        std::vector<uint8_t> z(512, 0);
        for (uint64_t h = 0; h < cfg.ndh; h++) {
            drv.write_block(z.data(), 8, profzone_hacked_addr(h));
        }
        drv.write_block(z.data(), 512, kProfzoneStageCtl);
    }
    if (cfg.coords && cfg.coords_bytes) {
        drv.write_block(cfg.coords, cfg.coords_bytes, kProfzoneMboxCoords);
    }
    // STICKY-SRC lookup table: lane L -> 8B packet the reader injects at each source switch.
    {
        const uint32_t nl = static_cast<uint32_t>(cfg.num_cores * kProfzoneNRisc);
        std::vector<uint8_t> lut(static_cast<size_t>(nl) * 8, 0);
        for (uint32_t L = 0; L < nl; L++) {
            detail::pz_pack<uint32_t>(lut, L * 8 + 0, pp_word0(PP_STICKY_SRC, L));
            detail::pz_pack<uint32_t>(lut, L * 8 + 4, L);
        }
        drv.write_block(lut.data(), static_cast<uint32_t>(lut.size()), kProfzoneSrcLutBase);
    }
    // params
    {
        std::vector<uint8_t> params(64, 0);
        detail::pz_pack<uint64_t>(params, kPOffPcieEnc, cfg.pcie_enc);
        detail::pz_pack<uint64_t>(params, kPOffHostBase, cfg.host_base);
        detail::pz_pack<uint64_t>(params, kPOffProfL1, cfg.prof_l1);
        detail::pz_pack<uint64_t>(params, kPOffNumCores, cfg.num_cores);
        detail::pz_pack<uint64_t>(params, kPOffHringWords, static_cast<uint64_t>(cfg.hring_words));
        detail::pz_pack<uint64_t>(params, kPOffStop, 0);  // resident: never stop at boot
        uint64_t nonce = cfg.read_noc | (cfg.direct ? kNonceDirect : 0) | (cfg.split_noc ? kNonceSplitNoc : 0) |
                         (cfg.wnoc1 ? kNonceWnoc1 : 0) | (cfg.nodrain ? kNonceNodrain : 0) |
                         (cfg.fullread ? kNonceFullread : 0) | (cfg.bulkcore ? kNonceBulkcore : 0) |
                         (cfg.dualrelay ? kNonceDualrelay : 0) | (cfg.adaptive ? kNonceAdaptive : 0);
        detail::pz_pack<uint64_t>(params, kPOffNonce, nonce);
        detail::pz_pack<uint64_t>(params, kPOffNRead, cfg.direct ? cfg.ndrain : cfg.nread);
        drv.write_block(params.data(), static_cast<uint32_t>(params.size()), kProfzoneMboxParams);
    }
    const uint64_t nrelay = cfg.dualrelay ? cfg.nread : 1;
    const uint64_t nharts = cfg.direct ? cfg.ndrain : (cfg.nread + nrelay);
    nharts_out = nharts;
    for (uint64_t h = 0; h < nharts; h++) {
        drv.lim_wr_u64(profzone_harthb_addr(h), 0);
    }
    if (!drv.handoff_to_active_fw(cfg.active_fw, milliseconds(3000))) {
        return false;
    }
    // verify every hart reached its work loop (heartbeat == 3) within 3s.
    auto deadline = steady_clock::now() + seconds(3);
    while (steady_clock::now() < deadline) {
        bool all = true;
        for (uint64_t h = 0; h < nharts; h++) {
            if (drv.lim_rd_u64(profzone_harthb_addr(h)) != 3) {
                all = false;
                break;
            }
        }
        if (all) {
            return true;
        }
        std::this_thread::sleep_for(milliseconds(2));
    }
    return false;
}

// Confirm profzone is still a resident active FW (RUNNING_ACTIVE_FW, not bounced to idle).
inline bool profzone_is_resident(X280Driver& drv) {
    return drv.lim_rd_u64(kX280BootPhaseAddr) == kX280PhaseRunningActive;
}

// Signal profzone to end its drain (teardown only -- breaks the drain loops on the next pass).
inline void profzone_stop(X280Driver& drv) { drv.lim_wr_u64(kProfzoneMboxParams + kPOffStop, 1); }

}  // namespace tt::tt_metal::profiler
