// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_boot.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal::experimental::x280 {

namespace {

// ---------------------------------------------------------------------------
// X280 (L2CPU) boot constants — mirror x280/host/loader.py and
// x280/host/clock.py. All AXI registers (reset unit at 0x80030000, PLL4 at
// 0x80020500) live in the ARC NOC XBAR region and are reachable via NOC
// writes to the ARC tile at noc0 (8, 0) on Blackhole. These are
// region-agnostic — LIM- and DRAM-mode boots both use the same ARC reset
// register, PLL4, and L2CPU tile NOC paths; only the firmware load address
// and sentinel layout differ (see BootProfile below).
// Source: tt-metal/.../umd/device/api/umd/device/arch/blackhole_implementation.hpp
//         ARC_NOC_XBAR_ADDRESS_START = 0x80000000, ARC_CORES_NOC0 = {{8, 0}}.
// ---------------------------------------------------------------------------

constexpr CoreCoord kArcTileNoc0{8, 0};
constexpr uint64_t kArcResetUnitBase = 0x80030000ULL;
constexpr uint64_t kArcL2CpuResetReg = kArcResetUnitBase + 0x14ULL;
constexpr uint64_t kArcPll4Base = 0x80020500ULL;
constexpr uint64_t kArcPllCntl1Addr = kArcPll4Base + 0x04ULL;
constexpr uint64_t kArcPllCntl5Addr = kArcPll4Base + 0x14ULL;

constexpr uint64_t kL2CpuRegBase = 0xFFFFF7FEFFF10000ULL;

// physical L2CPU index -> NOC0 (x, y). Matches loader.L2CPU_TILE_MAPPING; the
// physical index is also what the ARC reset register uses for bit positions
// (bit 4+N = L2CPU N).
constexpr std::array<CoreCoord, 4> kL2CpuTilesByPhysicalIndex = {
    CoreCoord{8, 3},  // physical 0
    CoreCoord{8, 9},  // physical 1
    CoreCoord{8, 5},  // physical 2
    CoreCoord{8, 7},  // physical 3
};

constexpr uint32_t kDefaultX280PllMhz = 1000;
// Reset-release frequency. Mirrors tt-bh-linux/boot.py::reset_x280, which
// slows PLL4 to 200 MHz immediately before clearing the reset bit and
// ramps back up afterwards. The L2CPU PLL has marginal start-up at high
// frequencies; releasing reset at 200 MHz and then ramping (one count at
// a time with 1ms between writes — see set_l2cpu_pll) is the
// SiFive-recommended sequence.
constexpr uint32_t kX280ResetReleasePllMhz = 200;
constexpr std::chrono::milliseconds kX280SentinelTimeout{5000};

// L2CPU L3-cache-controller config block, NOC-mapped on the L2CPU tile.
// 0x02010000 is the L3 Config base (used as a health probe in
// x280/host/loader.py::check_chip_health). Writing 0xf to +0x08 ungates
// the L2CPU's cache-controller clock domain. Kept here for diagnostic
// readback only; doing the write itself wedges already-running L2CPUs
// (see commentary in boot_idle()).
constexpr uint64_t kL2CpuL3ClockGateAddr = 0x02010008ULL;

// ---------------------------------------------------------------------------
// Per-region boot profile. Both regions go through the same Phase 1/2/3
// sequence; only the load address, sentinel address+value, trap CSR
// addresses, and the default firmware file basename differ.
//
// Lim profile mirrors `firmware-lim-idle.bin` (built from src/lim_idle.c
// linked with ld/x280.ld). DRAM profile mirrors `firmware-idle.bin`
// (built from src/idle.c linked with ld/x280-dram.ld) and the pyluwen
// boot reference x280/host/boot_idle_x280.py.
//
// Trap CSR addresses below are taken from `riscv64-unknown-elf-nm` on the
// corresponding .elf — entry.S's _trap_normal writes mcause/mepc/mtval to
// these labels, then CEASEs the hart. The DRAM-linked firmware places its
// BSS (and therefore the trap CSR readback area) in DRAM, NOT in LIM as
// the older x280/host/loader.py constants suggest; the LIM/DRAM split here
// reflects the *current* on-tree firmware (verified at plan time).
// ---------------------------------------------------------------------------

struct BootProfile {
    BootRegion region;
    uint64_t load_addr;  // also reset-vector entry target
    uint64_t sentinel_addr;
    uint64_t sentinel_value;
    uint64_t trap_mcause_addr;
    uint64_t trap_mepc_addr;
    uint64_t trap_mtval_addr;
    const char* default_fw_basename;
    const char* region_name;  // for log messages
};

constexpr BootProfile kLimProfile{
    BootRegion::Lim,
    0x08000000ULL,
    0x08100000ULL,
    0xDEADBEEFCAFEBABEULL,
    0x08000138ULL,
    0x08000140ULL,
    0x08000148ULL,
    "firmware-lim-idle.bin",
    "LIM",
};

constexpr BootProfile kDramProfile{
    BootRegion::Dram,
    0x400030000000ULL,                // DDR_BASE (x280.h)
    0x400030000000ULL + 0x5D8000ULL,  // DDR_BASE + IDLE_SENTINEL_OFF (idle.c)
    0xCAFEF00D00000099ULL,            // IDLE_SENTINEL_VAL (idle.c)
    0x4000300001D0ULL,
    0x4000300001D8ULL,
    0x4000300001E0ULL,  // nm firmware-idle.elf
    "firmware-idle.bin",
    "DRAM",
};

const BootProfile& profile_for(BootRegion region) { return region == BootRegion::Lim ? kLimProfile : kDramProfile; }

// PLL solutions ported verbatim from x280/host/clock.py: freq_mhz ->
// (fbdiv, postdiv0..3). The L2CPU PLL programming sequence is "ramp
// post-dividers up first, then step fbdiv, then ramp post-dividers down"
// to avoid out-of-range intermediate frequencies.
struct PllSolution {
    uint32_t freq_mhz;
    uint16_t fbdiv;
    std::array<uint8_t, 4> postdivs;
};
constexpr std::array<PllSolution, 4> kPllSolutions = {{
    {15, 120, {{99, 99, 99, 99}}},
    {200, 128, {{15, 15, 15, 15}}},
    {1000, 80, {{1, 1, 1, 1}}},
    {1750, 140, {{1, 1, 1, 1}}},
}};

// PLLCNTL1 layout (clock.py): refdiv[7:0], postdiv[15:8], fbdiv[31:16].
struct PllCntl1 {
    uint8_t refdiv;
    uint8_t postdiv;
    uint16_t fbdiv;
};
static_assert(sizeof(PllCntl1) == 4, "PLLCNTL1 must be 4 bytes");

// PLLCNTL5 layout (clock.py): postdiv[0..3] as 4 bytes.
struct PllCntl5 {
    std::array<uint8_t, 4> postdiv;
};
static_assert(sizeof(PllCntl5) == 4, "PLLCNTL5 must be 4 bytes");

// ---------------------------------------------------------------------------
// Helpers. These mirror what used to live as private members on
// RiscFirmwareInitializer; cluster_ becomes a Cluster& parameter so they
// stay decoupled from any one owner.
// ---------------------------------------------------------------------------

std::filesystem::path default_firmware_path(BootRegion region) {
    // Resolved relative to TT_METAL_RUNTIME_ROOT / repo root. Same scheme as
    // jit_build: tt-metal owns the artifact under runtime/hw/firmware/...
    // Per-region filename comes from the profile so adding a new BootRegion
    // is a single-line change here.
    const std::string& root = MetalContext::instance().rtoptions().get_root_dir();
    if (root.empty()) {
        return {};
    }
    std::filesystem::path p = std::filesystem::path(root) / "runtime" / "hw" / "firmware" / "blackhole" / "x280" /
                              profile_for(region).default_fw_basename;
    if (!std::filesystem::exists(p)) {
        return {};
    }
    return p;
}

std::vector<size_t> get_enabled_l2cpu_indices(Cluster& cluster, tt::ChipId device_id) {
    // SocDescriptor::get_cores(L2CPU, NOC0) returns NOC0 coords of all
    // non-harvested L2CPUs. We map each back to its physical index (0..3)
    // using the loader.py L2CPU_TILE_MAPPING ordering so the result lines
    // up with the ARC reset register bit positions.
    std::vector<size_t> result;
    result.reserve(4);
    const auto& soc_desc = cluster.get_soc_desc(device_id);
    const auto enabled = soc_desc.get_cores(CoreType::L2CPU, CoordSystem::NOC0);
    for (const auto& core : enabled) {
        for (size_t phys = 0; phys < kL2CpuTilesByPhysicalIndex.size(); phys++) {
            const auto& tile = kL2CpuTilesByPhysicalIndex[phys];
            if (core.x == tile.x && core.y == tile.y) {
                result.push_back(phys);
                break;
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void assert_l2cpu_reset(Cluster& cluster, tt::ChipId device_id, size_t l2cpu_index) {
    // Minimal: clear bit (l2cpu_index + 4) in the ARC L2CPU reset register
    // and nothing else.
    //
    // The L3 cache-controller clock-gate ungate (0x02010008 = 0xf on the
    // L2CPU tile) is intentionally NOT done here. Doing it on a *running*
    // L2CPU (e.g. one another consumer left active on chip 0) wedges the
    // core. Callers that have just put the core into reset and are about
    // to boot it would have to do the ungate themselves — see boot_idle()
    // for the diagnostic readback (the write itself is currently disabled).
    TT_ASSERT(l2cpu_index < 4, "l2cpu_index out of range: {}", l2cpu_index);
    uint32_t val = 0;
    cluster.read_reg(&val, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
    val &= ~(1u << (l2cpu_index + 4));
    cluster.write_reg(&val, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
}

void load_firmware(
    Cluster& cluster,
    tt::ChipId device_id,
    size_t l2cpu_index,
    uint64_t load_addr,
    const std::vector<uint8_t>& bin_padded) {
    // NOC-writes the firmware blob to the L2CPU tile at `load_addr`. For
    // LIM mode load_addr is the L2CPU's local LIM aperture (0x08000000);
    // for DRAM mode it is the L2CPU-local DRAM aperture (0x400030000000).
    // Same NOC routing either way — only the destination address differs.
    TT_ASSERT(l2cpu_index < 4, "l2cpu_index out of range: {}", l2cpu_index);
    TT_ASSERT(bin_padded.size() % 4 == 0, "X280 firmware must be 4-byte aligned");
    const CoreCoord tile = kL2CpuTilesByPhysicalIndex[l2cpu_index];
    cluster.write_core(
        bin_padded.data(), static_cast<uint32_t>(bin_padded.size()), tt_cxy_pair(device_id, tile), load_addr);
}

void set_reset_vectors(Cluster& cluster, tt::ChipId device_id, size_t l2cpu_index, uint64_t entry_addr) {
    // Program reset vectors for all 4 harts. Each hart has a (lo, hi) pair
    // at L2CPU_REG_BASE + 8*hart. Harts 1-3 will park in WFI per entry.S.
    //
    // Use write_core (memory-NOC path) not write_reg (uc_tlb AXI-register
    // path): pyluwen's loader.py uses noc_write32 here, and on multi-chip
    // hosts the uc_tlb path silently misroutes L2CPU-tile accesses on
    // chips != 0, leaving the reset vector unset.
    TT_ASSERT(l2cpu_index < 4, "l2cpu_index out of range: {}", l2cpu_index);
    const CoreCoord tile = kL2CpuTilesByPhysicalIndex[l2cpu_index];
    const uint32_t rv_lo = static_cast<uint32_t>(entry_addr & 0xFFFFFFFFULL);
    const uint32_t rv_hi = static_cast<uint32_t>(entry_addr >> 32);
    for (uint32_t hart = 0; hart < 4; hart++) {
        const uint64_t lo_addr = kL2CpuRegBase + 8 * hart;
        const uint64_t hi_addr = lo_addr + 4;
        cluster.write_core(&rv_lo, sizeof(rv_lo), tt_cxy_pair(device_id, tile), lo_addr);
        cluster.write_core(&rv_hi, sizeof(rv_hi), tt_cxy_pair(device_id, tile), hi_addr);
    }
}

void set_l2cpu_pll(Cluster& cluster, tt::ChipId device_id, uint32_t mhz) {
    // PLL programming sequence ported from x280/host/clock.py set_l2cpu_pll:
    //   1. Read current PLLCNTL1 (fbdiv) and PLLCNTL5 (post-dividers)
    //   2. Step post-dividers UP toward target (one at a time, 1ms apart)
    //   3. Step fbdiv toward target (one count at a time, 1ms apart)
    //   4. Step post-dividers DOWN toward target
    // This avoids out-of-range intermediate frequencies that would
    // de-lock the PLL.
    //
    // The ARC PLL4 we program here is a single per-chip clock that gates
    // ALL L2CPUs on that chip, so this function takes a chip id and target
    // frequency — not a per-L2CPU index. The chip id is required: on a
    // multi-chip host we must drive the ARC tile of the chip we are
    // booting, not chip 0.
    auto it = std::find_if(
        kPllSolutions.begin(), kPllSolutions.end(), [mhz](const PllSolution& s) { return s.freq_mhz == mhz; });
    TT_FATAL(it != kPllSolutions.end(), "No PLL solution registered for {} MHz. Available: 15, 200, 1000, 1750.", mhz);
    const PllSolution& target = *it;

    PllCntl1 cur1{};
    cluster.read_reg(reinterpret_cast<uint32_t*>(&cur1), tt_cxy_pair(device_id, kArcTileNoc0), kArcPllCntl1Addr);
    PllCntl5 cur5{};
    cluster.read_reg(reinterpret_cast<uint32_t*>(&cur5), tt_cxy_pair(device_id, kArcTileNoc0), kArcPllCntl5Addr);
    uint32_t cur1_raw = 0;
    uint32_t cur5_raw = 0;
    std::memcpy(&cur1_raw, &cur1, sizeof(cur1_raw));
    std::memcpy(&cur5_raw, &cur5, sizeof(cur5_raw));
    log_info(
        tt::LogMetal,
        "X280 dev {} PLL4 initial: CNTL1=0x{:08x} (fbdiv={}, postdiv={}, refdiv={}) CNTL5=0x{:08x} "
        "(postdivs=[{},{},{},{}])",
        device_id,
        cur1_raw,
        cur1.fbdiv,
        cur1.postdiv,
        cur1.refdiv,
        cur5_raw,
        cur5.postdiv[0],
        cur5.postdiv[1],
        cur5.postdiv[2],
        cur5.postdiv[3]);

    auto step_postdiv = [&cluster, device_id](PllCntl5& live, size_t idx, uint8_t target_val) {
        while (live.postdiv[idx] != target_val) {
            int delta = (target_val > live.postdiv[idx]) ? 1 : -1;
            live.postdiv[idx] = static_cast<uint8_t>(static_cast<int>(live.postdiv[idx]) + delta);
            cluster.write_reg(
                reinterpret_cast<const uint32_t*>(&live), tt_cxy_pair(device_id, kArcTileNoc0), kArcPllCntl5Addr);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };

    auto step_fbdiv = [&cluster, device_id](PllCntl1& live, uint16_t target_val) {
        while (live.fbdiv != target_val) {
            int delta = (target_val > live.fbdiv) ? 1 : -1;
            live.fbdiv = static_cast<uint16_t>(static_cast<int>(live.fbdiv) + delta);
            cluster.write_reg(
                reinterpret_cast<const uint32_t*>(&live), tt_cxy_pair(device_id, kArcTileNoc0), kArcPllCntl1Addr);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };

    // Increase post-dividers first (frequency goes down), then step fbdiv,
    // then decrease post-dividers (frequency goes up to target).
    for (size_t i = 0; i < 4; i++) {
        if (target.postdivs[i] > cur5.postdiv[i]) {
            step_postdiv(cur5, i, target.postdivs[i]);
        }
    }
    step_fbdiv(cur1, target.fbdiv);
    for (size_t i = 0; i < 4; i++) {
        if (target.postdivs[i] < cur5.postdiv[i]) {
            step_postdiv(cur5, i, target.postdivs[i]);
        }
    }

    // Read back the final register state to confirm the writes actually
    // landed at the target values (catches the case where the PLL register
    // bank silently NAKs writes on certain chips).
    PllCntl1 fin1{};
    cluster.read_reg(reinterpret_cast<uint32_t*>(&fin1), tt_cxy_pair(device_id, kArcTileNoc0), kArcPllCntl1Addr);
    PllCntl5 fin5{};
    cluster.read_reg(reinterpret_cast<uint32_t*>(&fin5), tt_cxy_pair(device_id, kArcTileNoc0), kArcPllCntl5Addr);
    uint32_t fin1_raw = 0;
    uint32_t fin5_raw = 0;
    std::memcpy(&fin1_raw, &fin1, sizeof(fin1_raw));
    std::memcpy(&fin5_raw, &fin5, sizeof(fin5_raw));
    log_info(
        tt::LogMetal,
        "X280 dev {} PLL4 final:   CNTL1=0x{:08x} (fbdiv={}, want fbdiv={}) CNTL5=0x{:08x} (postdivs=[{},{},{},{}], "
        "want=[{},{},{},{}])",
        device_id,
        fin1_raw,
        fin1.fbdiv,
        target.fbdiv,
        fin5_raw,
        fin5.postdiv[0],
        fin5.postdiv[1],
        fin5.postdiv[2],
        fin5.postdiv[3],
        target.postdivs[0],
        target.postdivs[1],
        target.postdivs[2],
        target.postdivs[3]);
}

bool poll_sentinel(
    Cluster& cluster,
    tt::ChipId device_id,
    size_t l2cpu_index,
    uint64_t sentinel_addr,
    uint64_t sentinel_value,
    std::chrono::milliseconds timeout,
    uint64_t* last_value_out) {
    // Sentinel lives at `sentinel_addr` on the L2CPU tile — read via the
    // memory-NOC path (read_core), not the uc_tlb AXI-register path
    // (read_reg). On multi-chip hosts read_reg silently misroutes L2CPU-
    // tile accesses on chips != 0. addr/value are passed in so the same
    // poller works for any BootProfile (LIM sentinel @ 0x08100000 or DRAM
    // sentinel @ DDR_BASE+0x5D8000).
    TT_ASSERT(l2cpu_index < 4, "l2cpu_index out of range: {}", l2cpu_index);
    const CoreCoord tile = kL2CpuTilesByPhysicalIndex[l2cpu_index];
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    uint64_t value = 0;
    do {
        cluster.read_core(&value, sizeof(value), tt_cxy_pair(device_id, tile), sentinel_addr);
        if (value == sentinel_value) {
            if (last_value_out) {
                *last_value_out = value;
            }
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    // One final read after the deadline.
    cluster.read_core(&value, sizeof(value), tt_cxy_pair(device_id, tile), sentinel_addr);
    if (last_value_out) {
        *last_value_out = value;
    }
    return value == sentinel_value;
}

std::size_t boot_idle_impl(
    Cluster& cluster,
    tt::ChipId device_id,
    BootRegion region,
    const std::filesystem::path& fw_path,
    std::uint8_t l2cpu_mask) {
    ZoneScopedN("X280 idle boot");

    const BootProfile& profile = profile_for(region);

    if (cluster.arch() != ARCH::BLACKHOLE) {
        return 0;
    }
    if (fw_path.empty() || !std::filesystem::exists(fw_path)) {
        log_warning(
            tt::LogMetal,
            "X280 {} firmware not found at '{}'; L2CPUs on device {} will remain in reset.",
            profile.region_name,
            fw_path.string(),
            device_id);
        return 0;
    }

    // Restrict the requested mask to the bottom 4 bits (one bit per physical
    // L2CPU). Bits above 3 don't map to any hardware L2CPU; ignore them
    // silently rather than fatal so callers can pass e.g. 0xFF safely.
    const std::uint8_t effective_mask = l2cpu_mask & 0x0Fu;
    if (effective_mask == 0) {
        log_info(
            tt::LogMetal, "X280 boot on device {} skipped: l2cpu_mask=0x{:x} selects no L2CPUs", device_id, l2cpu_mask);
        return 0;
    }

    std::vector<size_t> all_enabled = get_enabled_l2cpu_indices(cluster, device_id);
    if (all_enabled.empty()) {
        log_info(tt::LogMetal, "No enabled L2CPUs on device {}; skipping X280 init", device_id);
        return 0;
    }
    // Intersect enabled-L2CPU set with the requested mask. The mask is in
    // physical-index space, matching `kL2CpuTilesByPhysicalIndex` and the ARC
    // reset register bit positions.
    std::vector<size_t> l2cpu_indices;
    l2cpu_indices.reserve(all_enabled.size());
    for (size_t idx : all_enabled) {
        if (effective_mask & (1u << idx)) {
            l2cpu_indices.push_back(idx);
        }
    }
    if (l2cpu_indices.empty()) {
        log_info(
            tt::LogMetal,
            "X280 boot on device {} skipped: l2cpu_mask=0x{:x} selects no *enabled* L2CPUs "
            "(enabled set has {} L2CPU(s))",
            device_id,
            l2cpu_mask,
            all_enabled.size());
        return 0;
    }

    // Read the firmware once, pad to 4-byte alignment.
    std::vector<uint8_t> bin;
    {
        std::ifstream f(fw_path, std::ios::binary | std::ios::ate);
        TT_FATAL(f.good(), "Failed to open X280 firmware at {}", fw_path.string());
        std::streamsize size = f.tellg();
        TT_FATAL(size > 0, "X280 firmware {} is empty", fw_path.string());
        f.seekg(0, std::ios::beg);
        bin.resize(static_cast<size_t>(size));
        TT_FATAL(
            f.read(reinterpret_cast<char*>(bin.data()), size).good(),
            "Failed to read X280 firmware at {}",
            fw_path.string());
    }
    if (bin.size() % 4 != 0) {
        bin.resize(bin.size() + (4 - (bin.size() % 4)), 0);
    }

    log_info(
        tt::LogMetal,
        "Booting X280 on device {} in {} mode: {} of {} enabled L2CPU(s) (mask=0x{:x}), firmware={} ({} bytes), "
        "load_addr=0x{:016x}, sentinel_addr=0x{:016x}",
        device_id,
        profile.region_name,
        l2cpu_indices.size(),
        all_enabled.size(),
        l2cpu_mask,
        fw_path.string(),
        bin.size(),
        profile.load_addr,
        profile.sentinel_addr);

    // One-time diagnostic: log whether this chip is MMIO-capable or tunneled
    // remotely (e.g. Blackhole Galaxy chips 1-7 routed through chip 0). AXI
    // register access through the ARC tile takes a meaningfully different
    // code path on remote chips and is a known soft spot.
    const bool is_mmio = cluster.get_cluster_desc()->is_chip_mmio_capable(device_id);
    log_info(
        tt::LogMetal,
        "X280 boot device {} is_mmio_capable={} (remote chips use Ethernet-tunneled NOC access)",
        device_id,
        is_mmio);

    // Phase 1: per-L2CPU setup. Everything here is per-tile (assert reset,
    // load firmware to that tile's LIM, etc.) and must run BEFORE the
    // chip-wide reset release in Phase 2. The PLL and reset-register
    // release are deliberately NOT in this loop: PLL4 is a single per-chip
    // resource that gates ALL L2CPUs on the chip, and the ARC L2CPU_RESET
    // register is a single bitfield covering all four L2CPUs. Mirrors
    // tt-bh-linux/boot.py, which loads firmware for each L2CPU in a
    // per-tile loop and then calls reset_x280() ONCE.
    for (size_t l2cpu_index : l2cpu_indices) {
        const CoreCoord tile = kL2CpuTilesByPhysicalIndex[l2cpu_index];

        // Step 1: assert L2CPU reset and verify the register actually
        // changed (catches the case where AXI access via NOC0->ARC is silently
        // dropping writes on remote chips).
        uint32_t reset_before = 0;
        cluster.read_reg(&reset_before, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
        assert_l2cpu_reset(cluster, device_id, l2cpu_index);
        uint32_t reset_after_assert = 0;
        cluster.read_reg(&reset_after_assert, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
        log_info(
            tt::LogMetal,
            "X280 dev {} L2CPU {} reset reg: before=0x{:08x} after_assert=0x{:08x} (bit {} should be 0)",
            device_id,
            l2cpu_index,
            reset_before,
            reset_after_assert,
            l2cpu_index + 4);

        // Step 1b: read and log this L2CPU's L3 cache-controller clock-gate
        // register. We intentionally do NOT write it here — doing so on a
        // running L2CPU wedges the core, and ARC FW is expected to bring
        // up the L3 clock domain via TT_SMC_MSG_POWER_SETTING during
        // device-open. This readback is for diagnostic only. See
        // x280/host/boot_idle_x280.py:48 and tt-bh-linux/boot.py:211 for
        // the analogous pyluwen path that does write 0xf to this register.
        uint32_t l3_gate_before = 0;
        cluster.read_core(&l3_gate_before, sizeof(l3_gate_before), tt_cxy_pair(device_id, tile), kL2CpuL3ClockGateAddr);
        log_info(
            tt::LogMetal,
            "X280 dev {} L2CPU {} L3 clock-gate (0x{:08x}) before ungate: 0x{:08x}",
            device_id,
            l2cpu_index,
            static_cast<uint32_t>(kL2CpuL3ClockGateAddr),
            l3_gate_before);

        // Step 2: load firmware to `profile.load_addr` (LIM 0x08000000 or
        // DRAM 0x400030000000), then read back the first 16 bytes and verify
        // they match what we wrote. If they don't, the NOC write to the
        // L2CPU tile is not landing — boot will hang regardless of PLL/reset.
        load_firmware(cluster, device_id, l2cpu_index, profile.load_addr, bin);
        std::array<uint32_t, 4> load_readback{};
        cluster.read_core(
            load_readback.data(),
            static_cast<uint32_t>(load_readback.size() * sizeof(uint32_t)),
            tt_cxy_pair(device_id, tile),
            profile.load_addr);
        std::array<uint32_t, 4> load_expected{};
        std::memcpy(load_expected.data(), bin.data(), load_expected.size() * sizeof(uint32_t));
        const bool load_ok = load_readback == load_expected;
        log_info(
            tt::LogMetal,
            "X280 dev {} L2CPU {} {}[0..16] @ 0x{:016x} readback={:08x} {:08x} {:08x} {:08x} expected={:08x} {:08x} "
            "{:08x} {:08x} match={}",
            device_id,
            l2cpu_index,
            profile.region_name,
            profile.load_addr,
            load_readback[0],
            load_readback[1],
            load_readback[2],
            load_readback[3],
            load_expected[0],
            load_expected[1],
            load_expected[2],
            load_expected[3],
            load_ok);

        // Zero the sentinel so a stale value from a prior run doesn't satisfy
        // our poll immediately. Use write_core (memory-NOC) not write_reg
        // (uc_tlb AXI-register) — see set_reset_vectors comment for why.
        const uint64_t zero64 = 0;
        cluster.write_core(&zero64, sizeof(zero64), tt_cxy_pair(device_id, tile), profile.sentinel_addr);

        // Pre-zero the trap-CSR readback region (_trap_mcause / _trap_mepc /
        // _trap_mtval, 3 x u64 starting at profile.trap_mcause_addr) so the
        // timeout diagnostic is unambiguous. These symbols live in .bss and
        // the firmware .bin doesn't carry .bss bytes, so without this step
        // we'd be reading raw SRAM/DRAM noise from before the firmware
        // load. With pre-zero in place, on a timeout:
        //   * all three == 0  -> the X280 hart never executed (no
        //                        instruction fetched, BSS zero loop never
        //                        ran, trap handler never ran);
        //   * mcause is a legal RISC-V cause code AND mepc/mtval are
        //     coherent  -> _trap_handler ran on a real exception.
        // entry.S's BSS zero loop will overwrite these to 0 again on a
        // successful boot, so this pre-zero never races with normal
        // execution.
        const std::array<uint64_t, 3> trap_zero{0, 0, 0};
        cluster.write_core(
            trap_zero.data(),
            static_cast<uint32_t>(trap_zero.size() * sizeof(uint64_t)),
            tt_cxy_pair(device_id, tile),
            profile.trap_mcause_addr);

        // Step 3: program reset vectors and verify hart-0's pair reads back.
        // Entry address = the same address we just loaded firmware to.
        // Read back via the same memory-NOC path the writer uses.
        set_reset_vectors(cluster, device_id, l2cpu_index, profile.load_addr);
        uint32_t rv_lo_rb = 0;
        uint32_t rv_hi_rb = 0;
        cluster.read_core(&rv_lo_rb, sizeof(rv_lo_rb), tt_cxy_pair(device_id, tile), kL2CpuRegBase);
        cluster.read_core(&rv_hi_rb, sizeof(rv_hi_rb), tt_cxy_pair(device_id, tile), kL2CpuRegBase + 4);
        log_info(
            tt::LogMetal,
            "X280 dev {} L2CPU {} hart0 reset vector readback: 0x{:08x}{:08x} (expected 0x{:016x})",
            device_id,
            l2cpu_index,
            rv_hi_rb,
            rv_lo_rb,
            profile.load_addr);

        // Step 3b: verify the sentinel really is zero after our zero-write,
        // before we release reset. If we read non-zero here, either the
        // zero-write was misrouted, or there's stale data from a previous
        // boot that our zero-write didn't clear. Either way, the post-release
        // sentinel poll would be meaningless.
        uint64_t sentinel_pre_release = 0;
        cluster.read_core(
            &sentinel_pre_release, sizeof(sentinel_pre_release), tt_cxy_pair(device_id, tile), profile.sentinel_addr);
        log_info(
            tt::LogMetal,
            "X280 dev {} L2CPU {} sentinel pre-release @ 0x{:016x}: 0x{:016x} (expected 0x0)",
            device_id,
            l2cpu_index,
            profile.sentinel_addr,
            sentinel_pre_release);
    }

    // Phase 2: bracket the chip-wide reset release with a single PLL
    // down/up. Mirrors tt-bh-linux/boot.py::reset_x280:
    //
    //     clock.set_l2cpu_pll(chip, 200)                # ONCE
    //     v = axi_read32(L2CPU_RESET)
    //     for idx in l2cpu_indices: v |= 1 << (idx + 4)
    //     axi_write32(L2CPU_RESET, v)                   # ONE write
    //     axi_read32(L2CPU_RESET)                       # flush
    //     clock.set_l2cpu_pll(chip, 1750)               # ONCE
    //
    // The two PLL calls and the reset register R-M-W are per-chip, not
    // per-L2CPU; nesting them inside the Phase 1 loop (as an earlier
    // version of this port did) would ramp PLL4 down-and-back-up once
    // per enabled L2CPU on the chip, which is wasted work at best and
    // can leave the PLL transiently de-locked at worst.

    // Step 4: PLL down to the reset-release frequency.
    set_l2cpu_pll(cluster, device_id, kX280ResetReleasePllMhz);

    // Step 5: batched reset release. Read current value, OR in every
    // selected L2CPU's bit, write once, then read back as a posted-write
    // flush. A per-bit R-M-W loop would issue len(l2cpu_indices) separate
    // register writes for no benefit.
    uint32_t release_before = 0;
    cluster.read_reg(&release_before, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
    uint32_t release_val = release_before;
    for (size_t l2cpu_index : l2cpu_indices) {
        release_val |= (1u << (l2cpu_index + 4));
    }
    cluster.write_reg(&release_val, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
    uint32_t release_after = 0;
    cluster.read_reg(&release_after, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
    log_info(
        tt::LogMetal,
        "X280 dev {} batched L2CPU release ({} L2CPU(s)): before=0x{:08x} wrote=0x{:08x} after=0x{:08x}",
        device_id,
        l2cpu_indices.size(),
        release_before,
        release_val,
        release_after);

    // Step 5b: ramp PLL4 up to the steady-state target. set_l2cpu_pll
    // steps one divider count at a time (1ms between writes — see
    // clock.py step_postdiv / step_fbdiv), so this is not a hard
    // frequency jump. Doing the ramp AFTER reset release matches
    // tt-bh-linux/boot.py:85.
    set_l2cpu_pll(cluster, device_id, kDefaultX280PllMhz);

    // Phase 3: per-L2CPU sentinel verification + diagnostic dump on
    // timeout. Each L2CPU writes its sentinel independently, so this
    // has to be per-tile. A timeout on any L2CPU still throws; the
    // diagnostic from the first failure is what a human operator wants
    // to see.
    std::size_t booted = 0;
    for (size_t l2cpu_index : l2cpu_indices) {
        const CoreCoord tile = kL2CpuTilesByPhysicalIndex[l2cpu_index];

        uint64_t last_value = 0;
        if (!poll_sentinel(
                cluster,
                device_id,
                l2cpu_index,
                profile.sentinel_addr,
                profile.sentinel_value,
                kX280SentinelTimeout,
                &last_value)) {
            // Timeout path: dump enough state to determine *why* the L2CPU
            // didn't write the sentinel.
            //
            // 1. Re-read load_addr[0..16]: should still equal what we
            //    loaded. If it differs, something is mutating the load
            //    region (cache writeback, other initiator, etc.).
            // 2. Re-read the reset register and the reset vector: confirms
            //    they're still in the post-release state we expected.
            // 3. Read entry.S's trap CSR readback area. If mcause is
            //    nonzero, the L2CPU ran far enough to enter the trap
            //    handler and tells us why it bailed.
            std::array<uint32_t, 4> load_post{};
            cluster.read_core(
                load_post.data(),
                static_cast<uint32_t>(load_post.size() * sizeof(uint32_t)),
                tt_cxy_pair(device_id, tile),
                profile.load_addr);
            uint32_t reset_post = 0;
            cluster.read_reg(&reset_post, tt_cxy_pair(device_id, kArcTileNoc0), kArcL2CpuResetReg);
            uint32_t rv_lo_post = 0, rv_hi_post = 0;
            cluster.read_core(&rv_lo_post, sizeof(rv_lo_post), tt_cxy_pair(device_id, tile), kL2CpuRegBase);
            cluster.read_core(&rv_hi_post, sizeof(rv_hi_post), tt_cxy_pair(device_id, tile), kL2CpuRegBase + 4);
            uint64_t mcause = 0, mepc = 0, mtval = 0;
            cluster.read_core(&mcause, sizeof(mcause), tt_cxy_pair(device_id, tile), profile.trap_mcause_addr);
            cluster.read_core(&mepc, sizeof(mepc), tt_cxy_pair(device_id, tile), profile.trap_mepc_addr);
            cluster.read_core(&mtval, sizeof(mtval), tt_cxy_pair(device_id, tile), profile.trap_mtval_addr);
            // Classify the trap-CSR readback. We pre-zero these before
            // releasing reset (see Phase 1 pre-zero), so:
            //   * all three == 0  -> hart never executed any instruction
            //                        (no fetch, no BSS zero loop, no trap).
            //   * mcause is a legal cause AND mepc/mtval are coherent
            //     -> trap handler ran on a real exception.
            //   * anything else -> memory noise, hart still didn't run.
            const bool hart_never_ran = (mcause == 0 && mepc == 0 && mtval == 0);
            const uint64_t mcause_code = mcause & ~(1ULL << 63);
            const bool mcause_is_legal_exception = ((mcause >> 63) == 0) && (mcause_code <= 15);
            const bool mcause_is_legal_interrupt = ((mcause >> 63) == 1) && (mcause_code <= 15);
            const char* trap_class = hart_never_ran ? "HART NEVER EXECUTED (trap pre-zero intact)"
                                     : (mcause_is_legal_exception || mcause_is_legal_interrupt)
                                         ? "REAL TRAP (mcause is legal cause)"
                                         : "MEMORY NOISE (mcause not a legal cause)";
            log_warning(
                tt::LogMetal,
                "X280 timeout diag dev {} L2CPU {} ({} mode):\n"
                "  {}[0..16] @ 0x{:016x} (re-read) = {:08x} {:08x} {:08x} {:08x}\n"
                "  reset reg (re-read)              = 0x{:08x} (bit {} should be 1)\n"
                "  reset vector hart0               = 0x{:08x}{:08x}\n"
                "  trap mcause @ 0x{:016x}          = 0x{:016x}\n"
                "  trap mepc   @ 0x{:016x}          = 0x{:016x}\n"
                "  trap mtval  @ 0x{:016x}          = 0x{:016x}\n"
                "  diagnosis                        = {}",
                device_id,
                l2cpu_index,
                profile.region_name,
                profile.region_name,
                profile.load_addr,
                load_post[0],
                load_post[1],
                load_post[2],
                load_post[3],
                reset_post,
                l2cpu_index + 4,
                rv_hi_post,
                rv_lo_post,
                profile.trap_mcause_addr,
                mcause,
                profile.trap_mepc_addr,
                mepc,
                profile.trap_mtval_addr,
                mtval,
                trap_class);
            TT_THROW(
                "X280 boot failed on device {} L2CPU {} ({} mode): sentinel @ 0x{:016x} timeout after {}ms "
                "(expected 0x{:016x}, got 0x{:016x}). Check x280/.planning/research/BUGS_AND_LIMITATIONS.md.",
                device_id,
                l2cpu_index,
                profile.region_name,
                profile.sentinel_addr,
                kX280SentinelTimeout.count(),
                profile.sentinel_value,
                last_value);
        }
        log_info(
            tt::LogMetal,
            "X280 device {} L2CPU {} {} sentinel OK (0x{:016x})",
            device_id,
            l2cpu_index,
            profile.region_name,
            last_value);
        ++booted;
    }
    return booted;
}

}  // namespace

std::size_t boot_idle(tt::ChipId device_id, BootRegion region, std::uint8_t l2cpu_mask) {
    const BootProfile& profile = profile_for(region);
    std::filesystem::path fw_path = default_firmware_path(region);
    if (fw_path.empty()) {
        log_warning(
            tt::LogMetal,
            "X280 {} not found under runtime/hw/firmware/blackhole/x280/; "
            "L2CPUs on device {} will remain in reset. Rebuild with TT_METAL_BUILD_X280_FW=ON "
            "to ship the firmware.",
            profile.default_fw_basename,
            device_id);
        return 0;
    }
    return boot_idle(device_id, region, fw_path, l2cpu_mask);
}

std::size_t boot_idle(
    tt::ChipId device_id, BootRegion region, const std::filesystem::path& firmware_bin, std::uint8_t l2cpu_mask) {
    Cluster& cluster = MetalContext::instance().get_cluster();
    return boot_idle_impl(cluster, device_id, region, firmware_bin, l2cpu_mask);
}

}  // namespace tt::tt_metal::experimental::x280
