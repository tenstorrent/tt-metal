// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Remote atomics on Keraunos (Phase 3 of keraunos_plan.md). Two mechanisms:
//  * IDMA L1 accumulate (idma_l1_atomic_accum_cmdbuf). cce-dv's l1_atomic_accum.cpp gives the formats: 3 INT32,
//    4 INT32_NORMAL, 5 UINT32, 6 INT8, 7 UINT8, with operation 9. The probe tries them (and wrapping, saturation and the
//    address form) against a word block that does not start at zero, so an add and a copy differ.
//  * NOC atomic increment (noc_semaphore_inc), the call behind semaphores and fabric credits.

#include "keraunos_cce_fixture.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

constexpr uint32_t kWords = 16;
constexpr uint32_t kRepeats = 3;
constexpr uint32_t kStart = 100;
// cce-dv's SRAM_LBASE in Keraunos SoC mode: HSIO 0 CCE 0 in the SoC map.
constexpr uint64_t kSocLbase = 0x22800000ULL;

class KeraunosAtomicsProbe : public KeraunosCceTest {
protected:
    struct Combo {
        uint32_t fmt;
        bool soc_addr;
        bool wrapping;
        bool no_sat;
    };

    // Runs one accumulate (src words all 1) into a block holding kStart + i, and returns the destination words.
    std::vector<uint32_t> run_accumulate(const Hsio& a, const Hsio& b, const Combo& c, uint32_t status_offset) {
        write_words(a, kSrcOffset, std::vector<uint32_t>(kWords, 1));
        std::vector<uint32_t> start(kWords);
        for (uint32_t i = 0; i < kWords; i++) {
            start[i] = kStart + i;
        }
        write_words(b, kDstOffset, start);
        write_words(a, status_offset, std::vector<uint32_t>(kStatusWords, 0));
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_idma_accum_cfg.cpp",
            a.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        const uint64_t src = c.soc_addr ? kSocLbase + unreserved_ + kSrcOffset : global_addr(a, kSrcOffset);
        const uint64_t dst = c.soc_addr ? kSocLbase + unreserved_ + kDstOffset : global_addr(b, kDstOffset);
        std::vector<uint32_t> args = split(src);
        const auto d = split(dst);
        args.insert(args.end(), d.begin(), d.end());
        args.insert(
            args.end(),
            {kWords * 4,
             kRepeats,
             c.fmt,
             /*op=*/9,
             (c.wrapping ? 1u : 0u) | (c.no_sat ? 2u : 0u),
             unreserved_ + status_offset});
        SetRuntimeArgs(program, k, a.logical, args);
        launch(program, a, status_offset);
        return read_words(b, kDstOffset, kWords);
    }

    bool adds(const std::vector<uint32_t>& got) const {
        for (uint32_t i = 0; i < kWords; i++) {
            if (got[i] != kStart + i + kRepeats) {
                return false;
            }
        }
        return true;
    }
};

// Sweeps the accumulate settings on HSIO 0 CCE 0's own SRAM and logs what each one did. Passes when some setting adds.
TEST_F(KeraunosAtomicsProbe, AccumulateOwnSramSweep) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    std::vector<std::string> adders;
    for (const uint32_t fmt : {3u, 4u, 5u, 6u, 7u, 0u}) {
        for (const bool soc : {false, true}) {
            for (const bool wrap : {false, true}) {
                for (const bool no_sat : {false, true}) {
                    const Combo c{fmt, soc, wrap, no_sat};
                    const auto got = run_accumulate(*a, *a, c, kStatusOffset);
                    const bool ok = adds(got);
                    const std::string line = fmt::format(
                        "fmt {} addr {} wrap {} no_sat {}: first words {} {} {} {} -> {}",
                        fmt,
                        soc ? "soc" : "global",
                        wrap,
                        no_sat,
                        got[0],
                        got[1],
                        got[2],
                        got[3],
                        ok ? "ADDS" : (got[0] == 1 ? "copy" : "other"));
                    std::cout << line << std::endl;
                    if (ok) {
                        adders.push_back(line);
                    }
                }
            }
        }
    }
    EXPECT_FALSE(adders.empty()) << "no accumulate setting added";
}

// Accumulates into HSIO 1's SRAM through the package address, with the settings given by KERAUNOS_ACCUM_FMT (default 3).
// Disabled on mmk (2026-11-20): package route to another tile copies instead of adding.
TEST_F(KeraunosAtomicsProbe, DISABLED_AccumulateOtherTileSweep) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(1, 0);
    ASSERT_TRUE(a.has_value() && b.has_value());
    std::vector<std::string> adders;
    for (const uint32_t fmt : {3u, 4u, 5u}) {
        for (const bool wrap : {false, true}) {
            for (const bool no_sat : {false, true}) {
                const Combo c{fmt, false, wrap, no_sat};
                const auto got = run_accumulate(*a, *b, c, kStatusOffset);
                const bool ok = adds(got);
                std::cout << fmt::format(
                                 "other tile: fmt {} wrap {} no_sat {}: first words {} {} {} {} -> {}",
                                 fmt, wrap, no_sat, got[0], got[1], got[2], got[3],
                                 ok ? "ADDS" : (got[0] == 1 ? "copy" : "other"))
                          << std::endl;
                if (ok) {
                    adders.push_back("x");
                }
            }
        }
    }
    EXPECT_FALSE(adders.empty()) << "no accumulate setting added into another tile";
}

class KeraunosNocAtomic : public KeraunosCceTest {
protected:
    void expect_increment(const Hsio& src, const Hsio& dst, bool posted = true) {
        constexpr uint32_t kCount = 5;
        constexpr uint32_t kIncr = 3;
        constexpr uint32_t kBase = 40;
        write_words(dst, kDstOffset, {kBase});
        write_words(src, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        const CoreCoord noc = device_->virtual_core_from_logical_core(dst.logical, CoreType::ETH);
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_noc_atomic_inc.cpp",
            src.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            k,
            src.logical,
            {static_cast<uint32_t>(noc.x),
             static_cast<uint32_t>(noc.y),
             unreserved_ + kDstOffset,
             kCount,
             kIncr,
             unreserved_ + kStatusOffset,
             posted ? 1u : 0u});
        launch(program, src);
        const auto status = read_words(src, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        // Posted increments may still be in flight when the kernel ends, so poll the destination.
        uint32_t value = 0;
        for (int i = 0; i < 100; i++) {
            value = read_words(dst, kDstOffset, 1)[0];
            if (value == kBase + kCount * kIncr) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        EXPECT_EQ(value, kBase + kCount * kIncr) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    }
};

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs until the launch times out.
TEST_F(KeraunosNocAtomic, DISABLED_IncrementOwnSram) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    expect_increment(*a, *a);
}

// Non-posted increments wait for an ack. On mmk the first run of this never finished (2026-11-19), so it is disabled.
TEST_F(KeraunosNocAtomic, DISABLED_NonPostedIncrementOwnSram) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    expect_increment(*a, *a, /*posted=*/false);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocAtomic, DISABLED_IncrementPeerCceSram) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*a, *b);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocAtomic, DISABLED_IncrementCce1OwnSram) {
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(b.has_value());
    expect_increment(*b, *b);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocAtomic, DISABLED_IncrementCce0FromCce1) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*b, *a);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocAtomic, DISABLED_IncrementOtherTileSram) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(1, 0);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*a, *b);
}

// IDMA accumulate through cce-dv's SoC-map SRAM addresses (KERAUNOS_TEST: HSIO tiles 0x4000000 apart from 0x20000000,
// SRAM 0x2800000 into a tile, CCE1 0x400000 above CCE0). On HSIO 0 CCE 0's own SRAM this form adds where the package
// address only copies; these cases find out which other destinations it reaches.
class KeraunosAccumulateForms : public KeraunosCceTest {
protected:
    static uint64_t soc_sram(uint32_t hsio, uint32_t cce) {
        return 0x20000000ULL + hsio * 0x4000000ULL + 0x2800000ULL + cce * 0x400000ULL;
    }

    // Writes `src_words` at kSrcOffset of `src_core` and `start` at kDstOffset of `dst_core`, has `runner` accumulate
    // `repeats` times from `src` to `dst`, and returns the destination words.
    std::vector<uint32_t> accumulate(
        const Hsio& runner,
        const Hsio& src_core,
        uint64_t src,
        const Hsio& dst_core,
        uint64_t dst,
        const std::vector<uint32_t>& src_words,
        const std::vector<uint32_t>& start,
        uint32_t fmt,
        uint32_t repeats,
        bool no_sat) {
        write_words(src_core, kSrcOffset, src_words);
        write_words(dst_core, kDstOffset, start);
        write_words(runner, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_idma_accum_cfg.cpp",
            runner.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        std::vector<uint32_t> args = split(src);
        const auto d = split(dst);
        args.insert(args.end(), d.begin(), d.end());
        args.insert(
            args.end(),
            {static_cast<uint32_t>(src_words.size() * 4),
             repeats,
             fmt,
             /*op=*/9,
             no_sat ? 2u : 0u,
             unreserved_ + kStatusOffset});
        SetRuntimeArgs(program, k, runner.logical, args);
        launch(program, runner);
        const auto status = read_words(runner, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        return read_words(dst_core, kDstOffset, static_cast<uint32_t>(start.size()));
    }

    // Accumulates ones into kStart + i three times with format 4 (INT32_NORMAL) and checks for kStart + i + 3.
    void expect_adds(const Hsio& runner, const Hsio& src_core, uint64_t src, const Hsio& dst_core, uint64_t dst) {
        std::vector<uint32_t> start(kWords);
        std::vector<uint32_t> expected(kWords);
        for (uint32_t i = 0; i < kWords; i++) {
            start[i] = kStart + i;
            expected[i] = kStart + i + kRepeats;
        }
        const auto got = accumulate(
            runner, src_core, src, dst_core, dst, std::vector<uint32_t>(kWords, 1), start, 4, kRepeats, false);
        std::cout << fmt::format("dst {:#x}: {}", dst, fmt::join(got, " ")) << std::endl;
        EXPECT_EQ(got, expected) << (got[0] == 1 ? "acts as a copy" : "neither a copy nor an add");
    }
};

TEST_F(KeraunosAccumulateForms, Cce0OwnSram) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    expect_adds(*a, *a, soc_sram(0, 0) + unreserved_ + kSrcOffset, *a, soc_sram(0, 0) + unreserved_ + kDstOffset);
}

TEST_F(KeraunosAccumulateForms, Cce0ToPeerCce1) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_adds(*a, *a, soc_sram(0, 0) + unreserved_ + kSrcOffset, *b, soc_sram(0, 1) + unreserved_ + kDstOffset);
}

TEST_F(KeraunosAccumulateForms, Cce1OwnSram) {
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(b.has_value());
    expect_adds(*b, *b, soc_sram(0, 1) + unreserved_ + kSrcOffset, *b, soc_sram(0, 1) + unreserved_ + kDstOffset);
}

// Disabled on mmk (2026-11-20): from CCE1, CCE0's SoC address is not reached (a plain copy does not land either).
TEST_F(KeraunosAccumulateForms, DISABLED_Cce1ToCce0) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_adds(*b, *b, soc_sram(0, 1) + unreserved_ + kSrcOffset, *a, soc_sram(0, 0) + unreserved_ + kDstOffset);
}

// Disabled on mmk (2026-11-20): another tile's NoC-form address is not reached on mmk (a plain copy does not land either).
TEST_F(KeraunosAccumulateForms, DISABLED_Cce0ToOtherTileSocForm) {
    const auto a = hsio_core(0, 0);
    const auto c = hsio_core(1, 0);
    ASSERT_TRUE(a.has_value() && c.has_value());
    expect_adds(*a, *a, soc_sram(0, 0) + unreserved_ + kSrcOffset, *c, soc_sram(1, 0) + unreserved_ + kDstOffset);
}

// A plain IDMA copy (no accumulate) to HSIO 1 in its NoC form (cce-dv TILE1_SRAM 0x26800000), to tell whether that
// route works at all; the accumulate there left the destination unchanged.
// Disabled on mmk (2026-11-20): another tile's NoC-form address is not reached on mmk.
TEST_F(KeraunosAccumulateForms, DISABLED_CopyToOtherTileSocForm) {
    const auto a = hsio_core(0, 0);
    const auto c = hsio_core(1, 0);
    ASSERT_TRUE(a.has_value() && c.has_value());
    std::vector<uint32_t> data(kWords);
    for (uint32_t i = 0; i < kWords; i++) {
        data[i] = 0xC0F10000u | i;
    }
    write_words(*a, kSrcOffset, data);
    write_words(*c, kDstOffset, std::vector<uint32_t>(kWords, 0));
    write_words(*a, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    add_idma_copy(
        program,
        *a,
        {.src = soc_sram(0, 0) + unreserved_ + kSrcOffset,
         .dst = soc_sram(1, 0) + unreserved_ + kDstOffset,
         .len = kWords * 4});
    launch(program, *a);
    const auto status = read_words(*a, kStatusOffset, kStatusWords);
    EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    const auto got = read_words(*c, kDstOffset, kWords);
    std::cout << fmt::format("copy to {:#x}: {}", soc_sram(1, 0) + unreserved_ + kDstOffset, fmt::join(got, " "))
              << std::endl;
    EXPECT_EQ(got, data);
}

// The CCE CPU-view form: plain local SRAM offsets, where each CCE sees CCE0's SRAM at [0, 4 MiB) and CCE1's at
// [4, 8 MiB). cce-dv's IP-level tests use this form (SRAM_LBASE 0).
// Disabled on mmk (2026-11-20): IDMA does not take CPU-view (local offset) addresses.
TEST_F(KeraunosAccumulateForms, DISABLED_Cce0OwnCpuForm) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    expect_adds(*a, *a, unreserved_ + kSrcOffset, *a, unreserved_ + kDstOffset);
}

// Disabled on mmk (2026-11-20): IDMA does not take CPU-view (local offset) addresses.
TEST_F(KeraunosAccumulateForms, DISABLED_Cce0ToCce1CpuForm) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_adds(*a, *a, unreserved_ + kSrcOffset, *b, 0x400000ULL + unreserved_ + kDstOffset);
}

// Disabled on mmk (2026-11-20): IDMA does not take CPU-view (local offset) addresses.
TEST_F(KeraunosAccumulateForms, DISABLED_Cce1ToCce0CpuForm) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_adds(*b, *b, 0x400000ULL + unreserved_ + kSrcOffset, *a, unreserved_ + kDstOffset);
}

// A plain IDMA copy from CCE1 to CCE0's SoC form: tells whether that route exists at all.
// Disabled on mmk (2026-11-20): from CCE1, CCE0's SoC address is not reached.
TEST_F(KeraunosAccumulateForms, DISABLED_CopyCce1ToCce0SocForm) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    std::vector<uint32_t> data(kWords);
    for (uint32_t i = 0; i < kWords; i++) {
        data[i] = 0xC1C00000u | i;
    }
    write_words(*b, kSrcOffset, data);
    write_words(*a, kDstOffset, std::vector<uint32_t>(kWords, 0));
    write_words(*b, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    add_idma_copy(
        program,
        *b,
        {.src = soc_sram(0, 1) + unreserved_ + kSrcOffset,
         .dst = soc_sram(0, 0) + unreserved_ + kDstOffset,
         .len = kWords * 4});
    launch(program, *b);
    const auto got = read_words(*a, kDstOffset, kWords);
    std::cout << fmt::format("copy CCE1 -> {:#x}: {}", soc_sram(0, 0) + unreserved_ + kDstOffset, fmt::join(got, " "))
              << std::endl;
    EXPECT_EQ(got, data);
}

// Source in the SoC form, destination another tile through the package address (how cce-dv reaches other tiles).
// Disabled on mmk (2026-11-20): package route to another tile copies instead of adding.
TEST_F(KeraunosAccumulateForms, DISABLED_Cce0ToOtherTilePackageDestination) {
    const auto a = hsio_core(0, 0);
    const auto c = hsio_core(1, 0);
    ASSERT_TRUE(a.has_value() && c.has_value());
    expect_adds(*a, *a, soc_sram(0, 0) + unreserved_ + kSrcOffset, *c, global_addr(*c, kDstOffset));
}

// Tells the int32 formats apart: 100 plus -1 (0xFFFFFFFF) once. Logs every result; checks only that format 4
// (INT32_NORMAL, two's complement) gives 99.
TEST_F(KeraunosAccumulateForms, Int32FormatsWithNegativeValue) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    const uint64_t src = soc_sram(0, 0) + unreserved_ + kSrcOffset;
    const uint64_t dst = soc_sram(0, 0) + unreserved_ + kDstOffset;
    for (const uint32_t fmt : {3u, 4u, 5u}) {
        for (const bool no_sat : {false, true}) {
            const auto got = accumulate(
                *a, *a, src, *a, dst, std::vector<uint32_t>(kWords, 0xFFFFFFFFu), std::vector<uint32_t>(kWords, 100u),
                fmt, 1, no_sat);
            std::cout << fmt::format("fmt {} no_sat {}: 100 + 0xffffffff -> {:#010x}", fmt, no_sat, got[0]) << std::endl;
            if (fmt == 4) {
                EXPECT_EQ(got[0], 99u) << "fmt 4 no_sat " << no_sat;
            }
        }
    }
}

// Plain NOC traffic (overlay command buffer, not IDMA) from a CCE. Run these last: a NOC path that does not work hangs
// the core until the launch times out.
class KeraunosNocProbe : public KeraunosCceTest {
protected:
    // Returns the status block; `dst_word` receives the destination word after the run.
    std::vector<uint32_t> run(const Hsio& src, const Hsio& dst, uint32_t mode, uint32_t n, uint32_t incr) {
        write_words(src, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        const CoreCoord my = device_->virtual_core_from_logical_core(src.logical, CoreType::ETH);
        const CoreCoord to = device_->virtual_core_from_logical_core(dst.logical, CoreType::ETH);
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_noc_probe.cpp",
            src.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            k,
            src.logical,
            {static_cast<uint32_t>(my.x),
             static_cast<uint32_t>(my.y),
             static_cast<uint32_t>(to.x),
             static_cast<uint32_t>(to.y),
             unreserved_ + kDstOffset,
             unreserved_ + kSrcOffset,
             mode,
             n,
             incr,
             unreserved_ + kStatusOffset});
        launch(program, src);
        return read_words(src, kStatusOffset, kStatusWords);
    }

    void expect_write(const Hsio& src, const Hsio& dst) {
        std::vector<uint32_t> data(16);
        for (uint32_t i = 0; i < data.size(); i++) {
            data[i] = 0x50C00000u | i;
        }
        write_words(src, kSrcOffset, data);
        write_words(dst, kDstOffset, std::vector<uint32_t>(data.size(), 0));
        const auto status = run(src, dst, 0, data.size() * 4, 0);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(read_words(dst, kDstOffset, data.size()), data);
    }

    void expect_increment(const Hsio& src, const Hsio& dst) {
        constexpr uint32_t kBase = 40;
        constexpr uint32_t kCount = 5;
        constexpr uint32_t kIncr = 3;
        write_words(dst, kDstOffset, {kBase});
        const auto status = run(src, dst, 1, kCount, kIncr);
        uint32_t value = 0;
        for (int i = 0; i < 100; i++) {
            value = read_words(dst, kDstOffset, 1)[0];
            if (value == kBase + kCount * kIncr) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(value, kBase + kCount * kIncr) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    }
};

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocProbe, DISABLED_WriteOwnSram) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    expect_write(*a, *a);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocProbe, DISABLED_WritePeerCce) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_write(*a, *b);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocProbe, DISABLED_AtomicIncrementOwnSram) {
    const auto a = hsio_core(0, 0);
    ASSERT_TRUE(a.has_value());
    expect_increment(*a, *a);
}

// Disabled on mmk (2026-11-20): no NOC (non-IDMA) traffic from a CCE completes; hangs.
TEST_F(KeraunosNocProbe, DISABLED_AtomicIncrementPeerCce) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*a, *b);
}

}  // namespace
}  // namespace tt::tt_metal
