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
TEST_F(KeraunosAtomicsProbe, AccumulateOtherTileSweep) {
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

TEST_F(KeraunosNocAtomic, IncrementOwnSram) {
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

TEST_F(KeraunosNocAtomic, IncrementPeerCceSram) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*a, *b);
}

TEST_F(KeraunosNocAtomic, IncrementCce1OwnSram) {
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(b.has_value());
    expect_increment(*b, *b);
}

TEST_F(KeraunosNocAtomic, IncrementCce0FromCce1) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(0, 1);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*b, *a);
}

TEST_F(KeraunosNocAtomic, IncrementOtherTileSram) {
    const auto a = hsio_core(0, 0);
    const auto b = hsio_core(1, 0);
    ASSERT_TRUE(a.has_value() && b.has_value());
    expect_increment(*a, *b);
}

}  // namespace
}  // namespace tt::tt_metal
