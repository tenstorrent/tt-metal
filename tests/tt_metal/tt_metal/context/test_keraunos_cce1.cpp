// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CCE1 on a Keraunos HSIO tile (Phase 4 of keraunos_plan.md).
//
// Each CCE of an HSIO tile is its own ethernet core with eight harts: CCE0 is descriptor core (4, hsio)
// and CCE1 is (5, hsio). Both run the same firmware and see the tile the same way: their own SRAM at local 0
// with the uncached alias 8 MiB up, and the sibling CCE's SRAM in the next 4 MiB window
// (MEM_IERISC_PEER_CCE_BASE) with its alias 8 MiB above that.

#include "keraunos_cce_fixture.hpp"

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

constexpr auto kHart0 = DataMovementProcessor::RISCV_0;
// Scratch in a CCE's own SRAM, well above its firmware and kernels (a core offset).
constexpr uint32_t kScratch = 0x200000;

class KeraunosCce1 : public KeraunosCceTest {
protected:
    // An offset relative to the unreserved base (what write_words/read_words take) for a core address.
    uint32_t rel(uint32_t core_offset) const { return core_offset - unreserved_; }

    void add_cpu_copy(Program& program, const Hsio& c, uint64_t src, uint64_t dst, uint32_t words, uint32_t local_flags) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_cpu_copy.cpp",
            c.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = kHart0});
        std::vector<uint32_t> args = split(src);
        const auto d = split(dst);
        args.insert(args.end(), d.begin(), d.end());
        args.insert(args.end(), {words, unreserved_ + kStatusOffset, local_flags});
        SetRuntimeArgs(program, k, c.logical, args);
    }

    static std::vector<uint32_t> pattern(uint32_t tag, uint32_t words) {
        std::vector<uint32_t> p(words);
        for (uint32_t i = 0; i < words; i++) {
            p[i] = (tag << 16) | i;
        }
        return p;
    }

    // CCE1 of HSIO 0 copies a block from CCE0's SRAM (its sibling window, through the uncached alias) to
    // `dst`, which the host then reads at `dst_core` offset `dst_offset`.
    void expect_cce1_copy(uint64_t dst, bool dst_local, const Hsio& dst_core, uint32_t dst_offset, uint32_t tag) {
        const auto cce0 = hsio_core(0, 0);
        const auto cce1 = hsio_core(0, 1);
        ASSERT_TRUE(cce0.has_value() && cce1.has_value());
        constexpr uint32_t kWords = 64;
        const auto data = pattern(tag, kWords);
        write_words(*cce0, kSrcOffset, data);
        write_words(dst_core, rel(dst_offset), std::vector<uint32_t>(kWords, 0));
        write_words(*cce1, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        Program program = CreateProgram();
        // The source is an offset in the sibling (CCE0) SRAM; the kernel adds the sibling window itself.
        add_cpu_copy(program, *cce1, local_addr(kSrcOffset), dst, kWords, 4u | (dst_local ? 2u : 0u));
        launch(program, *cce1);
        const auto status = read_words(*cce1, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(read_words(dst_core, rel(dst_offset), kWords), data);
    }
};

// CCE1 stores into its own SRAM through its uncached alias.
TEST_F(KeraunosCce1, Cce1HartWritesOwnSram) {
    const auto cce1 = hsio_core(0, 1);
    ASSERT_TRUE(cce1.has_value());
    expect_cce1_copy(kScratch, /*dst_local=*/true, *cce1, kScratch, 0xC1);
}

// CCE1 stores into HSIO 1's CCE1 SRAM through the package SPA (identity HSIO window).
TEST_F(KeraunosCce1, Cce1HartWritesOtherTileCce1Sram) {
    const auto peer = hsio_core(1, 1);
    ASSERT_TRUE(peer.has_value());
    const uint64_t spa = global_addr(*peer, rel(kScratch));
    expect_cce1_copy(spa, /*dst_local=*/false, *peer, kScratch, 0xC2);
}

// CCE1 runs an IDMA copy (its own command buffer) from CCE0's SRAM into its own, both as package SPAs.
TEST_F(KeraunosCce1, Cce1IdmaCopyCce0ToCce1) {
    const auto cce0 = hsio_core(0, 0);
    const auto cce1 = hsio_core(0, 1);
    ASSERT_TRUE(cce0.has_value() && cce1.has_value());
    constexpr uint32_t kBytes = 256;
    const auto data = pattern(0xC3, kBytes / sizeof(uint32_t));
    write_words(*cce0, kSrcOffset, data);
    write_words(*cce1, rel(kScratch), std::vector<uint32_t>(data.size(), 0));
    write_words(*cce1, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    add_idma_copy(
        program,
        *cce1,
        {.src = global_addr(*cce0, kSrcOffset), .dst = global_addr(*cce1, rel(kScratch)), .len = kBytes, .hart = kHart0});
    launch(program, *cce1);
    const auto status = read_words(*cce1, kStatusOffset, kStatusWords);
    EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    EXPECT_EQ(read_words(*cce1, rel(kScratch), data.size()), data);
}

// CCE0 and CCE1 of one tile hand a flag back and forth through CCE0's SRAM: the local sync a sender/receiver
// split (Blackhole erisc0/erisc1) needs. CCE0 reaches the flags in its own SRAM and CCE1 through its sibling
// window, in two programs on two cores launched together.
TEST_F(KeraunosCce1, Cce0Cce1PingPong) {
    const auto cce0 = hsio_core(0, 0);
    const auto cce1 = hsio_core(0, 1);
    ASSERT_TRUE(cce0.has_value() && cce1.has_value());
    constexpr uint32_t kRounds = 32;
    constexpr uint32_t kFlags = kSrcOffset;
    write_words(*cce0, kFlags, {0, 0});
    write_words(*cce0, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    write_words(*cce1, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    // role 0 is the sender (CCE0), role 1 the responder (CCE1); each core's local flag address differs.
    for (const auto& [core, role, flags_local] :
         {std::tuple{*cce0, 0u, unreserved_ + kFlags}, std::tuple{*cce1, 1u, unreserved_ + kFlags}}) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_peer_ping_pong.cpp",
            core.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = kHart0});
        SetRuntimeArgs(program, k, core.logical, {flags_local, role, kRounds, unreserved_ + kStatusOffset});
    }
    launch(program, *cce0);
    for (const Hsio& core : {*cce0, *cce1}) {
        const auto status = read_words(core, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[1], kRounds) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(status[4], kAcked);
    }
    EXPECT_EQ(read_words(*cce0, kFlags, 2), (std::vector<uint32_t>{kRounds, kRounds}));
}

}  // namespace
}  // namespace tt::tt_metal
