// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CCE1 on a Keraunos HSIO tile (Phase 4 of keraunos_plan.md).
//
// One ethernet core is the whole tile: CCE0's SRAM, then CCE1's, MEM_IERISC_ETH_CCE_BASE(1) up. Processors
// interleave the CCEs, so EthernetConfig RISCV_1 is CCE1 hart 0 and RISCV_0 is CCE0 hart 0. Both CCEs see
// the tile in the same native layout (CCE0 SRAM at local 0, CCE1 SRAM at 4 MiB, uncached aliases 8 MiB
// up), so one local address means the same byte to either CCE and to the host's core offset.

#include "keraunos_cce_fixture.hpp"

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

constexpr auto kEthCce0Hart0 = DataMovementProcessor::RISCV_0;
constexpr auto kEthCce1Hart0 = DataMovementProcessor::RISCV_1;
// Scratch inside CCE1's SRAM, well above its firmware and kernels.
constexpr uint32_t kEthCce1Scratch = MEM_IERISC_ETH_CCE_BASE(1) + 0x200000;

class KeraunosCce1 : public KeraunosCceTest {
protected:
    // An offset relative to the unreserved base (what write_words/read_words take) for a core address.
    uint32_t rel(uint32_t core_offset) const { return core_offset - unreserved_; }

    void add_cpu_copy(Program& program, const Hsio& c, DataMovementProcessor hart, uint64_t src, uint64_t dst, uint32_t words, uint32_t local_flags) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_cpu_copy.cpp",
            c.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = hart});
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

    // CCE1 hart 0 copies a block from CCE0's SRAM (through the uncached alias) to `dst`, which the host
    // then reads at `dst_core` offset `dst_offset`.
    void expect_eth_cce1_copy(uint64_t dst, bool dst_local, const Hsio& dst_core, uint32_t dst_offset, uint32_t tag) {
        const auto c = hsio_core(0);
        ASSERT_TRUE(c.has_value());
        constexpr uint32_t kWords = 64;
        const auto data = pattern(tag, kWords);
        write_words(*c, kSrcOffset, data);
        write_words(dst_core, rel(dst_offset), std::vector<uint32_t>(kWords, 0));
        write_words(*c, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        Program program = CreateProgram();
        add_cpu_copy(program, *c, kEthCce1Hart0, local_addr(kSrcOffset), dst, kWords, dst_local ? 3u : 1u);
        launch(program, *c);
        const auto status = read_words(*c, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(read_words(dst_core, rel(dst_offset), kWords), data);
    }
};

// CCE1 stores into its own SRAM through its uncached alias (local 12 MiB = 4 MiB + 8 MiB).
TEST_F(KeraunosCce1, Cce1HartWritesOwnSram) {
    const auto c = hsio_core(0);
    ASSERT_TRUE(c.has_value());
    expect_eth_cce1_copy(kEthCce1Scratch, /*dst_local=*/true, *c, kEthCce1Scratch, 0xC1);
}

// CCE1 stores into HSIO 1's CCE1 SRAM through the package SPA (identity HSIO window).
TEST_F(KeraunosCce1, Cce1HartWritesOtherTileCce1Sram) {
    const auto peer = hsio_core(1);
    ASSERT_TRUE(peer.has_value());
    const uint64_t spa = global_addr(*peer, rel(kEthCce1Scratch));
    expect_eth_cce1_copy(spa, /*dst_local=*/false, *peer, kEthCce1Scratch, 0xC2);
}

// CCE1 runs an IDMA copy (its own command buffer) from CCE0's SRAM into its own, both as package SPAs.
TEST_F(KeraunosCce1, Cce1IdmaCopyCce0ToCce1) {
    const auto c = hsio_core(0);
    ASSERT_TRUE(c.has_value());
    constexpr uint32_t kBytes = 256;
    const auto data = pattern(0xC3, kBytes / sizeof(uint32_t));
    write_words(*c, kSrcOffset, data);
    write_words(*c, rel(kEthCce1Scratch), std::vector<uint32_t>(data.size(), 0));
    write_words(*c, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    add_idma_copy(
        program,
        *c,
        {.src = global_addr(*c, kSrcOffset), .dst = global_addr(*c, rel(kEthCce1Scratch)), .len = kBytes, .hart = kEthCce1Hart0});
    launch(program, *c);
    const auto status = read_words(*c, kStatusOffset, kStatusWords);
    EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    EXPECT_EQ(read_words(*c, rel(kEthCce1Scratch), data.size()), data);
}

// CCE0 hart 0 and CCE1 hart 0 hand a flag back and forth through CCE0's SRAM: the local sync a
// sender/receiver split (Blackhole erisc0/erisc1) needs.
TEST_F(KeraunosCce1, Cce0Cce1PingPong) {
    const auto c = hsio_core(0);
    ASSERT_TRUE(c.has_value());
    constexpr uint32_t kRounds = 32;
    constexpr uint32_t kFlags = kSrcOffset;
    constexpr uint32_t kResponderStatus = kStatusOffset + kStatusWords * sizeof(uint32_t);
    write_words(*c, kFlags, {0, 0});
    write_words(*c, kStatusOffset, std::vector<uint32_t>(2 * kStatusWords, 0));
    Program program = CreateProgram();
    for (const auto& [hart, role, status] :
         {std::tuple{kEthCce0Hart0, 0u, kStatusOffset}, std::tuple{kEthCce1Hart0, 1u, kResponderStatus}}) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_peer_ping_pong.cpp",
            c->logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = hart});
        SetRuntimeArgs(program, k, c->logical, {unreserved_ + kFlags, role, kRounds, unreserved_ + status});
    }
    launch(program, *c);
    for (const uint32_t status_offset : {kStatusOffset, kResponderStatus}) {
        const auto status = read_words(*c, status_offset, kStatusWords);
        EXPECT_EQ(status[1], kRounds) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(status[4], kAcked);
    }
    EXPECT_EQ(read_words(*c, kFlags, 2), (std::vector<uint32_t>{kRounds, kRounds}));
}

}  // namespace
}  // namespace tt::tt_metal
