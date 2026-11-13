// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar overlay libraries (tt_metal/hw/inc/internal/tt-2xx/quasar/overlay) on the Keraunos CCE.
//
// The Keraunos CCE is a Grendel DMRISC cluster. Its tt_cluster_ctrl / tt_rocc_accel register map matches
// tt-2.0.0 (the GRENDEL IP variant headers), so the same command buffer, address generator and RoCC
// builtins the Quasar DM cores use are expected to work here. These tests run them on an idle-ethernet
// hart and check every result from the host.
//
// One MeshDevice is opened for the whole suite: opening it on the MMK emulator costs about 90 s. Tests
// are ordered from the configuration SysEng firmware already uses to the ones still being explored, so a
// hang late in the suite does not hide the earlier results.

#include "keraunos_cce_fixture.hpp"

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

class KeraunosOverlay : public KeraunosCceTest {};

// SysEng's ker_cce_dma_m2k_remap sequence: write VCs, explicit trid, global SPA addresses for both ends.
TEST_F(KeraunosOverlay, IdmaCopyGlobalAddressesWriteVcs) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    expect_idma_copy({.src = global_addr(*core, kSrcOffset), .dst = global_addr(*core, kDstOffset), .len = 256}, 0xA1);
}

// The composite helper idma_copy_cmdbuf (reset, copy mode, write VCs, trid, issue).
TEST_F(KeraunosOverlay, IdmaCopyHelper) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    expect_idma_copy(
        {.src = global_addr(*core, kSrcOffset), .dst = global_addr(*core, kDstOffset), .len = 256, .helper = true},
        0xA2);
}

// Command buffer 1 runs the same copy.
TEST_F(KeraunosOverlay, IdmaCopyCmdBuf1) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    expect_idma_copy(
        {.src = global_addr(*core, kSrcOffset), .dst = global_addr(*core, kDstOffset), .len = 256, .cmdbuf = 1},
        0xA3);
}

// Several issues with source/destination auto-increment, gated on free space, then all_idma_acked.
TEST_F(KeraunosOverlay, IdmaCopyAutoIncrementAllAcked) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    expect_idma_copy(
        {.src = global_addr(*core, kSrcOffset),
         .dst = global_addr(*core, kDstOffset),
         .len = 512,
         .wait_all = true,
         .copies = 8},
        0xA4);
}

// The same copy with chiplet-local AXI addresses (the host's view) instead of package SPAs.
TEST_F(KeraunosOverlay, IdmaCopyChipletLocalAddresses) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    expect_idma_copy(
        {.src = chiplet_local_addr(*core, kSrcOffset), .dst = chiplet_local_addr(*core, kDstOffset), .len = 256},
        0xA5);
}

// Address generator 0 drives command buffer 0: every other 64-byte element of the source, packed.
TEST_F(KeraunosOverlay, AddrgenStridedIdma) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    constexpr uint32_t kElem = 64;
    constexpr uint32_t kCount = 16;
    const auto pattern = stage(*core, 0xA6, 2 * kElem * kCount);
    Program program = CreateProgram();
    const KernelHandle k = CreateKernel(
        program,
        std::string(kKernelDir) + "keraunos_cce_idma_addrgen_strided.cpp",
        core->logical,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
    std::vector<uint32_t> args = split(global_addr(*core, kSrcOffset));
    const auto dst = split(global_addr(*core, kDstOffset));
    args.insert(args.end(), dst.begin(), dst.end());
    args.insert(args.end(), {kElem, kCount, 2 * kElem, kElem, unreserved_ + kStatusOffset});
    SetRuntimeArgs(program, k, core->logical, args);
    launch(program, *core);

    const auto status = read_words(*core, kStatusOffset, kStatusWords);
    EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
    std::vector<uint32_t> expected;
    const uint32_t words_per_elem = kElem / sizeof(uint32_t);
    for (uint32_t i = 0; i < kCount; i++) {
        const auto first = pattern.begin() + 2 * i * words_per_elem;
        expected.insert(expected.end(), first, first + words_per_elem);
    }
    EXPECT_EQ(read_words(*core, kDstOffset, kCount * words_per_elem), expected);
}

// Every hart EthernetConfig can name (processors 0-7: harts 0-3 of CCE0 and of CCE1) issues its own copy
// at the same time. Each CPU has its own command buffer registers.
TEST_F(KeraunosOverlay, AllHartsIdmaCopyInParallel) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    constexpr uint32_t harts = 8;
    constexpr uint32_t kBytes = 256;
    const auto pattern = stage(*core, 0xA7, harts * kBytes);
    for (uint32_t h = 0; h < harts; h++) {
        write_words(*core, kStatusOffset + h * kStatusWords * 4, std::vector<uint32_t>(kStatusWords, 0));
    }
    Program program = CreateProgram();
    for (uint32_t h = 0; h < harts; h++) {
        add_idma_copy(
            program,
            *core,
            {.src = global_addr(*core, kSrcOffset + h * kBytes),
             .dst = global_addr(*core, kDstOffset + h * kBytes),
             .len = kBytes,
             .status_offset = kStatusOffset + h * kStatusWords * 4,
             .hart = static_cast<DataMovementProcessor>(h)});
    }
    launch(program, *core);
    for (uint32_t h = 0; h < harts; h++) {
        const auto status = read_words(*core, kStatusOffset + h * kStatusWords * 4, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << "hart " << h << fmt::format(" status: {:#010x}", fmt::join(status, " "));
    }
    EXPECT_EQ(read_words(*core, kDstOffset, harts * kBytes / sizeof(uint32_t)), pattern);
}

// Exploratory, last: the VC set the original GDDR read kernel used (NocVcs::READ, response VC 12). An IDMA
// copy is a write transaction; if its ack is expected on the write-response VC this one never completes.
TEST_F(KeraunosOverlay, IdmaCopyReadVcs) {
    const auto core = hsio_core(0);
    ASSERT_TRUE(core.has_value());
    expect_idma_copy(
        {.src = global_addr(*core, kSrcOffset), .dst = global_addr(*core, kDstOffset), .len = 256, .write_vcs = false},
        0xAF);
}

// Exploratory, last: IDMA L1 atomic accumulate with the API defaults (format 0, operation 9), the
// candidate remote-credit primitive (Blackhole fabric's atomic increment). The source block is all ones
// and is applied three times to a destination that starts at 100 + i, so an integer add leaves 103 + i.
// The test prints what lands either way: the encodings are not documented.
class KeraunosAtomics : public KeraunosCceTest {
protected:
    void expect_accumulate(uint32_t dst_hsio) {
        const auto a = hsio_core(0);
        const auto b = hsio_core(dst_hsio);
        ASSERT_TRUE(a.has_value() && b.has_value());
        constexpr uint32_t kWords = 16;
        constexpr uint32_t kRepeats = 3;
        std::vector<uint32_t> start(kWords);
        std::vector<uint32_t> expected(kWords);
        for (uint32_t i = 0; i < kWords; i++) {
            start[i] = 100 + i;
            expected[i] = 100 + i + kRepeats;
        }
        write_words(*a, kSrcOffset, std::vector<uint32_t>(kWords, 1));
        write_words(*b, kDstOffset, start);
        write_words(*a, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        Program program = CreateProgram();
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_idma_atomic_accum.cpp",
            a->logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        std::vector<uint32_t> args = split(global_addr(*a, kSrcOffset));
        const auto dst = split(global_addr(*b, kDstOffset));
        args.insert(args.end(), dst.begin(), dst.end());
        args.insert(args.end(), {kWords * 4, kRepeats, /*fmt=*/0, /*op=*/9, unreserved_ + kStatusOffset});
        SetRuntimeArgs(program, k, a->logical, args);
        launch(program, *a);
        const auto status = read_words(*a, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        const auto got = read_words(*b, kDstOffset, kWords);
        EXPECT_EQ(got, expected) << fmt::format("destination: {}", fmt::join(got, " "));
    }
};

// Disabled: on MMK emulation (2026-11-06) both complete and are acked, but the destination reads 1 1 1 ...:
// format 0 / operation 9 acts as a plain copy. The format and operation encodings for an accumulate are
// not documented; re-enable with the right values once SysEng confirms them.
TEST_F(KeraunosAtomics, DISABLED_IdmaAtomicAccumulateOwnSram) { expect_accumulate(0); }
TEST_F(KeraunosAtomics, DISABLED_IdmaAtomicAccumulateOtherTileSram) { expect_accumulate(1); }

}  // namespace
}  // namespace tt::tt_metal
