// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Noc*"
//
// The DRAM-read tests are arch-specific (`_WH` = 32 B rule, `_BH` = 64 B rule).
// Each one queries the device arch at runtime and GTEST_SKIPs when it doesn't
// match, so this bare filter is safe to run on either a wormhole or a blackhole
// cluster — the wrong-arch DRAM variants simply skip.

// The NOC alignment check is ABSOLUTE per-side: each endpoint must meet its own
// memory type's NoC alignment (L1 = 16 B, DRAM read = 32 B WH / 64 B BH, DRAM
// write = 16 B) — NOT the old relative "low bits of src and dst must match"
// rule. These four death tests each misalign exactly one endpoint:
//   read  L1 destination (16 B),  write L1 source (16 B),
//   read  DRAM source (32 B, WH), write DRAM destination (16 B).

#include <gtest/gtest.h>
#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// L1->L1 read: the L1 destination 0x30001 is not 16-byte aligned -> abort.
// (The source 0x30000 IS 16-aligned, so only the destination branch fires.)
TEST_F(MeshDeviceFixture, NocRead_L1_Misaligned_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // src NOC addr offset = 0x30000, 16-byte aligned (passes)
            uint64_t src = get_noc_addr(0x30000);
            // dst L1 offset 0x30001 -- low 4 bits = 1, not 16-byte aligned
            uint32_t dst = 0x30001;
            noc_async_read(src, dst, 16);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*NOC Transfer Alignment.*L1 destination.*must be 16-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// L1->L1 write: the L1 source 0x30001 is not 16-byte aligned -> abort. Misaligns
// the SOURCE (not the destination) so this exercises the write check's L1-source
// branch, which no other death test covers.
TEST_F(MeshDeviceFixture, NocWrite_L1_Misaligned_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // src L1 offset 0x30001 -- low 4 bits = 1, not 16-byte aligned
            uint32_t src = 0x30001;
            // dst NOC addr offset = 0x30000, 16-byte aligned (passes)
            uint64_t dst = get_noc_addr(0x30000);
            noc_async_write(src, dst, 16);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*NOC Transfer Alignment.*L1 source.*must be 16-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// L1->L1 read: the L1 SOURCE 0x30001 is not 16-byte aligned -> abort. Misaligns
// the source (destination aligned) so the read check's L1-source branch fires —
// the branch reached only when the destination passes AND the source resolves to
// L1 (not DRAM). No other read test exercises it (the DRAM tests take the
// DRAM-source arm; NocRead_L1_Misaligned aborts on the destination first).
TEST_F(MeshDeviceFixture, NocRead_L1_Misaligned_Source_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // src L1 offset 0x30001 -- low 4 bits = 1, not 16-byte aligned
            uint64_t src = get_noc_addr(0x30001);
            // dst L1 offset 0x30000 -- 16-byte aligned (passes, so the source arm is reached)
            uint32_t dst = 0x30000;
            noc_async_read(src, dst, 16);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*NOC Transfer Alignment.*L1 source.*must be 16-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// L1->L1 write: the L1 DESTINATION 0x30001 is not 16-byte aligned -> abort.
// Misaligns the destination (source aligned) so the write check's destination
// branch fires with its L1 label ("%s destination" -> "L1 destination"). No
// other write test exercises the L1-destination path (NocWrite_L1_Misaligned
// aborts on the source first; NocWrite_DRAM_Misaligned takes the DRAM label).
TEST_F(MeshDeviceFixture, NocWrite_L1_Misaligned_Dest_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // src L1 offset 0x30000 -- 16-byte aligned (passes, so the dest branch is reached)
            uint32_t src = 0x30000;
            // dst L1 offset 0x30001 -- low 4 bits = 1, not 16-byte aligned
            uint64_t dst = get_noc_addr(0x30001);
            noc_async_write(src, dst, 16);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*NOC Transfer Alignment.*L1 destination.*must be 16-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// DRAM->L1 read (WH, 32-byte rule, mask 0x1F): the DRAM source offset 0x10 is
// 16-byte aligned but NOT 32-byte aligned, so it fails the per-side DRAM-read
// alignment (WH = 32 B) and must abort. The L1 destination 0x30020 IS 16-byte
// aligned, so its branch passes and only the DRAM-source branch fires.
// Constructs the DRAM NOC address from the host-side NOC XY of DRAM bank 0.
TEST_F(MeshDeviceFixture, NocRead_DRAM_Misaligned_SanityCheck_WH) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    // WH-only: the 32 B DRAM-read rule is compiled into the JIT kernel only under
    // a wormhole cluster. Under blackhole the same offset aborts with the 64 B
    // message (regex mismatch), so gate to WH and SKIP elsewhere rather than fail
    // — this keeps the bare `Noc*` glob green on any arch.
    if (device->arch() != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "WH-only (32 B DRAM-read alignment); device arch is not wormhole.";
    }
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Get DRAM bank 0 NOC coordinates and build a NOC address with offset 0x10
    // (16-byte aligned, NOT 32-byte aligned).
    // NOC encoding: (y << 6) | x, shifted by 36 for the local-address field.
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) | static_cast<uint32_t>(dram_noc_coord.x);
    uint64_t dram_src = (static_cast<uint64_t>(noc_xy) << 36) | 0x0010ULL;
    uint32_t dram_lo = static_cast<uint32_t>(dram_src & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_src >> 32);

    // L1 dst is 16-byte aligned (passes). DRAM src offset 0x10 & 0x1F = 0x10 != 0,
    // so the per-side WH DRAM-read check (mask 0x1F, 32 B) aborts.
    uint32_t l1_dst = 0x30020;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t dst = get_arg_val<uint32_t>(2);
            uint64_t src = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_read(src, dst, 64);
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_dst});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*NOC Transfer Alignment.*DRAM source.*must be 32-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// DRAM->L1 read (BH, 64-byte rule, mask 0x3F): the DRAM source offset 0x20 is
// 32-byte aligned but NOT 64-byte aligned, so it fails the per-side DRAM-read
// alignment (BH = 64 B) and must abort. This is the blackhole counterpart of
// NocRead_DRAM_Misaligned_SanityCheck_WH: under blackhole_P100.yaml the kernel
// JIT-compiles with ARCH_BLACKHOLE, so the src mask is 0x3F and the message
// reads "64-byte aligned". The 0x20 offset would PASS the WH 32 B rule, so this
// test only fires under blackhole (the runners gate _WH/_BH by arch).
TEST_F(MeshDeviceFixture, NocRead_DRAM_Misaligned_SanityCheck_BH) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    // BH-only: the 64 B DRAM-read rule is compiled into the JIT kernel only under
    // a blackhole cluster. Under wormhole the 32 B rule accepts this 32-aligned
    // offset, so the read proceeds and aborts as an OOB write instead — gate to
    // BH and SKIP elsewhere so the bare `Noc*` glob stays green on any arch.
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "BH-only (64 B DRAM-read alignment); device arch is not blackhole.";
    }
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Get DRAM bank 0 NOC coordinates and build a NOC address with offset 0x20
    // (32-byte aligned, NOT 64-byte aligned).
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) | static_cast<uint32_t>(dram_noc_coord.x);
    uint64_t dram_src = (static_cast<uint64_t>(noc_xy) << 36) | 0x0020ULL;
    uint32_t dram_lo = static_cast<uint32_t>(dram_src & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_src >> 32);

    // L1 dst is 16-byte aligned (passes). DRAM src offset 0x20 & 0x3F = 0x20 != 0,
    // so the per-side BH DRAM-read check (mask 0x3F, 64 B) aborts.
    uint32_t l1_dst = 0x30020;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t dst = get_arg_val<uint32_t>(2);
            uint64_t src = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_read(src, dst, 64);
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_dst});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program), ".*NOC Transfer Alignment.*DRAM source.*must be 64-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// L1->DRAM write (WH/BH): the DRAM destination offset 0x01 is not 16-byte aligned
// (DRAM write alignment is 16 B on both arches) -> abort. The L1 source 0x30000
// IS 16-aligned, so only the destination branch fires.
TEST_F(MeshDeviceFixture, NocWrite_DRAM_Misaligned_SanityCheck) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Build DRAM NOC address with offset 0x01 so lower 4 bits = 1.
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) | static_cast<uint32_t>(dram_noc_coord.x);
    uint64_t dram_dst = (static_cast<uint64_t>(noc_xy) << 36) | 0x0001ULL;
    uint32_t dram_lo = static_cast<uint32_t>(dram_dst & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_dst >> 32);

    // L1 src: lower 4 bits = 0 -- mismatches DRAM lower 4 bits (1)
    uint32_t l1_src = 0x30000;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t src = get_arg_val<uint32_t>(2);
            uint64_t dst = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_write(src, dst, 16);
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_src});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*NOC Transfer Alignment.*DRAM destination.*must be 16-byte aligned.*");

    unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control (WH): a DRAM->L1 read where BOTH endpoints are 32-byte
// aligned (lower 5 bits = 0) but differ in higher bits MUST NOT abort. This is
// the exact pattern that the old, too-strict 0xFF mask (256-byte) wrongly
// flagged across the TT-NN sweeps (e.g. DRAM off 0x40 -> L1 off 0xa0: both
// 32-aligned, but 0x40 != 0xa0 under 0xFF). The corrected 0x1F mask compares
// only the lower 5 bits (0 == 0) and lets it through.
//
// Guards against a regression back to an over-strict mask: if the mask is ever
// widened past 0x1F, the two offsets below diverge and LaunchProgram SIGABRTs,
// failing this test. The DRAM source is a real DRAM-bank NOC-XY (so the NOC
// resolver maps it instead of raising a Fabric violation) and the L1 dest sits
// inside an allocated L1 buffer (so the dst-side OOB-tensor check passes).
TEST_F(MeshDeviceFixture, NocRead_DRAM_Aligned_NoViolation_WH) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    // WH-only positive control (guards the 32 B / 0x1F mask). Skip on other arches.
    if (device->arch() != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "WH-only positive control; device arch is not wormhole.";
    }
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Allocate a real L1 buffer so the read destination is a live tensor (no
    // OOB-tensor abort). Pick a 32-byte-aligned dst inside it whose lower 8
    // bits (0x20) differ from the DRAM source's (0x40) — that divergence is
    // exactly what the broken 0xFF mask used to reject.
    constexpr uint32_t buf_size = 8192;
    auto l1_buf = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    uint32_t base = static_cast<uint32_t>(l1_buf->address());
    uint32_t l1_dst = ((base + 0xFFu) & ~0xFFu) + 0x20u;  // 256-align, +0x20 -> low5=0, low8=0x20
    ASSERT_GE(l1_dst, base);
    ASSERT_LT(l1_dst + 32u, base + buf_size);

    // DRAM source on bank 0, offset 0x40 (32-byte aligned, low5=0).
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) | static_cast<uint32_t>(dram_noc_coord.x);
    constexpr uint32_t dram_off = 0x40u;
    uint64_t dram_src = (static_cast<uint64_t>(noc_xy) << 36) | dram_off;
    uint32_t dram_lo = static_cast<uint32_t>(dram_src & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_src >> 32);

    // Same lower 5 bits (both 0) -> 32-byte rule satisfied; different lower 8
    // bits -> the old 256-byte mask would have falsely aborted.
    ASSERT_EQ(dram_off & 0x1Fu, l1_dst & 0x1Fu);
    ASSERT_NE(dram_off & 0xFFu, l1_dst & 0xFFu);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t dst = get_arg_val<uint32_t>(2);
            uint64_t src = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_read(src, dst, 32);
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_dst});

    // Must NOT abort. If the alignment mask regresses to something stricter than
    // 0x1F, this LaunchProgram SIGABRTs and the harness marks the test failed.
    detail::LaunchProgram(device, program);
    SUCCEED();

    unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control (BH): a DRAM->L1 read whose DRAM source is 64-byte aligned
// (lower 6 bits = 0) MUST NOT abort, even though it differs from the L1 dst in
// higher bits. Blackhole counterpart of NocRead_DRAM_Aligned_NoViolation_WH,
// guarding the corrected 0x3F mask: if the BH DRAM-read mask ever regresses to
// something stricter than 0x3F (e.g. a relative 0xFF), the 0x40 source offset
// diverges from the 0x80 L1 offset under that wider mask and LaunchProgram
// SIGABRTs, failing this test. As with the WH control, the DRAM source is a real
// DRAM-bank NOC-XY (mapped by the NOC resolver, not a Fabric violation) and the
// L1 dest sits inside an allocated L1 buffer (so the dst-side OOB check passes).
TEST_F(MeshDeviceFixture, NocRead_DRAM_Aligned_NoViolation_BH) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh = this->devices_.at(0);
    auto* device = mesh->get_devices()[0];
    // BH-only positive control (guards the 64 B / 0x3F mask). Skip on other arches.
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "BH-only positive control; device arch is not blackhole.";
    }
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Allocate a real L1 buffer so the read destination is a live tensor (no
    // OOB-tensor abort). Pick a 64-byte-aligned dst inside it whose lower 8 bits
    // (0x80) differ from the DRAM source's (0x40) — that divergence is exactly
    // what an over-strict relative mask would wrongly reject.
    constexpr uint32_t buf_size = 8192;
    auto l1_buf = Buffer::create(device, buf_size, buf_size, BufferType::L1);
    uint32_t base = static_cast<uint32_t>(l1_buf->address());
    uint32_t l1_dst = ((base + 0xFFu) & ~0xFFu) + 0x80u;  // 256-align, +0x80 -> low6=0, low8=0x80
    ASSERT_GE(l1_dst, base);
    ASSERT_LT(l1_dst + 64u, base + buf_size);

    // DRAM source on bank 0, offset 0x40 (64-byte aligned, low6=0).
    auto dram_noc_coord = mesh->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    uint32_t noc_xy = (static_cast<uint32_t>(dram_noc_coord.y) << 6) | static_cast<uint32_t>(dram_noc_coord.x);
    constexpr uint32_t dram_off = 0x40u;
    uint64_t dram_src = (static_cast<uint64_t>(noc_xy) << 36) | dram_off;
    uint32_t dram_lo = static_cast<uint32_t>(dram_src & 0xFFFFFFFFU);
    uint32_t dram_hi = static_cast<uint32_t>(dram_src >> 32);

    // Same lower 6 bits (both 0) -> 64-byte rule satisfied; different lower 8
    // bits -> a mask stricter than 0x3F would have falsely aborted.
    ASSERT_EQ(dram_off & 0x3Fu, l1_dst & 0x3Fu);
    ASSERT_NE(dram_off & 0xFFu, l1_dst & 0xFFu);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t lo  = get_arg_val<uint32_t>(0);
            uint32_t hi  = get_arg_val<uint32_t>(1);
            uint32_t dst = get_arg_val<uint32_t>(2);
            uint64_t src = (static_cast<uint64_t>(hi) << 32) | lo;
            noc_async_read(src, dst, 64);
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {dram_lo, dram_hi, l1_dst});

    // Must NOT abort. If the alignment mask regresses to something stricter than
    // 0x3F, this LaunchProgram SIGABRTs and the harness marks the test failed.
    detail::LaunchProgram(device, program);
    SUCCEED();

    unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control (L1 16B): a legitimately 16-byte-aligned L1->L1 read must NOT
// abort. Guards the L1 alignment rule against over-tightening (e.g. a regression
// to a 32 B mask on the L1 side) — nothing else asserts a correct L1 transfer
// passes, since the other positive controls are DRAM-source reads. Placed here,
// with the other non-death controls (after every death test), so the death
// tests' EXPECT_DEATH forks all precede any non-death LaunchProgram.
TEST_F(MeshDeviceFixture, NocRead_L1_Aligned_NoViolation) {
    setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Use in-bounds addresses of a real L1 buffer: unlike the misalignment death
    // tests (which abort before any resolution), this transfer actually runs, so
    // both endpoints must resolve to a live tensor or the OOB check (§4) would
    // fire for the wrong reason. The allocator aligns buffer starts to >=16 B, so
    // base and base+16 are both 16-byte aligned.
    constexpr uint32_t buffer_size = 1024;
    auto buf = Buffer::create(device, buffer_size, buffer_size, BufferType::L1);
    uint32_t base = static_cast<uint32_t>(buf->address());
    ASSERT_EQ(base & 0xF, 0u);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t b = get_arg_val<uint32_t>(0);
            uint64_t src = get_noc_addr(b);          // L1 source, 16-aligned
            uint32_t dst = b + 16;                    // L1 dest, 16-aligned, in-bounds
            noc_async_read(src, dst, 16);
            noc_async_read_barrier();
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {base});

    // Must NOT abort on the alignment check.
    detail::LaunchProgram(device, program);
    SUCCEED();

    unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
