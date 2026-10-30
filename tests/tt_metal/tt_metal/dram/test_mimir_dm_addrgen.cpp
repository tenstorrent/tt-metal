// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-8 quasar_addrgen / quasar_im2col port: address generator 0 on a CCE runs each upstream example pattern and pops
// its addresses into own TL1; the host compares them with the hardware loop of addrgen_api.hpp. Upstream only prints
// the addresses. Interleaved_Banking is not ported (it needs the NoC bank table).

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_addrgen_abi.h"

namespace tt::tt_metal::mimir_test::dm_addrgen {

namespace abi = cce_addrgen_abi;

inline constexpr const char* kAddrgenKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_addrgen.cpp";
inline constexpr uint32_t kCce = 0;
inline constexpr uint32_t kHart = kCceKernelHart;
// The kernel has no polls; this covers launch on a slow emulator.
inline constexpr std::chrono::seconds kHostDeadline{300};
inline constexpr uint32_t kResultPoison = 0xADD60000u;

struct AddrgenCase {
    const char* name;
    uint32_t pattern;
    bool src_strided;
    bool dest_strided;
    uint32_t num_addresses;
};

std::string case_name(const ::testing::TestParamInfo<AddrgenCase>& info) { return info.param.name; }

// The hardware loop (addrgen_api.hpp): every pop advances the inner loop; on reaching its end the inner index restarts
// at its start (0 here) and the outer loop advances; on the outer end the outer index restarts and the base advances by
// the face size. The addrgen_api.hpp comment says the index becomes (index + stride) mod end; on a CCE it restarts
// (Im2Col_Dilation1, whose ends are not a multiple of the stride).
struct SideModel {
    abi::SideSpec spec;
    uint64_t base = 0;
    uint64_t inner = 0;
    uint64_t outer = 0;

    SideModel(const abi::SideSpec& side, bool strided) :
        spec(strided ? side : abi::SideSpec{.base = side.base, .inner = {0, 0}, .outer = {0, 0}, .face_size = 0}),
        base(side.base) {}

    uint64_t pop() {
        const uint64_t address = base + outer + inner;
        inner += spec.inner.stride;
        if (inner >= spec.inner.end) {
            inner = 0;
            outer += spec.outer.stride;
            if (outer >= spec.outer.end) {
                outer = 0;
                base += spec.face_size;
            }
        }
        return address;
    }
};

struct AddressPair {
    uint64_t src = 0;
    uint64_t dest = 0;
};

std::vector<AddressPair> predicted_addresses(const AddrgenCase& addrgen_case) {
    const abi::PatternSpec& pattern = abi::kPatterns[addrgen_case.pattern];
    SideModel src(pattern.src, addrgen_case.src_strided);
    SideModel dest(pattern.dest, addrgen_case.dest_strided);
    std::vector<AddressPair> addresses;
    const uint32_t patch_pops = pattern.patch_pops == 0 ? addrgen_case.num_addresses : pattern.patch_pops;
    const uint32_t patch_row_count = pattern.patch_pops == 0 ? 1 : pattern.patch_row_count;
    const uint32_t patch_column_count = pattern.patch_pops == 0 ? 1 : pattern.patch_column_count;
    for (uint32_t patch_row = 0; patch_row < patch_row_count; ++patch_row) {
        for (uint32_t patch_column = 0; patch_column < patch_column_count; ++patch_column) {
            if (pattern.patch_pops != 0 && addrgen_case.src_strided) {
                src.base = pattern.src.base + patch_row * pattern.patch_row_stride +
                           patch_column * pattern.patch_column_stride;
            }
            for (uint32_t pop = 0; pop < patch_pops && addresses.size() < addrgen_case.num_addresses; ++pop) {
                const uint64_t src_address = src.pop();
                addresses.push_back({src_address, dest.pop()});
            }
        }
    }
    return addresses;
}

uint64_t result_word64(const std::vector<uint32_t>& words, uint32_t index64) {
    return (static_cast<uint64_t>(words[index64 * 2 + 1]) << 32) | words[index64 * 2];
}

void run_case(const AddrgenCase& addrgen_case) {
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();

    const uint64_t status_offset = tl1_done_block(kHart);
    const uint64_t result_offset = tl1_data(kHart);
    const uint32_t result_words = addrgen_case.num_addresses * abi::kResultEntryWords64 * 2;
    write_cce_tl1(device, kCce, result_offset, std::vector<uint32_t>(result_words, kResultPoison));

    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = abi::kModePattern;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(status_offset);
    args[abi::kArgResultOffset] = static_cast<uint32_t>(result_offset);
    args[abi::kArgPattern] = addrgen_case.pattern;
    args[abi::kArgSrcStrideEnable] = addrgen_case.src_strided ? 1 : 0;
    args[abi::kArgDestStrideEnable] = addrgen_case.dest_strided ? 1 : 0;
    args[abi::kArgNumAddresses] = addrgen_case.num_addresses;
    const std::string label =
        fmt::format("{} on CCE{} hart {} ({} addresses)", addrgen_case.name, kCce, kHart, addrgen_case.num_addresses);
    log_info(tt::LogTest, "MM8 addrgen {}", label);

    const CceKernelLaunch launch{
        .cce_index = kCce,
        .kernel_path = kAddrgenKernel,
        .runtime_args = args,
        .hart = kHart,
        .status_line = {.done_magic = abi::kDoneMagic, .num_words = abi::kStatusWords, .poison = abi::kStatusPoison}};
    Program program = CreateProgram();
    const CceStatuses statuses = run_cce_kernels_with_deadline(device, program, {launch}, kHostDeadline);
    ASSERT_TRUE(statuses.has_value()) << label << ": " << statuses.no_status();
    const std::vector<uint32_t>& status = statuses->front();
    ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone)
        << label << ": kernel code 0x" << std::hex << status[abi::kStatusCode];

    const std::vector<uint32_t> results = read_cce_tl1(device, kCce, result_offset, result_words * sizeof(uint32_t));
    const std::vector<AddressPair> predicted = predicted_addresses(addrgen_case);
    uint32_t mismatches = 0;
    std::string first_mismatch;
    for (uint32_t address = 0; address < addrgen_case.num_addresses; ++address) {
        const uint64_t src = result_word64(results, address * abi::kResultEntryWords64 + abi::kResultSrc);
        const uint64_t dest = result_word64(results, address * abi::kResultEntryWords64 + abi::kResultDest);
        if (src != predicted[address].src || dest != predicted[address].dest) {
            if (mismatches == 0) {
                first_mismatch = fmt::format(
                    "address {}: src 0x{:x} (predicted 0x{:x}) dest 0x{:x} (predicted 0x{:x})",
                    address,
                    src,
                    predicted[address].src,
                    dest,
                    predicted[address].dest);
            }
            mismatches++;
        }
    }
    log_info(
        tt::LogTest,
        "MM8 addrgen {}: {} of {} address pairs match the loop model in {} cycles{}{}",
        label,
        addrgen_case.num_addresses - mismatches,
        addrgen_case.num_addresses,
        status[abi::kStatusPopCycles],
        mismatches == 0 ? "" : "; first mismatch ",
        first_mismatch);
    EXPECT_EQ(mismatches, 0u) << label << ": first mismatch " << first_mismatch;
}

class MimirDmAddrgen : public ::testing::TestWithParam<AddrgenCase> {};
class MimirDmIm2col : public ::testing::TestWithParam<AddrgenCase> {};

TEST_P(MimirDmAddrgen, Pattern) {
    skip_unless_programmable_dram_package();
    MIMIR_SKIP_IF_WATCHER();
    run_case(GetParam());
}

TEST_P(MimirDmIm2col, Pattern) {
    skip_unless_programmable_dram_package();
    MIMIR_SKIP_IF_WATCHER();
    run_case(GetParam());
}

// The upstream QuasarAddrgenOps cases, Interleaved_Banking excepted.
INSTANTIATE_TEST_SUITE_P(
    MM8,
    MimirDmAddrgen,
    ::testing::Values(
        AddrgenCase{"Strided1D_SrcOnly", abi::kPattern1d, true, false, 10},
        AddrgenCase{"Strided1D_DstOnly", abi::kPattern1d, false, true, 10},
        AddrgenCase{"Strided1D_Both", abi::kPattern1d, true, true, 10},
        AddrgenCase{"Strided2D_SrcOnly", abi::kPattern2d, true, false, 16},
        AddrgenCase{"Strided2D_DstOnly", abi::kPattern2d, false, true, 16},
        AddrgenCase{"Strided2D_Both", abi::kPattern2d, true, true, 16},
        AddrgenCase{"Face_SrcOnly", abi::kPatternFace, true, false, 32},
        AddrgenCase{"Face_DstOnly", abi::kPatternFace, false, true, 32},
        AddrgenCase{"Face_Both", abi::kPatternFace, true, true, 32}),
    case_name);

// The upstream QuasarIm2ColOps cases.
INSTANTIATE_TEST_SUITE_P(
    MM8,
    MimirDmIm2col,
    ::testing::Values(
        AddrgenCase{"Im2Col_FullImage", abi::kPatternIm2col, true, true, 81},
        AddrgenCase{"Im2Col_Dilation1", abi::kPatternIm2colDilation1, true, true, 180}),
    case_name);

}  // namespace tt::tt_metal::mimir_test::dm_addrgen
