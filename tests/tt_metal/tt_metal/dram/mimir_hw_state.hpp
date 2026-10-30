// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test {

// Read-only. Nothing in this header or its .cpp writes to the device.

// SMC core for a Mimir, in TRANSLATED coordinates (M0 first, then M1).
CoreCoord mimir_smc_core(const metal_SocDescriptor& soc, uint32_t mimir_index);
// Non-empty when the watcher is enabled, since its host NOC sanitizer rejects SMC core reads. Call before any SMC read.
std::string mimir_watcher_skip_reason();

struct CceResetState {
    uint32_t cce_index = 0;
    uint32_t raw_lo = 0;
    bool uncore_released = false;  // bit 0, 1 = released
    uint32_t core_reset = 0;       // bits [8:1], 1 = hart released
};
std::vector<CceResetState> read_cce_reset_states(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);  // one per CCE
std::vector<uint64_t> read_cce_reset_vectors(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip, uint32_t cce_index);
// "" = proceed. Otherwise a skip reason: watcher enabled; any CCE core_reset != 0; or (require_uncore_released && any
// uncore bit 0).
std::string host_only_guard_reason(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip, bool require_uncore_released);
#define MIMIR_HOST_ONLY_GUARD(cluster, soc, chip, require_uncore_released)                                   \
    if (const std::string mimir_guard_reason =                                                               \
            ::tt::tt_metal::mimir_test::host_only_guard_reason(cluster, soc, chip, require_uncore_released); \
        !mimir_guard_reason.empty()) {                                                                       \
        GTEST_SKIP() << mimir_guard_reason;                                                                  \
    }

struct MimirRegister {
    std::string name;
    uint32_t mimir_index = 0;
    uint64_t address = 0;
    uint64_t value = 0;
    bool is_64bit = false;
};
// Setup registers only: SS_BAR, ATT and GDDR interleave. No DMRISC registers.
std::vector<MimirRegister> read_d2d_setup_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);
std::vector<MimirRegister> expected_d2d_setup_config(const metal_SocDescriptor& soc);
// GDDR interleave and capacity status registers only.
std::vector<MimirRegister> read_gddr_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);
// CCE DMRISC remap entries only: start, end, remap_start and attrs of entries 0-2 in each CCE.
std::vector<MimirRegister> read_cce_remap_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);
std::vector<MimirRegister> expected_cce_remap_config(const metal_SocDescriptor& soc);
// Setup plus CCE remap: every register in the D2D layout.
std::vector<MimirRegister> read_d2d_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);
std::vector<MimirRegister> expected_d2d_config(const metal_SocDescriptor& soc);
// Returns one line per mismatching register. CCE remap attrs and ATT.ENABLE compare only bit 0; other bits are logged.
std::vector<std::string> diff_d2d_config(
    const std::vector<MimirRegister>& expected, const std::vector<MimirRegister>& actual);
// Identity check: "" when ATT endpoint entry 64 == (4,6) and SS_BAR1 base == 0x8000_0000 + inst*8MiB on every
// live tile; otherwise the mismatch description.
std::string mimir_identity_mismatch(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);
// ATT endpoint-table entry a CCE copy is predicted to use, from the expected table in mimir_hw_state.cpp.
struct AttEndpointPrediction {
    uint32_t entry_index = 0;
    uint64_t value = 0;  // expected x | (y << 6)
};
AttEndpointPrediction att_endpoint_for_remote_sram(uint32_t dest_cce);  // entry 64 + destination CCE
// Entry for a GDDR copy from source_channel over d2d_port, the port the source CCE's CceSpaRoutes picks for the
// destination. nullopt when that (Mimir, port) pair is not in the MMK table (0 over D2D1 -> 113, 1 over D2D0 -> 96).
std::optional<AttEndpointPrediction> att_endpoint_for_remote_gddr(uint32_t source_channel, uint32_t d2d_port);
// Live D2D tiles: 1x1 -> only M0 D2D0; 2x -> all four. Never read a non-live tile.
struct D2dTile {
    uint32_t mimir_index = 0;
    uint32_t d2d_index = 0;
};  // d2d_index 0/1; D2D1 registers are +0x200000
std::vector<D2dTile> live_d2d_tiles(const metal_SocDescriptor& soc);
struct D2dErrorSnapshot {
    std::vector<MimirRegister> debug_misc;
    std::vector<MimirRegister> axi_errors;
};
D2dErrorSnapshot snapshot_d2d_errors(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);
struct D2dErrorDelta {
    std::vector<std::string> new_bits;
    std::vector<std::string> inconclusive_bits;
};  // inconclusive = already set before
D2dErrorDelta new_error_bits(const D2dErrorSnapshot& before, const D2dErrorSnapshot& after);
// Logs inconclusive bits and adds a failure for each new bit. Returns true when there are no new bits.
bool expect_no_new_d2d_error_bits(
    const D2dErrorSnapshot& before, const D2dErrorSnapshot& after, const std::string& context);
std::vector<MimirRegister> read_debug_last(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip);  // DEBUG_LAST_SRC/DEST/TRANSLATION on live tiles
void dump_debug_last(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip);  // log only; call on failure

// Host suites. Cached per process. "" = identity, setup and CCE remap match the Chippy values; otherwise the mismatch.
// Uses MetalContext::instance().get_cluster(), so no initialize(). Call through MIMIR_REQUIRE_IDENTITY.
const std::string& mimir_identity_result();
// Remote suites, before CreateDevice. Cached per process.
// "" = identity, SS_BAR, ATT and interleave match; otherwise the mismatch. No DMRISC registers. Call through
// MIMIR_REQUIRE_REMOTE_PREFLIGHT.
const std::string& mimir_remote_preflight_result();
// Skip when the watcher is enabled (before any SMC read), then fail the test on a mismatch. Use before any device is
// opened.
#define MIMIR_REQUIRE_IDENTITY()                                                                          \
    if (const std::string mimir_watcher_reason = ::tt::tt_metal::mimir_test::mimir_watcher_skip_reason(); \
        !mimir_watcher_reason.empty()) {                                                                  \
        GTEST_SKIP() << mimir_watcher_reason;                                                             \
    }                                                                                                     \
    ASSERT_TRUE(::tt::tt_metal::mimir_test::mimir_identity_result().empty())                              \
        << ::tt::tt_metal::mimir_test::mimir_identity_result()
#define MIMIR_REQUIRE_REMOTE_PREFLIGHT()                                                                  \
    if (const std::string mimir_watcher_reason = ::tt::tt_metal::mimir_test::mimir_watcher_skip_reason(); \
        !mimir_watcher_reason.empty()) {                                                                  \
        GTEST_SKIP() << mimir_watcher_reason;                                                             \
    }                                                                                                     \
    ASSERT_TRUE(::tt::tt_metal::mimir_test::mimir_remote_preflight_result().empty())                      \
        << ::tt::tt_metal::mimir_test::mimir_remote_preflight_result()
// Remote suites, after CreateDevice. Chippy bring-up programs the DMRISC remap entries and nothing rewrites them.
// Cached per process. Only valid after CreateDevice; call it once mimir_remote_preflight_result() is empty. "" = remap
// start, end, remap_start and valid bit match UMD's values.
const std::string& mimir_remote_remap_result();

}  // namespace tt::tt_metal::mimir_test
