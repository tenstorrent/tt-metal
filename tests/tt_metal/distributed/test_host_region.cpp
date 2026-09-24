// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-only half of HostRegion: the mapping, the alias bookkeeping an overlay declares
// against, and the header self-check. No device and no MPI, so this runs in ordinary CI.
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "tt_metal/distributed/host_region.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"

namespace tt::tt_metal::experimental {
namespace {

// One region per process, and reserved_base() refuses to resize, so the suite fixes a core
// count once and every case works against that same mapping.
constexpr uint32_t kTestCores = 4;

HostRegion& reserved() {
    HostRegion& r = HostRegion::storage();
    static uint8_t* const base = r.reserved_base(kTestCores);
    EXPECT_NE(base, nullptr);
    return r;
}

TEST(HostRegion, ReservedBaseIsIdempotentAnd2MiBAligned) {
    uint8_t* const first = reserved().reserved_base(kTestCores);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(reserved().reserved_base(kTestCores), first);
    // MADV_HUGEPAGE only acts on a range it covers completely.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(first) % kAlign2M, 0u);
}

TEST(HostRegion, ReservedBaseRefusesAResize) {
    // Growing would move the base, and every overlay already placed against the old one
    // would go on naming pages nothing owns.
    EXPECT_THROW(reserved().reserved_base(kTestCores + 1), std::runtime_error);
    EXPECT_EQ(reserved().reserved_cores(), kTestCores);
}

TEST(HostRegion, MappedExtentCoversTheCoresAskedFor) {
    EXPECT_GE(reserved().region_bytes(), pinned_bytes_for(kTestCores));
    EXPECT_EQ(reserved().region_bytes() % kAlign2M, 0u);
}

TEST(HostRegion, AnUndeclaredAliasLeavesTheWholeArenaFillable) {
    reserved().clear_aliases(AliasArena::Tx);
    EXPECT_EQ(reserved().alias_fill_bytes(AliasArena::Tx, 0), kArenaBytes);
    EXPECT_EQ(reserved().alias_tail_offset(AliasArena::Tx, 0), kArenaBytes);
}

TEST(HostRegion, DeclareAliasRoundTrips) {
    reserved().clear_aliases(AliasArena::Rx);
    reserved().declare_alias(AliasArena::Rx, 1, 4096, 8192);
    EXPECT_EQ(reserved().alias_fill_bytes(AliasArena::Rx, 1), 4096u);
    EXPECT_EQ(reserved().alias_tail_offset(AliasArena::Rx, 1), 8192u);
    reserved().clear_aliases(AliasArena::Rx);
}

TEST(HostRegion, AZeroFillRetractsTheDeclaration) {
    reserved().clear_aliases(AliasArena::Tx);
    reserved().declare_alias(AliasArena::Tx, 2, 4096, 8192);
    reserved().declare_alias(AliasArena::Tx, 2, 0, 8192);
    EXPECT_EQ(reserved().alias_fill_bytes(AliasArena::Tx, 2), kArenaBytes);
    EXPECT_EQ(reserved().alias_tail_offset(AliasArena::Tx, 2), kArenaBytes);
}

TEST(HostRegion, ClearAliasesTouchesOneArenaOnly) {
    reserved().declare_alias(AliasArena::Tx, 0, 4096, 4096);
    reserved().declare_alias(AliasArena::Rx, 0, 8192, 8192);
    reserved().clear_aliases(AliasArena::Tx);
    EXPECT_EQ(reserved().alias_fill_bytes(AliasArena::Tx, 0), kArenaBytes);
    EXPECT_EQ(reserved().alias_fill_bytes(AliasArena::Rx, 0), 8192u);
    reserved().clear_aliases(AliasArena::Rx);
}

TEST(HostRegion, DeclareAliasRefusesASpanItCannotHonour) {
    // Refused, not clamped: a clamp would quietly fill part of a ring's metadata.
    EXPECT_THROW(reserved().declare_alias(AliasArena::Tx, 0, 8192, 4096), std::runtime_error);
    EXPECT_THROW(reserved().declare_alias(AliasArena::Tx, 0, 4096, kArenaBytes + 1), std::runtime_error);
    // Against what was mapped, not kProvisionedCores: an arena past it has no pages behind it.
    EXPECT_THROW(reserved().declare_alias(AliasArena::Tx, kTestCores, 4096, 4096), std::runtime_error);
}

TEST(HostRegion, VerifyHeaderRejectsAnUnprovisionedRegion) {
    // The magic is the gate a peer polls, so the self-check must report its absence rather
    // than read the geometry behind it.
    EXPECT_FALSE(reserved().is_provisioned());
    EXPECT_FALSE(reserved().verify_header().empty());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
