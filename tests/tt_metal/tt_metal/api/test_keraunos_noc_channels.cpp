// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// keraunos_noc_core(), the rule a Keraunos CCE resolves a NOC coordinate by, over the records of the three chips of
// the vcluster line: A on HSIO tile 0, M on tiles 1 and 2, B on tile 3.

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>

#include "hostdev/keraunos_noc_channels.h"

namespace {

// The record of a chip whose Ethernet channels are `channels`, in order, as the host writes it.
KeraunosNocChannels record_of(std::initializer_list<uint8_t> channels) {
    KeraunosNocChannels record{};
    for (const uint8_t channel : channels) {
        record.channels[record.num_channels++] = channel;
        record.cores |= 1u << channel;
    }
    record.valid = KERAUNOS_NOC_CHANNELS_VALID;
    return record;
}

constexpr uint32_t kCce0X = KERAUNOS_NOC_CCE0_X;
constexpr uint32_t kCce1X = KERAUNOS_NOC_CCE0_X + 1;
constexpr uint32_t kChannel0X = KERAUNOS_NOC_CHANNEL0_X;
constexpr uint32_t kChannelY = KERAUNOS_NOC_CHANNEL_Y;

TEST(KeraunosNocChannels, CoreFormNamesEveryCoreOfTheChipOnAnyTile) {
    const KeraunosNocChannels a = record_of({keraunos_noc_channel(0, 0), keraunos_noc_channel(0, 1)});
    const KeraunosNocChannels m = record_of({keraunos_noc_channel(1, 0), keraunos_noc_channel(2, 0)});
    const KeraunosNocChannels b = record_of({keraunos_noc_channel(3, 0), keraunos_noc_channel(3, 1)});

    EXPECT_EQ(keraunos_noc_core(a, kCce0X, 0), keraunos_noc_channel(0, 0));
    EXPECT_EQ(keraunos_noc_core(a, kCce1X, 0), keraunos_noc_channel(0, 1));
    // A tile index at or above the number of the chip's tiles.
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, 1), keraunos_noc_channel(1, 0));
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, 2), keraunos_noc_channel(2, 0));
    EXPECT_EQ(keraunos_noc_core(b, kCce0X, 3), keraunos_noc_channel(3, 0));
    EXPECT_EQ(keraunos_noc_core(b, kCce1X, 3), keraunos_noc_channel(3, 1));
}

TEST(KeraunosNocChannels, CoreFormRefusesWhatIsNoCoreOfTheChip) {
    const KeraunosNocChannels m = record_of({keraunos_noc_channel(1, 0), keraunos_noc_channel(2, 0)});
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, 0), -1);  // chip A's tile
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, 3), -1);  // chip B's tile
    EXPECT_EQ(keraunos_noc_core(m, kCce1X, 1), -1);  // a CCE the chip does not list
    EXPECT_EQ(keraunos_noc_core(m, kCce0X - 1, 1), -1);
    EXPECT_EQ(keraunos_noc_core(m, kCce1X + 1, 1), -1);
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, KERAUNOS_NOC_CORES_MAX / 2), -1);  // no bit in cores
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, KERAUNOS_NOC_CORES_MAX / 2 + 1), -1);
}

TEST(KeraunosNocChannels, ChannelFormFollowsTheChipsChannelOrder) {
    const KeraunosNocChannels m = record_of({keraunos_noc_channel(2, 0), keraunos_noc_channel(1, 0)});
    EXPECT_EQ(keraunos_noc_core(m, kChannel0X, kChannelY), keraunos_noc_channel(2, 0));
    EXPECT_EQ(keraunos_noc_core(m, kChannel0X + 1, kChannelY), keraunos_noc_channel(1, 0));
    EXPECT_EQ(keraunos_noc_core(m, kChannel0X + 2, kChannelY), -1);
    EXPECT_EQ(keraunos_noc_core(m, kChannel0X - 1, kChannelY), -1);
    EXPECT_EQ(keraunos_noc_core(m, kChannel0X, kChannelY + 1), -1);
}

TEST(KeraunosNocChannels, AnUnwrittenRecordNamesNothing) {
    KeraunosNocChannels m = record_of({keraunos_noc_channel(1, 0)});
    m.valid = 0;
    EXPECT_EQ(keraunos_noc_core(m, kCce0X, 1), -1);
    EXPECT_EQ(keraunos_noc_core(m, kChannel0X, kChannelY), -1);
}

}  // namespace
