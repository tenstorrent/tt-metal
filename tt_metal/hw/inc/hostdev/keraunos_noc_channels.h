// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// The Ethernet channels of a Keraunos chip, for a CCE (a Quasar Ethernet core) that resolves NOC addresses itself
// (tt-2xx/quasar/cce_noc.h). The host writes it at MEM_IERISC_NOC_CHANNELS_BASE of every such core before launch,
// with KeraunosSpaRoutes (hostdev/keraunos_spa_routes.h), which holds the addresses.
//
// A NOC coordinate names an Ethernet core in one of two forms. A core's own coordinate is (KERAUNOS_NOC_CCE0_X +
// eth_cce, hsio tile), and cores says which of those are cores of the chip. An Ethernet channel's coordinate is
// (KERAUNOS_NOC_CHANNEL0_X + channel, KERAUNOS_NOC_CHANNEL_Y), the device's eth_chan_to_noc_xy, and the channel order
// is the chip's SoC descriptor's, so channels[] gives each channel's core. The device writes fault and fault_detail
// when it stops on a NOC call it cannot serve.
inline constexpr uint32_t KERAUNOS_NOC_CHANNELS_MAX = 12;
inline constexpr uint32_t KERAUNOS_NOC_CHANNELS_VALID = 0x4E414843;  // "CHAN"
inline constexpr uint32_t KERAUNOS_NOC_CORES_MAX = 32;               // channels[] entries cores has a bit for
inline constexpr uint32_t KERAUNOS_NOC_CCE0_X = 4;
inline constexpr uint32_t KERAUNOS_NOC_CHANNEL0_X = 20;
inline constexpr uint32_t KERAUNOS_NOC_CHANNEL_Y = 25;

struct KeraunosNocChannels {
    uint32_t valid;  // KERAUNOS_NOC_CHANNELS_VALID once written
    uint32_t num_channels;
    uint32_t fault;
    uint32_t cores;  // bit keraunos_noc_channel(hsio tile, eth_cce) set for each Ethernet core of the chip
    uint64_t fault_detail;
    uint8_t channels[KERAUNOS_NOC_CHANNELS_MAX];  // keraunos_noc_channel(hsio tile, eth_cce) of each channel
    uint8_t reserved_tail[4];
};

static_assert(sizeof(KeraunosNocChannels) == 40);
static_assert(offsetof(KeraunosNocChannels, fault) == 8);
static_assert(offsetof(KeraunosNocChannels, fault_detail) == 16);
static_assert(offsetof(KeraunosNocChannels, channels) == 24);

// A channels[] entry, and its two parts.
constexpr uint8_t keraunos_noc_channel(uint32_t hsio, uint32_t eth_cce) {
    return static_cast<uint8_t>((hsio << 1) | eth_cce);
}
constexpr uint32_t keraunos_noc_channel_hsio(uint8_t channel) { return channel >> 1; }
constexpr uint32_t keraunos_noc_channel_eth_cce(uint8_t channel) { return channel & 1; }

// The core the NOC coordinate (@p x, @p y) names on the chip of @p r, as a channels[] entry, in either form; -1 when
// @p r is not valid or the coordinate names no Ethernet core of the chip. Read-only.
inline int32_t keraunos_noc_core(const volatile KeraunosNocChannels& r, uint32_t x, uint32_t y) {
    if (r.valid != KERAUNOS_NOC_CHANNELS_VALID) {
        return -1;
    }
    // Unsigned differences: a coordinate below the first of its form wraps and fails the bound.
    const uint32_t channel = x - KERAUNOS_NOC_CHANNEL0_X;
    const uint32_t num_channels = r.num_channels;
    if (y == KERAUNOS_NOC_CHANNEL_Y && num_channels <= KERAUNOS_NOC_CHANNELS_MAX && channel < num_channels) {
        return r.channels[channel];
    }
    const uint32_t eth_cce = x - KERAUNOS_NOC_CCE0_X;
    if (eth_cce < 2 && y < KERAUNOS_NOC_CORES_MAX / 2 && ((r.cores >> keraunos_noc_channel(y, eth_cce)) & 1) != 0) {
        return keraunos_noc_channel(y, eth_cce);
    }
    return -1;
}
