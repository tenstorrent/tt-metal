// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// L1 layout and runtime-argument construction for the ttnvtop aggregator.
//
// ONE definition, shared by the two sides that must agree exactly:
//   - the tt-metal-linked producer that emits the launch artifact, and
//   - the UMD-only launcher that replays it onto a live chip.
//
// If these ever diverge the aggregator reads its scratch from the wrong addresses and
// produces confident nonsense, so they do not get to diverge.
//
// Deliberately dependency-free: <cstdint> plus util_aggregator.h. No tt-metal types,
// because the launcher must not link tt-metal (a monitoring process cannot take
// CHIP_IN_USE from the workload it is monitoring).

#pragma once

#include <cstdint>
#include <vector>

#include "util_aggregator.h"

namespace ttnvtop {

// The ethernet firmware's liveness word. UMD's topology discovery polls EVERY eth core
// waiting for this to change (eth_heartbeat_running), because the idle-erisc firmware
// increments it. A persistent kernel occupies ERISC0 and the firmware never runs, so
// the heartbeat freezes -- and every later device open, ours or tt-metal's or tt-smi's,
// stalls on that core until UMD's timeouts expire.
//
// So the aggregator maintains it. Honouring the invariant the firmware guarantees is
// what makes a persistent eth kernel indistinguishable from firmware, rather than
// something that quietly degrades the platform for everyone else.
//
// Wormhole only: Blackhole's address differs and its discovery skips the check
// ("Temporary - heartbeat check disabled for Blackhole"). Pass 0 to disable.
constexpr uint32_t kWormholeEthHeartbeatAddr = 0x1C;       // umd wormhole_eth.hpp
constexpr uint32_t kEthBaseFwHeartbeatSignature = 0xABCD;  // umd erisc_firmware.hpp

// Everything the aggregator needs in its own eth-core L1. All 16 B aligned: every one
// of these is a NOC read destination or a published structure.
// THE JOURNAL COMES FIRST, at exactly `base`.
//
// A reader has to find the journal without already knowing how big it is, and its size
// depends on num_cores, which is a field INSIDE it. Putting it anywhere but the base
// makes discovery circular: you cannot compute the offset without first reading the
// thing at that offset. Scratch goes after, where nobody has to locate it.
//
// So `journal == base == the idle-eth UNRESERVED address` is a fixed, well-known
// address a UMD-only reader can probe on any ethernet core with a single 64 B read.
struct AggL1 {
    uint32_t journal = 0;       // 64 B header + num_cores * 32 B states — AT `base`
    uint32_t last_head = 0;     // num_cores * 4
    uint32_t head_scratch = 0;  // num_cores * 16  (16, not 4 — NOC L1 reads need 16 B alignment)
    uint32_t seq_scratch = 0;   // num_cores * 4
    uint32_t last_wall = 0;     // num_cores * 4
    uint32_t last_fpu = 0;      // num_cores * 4
    uint32_t sample_pad = 0;    // 16
    uint32_t dbg = 0;           // 16
    uint32_t end = 0;
};

inline AggL1 agg_layout(uint32_t base, uint32_t num_cores) {
    auto a16 = [](uint32_t v) { return (v + 15u) & ~15u; };
    AggL1 l;
    uint32_t p = a16(base);
    l.journal = p;
    p = a16(p + util_agg_bytes_for(num_cores));
    l.last_head = p;
    p = a16(p + num_cores * 4u);
    l.head_scratch = p;
    p = a16(p + num_cores * 16u);
    l.seq_scratch = p;
    p = a16(p + num_cores * 4u);
    l.last_wall = p;
    p = a16(p + num_cores * 4u);
    l.last_fpu = p;
    p = a16(p + num_cores * 4u);
    l.sample_pad = p;
    p = a16(p + 16u);
    l.dbg = p;
    p = a16(p + 16u);
    l.end = p;
    return l;
}

// The live Tensix cores as a cross product of translated x and y coordinates.
// Harvesting removes whole rows (WH) or columns (BH), so nx + ny numbers describe all
// nx * ny cores — and the set is NOT a contiguous rectangle on Blackhole.
struct AggGrid {
    std::vector<uint32_t> xs, ys;
    uint32_t num_cores() const { return static_cast<uint32_t>(xs.size() * ys.size()); }
};

// Runtime args, in the exact order eth_aggregator.cpp reads them.
inline std::vector<uint32_t> agg_rt_args(
    const AggGrid& g,
    const AggL1& l1,
    uint32_t src_chip,
    uint32_t sweep_interval_cycles,
    uint32_t publish_every,
    uint32_t heartbeat_addr) {
    std::vector<uint32_t> a = {g.num_cores(), static_cast<uint32_t>(g.xs.size()), static_cast<uint32_t>(g.ys.size())};
    a.insert(a.end(), g.xs.begin(), g.xs.end());
    a.insert(a.end(), g.ys.begin(), g.ys.end());
    a.push_back(l1.last_head);
    a.push_back(l1.head_scratch);
    a.push_back(l1.seq_scratch);
    a.push_back(l1.last_wall);
    a.push_back(l1.last_fpu);
    a.push_back(l1.journal);
    a.push_back(g.num_cores());  // capacity == num_cores in v2
    a.push_back(src_chip);
    a.push_back(sweep_interval_cycles);
    a.push_back(publish_every);
    a.push_back(l1.sample_pad);
    a.push_back(l1.dbg);
    a.push_back(heartbeat_addr);
    return a;
}

}  // namespace ttnvtop
