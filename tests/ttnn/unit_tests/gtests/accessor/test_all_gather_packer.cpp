// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host proof of the all_gather packet packer (ttnn/.../all_gather/device/kernels/gather_packer.hpp):
//   - PacketPacker emits exactly the packet sequence of BOTH writers it replaced: the unicast FabricWriter
//     (runtime while-loop split of oversized segments, tail left pending) and the multicast FabricWriter
//     (compile-time `oversized` path), on random segment streams covering normal packing, the 4-chunk scatter
//     cap, chunk == packet, and oversized chunks.
//   - Properties, independent of any oracle: every packet's payload fits the packet; a scatter write carries
//     2..4 segments and its sizes add up to the payload; a packet's bytes are one contiguous stretch of L1;
//     every queued byte is sent exactly once, to the destination it was queued for; a flush leaves nothing
//     pending.
//
// No device: the packer is driven through a recording Sink.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#define HOST_KERNEL_SHIM_KEEP_MACROS  // the kernel headers need FORCE_INLINE / ASSERT
#include "host_kernel_shim.hpp"

#include "ttnn/operations/ccl/all_gather/device/kernels/gather_rules.hpp"
#include "ttnn/operations/ccl/all_gather/device/kernels/gather_packer.hpp"

namespace {

// ---------------------------------------------------------------------------------------------------
// Packet log
// ---------------------------------------------------------------------------------------------------

struct Ev {
    char kind;  // 'U' unicast write, 'S' scatter write, 'F' flush
    uint32_t l1;
    std::vector<uint64_t> dst;
    std::vector<uint16_t> sizes;  // the chunk_count-1 sizes that travel in the header
    uint32_t bytes;
    bool operator==(const Ev& o) const {
        return kind == o.kind && l1 == o.l1 && dst == o.dst && sizes == o.sizes && bytes == o.bytes;
    }
};
using Log = std::vector<Ev>;

struct MockSink {
    using ScatterSlots = GatherScatterSlots;
    Log log;
    void unicast_write(uint32_t l1, uint64_t dst, uint32_t bytes) { log.push_back({'U', l1, {dst}, {}, bytes}); }
    void scatter_write(uint32_t l1, const ScatterSlots& s, uint32_t payload) {
        Ev e{'S', l1, {}, {}, payload};
        for (uint32_t i = 0; i < s.chunk_count; ++i) {
            e.dst.push_back(s.noc_address[i]);
        }
        for (uint32_t i = 0; i + 1 < s.chunk_count; ++i) {
            e.sizes.push_back(s.chunk_size[i]);
        }
        log.push_back(e);
    }
    void flush() { log.push_back({'F', 0, {}, {}, 0}); }
};

// ---------------------------------------------------------------------------------------------------
// Oracles: the packing logic of the two FabricWriters PacketPacker replaced (unicast_common.hpp and
// multicast_common.hpp before the shared-layer refactor). Verbatim except that fabric calls become log
// entries. The per-send noc.async_writes_flushed() is the Sink's job in the new code, so neither side
// logs it; only flush_packet_and_wait()'s flush is logged.
// ---------------------------------------------------------------------------------------------------

template <uint32_t packet_size>
struct OldUnicastWriter {
    static constexpr uint32_t max_chunks = 4;  // NOC_SCATTER_WRITE_MAX_CHUNKS
    Log log;
    uint64_t noc_address[max_chunks]{};
    uint16_t chunk_size[max_chunks - 1]{};
    uint8_t chunk_count = 0;
    uint32_t payload = 0;
    uint32_t start_l1_addr = 0;

    void queue_segment(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        while (bytes > packet_size) {
            send();
            push(l1_addr, remote_noc_addr, packet_size);
            send();
            l1_addr += packet_size;
            remote_noc_addr += packet_size;
            bytes -= packet_size;
        }
        if (chunk_count == max_chunks || payload + bytes > packet_size) {
            send();
        }
        push(l1_addr, remote_noc_addr, bytes);
    }
    void flush_packet_and_wait() {
        send();
        log.push_back({'F', 0, {}, {}, 0});
    }
    void push(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if (chunk_count == 0) {
            start_l1_addr = l1_addr;
        }
        if (chunk_count < max_chunks - 1) {
            chunk_size[chunk_count] = static_cast<uint16_t>(bytes);
        }
        noc_address[chunk_count++] = remote_noc_addr;
        payload += bytes;
    }
    void send() {
        if (chunk_count == 0) {
            return;
        }
        if (chunk_count == 1) {
            log.push_back({'U', start_l1_addr, {noc_address[0]}, {}, payload});
        } else {
            Ev e{'S', start_l1_addr, {}, {}, payload};
            for (uint32_t i = 0; i < chunk_count; ++i) {
                e.dst.push_back(noc_address[i]);
            }
            for (uint32_t i = 0; i + 1 < chunk_count; ++i) {
                e.sizes.push_back(chunk_size[i]);
            }
            log.push_back(e);
        }
        chunk_count = 0;
        payload = 0;
    }
};

// Same push/send; differs in the oversized path (compile-time, one unicast write per packet).
template <uint32_t chunk_size, uint32_t packet_size>
struct OldMulticastWriter : OldUnicastWriter<packet_size> {
    static constexpr bool oversized = chunk_size > packet_size;
    static constexpr uint32_t packets_per_chunk = (chunk_size + packet_size - 1) / packet_size;
    static constexpr uint32_t last_payload_size = chunk_size - ((packets_per_chunk - 1) * packet_size);

    void queue_segment(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if constexpr (oversized) {
            for (uint32_t packet = 0; packet < packets_per_chunk; ++packet) {
                this->log.push_back(
                    {'U',
                     l1_addr,
                     {remote_noc_addr},
                     {},
                     (packet < packets_per_chunk - 1) ? packet_size : last_payload_size});
                l1_addr += packet_size;
                remote_noc_addr += packet_size;
            }
        } else {
            if (this->chunk_count == this->max_chunks || this->payload + bytes > packet_size) {
                this->send();
            }
            this->push(l1_addr, remote_noc_addr, bytes);
        }
    }
    void flush_packet_and_wait() {
        if constexpr (!oversized) {
            this->send();
        }
        this->log.push_back({'F', 0, {}, {}, 0});
    }
};

struct StreamStats {
    uint32_t unicast = 0, scatter = 0, scatter_full = 0;
};

StreamStats stats(const Log& log) {
    StreamStats s;
    for (const auto& e : log) {
        s.unicast += e.kind == 'U';
        s.scatter += e.kind == 'S';
        s.scatter_full += e.kind == 'S' && e.dst.size() == gather_scatter_max_chunks;
    }
    return s;
}

// Random segment stream: CB entries of up to 24 chunks, runs of 1..xfer chunks (1..3 with `short_runs`, so
// that small chunks run into the scatter-chunk cap before the payload limit) contiguous in L1, arbitrary
// destinations; a flush closes each entry, as the kernels do before pop_front.
template <uint32_t chunk_size, uint32_t packet_size, bool short_runs = false>
StreamStats packer_equivalence(uint32_t seed) {
    constexpr uint32_t xfer = max_xfer(packet_size, chunk_size, 0, 16384);
    std::mt19937 rng(seed);
    MockSink sink;
    OldUnicastWriter<packet_size> old_u;
    OldMulticastWriter<chunk_size, packet_size> old_m;
    {
        PacketPacker<packet_size, chunk_size, MockSink> packer(sink);
        for (int entry = 0; entry < 50; ++entry) {
            uint32_t l1 = 0x10000 + (entry % 3) * 0x40000;
            uint32_t left = 1 + rng() % 24;
            while (left > 0) {
                const uint32_t max_run = short_runs ? std::min<uint32_t>(xfer, 3) : xfer;
                const uint32_t run = std::min<uint32_t>(1 + rng() % max_run, left);
                const uint64_t dst = (uint64_t(rng() % 64) << 36) | (uint64_t(rng() % 4096) * 64);
                packer.queue_segment(l1, dst, run * chunk_size);
                old_u.queue_segment(l1, dst, run * chunk_size);
                old_m.queue_segment(l1, dst, run * chunk_size);
                l1 += run * chunk_size;
                left -= run;
            }
            packer.flush_packet_and_wait();
            old_u.flush_packet_and_wait();
            old_m.flush_packet_and_wait();
        }
    }
    EXPECT_FALSE(sink.log.empty());
    EXPECT_TRUE(sink.log == old_u.log) << "vs old unicast writer, chunk " << chunk_size << " packet " << packet_size
                                       << " seed " << seed;
    EXPECT_TRUE(sink.log == old_m.log) << "vs old multicast writer, chunk " << chunk_size << " packet " << packet_size
                                       << " seed " << seed;
    return stats(sink.log);
}

template <uint32_t chunk_size, uint32_t packet_size, bool short_runs = false>
StreamStats packer_equivalence_seeds() {
    StreamStats total;
    for (uint32_t seed = 1; seed <= 20; ++seed) {
        const auto s = packer_equivalence<chunk_size, packet_size, short_runs>(seed);
        total.unicast += s.unicast;
        total.scatter += s.scatter;
        total.scatter_full += s.scatter_full;
    }
    return total;
}

// ---------------------------------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------------------------------

// Replays a sink log byte by byte: every packet's (l1 range, destination) pairs, in order.
struct Byte {
    uint32_t l1;
    uint64_t dst;
    bool operator<(const Byte& o) const { return l1 != o.l1 ? l1 < o.l1 : dst < o.dst; }
    bool operator==(const Byte& o) const { return l1 == o.l1 && dst == o.dst; }
};

template <uint32_t chunk_size, uint32_t packet_size, bool short_runs = false>
void packer_properties(uint32_t seed) {
    SCOPED_TRACE(
        "chunk " + std::to_string(chunk_size) + " packet " + std::to_string(packet_size) + " seed " +
        std::to_string(seed));
    constexpr uint32_t xfer = max_xfer(packet_size, chunk_size, 0, 16384);
    std::mt19937 rng(seed);
    MockSink sink;
    std::vector<Byte> queued;
    std::vector<size_t> flush_at;  // log size after each flush_packet_and_wait()
    {
        PacketPacker<packet_size, chunk_size, MockSink> packer(sink);
        for (int entry = 0; entry < 40; ++entry) {
            uint32_t l1 = 0x10000 + (entry % 3) * 0x40000;
            uint32_t left = 1 + rng() % 24;
            while (left > 0) {
                const uint32_t max_run = short_runs ? std::min<uint32_t>(xfer, 3) : xfer;
                const uint32_t run = std::min<uint32_t>(1 + rng() % max_run, left);
                // Distinct, 64 B-aligned destinations so a byte's destination identifies it.
                const uint64_t dst = (uint64_t(entry) << 40) | (uint64_t(rng() % 4096) << 24) | (uint64_t(l1) << 0);
                packer.queue_segment(l1, dst, run * chunk_size);
                for (uint32_t b = 0; b < run * chunk_size; b += 16) {
                    queued.push_back({l1 + b, dst + b});
                }
                l1 += run * chunk_size;
                left -= run;
            }
            packer.flush_packet_and_wait();
            flush_at.push_back(sink.log.size());
        }
    }
    std::vector<Byte> sent;
    for (const auto& e : sink.log) {
        if (e.kind == 'F') {
            continue;
        }
        ASSERT_GT(e.bytes, 0u);
        ASSERT_LE(e.bytes, packet_size) << "payload larger than a packet";
        if (e.kind == 'U') {
            ASSERT_EQ(e.dst.size(), 1u);
            for (uint32_t b = 0; b < e.bytes; b += 16) {
                sent.push_back({e.l1 + b, e.dst[0] + b});
            }
            continue;
        }
        // Scatter: 2..4 segments; the header's sizes cover all but the last, implied by the payload.
        ASSERT_GE(e.dst.size(), gather_scatter_min_chunks);
        ASSERT_LE(e.dst.size(), gather_scatter_max_chunks);
        ASSERT_EQ(e.sizes.size(), e.dst.size() - 1);
        uint32_t off = 0;
        for (size_t i = 0; i < e.dst.size(); ++i) {
            const uint32_t sz = i + 1 < e.dst.size() ? e.sizes[i] : e.bytes - off;
            ASSERT_GT(sz, 0u);
            ASSERT_LE(off + sz, e.bytes);
            // One payload from e.l1: segment i's bytes follow segment i - 1's in L1.
            for (uint32_t b = 0; b < sz; b += 16) {
                sent.push_back({e.l1 + off + b, e.dst[i] + b});
            }
            off += sz;
        }
        ASSERT_EQ(off, e.bytes);
    }
    // A flush leaves nothing behind: everything queued before it was sent before it.
    for (size_t f = 0; f < flush_at.size(); ++f) {
        ASSERT_EQ(sink.log[flush_at[f] - 1].kind, 'F');
    }
    std::sort(queued.begin(), queued.end());
    std::sort(sent.begin(), sent.end());
    ASSERT_EQ(sent, queued) << "a byte was dropped, duplicated or sent to the wrong address";
}

template <uint32_t chunk_size, uint32_t packet_size, bool short_runs = false>
void packer_properties_seeds() {
    for (uint32_t seed = 1; seed <= 20; ++seed) {
        packer_properties<chunk_size, packet_size, short_runs>(seed);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

}  // namespace

TEST(AllGatherPacker, MatchesOldWritersNormal) {
    const auto s = packer_equivalence_seeds<2048, 4096>();  // 2 chunks per packet
    EXPECT_GT(s.scatter, 0u);
    EXPECT_GT(s.unicast, 0u);
}

TEST(AllGatherPacker, MatchesOldWritersScatterCap) {
    packer_equivalence_seeds<64, 4096>();
    // Many short segments: the 4-chunk scatter cap closes packets long before the payload fills.
    const auto s = packer_equivalence_seeds<64, 4096, /*short_runs=*/true>();
    EXPECT_GT(s.scatter_full, 0u);
}

TEST(AllGatherPacker, MatchesOldWritersChunkEqualsPacket) {
    const auto s = packer_equivalence_seeds<4096, 4096>();  // never scatters
    EXPECT_EQ(s.scatter, 0u);
}

TEST(AllGatherPacker, MatchesOldWritersOversized) {
    packer_equivalence_seeds<8192, 4096>();    // exact multiple
    packer_equivalence_seeds<10240, 4096>();   // ragged last packet
    packer_equivalence_seeds<32768, 15232>();  // BH packet size
}

TEST(AllGatherPacker, Properties) {
    packer_properties_seeds<2048, 4096>();
    packer_properties_seeds<2048, 15232>();
    packer_properties_seeds<64, 4096, /*short_runs=*/true>();
    packer_properties_seeds<64, 15232>();
    packer_properties_seeds<4096, 4096>();
    packer_properties_seeds<8192, 4096>();
    packer_properties_seeds<10240, 4096>();
    packer_properties_seeds<32768, 15232>();
}
