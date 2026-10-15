// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host proof of the all_gather packet packer and run loops (ttnn/.../all_gather/device/kernels/):
//   - PacketPacker (gather_packer.hpp) emits exactly the packet sequence of BOTH writers it replaced: the
//     unicast FabricWriter (runtime while-loop split of oversized segments, tail left pending) and the
//     multicast FabricWriter (compile-time `oversized` path), on random segment streams covering normal
//     packing, the 4-chunk scatter cap, chunk == packet, and oversized chunks.
//   - emit_runs / read_input_runs (gather_walk.hpp) visit the same runs as the inline loops they replaced.
//
// No device: the packer is driven through a recording Sink.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#define HOST_KERNEL_SHIM_KEEP_MACROS  // the kernel headers need FORCE_INLINE / ASSERT
#include "host_kernel_shim.hpp"

#include "ttnn/operations/ccl/all_gather/device/kernels/gather_walk.hpp"
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
    constexpr uint32_t xfer = chunks_per_transfer(packet_size, chunk_size, 0, 16384);
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
// emit_runs / read_input_runs vs the inline loops they replaced
// ---------------------------------------------------------------------------------------------------

// Contiguous runs of `run_pages` pages (models DRAM-bank / shard contiguity).
struct MockAccessor {
    uint32_t run_pages;
    uint32_t num_contiguous_pages(uint32_t page, uint32_t end_page) const {
        const uint32_t to_edge = run_pages - page % run_pages;
        const uint32_t room = end_page - page;
        return to_edge < room ? to_edge : room;
    }
};

struct Run {
    uint32_t l1, page, off, run;
    bool operator==(const Run& o) const { return l1 == o.l1 && page == o.page && off == o.off && run == o.run; }
};

template <uint32_t cpp, uint32_t chunk_size, uint32_t cps, uint32_t ndev>
void emit_equivalence(uint32_t stride, uint32_t xfer, uint32_t run_pages, bool whole_page, bool cross_page) {
    MockAccessor acc{run_pages};
    const RunSource src{whole_page, cross_page};
    const uint32_t first = 3, count = 4 * cps + 5, batch = 7;
    const uint32_t end_chunk = first + count;
    StripeMap<cps, ndev> map;
    map.init(1, end_chunk);
    auto addr = [](uint32_t page, uint32_t off) { return uint64_t(page) * cpp * chunk_size + off; };

    // k = 1, rot = 0, runtime stride/xfer: the v1 emission order (test_all_gather_walk.cpp MatchesLegacy).
    TiledWalk<1> w_old, w_new;
    w_old.init(end_chunk, stride, xfer, 0, walk_start(first));
    w_new = w_old;
    std::vector<Run> got_old, got_new;
    for (uint32_t done = 0; done < count; done += batch) {
        const uint32_t n = std::min(batch, count - done);
        // old inline loop (unicast_writer / multicast_writer / multicast_reader send_entry)
        uint32_t l1 = 0x1000;
        for (uint32_t left = n; left > 0;) {
            const auto pos = map.at(w_old.chunk());
            const uint32_t page = page_of<cpp>(pos.global);
            const uint32_t off = byte_off_of<cpp, chunk_size>(pos.global);
            const uint32_t run = next_run<1, cpp>(w_old, acc, src, pos.global, pos.bound, left);
            got_old.push_back({l1, page, off, run});
            l1 += run * chunk_size;
            left -= run;
            w_old.advance(run);
        }
        const uint32_t end = emit_runs<cpp, chunk_size, chunk_size>(
            w_new, map, acc, src, 0x1000u, n, addr, [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                got_new.push_back({l1, page, off, run});
            });
        EXPECT_EQ(end, l1);
    }
    EXPECT_TRUE(got_old == got_new);

    // read_input_runs vs the old input loops (unicast_reader from_input / multicast_reader fill_entry)
    constexpr uint32_t split = cpp;
    auto in_addr = [](uint32_t chunk) { return uint64_t(chunk) * chunk_size; };
    w_old.init(end_chunk, stride, xfer, 0, walk_start(first));
    w_new = w_old;
    std::vector<Run> in_old, in_new;
    for (uint32_t done = 0; done < count; done += batch) {
        const uint32_t n = std::min(batch, count - done);
        uint32_t l1 = 0x2000;
        for (uint32_t left = n; left > 0;) {
            const uint32_t chunk = w_old.chunk();
            const uint32_t run = next_run<1, split>(w_old, acc, src, chunk, end_chunk, left);
            in_old.push_back({l1, uint32_t(in_addr(chunk)), 0, run});
            l1 += run * chunk_size;
            left -= run;
            w_old.advance(run);
        }
        const uint32_t end = read_input_runs<split, chunk_size, chunk_size>(
            w_new, acc, src, end_chunk, 0x2000u, n, in_addr, [&](uint64_t s, uint32_t l1, uint32_t run) {
                in_new.push_back({l1, uint32_t(s), 0, run});
            });
        EXPECT_EQ(end, l1);
    }
    EXPECT_TRUE(in_old == in_new);
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

TEST(AllGatherPacker, EmitRunsMatchInlineLoops) {
    emit_equivalence<1, 2048, 6, 4>(1, 2, 3, false, true);
    emit_equivalence<1, 2048, 6, 4>(3, 2, 1, false, false);
    emit_equivalence<4, 512, 8, 2>(1, 4, 2, true, true);
    emit_equivalence<4, 512, 8, 2>(1, 3, 1, true, false);
}
