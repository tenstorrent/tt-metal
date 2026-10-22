// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host tests of the all_gather kernel library (walk.hpp, packetizer.hpp, slot_ring.hpp) against the real sharded
// TensorAccessor and the real fabric packet headers:
//   H1 the walk is a bijection, resumes at any position, its order does not depend on the stripe, and
//      gate targets grow monotonically;
//   H2 every dst run and every source piece is byte-linear;
//   H3 every packet is legal (<= P, <= 4 segments, cuts on A, reads congruent and inside their
//      pages) and delivers each unit to its own destination exactly once; headers round-trip;
//   H4 the slot ring never refills a slot the pump or a local copy may still read.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/operations/ccl/all_gather/device/all_gather_plan.hpp"

#include "host_kernel_prelude.hpp"

// NOLINTBEGIN(bugprone-macro-parentheses)

#include "ttnn/operations/ccl/all_gather/device/kernels/packetizer.hpp"

// The CB and NOC calls slot_ring.hpp makes, against a simulated CB and pump (H4).
#ifndef NOC_MAX_BURST_SIZE
#define NOC_MAX_BURST_SIZE 16384
#endif
struct RingSim {
    uint32_t received = 0, acked = 0, idle_polls = 0, last_barrier = 0;
    bool fenced = false;
    std::vector<bool> unflushed;  // per slot: a local-copy write from it not yet flushed
    std::function<void()> pump;   // pops a pushed slot, sometimes
    std::function<void(uint32_t)> on_write;
};
static RingSim ring_sim;
constexpr uint32_t read_cmd_buf = 1;
static uint32_t get_write_ptr(uint32_t) { return 0x20000; }
static volatile uint32_t* get_cb_tiles_received_ptr(uint32_t) { return &ring_sim.received; }
static volatile uint32_t* get_cb_tiles_acked_ptr(uint32_t) { return &ring_sim.acked; }
static void invalidate_l1_cache() { ring_sim.pump(); }
static bool noc_cmd_buf_ready(uint32_t, uint32_t cmd_buf) { return ring_sim.fenced = cmd_buf == read_cmd_buf; }
static void noc_async_writes_flushed() { ring_sim.unflushed.assign(ring_sim.unflushed.size(), false); }
static void noc_async_read_barrier_with_trid(uint32_t trid) { ring_sim.last_barrier = trid; }
template <uint32_t>
static void noc_async_write(uint32_t src, uint64_t, uint32_t) {
    ring_sim.on_write(src);
}
#include "ttnn/operations/ccl/all_gather/device/kernels/slot_ring.hpp"

#undef get_compile_time_arg_val
#undef noc_index
#undef ASSERT
#undef DPRINT
#undef DPRINT_DATA0
#undef DPRINT_DATA1
#undef DPRINT_MATH
#undef FORCE_INLINE
#undef DYNAMIC_NOC_X
#undef DYNAMIC_NOC_Y
#undef NOC_XY_ADDR

// NOLINTEND(bugprone-macro-parentheses)

namespace {

void check(bool ok, const std::string& what) {
    if (!ok) {
        throw std::runtime_error(what);
    }
}

// The device's interleaved TensorAccessor (KERNEL_BUILD-only in tensor_accessor.h), built with the
// bank pitch as its page size, as the kernels build it. Pages round-robin over banks.
struct InterleavedAcc {
    uint32_t banks, aligned, volume;
    uint32_t contiguous_page_stride() const { return banks; }
    uint32_t num_contiguous_pages(uint32_t page, uint32_t end) const {
        // The device does not check `end`; the sharded accessor does, so hold the walk to it here too.
        check(page < end && end <= volume, "interleaved num_contiguous_pages out of range");
        return (end - page - 1) / banks + 1;
    }
    uint32_t get_aligned_page_size() const { return aligned; }
    uint64_t get_noc_addr(uint32_t page, uint32_t off = 0) const {
        check(page < volume, "interleaved page out of range");
        return (uint64_t{page % banks} << 40) + uint64_t{page / banks} * aligned + off;
    }
};

// The real (fully dynamic) sharded TensorAccessor, with its shape storage.
using ShardedAcc = TensorAccessor<tensor_accessor::DistributionSpec<>>;
struct Sharded {
    std::vector<uint32_t> tensor, shard;
    std::vector<uint16_t> banks;
    std::unique_ptr<ShardedAcc> acc;
};

std::mt19937 rng(20261021);
uint32_t rnd(uint32_t lo, uint32_t hi) { return lo + rng() % (hi - lo + 1); }
uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

std::unique_ptr<Sharded> random_sharded(uint32_t volume, uint32_t aligned, std::string& desc) {
    auto s = std::make_unique<Sharded>();
    const uint32_t rank = rnd(1, 4);
    s->tensor.assign(rank, 1);
    for (uint32_t v = volume, p = 2; v > 1;) {
        if (v % p == 0) {
            s->tensor[rng() % rank] *= p;
            v /= p;
        } else {
            ++p;
        }
    }
    uint64_t shards = 1;
    s->shard.resize(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        const uint32_t t = s->tensor[i];
        const uint32_t m = rng() % 8;
        s->shard[i] = m == 0 ? 1 : m == 1 ? t : m == 2 ? t + 1 : rnd(1, t);
        if (rank > 1 && i == rank - 1 && rng() % 4 == 0) {
            s->shard[i] = 1;  // one page wide: contiguous_page_stride() > 1
        }
        shards *= (t + s->shard[i] - 1) / s->shard[i];
    }
    const uint32_t nb = rnd(1, 12);
    const bool contig = shards % nb == 0 && rng() % 2;
    s->banks.resize(nb);
    for (uint32_t i = 0; i < nb; ++i) {
        s->banks[i] = static_cast<uint16_t>((i << 8) | i);
    }
    tensor_accessor::DistributionSpec<> ds(rank, nb, s->tensor.data(), s->shard.data(), s->banks.data(), contig);
    s->acc = std::make_unique<ShardedAcc>(std::move(ds), 0, aligned);
    std::ostringstream o;
    o << "sharded[";
    for (uint32_t i = 0; i < rank; ++i) {
        o << s->tensor[i] << "/" << s->shard[i] << (i + 1 < rank ? "," : "");
    }
    o << "]b" << nb << (contig ? "c" : "");
    desc = o.str();
    return s;
}

struct Case {
    uint32_t A, P, ri, ro;  // cut quantum, fabric payload, source read alignments (input, output)
    uint32_t N, S, R, Ui, Uo, E, Pps;
    uint32_t ci, co, ai, ao;  // page content and bank pitch, bytes
    bool dst_cross, src_cross;
    uint32_t links;
    std::string desc;
};

// ---- recording Ring and Io ----

struct Read {
    uint64_t src;
    uint32_t l1, bytes, trid;
};
struct Packet {
    uint32_t slot, bytes;
    std::vector<Read> reads;
    std::vector<Seg> segs;
};

constexpr uint32_t kSlots = 6, kSlotBytes = 16384, kSlotBase = 0x10000, kPayloadOff = 64;

struct RecRing {
    uint32_t count = 0;
    bool open = false;
    uint32_t acquire() {
        check(!open, "acquire with a slot open");
        open = true;
        return kSlotBase + (count % kSlots) * kSlotBytes;
    }
    uint8_t trid() const { return 1 + count % kSlots; }
    void commit(const Seg*, uint32_t) {
        check(open, "commit without a slot");
        open = false;
        ++count;
    }
};

template <class H>
void round_trip(const Seg* segs, uint32_t nseg, uint32_t bytes) {
    alignas(16) uint8_t buf[sizeof(H)];
    std::memset(buf, 0xA5, sizeof(buf));  // stale fields must not matter
    auto* hdr = reinterpret_cast<volatile H*>(buf);
    encode_segments(hdr, segs, nseg, bytes);
    uint32_t i = 0, next = 0;
    for_each_segment(hdr, [&](uint32_t off, uint64_t addr, uint32_t len) {
        check(i < nseg && off == next && addr == segs[i].dst && len == segs[i].bytes, "header round trip");
        next += len;
        ++i;
    });
    check(i == nseg && next == bytes, "header round trip: segment count");
}

struct RecIo {
    static constexpr uint32_t kReadsPerPoll = 128;
    std::vector<Packet> packets;
    Packet cur;
    uint32_t trid = 0, since_poll = 0;
    void start(uint8_t t) {
        check(cur.reads.empty(), "start with reads pending");
        trid = t;
        since_poll = 0;
    }
    void throttle(uint8_t t) {
        check(t == trid && since_poll == kReadsPerPoll - 1, "throttle cadence");
        since_poll = 0;
    }
    void read(uint64_t src, uint32_t l1, uint32_t bytes, uint8_t t) {
        check(t == trid, "read trid differs from the open packet's");
        check(++since_poll < kReadsPerPoll, "reads past the trid limit without a throttle");
        cur.reads.push_back({src, l1, bytes, t});
    }
    void write_headers(uint32_t slot, const Seg* segs, uint32_t nseg, uint32_t bytes) {
        round_trip<tt::tt_fabric::LowLatencyPacketHeader>(segs, nseg, bytes);
        round_trip<tt::tt_fabric::HybridMeshPacketHeader>(segs, nseg, bytes);
        cur.slot = slot;
        cur.bytes = bytes;
        cur.segs.assign(segs, segs + nseg);
        packets.push_back(std::move(cur));
        cur = {};
    }
};

// ---- one (geometry, stripe, slice, positions, source) pass ----

struct Stats {
    uint64_t runs = 0, units = 0, packets = 0, packet_bound = 0, passes = 0;
};

template <class Out, class In>
std::vector<uint32_t> run_pass(
    const Case& c,
    const Out& out,
    const In& in,
    uint32_t stripe,
    uint32_t e0,
    uint32_t ne,
    uint32_t skip,
    uint32_t take,
    bool relay,
    Stats& st) {
    const uint32_t A = c.A;
    auto global = [&](uint32_t u) { return (u / c.S) * c.N * c.S + stripe * c.S + u % c.S; };
    auto out_addr = [&](uint32_t g) { return out.get_noc_addr(g / c.Uo, (g % c.Uo) * A); };
    auto in_addr = [&](uint32_t u) { return in.get_noc_addr(u / c.Ui, (u % c.Ui) * A); };

    Walk<Out> walk(out, WalkGeo{A, c.Uo, c.S, c.N, c.E, c.P, c.dst_cross});
    walk.init(stripe, e0, ne, skip, take);
    InputPieces<In> input(in, A, c.Ui, c.E, c.src_cross);
    RelayPieces relay_src{A, c.E};
    RecRing ring;
    RecIo io;
    Packetizer<RecRing, RecIo> pk(ring, io, c.P, kPayloadOff);

    std::vector<uint32_t> seq;  // elements in walk order
    uint32_t pos = 0, last_target = 0;
    // Checks each run as the packetizer takes it.
    auto before = [&](const DstRun& r) {
        // H1: positions advance, so gate targets (base + pos + units) grow.
        check(r.n >= 1 && pos + r.n <= take, "run length");
        const uint32_t target = (pos + r.n) * c.E;
        check(target > last_target, "gate target not monotone");
        last_target = target;
        pos += r.n;
        // H2: the run is linear at the destination.
        std::vector<uint32_t> units;
        for (uint32_t k = 0; k < r.n; ++k) {
            const uint32_t l = r.l0 + k * r.lambda;
            check(l >= e0 && l < e0 + ne, "element outside the slice");
            seq.push_back(l);
            for (uint32_t j = 0; j < c.E; ++j) {
                units.push_back(l * c.E + j);
            }
        }
        for (uint32_t i = 0; i < units.size(); ++i) {
            check(out_addr(global(units[i])) == r.addr + uint64_t{i} * A, "dst run not linear");
        }
        // H2: the pieces cover the run in order, each linear in its source.
        uint32_t i = 0;
        auto piece_check = [&](const Piece& p) {
            check(p.bytes >= A && p.bytes % A == 0 && p.readable_bytes <= p.bytes, "piece size");
            for (uint32_t k = 0; k < p.bytes / A; ++k, ++i) {
                check(i < units.size(), "pieces overrun the run");
                const uint64_t want = relay ? out_addr(global(units[i])) : in_addr(units[i]);
                check(p.src + uint64_t{k} * A == want, "piece not linear in its source");
            }
        };
        if (relay) {
            relay_src.for_each(r, piece_check);
        } else {
            input.for_each(r, piece_check);
        }
        check(i == units.size(), "pieces do not cover the run");
        ++st.runs;
    };
    if (relay) {
        pk.add_all(walk, relay_src, before);
    } else {
        pk.add_all(walk, input, before);
    }
    pk.close();
    check(pos == take, "walk did not take `take` elements");
    check(!ring.open && ring.count == io.packets.size(), "ring/packet count");

    // H3: packets.
    std::unordered_map<uint64_t, uint32_t> dst_unit;  // output address -> global unit, this stripe
    for (uint32_t u = 0; u < c.R * c.S; ++u) {
        dst_unit[out_addr(global(u))] = u;
    }
    const uint32_t P = c.P;
    std::set<uint32_t> seen;
    uint64_t bytes_total = 0;
    for (uint32_t p = 0; p < io.packets.size(); ++p) {
        const Packet& pkt = io.packets[p];
        check(pkt.bytes > 0 && pkt.bytes <= P && pkt.bytes % A == 0, "packet size");
        check(!pkt.segs.empty() && pkt.segs.size() <= NOC_SCATTER_WRITE_MAX_CHUNKS, "segment count");
        check(pkt.slot == kSlotBase + (p % kSlots) * kSlotBytes, "slot order");
        const uint32_t payload = pkt.slot + kPayloadOff;
        const uint32_t ra = relay ? c.ro : c.ri;
        struct Got {
            uint64_t src;
            uint32_t bytes;
        };
        std::vector<Got> got(pkt.bytes / A, Got{0, 0});
        for (const Read& rd : pkt.reads) {
            check(rd.trid == 1 + p % kSlots, "read trid");
            check(rd.l1 >= payload && rd.l1 + rd.bytes <= payload + pkt.bytes, "read outside the payload");
            check((rd.l1 - payload) % A == 0, "read L1 offset not on A");
            check((rd.l1 - rd.src) % ra == 0, "read not congruent with its source");
            check(rd.bytes > 0 && rd.bytes <= P, "read size");
            for (uint32_t b = 0; b < rd.bytes; b += A) {
                Got& gu = got[(rd.l1 - payload + b) / A];
                check(gu.bytes == 0, "two reads into one unit");
                gu = {rd.src + b, std::min(A, rd.bytes - b)};
            }
        }
        uint32_t seg_off = 0;
        for (const Seg& s : pkt.segs) {
            check(s.bytes > 0 && s.bytes % A == 0, "segment size");
            for (uint32_t b = 0; b < s.bytes; b += A) {
                const auto it = dst_unit.find(s.dst + b);
                check(it != dst_unit.end(), "segment writes outside this stripe");
                const uint32_t u = it->second;
                check(seen.insert(u).second, "unit delivered twice");
                const uint32_t g = global(u);
                // The unit's own bytes in its source page: offset, content and allocation.
                const uint64_t want = relay ? out_addr(g) : in_addr(u);
                const uint32_t off = relay ? (g % c.Uo) * A : (u % c.Ui) * A;
                const uint32_t content = relay ? c.co : c.ci, alloc = relay ? c.ao : c.ai;
                const Got& gu = got[(seg_off + b) / A];
                if (gu.bytes == 0) {
                    check(off >= content, "unit with content never read");
                } else {
                    check(gu.src == want, "slot unit holds another unit's data");
                    check(off + gu.bytes <= alloc, "read past the source page");
                    check(off >= content || gu.bytes >= std::min(A, content - off), "read misses content");
                }
            }
            seg_off += s.bytes;
        }
        check(seg_off == pkt.bytes, "segments do not add up to the payload");
        bytes_total += pkt.bytes;
    }
    check(seen.size() == uint64_t{take} * c.E, "units not delivered exactly once");
    for (uint32_t l : seq) {
        for (uint32_t j = 0; j < c.E; ++j) {
            check(seen.count(l * c.E + j), "a walked unit was not delivered");
        }
    }
    st.units += uint64_t{take} * c.E;
    st.packets += io.packets.size();
    st.packet_bound += (bytes_total + P - 1) / P;
    ++st.passes;
    return seq;
}

template <class Out, class In>
void run_case(const Case& c, const Out& out, const In& in, Stats& st) {
    const uint32_t elements = c.R * c.S / c.E;
    std::vector<uint32_t> stripes = {0, c.N - 1, rnd(0, c.N - 1)};
    for (const auto [e0, ne] :
         ttnn::operations::ccl::all_gather_plan::make_slices(elements, std::min(c.links, elements))) {
        std::vector<uint32_t> full;
        for (uint32_t stripe : stripes) {
            for (bool relay : {false, true}) {
                auto seq = run_pass(c, out, in, stripe, e0, ne, 0, ne, relay, st);
                if (full.empty()) {
                    // H1: a bijection on the slice.
                    full = seq;
                    auto sorted = seq;
                    std::sort(sorted.begin(), sorted.end());
                    for (uint32_t i = 0; i < ne; ++i) {
                        check(sorted[i] == e0 + i, "walk is not a bijection on the slice");
                    }
                }
                check(seq == full, "walk order depends on the stripe");
            }
            // H1: the even-ring halves partition the walk, and any position resumes in order.
            const uint32_t half = ne / 2;
            const uint32_t at = rnd(0, ne - 1);
            for (auto [skip, take] : {std::pair{0u, half}, std::pair{half, ne - half}, std::pair{at, ne - at}}) {
                const bool relay = rng() % 2;
                auto seq = run_pass(c, out, in, stripe, e0, ne, skip, take, relay, st);
                check(std::equal(seq.begin(), seq.end(), full.begin() + skip), "resumed walk differs");
            }
        }
    }
}

// Fills S, E, the cross flags and desc from the rest.
void finish(Case& c, const char* kind) {
    check(c.P % c.A == 0, "P is a multiple of A (the plan rounds it)");
    c.S = c.Pps * c.Ui;
    c.E = c.S % c.Uo == 0 ? c.Uo : 1;
    c.dst_cross = c.Uo * c.A == c.ao;
    c.src_cross = c.Ui * c.A == c.ai;
    std::ostringstream o;
    o << kind << " A" << c.A << " P" << c.P << " ri" << c.ri << " ro" << c.ro << " N" << c.N << " R" << c.R << " S"
      << c.S << " Ui" << c.Ui << " Uo" << c.Uo << " E" << c.E << " ci" << c.ci << " ai" << c.ai << " co" << c.co
      << " ao" << c.ao << " links" << c.links;
    c.desc = o.str();
}

// Random geometry: sub-page (A divides both pages' content) or whole-page (pages match, cut inside a
// page at A), interleaved or sharded on each side.
Case random_case() {
    Case c{};
    c.A = std::vector<uint32_t>{64, 32, 16}[rng() % 3];
    c.P = rng() % 2 ? 15168 : 7616;
    auto pick_align = [&] {
        std::vector<uint32_t> d;
        for (uint32_t a : {16u, 32u, 64u}) {
            if (c.A % a == 0) {
                d.push_back(a);
            }
        }
        return d[rng() % d.size()];
    };
    c.ri = pick_align();
    c.ro = pick_align();
    c.N = std::vector<uint32_t>{2, 3, 4, 5, 8}[rng() % 5];
    c.R = rnd(1, 12);
    c.Pps = rnd(1, 6);
    const bool whole = rng() % 3 == 0;
    if (whole) {
        // Content not a multiple of A; sometimes larger than a packet, so cuts fall inside pages.
        c.ci = c.co = rng() % 4 == 0 ? rnd(c.P / 2, 2 * c.P) : rnd(1, 4 * c.A);
        const uint32_t s = align_up(c.ci, c.A);
        c.Ui = c.Uo = s / c.A;
        c.ai = align_up(c.ci, c.ri);
        c.ao = s + (rng() % 4 == 0 ? c.ro * rnd(1, 2) : 0);  // legal only while s <= ao
    } else {
        c.Ui = rnd(1, 6);
        c.S = c.Pps * c.Ui;
        std::vector<uint32_t> divs;  // output pages tile each output row
        for (uint32_t d = 1; d <= c.N * c.S; ++d) {
            if (c.N * c.S % d == 0) {
                divs.push_back(d);
            }
        }
        c.Uo = divs[rng() % divs.size()];
        c.ci = c.Ui * c.A;
        c.co = c.Uo * c.A;
        c.ai = c.ci + (rng() % 6 == 0 ? c.ri : 0);
        c.ao = c.co + (rng() % 6 == 0 ? c.ro : 0);
    }
    c.links = rnd(1, 3);
    finish(c, whole ? "whole" : "sub");
    return c;
}

// Calls f with an interleaved or a random sharded accessor of `volume` pages.
template <class F>
void with_random_acc(uint32_t volume, uint32_t aligned, std::string& desc, F&& f) {
    if (rng() % 2) {
        const InterleavedAcc acc{std::vector<uint32_t>{1, 7, 8, 12, 64}[rng() % 5], aligned, volume};
        desc = "interleaved" + std::to_string(acc.banks);
        f(acc);
    } else {
        auto acc = random_sharded(volume, aligned, desc);
        f(*acc->acc);
    }
}

void run_random(uint32_t iters, Stats& st) {
    for (uint32_t it = 0; it < iters; ++it) {
        const Case c = random_case();
        std::string out_desc, in_desc;
        try {
            with_random_acc(c.R * c.N * c.S / c.Uo, c.ao, out_desc, [&](const auto& out) {
                with_random_acc(c.R * c.Pps, c.ai, in_desc, [&](const auto& in) { run_case(c, out, in, st); });
            });
        } catch (const std::exception& e) {
            FAIL() << e.what() << "\n  " << c.desc << "\n  out " << out_desc << "  in " << in_desc;
        }
    }
}

}  // namespace

#ifndef AG_KERNEL_LIB_ITERS
#define AG_KERNEL_LIB_ITERS 200
#endif

TEST(AllGatherKernelLib, RandomGeometries) {
    Stats st;
    run_random(AG_KERNEL_LIB_ITERS, st);
    std::cout << "passes " << st.passes << " runs " << st.runs << " units " << st.units << " packets " << st.packets
              << " (lower bound " << st.packet_bound << ")\n";
}

// Shapes from the device matrix, interleaved DRAM (8 banks) on both sides.
TEST(AllGatherKernelLib, MatrixShapes) {
    struct Shape {
        uint32_t ci, Pps, R;
    };
    const std::vector<Shape> shapes = {
        {2048, 10, 32},  // D1 tile bf16 [1024, 2560] d3: stripe 10, no cross_row (8 % 80)
        {1088, 1, 8},    // D2 bfp8 [256, 256] d3: S = 1, cross_row
        {8192, 1, 32},   // D3 RM 8 KB rows d0
        {20000, 1, 64},  // D3 20000 B rows: whole-page, cuts inside pages
        {48, 1, 512},    // D4 48 B rows: whole-page, s == ao merges
        {16, 1, 64},     // D9 16 B rows
    };
    Stats st;
    for (const Shape& sh : shapes) {
        for (uint32_t P : {15168u, 7616u}) {
            Case c{};
            c.A = c.ri = c.ro = 64;
            c.P = P;
            c.N = 8;
            c.R = sh.R;
            c.Pps = sh.Pps;
            c.ci = c.co = sh.ci;
            c.ai = c.ao = align_up(sh.ci, c.A);
            c.Ui = c.Uo = c.ai / c.A;
            c.links = 2;
            finish(c, "matrix");
            const InterleavedAcc out{8, c.ao, c.R * c.N * c.Pps};
            const InterleavedAcc in{8, c.ai, c.R * c.Pps};
            try {
                run_case(c, out, in, st);
            } catch (const std::exception& e) {
                FAIL() << e.what() << "\n  " << c.desc;
            }
        }
    }
}

// H4: the slot ring against a simulated CB whose pump pops pushed slots at random times. A slot is refilled only
// once the pump has popped it and its local-copy writes are flushed; a held slot's writes follow its read barrier;
// at most `inflight` slots are held; without local copy every slot is pushed at once, behind the read fence.
struct RingCt {
    static constexpr uint32_t cb = 0, slots = 6, slot_bytes = 1024, payload_off = 64, inflight = 3;
};

TEST(AllGatherKernelLib, SlotRing) {
    using Ct = RingCt;
    for (const bool copy : {false, true}) {
        ring_sim = RingSim{};
        ring_sim.unflushed.assign(Ct::slots, false);
        ring_sim.pump = [] {
            if (ring_sim.acked != ring_sim.received && rng() % 2) {
                ++ring_sim.acked;
            } else if (++ring_sim.idle_polls > 1000) {
                throw std::runtime_error("the ring waits for a slot nobody will free");
            }
        };
        ring_sim.on_write = [](uint32_t src) {
            const uint32_t h = (src - get_write_ptr(0)) / Ct::slot_bytes;
            check(ring_sim.received % Ct::slots == h, "a local copy reads a slot other than the oldest held");
            check(ring_sim.last_barrier == 1 + h, "a local copy reads its slot before the slot's reads landed");
            ring_sim.unflushed[h] = true;
        };
        try {
            CbSlotRing<Ct> ring(copy);
            Seg segs[NOC_SCATTER_WRITE_MAX_CHUNKS];
            for (uint32_t committed = 0; committed < 500;) {
                ring_sim.idle_polls = 0;
                const uint32_t k = (ring.acquire() - get_write_ptr(0)) / Ct::slot_bytes;
                check(k == committed % Ct::slots && ring.trid() == 1 + k, "slots fill in ring order");
                check(committed - ring_sim.acked < Ct::slots, "refilled a slot the pump has not popped");
                check(!ring_sim.unflushed[k], "refilled a slot a local copy may still read");
                const uint32_t nseg = rnd(1, NOC_SCATTER_WRITE_MAX_CHUNKS);
                for (uint32_t i = 0; i < nseg; ++i) {
                    segs[i] = {uint64_t{i} * 64, 64};
                }
                ring_sim.fenced = false;
                ring.commit(segs, nseg);
                ++committed;
                check(copy || (ring_sim.received == committed && ring_sim.fenced), "push at once, behind the fence");
                check(committed - ring_sim.received <= (copy ? Ct::inflight : 0), "too many held slots");
            }
            ring.drain();
            check(ring_sim.received == 500, "drain pushes every held slot");
        } catch (const std::exception& e) {
            FAIL() << e.what() << (copy ? " (local copy)" : "");
        }
    }
}
