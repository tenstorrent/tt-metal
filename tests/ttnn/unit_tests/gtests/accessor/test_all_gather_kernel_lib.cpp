// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host byte-level sim of the all_gather kernel library (ag_walk.hpp, ag_packet.hpp, ag_sync.hpp) against the real
// sharded TensorAccessor and the real fabric scatter command header. A reader model fills CB slots exactly as
// ag_reader.cpp does (k packets per slot, records and payload in the slot), and a writer model decodes each record
// and checks every delivered byte:
//   W  the walk is a bijection on the slice, resumes at any position, its order does not depend on the stripe,
//      every run is linear at the destination and within the column cap, and mid-row slices walk whole rows;
//   P  every packet is <= P_A bytes and <= 4 segments, cuts on A, reads congruent with their sources and inside
//      their pages, and delivers each unit once, with its own bytes, in walk order;
//   S  a slot carries at most max_slot_units() units of one iteration (so a relay's wait target covers every read),
//      the reader never writes the writer's header region, and SignalCadence signals every unit by iteration end.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Pure headers first, before the macros the device accessor needs on host.
#include "ttnn/operations/ccl/all_gather/device/kernels/ag_packet.hpp"
#include "ttnn/operations/ccl/all_gather/device/kernels/ag_sync.hpp"

// NOLINTBEGIN(bugprone-macro-parentheses)

// Defines to include api/tensor/tensor_accessor.h on host (as test_tensor_accessor.cpp does).
template <int N>
constexpr auto get_ct_arg();
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

namespace tensor_accessor {
uint64_t get_dram_bank_base_offset(uint32_t bank_id, uint8_t noc);
}

[[maybe_unused]] static uint32_t my_x[1] = {0};
[[maybe_unused]] static uint32_t my_y[1] = {0};

static void host_assert(bool condition, const char* expression) {
    if (!condition) {
        throw std::out_of_range(expression);
    }
}

#define noc_index 0
#define ASSERT(condition, ...) host_assert(condition, #condition)
#define FORCE_INLINE inline __attribute__((always_inline))
#define DPRINT std::cout
#define ENDL() std::endl
#define DPRINT_DATA0(x) x
#define DPRINT_DATA1(x) x
#define DPRINT_MATH(x) x
#define NOC_UNICAST_ADDR_X(addr) addr
#define NOC_UNICAST_ADDR_Y(addr) addr
#define DYNAMIC_NOC_X(noc, x) (x)
#define DYNAMIC_NOC_Y(noc, y) (y)
#define NOC_XY_ADDR(x, y, addr) \
    (static_cast<uint64_t>(x) << 32 | static_cast<uint64_t>(y) << 16 | static_cast<uint64_t>(addr))
namespace noc_address_backend {
[[maybe_unused]] static bool is_local(uint64_t noc_addr, uint8_t noc) {
    return NOC_UNICAST_ADDR_X(noc_addr) == my_x[noc] && NOC_UNICAST_ADDR_Y(noc_addr) == my_y[noc];
}
[[maybe_unused]] static bool is_local_coordinate(uint32_t x, uint32_t y, uint8_t noc) {
    return x == my_x[noc] && y == my_y[noc];
}
[[maybe_unused]] static uint64_t worker_address(uint32_t x, uint32_t y, uint32_t addr, [[maybe_unused]] uint8_t noc) {
    return NOC_XY_ADDR(DYNAMIC_NOC_X(noc, x), DYNAMIC_NOC_Y(noc, y), addr);
}
}  // namespace noc_address_backend

#include "api/tensor/tensor_accessor.h"

#undef get_compile_time_arg_val
#undef noc_index
#undef ASSERT
#undef DPRINT
#undef ENDL
#undef DPRINT_DATA0
#undef DPRINT_DATA1
#undef DPRINT_MATH
#undef FORCE_INLINE
#undef NOC_UNICAST_ADDR_X
#undef NOC_UNICAST_ADDR_Y
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

std::mt19937 rng(20261023);
uint32_t rnd(uint32_t lo, uint32_t hi) { return lo + rng() % (hi - lo + 1); }

// What memory holds at a NOC address: a fixed hash, so a byte names the address it came from.
uint8_t byte_at(uint64_t a) {
    a ^= a >> 33;
    a *= 0xff51afd7ed558ccdULL;
    a ^= a >> 33;
    return static_cast<uint8_t>(a);
}

// The device's interleaved TensorAccessor (KERNEL_BUILD-only in tensor_accessor.h). Pages round-robin over banks
// at the bank pitch. pitch_runs = false models a ctor page size that is not the pitch: the device then reports
// every run as one page.
struct InterleavedAcc {
    uint32_t banks, pitch, volume;
    bool pitch_runs = true;
    uint32_t contiguous_page_stride() const { return banks; }
    uint32_t num_contiguous_pages(uint32_t page, uint32_t end) const {
        // The device does not check `end`; the sharded accessor does, so hold the walk to it here too.
        check(page < end && end <= volume, "interleaved num_contiguous_pages out of range");
        return pitch_runs ? (end - page - 1) / banks + 1 : 1;
    }
    uint32_t get_aligned_page_size() const { return pitch; }
    uint64_t get_noc_addr(uint32_t page, uint32_t off = 0) const {
        check(page < volume, "interleaved page out of range");
        return (uint64_t{page % banks + 1} << 40) + uint64_t{page / banks} * pitch + off;
    }
};

// The real (fully dynamic) sharded TensorAccessor, with its shape storage.
using ShardedAcc = TensorAccessor<tensor_accessor::DistributionSpec<>>;
struct Sharded {
    std::vector<uint32_t> tensor, shard;
    std::vector<uint16_t> banks;
    std::unique_ptr<ShardedAcc> acc;
};

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
    o << "]b" << nb << (contig ? "c" : "") << " L" << s->acc->contiguous_page_stride();
    desc = o.str();
    return s;
}

// Read alignments and NOC burst per architecture. "X" stands in for an uncalibrated arch with larger alignments.
struct Arch {
    const char* name;
    uint32_t dram, l1, burst;
};
constexpr Arch kWH{"WH", 32, 16, 8192}, kBH{"BH", 64, 16, 16384}, kX{"X", 128, 32, 16384};
constexpr uint32_t kPayloads[] = {4352, 7616, 15168};
constexpr uint32_t kHeaderBytes[] = {
    sizeof(tt::tt_fabric::LowLatencyPacketHeader), sizeof(tt::tt_fabric::HybridMeshPacketHeader)};

struct Case {
    uint32_t A, P_A, ri, ro;  // cut quantum, payload cap, source read alignments (input, output)
    uint32_t N, S, R, Ui, Uo, E, Pps;
    uint32_t ci, co, ai, ao;           // page content and bank pitch, bytes
    bool dst_cross, src_cross, relay;  // relay: unicast (re-reads its output); else multicast (input only)
    uint32_t workers, k, hdr_bytes, cadence;
    std::string desc;
};

// Fills S, E, the cross flags and desc from the rest.
void finish(Case& c, const char* kind) {
    check(c.P_A % c.A == 0 && c.P_A >= c.A, "P_A is a positive multiple of A");
    check(c.Uo * c.A <= c.ao, "an output page's footprint fits its pitch (host legality)");
    c.S = c.Pps * c.Ui;
    c.E = c.S % c.Uo == 0 ? c.Uo : 1;
    c.dst_cross = c.Uo * c.A == c.ao;
    c.src_cross = c.Ui * c.A == c.ai;
    std::ostringstream o;
    o << kind << " A" << c.A << " P_A" << c.P_A << " ri" << c.ri << " ro" << c.ro << " N" << c.N << " R" << c.R << " S"
      << c.S << " Ui" << c.Ui << " Uo" << c.Uo << " E" << c.E << " ci" << c.ci << " ai" << c.ai << " co" << c.co
      << " ao" << c.ao << " w" << c.workers << " k" << c.k << " hdr" << c.hdr_bytes << " cad" << c.cadence
      << (c.relay ? " uc" : " mc");
    c.desc = o.str();
}

ag::WalkGeo geo(const Case& c) { return {c.A, c.Uo, c.S, c.N, c.E, c.P_A, c.dst_cross}; }

// Near-equal contiguous element ranges; parts > elements leaves empty slices, as the plan does.
std::vector<std::pair<uint32_t, uint32_t>> make_slices(uint32_t elements, uint32_t parts) {
    std::vector<std::pair<uint32_t, uint32_t>> s;
    for (uint32_t i = 0, e0 = 0; i < parts; ++i) {
        const uint32_t ne = elements / parts + (i < elements % parts ? 1 : 0);
        s.emplace_back(e0, ne);
        e0 += ne;
    }
    return s;
}

// ---- proxies that record what the builder pulls ----

template <class W>
struct RecWalk {
    W& w;
    std::vector<ag::DstRun> runs;
    bool next(ag::DstRun& r) {
        const bool ok = w.next(r);
        if (ok) {
            runs.push_back(r);
        }
        return ok;
    }
};

// Checks that each piece is linear in its source and that a run's pieces cover it in order.
template <class Src>
struct RecSrc {
    Src& s;
    std::function<std::vector<uint64_t>(const ag::DstRun&)> sources;  // each unit's source address
    uint32_t A;
    std::vector<uint64_t> want;
    size_t at = 0;
    void start(const ag::DstRun& r) {
        check(at == want.size(), "pieces did not cover the previous run");
        s.start(r);
        want = sources(r);
        at = 0;
    }
    bool next(ag::Piece& p) {
        if (!s.next(p)) {
            check(at == want.size(), "pieces do not cover the run");
            return false;
        }
        check(p.bytes >= A && p.bytes % A == 0 && p.readable_bytes <= p.bytes, "piece size");
        for (uint32_t b = 0; b < p.bytes; b += A, ++at) {
            check(at < want.size(), "pieces overrun the run");
            check(p.src + b == want[at], "piece not linear in its source");
        }
        return true;
    }
};

struct Stats {
    uint64_t runs = 0, units = 0, packets = 0, segments = 0, slots = 0, packet_bound = 0, passes = 0;
};

// One reader: a CB of two slots, one walk, one builder per source, as ag_reader.cpp holds them. Iterations run
// back to back through the same objects, so builder state leaking across an iteration boundary would show.
template <class Out, class In>
class ReaderSim {
public:
    ReaderSim(const Case& c, const Out& out, const In& in, Stats& st) :
        c_(c),
        out_(out),
        in_(in),
        st_(st),
        walk_(out, geo(c)),
        input_(in, c.A, c.Ui, c.E, c.src_cross),
        relay_(c.A, c.E),
        rwalk_{walk_, {}},
        rinput_{input_, [this](const ag::DstRun& r) { return run_sources(r); }, c.A, {}, 0},
        rrelay_{relay_, [this](const ag::DstRun& r) { return run_sources(r); }, c.A, {}, 0},
        in_b_(rwalk_, rinput_, c.P_A),
        relay_b_(rwalk_, rrelay_, c.P_A),
        lay_(ag::slot_offsets(c.k, c.A, c.hdr_bytes, c.P_A)),
        cadence_(c.cadence) {
        check(lay_.slot_bytes % c.A == 0 && lay_.pay_off % c.A == 0 && lay_.hdr_off % c.A == 0, "slot offsets on A");
        check(lay_.rec_off + c.k * ag::kPacketRecordBytes <= lay_.hdr_off, "records overlap the headers");
        check(lay_.hdr_stride >= c.hdr_bytes && lay_.pay_off == lay_.hdr_off + c.k * lay_.hdr_stride, "header slots");
        mem_.assign(kSlots * lay_.slot_bytes / sizeof(uint64_t), 0);
        // The writer's headers: a pattern the reader must never disturb.
        for (uint32_t s = 0; s < kSlots; ++s) {
            for (uint32_t b = lay_.hdr_off; b < lay_.pay_off; ++b) {
                slot(s)[b] = header_byte(s, b);
            }
        }
    }

    // One iteration: walk positions [skip, skip + take) of elements [e0, e0 + ne) for `stripe`, reading the input
    // (relay = false) or re-reading our own output (relay = true). Returns the elements in walk order.
    std::vector<uint32_t> iteration(
        uint32_t stripe, uint32_t e0, uint32_t ne, uint32_t skip, uint32_t take, bool relay) {
        stripe_ = stripe;
        relay_iter_ = relay;
        e0_ = e0;
        ne_ = ne;
        walk_.init(stripe, e0, ne, skip, take);
        rwalk_.runs.clear();
        run_seen_ = 0;
        pos_.clear();
        next_pos_ = 0;
        delivered_.clear();
        bytes_ = 0;
        const uint32_t take_units = take * c_.E;
        uint32_t consumed = 0;
        while (consumed < take_units) {
            // The relay wait target taken before reserving: every unit this slot reads must lie below it.
            const uint32_t target = std::min(consumed + ag::max_slot_units(c_.k, c_.A, c_.P_A), take_units);
            const uint32_t s = slots_++ % kSlots;
            uint32_t packets = 0, units = 0;
            for (; packets < c_.k; ++packets) {
                const uint32_t bytes = relay ? fill_packet(relay_b_, s, packets) : fill_packet(in_b_, s, packets);
                if (bytes == 0) {
                    break;
                }
                units += bytes / c_.A;
            }
            check(packets != 0, "reserved a slot with nothing left to send");
            consumed += units;
            check(next_pos_ == consumed && consumed <= target, "a slot read past the relay wait target");
            const ag::SlotInfo info{
                static_cast<uint16_t>(packets), static_cast<uint16_t>(consumed == take_units ? 1 : 0), units};
            std::memcpy(slot(s), &info, sizeof(info));
            check_headers(s);
            writer_pop(info);
            ++st_.slots;
        }
        // The walk is exhausted exactly at the iteration's units, and a further call leaves the record alone.
        alignas(8) uint8_t spare[ag::kPacketRecordBytes];
        std::memset(spare, 0x5A, sizeof(spare));
        auto* spare_rec = reinterpret_cast<ag::PacketRecord*>(spare);
        const auto no_read = [](uint64_t, uint32_t, uint32_t) { throw std::runtime_error("read past the walk"); };
        const uint32_t more = relay ? relay_b_.next(spare_rec, no_read) : in_b_.next(spare_rec, no_read);
        check(more == 0 && spare[0] == 0x5A, "builder holds data past the iteration");
        total_units_ += consumed;
        check(signalled_ == total_units_, "SignalCadence did not signal every unit by the iteration end");
        check(delivered_.size() == take_units, "units not delivered exactly once");

        std::vector<uint32_t> seq;
        for (const ag::DstRun& r : rwalk_.runs) {
            for (uint32_t i = 0; i < r.n; ++i) {
                seq.push_back(r.l0 + i * r.lambda);
            }
        }
        check(seq.size() == take, "walk did not take `take` elements");
        st_.runs += rwalk_.runs.size();
        st_.units += take_units;
        st_.packet_bound += (bytes_ + c_.P_A - 1) / c_.P_A;
        ++st_.passes;
        return seq;
    }

private:
    static constexpr uint32_t kSlots = 2;         // CB_SLOTS
    static constexpr uint32_t kCbBase = 0x40000;  // DRAM-aligned, as program.cpp places CBs

    uint8_t* slot(uint32_t s) { return reinterpret_cast<uint8_t*>(mem_.data()) + s * lay_.slot_bytes; }
    static uint8_t header_byte(uint32_t s, uint32_t b) { return static_cast<uint8_t>(0xC3 ^ (s * 31 + b)); }

    void check_headers(uint32_t s) {
        for (uint32_t b = lay_.hdr_off; b < lay_.pay_off; ++b) {
            check(slot(s)[b] == header_byte(s, b), "the reader wrote the writer's header region");
        }
    }

    uint32_t global(uint32_t u) const { return (u / c_.S) * c_.N * c_.S + stripe_ * c_.S + u % c_.S; }
    uint64_t out_addr(uint32_t g) const { return out_.get_noc_addr(g / c_.Uo, (g % c_.Uo) * c_.A); }
    uint64_t in_addr(uint32_t u) const { return in_.get_noc_addr(u / c_.Ui, (u % c_.Ui) * c_.A); }

    std::vector<uint64_t> run_sources(const ag::DstRun& r) const {
        std::vector<uint64_t> want;
        for (uint32_t k = 0; k < r.n; ++k) {
            for (uint32_t j = 0; j < c_.E; ++j) {
                const uint32_t u = (r.l0 + k * r.lambda) * c_.E + j;
                want.push_back(relay_iter_ ? out_addr(global(u)) : in_addr(u));
            }
        }
        return want;
    }

    // Checks the runs the builder pulled since the last call and indexes their units by destination address.
    void index_runs() {
        const ag::WalkShape shape = ag::walk_shape(geo(c_), out_.contiguous_page_stride());
        for (; run_seen_ < rwalk_.runs.size(); ++run_seen_) {
            const ag::DstRun& r = rwalk_.runs[run_seen_];
            check(r.n >= 1 && r.n <= shape.h && r.lambda == shape.lambda, "run longer than the column cap");
            for (uint32_t k = 0; k < r.n; ++k) {
                const uint32_t l = r.l0 + k * r.lambda;
                check(l >= e0_ && l < e0_ + ne_, "element outside the slice");
                for (uint32_t j = 0; j < c_.E; ++j) {
                    const uint32_t u = l * c_.E + j;
                    const uint64_t dst = r.addr + uint64_t{k * c_.E + j} * c_.A;
                    check(out_addr(global(u)) == dst, "dst run not linear");
                    const uint32_t pos = static_cast<uint32_t>(pos_.size());
                    check(pos_.emplace(dst, std::pair{u, pos}).second, "two walked units at one address");
                }
            }
        }
    }

    // Builds packet j of slot s as the reader does (reads into the slot payload, record in place), then plays the
    // writer: decodes the record and checks every unit it delivers, byte by byte.
    template <class B>
    uint32_t fill_packet(B& b, uint32_t s, uint32_t j) {
        struct Rd {
            uint64_t src;
            uint32_t off, bytes;
        };
        std::vector<Rd> reads;
        uint8_t* payload = slot(s) + lay_.pay_off + j * c_.P_A;
        const uint32_t payload_l1 = kCbBase + s * lay_.slot_bytes + lay_.pay_off + j * c_.P_A;
        auto* rec = reinterpret_cast<ag::PacketRecord*>(slot(s) + lay_.rec_off + j * ag::kPacketRecordBytes);
        const uint32_t ra = relay_iter_ ? c_.ro : c_.ri;
        uint64_t local[NOC_SCATTER_WRITE_MAX_CHUNKS] = {};
        auto read = [&](uint64_t src, uint32_t off, uint32_t n) {
            check(n > 0 && off % c_.A == 0 && off + n <= c_.P_A, "read outside the packet payload");
            check((payload_l1 + off - src) % ra == 0, "read not congruent with its source");
            for (uint32_t i = 0; i < n; ++i) {
                payload[off + i] = byte_at(src + i);
            }
            reads.push_back({src, off, n});
        };
        const uint32_t bytes = b.next(rec, read, local);
        index_runs();
        if (bytes == 0) {
            return 0;
        }
        // The host walk's own-NOC and fabric frames coincide, so the reported local segment starts must be the
        // record's.
        for (uint32_t i = 0; i < rec->cmd.chunk_count; ++i) {
            check(local[i] == rec->cmd.noc_address[i], "local segment address differs from the record's");
        }
        ++st_.packets;
        check(bytes <= c_.P_A && bytes % c_.A == 0, "packet size");
        check(rec->payload_bytes == bytes, "record payload size");
        check(rec->cmd.chunk_count >= 1 && rec->cmd.chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS, "segment count");
        st_.segments += rec->cmd.chunk_count;

        // Which payload bytes the reads filled, and from where.
        std::vector<int64_t> src_of(bytes, -1);
        for (const Rd& r : reads) {
            check(r.off + r.bytes <= bytes, "read past the packet's payload bytes");
            for (uint32_t i = 0; i < r.bytes; ++i) {
                check(src_of[r.off + i] < 0, "two reads into one byte");
                src_of[r.off + i] = static_cast<int64_t>(r.src + i);
            }
        }
        uint32_t seg_end = 0;
        ag::for_each_segment(*rec, rec->cmd.noc_address, [&](uint32_t off, uint64_t dst, uint32_t len) {
            check(off == seg_end && len > 0 && len % c_.A == 0, "segment not cut on A");
            seg_end += len;
            for (uint32_t ub = 0; ub < len; ub += c_.A) {
                const auto it = pos_.find(dst + ub);
                check(it != pos_.end(), "segment writes outside this stripe's walked units");
                const auto [u, pos] = it->second;
                check(pos == next_pos_++, "units delivered out of walk order");
                check(delivered_.insert(u).second, "unit delivered twice");
                const uint32_t g = global(u);
                const uint64_t src = relay_iter_ ? out_addr(g) : in_addr(u);
                const uint32_t page_off = relay_iter_ ? (g % c_.Uo) * c_.A : (u % c_.Ui) * c_.A;
                const uint32_t content = relay_iter_ ? c_.co : c_.ci;
                const uint32_t alloc = relay_iter_ ? c_.ao : c_.ai;
                for (uint32_t i = 0; i < c_.A; ++i) {
                    const int64_t from = src_of[off + ub + i];
                    check(from < 0 || page_off + i < alloc, "read past the source page");
                    if (page_off + i < content) {
                        check(from == static_cast<int64_t>(src + i), "a content byte was not read from its source");
                        check(payload[off + ub + i] == byte_at(src + i), "payload byte differs from its source");
                    }
                }
            }
        });
        check(seg_end == bytes, "segments do not add up to the payload");
        bytes_ += bytes;
        return bytes;
    }

    // The writer pops the slot: SignalCadence must signal at least every `cadence` slots.
    void writer_pop(const ag::SlotInfo& info) {
        const uint32_t units = cadence_.on_slot(info);
        signalled_ += units;
        since_signal_ = units == 0 ? since_signal_ + 1 : 0;
        check(since_signal_ < c_.cadence, "SignalCadence skipped a signal");
        check(info.iter_end == 0 || units != 0 || info.units == 0, "no signal at the iteration end");
    }

    const Case& c_;
    const Out& out_;
    const In& in_;
    Stats& st_;
    ag::Walk<Out> walk_;
    ag::InputPieces<In> input_;
    ag::RelayPieces relay_;
    RecWalk<ag::Walk<Out>> rwalk_;
    RecSrc<ag::InputPieces<In>> rinput_;
    RecSrc<ag::RelayPieces> rrelay_;
    ag::PacketBuilder<RecWalk<ag::Walk<Out>>, RecSrc<ag::InputPieces<In>>> in_b_;
    ag::PacketBuilder<RecWalk<ag::Walk<Out>>, RecSrc<ag::RelayPieces>> relay_b_;
    ag::SlotLayout lay_;
    ag::SignalCadence cadence_;
    std::vector<uint64_t> mem_;  // the two slots; uint64_t keeps records 8-aligned

    uint32_t stripe_ = 0, e0_ = 0, ne_ = 0, slots_ = 0, next_pos_ = 0, total_units_ = 0, signalled_ = 0,
             since_signal_ = 0;
    uint64_t bytes_ = 0;
    size_t run_seen_ = 0;
    bool relay_iter_ = false;
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> pos_;  // dst unit address -> (local unit, position)
    std::unordered_set<uint32_t> delivered_;
};

// Every worker of a case: its own stripe from the input, then relays of other stripes, the even-ring halves and a
// resume at a random position, all through one ReaderSim per worker.
template <class Out, class In>
void run_case(const Case& c, const Out& out, const In& in, Stats& st) {
    const uint32_t elements = c.R * c.S / c.E;
    for (const auto& [e0, ne] : make_slices(elements, c.workers)) {
        ReaderSim<Out, In> sim(c, out, in, st);
        const uint32_t dev = rnd(0, c.N - 1);
        const auto full = sim.iteration(dev, e0, ne, 0, ne, false);
        auto sorted = full;
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < ne; ++i) {
            check(sorted[i] == e0 + i, "walk is not a bijection on the slice");
        }
        for (const uint32_t stripe : {(dev + 1) % c.N, rnd(0, c.N - 1)}) {
            check(sim.iteration(stripe, e0, ne, 0, ne, c.relay) == full, "walk order depends on the stripe");
        }
        const uint32_t half = ne / 2;
        const uint32_t at = rnd(0, ne);
        for (const auto& [skip, take] : {std::pair{0u, half}, std::pair{half, ne - half}, std::pair{at, ne - at}}) {
            const auto seq = sim.iteration(rnd(0, c.N - 1), e0, ne, skip, take, c.relay && rng() % 2);
            check(std::equal(seq.begin(), seq.end(), full.begin() + skip), "resumed walk differs");
        }
    }
}

uint32_t pick(std::initializer_list<uint32_t> v) { return *(v.begin() + rng() % v.size()); }

// A random legal geometry: sub-page (A divides both pages) or whole-page (pages match, cut inside a page at A),
// on a random arch, payload, k, header type and cadence.
Case random_case() {
    while (true) {
        Case c{};
        const Arch a = std::vector<Arch>{kWH, kBH, kX}[rng() % 3];
        c.ri = rng() % 2 ? a.dram : a.l1;
        c.ro = rng() % 2 ? a.dram : a.l1;
        c.relay = rng() % 4 != 0;
        // A = lcm(16, every source's read alignment); multicast never reads its output.
        c.A = std::lcm(16u, c.relay ? std::lcm(c.ri, c.ro) : c.ri);
        std::vector<uint32_t> payloads;
        for (const uint32_t p : kPayloads) {
            if (p <= a.burst) {
                payloads.push_back(p);
            }
        }
        const uint32_t P = payloads[rng() % payloads.size()];
        c.P_A = P / c.A * c.A;
        c.N = pick({2, 3, 4, 5, 8});
        c.R = rnd(1, 12);
        c.Pps = rnd(1, 6);
        const bool whole = rng() % 3 == 0;
        if (whole) {
            // Content not a multiple of A; sometimes larger than a packet, so cuts fall inside pages.
            c.ci = c.co = rng() % 4 == 0 ? rnd(P / 2, 2 * P) : rnd(1, 4 * c.A);
            const uint32_t s = ag::align_up(c.ci, c.A);
            c.Ui = c.Uo = s / c.A;
            c.ai = ag::align_up(c.ci, c.ri);
            c.ao = ag::align_up(c.co, c.ro) + (rng() % 4 == 0 ? c.ro * rnd(1, 2) : 0);
            if (c.ao < s) {
                continue;  // the host refuses a unit that would spill into the next output page
            }
        } else {
            c.Ui = rnd(1, 6);
            const uint32_t S = c.Pps * c.Ui;
            std::vector<uint32_t> divs;  // output pages tile each output row
            for (uint32_t d = 1; d <= c.N * S; ++d) {
                if (c.N * S % d == 0) {
                    divs.push_back(d);
                }
            }
            c.Uo = divs[rng() % divs.size()];
            c.ci = c.Ui * c.A;
            c.co = c.Uo * c.A;
            c.ai = c.ci + (rng() % 6 == 0 ? c.ri : 0);
            c.ao = c.co + (rng() % 6 == 0 ? c.ro : 0);
        }
        c.workers = rnd(1, 6);
        c.k = rnd(1, 4);
        c.hdr_bytes = kHeaderBytes[rng() % 2];
        c.cadence = rnd(1, 4);
        finish(c, (std::string(a.name) + (whole ? " whole" : " sub")).c_str());
        return c;
    }
}

// Calls f with an interleaved or a random sharded accessor of `volume` pages at `pitch`.
template <class F>
void with_random_acc(uint32_t volume, uint32_t pitch, std::string& desc, F&& f) {
    if (rng() % 2) {
        const InterleavedAcc acc{pick({1, 7, 8, 12, 64}), pitch, volume, rng() % 8 != 0};
        desc = "interleaved" + std::to_string(acc.banks) + (acc.pitch_runs ? "" : " 1-page runs");
        f(acc);
    } else {
        auto acc = random_sharded(volume, pitch, desc);
        f(*acc->acc);
    }
}

}  // namespace

#ifndef AG_KERNEL_LIB_ITERS
#define AG_KERNEL_LIB_ITERS 300
#endif

TEST(AllGatherKernelLib, RandomGeometries) {
    Stats st;
    for (uint32_t it = 0; it < AG_KERNEL_LIB_ITERS; ++it) {
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
    std::cout << "passes " << st.passes << " runs " << st.runs << " units " << st.units << " slots " << st.slots
              << " packets " << st.packets << " (lower bound " << st.packet_bound << ") segments " << st.segments
              << "\n";
}

// Device-matrix shapes on interleaved DRAM: BH (8 banks, A = 64) at every payload, WH (12 banks, A = 32) at the
// payloads its 8 KB burst allows; k = 1, 2 and 4 (the plan's range); 2 links x n = 2 and 3.
TEST(AllGatherKernelLib, DeviceShapes) {
    struct Shape {
        uint32_t ci, Pps, R;
    };
    const std::vector<Shape> shapes = {
        {2048, 10, 32},  // tile bf16 [1024, 2560] d3: stripe 10 pages, no cross_row on 8 banks (8 % 80)
        {1088, 1, 8},    // bfp8 [256, 256] d3: one page per stripe-row, cross_row
        {8192, 1, 32},   // RM 8 KB rows
        {20000, 1, 64},  // RM 20000 B rows: whole-page, cuts inside pages
        {48, 1, 512},    // RM 48 B rows
        {16, 1, 64},     // RM 16 B rows
        {2048, 32, 13},  // tile bf16 [416, 8192] d3: odd rows, slices start mid-row
    };
    Stats st;
    for (const Arch& a : {kBH, kWH}) {
        const uint32_t banks = a.burst == kBH.burst ? 8 : 12;
        for (const uint32_t P : kPayloads) {
            if (P > a.burst) {
                continue;
            }
            for (const Shape& sh : shapes) {
                for (const auto [n, k] : {std::pair{2u, 1u}, {2u, 2u}, {3u, 4u}}) {
                    Case c{};
                    c.A = c.ri = c.ro = a.dram;
                    c.P_A = P / c.A * c.A;
                    c.N = 8;
                    c.R = sh.R;
                    c.Pps = sh.Pps;
                    c.ci = c.co = sh.ci;
                    c.ai = c.ao = ag::align_up(sh.ci, c.A);
                    c.Ui = c.Uo = c.ai / c.A;
                    c.relay = true;
                    c.workers = 2 * n;
                    c.k = k;
                    c.hdr_bytes = kHeaderBytes[n % 2];
                    c.cadence = 2;
                    finish(c, a.name);
                    const InterleavedAcc out{banks, c.ao, c.R * c.N * c.Pps};
                    const InterleavedAcc in{banks, c.ai, c.R * c.Pps};
                    try {
                        run_case(c, out, in, st);
                    } catch (const std::exception& e) {
                        FAIL() << e.what() << "\n  " << c.desc;
                    }
                }
            }
        }
    }
    std::cout << "passes " << st.passes << " runs " << st.runs << " packets " << st.packets << " (lower bound "
              << st.packet_bound << ")\n";
}

// A slice that starts mid-row (2 slices, odd row count) walks as many runs as one that starts a row: tiles anchored
// mid-row would straddle every row end (bf16 tile [416, 8192] d3 on BH: 104 runs of 2 pages, 40% slower).
TEST(AllGatherKernelLib, MidRowSlice) {
    for (const uint32_t R : {1u, 3u, 13u}) {
        Case c{};
        c.A = c.ri = c.ro = 64;
        c.P_A = 15168;
        c.N = 8;
        c.R = R;
        c.Pps = 32;
        c.ci = c.co = c.ai = c.ao = 2048;
        c.Ui = c.Uo = 32;
        c.relay = true;
        c.workers = 2;
        c.k = 2;
        c.hdr_bytes = kHeaderBytes[0];
        c.cadence = 2;
        finish(c, "midrow");
        const InterleavedAcc out{8, c.ao, c.R * c.N * c.Pps};
        const InterleavedAcc in{8, c.ai, c.R * c.Pps};
        std::vector<uint64_t> runs;
        for (const auto& [e0, ne] : make_slices(c.R * c.Pps, c.workers)) {
            Stats st;
            ReaderSim<InterleavedAcc, InterleavedAcc> sim(c, out, in, st);
            sim.iteration(0, e0, ne, 0, ne, false);
            runs.push_back(st.runs);
        }
        ASSERT_EQ(runs.size(), 2u);
        EXPECT_EQ(runs[0], runs[1]) << c.desc;
        if (R == 13) {
            EXPECT_EQ(runs[1], 56u);  // 6.5 rows x 8 lanes, one run per lane per row
        }
    }
}

// walk_shape on the cases the cost model depends on.
TEST(AllGatherKernelLib, WalkShape) {
    // 2 KB tiles, 10 pages per stripe-row, 8 banks: rows do not tile the banks, so runs stop at the row end.
    constexpr auto tile = ag::walk_shape({64, 32, 320, 8, 32, 15168, true}, 8);
    static_assert(!tile.cross_row && tile.lambda == 8 && tile.h == 2);  // ceil(10 pages / 8 lanes)
    // One page per stripe-row: L / N rows apart is the next page of the same bank.
    constexpr auto narrow = ag::walk_shape({64, 17, 17, 8, 17, 15168, true}, 8);
    static_assert(narrow.cross_row && narrow.lambda == 1 && narrow.h == 15168 / (17 * 64));
    // Sub-page elements walk ascending units, capped at a packet.
    constexpr auto sub = ag::walk_shape({32, 4, 6, 4, 1, 7616, true}, 12);
    static_assert(!sub.cross_row && sub.lambda == 1 && sub.h == 7616 / 32);
    // A page bigger than the packet still walks one element per column.
    constexpr auto big = ag::walk_shape({64, 313, 313, 8, 313, 15168, true}, 8);
    static_assert(big.h == 1);
    // Slot layout: offsets on A, records before headers, payloads last.
    constexpr auto lay = ag::slot_offsets(2, 64, 96, 15168);
    static_assert(lay.rec_off == 8 && lay.hdr_off == 128 && lay.hdr_stride == 128 && lay.pay_off == 384);
    static_assert(lay.pay_stride == 15168 && lay.slot_bytes == 384 + 2 * 15168 && lay.slot_bytes % 64 == 0);
    static_assert(lay.record(1000, 1) == 1000 + 8 + 48 && lay.header(1000, 1) == 1000 + 256);
    static_assert(lay.payload(1000, 1) == 1000 + 384 + 15168);
    static_assert(ag::max_slot_units(2, 64, 15168) == 2 * 237);
    SUCCEED();
}
