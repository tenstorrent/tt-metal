// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host proof of the all_gather page walk (ttnn/.../all_gather/device/kernels/gather_walk.hpp, v2) and of the
// integer rules the host and kernels share (kernels/gather_rules.hpp):
//   - the k = 1, rot = 0 order is the v1 TiledWalk (LegacyTiledWalk, a verbatim copy of it, is the oracle);
//   - the walk is a bijection for every k and rot, and walk_seek() resumes exactly;
//   - every run next_run() forms is one linear stretch of memory on interleaved (mock) and sharded (real
//     TensorAccessor) tensors, on the input (split) and output (StripeMap, concat) sides;
//   - reader and writer group a CB entry the same way (packed CB), including k > 1, rot and resume;
//   - the even-ring halves partition the slice under the antipode rule; StripeMap scales by k;
//   - the host policy rules (choose_k, xfer_for_segments, rotation targets, the antipode rule, run-source CT
//     encoding, the padded-slot run loops) and the kernel arg tables.
//
// No device: the walk, StripeMap and run queries are pure integer code. Contiguity comes from a mock of the
// interleaved accessor (whose real specialization only exists under KERNEL_BUILD) and from real sharded
// TensorAccessor instances, checked against their own get_bank_and_offset(). WalkGeometry::make (which needs
// Buffers and hal) is not built here; the rules it applies are.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define HOST_KERNEL_SHIM_KEEP_MACROS  // gather_walk.hpp and the verbatim walk copy below need them
#include "host_kernel_shim.hpp"

#include "ttnn/operations/ccl/all_gather/device/kernels/gather_args.hpp"
#include "ttnn/operations/ccl/all_gather/device/kernels/gather_walk.hpp"

namespace {

// ---------------------------------------------------------------------------------------------------
// LegacyTiledWalk: verbatim copy of v1 TiledWalk (gather_walk.hpp at 34d37d84a89), renamed only.
// ---------------------------------------------------------------------------------------------------
class LegacyTiledWalk {
public:
    FORCE_INLINE void init(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer) {
        stride_ = stride;
        xfer_ = xfer;
        tile_ = xfer * stride;
        const uint32_t full = count / tile_;
        const uint32_t rem = count - full * tile_;
        tail_first_ = first + full * tile_;
        hb_ = rem / stride;
        he_ = rem - hb_ * stride;
        lane_ = 0;
        k_ = 0;
        if (count == 0) {
            tile_first_ = c_ = first;
            return;
        }
        if (skip < full * tile_) {
            const uint32_t tile = skip / tile_;
            const uint32_t in_tile = skip - tile * tile_;
            tile_first_ = first + tile * tile_;
            lane_ = in_tile / xfer;
            k_ = in_tile - lane_ * xfer;
        } else {
            tile_first_ = tail_first_;
            const uint32_t s = skip - full * tile_;
            const uint32_t tall = he_ * (hb_ + 1);
            if (s < tall) {
                lane_ = s / (hb_ + 1);
                k_ = s - lane_ * (hb_ + 1);
            } else {
                const uint32_t rest = s - tall;
                lane_ = he_ + rest / hb_;
                k_ = rest - (lane_ - he_) * hb_;
            }
        }
        c_ = tile_first_ + lane_ + k_ * stride_;
    }

    FORCE_INLINE uint32_t chunk() const { return c_; }
    FORCE_INLINE uint32_t run_limit() const { return col() - k_; }

    FORCE_INLINE void advance(uint32_t n) {
        ASSERT(n != 0 && n <= run_limit());
        k_ += n;
        if (k_ < col()) {
            c_ += n * stride_;
            return;
        }
        k_ = 0;
        if (++lane_ == lanes()) {
            lane_ = 0;
            tile_first_ += tile_;
        }
        c_ = tile_first_ + lane_;
    }

private:
    FORCE_INLINE bool in_tail() const { return tile_first_ == tail_first_; }
    FORCE_INLINE uint32_t col() const { return in_tail() ? hb_ + (lane_ < he_ ? 1u : 0u) : xfer_; }
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && hb_ == 0) ? he_ : stride_; }

    uint32_t c_, stride_, xfer_, tile_, lane_, k_, tile_first_, tail_first_, hb_, he_;
};

// ---------------------------------------------------------------------------------------------------
// Spec: the v2 order by definition, independent of the incremental walk.
// ---------------------------------------------------------------------------------------------------
uint32_t chunk_at(uint32_t first, uint32_t lane, uint32_t q, uint32_t stride, uint32_t k) {
    return (first / k + lane + (q / k) * stride) * k + q % k;
}

std::vector<uint32_t> lane_lens(uint32_t count, uint32_t stride, uint32_t k) {
    const uint32_t n = count / k;
    std::vector<uint32_t> lens(stride);
    for (uint32_t l = 0; l < stride; ++l) {
        lens[l] = l < n ? k * ((n - l + stride - 1) / stride) : 0;
    }
    return lens;
}

uint32_t full_tiles(uint32_t count, uint32_t stride, uint32_t xfer, uint32_t k) {
    return k * ((count / k) / stride) / xfer;
}

std::vector<std::vector<uint32_t>> spec_columns(
    uint32_t first, uint32_t count, uint32_t stride, uint32_t xfer, uint32_t k, uint32_t rot) {
    const auto lens = lane_lens(count, stride, k);
    const uint32_t F = full_tiles(count, stride, xfer, k);
    std::vector<std::vector<uint32_t>> cols;
    for (uint32_t t = 0; t * xfer < lens[0]; ++t) {
        const uint32_t q0 = t * xfer;
        for (uint32_t j = 0; j < stride; ++j) {
            const uint32_t l = t < F ? (j + rot) % stride : j;
            std::vector<uint32_t> col;
            for (uint32_t q = q0; q < std::min(q0 + xfer, lens[l]); ++q) {
                col.push_back(chunk_at(first, l, q, stride, k));
            }
            if (!col.empty()) {
                cols.push_back(std::move(col));
            }
        }
    }
    return cols;
}

// Exact chunks left in c's lane below end: what an end-clipped run_limit would be.
template <uint32_t K>
uint32_t lane_room(uint32_t c, uint32_t end, uint32_t stride) {
    uint32_t n = 0;
    while (c < end) {
        ++n;
        c = c + 1 + (((c & (K - 1)) + 1) / K) * ((stride - 1) * K);
    }
    return n;
}

template <uint32_t K>
TiledWalk<K> make_walk(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer, uint32_t rot) {
    TiledWalk<K> w;
    w.init(first + count, stride, xfer, rot, walk_seek(first, count, skip, stride, xfer, K));
    return w;
}

template <uint32_t K>
std::vector<uint32_t> trace(
    uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer, uint32_t rot) {
    auto w = make_walk<K>(first, count, skip, stride, xfer, rot);
    std::vector<uint32_t> out;
    for (uint32_t s = skip; s < count; ++s) {
        out.push_back(w.chunk());
        w.advance(1);
    }
    return out;
}

std::string ctx(
    uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer, uint32_t k, uint32_t rot) {
    return "first=" + std::to_string(first) + " count=" + std::to_string(count) + " skip=" + std::to_string(skip) +
           " stride=" + std::to_string(stride) + " xfer=" + std::to_string(xfer) + " k=" + std::to_string(k) +
           " rot=" + std::to_string(rot);
}

constexpr uint32_t kStrides[] = {1, 2, 3, 7, 8, 12};
constexpr uint32_t kStridesDense[] = {1, 2, 3, 7, 8, 12, 13};
constexpr uint32_t kXfers[] = {1, 2, 3, 4, 7, 9, 14};

}  // namespace

// ---------------------------------------------------------------------------------------------------
// (a) k = 1, rot = 0: same chunks as LegacyTiledWalk; run_limit clipped at end == Legacy's run_limit;
// resuming at every skip lands in the stepped walk's exact state (=> same suffix), and on Legacy's chunk.
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, MatchesLegacy) {
    for (uint32_t stride : kStrides) {
        for (uint32_t xfer : kXfers) {
            for (uint32_t first : {0u, 5u, 1000u}) {
                for (uint32_t count = 0; count < 300; ++count) {
                    SCOPED_TRACE(ctx(first, count, 0, stride, xfer, 1, 0));
                    const uint32_t end = first + count;
                    LegacyTiledWalk legacy;
                    legacy.init(first, count, 0, stride, xfer);
                    auto w = make_walk<1>(first, count, 0, stride, xfer, 0);
                    std::vector<TiledWalk<1>> states;
                    states.reserve(count);
                    for (uint32_t s = 0; s < count; ++s) {
                        ASSERT_EQ(w.chunk(), legacy.chunk()) << "seqno " << s;
                        ASSERT_EQ(std::min(w.run_limit(), lane_room<1>(w.chunk(), end, stride)), legacy.run_limit())
                            << "seqno " << s;
                        states.push_back(w);
                        w.advance(1);
                        legacy.advance(1);
                    }
                    for (uint32_t skip = 0; skip < count; ++skip) {
                        const auto resumed = make_walk<1>(first, count, skip, stride, xfer, 0);
                        ASSERT_TRUE(resumed == states[skip]) << "skip " << skip;
                        LegacyTiledWalk lg;
                        lg.init(first, count, skip, stride, xfer);
                        ASSERT_EQ(resumed.chunk(), lg.chunk()) << "skip " << skip;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// (b) + (c): for k in {1,2,4,8} and every rot: the walk is the spec's column order (so a bijection onto
// the slice), run_limit is the column room (exact once clipped at end), whole-column steps land on
// column starts, and walk_seek at every skip reproduces the stepped state.
// ---------------------------------------------------------------------------------------------------
namespace {
template <uint32_t K>
void bijection_and_seek(uint64_t& walks, uint64_t& seeks) {
    for (uint32_t stride : kStrides) {
        for (uint32_t xfer : kXfers) {
            for (uint32_t first_page : {0u, 3u}) {
                for (uint32_t pages = 0; pages < 40; ++pages) {
                    const uint32_t first = first_page * K, count = pages * K, end = first + count;
                    for (uint32_t rot = 0; rot < stride; ++rot) {
                        SCOPED_TRACE(ctx(first, count, 0, stride, xfer, K, rot));
                        const auto cols = spec_columns(first, count, stride, xfer, K, rot);
                        std::vector<uint32_t> flat;
                        for (const auto& c : cols) {
                            flat.insert(flat.end(), c.begin(), c.end());
                        }
                        std::vector<uint32_t> sorted = flat;
                        std::sort(sorted.begin(), sorted.end());
                        std::vector<uint32_t> want(count);
                        std::iota(want.begin(), want.end(), first);
                        ASSERT_EQ(sorted, want) << "spec is not a permutation";

                        auto w = make_walk<K>(first, count, 0, stride, xfer, rot);
                        std::vector<TiledWalk<K>> states;
                        for (const auto& col : cols) {
                            for (size_t j = 0; j < col.size(); ++j) {
                                ASSERT_EQ(w.chunk(), col[j]);
                                ASSERT_LE(w.run_limit(), xfer);
                                ASSERT_EQ(
                                    std::min(w.run_limit(), lane_room<K>(w.chunk(), end, stride)), col.size() - j);
                                states.push_back(w);
                                w.advance(1);
                            }
                        }
                        auto w2 = make_walk<K>(first, count, 0, stride, xfer, rot);
                        for (const auto& col : cols) {
                            ASSERT_EQ(w2.chunk(), col[0]);
                            w2.advance(col.size());
                        }
                        ++walks;
                        for (uint32_t skip = 0; skip < count; ++skip) {
                            ASSERT_TRUE(make_walk<K>(first, count, skip, stride, xfer, rot) == states[skip])
                                << "skip " << skip;
                            ++seeks;
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

TEST(AllGatherWalk, BijectionAndSeek) {
    uint64_t walks = 0, seeks = 0;
    bijection_and_seek<1>(walks, seeks);
    bijection_and_seek<2>(walks, seeks);
    bijection_and_seek<4>(walks, seeks);
    bijection_and_seek<8>(walks, seeks);
    EXPECT_GT(walks, 0u);
    EXPECT_GT(seeks, 0u);
}

// CT stride/xfer (the kernels' form) is the same walk as runtime stride/xfer.
TEST(AllGatherWalk, CompileTimeParamsMatchRuntime) {
    for (uint32_t pages = 0; pages < 100; ++pages) {
        for (uint32_t rot = 0; rot < 8; ++rot) {
            const uint32_t count = pages * 4;
            TiledWalk<4, 8, 7> ct;
            ct.init(count, rot, walk_start(0));
            auto rt = make_walk<4>(0, count, 0, 8, 7, rot);
            for (uint32_t s = 0; s < count; ++s) {
                ASSERT_EQ(ct.chunk(), rt.chunk());
                ASSERT_EQ(ct.run_limit(), rt.run_limit());
                const uint32_t n = std::min(ct.run_limit(), lane_room<4>(ct.chunk(), count, 8));
                ct.advance(n);
                rt.advance(n);
                s += n - 1;
            }
        }
    }
}

TEST(AllGatherWalk, AdvancePreconditionAsserts) {
    auto w = make_walk<2>(0, 40, 0, 3, 4, 1);
    EXPECT_THROW(w.advance(0), std::out_of_range);
    EXPECT_THROW(w.advance(w.run_limit() + 1), std::out_of_range);
    EXPECT_NO_THROW(w.advance(w.run_limit()));
    // A run past the slice end (a caller bound that was not the slice end) is caught.
    // stride 3, xfer 4, 10 chunks: lane 1 is 1, 4, 7 -- at 7 the column has room 2 but the lane 1.
    auto t = make_walk<1>(0, 10, 6, 3, 4, 0);
    ASSERT_EQ(t.chunk(), 7u);
    ASSERT_GT(t.run_limit(), 1u);
    EXPECT_THROW(t.advance(2), std::out_of_range);
}

// ---------------------------------------------------------------------------------------------------
// (i) BIJECTION at k = 1, dense in count: the walk over [first, first + count) emits each chunk exactly
// once; its columns are `stride` apart and at most `xfer` tall (end-clipped room: min(run_limit, lane
// room)); stepping a whole column at once gives the same order.
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, Bijection) {
    for (uint32_t stride : kStridesDense) {
        for (uint32_t xfer : kXfers) {
            for (uint32_t first : {0u, 5u, 1000u}) {
                for (uint32_t count = 0; count <= 300; ++count) {
                    SCOPED_TRACE(ctx(first, count, 0, stride, xfer, 1, 0));
                    const uint32_t end = first + count;
                    std::vector<uint32_t> chunks, limits;
                    auto w = make_walk<1>(first, count, 0, stride, xfer, 0);
                    for (uint32_t s = 0; s < count; ++s) {
                        chunks.push_back(w.chunk());
                        limits.push_back(std::min(w.run_limit(), lane_room<1>(w.chunk(), end, stride)));
                        w.advance(1);
                    }
                    std::vector<uint32_t> sorted = chunks;
                    std::sort(sorted.begin(), sorted.end());
                    for (uint32_t i = 0; i < count; ++i) {
                        ASSERT_EQ(sorted[i], first + i) << "not a permutation of the range";
                    }
                    for (uint32_t s = 0; s < count; ++s) {
                        ASSERT_GE(limits[s], 1u) << "seqno " << s;
                        ASSERT_LE(limits[s], xfer) << "seqno " << s;
                        ASSERT_LE(limits[s], count - s) << "seqno " << s << ": column reaches past the range";
                        if (limits[s] > 1) {
                            ASSERT_EQ(chunks[s + 1], chunks[s] + stride) << "seqno " << s;
                            ASSERT_EQ(limits[s + 1], limits[s] - 1) << "seqno " << s;
                        }
                    }
                    // Whole-column steps (advance(n > 1)) land on the same chunks.
                    auto w2 = make_walk<1>(first, count, 0, stride, xfer, 0);
                    for (uint32_t s = 0; s < count;) {
                        ASSERT_EQ(w2.chunk(), chunks[s]) << "seqno " << s;
                        const uint32_t n = limits[s];
                        s += n;
                        w2.advance(n);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// (ii) SEEK at k = 1 against the oracle: resuming at seqno `skip` (host walk_seek -> init) gives the same
// chunks and (end-clipped) run limits as LegacyTiledWalk::init(skip) from there on. The unicast even-ring
// split depends on this (final_skip -> final_pos).
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, Seek) {
    for (uint32_t stride : {1u, 3u, 7u, 8u}) {
        for (uint32_t xfer : {1u, 2u, 3u, 7u}) {
            for (uint32_t first : {0u, 1000u}) {
                for (uint32_t count = 0; count <= 80; ++count) {
                    const uint32_t end = first + count;
                    for (uint32_t skip = 0; skip < count; ++skip) {
                        SCOPED_TRACE(ctx(first, count, skip, stride, xfer, 1, 0));
                        auto w = make_walk<1>(first, count, skip, stride, xfer, 0);
                        LegacyTiledWalk lg;
                        lg.init(first, count, skip, stride, xfer);
                        for (uint32_t s = skip; s < count; ++s) {
                            ASSERT_EQ(w.chunk(), lg.chunk()) << "seqno " << s;
                            ASSERT_EQ(std::min(w.run_limit(), lane_room<1>(w.chunk(), end, stride)), lg.run_limit());
                            w.advance(1);
                            lg.advance(1);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// (iv) STRIPE_MAP: over every stripe, StripeMap::at is a bijection from local chunk ids onto the output's
// chunks, keeps rows, and (with the slice end past the last row) bound is the end of our stripe in that row.
// ---------------------------------------------------------------------------------------------------
namespace {
template <uint32_t cps, uint32_t N>
void check_stripe_map() {
    SCOPED_TRACE("cps=" + std::to_string(cps) + " N=" + std::to_string(N));
    constexpr uint32_t rows = 6;
    std::vector<bool> seen(rows * N * cps, false);
    for (uint32_t stripe = 0; stripe < N; ++stripe) {
        StripeMap<cps, N> map;
        map.init(stripe, rows * cps);
        for (uint32_t local = 0; local < rows * cps; ++local) {
            const auto pos = map.at(local);
            ASSERT_LT(pos.global, seen.size()) << "stripe " << stripe << " local " << local;
            ASSERT_FALSE(seen[pos.global]) << "stripe " << stripe << " local " << local << " collides";
            seen[pos.global] = true;
            ASSERT_LT(pos.global, pos.bound) << "stripe " << stripe << " local " << local;
            ASSERT_EQ(pos.bound - pos.global, cps - local % cps) << "stripe " << stripe << " local " << local;
            ASSERT_EQ(pos.global / (N * cps), local / cps) << "row changed";
            ASSERT_EQ(pos.global % (N * cps) / cps, stripe) << "landed in another stripe";
        }
    }
    for (size_t g = 0; g < seen.size(); ++g) {
        ASSERT_TRUE(seen[g]) << "global " << g << " never written";
    }
}

template <uint32_t cps, uint32_t... Ns>
void check_stripe_maps() {
    (check_stripe_map<cps, Ns>(), ...);
}
}  // namespace

TEST(AllGatherWalk, StripeMap) {
    check_stripe_maps<1, 2, 3, 4, 8>();
    check_stripe_maps<2, 2, 3, 4, 8>();
    check_stripe_maps<5, 2, 3, 4, 8>();
    check_stripe_maps<8, 2, 3, 4, 8>();
    check_stripe_maps<10, 2, 3, 4, 8>();
    check_stripe_maps<16, 2, 3, 4, 8>();
}

// ---------------------------------------------------------------------------------------------------
// chunks_per_transfer: min(packet, burst, cap) / unit, at least 1, for both arch bursts.
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, ChunksPerTransfer) {
    for (uint32_t burst : {host_kernel_shim::kNocMaxBurstSizeWormhole, host_kernel_shim::kNocMaxBurstSizeBlackhole}) {
        for (uint32_t packet : {2048u, 4096u, 7616u, 15232u, 16384u, 32768u}) {
            for (uint32_t chunk : {16u, 32u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u}) {
                for (uint32_t cap : {0u, 1024u, 4096u, 100000u}) {
                    uint32_t limit = std::min(packet, burst);
                    if (cap != 0) {
                        limit = std::min(limit, cap);
                    }
                    const uint32_t want = std::max(1u, limit / chunk);
                    EXPECT_EQ(chunks_per_transfer(packet, chunk, cap, burst), want)
                        << "burst " << burst << " packet " << packet << " chunk " << chunk << " cap " << cap;
                }
            }
        }
    }
    static_assert(chunks_per_transfer(15232, 2048, 0, host_kernel_shim::kNocMaxBurstSizeBlackhole) == 7);
    static_assert(chunks_per_transfer(15232, 2048, 0, host_kernel_shim::kNocMaxBurstSizeWormhole) == 4);
}

// ---------------------------------------------------------------------------------------------------
// (d) RUN LINEARITY
// ---------------------------------------------------------------------------------------------------
namespace {

struct MockInterleavedAccessor {
    uint32_t num_banks;
    uint32_t aligned_page;
    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;
    };
    uint32_t contiguous_page_stride() const { return num_banks; }
    uint32_t num_contiguous_pages(uint32_t page_id, uint32_t end_page_id) const {
        ASSERT(page_id < end_page_id);
        return (end_page_id - page_id + num_banks - 1) / num_banks;
    }
    uint32_t get_aligned_page_size() const { return aligned_page; }
    PageMapping get_bank_and_offset(uint32_t page_id) const { return {page_id % num_banks, page_id / num_banks}; }
};

template <size_t... Dims>
using U32 = tensor_accessor::ArrayStaticWrapperU32<Dims...>;
template <size_t... Dims>
using U16 = tensor_accessor::ArrayStaticWrapperU16<Dims...>;

constexpr uint32_t kChunk = 32;

struct Loc {
    size_t bank;
    size_t byte;
    bool operator==(const Loc&) const = default;
};

template <uint32_t cpp, typename Accessor>
Loc chunk_loc(const Accessor& acc, uint32_t chunk) {
    const auto m = acc.get_bank_and_offset(page_of<cpp>(chunk));
    return {
        static_cast<size_t>(m.bank_id),
        static_cast<size_t>(m.bank_page_offset) * acc.get_aligned_page_size() + byte_off_of<cpp, kChunk>(chunk)};
}

uint64_t pack(Loc l) { return (static_cast<uint64_t>(l.bank) << 40) | static_cast<uint64_t>(l.byte); }

struct Target {
    uint32_t chunk;
    uint32_t bound;
};

struct RunStats {
    uint64_t cases = 0, runs = 0, multi = 0, cross = 0;
};

// The kernels' inner loop: batches of `batch`, `left` counting down, next_run; every run linear, below
// its bound, inside the column; the runs cover exactly the walk's seqnos [skip, count).
template <uint32_t K, uint32_t cpp, typename Accessor, typename ToAcc>
void check_runs(
    const Accessor& acc,
    RunSource src,
    uint32_t first,
    uint32_t count,
    uint32_t skip,
    uint32_t stride,
    uint32_t xfer,
    uint32_t rot,
    uint32_t batch_size,
    ToAcc to_acc,
    RunStats& stats) {
    auto walk = make_walk<K>(first, count, skip, stride, xfer, rot);
    std::vector<uint32_t> seen;
    const uint32_t take = count - skip;
    for (uint32_t sent = 0; sent < take;) {
        const uint32_t batch = std::min(batch_size, take - sent);
        for (uint32_t left = batch; left > 0;) {
            const Target t = to_acc(walk.chunk());
            const uint32_t limit = walk.run_limit();
            const uint32_t run = next_run<K, cpp>(walk, acc, src, t.chunk, t.bound, left);
            ASSERT_GE(run, 1u);
            ASSERT_LE(run, limit);
            ASSERT_LE(run, left);
            const Loc base = chunk_loc<cpp>(acc, t.chunk);
            auto probe = walk;
            for (uint32_t i = 0; i < run; ++i) {
                const Target ti = to_acc(probe.chunk());
                ASSERT_LT(ti.chunk, t.bound) << "run from " << t.chunk << " crosses its bound at i " << i;
                ASSERT_EQ(chunk_loc<cpp>(acc, ti.chunk), (Loc{base.bank, base.byte + i * kChunk}))
                    << "run from " << t.chunk << " is not linear at i " << i;
                seen.push_back(probe.chunk());
                probe.advance(1);
            }
            auto addr = [&](uint32_t local) { return pack(chunk_loc<cpp>(acc, to_acc(local).chunk)); };
            ASSERT_TRUE(run_is_linear(walk, run, kChunk, pack(base), addr));
            ++stats.runs;
            stats.multi += run > 1;
            stats.cross += run > K - t.chunk % K;
            walk.advance(run);
            left -= run;
        }
        sent += batch;
    }
    std::vector<uint32_t> want = trace<K>(first, count, skip, stride, xfer, rot);
    std::sort(want.begin(), want.end());
    std::sort(seen.begin(), seen.end());
    ASSERT_EQ(seen, want) << "runs do not cover the walk exactly once";
    ++stats.cases;
}

constexpr uint32_t kRunXfers[] = {1, 2, 3, 5, 8, 12};
constexpr uint32_t kBatches[] = {1, 3, 16, 1u << 20};

// Input side (reader iteration 0): the walk runs over local chunk ids == input chunk ids, an input page
// is `split` walk pages (cpp = split * K); bound = slice end. Walk strides: the natural one and others.
template <uint32_t K, uint32_t split, typename Accessor>
void sweep_input(const Accessor& acc, uint32_t in_pages, RunStats& stats) {
    constexpr uint32_t cpp = split * K;
    const uint32_t page_stride = acc.contiguous_page_stride();
    const bool packed = acc.get_aligned_page_size() == cpp * kChunk;
    std::vector<uint32_t> strides = {page_stride, 1u, 2u, 3u};
    std::sort(strides.begin(), strides.end());
    strides.erase(std::unique(strides.begin(), strides.end()), strides.end());
    const uint32_t walk_pages = in_pages * split;
    for (uint32_t stride : strides) {
        const RunSource src = run_source(packed, K, cpp, page_stride, stride);
        for (uint32_t workers : {1u, 3u}) {
            for (uint32_t w = 0; w < workers; ++w) {
                // Slices are cut in input pages.
                const uint32_t fp = in_pages * w / workers * split, ep = in_pages * (w + 1) / workers * split;
                (void)walk_pages;
                const uint32_t first = fp * K, count = (ep - fp) * K, end = first + count;
                auto to_acc = [end](uint32_t local) { return Target{local, end}; };
                for (uint32_t xfer : kRunXfers) {
                    for (uint32_t rot : {0u, stride - 1, stride / 2}) {
                        for (uint32_t skip : {0u, count / 2, count ? count - 1 : 0u}) {
                            if (count != 0 && skip >= count) {
                                continue;
                            }
                            for (uint32_t batch : kBatches) {
                                SCOPED_TRACE(
                                    "input split=" + std::to_string(split) + " packed=" + std::to_string(packed) +
                                    " batch=" + std::to_string(batch) + " " +
                                    ctx(first, count, skip, stride, xfer, K, rot));
                                check_runs<K, cpp>(
                                    acc, src, first, count, skip, stride, xfer, rot, batch, to_acc, stats);
                                if (::testing::Test::HasFatalFailure()) {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Output side (writer, relaying reader, multicast sender): StripeMap with cps = cps_pages * K and the
// slice-end bound; an output page is m walk pages (m > 1: concat, walk stride 1).
template <uint32_t K, uint32_t m, uint32_t cps_pages, uint32_t N, typename Accessor>
void sweep_output_one(const Accessor& acc, uint32_t out_pages, RunStats& stats) {
    constexpr uint32_t cpp = m * K, cps = cps_pages * K;
    const uint32_t total = out_pages * cpp;
    if (total % (cps * N) != 0) {
        return;
    }
    const uint32_t stride = m > 1 ? 1u : acc.contiguous_page_stride();
    // As WalkGeometry::make derives it (slot == chunk).
    const RunSource src =
        run_source(acc.get_aligned_page_size() == cpp * kChunk, K, cpp, acc.contiguous_page_stride(), stride);
    const uint32_t local_pages = total / N / K;
    for (uint32_t stripe : {0u, N - 1}) {
        for (uint32_t workers : {1u, 3u}) {
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t fp = local_pages * w / workers, ep = local_pages * (w + 1) / workers;
                const uint32_t first = fp * K, count = (ep - fp) * K;
                StripeMap<cps, N> map;
                map.init(stripe, first + count);
                auto to_acc = [&map](uint32_t local) {
                    const auto pos = map.at(local);
                    return Target{pos.global, pos.bound};
                };
                for (uint32_t xfer : kRunXfers) {
                    for (uint32_t rot : {0u, stride - 1}) {
                        for (uint32_t skip : {0u, count / 2}) {
                            if (count != 0 && skip >= count) {
                                continue;
                            }
                            for (uint32_t batch : {3u, 1u << 20}) {
                                SCOPED_TRACE(
                                    "output m=" + std::to_string(m) + " cps_pages=" + std::to_string(cps_pages) +
                                    " N=" + std::to_string(N) + " stripe=" + std::to_string(stripe) + " batch=" +
                                    std::to_string(batch) + " " + ctx(first, count, skip, stride, xfer, K, rot));
                                check_runs<K, cpp>(
                                    acc, src, first, count, skip, stride, xfer, rot, batch, to_acc, stats);
                                if (::testing::Test::HasFatalFailure()) {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <uint32_t K, uint32_t m, typename Accessor>
void sweep_output(const Accessor& acc, uint32_t out_pages, RunStats& stats) {
    sweep_output_one<K, m, 1, 2>(acc, out_pages, stats);
    sweep_output_one<K, m, 2, 2>(acc, out_pages, stats);
    sweep_output_one<K, m, 3, 2>(acc, out_pages, stats);
    sweep_output_one<K, m, 6, 2>(acc, out_pages, stats);
    sweep_output_one<K, m, 1, 3>(acc, out_pages, stats);
    sweep_output_one<K, m, 2, 3>(acc, out_pages, stats);
    sweep_output_one<K, m, 4, 3>(acc, out_pages, stats);
}

// One accessor page size: input (split) and output (m), packed and padded.
template <uint32_t K, uint32_t r, typename MakeAcc>
void sweep_k_ratio(MakeAcc make_acc, uint32_t num_pages, RunStats& stats) {
    for (uint32_t pad : {0u, 32u}) {
        const auto acc = make_acc(r * K * kChunk + pad);
        SCOPED_TRACE("K=" + std::to_string(K) + " ratio=" + std::to_string(r) + " pad=" + std::to_string(pad));
        sweep_input<K, r>(acc, num_pages, stats);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        if (pad == 0) {  // the output is packed: the host enforces it for unicast
            sweep_output<K, r>(acc, num_pages, stats);
            if (::testing::Test::HasFatalFailure()) {
                return;
            }
        }
    }
}

template <typename MakeAcc>
void sweep_all(MakeAcc make_acc, uint32_t num_pages, RunStats& stats) {
    sweep_k_ratio<1, 1>(make_acc, num_pages, stats);
    sweep_k_ratio<1, 2>(make_acc, num_pages, stats);
    sweep_k_ratio<2, 1>(make_acc, num_pages, stats);
    sweep_k_ratio<2, 2>(make_acc, num_pages, stats);
    sweep_k_ratio<4, 1>(make_acc, num_pages, stats);
    sweep_k_ratio<4, 2>(make_acc, num_pages, stats);
}

template <typename DSpec>
void sweep_sharded(const char* name, RunStats& stats) {
    SCOPED_TRACE(name);
    const uint32_t volume = DSpec{}.tensor_volume();
    sweep_all([](uint32_t page_size) { return TensorAccessor<DSpec>(0, page_size); }, volume, stats);
}

}  // namespace

TEST(AllGatherWalk, RunLinearityInterleaved) {
    RunStats stats;
    for (uint32_t banks : {1u, 2u, 3u, 7u, 8u, 12u}) {
        for (uint32_t pages : {1u, 7u, 24u, 60u}) {
            SCOPED_TRACE("banks=" + std::to_string(banks) + " pages=" + std::to_string(pages));
            sweep_all([banks](uint32_t page_size) { return MockInterleavedAccessor{banks, page_size}; }, pages, stats);
            if (HasFatalFailure()) {
                return;
            }
        }
    }
    EXPECT_GT(stats.multi, 0u);
    EXPECT_GT(stats.cross, 0u);
    std::cout << "[ interleaved ] cases " << stats.cases << " runs " << stats.runs << " multi " << stats.multi
              << " cross-page " << stats.cross << std::endl;
}

TEST(AllGatherWalk, RunLinearitySharded) {
    using namespace tensor_accessor;
    RunStats stats;
    sweep_sharded<DistributionSpec<2, 3, U32<3, 5>, U32<2, 2>, U16<0, 1, 2>>>("padded 2D", stats);
    sweep_sharded<DistributionSpec<3, 3, U32<4, 3, 4>, U32<1, 3, 4>, U16<0, 1, 2>>>("rank 3", stats);
    sweep_sharded<DistributionSpec<3, 1, U32<2, 3, 4>, U32<2, 3, 4>, U16<0>>>("one shard", stats);
    sweep_sharded<DistributionSpec<2, 6, U32<4, 6>, U32<4, 1>, U16<0, 1, 2, 3, 4, 5>>>("one page wide", stats);
    sweep_sharded<DistributionSpec<4, 4, U32<4, 3, 2, 2>, U32<2, 1, 1, 1>, U16<0, 1, 2, 3>>>("rank 4", stats);
    sweep_sharded<DistributionSpec<2, 2, U32<4, 4>, U32<1, 4>, U16<0, 1>, false, false, true>>(
        "shard contiguous", stats);
    sweep_sharded<DistributionSpec<1, 2, U32<10>, U32<5>, U16<0, 1>>>("rank 1", stats);
    sweep_sharded<DistributionSpec<2, 4, U32<8, 4>, U32<2, 4>, U16<0, 1, 2, 3>>>("height sharded", stats);
    sweep_sharded<DistributionSpec<2, 4, U32<4, 6>, U32<2, 3>, U16<0, 1, 2, 3>>>("block sharded", stats);
    sweep_sharded<DistributionSpec<2, 3, U32<4, 6>, U32<4, 2>, U16<0, 1, 2>>>("width sharded", stats);
    sweep_sharded<DistributionSpec<3, 4, U32<3, 5, 3>, U32<2, 2, 2>, U16<0, 1, 2, 3>>>("padded rank 3", stats);
    sweep_sharded<DistributionSpec<2, 4, U32<5, 4>, U32<2, 1>, U16<0, 1, 2, 3>>>("padded one page wide", stats);
    EXPECT_GT(stats.multi, 0u);
    EXPECT_GT(stats.cross, 0u);
    std::cout << "[ sharded ] cases " << stats.cases << " runs " << stats.runs << " multi " << stats.multi
              << " cross-page " << stats.cross << std::endl;
}

// ---------------------------------------------------------------------------------------------------
// (e) ANTIPODE: fwd emits seqnos [0, h) of its walk (rot_f), bwd [h, count) of its own (rot_b). With
// rot_f != rot_b they partition the slice iff h is on a tile boundary or inside the unrotated tail; the
// off-boundary case is shown to break, so the rule is necessary, not just sufficient.
// ---------------------------------------------------------------------------------------------------
namespace {
template <uint32_t K>
void antipode(uint64_t& ok, uint64_t& off_bad) {
    for (uint32_t stride : {2u, 3u, 7u, 8u, 12u}) {
        for (uint32_t xfer : {1u, 2u, 3u, 7u}) {
            const uint32_t tile = stride * xfer;
            for (uint32_t pages = 1; pages < 40; ++pages) {
                const uint32_t count = pages * K;
                const uint32_t tail_start = full_tiles(count, stride, xfer, K) * tile;
                const std::pair<uint32_t, uint32_t> rots[] = {
                    {0, stride - 1}, {1 % stride, stride / 2}, {stride - 1, 0}};
                for (auto [rf, rb] : rots) {
                    const auto fw = trace<K>(0, count, 0, stride, xfer, rf);
                    for (uint32_t h = 0; h <= count; ++h) {
                        std::vector<uint32_t> got(fw.begin(), fw.begin() + h);
                        const auto bw = trace<K>(0, count, h, stride, xfer, rb);
                        got.insert(got.end(), bw.begin(), bw.end());
                        std::sort(got.begin(), got.end());
                        std::vector<uint32_t> want(count);
                        std::iota(want.begin(), want.end(), 0u);
                        const bool part = got == want;
                        if (h % tile == 0 || h >= tail_start || rf == rb) {
                            ASSERT_TRUE(part) << ctx(0, count, h, stride, xfer, K, rf) << " rot_b=" << rb;
                            ++ok;
                        } else {
                            off_bad += !part;
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

TEST(AllGatherWalk, AntipodeSplitPartitions) {
    uint64_t ok = 0, off_bad = 0;
    antipode<1>(ok, off_bad);
    antipode<2>(ok, off_bad);
    antipode<4>(ok, off_bad);
    EXPECT_GT(ok, 0u);
    EXPECT_GT(off_bad, 0u) << "the off-boundary case should break, else the rule is vacuous";
}

// ---------------------------------------------------------------------------------------------------
// (f) STRIPE_MAP: StripeMap at cps * k is k * StripeMap at cps plus the sub-chunk (rows are whole
// pages); it is a bijection onto the output; bound = min(row end, slice end).
// ---------------------------------------------------------------------------------------------------
namespace {
template <uint32_t cps, uint32_t N, uint32_t K>
void stripe_map_k() {
    SCOPED_TRACE("cps=" + std::to_string(cps) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
    constexpr uint32_t rows = 5;
    std::vector<bool> seen(rows * N * cps * K, false);
    for (uint32_t stripe = 0; stripe < N; ++stripe) {
        for (uint32_t end_pages : {rows * cps, rows * cps - 1, cps + 1}) {
            const uint32_t end = end_pages * K;
            StripeMap<cps, N> m1;
            m1.init(stripe, end_pages);
            StripeMap<cps * K, N> mk;
            mk.init(stripe, end);
            for (uint32_t p = 0; p < end_pages; ++p) {
                const auto a = m1.at(p);
                for (uint32_t s = 0; s < K; ++s) {
                    const auto b = mk.at(p * K + s);
                    ASSERT_EQ(b.global, a.global * K + s);
                    ASSERT_EQ(b.bound, a.bound * K);
                    ASSERT_EQ(b.bound % K, 0u);
                    const uint32_t row_end = (b.global / (N * cps * K)) * N * cps * K + (stripe + 1) * cps * K;
                    const uint32_t end_global = b.global + (end - (p * K + s));
                    ASSERT_EQ(b.bound, std::min(row_end, end_global));
                    if (end_pages == rows * cps) {
                        ASSERT_FALSE(seen[b.global]);
                        seen[b.global] = true;
                    }
                }
            }
        }
    }
    for (size_t g = 0; g < seen.size(); ++g) {
        ASSERT_TRUE(seen[g]) << "global " << g;
    }
}
template <uint32_t cps, uint32_t N>
void stripe_map_ks() {
    stripe_map_k<cps, N, 1>();
    stripe_map_k<cps, N, 2>();
    stripe_map_k<cps, N, 4>();
    stripe_map_k<cps, N, 8>();
}
}  // namespace

TEST(AllGatherWalk, StripeMapScalesByK) {
    stripe_map_ks<1, 2>();
    stripe_map_ks<2, 3>();
    stripe_map_ks<5, 4>();
    stripe_map_ks<8, 8>();
}

// ---------------------------------------------------------------------------------------------------
// (g) P6 SCATTER CAP: without cross-page runs (page stride != walk stride) a column of the capped xfer
// holds at most 4 runs, measured through next_run; one chunk more makes some column hold 5. The
// general form agrees with the simple one at contig_pages == 1 and is uncapped at 0.
// ---------------------------------------------------------------------------------------------------
namespace {
template <uint32_t K>
uint32_t worst_column_runs(uint32_t xfer, uint32_t stride, uint32_t pages) {
    const MockInterleavedAccessor acc{5, K * kChunk};
    const RunSource src = run_source(true, K, K, 5, stride);
    EXPECT_TRUE(src.whole_page && !src.cross_page);
    const uint32_t count = pages * K;
    auto walk = make_walk<K>(0, count, 0, stride, xfer, 0);
    uint32_t worst = 0;
    for (const auto& col : spec_columns(0, count, stride, xfer, K, 0)) {
        uint32_t segs = 0;
        for (uint32_t left = col.size(); left > 0;) {
            const uint32_t run = next_run<K, K>(walk, acc, src, walk.chunk(), count, left);
            walk.advance(run);
            left -= run;
            ++segs;
        }
        worst = std::max(worst, segs);
    }
    return worst;
}
template <uint32_t K>
void scatter_cap() {
    const uint32_t cap = xfer_cap_for_segments(K, false);
    EXPECT_EQ(cap, 4 * K);
    EXPECT_EQ(xfer_cap_for_segments(K, true), 0u);
    EXPECT_EQ(xfer_for_segments(1000, K, 1), cap);
    EXPECT_EQ(xfer_for_segments(1000, K, 0), 1000u);
    for (uint32_t stride : {2u, 3u}) {
        for (uint32_t pages : {37u, 64u}) {
            EXPECT_LE(worst_column_runs<K>(cap, stride, pages), 4u) << "K=" << K;
        }
    }
    EXPECT_GT(worst_column_runs<K>(cap + 1, 2, 64), 4u) << "K=" << K;
    // contig_pages == 2: a column may span (4 - 1) * 2 + 1 = 7 pages.
    EXPECT_EQ(column_segments(xfer_for_segments(1000, K, 2), K, 2), 4u);
}
}  // namespace

TEST(AllGatherWalk, ScatterCap) {
    scatter_cap<1>();
    scatter_cap<2>();
    scatter_cap<4>();
    scatter_cap<8>();
}

// ---------------------------------------------------------------------------------------------------
// (h) GROUPING: the CB producer and consumer group the same walk into runs independently -- the reader by
// input contiguity, the writer by output contiguity -- and only agree because the CB is packed. Emulate
// both over the same slice and check every CB entry holds the same seqnos, chunk j at j * chunk_size.
// k > 1, rot, resume (the even-ring halves), padded input pages, interleaved and sharded outputs.
//
// Mirrored kernel loops (ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/): the unicast reader's
// iteration-0 batch loop (read_input_runs on the input, in_src, the slice end) and the unicast writer's
// (emit_runs on the output, out_src, StripeMap's bound); the same loops in multicast_reader.cpp (fill_entry /
// send_entry) and multicast_writer.cpp with skip = 0. The relaying reader calls emit_runs with exactly the
// writer's arguments, so it groups like the writer by construction and is not emulated separately.
// ---------------------------------------------------------------------------------------------------
namespace {
struct Placed {
    uint32_t chunk, l1;
    bool operator==(const Placed&) const = default;
};

template <uint32_t K, uint32_t cpp, typename Accessor, typename ToAcc>
std::vector<std::vector<Placed>> emulate_side(
    const Accessor& acc,
    RunSource src,
    uint32_t first,
    uint32_t count,
    uint32_t skip,
    uint32_t take,
    uint32_t stride,
    uint32_t xfer,
    uint32_t rot,
    uint32_t cpe,
    ToAcc to_acc,
    uint64_t& runs) {
    std::vector<std::vector<Placed>> out;
    auto walk = make_walk<K>(first, count, skip, stride, xfer, rot);
    for (uint32_t done = 0; done < take;) {
        const uint32_t batch = std::min(cpe, take - done);
        std::vector<Placed> entry;
        uint32_t l1 = 0;  // cb.get_write_ptr() / cb.get_read_ptr()
        for (uint32_t left = batch; left > 0;) {
            const Target t = to_acc(walk.chunk());
            const uint32_t run = next_run<K, cpp>(walk, acc, src, t.chunk, t.bound, left);
            if (run == 0 || run > left) {
                ADD_FAILURE() << "bad run " << run << " with " << left << " left";
                return out;
            }
            auto probe = walk;
            for (uint32_t i = 0; i < run; ++i) {
                entry.push_back({probe.chunk(), l1 + i * kChunk});
                probe.advance(1);
            }
            ++runs;
            l1 += run * kChunk;
            left -= run;
            walk.advance(run);
        }
        out.push_back(std::move(entry));
        done += batch;
    }
    return out;
}

struct GroupingStats {
    uint64_t cases = 0;
    uint64_t differing = 0;  // cases the two sides grouped differently: what the packed CB rescues
};

// One device's slice of `rows` stripe rows. Input: split walk pages per input page (in_cpp = split * K).
// Output: m walk pages per output page, rows of N stripes of cps_pages walk pages. Run sources exactly as
// WalkGeometry::make derives them (slot == chunk).
template <uint32_t K, uint32_t split, uint32_t m, uint32_t cps_pages, uint32_t N, typename InAcc, typename OutAcc>
void grouping_case(const InAcc& in, const OutAcc& out, uint32_t rows, GroupingStats& stats) {
    constexpr uint32_t in_cpp = split * K, out_cpp = m * K, cps = cps_pages * K;
    const uint32_t local_pages = rows * cps_pages;  // walk pages
    if (local_pages % split != 0 || (rows * N * cps) % out_cpp != 0) {
        return;
    }
    const uint32_t stride = m > 1 ? 1u : out.contiguous_page_stride();
    const RunSource out_src =
        run_source(out.get_aligned_page_size() == out_cpp * kChunk, K, out_cpp, out.contiguous_page_stride(), stride);
    const RunSource in_src =
        run_source(in.get_aligned_page_size() == in_cpp * kChunk, K, in_cpp, in.contiguous_page_stride(), stride);
    for (uint32_t stripe : {0u, N - 1}) {
        for (uint32_t workers : {1u, 3u}) {
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t in_pages = local_pages / split;
                const uint32_t first = in_pages * w / workers * split * K;
                const uint32_t count = (in_pages * (w + 1) / workers * split) * K - first;
                const uint32_t end = first + count;
                StripeMap<cps, N> map;
                map.init(stripe, end);
                auto out_t = [&map](uint32_t l) {
                    const auto p = map.at(l);
                    return Target{p.global, p.bound};
                };
                auto in_t = [end](uint32_t l) { return Target{l, end}; };
                for (uint32_t xfer : {1u, 3u, 8u}) {
                    for (uint32_t rot : {0u, stride - 1}) {
                        const auto order = trace<K>(first, count, 0, stride, xfer, rot);
                        for (uint32_t cpe : {1u, 3u, xfer, 2 * xfer, 7u}) {
                            // (skip, take): whole stripe, even-ring prefix half, even-ring suffix half.
                            const std::pair<uint32_t, uint32_t> iters[] = {
                                {0, count}, {0, count / 2}, {count / 2, count - count / 2}};
                            for (auto [skip, take] : iters) {
                                if (count != 0 && skip >= count) {
                                    continue;
                                }
                                SCOPED_TRACE(
                                    "split=" + std::to_string(split) + " m=" + std::to_string(m) +
                                    " cps_pages=" + std::to_string(cps_pages) + " N=" + std::to_string(N) +
                                    " stripe=" + std::to_string(stripe) + " cpe=" + std::to_string(cpe) + " take=" +
                                    std::to_string(take) + " " + ctx(first, count, skip, stride, xfer, K, rot));
                                uint64_t rr = 0, rw = 0;
                                const auto r = emulate_side<K, in_cpp>(
                                    in, in_src, first, count, skip, take, stride, xfer, rot, cpe, in_t, rr);
                                const auto wr = emulate_side<K, out_cpp>(
                                    out, out_src, first, count, skip, take, stride, xfer, rot, cpe, out_t, rw);
                                if (::testing::Test::HasFailure()) {
                                    return;
                                }
                                ASSERT_EQ(r.size(), (take + cpe - 1) / cpe);
                                ASSERT_EQ(r, wr);
                                for (size_t e = 0; e < r.size(); ++e) {
                                    ASSERT_EQ(r[e].size(), std::min<uint32_t>(cpe, take - e * cpe)) << "entry " << e;
                                    for (size_t j = 0; j < r[e].size(); ++j) {
                                        ASSERT_EQ(r[e][j].l1, j * kChunk);
                                        ASSERT_EQ(r[e][j].chunk, order[skip + e * cpe + j]) << "wrong seqno";
                                    }
                                }
                                stats.differing += rr != rw;
                                ++stats.cases;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <uint32_t K, uint32_t split, uint32_t m, uint32_t cps_pages, uint32_t N>
void grouping_mocks(GroupingStats& stats) {
    for (uint32_t in_banks : {1u, 3u, 8u}) {
        for (uint32_t out_banks : {1u, 3u, 8u}) {
            for (uint32_t in_pad : {0u, 32u}) {
                for (uint32_t rows : {1u, 2u, 3u, 5u}) {
                    const MockInterleavedAccessor in{in_banks, split * K * kChunk + in_pad};
                    const MockInterleavedAccessor out{out_banks, m * K * kChunk};
                    SCOPED_TRACE(
                        "in_banks=" + std::to_string(in_banks) + " out_banks=" + std::to_string(out_banks) +
                        " in_pad=" + std::to_string(in_pad) + " rows=" + std::to_string(rows));
                    grouping_case<K, split, m, cps_pages, N>(in, out, rows, stats);
                    if (::testing::Test::HasFatalFailure()) {
                        return;
                    }
                }
            }
        }
    }
}

// Sharded output: rows follow from the tensor volume (in output pages).
template <uint32_t K, uint32_t split, uint32_t m, uint32_t cps_pages, uint32_t N, typename DSpec>
void grouping_sharded(const char* name, GroupingStats& stats) {
    const uint32_t total_pages = DSpec{}.tensor_volume() * m;  // walk pages
    if (total_pages % (N * cps_pages) != 0) {
        return;
    }
    SCOPED_TRACE(name);
    const auto out = TensorAccessor<DSpec>(0, m * K * kChunk);
    for (uint32_t in_banks : {1u, 3u, 8u}) {
        const MockInterleavedAccessor in{in_banks, split * K * kChunk};
        grouping_case<K, split, m, cps_pages, N>(in, out, total_pages / (N * cps_pages), stats);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

template <uint32_t K, uint32_t split, uint32_t m>
void grouping_all(GroupingStats& stats) {
    using namespace tensor_accessor;
    using OnePageWide = DistributionSpec<2, 6, U32<4, 6>, U32<4, 1>, U16<0, 1, 2, 3, 4, 5>>;
    using Block = DistributionSpec<2, 4, U32<4, 6>, U32<2, 3>, U16<0, 1, 2, 3>>;
    using Rank3 = DistributionSpec<3, 3, U32<4, 3, 4>, U32<1, 3, 4>, U16<0, 1, 2>>;
    SCOPED_TRACE("K=" + std::to_string(K) + " split=" + std::to_string(split) + " m=" + std::to_string(m));
    grouping_mocks<K, split, m, 4, 2>(stats);
    grouping_mocks<K, split, m, 4, 3>(stats);
    grouping_mocks<K, split, m, 8, 3>(stats);
    grouping_mocks<K, split, m, 12, 3>(stats);
    grouping_sharded<K, split, m, 4, 2, OnePageWide>("one page wide", stats);
    grouping_sharded<K, split, m, 4, 3, Block>("block", stats);
    grouping_sharded<K, split, m, 6, 2, Rank3>("rank 3", stats);
    grouping_sharded<K, split, m, 4, 3, Rank3>("rank 3", stats);
}
}  // namespace

TEST(AllGatherWalk, ReaderWriterGrouping) {
    GroupingStats stats;
    grouping_all<1, 1, 1>(stats);
    grouping_all<1, 2, 1>(stats);
    grouping_all<1, 4, 1>(stats);
    grouping_all<1, 1, 2>(stats);
    grouping_all<1, 1, 4>(stats);
    grouping_all<2, 1, 1>(stats);
    grouping_all<2, 2, 1>(stats);
    grouping_all<2, 1, 2>(stats);
    grouping_all<4, 1, 1>(stats);
    grouping_all<4, 2, 1>(stats);
    EXPECT_GT(stats.cases, 0u);
    // The sweep has to reach entries the two sides group differently, else it proves nothing.
    EXPECT_GT(stats.differing, 0u);
}

// ---------------------------------------------------------------------------------------------------
// Host rules: chunks_per_transfer (unchanged) and the k choice.
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, HostRules) {
    static_assert(chunks_per_transfer(15232, 2048, 0, host_kernel_shim::kNocMaxBurstSizeBlackhole) == 7);
    static_assert(chunks_per_transfer(15232, 2048, 0, host_kernel_shim::kNocMaxBurstSizeWormhole) == 4);
    constexpr uint32_t bh = host_kernel_shim::kNocMaxBurstSizeBlackhole;
    // fill(k): 8192 -> k=1 0.54, k=2 0.81, k=4 0.94; 4096 -> k=1 0.81, k=2 0.94; 2048 -> k=1 0.94.
    static_assert(choose_k(8192, 15232, bh, 64) == 4);
    static_assert(choose_k(4096, 15232, bh, 64) == 2);
    static_assert(choose_k(2048, 15232, bh, 64) == 1);
    static_assert(choose_k(64, 15232, bh, 64) == 1);  // cannot split below the alignment
    // 1088 B (bfp8 tile): 13 * 1088 = 14144 -> 0.93 at k = 1.
    static_assert(choose_k(1088, 15232, bh, 64) == 1);
    // A page that cannot reach 0.9 with aligned chunks falls back to the best fill.
    static_assert(choose_k(96, 15232, bh, 32) == 1);
    for (uint32_t page : {64u, 128u, 1024u, 1088u, 2048u, 4096u, 8192u, 16384u, 32768u}) {
        const uint32_t k = choose_k(page, 15232, bh, 64);
        EXPECT_EQ(page % k, 0u);
        EXPECT_EQ((page / k) % 64, 0u);
    }
}

// ---------------------------------------------------------------------------------------------------
// (j) HOST POLICY RULES (gather_rules.hpp): what WalkGeometry::make and the factories compute from the walk
// geometry. WalkGeometry::make itself needs real Buffers (contiguous_page_stride, hal), so it is not built
// here; every integer rule it applies is, from the same header the kernels include.
// ---------------------------------------------------------------------------------------------------

// choose_k on both archs (BH packet 15232 / burst 16384, WH packet 7616 / burst 8192), with the run-cap
// budget the factories pass (min(packet, run_cap)).
TEST(AllGatherWalk, ChooseK) {
    constexpr uint32_t bh = host_kernel_shim::kNocMaxBurstSizeBlackhole;
    constexpr uint32_t wh = host_kernel_shim::kNocMaxBurstSizeWormhole;
    // BH: 8 KB -> 4 (0.94), 4 KB -> 2 (0.94), 2 KB -> 1 (0.94), fp32 tile (4 KB) -> 2, bfp8 tile -> 1.
    static_assert(choose_k(8192, 15232, bh, 64) == 4);
    static_assert(choose_k(4096, 15232, bh, 64) == 2);
    static_assert(choose_k(2048, 15232, bh, 64) == 1);
    static_assert(choose_k(1088, 15232, bh, 64) == 1);
    static_assert(choose_k(64, 15232, bh, 64) == 1);  // cannot split below the alignment
    static_assert(choose_k(96, 15232, bh, 32) == 1);  // best fill when 0.9 is out of reach
    // WH: 7616 B packet. 4 KB -> k 4 (7 x 1 KB = 0.94); 2 KB -> 2 (7 x 1 KB).
    static_assert(choose_k(4096, 7616, wh, 32) == 4);
    static_assert(choose_k(2048, 7616, wh, 32) == 2);
    // A run cap of 8 KB (BH multicast, pages >= 1 KB) is the budget: whole pages already fill it.
    static_assert(choose_k(8192, 8192, bh, 64) == 1);
    static_assert(choose_k(4096, 8192, bh, 64) == 1);
    static_assert(choose_k(2048, 8192, bh, 64) == 1);
    for (uint32_t page : {64u, 128u, 1024u, 1088u, 2048u, 4096u, 8192u, 16384u, 32768u}) {
        for (uint32_t align : {16u, 32u, 64u}) {
            const uint32_t k = choose_k(page, 15232, bh, align);
            EXPECT_TRUE(k_is_valid(k, page, align)) << "page " << page << " align " << align << " k " << k;
            // k is minimal: a smaller valid k does not reach 0.9 fill.
            for (uint32_t s = 1; s < k; s *= 2) {
                const uint32_t c = page / s;
                const uint32_t cap = 15232;
                const bool fills = c <= cap ? (cap / c) * c * 10 >= cap * 9 : false;
                EXPECT_FALSE(fills && k_is_valid(s, page, align)) << "page " << page << " k " << k << " vs " << s;
            }
        }
    }
    EXPECT_FALSE(k_is_valid(3, 3072, 16));
    EXPECT_FALSE(k_is_valid(4, 128, 64));  // 32 B chunks miss a 64 B alignment
    EXPECT_TRUE(k_is_valid(2, 128, 64));
}

// xfer under the scatter cap: what XferPolicy::Segments computes from xfer_max, k and the contig pages.
TEST(AllGatherWalk, XferForSegments) {
    // No cross-page runs (contig 1): 4 pages per column at xfer % k == 0.
    EXPECT_EQ(xfer_for_segments(7, 1, 1), 4u);
    EXPECT_EQ(xfer_for_segments(3, 1, 1), 3u);
    EXPECT_EQ(xfer_for_segments(16, 4, 1), 16u);  // 4k: 4 whole pages
    EXPECT_EQ(xfer_for_segments(28, 4, 1), 16u);
    EXPECT_EQ(column_segments(16, 4, 1), 4u);
    EXPECT_EQ(column_segments(17, 4, 1), 5u);  // starts mid-page: one page more
    // Concat (whole_page, m walk pages per output page): contig m.
    EXPECT_EQ(xfer_for_segments(7, 1, 4), 7u);
    // Cross-page with a long stripe: no cap; with a 1-page lane run per row: the contig-1 cap.
    EXPECT_EQ(xfer_for_segments(7, 1, 0), 7u);
    EXPECT_EQ(xfer_for_segments(7, 1, 100), 7u);
    // Every capped xfer keeps <= 4 segments, and the next one up would not.
    for (uint32_t k : {1u, 2u, 4u, 8u}) {
        for (uint32_t contig : {1u, 2u, 3u, 5u}) {
            for (uint32_t xmax : {1u, 2u, 7u, 13u, 28u, 64u}) {
                const uint32_t x = xfer_for_segments(xmax, k, contig);
                EXPECT_GE(x, 1u);
                EXPECT_LE(x, xmax);
                EXPECT_TRUE(x == 1 || column_segments(x, k, contig) <= gather_scatter_max_chunks);
                if (x < xmax) {
                    EXPECT_GT(column_segments(x + 1, k, contig), gather_scatter_max_chunks)
                        << "k " << k << " contig " << contig << " xmax " << xmax;
                }
            }
        }
    }
}

// Rotation targets: slices spread over the lanes; the stagger moves only direction 1; rot_for_target puts
// the walk's first full-tile lane on the target, which the walk then really starts on.
TEST(AllGatherWalk, RotationRules) {
    EXPECT_EQ(rot_target(0, 4, 1, true, 1), 0u);  // stride 1: nothing to rotate
    EXPECT_EQ(rot_target(3, 4, 8, false, 0), 6u);
    EXPECT_EQ(rot_target(3, 4, 8, false, 1), 6u);  // no stagger
    EXPECT_EQ(rot_target(3, 4, 8, true, 0), 6u);
    EXPECT_EQ(rot_target(3, 4, 8, true, 1), 7u);   // + 8 / 8
    EXPECT_EQ(rot_target(0, 8, 8, true, 1), 1u);   // + max(1, 8 / 16)
    EXPECT_EQ(rot_target(7, 8, 8, true, 1), 0u);   // wraps
    EXPECT_EQ(rot_target(1, 2, 12, true, 1), 9u);  // 6 + 3
    for (uint32_t stride : {2u, 3u, 7u, 8u, 12u}) {
        for (uint32_t n : {1u, 2u, 3u, 4u, 8u, 16u}) {
            std::vector<uint32_t> targets;
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t dir : {0u, 1u}) {
                    const uint32_t t = rot_target(i, n, stride, true, dir);
                    ASSERT_LT(t, stride);
                    if (dir == 0) {
                        targets.push_back(t);
                    }
                }
            }
            // Non-decreasing and spread: n <= stride slices land on distinct lanes.
            ASSERT_TRUE(std::is_sorted(targets.begin(), targets.end()));
            if (n <= stride) {
                ASSERT_EQ(std::set<uint32_t>(targets.begin(), targets.end()).size(), n) << "stride " << stride;
            }
            for (uint32_t first_page : {0u, 1u, 5u, 1000u}) {
                for (uint32_t t = 0; t < stride; ++t) {
                    const uint32_t rot = rot_for_target(t, first_page, stride);
                    ASSERT_LT(rot, stride);
                    ASSERT_EQ((first_page + rot) % stride, t);
                    // The walk's first chunk (a full tile exists) sits on absolute lane t.
                    for (uint32_t k : {1u, 4u}) {
                        const uint32_t pages = 3 * stride;
                        const uint32_t first = first_page * k;
                        uint32_t c = 0;
                        if (k == 1) {
                            c = make_walk<1>(first, pages * k, 0, stride, 2, rot).chunk();
                        } else {
                            c = make_walk<4>(first, pages * k, 0, stride, 2, rot).chunk();
                        }
                        ASSERT_EQ((c / k) % stride, t) << "stride " << stride << " k " << k << " first " << first;
                    }
                }
            }
        }
    }
}

// The antipode rule as the unicast factory applies it: antipode_split_ok is exactly "the halves partition the
// slice" wherever the walk tests showed that (tile boundary, tail, equal rots), and antipode_rot_b only ever
// replaces rot_b where that fails -- after which the halves always partition.
namespace {
template <uint32_t K>
void antipode_rule(uint64_t& changed) {
    for (uint32_t stride : {2u, 3u, 8u}) {
        for (uint32_t xfer : {1u, 3u, 7u}) {
            for (uint32_t pages = 1; pages < 30; ++pages) {
                const uint32_t count = pages * K;
                for (uint32_t rf = 0; rf < stride; ++rf) {
                    for (uint32_t rb = 0; rb < stride; ++rb) {
                        const auto fw = trace<K>(0, count, 0, stride, xfer, rf);
                        for (uint32_t h : {count / 2, count / 3, count - count / 4}) {
                            const uint32_t rb2 = antipode_rot_b(count, h, stride, xfer, K, rf, rb);
                            changed += rb2 != rb;
                            ASSERT_TRUE(rb2 == rb || rb2 == rf);
                            if (antipode_split_ok(count, h, stride, xfer, K, rf, rb)) {
                                ASSERT_EQ(rb2, rb);
                            }
                            std::vector<uint32_t> got(fw.begin(), fw.begin() + h);
                            const auto bw = trace<K>(0, count, h, stride, xfer, rb2);
                            got.insert(got.end(), bw.begin(), bw.end());
                            std::sort(got.begin(), got.end());
                            std::vector<uint32_t> want(count);
                            std::iota(want.begin(), want.end(), 0u);
                            ASSERT_EQ(got, want) << ctx(0, count, h, stride, xfer, K, rf) << " rot_b " << rb2;
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

TEST(AllGatherWalk, AntipodeRule) {
    uint64_t changed = 0;
    antipode_rule<1>(changed);
    antipode_rule<2>(changed);
    antipode_rule<4>(changed);
    EXPECT_GT(changed, 0u) << "the rule never had to step in: the sweep proves nothing";
}

// full_tile_seqnos is the spec's full-tile prefix; walk_seek of a tile boundary is the tile anchor.
TEST(AllGatherWalk, SeekRules) {
    for (uint32_t k : {1u, 2u, 4u}) {
        for (uint32_t stride : {1u, 3u, 8u}) {
            for (uint32_t xfer : {1u, 3u, 7u}) {
                for (uint32_t pages = 0; pages < 40; ++pages) {
                    const uint32_t count = pages * k;
                    const uint32_t F = full_tiles(count, stride, xfer, k);
                    ASSERT_EQ(full_tile_seqnos(count, stride, xfer, k), F * stride * xfer);
                    for (uint32_t t = 0; t < F; ++t) {
                        const WalkPos p = walk_seek(8 * k, count, t * stride * xfer, stride, xfer, k);
                        ASSERT_EQ(p.slot, 0u);
                        ASSERT_EQ(p.i, 0u);
                        ASSERT_EQ(p.base, chunk_at(8 * k, 0, t * xfer, stride, k));
                    }
                    const WalkPos z = walk_seek(8 * k, count, 0, stride, xfer, k);
                    ASSERT_TRUE(z.base == 8 * k && z.slot == 0 && z.i == 0) << "skip 0 is walk_start";
                }
            }
        }
    }
    static_assert(walk_seek(0, 0, 0, 8, 7, 4).base == 0);
}

TEST(AllGatherWalk, RunSourceBits) {
    for (bool wp : {false, true}) {
        for (bool cp : {false, true}) {
            const RunSource s = run_source_from_bits(run_source_bits({wp, cp}));
            EXPECT_EQ(s.whole_page, wp);
            EXPECT_EQ(s.cross_page, cp);
        }
    }
    // k = 1 is step 1's rule for cross_page; whole_page no longer needs `packed` (a padded page's own bytes are
    // contiguous), which only matters for split inputs at stride 1.
    EXPECT_TRUE(run_source(true, 1, 1, 8, 8).cross_page);
    EXPECT_FALSE(run_source(false, 1, 1, 8, 8).cross_page);
    EXPECT_TRUE(run_source(false, 1, 4, 8, 1).whole_page);
    EXPECT_FALSE(run_source(true, 1, 4, 8, 8).whole_page);  // split input at stride 8: walk-page runs only
    EXPECT_TRUE(run_source(true, 4, 4, 8, 8).whole_page);   // k = page: in-page runs at any stride
}

// The kernels' run loops themselves (gather_walk.hpp emit_runs / read_input_runs), host-built: their
// live ASSERT(run_is_linear) checks every run at the slot stride. Padded slot: chunk 32 B in a 64 B output
// page; the output steps by the slot, so output runs cross pages; the input (32 B aligned pages) does not,
// so it reads one chunk per run; every CB slot holds the right chunk at j * slot.
TEST(AllGatherWalk, PaddedSlotRunLoops) {
    constexpr uint32_t chunk = 32, slot = 64, N = 3, cps = 6;
    for (uint32_t banks : {1u, 3u, 8u}) {
        const MockInterleavedAccessor in{banks, chunk};  // matched, L1 input: aligned == chunk
        const MockInterleavedAccessor out{banks, slot};  // DRAM output: 32 B content in a 64 B page
        const uint32_t stride = out.contiguous_page_stride();
        // WalkGeometry::make: out packed against the slot; in runs only where in_aligned == slot.
        const RunSource out_src = run_source(out.get_aligned_page_size() == 1 * slot, 1, 1, stride, stride);
        const RunSource in_src = run_source(in.get_aligned_page_size() == 1 * slot, 1, 1, stride, stride);
        ASSERT_TRUE(out_src.cross_page);
        ASSERT_FALSE(in_src.cross_page);
        for (uint32_t rows : {1u, 4u}) {
            const uint32_t count = rows * cps;
            for (uint32_t xfer : {1u, 5u, 16u}) {
                for (uint32_t cpe : {3u, 16u}) {
                    SCOPED_TRACE(
                        "banks " + std::to_string(banks) + " rows " + std::to_string(rows) + " xfer " +
                        std::to_string(xfer) + " cpe " + std::to_string(cpe));
                    StripeMap<cps, N> map;
                    map.init(1, count);
                    auto out_loc = [&](uint32_t page, uint32_t off) {
                        const auto m = out.get_bank_and_offset(page);
                        return pack(Loc{m.bank_id, m.bank_page_offset * slot + off});
                    };
                    auto in_loc = [&](uint32_t c) {
                        const auto m = in.get_bank_and_offset(c);
                        return pack(Loc{m.bank_id, m.bank_page_offset * chunk});
                    };
                    auto rw = make_walk<1>(0, count, 0, stride, xfer, 0);
                    auto ww = rw;
                    std::vector<uint32_t> cb_in, cb_out;  // chunk id per CB slot
                    uint64_t out_multi = 0;
                    for (uint32_t done = 0; done < count;) {
                        const uint32_t batch = std::min(cpe, count - done);
                        std::vector<uint32_t> slots_in(batch, ~0u), slots_out(batch, ~0u);
                        auto rprobe = rw;
                        const uint32_t r_end = read_input_runs<1, chunk, slot>(
                            rw, in, in_src, count, 0, batch, in_loc, [&](uint64_t, uint32_t l1, uint32_t run) {
                                ASSERT_EQ(run, 1u) << "input steps by 32 B, the CB by 64 B: one chunk per read";
                                slots_in[l1 / slot] = rprobe.chunk();
                                rprobe.advance(1);
                            });
                        auto wprobe = ww;
                        const uint32_t w_end = emit_runs<1, chunk, slot>(
                            ww,
                            map,
                            out,
                            out_src,
                            0,
                            batch,
                            out_loc,
                            [&](uint32_t l1, uint32_t, uint32_t off, uint32_t run) {
                                ASSERT_EQ(off, 0u);
                                out_multi += run > 1;
                                for (uint32_t i = 0; i < run; ++i) {
                                    slots_out[l1 / slot + i] = wprobe.chunk();
                                    wprobe.advance(1);
                                }
                            });
                        ASSERT_EQ(r_end, batch * slot);
                        ASSERT_EQ(w_end, batch * slot);
                        ASSERT_EQ(slots_in, slots_out);
                        done += batch;
                    }
                    if (xfer > 1 && banks == 1) {
                        EXPECT_GT(out_multi, 0u) << "padded slot should give multi-page output runs";
                    }
                }
            }
        }
    }
}

// The arg tables are dense (every index once) and their tails where the factories append.
TEST(AllGatherWalk, ArgTables) {
    namespace ga = gather_args;
    static_assert(ga::unicast_reader::ct::tensor_args == 13);
    static_assert(ga::unicast_reader::rt::count == 16);
    static_assert(ga::unicast_writer::ct::tensor_args == 13);
    static_assert(ga::unicast_writer::rt::num_granular_sends == 19 && ga::unicast_writer::rt::fabric_args == 22);
    static_assert(ga::multicast_reader::ct::tensor_args == 17);
    static_assert(ga::multicast_reader::rt::route_args == 10 && ga::multicast_reader::rt::fabric_args == 22);
    static_assert(ga::multicast_writer::ct::tensor_args == 14);
    static_assert(ga::multicast_writer::rt::route_args == 8 && ga::multicast_writer::rt::fabric_args == 20);
}

// Host-only: the unicast relay rule (gather_rules.hpp relay_read_ok) and the relay predicate, on the nightly
// alignment cases (BH: DRAM 64 B, L1 16 B; WH: DRAM 32 B).
TEST(AllGatherWalk, RelayRules) {
    struct Case {
        const char* name;
        uint32_t chunk, out_cpp, out_aligned_page, out_align;
        bool ok, padded;
    };
    const Case cases[] = {
        // name                               chunk cpp aligned align  ok     padded
        {"matched_32B_l1_to_dram BH", 32, 1, 64, 64, true, true},
        {"matched_32B_l1_to_l1", 32, 1, 32, 16, true, false},
        {"matched_32B_dram_to_dram BH", 64, 1, 64, 64, true, false},  // chunk = the input's aligned page
        {"split_32B_out_page BH", 32, 1, 64, 64, true, true},
        {"concat_48B_page BH", 16, 3, 64, 64, false, false},
        {"concat_48B_page WH", 16, 3, 64, 32, false, false},
        {"concat 32B L1 -> 128B DRAM BH", 32, 4, 128, 64, false, false},  // packed, offset misaligned
        {"concat 32B L1 -> 128B DRAM WH", 32, 4, 128, 32, true, false},
        {"concat 16B L1 -> 64B L1", 16, 4, 64, 16, true, false},
        {"concat 64B DRAM -> 256B DRAM BH", 64, 4, 256, 64, true, false},
        {"degenerate chunk > page", 128, 1, 64, 64, false, false},
    };
    for (const auto& c : cases) {
        EXPECT_EQ(relay_read_ok(c.chunk, c.out_cpp, c.out_aligned_page, c.out_align), c.ok) << c.name;
        EXPECT_EQ(relay_needs_padded_slot(c.chunk, c.out_cpp, c.out_aligned_page), c.padded) << c.name;
    }
    // Relaying topologies: a line from 3 devices, a ring from 4 (N / 2 stripes per direction, the first local).
    EXPECT_FALSE(unicast_relays(2, false));
    EXPECT_TRUE(unicast_relays(3, false));
    EXPECT_FALSE(unicast_relays(2, true));
    EXPECT_FALSE(unicast_relays(3, true));
    EXPECT_TRUE(unicast_relays(4, true));
    EXPECT_TRUE(unicast_relays(8, true));
}
