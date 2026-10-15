// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host proof of the all_gather page walk (ttnn/.../all_gather/device/kernels/gather_walk.hpp): the order
// TiledWalk emits, that every run the kernels form from it is one linear stretch of memory, and that the
// reader's and the writer's run groupings agree on what sits where in a CB entry.
//
// No device: the walk, StripeMap and run queries are pure integer code. Contiguity comes from a mock of
// the interleaved accessor (whose real specialization only exists under KERNEL_BUILD) and from real
// sharded TensorAccessor instances, checked against their own get_bank_and_offset().

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#define HOST_KERNEL_SHIM_KEEP_MACROS  // gather_walk.hpp and the verbatim walk copy below need them
#include "host_kernel_shim.hpp"

#include "ttnn/operations/ccl/all_gather/device/kernels/gather_walk.hpp"

namespace {

// ---------------------------------------------------------------------------------------------------
// LegacyTiledWalk: a verbatim copy of TiledWalk as of this commit, renamed only. Oracle for a later
// rewrite of the walk: WalkOracle.MatchesLegacy pins the emission order and run_limit() to it.
// ---------------------------------------------------------------------------------------------------

// Walks a chunk id range as tiles of `xfer * stride`, reading each tile column-major: `xfer` chunks
// `stride` apart, then the next lane. One column is one run. `xfer == 1` is plain ascending order.
//
// The last tile is ragged -- its columns are `hb_` or `hb_ + 1` tall -- which is the only reason this
// holds state beyond the position.
class LegacyTiledWalk {
public:
    // Walk [first, first + count) from seqno `skip`. `stride` and `xfer` are at least 1.
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
            // Lanes below he_ hold one extra chunk, so seqnos before lane r are r*hb_ + min(r, he_).
            tile_first_ = tail_first_;
            const uint32_t s = skip - full * tile_;
            const uint32_t tall = he_ * (hb_ + 1);
            if (s < tall) {
                lane_ = s / (hb_ + 1);
                k_ = s - lane_ * (hb_ + 1);
            } else {
                // hb_ > 0 here: at hb_ == 0 the tile holds only its taller lanes, so `s < tall`.
                const uint32_t rest = s - tall;
                lane_ = he_ + rest / hb_;
                k_ = rest - (lane_ - he_) * hb_;
            }
        }
        c_ = tile_first_ + lane_ + k_ * stride_;
    }

    FORCE_INLINE uint32_t chunk() const { return c_; }

    // Chunks left in this column, i.e. the longest run still allowed here.
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
    // Empty columns can only be a tail of lanes, so a lane step never has to skip one.
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && hb_ == 0) ? he_ : stride_; }

    uint32_t c_, stride_, xfer_, tile_, lane_, k_, tile_first_, tail_first_, hb_, he_;
};

// ---------------------------------------------------------------------------------------------------
// MockInterleavedAccessor: the interleaved TensorAccessor specialization is KERNEL_BUILD-only (it derives
// from InterleavedAddrGen), so mirror its contiguity API. From tt_metal/hw/inc/api/tensor/tensor_accessor.h:
//
//     // Pages round-robin across banks, so within a bank every num_banks'th page id is contiguous.
//     uint32_t contiguous_page_stride() const { return IsDram ? NUM_DRAM_BANKS : NUM_L1_BANKS; }
//
//     // See the sharded overload. end_page_id is required: an interleaved accessor has no shape.
//     uint32_t num_contiguous_pages(uint32_t page_id, uint32_t end_page_id) const {
//         ASSERT(page_id < end_page_id);
//         const uint32_t stride = contiguous_page_stride();
//         return (end_page_id - page_id + stride - 1) / stride;
//     }
//
// Placement is round-robin: page p lives in bank p % N, at page slot p / N of that bank. The mapping
// reports the slot in pages, like the sharded accessor's PageMapping::bank_page_offset, so one helper
// (chunk_loc) turns both into a byte offset: slot * aligned_page.
// ---------------------------------------------------------------------------------------------------
struct MockInterleavedAccessor {
    uint32_t num_banks;
    uint32_t aligned_page;

    struct PageMapping {
        size_t bank_id;
        size_t bank_page_offset;  // in pages; byte offset is bank_page_offset * aligned_page
    };

    uint32_t contiguous_page_stride() const { return num_banks; }
    uint32_t num_contiguous_pages(uint32_t page_id, uint32_t end_page_id) const {
        ASSERT(page_id < end_page_id);
        const uint32_t stride = contiguous_page_stride();
        return (end_page_id - page_id + stride - 1) / stride;
    }
    uint32_t get_aligned_page_size() const { return aligned_page; }
    PageMapping get_bank_and_offset(uint32_t page_id) const { return {page_id % num_banks, page_id / num_banks}; }
};

template <size_t... Dims>
using U32 = tensor_accessor::ArrayStaticWrapperU32<Dims...>;
template <size_t... Dims>
using U16 = tensor_accessor::ArrayStaticWrapperU16<Dims...>;

constexpr uint32_t kChunk = 32;  // chunk_size in bytes; any value works, the walk never looks at it

// Where a chunk sits: its page's bank, and its byte offset in that bank. A run is linear iff its i-th
// chunk sits at the first one's bank, i * chunk_size further on.
struct Loc {
    size_t bank;
    size_t byte;
    bool operator==(const Loc&) const = default;
};

template <uint32_t chunks_per_page, typename Accessor>
Loc chunk_loc(const Accessor& acc, uint32_t chunk) {
    const auto m = acc.get_bank_and_offset(page_of<chunks_per_page>(chunk));
    return {
        static_cast<size_t>(m.bank_id),
        static_cast<size_t>(m.bank_page_offset) * acc.get_aligned_page_size() +
            byte_off_of<chunks_per_page, kChunk>(chunk)};
}

uint64_t pack(Loc l) { return (static_cast<uint64_t>(l.bank) << 40) | static_cast<uint64_t>(l.byte); }

// The seqno-ordered chunks and run_limit() of a walk, stepping one chunk at a time.
struct Trace {
    std::vector<uint32_t> chunks;
    std::vector<uint32_t> limits;
};

template <typename Walk>
Trace walk_trace(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer) {
    Trace t;
    Walk w;
    w.init(first, count, skip, stride, xfer);
    for (uint32_t s = skip; s < count; ++s) {
        t.chunks.push_back(w.chunk());
        t.limits.push_back(w.run_limit());
        w.advance(1);
    }
    return t;
}

std::string ctx(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer) {
    return "first=" + std::to_string(first) + " count=" + std::to_string(count) + " skip=" + std::to_string(skip) +
           " stride=" + std::to_string(stride) + " xfer=" + std::to_string(xfer);
}

constexpr uint32_t kStrides[] = {1, 2, 3, 7, 8, 12, 13};
constexpr uint32_t kXfers[] = {1, 2, 3, 4, 7, 9};
constexpr uint32_t kFirsts[] = {0, 5, 1000};

}  // namespace

// ---------------------------------------------------------------------------------------------------
// (i) BIJECTION: the walk over [first, first + count) emits each chunk exactly once; its columns are
// `stride` apart and at most `xfer` tall; stepping a whole column at once gives the same order.
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, Bijection) {
    for (uint32_t stride : kStrides) {
        for (uint32_t xfer : kXfers) {
            for (uint32_t first : kFirsts) {
                for (uint32_t count = 0; count <= 300; ++count) {
                    SCOPED_TRACE(ctx(first, count, 0, stride, xfer));
                    const Trace t = walk_trace<TiledWalk>(first, count, 0, stride, xfer);
                    ASSERT_EQ(t.chunks.size(), count);

                    std::vector<uint32_t> sorted = t.chunks;
                    std::sort(sorted.begin(), sorted.end());
                    for (uint32_t i = 0; i < count; ++i) {
                        ASSERT_EQ(sorted[i], first + i) << "not a permutation of the range";
                    }

                    for (uint32_t s = 0; s < count; ++s) {
                        ASSERT_GE(t.limits[s], 1u) << "seqno " << s;
                        ASSERT_LE(t.limits[s], xfer) << "seqno " << s;
                        ASSERT_LE(t.limits[s], count - s) << "seqno " << s << ": column reaches past the range";
                        if (t.limits[s] > 1) {
                            ASSERT_EQ(t.chunks[s + 1], t.chunks[s] + stride) << "seqno " << s;
                            ASSERT_EQ(t.limits[s + 1], t.limits[s] - 1) << "seqno " << s;
                        }
                    }

                    // Whole-column steps (advance(n > 1)) land on the same chunks.
                    TiledWalk w;
                    w.init(first, count, 0, stride, xfer);
                    for (uint32_t s = 0; s < count;) {
                        ASSERT_EQ(w.chunk(), t.chunks[s]) << "seqno " << s;
                        const uint32_t n = w.run_limit();
                        s += n;
                        w.advance(n);
                    }
                }
            }
        }
    }
}

// The kernel ASSERT in advance() is live on host (the shim throws), so a bad step is caught.
TEST(AllGatherWalk, AdvancePreconditionAsserts) {
    TiledWalk w;
    w.init(0, 40, 0, 3, 4);
    EXPECT_THROW(w.advance(0), std::out_of_range);
    EXPECT_THROW(w.advance(w.run_limit() + 1), std::out_of_range);
    EXPECT_NO_THROW(w.advance(w.run_limit()));
}

// Oracle for a later walk rewrite: today TiledWalk *is* LegacyTiledWalk, so this pins the copy.
TEST(AllGatherWalk, MatchesLegacy) {
    for (uint32_t stride : kStrides) {
        for (uint32_t xfer : kXfers) {
            for (uint32_t first : kFirsts) {
                for (uint32_t count = 0; count <= 300; ++count) {
                    SCOPED_TRACE(ctx(first, count, 0, stride, xfer));
                    const Trace now = walk_trace<TiledWalk>(first, count, 0, stride, xfer);
                    const Trace legacy = walk_trace<LegacyTiledWalk>(first, count, 0, stride, xfer);
                    ASSERT_EQ(now.chunks, legacy.chunks);
                    ASSERT_EQ(now.limits, legacy.limits);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// (ii) SEEK: init(skip) resumes exactly where the skip-0 walk is at seqno `skip` -- same chunks and the
// same run_limit() from there on. The unicast even-ring split depends on this (final_skip).
// ---------------------------------------------------------------------------------------------------
TEST(AllGatherWalk, Seek) {
    for (uint32_t stride : {1u, 3u, 7u, 8u}) {
        for (uint32_t xfer : {1u, 2u, 3u, 7u}) {
            for (uint32_t first : {0u, 1000u}) {
                for (uint32_t count = 0; count <= 80; ++count) {
                    const Trace full = walk_trace<TiledWalk>(first, count, 0, stride, xfer);
                    for (uint32_t skip = 0; skip < count; ++skip) {
                        SCOPED_TRACE(ctx(first, count, skip, stride, xfer));
                        const Trace part = walk_trace<TiledWalk>(first, count, skip, stride, xfer);
                        ASSERT_TRUE(std::equal(part.chunks.begin(), part.chunks.end(), full.chunks.begin() + skip));
                        ASSERT_TRUE(std::equal(part.limits.begin(), part.limits.end(), full.limits.begin() + skip));
                        ASSERT_EQ(part.chunks.size(), count - skip);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// (iii) RUN_LINEARITY
// ---------------------------------------------------------------------------------------------------
namespace {

// A walk chunk in the accessor's chunk space, plus the bound a run from it must stop at.
struct Target {
    uint32_t chunk;
    uint32_t end;
};

struct RunStats {
    uint64_t runs = 0;
    uint64_t multi = 0;  // runs longer than one chunk: proves the sweep is not vacuous
};

// Drives `walk` + next_run the way the kernels' inner loops do (batch of `batch_size`, `left` counting
// down), and checks each run:
//   - 1 <= run <= min(run_limit(), left)
//   - its chunks are stride apart in accessor space and below `end` (no row/slice crossing)
//   - they are linear in memory: same bank, chunk_size apart (and run_is_linear agrees)
// and that the runs cover the walk's seqnos [skip, count) exactly once.
template <uint32_t chunks_per_page, typename Accessor, typename ToAcc>
void check_runs(
    const Accessor& acc,
    RunSource src,
    uint32_t first,
    uint32_t count,
    uint32_t skip,
    uint32_t stride,
    uint32_t xfer,
    uint32_t batch_size,
    ToAcc to_acc,
    RunStats& stats) {
    TiledWalk walk;
    walk.init(first, count, skip, stride, xfer);
    std::vector<uint32_t> seen;
    const uint32_t take = count - skip;
    for (uint32_t sent = 0; sent < take;) {
        const uint32_t batch = std::min(batch_size, take - sent);
        for (uint32_t left = batch; left > 0;) {
            const Target t = to_acc(walk.chunk());
            const uint32_t limit = walk.run_limit();
            const uint32_t run = next_run<chunks_per_page>(walk, acc, src, t.chunk, t.end, left);
            ASSERT_GE(run, 1u) << "chunk " << t.chunk;
            ASSERT_LE(run, limit) << "chunk " << t.chunk << ": run longer than the walk column";
            ASSERT_LE(run, left) << "chunk " << t.chunk << ": run spills out of the CB entry";

            const Loc base = chunk_loc<chunks_per_page>(acc, t.chunk);
            TiledWalk probe = walk;
            for (uint32_t i = 0; i < run; ++i) {
                const Target ti = to_acc(probe.chunk());
                ASSERT_EQ(ti.chunk, t.chunk + i * stride) << "run from " << t.chunk << ", i " << i;
                ASSERT_LT(ti.chunk, t.end) << "run from " << t.chunk << " crosses end, i " << i;
                ASSERT_EQ(chunk_loc<chunks_per_page>(acc, ti.chunk), (Loc{base.bank, base.byte + i * kChunk}))
                    << "run from " << t.chunk << " is not linear at i " << i;
                seen.push_back(probe.chunk());
                probe.advance(1);
            }
            // The kernels' own debug check, fed host addresses.
            auto addr = [&](uint32_t local) { return pack(chunk_loc<chunks_per_page>(acc, to_acc(local).chunk)); };
            ASSERT_TRUE(run_is_linear(walk, run, kChunk, pack(base), addr));

            ++stats.runs;
            stats.multi += run > 1;
            walk.advance(run);
            left -= run;
        }
        sent += batch;
    }

    std::vector<uint32_t> want = walk_trace<TiledWalk>(first, count, skip, stride, xfer).chunks;
    std::sort(want.begin(), want.end());
    std::sort(seen.begin(), seen.end());
    ASSERT_EQ(seen, want) << "runs do not cover the walk exactly once";
}

constexpr uint32_t kRunXfers[] = {1, 2, 3, 5, 8};
constexpr uint32_t kBatches[] = {1, 3, 16, 1u << 20};
constexpr uint32_t kWorkers[] = {1, 3};

// Reader iteration 0 (unicast_reader.cpp:151-154, multicast_reader.cpp:234-238): the walk runs over
// local chunk ids, which are also the input accessor's chunk ids; a run is bounded by the slice end.
// The walk stride is the *output's* (plan.stride), so the natural input stride and some others are tried.
template <uint32_t chunks_per_page, typename Accessor>
void sweep_input(const Accessor& acc, uint32_t total_chunks, RunStats& stats) {
    const uint32_t page_stride = acc.contiguous_page_stride();
    const bool packed = acc.get_aligned_page_size() == chunks_per_page * kChunk;
    std::vector<uint32_t> strides = {chunks_per_page > 1 ? 1u : page_stride, 1u, 2u, 3u};
    std::sort(strides.begin(), strides.end());
    strides.erase(std::unique(strides.begin(), strides.end()), strides.end());
    for (uint32_t stride : strides) {
        const RunSource src = run_source(packed, chunks_per_page, page_stride, stride);
        for (uint32_t workers : kWorkers) {
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t first = total_chunks * w / workers;
                const uint32_t count = total_chunks * (w + 1) / workers - first;
                const uint32_t end = first + count;
                auto to_acc = [end](uint32_t local) { return Target{local, end}; };
                for (uint32_t xfer : kRunXfers) {
                    for (uint32_t skip : {0u, count / 2}) {
                        if (count != 0 && skip >= count) {
                            continue;
                        }
                        for (uint32_t batch : kBatches) {
                            SCOPED_TRACE(
                                "input cpp=" + std::to_string(chunks_per_page) + " " +
                                ctx(first, count, skip, stride, xfer) + " batch=" + std::to_string(batch));
                            check_runs<chunks_per_page>(
                                acc, src, first, count, skip, stride, xfer, batch, to_acc, stats);
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

// walk_plan takes xfer_max as a template argument.
template <uint32_t chunks_per_page, typename Accessor>
WalkPlan plan_for(const Accessor& acc, uint32_t xfer_max) {
    switch (xfer_max) {
        case 1: return walk_plan<chunks_per_page, kChunk, 1>(acc);
        case 2: return walk_plan<chunks_per_page, kChunk, 2>(acc);
        case 3: return walk_plan<chunks_per_page, kChunk, 3>(acc);
        case 5: return walk_plan<chunks_per_page, kChunk, 5>(acc);
        case 8: return walk_plan<chunks_per_page, kChunk, 8>(acc);
        default: ADD_FAILURE() << "no walk_plan instance for xfer_max " << xfer_max; return {1, 1, {false, false}};
    }
}

// Writer / relaying reader (unicast_writer.cpp:366-370, unicast_reader.cpp:157-160,
// multicast_writer.cpp:205-207, multicast_reader.cpp:262-266): the walk runs over local chunk ids;
// StripeMap turns each into an output chunk id, and a run is bounded by the row's end of our stripe.
// The output must hold rows * N * cps chunks; (cps, N) pairs that do not tile it are skipped.
template <uint32_t chunks_per_page, uint32_t cps, uint32_t N, typename Accessor>
void sweep_output_one(const Accessor& acc, uint32_t total_chunks, RunStats& stats) {
    if (total_chunks % (cps * N) != 0) {
        return;
    }
    const uint32_t local_chunks = total_chunks / N;
    const bool packed = acc.get_aligned_page_size() == chunks_per_page * kChunk;
    for (uint32_t stripe : {0u, N - 1}) {
        StripeMap<cps, N> map;
        map.init(stripe);
        auto to_acc = [&map](uint32_t local) {
            const auto pos = map.at(local);
            return Target{pos.global, pos.row_end};
        };
        for (uint32_t xfer_max : kRunXfers) {
            const WalkPlan plan = plan_for<chunks_per_page>(acc, xfer_max);
            ASSERT_EQ(plan.stride, chunks_per_page > 1 ? 1u : acc.contiguous_page_stride());
            ASSERT_EQ(plan.xfer, packed ? xfer_max : 1u);
            for (uint32_t workers : kWorkers) {
                for (uint32_t w = 0; w < workers; ++w) {
                    const uint32_t first = local_chunks * w / workers;
                    const uint32_t count = local_chunks * (w + 1) / workers - first;
                    for (uint32_t skip : {0u, count / 2}) {
                        if (count != 0 && skip >= count) {
                            continue;
                        }
                        for (uint32_t batch : kBatches) {
                            SCOPED_TRACE(
                                "output cpp=" + std::to_string(chunks_per_page) + " cps=" + std::to_string(cps) +
                                " N=" + std::to_string(N) + " stripe=" + std::to_string(stripe) + " " +
                                ctx(first, count, skip, plan.stride, plan.xfer) + " batch=" + std::to_string(batch));
                            check_runs<chunks_per_page>(
                                acc, plan.out, first, count, skip, plan.stride, plan.xfer, batch, to_acc, stats);
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

template <uint32_t chunks_per_page, typename Accessor>
void sweep_output(const Accessor& acc, uint32_t total_chunks, RunStats& stats) {
    sweep_output_one<chunks_per_page, 1, 2>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 2, 2>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 3, 2>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 4, 2>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 6, 2>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 8, 2>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 1, 3>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 2, 3>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 5, 3>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 1, 4>(acc, total_chunks, stats);
    sweep_output_one<chunks_per_page, 3, 4>(acc, total_chunks, stats);
}

// Both roles, packed and padded (aligned page one NoC word past the chunks: runs must collapse to 1).
template <uint32_t chunks_per_page, typename MakeAcc>
void sweep_both(MakeAcc make_acc, uint32_t num_pages, RunStats& stats) {
    for (uint32_t pad : {0u, 32u}) {
        const auto acc = make_acc(chunks_per_page * kChunk + pad);
        SCOPED_TRACE("pad=" + std::to_string(pad));
        sweep_input<chunks_per_page>(acc, num_pages * chunks_per_page, stats);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        sweep_output<chunks_per_page>(acc, num_pages * chunks_per_page, stats);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

template <typename MakeAcc>
void sweep_all_cpp(MakeAcc make_acc, uint32_t num_pages, RunStats& stats) {
    {
        SCOPED_TRACE("chunks_per_page=1");
        sweep_both<1>(make_acc, num_pages, stats);
    }
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    {
        SCOPED_TRACE("chunks_per_page=2");
        sweep_both<2>(make_acc, num_pages, stats);
    }
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    {
        SCOPED_TRACE("chunks_per_page=4");
        sweep_both<4>(make_acc, num_pages, stats);
    }
}

template <typename DSpec>
void sweep_sharded(const char* name, RunStats& stats) {
    SCOPED_TRACE(name);
    const uint32_t volume = DSpec{}.tensor_volume();
    sweep_all_cpp([](uint32_t page_size) { return TensorAccessor<DSpec>(0, page_size); }, volume, stats);
}

}  // namespace

TEST(AllGatherWalk, RunLinearityInterleaved) {
    RunStats stats;
    for (uint32_t banks : {1u, 2u, 3u, 7u, 8u, 12u}) {
        for (uint32_t pages : {1u, 7u, 24u, 60u}) {
            SCOPED_TRACE("banks=" + std::to_string(banks) + " pages=" + std::to_string(pages));
            sweep_all_cpp(
                [banks](uint32_t page_size) { return MockInterleavedAccessor{banks, page_size}; }, pages, stats);
            if (HasFatalFailure()) {
                return;
            }
        }
    }
    EXPECT_GT(stats.multi, 0u);
}

// Shapes reused from test_tensor_accessor.cpp, plus explicit height/block/width-sharded layouts.
TEST(AllGatherWalk, RunLinearitySharded) {
    using namespace tensor_accessor;
    RunStats stats;
    // padded on both dims
    sweep_sharded<DistributionSpec<2, 3, U32<3, 5>, U32<2, 2>, U16<0, 1, 2>>>("padded 2D", stats);
    // ND: shard spans dims 1 and 2 whole
    sweep_sharded<DistributionSpec<3, 3, U32<4, 3, 4>, U32<1, 3, 4>, U16<0, 1, 2>>>("rank 3", stats);
    // whole tensor is one shard
    sweep_sharded<DistributionSpec<3, 1, U32<2, 3, 4>, U32<2, 3, 4>, U16<0>>>("one shard", stats);
    // one page wide (row-major width/block sharding): page stride 6
    sweep_sharded<DistributionSpec<2, 6, U32<4, 6>, U32<4, 1>, U16<0, 1, 2, 3, 4, 5>>>("one page wide", stats);
    // rank 4, one-page-wide trailing dims: page stride 12
    sweep_sharded<DistributionSpec<4, 4, U32<4, 3, 2, 2>, U32<2, 1, 1, 1>, U16<0, 1, 2, 3>>>("rank 4", stats);
    // shard-contiguous placement: runs stop at the shard edge although memory carries on
    sweep_sharded<DistributionSpec<2, 2, U32<4, 4>, U32<1, 4>, U16<0, 1>, false, false, true>>(
        "shard contiguous", stats);
    // rank 1
    sweep_sharded<DistributionSpec<1, 2, U32<10>, U32<5>, U16<0, 1>>>("rank 1", stats);
    // height sharded: shard spans the full width
    sweep_sharded<DistributionSpec<2, 4, U32<8, 4>, U32<2, 4>, U16<0, 1, 2, 3>>>("height sharded", stats);
    // block sharded
    sweep_sharded<DistributionSpec<2, 4, U32<4, 6>, U32<2, 3>, U16<0, 1, 2, 3>>>("block sharded", stats);
    // width sharded, two pages wide
    sweep_sharded<DistributionSpec<2, 3, U32<4, 6>, U32<4, 2>, U16<0, 1, 2>>>("width sharded", stats);
    // padded ND, more shards than banks
    sweep_sharded<DistributionSpec<3, 4, U32<3, 5, 3>, U32<2, 2, 2>, U16<0, 1, 2, 3>>>("padded rank 3", stats);
    // padded one-page-wide: page stride 4, ragged last shard row
    sweep_sharded<DistributionSpec<2, 4, U32<5, 4>, U32<2, 1>, U16<0, 1, 2, 3>>>("padded one page wide", stats);
    EXPECT_GT(stats.multi, 0u);
}

// ---------------------------------------------------------------------------------------------------
// (iv) STRIPE_MAP: over every stripe, StripeMap::at is a bijection from local chunk ids onto the output's
// chunks, keeps rows, and row_end is the end of our stripe in that row.
// ---------------------------------------------------------------------------------------------------
namespace {

template <uint32_t cps, uint32_t N>
void check_stripe_map() {
    SCOPED_TRACE("cps=" + std::to_string(cps) + " N=" + std::to_string(N));
    constexpr uint32_t rows = 6;
    std::vector<bool> seen(rows * N * cps, false);
    for (uint32_t stripe = 0; stripe < N; ++stripe) {
        StripeMap<cps, N> map;
        map.init(stripe);
        for (uint32_t local = 0; local < rows * cps; ++local) {
            const auto pos = map.at(local);
            ASSERT_LT(pos.global, seen.size()) << "stripe " << stripe << " local " << local;
            ASSERT_FALSE(seen[pos.global]) << "stripe " << stripe << " local " << local << " collides";
            seen[pos.global] = true;
            ASSERT_LT(pos.global, pos.row_end) << "stripe " << stripe << " local " << local;
            ASSERT_LE(pos.row_end - pos.global, cps) << "stripe " << stripe << " local " << local;
            ASSERT_EQ(pos.row_end - pos.global, cps - local % cps) << "stripe " << stripe << " local " << local;
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
// (v) GROUPING: the CB producer and consumer group the same walk into runs independently -- the reader by
// input contiguity, the writer by output contiguity -- and only agree because the CB is packed. Emulate
// both over the same slice and check every CB entry holds the same seqnos, chunk j at j * chunk_size.
//
// Mirrored kernel lines (ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/):
//   reader, unicast_reader.cpp:126-171 -- per-iteration skip/take (128-130), map.init/walk.init (136-137),
//       batch loop (139-140), l1_write_addr from the entry base (146), run loop (147-167) with the
//       from_input branch (151-154: next_run<split_factor> on the input accessor, in_src, input_end_chunk);
//       in_src from run_source(...) at 75-80.
//   writer, unicast_writer.cpp:346-397 -- skip/take (351-352), map.init/walk.init (357-358), batch loop
//       (361-362), l1_read_addr from the entry base (364), run loop (365-383: next_run<output_chunks_per_page>
//       on the output accessor, plan.out, pos.global, pos.row_end).
//   The same loops in multicast_reader.cpp:231-248 (fill_entry), 251-283 (send_entry) and
//   multicast_writer.cpp:199-230, with skip = 0, take = slice_chunks. The relaying reader
//   (unicast_reader.cpp:155-162) calls next_run with exactly the writer's arguments, so it groups like the
//   writer by construction and is not emulated separately.
// ---------------------------------------------------------------------------------------------------
namespace {

struct Placed {
    uint32_t chunk;  // local chunk id
    uint32_t l1;     // byte offset from the CB entry's base
    bool operator==(const Placed&) const = default;
};

struct SideTrace {
    std::vector<std::vector<Placed>> entries;
    std::vector<std::vector<uint32_t>> runs;  // run lengths per entry
};

template <uint32_t chunks_per_page, typename Accessor, typename ToAcc>
SideTrace emulate_side(
    const Accessor& acc,
    RunSource src,
    uint32_t first,
    uint32_t count,
    uint32_t skip,
    uint32_t take,
    uint32_t stride,
    uint32_t xfer,
    uint32_t chunks_per_cb_entry,
    ToAcc to_acc) {
    SideTrace out;
    TiledWalk walk;
    walk.init(first, count, skip, stride, xfer);
    for (uint32_t done = 0; done < take;) {
        const uint32_t batch = std::min(chunks_per_cb_entry, take - done);
        std::vector<Placed> entry;
        std::vector<uint32_t> runs;
        uint32_t l1 = 0;  // cb.get_write_ptr() / cb.get_read_ptr()
        for (uint32_t left = batch; left > 0;) {
            const Target t = to_acc(walk.chunk());
            const uint32_t run = next_run<chunks_per_page>(walk, acc, src, t.chunk, t.end, left);
            if (run == 0 || run > left) {
                ADD_FAILURE() << "bad run " << run << " with " << left << " left";
                return out;
            }
            TiledWalk probe = walk;
            for (uint32_t i = 0; i < run; ++i) {
                entry.push_back({probe.chunk(), l1 + i * kChunk});
                probe.advance(1);
            }
            runs.push_back(run);
            l1 += run * kChunk;
            left -= run;
            walk.advance(run);
        }
        out.entries.push_back(std::move(entry));
        out.runs.push_back(std::move(runs));
        done += batch;
    }
    return out;
}

struct GroupingStats {
    uint64_t cases = 0;
    uint64_t entries = 0;
    uint64_t differing = 0;  // entries the two sides grouped differently: the case the packed CB rescues
};

// One device's slice. Input: split chunks per input page. Output: out_cpp chunks per output page, laid
// out as rows of N stripes of cps chunks. Exactly one of split / out_cpp exceeds 1 (chunk = min of pages).
template <uint32_t split, uint32_t out_cpp, uint32_t cps, uint32_t N, typename InAcc, typename OutAcc>
void grouping_case(const InAcc& in, const OutAcc& out, uint32_t rows, GroupingStats& stats) {
    const uint32_t local_chunks = rows * cps;
    if (local_chunks % split != 0 || (rows * N * cps) % out_cpp != 0) {
        return;
    }
    // unicast_reader.cpp:75-79
    const auto mk_in_src = [&](uint32_t stride) {
        return run_source(in.get_aligned_page_size() == split * kChunk, split, in.contiguous_page_stride(), stride);
    };
    for (uint32_t stripe : {0u, N - 1}) {
        StripeMap<cps, N> map;
        map.init(stripe);
        auto out_target = [&map](uint32_t local) {
            const auto pos = map.at(local);
            return Target{pos.global, pos.row_end};
        };
        for (uint32_t xfer_max : {1u, 3u, 8u}) {
            const WalkPlan plan = plan_for<out_cpp>(out, xfer_max);
            const RunSource in_src = mk_in_src(plan.stride);
            for (uint32_t cpe : {1u, 3u, xfer_max, 2 * xfer_max, 7u}) {
                for (uint32_t workers : {1u, 3u}) {
                    for (uint32_t w = 0; w < workers; ++w) {
                        const uint32_t first = local_chunks * w / workers;
                        const uint32_t count = local_chunks * (w + 1) / workers - first;
                        const uint32_t input_end_chunk = first + count;
                        auto in_target = [input_end_chunk](uint32_t local) { return Target{local, input_end_chunk}; };
                        // (skip, take): whole stripe, even-ring prefix half, even-ring suffix half.
                        const std::pair<uint32_t, uint32_t> iters[] = {
                            {0, count}, {0, count / 2}, {count / 2, count - count / 2}};
                        const Trace order = walk_trace<TiledWalk>(first, count, 0, plan.stride, plan.xfer);
                        for (auto [skip, take] : iters) {
                            if (count != 0 && skip >= count) {
                                continue;
                            }
                            SCOPED_TRACE(
                                "split=" + std::to_string(split) + " out_cpp=" + std::to_string(out_cpp) + " cps=" +
                                std::to_string(cps) + " N=" + std::to_string(N) + " stripe=" + std::to_string(stripe) +
                                " cpe=" + std::to_string(cpe) + " take=" + std::to_string(take) + " " +
                                ctx(first, count, skip, plan.stride, plan.xfer));
                            const SideTrace reader = emulate_side<split>(
                                in, in_src, first, count, skip, take, plan.stride, plan.xfer, cpe, in_target);
                            const SideTrace writer = emulate_side<out_cpp>(
                                out, plan.out, first, count, skip, take, plan.stride, plan.xfer, cpe, out_target);
                            if (::testing::Test::HasFailure()) {
                                return;
                            }
                            const size_t want_entries = (take + cpe - 1) / cpe;
                            ASSERT_EQ(reader.entries.size(), want_entries);
                            ASSERT_EQ(writer.entries.size(), want_entries);
                            for (size_t e = 0; e < want_entries; ++e) {
                                ASSERT_EQ(reader.entries[e], writer.entries[e]) << "entry " << e;
                                const auto& entry = reader.entries[e];
                                ASSERT_EQ(entry.size(), std::min<uint32_t>(cpe, take - e * cpe)) << "entry " << e;
                                for (uint32_t j = 0; j < entry.size(); ++j) {
                                    ASSERT_EQ(entry[j].l1, j * kChunk) << "entry " << e << " chunk " << j;
                                    ASSERT_EQ(entry[j].chunk, order.chunks[skip + e * cpe + j])
                                        << "entry " << e << " chunk " << j << ": wrong seqno";
                                }
                                stats.differing += reader.runs[e] != writer.runs[e];
                            }
                            ++stats.cases;
                            stats.entries += want_entries;
                        }
                    }
                }
            }
        }
    }
}

template <uint32_t split, uint32_t out_cpp, uint32_t cps, uint32_t N>
void grouping_mocks(GroupingStats& stats) {
    for (uint32_t in_banks : {1u, 3u, 8u}) {
        for (uint32_t out_banks : {1u, 3u, 8u}) {
            for (uint32_t in_pad : {0u, 32u}) {
                for (uint32_t rows : {1u, 2u, 3u, 5u}) {
                    const MockInterleavedAccessor in{in_banks, split * kChunk + in_pad};
                    const MockInterleavedAccessor out{out_banks, out_cpp * kChunk};
                    SCOPED_TRACE(
                        "in_banks=" + std::to_string(in_banks) + " out_banks=" + std::to_string(out_banks) +
                        " in_pad=" + std::to_string(in_pad) + " rows=" + std::to_string(rows));
                    grouping_case<split, out_cpp, cps, N>(in, out, rows, stats);
                    if (::testing::Test::HasFatalFailure()) {
                        return;
                    }
                }
            }
        }
    }
}

// Sharded output: rows follow from the tensor volume.
template <uint32_t split, uint32_t out_cpp, uint32_t cps, uint32_t N, typename DSpec>
void grouping_sharded(const char* name, GroupingStats& stats) {
    const uint32_t total = DSpec{}.tensor_volume() * out_cpp;
    if (total % (N * cps) != 0) {
        return;
    }
    SCOPED_TRACE(name);
    const auto out = TensorAccessor<DSpec>(0, out_cpp * kChunk);
    for (uint32_t in_banks : {1u, 3u, 8u}) {
        const MockInterleavedAccessor in{in_banks, split * kChunk};
        grouping_case<split, out_cpp, cps, N>(in, out, total / (N * cps), stats);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

template <uint32_t split, uint32_t out_cpp>
void grouping_all(GroupingStats& stats) {
    using namespace tensor_accessor;
    using OnePageWide = DistributionSpec<2, 6, U32<4, 6>, U32<4, 1>, U16<0, 1, 2, 3, 4, 5>>;
    using Block = DistributionSpec<2, 4, U32<4, 6>, U32<2, 3>, U16<0, 1, 2, 3>>;
    using Rank3 = DistributionSpec<3, 3, U32<4, 3, 4>, U32<1, 3, 4>, U16<0, 1, 2>>;
    SCOPED_TRACE("split=" + std::to_string(split) + " out_cpp=" + std::to_string(out_cpp));
    grouping_mocks<split, out_cpp, 4, 2>(stats);
    grouping_mocks<split, out_cpp, 4, 3>(stats);
    grouping_mocks<split, out_cpp, 8, 2>(stats);
    grouping_mocks<split, out_cpp, 12, 3>(stats);
    grouping_sharded<split, out_cpp, 4, 2, OnePageWide>("one page wide", stats);
    grouping_sharded<split, out_cpp, 4, 3, Block>("block", stats);
    grouping_sharded<split, out_cpp, 6, 2, Rank3>("rank 3", stats);
    grouping_sharded<split, out_cpp, 4, 3, Rank3>("rank 3", stats);
}

}  // namespace

TEST(AllGatherWalk, ReaderWriterGrouping) {
    GroupingStats stats;
    grouping_all<1, 1>(stats);
    grouping_all<2, 1>(stats);
    grouping_all<4, 1>(stats);
    grouping_all<1, 2>(stats);
    grouping_all<1, 4>(stats);
    EXPECT_GT(stats.cases, 0u);
    // The sweep has to reach entries the two sides group differently, else it proves nothing.
    EXPECT_GT(stats.differing, 0u);
}

// ---------------------------------------------------------------------------------------------------
// chunks_per_transfer: min(packet, burst, cap) / chunk, at least 1, for both arch bursts. The 3-argument
// form the kernels call uses NOC_MAX_BURST_SIZE (the shim's, on host).
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
                EXPECT_EQ(
                    chunks_per_transfer(packet, chunk), chunks_per_transfer(packet, chunk, 0, NOC_MAX_BURST_SIZE));
            }
        }
    }
    static_assert(chunks_per_transfer(15232, 2048, 0, host_kernel_shim::kNocMaxBurstSizeBlackhole) == 7);
    static_assert(chunks_per_transfer(15232, 2048, 0, host_kernel_shim::kNocMaxBurstSizeWormhole) == 4);
}
