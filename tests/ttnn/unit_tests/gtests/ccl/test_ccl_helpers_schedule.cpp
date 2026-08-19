// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host equivalence sweep for ttnn::ccl::schedule — the bidirectional ring reduce-scatter schedule
// shared by reduce_scatter_minimal_async's ring reader / compute / writer kernels.
//
// GOLDEN: the PRE-MIGRATION loops of those three kernels, transcribed verbatim from their last
// unmigrated revision (bfcc8253931) — control flow and tile-id arithmetic only, with the CB/NoC/
// fabric calls stripped. Both the reader's and the writer's hand-maintained step state machines
// are transcribed SEPARATELY and asserted to agree on their shared fields, turning the
// "byte-identical copies" property the kernels used to maintain by hand into an executable check.
//
// CANDIDATE: RingRsSchedule + RingSliceCursor + SliceRowWalker / SequentialTileWalker, driven
// exactly as the migrated kernels drive them.
//
// COMPARED: the full per-chunk trace — batch/step/slice/channel indices, chunk parity, tile
// count, skip / reduce / write flags, and EVERY tile id emitted by the input/interm/output
// walkers on active chunks. This sweep (in its original, uncommitted form) is what caught the
// hoisted-RingSliceCursor bug: constructed outside the batch loop, batch N+1 silently continues
// on the wrong slice with no hang and no CB mismatch — invisible to a B=1 device probe.
//
// The default grid keeps unit-test runtime small; set TT_CCL_SCHEDULE_SWEEP_FULL=1 for the full
// grid (~835k configs / ~280M chunk records, tens of seconds).

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace {

namespace sched = ttnn::ccl::schedule;

struct ChunkRec {
    uint32_t b, step, slice_idx, c;
    bool is_even;
    uint32_t tiles;
    bool skip;
    bool reduce_interm, reduce_output;
    bool write_to_remote, write_to_interm, separate_sems;
    bool even_chunks, odd_chunks;
    std::vector<uint32_t> ids;  // interleaved (input, interm, output) per tile, active chunks only

    bool operator==(const ChunkRec& o) const {
        return b == o.b && step == o.step && slice_idx == o.slice_idx && c == o.c && is_even == o.is_even &&
               tiles == o.tiles && skip == o.skip && reduce_interm == o.reduce_interm &&
               reduce_output == o.reduce_output && write_to_remote == o.write_to_remote &&
               write_to_interm == o.write_to_interm && separate_sems == o.separate_sems &&
               even_chunks == o.even_chunks && odd_chunks == o.odd_chunks && ids == o.ids;
    }
};

struct Cfg {
    uint32_t ring, chip;
    bool dir;
    uint32_t dim, B, C, gran;
    uint32_t start, end;  // start_tiles_read, start_tiles_to_read
    uint32_t slice_Wt, tensor_Wt, slice_Ht;
    uint32_t in_batch, out_batch, in_chan, out_chan;
    uint32_t start_pages, start_row;
};

std::string cfg_str(const Cfg& g) {
    std::ostringstream os;
    os << "ring=" << g.ring << " chip=" << g.chip << " dir=" << g.dir << " dim=" << g.dim << " B=" << g.B
       << " C=" << g.C << " gran=" << g.gran << " start=" << g.start << " end=" << g.end;
    return os.str();
}

std::string rec_str(const ChunkRec& r) {
    std::ostringstream os;
    os << "b=" << r.b << " step=" << r.step << " slice=" << r.slice_idx << " c=" << r.c << " even=" << r.is_even
       << " tiles=" << r.tiles << " skip=" << r.skip << " ri=" << r.reduce_interm << " ro=" << r.reduce_output
       << " wr=" << r.write_to_remote << " wi=" << r.write_to_interm << " sep=" << r.separate_sems
       << " ec=" << r.even_chunks << " oc=" << r.odd_chunks << " ids=[";
    for (auto v : r.ids) {
        os << v << ",";
    }
    os << "]";
    return os.str();
}

// -----------------------------------------------------------------------------------------
// GOLDEN — transcription of the pre-migration kernels (bfcc8253931). Line references are to
// ring_reduce_scatter_minimal_async_reader.cpp / _writer.cpp at that revision.
// -----------------------------------------------------------------------------------------
void golden(const Cfg& g, std::vector<ChunkRec>& out, uint64_t& state_machine_disagreements) {
    out.clear();
    const uint32_t ring_size_by_2 = g.ring / 2;
    for (uint32_t b = 0; b < g.B; ++b) {
        // reader.cpp:94 — slice_idx is (re)seeded INSIDE the batch loop.
        int slice_idx = static_cast<int>(g.chip + ring_size_by_2);
        const uint32_t num_iters = ring_size_by_2 + 1;
        for (uint32_t i = 0; i < num_iters; ++i) {
            // READER state machine (reader.cpp:98-124; ring_reduction.cpp:34-60 was byte-identical).
            bool r_even, r_odd, r_re, r_ro, r_out;
            if (i == 0) {
                r_even = g.dir;
                r_odd = !g.dir;
                r_re = false;
                r_ro = false;
                r_out = false;
            } else if (i == ring_size_by_2) {
                r_even = g.dir;
                r_odd = !g.dir;
                r_re = r_even;
                r_ro = r_odd;
                r_out = true;
            } else if (i == 1) {
                r_even = true;
                r_odd = true;
                r_re = g.dir;
                r_ro = !g.dir;
                r_out = false;
            } else {
                r_even = true;
                r_odd = true;
                r_re = r_even;
                r_ro = r_odd;
                r_out = false;
            }

            // WRITER state machine (writer.cpp:232-267) — note the combined
            // `i == 1 || i == ring_size_by_2 - 1` branch whose two conditions coincide at ring 4.
            bool w_even, w_odd, w_re, w_ro, w_remote, w_interm, w_sep;
            if (i == 0) {
                w_even = g.dir;
                w_odd = !g.dir;
                w_re = false;
                w_ro = false;
                w_remote = true;
                w_interm = true;
                w_sep = false;
            } else if (i == ring_size_by_2) {
                w_even = g.dir;
                w_odd = !g.dir;
                w_re = w_even;
                w_ro = w_odd;
                w_remote = false;
                w_interm = false;
                w_sep = false;
            } else if (i == 1 || i == ring_size_by_2 - 1) {
                w_even = true;
                w_odd = true;
                w_re = (i == 1) ? g.dir : w_even;
                w_ro = (i == 1) ? !g.dir : w_odd;
                w_remote = true;
                w_interm = (i == ring_size_by_2 - 1) ? g.dir : true;
                w_sep = (i == ring_size_by_2 - 1);
            } else {
                w_even = true;
                w_odd = true;
                w_re = w_even;
                w_ro = w_odd;
                w_remote = true;
                w_interm = true;
                w_sep = false;
            }

            // The kernels carried these as separate hand-maintained copies; assert agreement.
            if (r_even != w_even || r_odd != w_odd || r_re != w_re || r_ro != w_ro) {
                ++state_machine_disagreements;
            }

            // reader.cpp:127-131 — wrap.
            if (slice_idx < 0) {
                slice_idx += static_cast<int>(g.ring);
            } else if (slice_idx >= static_cast<int>(g.ring)) {
                slice_idx = static_cast<int>(static_cast<uint32_t>(slice_idx) - g.ring);
            }

            // reader.cpp:134-146 — walker bases. Input gets the batch offset; interm does not.
            uint32_t base = 0;
            if (g.dim == 3) {
                base = static_cast<uint32_t>(slice_idx) * g.slice_Wt;
            } else if (g.dim == 2) {
                base = static_cast<uint32_t>(slice_idx) * g.slice_Ht * g.slice_Wt;
            } else {  // dim == 1
                base = static_cast<uint32_t>(slice_idx) * g.C * g.slice_Ht * g.slice_Wt;
            }
            uint32_t input_tile_id_start = base + g.in_batch * b;
            uint32_t interm_tile_id_start = base;

            uint32_t input_pages_read_in_row = g.start_pages, interm_pages_read_in_row = g.start_pages;
            uint32_t input_row_offset = g.start_row, interm_row_offset = g.start_row;
            auto get_next_input_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                ++input_pages_read_in_row;
                if (input_pages_read_in_row == g.slice_Wt) {
                    input_row_offset += g.tensor_Wt;
                    input_pages_read_in_row -= g.slice_Wt;
                }
                return tile_id;
            };
            auto get_next_interm_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = interm_tile_id_start + interm_row_offset + interm_pages_read_in_row;
                ++interm_pages_read_in_row;
                if (interm_pages_read_in_row == g.slice_Wt) {
                    interm_row_offset += g.tensor_Wt;
                    interm_pages_read_in_row -= g.slice_Wt;
                }
                return tile_id;
            };

            uint32_t output_tile_id_start = b * g.out_batch;
            uint32_t output_tiles_read = g.start;
            auto get_next_output_tile_id = [&]() -> uint32_t { return output_tile_id_start + (output_tiles_read++); };

            for (uint32_t c = 0; c < g.C; ++c) {
                // reader.cpp:176-181 — per-channel resets.
                input_pages_read_in_row = interm_pages_read_in_row = g.start_pages;
                input_row_offset = interm_row_offset = g.start_row;
                output_tiles_read = g.start;
                uint32_t tiles_read = g.start;
                const uint32_t total_tiles_to_read = g.end;

                bool is_even_chunk = true;
                while (tiles_read < total_tiles_to_read) {
                    const uint32_t tiles_remaining = total_tiles_to_read - tiles_read;
                    const uint32_t tiles_to_read =
                        is_even_chunk ? std::min(tiles_remaining / 2, g.gran) : std::min(tiles_remaining, g.gran);

                    ChunkRec rec{};
                    rec.b = b;
                    rec.step = i;
                    rec.slice_idx = static_cast<uint32_t>(slice_idx);
                    rec.c = c;
                    rec.is_even = is_even_chunk;
                    rec.tiles = tiles_to_read;
                    rec.write_to_remote = w_remote;
                    rec.write_to_interm = w_interm;
                    rec.separate_sems = w_sep;
                    rec.even_chunks = w_even;
                    rec.odd_chunks = w_odd;
                    rec.reduce_output = r_out;

                    if ((is_even_chunk && !w_even) || (!is_even_chunk && !w_odd) || tiles_to_read == 0) {
                        rec.skip = true;
                        rec.reduce_interm = false;
                        tiles_read += tiles_to_read;
                        for (uint32_t k = 0; k < tiles_to_read; ++k) {
                            get_next_input_tile_id();
                            get_next_interm_tile_id();
                            get_next_output_tile_id();
                        }
                    } else {
                        rec.skip = false;
                        rec.reduce_interm = (is_even_chunk && w_re) || (!is_even_chunk && w_ro);
                        for (uint32_t j = 0; j < tiles_to_read; ++j) {
                            rec.ids.push_back(get_next_input_tile_id());
                            rec.ids.push_back(get_next_interm_tile_id());
                            rec.ids.push_back(get_next_output_tile_id());
                        }
                        tiles_read += tiles_to_read;
                    }
                    out.push_back(std::move(rec));
                    is_even_chunk = !is_even_chunk;
                }

                // reader.cpp:278-280 — per-channel base accumulation.
                input_tile_id_start += g.in_chan;
                interm_tile_id_start += g.in_chan;
                output_tile_id_start += g.out_chan;
            }

            slice_idx = g.dir ? (slice_idx - 1) : (slice_idx + 1);
        }
    }
}

// -----------------------------------------------------------------------------------------
// CANDIDATE — the schedule helper, driven exactly as the migrated kernels drive it.
// -----------------------------------------------------------------------------------------
void candidate(const Cfg& g, std::vector<ChunkRec>& out) {
    out.clear();
    sched::RingRsSchedule schedule(g.ring, g.B, g.C, g.gran, g.start, g.end, g.dir);
    sched::SliceRowWalker input_walker(g.slice_Wt, g.tensor_Wt);
    sched::SliceRowWalker interm_walker(g.slice_Wt, g.tensor_Wt);
    sched::SequentialTileWalker output_walker;

    while (schedule.next_batch()) {
        const uint32_t b = schedule.batch_idx();
        sched::RingSliceCursor slice_cursor(g.chip, g.ring, g.dir);
        while (schedule.next_step()) {
            const auto& f = schedule.flags();
            const uint32_t slice_idx = slice_cursor.wrap();
            const uint32_t slice_offset = sched::slice_tile_offset(g.dim, slice_idx, g.C, g.slice_Ht, g.slice_Wt);
            input_walker.set_base(slice_offset + g.in_batch * b);
            interm_walker.set_base(slice_offset);
            output_walker.set_base(b * g.out_batch);

            while (schedule.next_channel()) {
                input_walker.reset_offsets(g.start_pages, g.start_row);
                interm_walker.reset_offsets(g.start_pages, g.start_row);
                output_walker.reset_offsets(g.start);

                while (schedule.next_chunk()) {
                    const uint32_t tiles = schedule.tiles_this_chunk();

                    ChunkRec rec{};
                    rec.b = b;
                    rec.step = schedule.step_idx();
                    rec.slice_idx = slice_idx;
                    rec.c = schedule.channel_idx();
                    rec.is_even = schedule.is_even_chunk();
                    rec.tiles = tiles;
                    rec.write_to_remote = f.write_to_remote;
                    rec.write_to_interm = f.write_to_interm;
                    rec.separate_sems = f.separate_even_odd_sems;
                    rec.even_chunks = f.even_chunks;
                    rec.odd_chunks = f.odd_chunks;
                    rec.reduce_output = f.reduce_output;

                    if (schedule.skip()) {
                        rec.skip = true;
                        rec.reduce_interm = false;
                        input_walker.advance(tiles);
                        interm_walker.advance(tiles);
                        output_walker.advance(tiles);
                    } else {
                        rec.skip = false;
                        rec.reduce_interm = schedule.reduce_interm();
                        for (uint32_t j = 0; j < tiles; ++j) {
                            rec.ids.push_back(input_walker.next());
                            rec.ids.push_back(interm_walker.next());
                            rec.ids.push_back(output_walker.next());
                        }
                    }
                    out.push_back(std::move(rec));
                }

                input_walker.bump_base(g.in_chan);
                interm_walker.bump_base(g.in_chan);
                output_walker.bump_base(g.out_chan);
            }

            slice_cursor.advance();
        }
    }
}

void run_sweep(const std::vector<uint32_t>& rings, const std::vector<uint32_t>& grans, uint32_t max_total) {
    std::vector<ChunkRec> gold, cand;
    uint64_t configs = 0, records = 0, state_machine_disagreements = 0;

    const uint32_t dims[] = {1, 2, 3};
    // (start offset, start_pages_read_in_row, start_row_offset as a multiple of tensor_Wt)
    const uint32_t starts[][3] = {{0, 0, 0}, {5, 1, 1}};
    // (batches, channels) — B=2 is what catches per-batch-cursor bugs; a B=1 grid cannot.
    const uint32_t BC[][2] = {{1, 1}, {2, 3}};

    for (uint32_t ring : rings) {
        for (uint32_t chip = 0; chip < ring; ++chip) {
            for (int dir = 0; dir <= 1; ++dir) {
                for (uint32_t dim : dims) {
                    for (uint32_t gran : grans) {
                        for (const auto& bc : BC) {
                            for (const auto& st : starts) {
                                for (uint32_t total = 0; total < max_total; ++total) {
                                    Cfg g{};
                                    g.ring = ring;
                                    g.chip = chip;
                                    g.dir = dir;
                                    g.dim = dim;
                                    g.B = bc[0];
                                    g.C = bc[1];
                                    g.gran = gran;
                                    g.start = st[0];
                                    g.end = st[0] + total;
                                    g.slice_Wt = 3;
                                    g.tensor_Wt = g.slice_Wt * ring;
                                    g.slice_Ht = 2;
                                    g.in_batch = g.C * g.slice_Ht * g.tensor_Wt;
                                    g.out_batch = g.C * g.slice_Ht * g.slice_Wt;
                                    g.in_chan = g.slice_Ht * g.tensor_Wt;
                                    g.out_chan = g.slice_Ht * g.slice_Wt;
                                    g.start_pages = st[1];
                                    g.start_row = st[2] * g.tensor_Wt;

                                    golden(g, gold, state_machine_disagreements);
                                    candidate(g, cand);
                                    ++configs;
                                    records += gold.size();

                                    ASSERT_EQ(gold.size(), cand.size()) << cfg_str(g);
                                    for (size_t k = 0; k < gold.size(); ++k) {
                                        ASSERT_TRUE(gold[k] == cand[k])
                                            << cfg_str(g) << "\n  record " << k << "\n  gold: " << rec_str(gold[k])
                                            << "\n  cand: " << rec_str(cand[k]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ASSERT_EQ(state_machine_disagreements, 0u);
    // Guard against a silently-degenerate grid.
    ASSERT_GT(configs, 0u);
    ASSERT_GT(records, 0u);
}

}  // namespace

// The default grid keeps runtime in unit-test territory while covering every chip of several ring
// sizes, both directions, all three scatter dims, granularities around DEST capacity, multi-batch,
// multi-channel, and both zero and non-zero worker start offsets.
TEST(CclRingSchedule, GoldenEquivalenceSweep) {
    const bool full = std::getenv("TT_CCL_SCHEDULE_SWEEP_FULL") != nullptr;
    if (full) {
        std::vector<uint32_t> rings;
        for (uint32_t r = 2; r <= 32; r += 2) {
            rings.push_back(r);
        }
        run_sweep(rings, {1, 2, 4, 8}, 32);
    } else {
        run_sweep({2, 4, 8}, {1, 4, 8}, 24);
    }
}

// The bug the original (uncommitted) sweep caught, as a named regression test: RingSliceCursor is
// PER-BATCH. Every batch must restart the ring walk at the same first slice; a cursor hoisted out
// of the batch loop makes batch N+1 continue where batch N stopped — silently, with no hang.
TEST(CclRingSchedule, SliceCursorRestartsEveryBatch) {
    constexpr uint32_t ring = 8;
    constexpr uint32_t chip = 3;
    for (int dir = 0; dir <= 1; ++dir) {
        std::vector<uint32_t> batch0, batch1;
        sched::RingRsSchedule schedule(ring, /*batches=*/2, /*channels=*/1, /*gran=*/4, 0, 8, dir);
        while (schedule.next_batch()) {
            sched::RingSliceCursor cursor(chip, ring, dir);
            auto& sink = schedule.batch_idx() == 0 ? batch0 : batch1;
            while (schedule.next_step()) {
                sink.push_back(cursor.wrap());
                while (schedule.next_channel()) {
                    while (schedule.next_chunk()) {
                    }
                }
                cursor.advance();
            }
        }
        ASSERT_EQ(batch0, batch1);
        ASSERT_EQ(batch0.front(), (chip + ring / 2) % ring);
    }
}

// At ring_size == 4, step 1 IS step half-1: the handover step and the separate-sems step coincide,
// and both of their effects must apply — matching the pre-migration writer's combined branch.
TEST(CclRingSchedule, Ring4Step1Coincidence) {
    for (int dir = 0; dir <= 1; ++dir) {
        const auto f = sched::ring_rs_step_flags(/*step=*/1, /*ring_size=*/4, dir);
        EXPECT_TRUE(f.even_chunks);
        EXPECT_TRUE(f.odd_chunks);
        EXPECT_EQ(f.reduce_even_chunks, static_cast<bool>(dir));  // step==1 behaviour
        EXPECT_EQ(f.reduce_odd_chunks, !static_cast<bool>(dir));  // step==1 behaviour
        EXPECT_TRUE(f.write_to_remote);
        EXPECT_EQ(f.write_to_interm, static_cast<bool>(dir));  // step==half-1 behaviour
        EXPECT_TRUE(f.separate_even_odd_sems);                 // step==half-1 behaviour
        EXPECT_FALSE(f.reduce_output);
    }
}
