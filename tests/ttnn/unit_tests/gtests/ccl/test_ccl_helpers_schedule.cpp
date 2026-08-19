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

// =============================================================================================
// LINE schedule equivalence sweep.
//
// GOLDEN: the pre-migration line_reduce_scatter_minimal_async reader + writer loops, transcribed
// verbatim from their last unmigrated revision (this branch's parent, 831a7c69301) — control flow,
// tile-id arithmetic, and the chunks-per-sync wait/signal cadence only. CANDIDATE: LineSliceCursor
// + LineChannelWalk + SyncCadence + SliceRowWalker/SequentialTileWalker + the S7 predicates,
// composed exactly as the migrated kernels drive them.
//
// COMPARED per chunk: batch/phase/target/slice/channel indices, tile count, the reader's
// out_ready-wait and fwd/bwd-wait flags, the writer's inc-after/tail-inc/fwd-bwd-inc flags, the
// writer's per-packet split (contig_pages_advanced), and EVERY tile id emitted by the input /
// intermediate / output walks.
// =============================================================================================

namespace {

struct LineCfg {
    uint32_t ring, chip;
    bool is_forward, is_first_dev, do_final, sync_other;
    uint32_t num_targets;
    uint32_t B, C, gran, cps, contig;
    uint32_t start, end;
    uint32_t dim;
    uint32_t slice_Wt, tensor_Wt, slice_Ht;
    uint32_t in_num_pages, in_batch, in_chan, out_batch, out_chan;
    uint32_t start_pages, start_row;
};

std::string line_cfg_str(const LineCfg& g) {
    std::ostringstream os;
    os << "ring=" << g.ring << " chip=" << g.chip << " fwd=" << g.is_forward << " first=" << g.is_first_dev
       << " final=" << g.do_final << " sync=" << g.sync_other << " targets=" << g.num_targets << " dim=" << g.dim
       << " B=" << g.B << " C=" << g.C << " gran=" << g.gran << " cps=" << g.cps << " contig=" << g.contig
       << " start=" << g.start << " end=" << g.end;
    return os.str();
}

struct LineEv {
    // kind: 0 = reader chunk, 1 = writer chunk, 2 = writer tail-inc point, 3 = writer final chunk,
    //       4 = reader final chunk
    uint32_t kind, b, target, slice_idx, c;
    uint32_t tiles;
    bool waited, fwdbwd, inc_after, tail_inc;
    std::vector<uint32_t> ids;      // reader: main ids; writer: packet-flattened interm/output ids
    std::vector<uint32_t> ids2;     // reader: intermediate ids
    std::vector<uint32_t> packets;  // writer phase 1: pages per packet

    bool operator==(const LineEv& o) const {
        return kind == o.kind && b == o.b && target == o.target && slice_idx == o.slice_idx && c == o.c &&
               tiles == o.tiles && waited == o.waited && fwdbwd == o.fwdbwd && inc_after == o.inc_after &&
               tail_inc == o.tail_inc && ids == o.ids && ids2 == o.ids2 && packets == o.packets;
    }
};

std::string line_ev_str(const LineEv& e) {
    std::ostringstream os;
    os << "kind=" << e.kind << " b=" << e.b << " t=" << e.target << " slice=" << e.slice_idx << " c=" << e.c
       << " tiles=" << e.tiles << " waited=" << e.waited << " fwdbwd=" << e.fwdbwd << " inc=" << e.inc_after
       << " tail=" << e.tail_inc << " ids=[";
    for (auto v : e.ids) {
        os << v << ",";
    }
    os << "] ids2=[";
    for (auto v : e.ids2) {
        os << v << ",";
    }
    os << "] pkts=[";
    for (auto v : e.packets) {
        os << v << ",";
    }
    os << "]";
    return os.str();
}

uint32_t line_slice_off(const LineCfg& g, uint32_t slice_idx) {
    if (g.dim == 3) {
        return slice_idx * g.slice_Wt;
    }
    if (g.dim == 2) {
        return slice_idx * g.slice_Ht * g.slice_Wt;
    }
    return slice_idx * g.C * g.slice_Ht * g.slice_Wt;  // dim == 1
}

// ------------------------------ GOLDEN: pre-migration reader --------------------------------
void line_reader_golden(const LineCfg& g, std::vector<LineEv>& out) {
    out.clear();
    const uint32_t interm_full = g.is_forward ? 0 : g.in_num_pages;
    uint32_t chunk_count = 0;
    uint32_t fwd_sync_cnt = 0;
    uint32_t sem_target = 0;
    (void)fwd_sync_cnt;
    (void)sem_target;

    for (uint32_t b = 0; b < g.B; ++b) {
        int slice_idx = g.is_forward ? static_cast<int>(g.ring) - 1 : 0;
        const uint32_t batch_offset = g.in_batch * b;

        for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
            chunk_count = 0;
            uint32_t input_start = line_slice_off(g, static_cast<uint32_t>(slice_idx)) + batch_offset;
            uint32_t interm_start = input_start + interm_full;

            for (uint32_t c = 0; c < g.C; ++c) {
                uint32_t in_pages = g.start_pages, in_row = g.start_row;
                uint32_t it_pages = g.start_pages, it_row = g.start_row;
                uint32_t tiles_read = g.start;

                while (tiles_read < g.end) {
                    const uint32_t n = std::min(g.end - tiles_read, g.gran);
                    LineEv ev{};
                    ev.kind = 0;
                    ev.b = b;
                    ev.target = iter;
                    ev.slice_idx = static_cast<uint32_t>(slice_idx);
                    ev.c = c;
                    ev.tiles = n;
                    if (!g.is_first_dev) {
                        // reader.cpp: wait when chunk_count % cps == 0, THEN chunk_count++.
                        ev.waited = (chunk_count % g.cps == 0);
                        ++chunk_count;
                    }
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(input_start + in_row + in_pages);
                        if (++in_pages == g.slice_Wt) {
                            in_row += g.tensor_Wt;
                            in_pages -= g.slice_Wt;
                        }
                        if (!g.is_first_dev) {
                            ev.ids2.push_back(interm_start + it_row + it_pages);
                            if (++it_pages == g.slice_Wt) {
                                it_row += g.tensor_Wt;
                                it_pages -= g.slice_Wt;
                            }
                        }
                    }
                    tiles_read += n;
                    out.push_back(std::move(ev));
                }
                input_start += g.in_chan;
                interm_start += g.in_chan;
            }
            slice_idx += g.is_forward ? -1 : 1;
        }

        if (g.do_final) {
            chunk_count = 0;
            const uint32_t my_off = line_slice_off(g, g.chip);
            uint32_t input_start = my_off + batch_offset;
            uint32_t interm_start = input_start + interm_full;
            uint32_t output_start = b * g.out_batch;
            const bool acc = g.sync_other && !g.is_forward;

            uint32_t main_start = acc ? output_start : input_start;
            const uint32_t main_stride = acc ? g.slice_Wt : g.tensor_Wt;
            const uint32_t main_chan = acc ? g.out_chan : g.in_chan;
            const uint32_t main_row0 = acc ? (g.start_row / g.tensor_Wt * g.slice_Wt) : g.start_row;

            for (uint32_t c = 0; c < g.C; ++c) {
                uint32_t pages = g.start_pages, row = main_row0;
                uint32_t it_pages = g.start_pages, it_row = g.start_row;
                uint32_t tiles_read = g.start;

                while (tiles_read < g.end) {
                    const uint32_t n = std::min(g.end - tiles_read, g.gran);
                    LineEv ev{};
                    ev.kind = 4;
                    ev.b = b;
                    ev.c = c;
                    ev.tiles = n;
                    ev.slice_idx = g.chip;
                    ev.fwdbwd = acc;  // one fwd/bwd wait per chunk when accumulating
                    ev.waited = (chunk_count % g.cps == 0);
                    ++chunk_count;
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(main_start + row + pages);
                        if (++pages == g.slice_Wt) {
                            row += main_stride;
                            pages -= g.slice_Wt;
                        }
                        ev.ids2.push_back(interm_start + it_row + it_pages);
                        if (++it_pages == g.slice_Wt) {
                            it_row += g.tensor_Wt;
                            it_pages -= g.slice_Wt;
                        }
                    }
                    tiles_read += n;
                    out.push_back(std::move(ev));
                }
                main_start += main_chan;
                interm_start += g.in_chan;
            }
        }
    }
}

// ------------------------------ GOLDEN: pre-migration writer --------------------------------
void line_writer_golden(const LineCfg& g, std::vector<LineEv>& out) {
    out.clear();
    const uint32_t interm_full = g.is_forward ? 0 : g.in_num_pages;

    for (uint32_t b = 0; b < g.B; ++b) {
        int slice_idx = g.is_forward ? static_cast<int>(g.ring) - 1 : 0;
        const uint32_t batch_offset = g.in_batch * b;

        for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
            uint32_t chunk_count = 0;
            uint32_t interm_start = line_slice_off(g, static_cast<uint32_t>(slice_idx)) + batch_offset + interm_full;

            for (uint32_t c = 0; c < g.C; ++c) {
                uint32_t it_pages = g.start_pages, it_row = g.start_row;
                uint32_t tiles_read = g.start;

                while (tiles_read < g.end) {
                    const uint32_t n = std::min(g.end - tiles_read, g.gran);
                    LineEv ev{};
                    ev.kind = 1;
                    ev.b = b;
                    ev.target = iter;
                    ev.slice_idx = static_cast<uint32_t>(slice_idx);
                    ev.c = c;
                    ev.tiles = n;
                    for (uint32_t j = 0; j < n; j += g.contig) {
                        const uint32_t p = std::min(g.contig, n - j);
                        ev.packets.push_back(p);
                        for (uint32_t k = 0; k < p; ++k) {
                            ev.ids.push_back(interm_start + it_row + it_pages);
                            if (++it_pages == g.slice_Wt) {
                                it_row += g.tensor_Wt;
                                it_pages -= g.slice_Wt;
                            }
                        }
                    }
                    tiles_read += n;
                    ++chunk_count;
                    ev.inc_after = (chunk_count % g.cps == 0);
                    out.push_back(std::move(ev));
                }
                interm_start += g.in_chan;
            }

            LineEv tail{};
            tail.kind = 2;
            tail.b = b;
            tail.target = iter;
            tail.slice_idx = static_cast<uint32_t>(slice_idx);
            tail.tail_inc = (chunk_count % g.cps != 0);
            out.push_back(std::move(tail));

            slice_idx += g.is_forward ? -1 : 1;
        }

        if (g.do_final) {
            uint32_t output_start = b * g.out_batch;
            const bool hands_off = g.sync_other && g.is_forward;
            for (uint32_t c = 0; c < g.C; ++c) {
                uint32_t tiles_read = g.start;
                while (tiles_read < g.end) {
                    const uint32_t n = std::min(g.end - tiles_read, g.gran);
                    LineEv ev{};
                    ev.kind = 3;
                    ev.b = b;
                    ev.c = c;
                    ev.tiles = n;
                    ev.fwdbwd = hands_off;  // one fwd/bwd inc per chunk when handing off
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(output_start + tiles_read);
                        ++tiles_read;
                    }
                    out.push_back(std::move(ev));
                }
                output_start += g.out_chan;
            }
        }
    }
}

// ----------------------- CANDIDATE: the schedule pieces, kernel-shaped -----------------------
void line_reader_candidate(const LineCfg& g, std::vector<LineEv>& out) {
    out.clear();
    const uint32_t interm_full = g.is_forward ? 0 : g.in_num_pages;
    sched::LineChannelWalk walk(g.C, g.gran, g.start, g.end);
    sched::SyncCadence cadence(g.cps);
    sched::SliceRowWalker input_walker(g.slice_Wt, g.tensor_Wt);
    sched::SliceRowWalker interm_walker(g.slice_Wt, g.tensor_Wt);
    uint32_t sem_target = 0;
    uint32_t fwd_sync_cnt = 0;
    (void)sem_target;
    (void)fwd_sync_cnt;

    for (uint32_t b = 0; b < g.B; ++b) {
        sched::LineSliceCursor cursor(g.is_forward, g.ring);
        const uint32_t batch_offset = g.in_batch * b;

        for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
            cadence.reset();
            const uint32_t slice_offset = line_slice_off(g, cursor.slice()) + batch_offset;
            input_walker.set_base(slice_offset);
            interm_walker.set_base(slice_offset + interm_full);

            walk.reset();
            while (walk.next_channel()) {
                input_walker.reset_offsets(g.start_pages, g.start_row);
                interm_walker.reset_offsets(g.start_pages, g.start_row);
                while (walk.next_chunk()) {
                    const uint32_t n = walk.tiles_this_chunk();
                    LineEv ev{};
                    ev.kind = 0;
                    ev.b = b;
                    ev.target = iter;
                    ev.slice_idx = cursor.slice();
                    ev.c = walk.channel_idx();
                    ev.tiles = n;
                    if (!g.is_first_dev) {
                        ev.waited = cadence.wait_due();
                        if (ev.waited) {
                            ++sem_target;
                        }
                        cadence.advance();
                    }
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(input_walker.next());
                        if (!g.is_first_dev) {
                            ev.ids2.push_back(interm_walker.next());
                        }
                    }
                    // A first-device reader never consumes the intermediate walker; keep it in
                    // step anyway (harmless), matching a uniform kernel body.
                    if (g.is_first_dev) {
                        interm_walker.advance(n);
                    }
                    out.push_back(std::move(ev));
                }
                input_walker.bump_base(g.in_chan);
                interm_walker.bump_base(g.in_chan);
            }
            cursor.advance();
        }

        if (g.do_final) {
            cadence.reset();
            const bool acc = sched::line_rs_accumulate_output(g.sync_other, g.is_forward);
            const uint32_t my_offset = line_slice_off(g, g.chip) + batch_offset;

            // The accumulate path reads the OUTPUT tensor: dense slice, stride slice_Wt, and the
            // worker's start_row_offset re-based from input rows onto slice rows.
            sched::SliceRowWalker main_walker(g.slice_Wt, acc ? g.slice_Wt : g.tensor_Wt);
            main_walker.set_base(acc ? b * g.out_batch : my_offset);
            const uint32_t main_row0 =
                acc ? sched::rebase_row_offset(g.start_row, g.tensor_Wt, g.slice_Wt) : g.start_row;
            const uint32_t main_chan = acc ? g.out_chan : g.in_chan;
            interm_walker.set_base(my_offset + interm_full);

            walk.reset();
            while (walk.next_channel()) {
                main_walker.reset_offsets(g.start_pages, main_row0);
                interm_walker.reset_offsets(g.start_pages, g.start_row);
                while (walk.next_chunk()) {
                    const uint32_t n = walk.tiles_this_chunk();
                    LineEv ev{};
                    ev.kind = 4;
                    ev.b = b;
                    ev.c = walk.channel_idx();
                    ev.tiles = n;
                    ev.slice_idx = g.chip;
                    ev.fwdbwd = acc;
                    if (acc) {
                        ++fwd_sync_cnt;
                    }
                    ev.waited = cadence.wait_due();
                    if (ev.waited) {
                        ++sem_target;
                    }
                    cadence.advance();
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(main_walker.next());
                        ev.ids2.push_back(interm_walker.next());
                    }
                    out.push_back(std::move(ev));
                }
                main_walker.bump_base(main_chan);
                interm_walker.bump_base(g.in_chan);
            }
        }
    }
}

void line_writer_candidate(const LineCfg& g, std::vector<LineEv>& out) {
    out.clear();
    const uint32_t interm_full = g.is_forward ? 0 : g.in_num_pages;
    sched::LineChannelWalk walk(g.C, g.gran, g.start, g.end);
    sched::SyncCadence cadence(g.cps);
    sched::SliceRowWalker interm_walker(g.slice_Wt, g.tensor_Wt);
    sched::SequentialTileWalker output_walker;

    for (uint32_t b = 0; b < g.B; ++b) {
        sched::LineSliceCursor cursor(g.is_forward, g.ring);
        const uint32_t batch_offset = g.in_batch * b;

        for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
            cadence.reset();
            interm_walker.set_base(line_slice_off(g, cursor.slice()) + batch_offset + interm_full);

            walk.reset();
            while (walk.next_channel()) {
                interm_walker.reset_offsets(g.start_pages, g.start_row);
                while (walk.next_chunk()) {
                    const uint32_t n = walk.tiles_this_chunk();
                    LineEv ev{};
                    ev.kind = 1;
                    ev.b = b;
                    ev.target = iter;
                    ev.slice_idx = cursor.slice();
                    ev.c = walk.channel_idx();
                    ev.tiles = n;
                    for (uint32_t j = 0; j < n; j += g.contig) {
                        const uint32_t p = std::min(g.contig, n - j);
                        ev.packets.push_back(p);
                        for (uint32_t k = 0; k < p; ++k) {
                            ev.ids.push_back(interm_walker.next());
                        }
                    }
                    cadence.advance();
                    ev.inc_after = cadence.signal_due();
                    out.push_back(std::move(ev));
                }
                interm_walker.bump_base(g.in_chan);
            }

            LineEv tail{};
            tail.kind = 2;
            tail.b = b;
            tail.target = iter;
            tail.slice_idx = cursor.slice();
            tail.tail_inc = cadence.tail_due();
            out.push_back(std::move(tail));

            cursor.advance();
        }

        if (g.do_final) {
            const bool hands_off = sched::line_rs_forward_hands_off(g.sync_other, g.is_forward);
            output_walker.set_base(b * g.out_batch);
            walk.reset();
            while (walk.next_channel()) {
                output_walker.reset_offsets(g.start);
                while (walk.next_chunk()) {
                    const uint32_t n = walk.tiles_this_chunk();
                    LineEv ev{};
                    ev.kind = 3;
                    ev.b = b;
                    ev.c = walk.channel_idx();
                    ev.tiles = n;
                    ev.fwdbwd = hands_off;
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(output_walker.next());
                    }
                    out.push_back(std::move(ev));
                }
                output_walker.bump_base(g.out_chan);
            }
        }
    }
}

void run_line_sweep() {
    std::vector<LineEv> gold, cand;
    uint64_t configs = 0, events = 0;

    for (uint32_t ring : {2u, 3u, 4u, 8u}) {
        for (uint32_t chip = 0; chip < ring; ++chip) {
            for (int fwd = 0; fwd <= 1; ++fwd) {
                // Host-legal target counts for a line: forward walks ring-1..chip+1, backward
                // 0..chip-1 — but the kernels take num_targets as a plain arg, so sweep the whole
                // range (a superset of host-reachable configs).
                for (uint32_t targets = 0; targets <= ring - 1; ++targets) {
                    for (uint32_t dim : {1u, 3u}) {
                        for (uint32_t gran : {2u, 4u}) {
                            for (uint32_t cps : {1u, 2u, 3u}) {
                                for (uint32_t contig : {1u, 2u}) {
                                    for (int first = 0; first <= 1; ++first) {
                                        for (int fin = 0; fin <= 1; ++fin) {
                                            for (int sync = 0; sync <= 1; ++sync) {
                                                for (uint32_t total : {0u, 5u, 8u, 13u}) {
                                                    LineCfg g{};
                                                    g.ring = ring;
                                                    g.chip = chip;
                                                    g.is_forward = fwd;
                                                    g.is_first_dev = first;
                                                    g.do_final = fin;
                                                    g.sync_other = sync;
                                                    g.num_targets = targets;
                                                    g.B = 2;
                                                    g.C = 2;
                                                    g.gran = gran;
                                                    g.cps = cps;
                                                    g.contig = contig;
                                                    g.start = 3;
                                                    g.end = 3 + total;
                                                    g.dim = dim;
                                                    g.slice_Wt = 3;
                                                    g.tensor_Wt = 3 * ring;
                                                    g.slice_Ht = 2;
                                                    g.in_num_pages = g.C * g.slice_Ht * g.tensor_Wt * g.B;
                                                    g.in_batch = g.C * g.slice_Ht * g.tensor_Wt;
                                                    g.in_chan = g.slice_Ht * g.tensor_Wt;
                                                    g.out_batch = g.C * g.slice_Ht * g.slice_Wt;
                                                    g.out_chan = g.slice_Ht * g.slice_Wt;
                                                    g.start_pages = 1;
                                                    g.start_row = g.tensor_Wt;

                                                    line_reader_golden(g, gold);
                                                    line_reader_candidate(g, cand);
                                                    ASSERT_EQ(gold.size(), cand.size()) << "reader " << line_cfg_str(g);
                                                    for (size_t k = 0; k < gold.size(); ++k) {
                                                        ASSERT_TRUE(gold[k] == cand[k])
                                                            << "reader " << line_cfg_str(g) << "\n  record " << k
                                                            << "\n  gold: " << line_ev_str(gold[k])
                                                            << "\n  cand: " << line_ev_str(cand[k]);
                                                    }
                                                    events += gold.size();

                                                    line_writer_golden(g, gold);
                                                    line_writer_candidate(g, cand);
                                                    ASSERT_EQ(gold.size(), cand.size()) << "writer " << line_cfg_str(g);
                                                    for (size_t k = 0; k < gold.size(); ++k) {
                                                        ASSERT_TRUE(gold[k] == cand[k])
                                                            << "writer " << line_cfg_str(g) << "\n  record " << k
                                                            << "\n  gold: " << line_ev_str(gold[k])
                                                            << "\n  cand: " << line_ev_str(cand[k]);
                                                    }
                                                    events += gold.size();
                                                    ++configs;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ASSERT_GT(configs, 0u);
    ASSERT_GT(events, 0u);
}

}  // namespace

TEST(CclLineSchedule, GoldenEquivalenceSweep) { run_line_sweep(); }

// The wait/signal cadence pairing invariant, standalone: for any (chunks, cps), the number of
// reader waits equals the number of writer signals (loop signals + the tail signal).
TEST(CclLineSchedule, SyncCadencePairing) {
    for (uint32_t cps = 1; cps <= 5; ++cps) {
        for (uint32_t chunks = 0; chunks <= 17; ++chunks) {
            sched::SyncCadence reader(cps), writer(cps);
            uint32_t waits = 0, signals = 0;
            for (uint32_t k = 0; k < chunks; ++k) {
                if (reader.wait_due()) {
                    ++waits;
                }
                reader.advance();
                writer.advance();
                if (writer.signal_due()) {
                    ++signals;
                }
                // At every prefix the reader may never have waited for more than was signalled,
                // except for the one wait admitting the in-flight group.
                ASSERT_LE(waits, signals + 1);
            }
            if (writer.tail_due()) {
                ++signals;
            }
            ASSERT_EQ(waits, signals) << "cps=" << cps << " chunks=" << chunks;
        }
    }
}

// =============================================================================================
// DIM-ZERO schedule equivalence sweeps.
//
// GOLDEN: the pre-migration dim_zero_ring / dim_zero_line reader + writer loops, transcribed
// verbatim from their last unmigrated revision (this branch, pre-migration) — the interleaved
// own/other chunk pairing (ring), the plain line walk with slice_B in the channel role (line),
// dense tile-id arithmetic, and the chunks-per-sync cadence. CANDIDATE: DimZeroChunkWalk +
// RingSliceCursor::starting_at (ring) / LineSliceCursor + LineChannelWalk (line) + SyncCadence +
// SequentialTileWalker, composed exactly as the migrated kernels drive them.
// =============================================================================================

namespace {

struct DzCfg {
    uint32_t ring, chip;
    bool dir;  // ring: direction; line: is_forward
    bool is_first_dev, do_final, sync_other;
    uint32_t num_targets;  // line only
    uint32_t sliceB, gran, cps, npack;
    uint32_t start, end;
    uint32_t out_num_pages, batch_num_pages, in_num_pages;
};

std::string dz_cfg_str(const DzCfg& g) {
    std::ostringstream os;
    os << "ring=" << g.ring << " chip=" << g.chip << " dir=" << g.dir << " first=" << g.is_first_dev
       << " final=" << g.do_final << " sync=" << g.sync_other << " targets=" << g.num_targets << " B=" << g.sliceB
       << " gran=" << g.gran << " cps=" << g.cps << " npack=" << g.npack << " start=" << g.start << " end=" << g.end;
    return os.str();
}

// One record per own-chunk (ring) or chunk (line); tail records carry only tail_inc.
struct DzEv {
    uint32_t kind, step, slice_idx, b;
    uint32_t tiles, pos;
    bool waited, fwdbwd, inc_after, tail_inc, do_reduce;
    std::vector<uint32_t> ids;   // main (input / intermediate-write / output) ids
    std::vector<uint32_t> ids2;  // reader: intermediate-read ids
    std::vector<uint32_t> packets;

    bool operator==(const DzEv& o) const {
        return kind == o.kind && step == o.step && slice_idx == o.slice_idx && b == o.b && tiles == o.tiles &&
               pos == o.pos && waited == o.waited && fwdbwd == o.fwdbwd && inc_after == o.inc_after &&
               tail_inc == o.tail_inc && do_reduce == o.do_reduce && ids == o.ids && ids2 == o.ids2 &&
               packets == o.packets;
    }
};

std::string dz_ev_str(const DzEv& e) {
    std::ostringstream os;
    os << "kind=" << e.kind << " step=" << e.step << " slice=" << e.slice_idx << " b=" << e.b << " tiles=" << e.tiles
       << " pos=" << e.pos << " waited=" << e.waited << " fwdbwd=" << e.fwdbwd << " inc=" << e.inc_after
       << " tail=" << e.tail_inc << " red=" << e.do_reduce << " ids=[";
    for (auto v : e.ids) {
        os << v << ",";
    }
    os << "] ids2=[";
    for (auto v : e.ids2) {
        os << v << ",";
    }
    os << "] pkts=[";
    for (auto v : e.packets) {
        os << v << ",";
    }
    os << "]";
    return os.str();
}

// --------------------------- dim-zero RING: pre-migration goldens ---------------------------
void dz_ring_reader_golden(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    uint32_t sem_target = 0;
    (void)sem_target;
    int slice_idx = g.dir ? static_cast<int>(g.chip) - 1 : static_cast<int>(g.chip) + 1;
    for (uint32_t i = 0; i < g.ring; ++i) {
        const bool do_reduce = i != 0;
        uint32_t actual;
        if (g.dir) {
            actual = slice_idx < 0 ? static_cast<uint32_t>(slice_idx + static_cast<int>(g.ring))
                                   : static_cast<uint32_t>(slice_idx);
        } else {
            actual = slice_idx >= static_cast<int>(g.ring) ? static_cast<uint32_t>(slice_idx) - g.ring
                                                           : static_cast<uint32_t>(slice_idx);
        }
        uint32_t base = actual * g.out_num_pages;
        uint32_t chunk_count = 0;
        for (uint32_t b = 0; b < g.sliceB; ++b) {
            uint32_t tiles_read = g.start;
            if (!g.dir && g.end > tiles_read) {
                tiles_read += std::min((g.end - tiles_read) / 2, g.gran);
            }
            while (tiles_read < g.end) {
                DzEv ev{};
                ev.kind = 0;
                ev.step = i;
                ev.slice_idx = actual;
                ev.b = b;
                ev.do_reduce = do_reduce;
                if (do_reduce && (chunk_count % g.cps == 0)) {
                    ev.waited = true;
                    ++sem_target;
                }
                ++chunk_count;
                const uint32_t rem = g.end - tiles_read;
                const uint32_t own = g.dir ? std::min(rem / 2, g.gran) : std::min(rem, g.gran);
                ev.tiles = own;
                ev.pos = tiles_read;
                for (uint32_t j = 0; j < own; ++j) {
                    ev.ids.push_back(base + tiles_read + j);
                    if (do_reduce) {
                        ev.ids2.push_back(base + tiles_read + j);
                    }
                }
                tiles_read += own;
                const uint32_t rem2 = g.end > tiles_read ? g.end - tiles_read : 0;
                if (rem2 > 0) {
                    tiles_read += g.dir ? std::min(rem2, g.gran) : std::min(rem2 / 2, g.gran);
                }
                out.push_back(std::move(ev));
            }
            base += g.batch_num_pages;
        }
        slice_idx += g.dir ? -1 : 1;
    }
}

void dz_ring_writer_golden(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    int slice_idx = g.dir ? static_cast<int>(g.chip) - 1 : static_cast<int>(g.chip) + 1;
    for (uint32_t i = 0; i < g.ring; ++i) {
        uint32_t actual;
        if (g.dir) {
            actual = slice_idx < 0 ? static_cast<uint32_t>(slice_idx + static_cast<int>(g.ring))
                                   : static_cast<uint32_t>(slice_idx);
        } else {
            actual = slice_idx >= static_cast<int>(g.ring) ? static_cast<uint32_t>(slice_idx) - g.ring
                                                           : static_cast<uint32_t>(slice_idx);
        }
        if (i < g.ring - 1) {
            uint32_t base = actual * g.out_num_pages;
            uint32_t chunk_count = 0;
            for (uint32_t b = 0; b < g.sliceB; ++b) {
                uint32_t tiles_read = g.start;
                if (!g.dir && g.end > tiles_read) {
                    tiles_read += std::min((g.end - tiles_read) / 2, g.gran);
                }
                while (tiles_read < g.end) {
                    DzEv ev{};
                    ev.kind = 1;
                    ev.step = i;
                    ev.slice_idx = actual;
                    ev.b = b;
                    const uint32_t rem = g.end - tiles_read;
                    const uint32_t own = g.dir ? std::min(rem / 2, g.gran) : std::min(rem, g.gran);
                    ev.tiles = own;
                    ev.pos = tiles_read;
                    uint32_t done = 0;
                    while (done < own) {
                        const uint32_t p = std::min(own - done, g.npack);
                        ev.packets.push_back(p);
                        for (uint32_t k = 0; k < p; ++k) {
                            ev.ids.push_back(base + tiles_read);
                            ++tiles_read;
                            ++done;
                        }
                    }
                    const uint32_t rem2 = g.end > tiles_read ? g.end - tiles_read : 0;
                    if (rem2 > 0) {
                        tiles_read += g.dir ? std::min(rem2, g.gran) : std::min(rem2 / 2, g.gran);
                    }
                    ++chunk_count;
                    ev.inc_after = (chunk_count % g.cps == 0);
                    out.push_back(std::move(ev));
                }
                base += g.batch_num_pages;
            }
            DzEv tail{};
            tail.kind = 2;
            tail.step = i;
            tail.slice_idx = actual;
            tail.tail_inc = (chunk_count % g.cps != 0);
            out.push_back(std::move(tail));
        } else {
            uint32_t base = 0;
            for (uint32_t b = 0; b < g.sliceB; ++b) {
                uint32_t tiles_read = g.start;
                if (!g.dir && g.end > tiles_read) {
                    tiles_read += std::min((g.end - tiles_read) / 2, g.gran);
                }
                while (tiles_read < g.end) {
                    DzEv ev{};
                    ev.kind = 3;
                    ev.step = i;
                    ev.slice_idx = actual;
                    ev.b = b;
                    const uint32_t rem = g.end - tiles_read;
                    const uint32_t own = g.dir ? std::min(rem / 2, g.gran) : std::min(rem, g.gran);
                    ev.tiles = own;
                    ev.pos = tiles_read;
                    for (uint32_t j = 0; j < own; ++j) {
                        ev.ids.push_back(base + tiles_read);
                        ++tiles_read;
                    }
                    const uint32_t rem2 = g.end > tiles_read ? g.end - tiles_read : 0;
                    if (rem2 > 0) {
                        tiles_read += g.dir ? std::min(rem2, g.gran) : std::min(rem2 / 2, g.gran);
                    }
                    out.push_back(std::move(ev));
                }
                base += g.batch_num_pages;
            }
        }
        slice_idx += g.dir ? -1 : 1;
    }
}

// ----------------------- dim-zero RING: candidates (kernel-shaped) --------------------------
void dz_ring_reader_candidate(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    auto cursor = sched::RingSliceCursor::starting_at(sched::dim_zero_ring_first_slice(g.chip, g.dir), g.ring, g.dir);
    sched::DimZeroChunkWalk walk(g.sliceB, g.gran, g.start, g.end, g.dir);
    sched::SyncCadence cadence(g.cps);
    uint32_t sem_target = 0;
    (void)sem_target;

    for (uint32_t i = 0; i < g.ring; ++i) {
        const bool do_reduce = i != 0;
        const uint32_t slice = cursor.wrap();
        uint32_t base = slice * g.out_num_pages;
        cadence.reset();
        walk.reset();
        while (walk.next_batch()) {
            while (walk.next_chunk()) {
                DzEv ev{};
                ev.kind = 0;
                ev.step = i;
                ev.slice_idx = slice;
                ev.b = walk.batch_idx();
                ev.do_reduce = do_reduce;
                if (do_reduce && cadence.wait_due()) {
                    ev.waited = true;
                    ++sem_target;
                }
                cadence.advance();
                const uint32_t n = walk.tiles_this_chunk();
                ev.tiles = n;
                ev.pos = walk.position();
                for (uint32_t j = 0; j < n; ++j) {
                    ev.ids.push_back(base + walk.position() + j);
                    if (do_reduce) {
                        ev.ids2.push_back(base + walk.position() + j);
                    }
                }
                out.push_back(std::move(ev));
            }
            base += g.batch_num_pages;
        }
        cursor.advance();
    }
}

void dz_ring_writer_candidate(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    auto cursor = sched::RingSliceCursor::starting_at(sched::dim_zero_ring_first_slice(g.chip, g.dir), g.ring, g.dir);
    sched::DimZeroChunkWalk walk(g.sliceB, g.gran, g.start, g.end, g.dir);
    sched::SyncCadence cadence(g.cps);

    for (uint32_t i = 0; i < g.ring; ++i) {
        const uint32_t slice = cursor.wrap();
        if (i < g.ring - 1) {
            uint32_t base = slice * g.out_num_pages;
            cadence.reset();
            walk.reset();
            while (walk.next_batch()) {
                while (walk.next_chunk()) {
                    DzEv ev{};
                    ev.kind = 1;
                    ev.step = i;
                    ev.slice_idx = slice;
                    ev.b = walk.batch_idx();
                    const uint32_t n = walk.tiles_this_chunk();
                    ev.tiles = n;
                    ev.pos = walk.position();
                    for (uint32_t j = 0; j < n; j += g.npack) {
                        const uint32_t p = std::min(g.npack, n - j);
                        ev.packets.push_back(p);
                        for (uint32_t k = 0; k < p; ++k) {
                            ev.ids.push_back(base + walk.position() + j + k);
                        }
                    }
                    cadence.advance();
                    ev.inc_after = cadence.signal_due();
                    out.push_back(std::move(ev));
                }
                base += g.batch_num_pages;
            }
            DzEv tail{};
            tail.kind = 2;
            tail.step = i;
            tail.slice_idx = slice;
            tail.tail_inc = cadence.tail_due();
            out.push_back(std::move(tail));
        } else {
            uint32_t base = 0;
            walk.reset();
            while (walk.next_batch()) {
                while (walk.next_chunk()) {
                    DzEv ev{};
                    ev.kind = 3;
                    ev.step = i;
                    ev.slice_idx = slice;
                    ev.b = walk.batch_idx();
                    const uint32_t n = walk.tiles_this_chunk();
                    ev.tiles = n;
                    ev.pos = walk.position();
                    for (uint32_t j = 0; j < n; ++j) {
                        ev.ids.push_back(base + walk.position() + j);
                    }
                    out.push_back(std::move(ev));
                }
                base += g.batch_num_pages;
            }
        }
        cursor.advance();
    }
}

// --------------------------- dim-zero LINE: pre-migration goldens ---------------------------
void dz_line_reader_golden(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    const uint32_t interm_full = g.dir ? 0 : g.in_num_pages;
    uint32_t chunk_count = 0;
    uint32_t sem_target = 0;
    uint32_t fwd_sync_cnt = 0;
    (void)sem_target;
    (void)fwd_sync_cnt;
    int slice_idx = g.dir ? static_cast<int>(g.ring) - 1 : 0;

    for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
        chunk_count = 0;
        uint32_t in_base = static_cast<uint32_t>(slice_idx) * g.out_num_pages;
        uint32_t it_base = in_base + interm_full;
        for (uint32_t b = 0; b < g.sliceB; ++b) {
            uint32_t tiles_read = g.start;
            while (tiles_read < g.end) {
                const uint32_t n = std::min(g.end - tiles_read, g.gran);
                DzEv ev{};
                ev.kind = 0;
                ev.step = iter;
                ev.slice_idx = static_cast<uint32_t>(slice_idx);
                ev.b = b;
                ev.tiles = n;
                ev.pos = tiles_read;
                if (!g.is_first_dev) {
                    if (chunk_count % g.cps == 0) {
                        ev.waited = true;
                        ++sem_target;
                    }
                    ++chunk_count;
                }
                for (uint32_t j = 0; j < n; ++j) {
                    ev.ids.push_back(in_base + tiles_read + j);
                    if (!g.is_first_dev) {
                        ev.ids2.push_back(it_base + tiles_read + j);
                    }
                }
                tiles_read += n;
                out.push_back(std::move(ev));
            }
            in_base += g.batch_num_pages;
            it_base += g.batch_num_pages;
        }
        slice_idx += g.dir ? -1 : 1;
    }

    if (g.do_final) {
        chunk_count = 0;
        const bool acc = g.sync_other && !g.dir;
        uint32_t in_base = g.chip * g.out_num_pages;
        uint32_t it_base = in_base + interm_full;
        uint32_t main_base = acc ? 0 : in_base;
        for (uint32_t b = 0; b < g.sliceB; ++b) {
            uint32_t tiles_read = g.start;
            while (tiles_read < g.end) {
                const uint32_t n = std::min(g.end - tiles_read, g.gran);
                DzEv ev{};
                ev.kind = 4;
                ev.slice_idx = g.chip;
                ev.b = b;
                ev.tiles = n;
                ev.pos = tiles_read;
                ev.fwdbwd = acc;
                if (acc) {
                    ++fwd_sync_cnt;
                }
                if (chunk_count % g.cps == 0) {
                    ev.waited = true;
                    ++sem_target;
                }
                ++chunk_count;
                for (uint32_t j = 0; j < n; ++j) {
                    ev.ids.push_back(main_base + tiles_read + j);
                    ev.ids2.push_back(it_base + tiles_read + j);
                }
                tiles_read += n;
                out.push_back(std::move(ev));
            }
            main_base += g.batch_num_pages;
            it_base += g.batch_num_pages;
        }
    }
}

void dz_line_writer_golden(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    const uint32_t interm_full = g.dir ? 0 : g.in_num_pages;
    int slice_idx = g.dir ? static_cast<int>(g.ring) - 1 : 0;

    for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
        uint32_t chunk_count = 0;
        uint32_t it_base = static_cast<uint32_t>(slice_idx) * g.out_num_pages + interm_full;
        for (uint32_t b = 0; b < g.sliceB; ++b) {
            uint32_t tiles_read = g.start;
            while (tiles_read < g.end) {
                const uint32_t n = std::min(g.end - tiles_read, g.gran);
                DzEv ev{};
                ev.kind = 1;
                ev.step = iter;
                ev.slice_idx = static_cast<uint32_t>(slice_idx);
                ev.b = b;
                ev.tiles = n;
                ev.pos = tiles_read;
                for (uint32_t j = 0; j < n; j += g.npack) {
                    const uint32_t p = std::min(g.npack, n - j);
                    ev.packets.push_back(p);
                    for (uint32_t k = 0; k < p; ++k) {
                        ev.ids.push_back(it_base + tiles_read);
                        ++tiles_read;
                    }
                }
                ++chunk_count;
                ev.inc_after = (chunk_count % g.cps == 0);
                out.push_back(std::move(ev));
            }
            it_base += g.batch_num_pages;
        }
        DzEv tail{};
        tail.kind = 2;
        tail.step = iter;
        tail.slice_idx = static_cast<uint32_t>(slice_idx);
        tail.tail_inc = (chunk_count % g.cps != 0);
        out.push_back(std::move(tail));
        slice_idx += g.dir ? -1 : 1;
    }

    if (g.do_final) {
        const bool hands_off = g.sync_other && g.dir;
        uint32_t out_base = 0;
        for (uint32_t b = 0; b < g.sliceB; ++b) {
            uint32_t tiles_read = g.start;
            while (tiles_read < g.end) {
                const uint32_t n = std::min(g.end - tiles_read, g.gran);
                DzEv ev{};
                ev.kind = 3;
                ev.b = b;
                ev.tiles = n;
                ev.pos = tiles_read;
                ev.fwdbwd = hands_off;
                for (uint32_t j = 0; j < n; ++j) {
                    ev.ids.push_back(out_base + tiles_read);
                    ++tiles_read;
                }
                out.push_back(std::move(ev));
            }
            out_base += g.batch_num_pages;
        }
    }
}

// ----------------------- dim-zero LINE: candidates (kernel-shaped) --------------------------
void dz_line_reader_candidate(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    const uint32_t interm_full = g.dir ? 0 : g.in_num_pages;
    sched::LineSliceCursor cursor(g.dir, g.ring);
    sched::LineChannelWalk walk(g.sliceB, g.gran, g.start, g.end);  // slice_B plays the channel role
    sched::SyncCadence cadence(g.cps);
    sched::SequentialTileWalker in_walker, it_walker;
    uint32_t sem_target = 0;
    uint32_t fwd_sync_cnt = 0;
    (void)sem_target;
    (void)fwd_sync_cnt;

    for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
        cadence.reset();
        in_walker.set_base(cursor.slice() * g.out_num_pages);
        it_walker.set_base(cursor.slice() * g.out_num_pages + interm_full);
        walk.reset();
        while (walk.next_channel()) {
            in_walker.reset_offsets(g.start);
            it_walker.reset_offsets(g.start);
            while (walk.next_chunk()) {
                const uint32_t n = walk.tiles_this_chunk();
                DzEv ev{};
                ev.kind = 0;
                ev.step = iter;
                ev.slice_idx = cursor.slice();
                ev.b = walk.channel_idx();
                ev.tiles = n;
                if (!g.is_first_dev) {
                    if (cadence.wait_due()) {
                        ev.waited = true;
                        ++sem_target;
                    }
                    cadence.advance();
                }
                // Dense ids straight off the walkers (walkers ARE the position).
                uint32_t first_id = 0;
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t id = in_walker.next();
                    if (j == 0) {
                        first_id = id;
                    }
                    ev.ids.push_back(id);
                    if (!g.is_first_dev) {
                        ev.ids2.push_back(it_walker.next());
                    }
                }
                if (g.is_first_dev) {
                    it_walker.advance(n);
                }
                ev.pos = first_id - (cursor.slice() * g.out_num_pages) -
                         (walk.channel_idx() * g.batch_num_pages);  // recover position for the trace
                out.push_back(std::move(ev));
            }
            in_walker.bump_base(g.batch_num_pages);
            it_walker.bump_base(g.batch_num_pages);
        }
        cursor.advance();
    }

    if (g.do_final) {
        cadence.reset();
        const bool acc = sched::line_rs_accumulate_output(g.sync_other, g.dir);
        sched::SequentialTileWalker main_walker;
        main_walker.set_base(acc ? 0 : g.chip * g.out_num_pages);
        it_walker.set_base(g.chip * g.out_num_pages + interm_full);
        walk.reset();
        while (walk.next_channel()) {
            main_walker.reset_offsets(g.start);
            it_walker.reset_offsets(g.start);
            while (walk.next_chunk()) {
                const uint32_t n = walk.tiles_this_chunk();
                DzEv ev{};
                ev.kind = 4;
                ev.slice_idx = g.chip;
                ev.b = walk.channel_idx();
                ev.tiles = n;
                ev.fwdbwd = acc;
                if (acc) {
                    ++fwd_sync_cnt;
                }
                if (cadence.wait_due()) {
                    ev.waited = true;
                    ++sem_target;
                }
                cadence.advance();
                uint32_t first_id = 0;
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t id = main_walker.next();
                    if (j == 0) {
                        first_id = id;
                    }
                    ev.ids.push_back(id);
                    ev.ids2.push_back(it_walker.next());
                }
                ev.pos = first_id - (acc ? 0 : g.chip * g.out_num_pages) - (walk.channel_idx() * g.batch_num_pages);
                out.push_back(std::move(ev));
            }
            main_walker.bump_base(g.batch_num_pages);
            it_walker.bump_base(g.batch_num_pages);
        }
    }
}

void dz_line_writer_candidate(const DzCfg& g, std::vector<DzEv>& out) {
    out.clear();
    const uint32_t interm_full = g.dir ? 0 : g.in_num_pages;
    sched::LineSliceCursor cursor(g.dir, g.ring);
    sched::LineChannelWalk walk(g.sliceB, g.gran, g.start, g.end);
    sched::SyncCadence cadence(g.cps);
    sched::SequentialTileWalker it_walker, out_walker;

    for (uint32_t iter = 0; iter < g.num_targets; ++iter) {
        cadence.reset();
        it_walker.set_base(cursor.slice() * g.out_num_pages + interm_full);
        walk.reset();
        while (walk.next_channel()) {
            it_walker.reset_offsets(g.start);
            while (walk.next_chunk()) {
                const uint32_t n = walk.tiles_this_chunk();
                DzEv ev{};
                ev.kind = 1;
                ev.step = iter;
                ev.slice_idx = cursor.slice();
                ev.b = walk.channel_idx();
                ev.tiles = n;
                uint32_t first_id = 0;
                bool got_first = false;
                for (uint32_t j = 0; j < n; j += g.npack) {
                    const uint32_t p = std::min(g.npack, n - j);
                    ev.packets.push_back(p);
                    for (uint32_t k = 0; k < p; ++k) {
                        const uint32_t id = it_walker.next();
                        if (!got_first) {
                            first_id = id;
                            got_first = true;
                        }
                        ev.ids.push_back(id);
                    }
                }
                ev.pos = got_first ? first_id - (cursor.slice() * g.out_num_pages + interm_full) -
                                         (walk.channel_idx() * g.batch_num_pages)
                                   : 0;
                cadence.advance();
                ev.inc_after = cadence.signal_due();
                out.push_back(std::move(ev));
            }
            it_walker.bump_base(g.batch_num_pages);
        }
        DzEv tail{};
        tail.kind = 2;
        tail.step = iter;
        tail.slice_idx = cursor.slice();
        tail.tail_inc = cadence.tail_due();
        out.push_back(std::move(tail));
        cursor.advance();
    }

    if (g.do_final) {
        const bool hands_off = sched::line_rs_forward_hands_off(g.sync_other, g.dir);
        out_walker.set_base(0);
        walk.reset();
        while (walk.next_channel()) {
            out_walker.reset_offsets(g.start);
            while (walk.next_chunk()) {
                const uint32_t n = walk.tiles_this_chunk();
                DzEv ev{};
                ev.kind = 3;
                ev.b = walk.channel_idx();
                ev.tiles = n;
                ev.fwdbwd = hands_off;
                uint32_t first_id = 0;
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t id = out_walker.next();
                    if (j == 0) {
                        first_id = id;
                    }
                    ev.ids.push_back(id);
                }
                ev.pos = first_id - (walk.channel_idx() * g.batch_num_pages);
                out.push_back(std::move(ev));
            }
            out_walker.bump_base(g.batch_num_pages);
        }
    }
}

void dz_compare(const char* who, const DzCfg& g, const std::vector<DzEv>& gold, const std::vector<DzEv>& cand) {
    ASSERT_EQ(gold.size(), cand.size()) << who << " " << dz_cfg_str(g);
    for (size_t k = 0; k < gold.size(); ++k) {
        ASSERT_TRUE(gold[k] == cand[k]) << who << " " << dz_cfg_str(g) << "\n  record " << k
                                        << "\n  gold: " << dz_ev_str(gold[k]) << "\n  cand: " << dz_ev_str(cand[k]);
    }
}

}  // namespace

TEST(CclDimZeroSchedule, RingGoldenEquivalenceSweep) {
    std::vector<DzEv> gold, cand;
    uint64_t configs = 0;
    for (uint32_t ring : {2u, 3u, 4u, 8u}) {
        for (uint32_t chip = 0; chip < ring; ++chip) {
            for (int dir = 0; dir <= 1; ++dir) {
                for (uint32_t gran : {1u, 2u, 4u}) {
                    for (uint32_t cps : {1u, 2u, 3u}) {
                        for (uint32_t npack : {1u, 2u}) {
                            for (uint32_t sliceB : {1u, 2u, 3u}) {
                                for (uint32_t total : {0u, 1u, 5u, 8u, 13u}) {
                                    DzCfg g{};
                                    g.ring = ring;
                                    g.chip = chip;
                                    g.dir = dir;
                                    g.sliceB = sliceB;
                                    g.gran = gran;
                                    g.cps = cps;
                                    g.npack = npack;
                                    g.start = 2;
                                    g.end = 2 + total;
                                    g.out_num_pages = 64;
                                    g.batch_num_pages = 16;
                                    dz_ring_reader_golden(g, gold);
                                    dz_ring_reader_candidate(g, cand);
                                    dz_compare("dz_ring_reader", g, gold, cand);
                                    dz_ring_writer_golden(g, gold);
                                    dz_ring_writer_candidate(g, cand);
                                    dz_compare("dz_ring_writer", g, gold, cand);
                                    ++configs;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ASSERT_GT(configs, 0u);
}

TEST(CclDimZeroSchedule, LineGoldenEquivalenceSweep) {
    std::vector<DzEv> gold, cand;
    uint64_t configs = 0;
    for (uint32_t ring : {2u, 3u, 4u, 8u}) {
        for (uint32_t chip = 0; chip < ring; ++chip) {
            for (int dir = 0; dir <= 1; ++dir) {
                for (uint32_t targets = 0; targets <= ring - 1; ++targets) {
                    for (uint32_t gran : {2u, 4u}) {
                        for (uint32_t cps : {1u, 3u}) {
                            for (uint32_t npack : {1u, 2u}) {
                                for (int first = 0; first <= 1; ++first) {
                                    for (int fin = 0; fin <= 1; ++fin) {
                                        for (int sync = 0; sync <= 1; ++sync) {
                                            for (uint32_t total : {0u, 5u, 13u}) {
                                                DzCfg g{};
                                                g.ring = ring;
                                                g.chip = chip;
                                                g.dir = dir;
                                                g.is_first_dev = first;
                                                g.do_final = fin;
                                                g.sync_other = sync;
                                                g.num_targets = targets;
                                                g.sliceB = 2;
                                                g.gran = gran;
                                                g.cps = cps;
                                                g.npack = npack;
                                                g.start = 3;
                                                g.end = 3 + total;
                                                g.out_num_pages = 64;
                                                g.batch_num_pages = 16;
                                                g.in_num_pages = 128;
                                                dz_line_reader_golden(g, gold);
                                                dz_line_reader_candidate(g, cand);
                                                dz_compare("dz_line_reader", g, gold, cand);
                                                dz_line_writer_golden(g, gold);
                                                dz_line_writer_candidate(g, cand);
                                                dz_compare("dz_line_writer", g, gold, cand);
                                                ++configs;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ASSERT_GT(configs, 0u);
}
