// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar narrow-row pack-untilize, end to end, via iDMA (test id 918).
//
// THE PROBLEM. `_llk_pack_untilize_` cannot produce an untilized row narrower than a whole
// number of tiles on Quasar. Its output stride is face-granular and the compute API exposes
// it only in whole tiles (api/compute/pack_untilize.h static_asserts narrow_row == false for
// ARCH_QUASAR); Wormhole/Blackhole had a 16-byte output stride register, Quasar does not. So
// untilizing a matrix whose width is not a multiple of 32 leaves junk at the end of every
// row, and the current workaround is for the consumer's NOC to issue a separate read per row
// to step over it -- one transaction per row, where one would do.
//
// WHAT THIS TEST DOES. Two stages in one program:
//
//   stage 1  stock whole-tile HW pack_untilize  ->  32 rows x (ct_dim*32) datums, padded
//   stage 2  ONE iDMA transaction               ->  32 rows x matrix_w datums, dense
//
// with matrix_w = (ct_dim - 1) * 32 + last_tile_w. The output is the same dense narrow-row
// matrix the RV_PACR per-face-row path builds, and the host checks it datum for datum against
// a golden, for every shape and every engine.
//
// WHY iDMA RATHER THAN RV_PACR. RV_PACR drives the packer by hand, one 16-datum DEST face-row
// per op, 64 ops per tile, with the face de-interleave and the row stride computed in software
// per op -- it works, but tt-llk measures 1439-2418 cyc/tile against 77.4 for the hardware
// path. Here the hardware does the untilize at full speed and iDMA only moves 32 rows
// afterwards. iDMA also addresses L1 by the BYTE where RV_PACR's output address is 16-byte
// granular, so this path has no minimum narrow width, no spill into the next row and no
// two-pass write ordering (see SubFaceWidths).
//
// THE ENGINE AXIS. All three engines produce identical output and all three are verified:
//   0 IDMA_SCATTER  one transaction; the hardware walks a 32-entry address list  <- proposal
//   1 IDMA_PER_ROW  32 transactions from the address generator, one drain
//   2 NOC_PER_ROW   32 stateful NOC reads == the current workaround == the bar to beat
//
// Running, reporting and the layout contract: see README.md in this directory.

#include <algorithm>
#include <cstdint>
#include <vector>
#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_narrow_row {

constexpr std::uint32_t TEST_ID = 918;

constexpr std::uint32_t TILE_H = 32;
constexpr std::uint32_t TILE_W = 32;
constexpr std::uint32_t FACE_DIM = 16;
constexpr std::uint32_t DATUM_BYTES = 2;  // Float16_b
constexpr std::uint32_t TILE_BYTES = TILE_H * TILE_W * DATUM_BYTES;
constexpr std::uint32_t OUT_ROWS = TILE_H;

// Stage 2 is idempotent (an L1->L1 gather), so it can be repeated inside the timed zone and
// still leave the correct answer behind: every run is a correctness test AND a steady-state
// timing. Stage 1 cannot -- see COMPUTE_LOOP_FACTOR.
constexpr std::uint32_t COMPACT_ITERATIONS = 16;
// 1, and not a tuning choice: the Quasar tile-counter model consumes a tile per unpack, so
// re-packing resident tiles without re-streaming them gives valid timing but undefined data
// from the second iteration on. Stage 1 is therefore measured single-shot, which includes
// pipeline fill and reads high against a steady-state number.
constexpr std::uint32_t COMPUTE_LOOP_FACTOR = 1;

constexpr std::uint32_t ENGINE_IDMA_SCATTER = 0;
constexpr std::uint32_t ENGINE_IDMA_PER_ROW = 1;
constexpr std::uint32_t ENGINE_NOC_PER_ROW = 2;

// All 8 iDMA backend VCs. Each carries 16 B/cycle, so this is the 128 B/cycle ceiling.
constexpr std::uint32_t CHANNELS_ALL = 8;
// Floor on a sub-split packet: below this the issue cost swamps whatever the split buys. It
// is also what makes the split arm of ChannelSweep a no-op on short rows, since a floor at or
// above the row size is not a split.
constexpr std::uint32_t MIN_SPLIT_PACKET_BYTES = 64;

// The reference workload: a 32 x 252 Float16_b matrix. 252 datums needs 8 tiles to cover
// (7 x 32 = 224, + 28), so ct_dim 8 -- right at the half-sync 16-bit DEST limit for
// pack_untilize -- with 28 datums kept from the last tile. The padded row is 256 datums
// (512 B), the dense row 252 (504 B), so every row drops 4 datums and row r shifts back 4r.
constexpr std::uint32_t LAST_W_252 = 28;

const char* engine_name(std::uint32_t e) {
    switch (e) {
        case ENGINE_IDMA_SCATTER: return "iDMA scatter-list";
        case ENGINE_IDMA_PER_ROW: return "iDMA per-row";
        default: return "NOC per-row";
    }
}

constexpr CoreCoord CORE = {0, 0};

// The scatter list is 32 entries x 8 B = 256 B.
constexpr std::uint32_t LIST_SLOT_BYTES = 256;
// Every program run in the PROCESS takes a fresh slot. The iDMA fetches a scatter list once
// per address and later writes to that address do not take effect, so two runs sharing a list
// address make the second silently replay the first one's list -- which looks exactly like a
// hardware result. A process-global counter (not per-test) is what makes that impossible even
// when the allocator hands two tests the same buffer base.
constexpr std::uint32_t LIST_SLOTS = 128;
std::uint32_t g_next_list_slot = 0;

// Guard band after the dense output. A gather that writes past its row must not go unnoticed.
constexpr std::uint32_t GUARD_BYTES = 256;
constexpr std::uint32_t GUARD_FILL = 0xA5A5A5A5;
// Fills the DRAM input buffer past the tiles this run actually uses. Never read by the
// reader, so it should never appear anywhere; distinct from GUARD_FILL to tell the two apart.
constexpr std::uint32_t SRC_PAD_FILL = 0xDEADBEEF;

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return std::getenv("TT_METAL_SIMULATOR") == nullptr;
}

// Unique, bit-stable Float16_b stimulus: each datum names its own position in the PADDED
// untilized matrix, so a datum that lands in the wrong place is caught by value, not just by
// a checksum. 0x4000 | (idx & 0x1FFF) keeps the bf16 exponent field in [0x80, 0xBF] -- always
// normal, never denormal, Inf or NaN -- so the L1 -> SrcA -> DEST -> pack -> L1 round trip is
// bit-exact. That matters: the Quasar FPU writes no denormal to DEST, so a denormal pattern
// would come back as zero and read as a data-movement bug.
// idx < 32 * 256 = 8192 fits 13 bits exactly at the widest shape tested (ct_dim = 8).
inline std::uint16_t datum_at(std::uint32_t r, std::uint32_t c, std::uint32_t pad_w) {
    return static_cast<std::uint16_t>(0x4000u | ((r * pad_w + c) & 0x1FFFu));
}

// Tilized stimulus for a ct_dim-wide tile-row, in the layout the unpacker expects: per tile,
// four 16x16 faces in order top-left, top-right, bottom-left, bottom-right, each row-major,
// two datums per uint32 with the even datum in the low half (pack_two_bfloat16_into_uint32).
// Matches gold_standard_tilize; hand-rolled so this test does not depend on the llk test
// target's translation units.
std::vector<std::uint32_t> make_tilized_input(std::uint32_t ct_dim) {
    const std::uint32_t pad_w = ct_dim * TILE_W;
    std::vector<std::uint16_t> tilized(ct_dim * TILE_H * TILE_W);
    for (std::uint32_t t = 0; t < ct_dim; t++) {
        for (std::uint32_t f = 0; f < 4; f++) {
            for (std::uint32_t fr = 0; fr < FACE_DIM; fr++) {
                for (std::uint32_t fc = 0; fc < FACE_DIM; fc++) {
                    const std::uint32_t r = (f / 2) * FACE_DIM + fr;
                    const std::uint32_t c = t * TILE_W + (f % 2) * FACE_DIM + fc;
                    tilized[t * TILE_H * TILE_W + f * FACE_DIM * FACE_DIM + fr * FACE_DIM + fc] = datum_at(r, c, pad_w);
                }
            }
        }
    }
    std::vector<std::uint32_t> packed(tilized.size() / 2);
    for (std::size_t i = 0; i < packed.size(); i++) {
        packed[i] = static_cast<std::uint32_t>(tilized[2 * i]) | (static_cast<std::uint32_t>(tilized[2 * i + 1]) << 16);
    }
    return packed;
}

struct Buffers {
    std::shared_ptr<distributed::MeshBuffer> src_dram;  // tilized input
    std::shared_ptr<distributed::MeshBuffer> out_l1;    // dense narrow-row output + guard
    std::shared_ptr<distributed::MeshBuffer> list_l1;   // scatter lists, one slot per run
};

Buffers make_buffers(const std::shared_ptr<distributed::MeshDevice>& mesh_device, std::uint32_t max_ct_dim) {
    auto mk = [&](std::uint32_t bytes, BufferType type) {
        // page_size == the whole buffer keeps it in a single bank, which is what the direct
        // reader (bank_id 0) and the L1 read-back both assume.
        distributed::DeviceLocalBufferConfig local{.page_size = bytes, .buffer_type = type};
        distributed::ReplicatedBufferConfig cfg{.size = bytes};
        return distributed::MeshBuffer::create(cfg, local, mesh_device.get());
    };
    // Sized for the widest shape: matrix_w <= max_ct_dim * 32 datums.
    const std::uint32_t max_out_bytes = OUT_ROWS * max_ct_dim * TILE_W * DATUM_BYTES + GUARD_BYTES;
    return {
        .src_dram = mk(max_ct_dim * TILE_BYTES, BufferType::DRAM),
        .out_l1 = mk(max_out_bytes, BufferType::L1),
        .list_l1 = mk(LIST_SLOTS * LIST_SLOT_BYTES, BufferType::L1),
    };
}

struct RunConfig {
    std::uint32_t ct_dim = 1;
    std::uint32_t last_tile_w = 32;  // datums kept from the LAST tile; matrix_w derives from it
    std::uint32_t engine_mode = ENGINE_IDMA_SCATTER;
    std::uint32_t num_channels = 1;
    // 0 => the default policy: one packet per row (max_packet_bytes = out_row_bytes). That is
    // the RIGHT default even when fanning out, because both iDMA engines already emit one
    // packet per row, so 32 rows always give the round-robin more packets than it has
    // channels. Set this only to deliberately sub-split a row -- see ChannelSweep.
    std::uint32_t max_packet_bytes = 0;
};

// Builds and runs the two-stage program, then checks the dense output datum for datum.
// Returns true iff the narrow-row matrix is exactly right.
bool run_narrow_row(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Buffers& buffers, const RunConfig& cfg) {
    IDevice* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t ct_dim = cfg.ct_dim;
    const std::uint32_t pad_w = ct_dim * TILE_W;                             // padded row, datums
    const std::uint32_t matrix_w = (ct_dim - 1) * TILE_W + cfg.last_tile_w;  // dense row, datums
    const std::uint32_t pad_row_bytes = pad_w * DATUM_BYTES;
    const std::uint32_t out_row_bytes = matrix_w * DATUM_BYTES;
    const std::uint32_t out_bytes = OUT_ROWS * out_row_bytes;

    const std::uint32_t src_addr = buffers.src_dram->address();
    const std::uint32_t out_addr = buffers.out_l1->address();
    if (g_next_list_slot >= LIST_SLOTS) {
        log_warning(tt::LogTest, "scatter-list slots exhausted ({}); wrapping may replay a stale list", LIST_SLOTS);
        g_next_list_slot = 0;
    }
    const std::uint32_t list_addr = buffers.list_l1->address() + (g_next_list_slot++) * LIST_SLOT_BYTES;

    // The NOC engine reads this core's own L1 (loopback), so it needs PHYSICAL noc coords --
    // logical {0,0} is physical (0,1) on the 1x3 emu.
    const CoreCoord physical_core = device->worker_core_from_logical_core(CORE);
    const std::uint32_t packed_coords = ((std::uint32_t)physical_core.x << 16) | (std::uint32_t)physical_core.y;

    const std::uint32_t max_packet_bytes = cfg.max_packet_bytes != 0 ? cfg.max_packet_bytes : out_row_bytes;

    // ---- stimulus -----------------------------------------------------------------------
    std::vector<std::uint32_t> src_vec = make_tilized_input(ct_dim);
    // EnqueueWriteMeshBuffer takes the buffer by non-const reference, so it needs a copy of
    // the handle rather than the const one reachable through `buffers`.
    auto src_dram = buffers.src_dram;
    // It also asserts the source FILLS the buffer (src.size()*sizeof >= buffer->size()), and
    // the buffer is sized for the widest shape the calling test body uses -- so a body that
    // mixes ct_dim throws on its first narrower run unless the tail is padded. Pad with a
    // poison value rather than zeros: the reader only streams ct_dim tiles, so any of this
    // reaching the output is a bug worth seeing rather than a plausible-looking zero.
    src_vec.resize(src_dram->size() / sizeof(std::uint32_t), SRC_PAD_FILL);
    distributed::EnqueueWriteMeshBuffer(cq, src_dram, src_vec, /*blocking=*/true);

    // Prefill output + guard band, so "the gather never ran" and "the gather overran its row"
    // are both visible rather than passing on stale data.
    std::vector<std::uint32_t> out_init((out_bytes + GUARD_BYTES) / sizeof(std::uint32_t), GUARD_FILL);
    tt_metal::detail::WriteToDeviceL1(device, CORE, out_addr, out_init);

    // The scatter list: entry r is the OFFSET of padded row r. The kernel passes the pad DFB's
    // L1 base as SCATTER_BASE_ADDR, so the hardware forms base + offset -- which means the
    // host can build the list without knowing where the DFB landed. Entries are 8 B and the
    // engine takes the low 32 bits (resolved empirically; the encoding is not documented).
    std::vector<std::uint32_t> list_words(LIST_SLOT_BYTES / sizeof(std::uint32_t), 0);
    for (std::uint32_t r = 0; r < OUT_ROWS; r++) {
        list_words[r * 2] = r * pad_row_bytes;  // low word of an 8 B entry; high word stays 0
    }
    tt_metal::detail::WriteToDeviceL1(device, CORE, list_addr, list_words);

    // ---- program ------------------------------------------------------------------------
    const experimental::DFBSpecName SRC_DFB{"src_dfb"};
    const experimental::DFBSpecName PAD_DFB{"pad_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName COMPUTE{"narrow_row_untilize"};
    const experimental::KernelSpecName COMPACT{"narrow_row_compact"};

    experimental::DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = TILE_BYTES,
        .num_entries = ct_dim,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    // The untilized output is addressed in ROWS, not tiles: one entry per output row, TILE_H
    // of them. Same convention the shipping pack_untilize consumers use (RM_VALUE_OUTPUT in
    // sort_program_factory.cpp). Total L1 matches a tile-granular spec, but reserve/push counts
    // follow the entry granularity, so it has to be stated this way.
    experimental::DataflowBufferSpec pad_dfb_spec{
        .unique_id = PAD_DFB,
        .entry_size = pad_row_bytes,
        .num_entries = OUT_ROWS,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(SRC_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_narrow_row/kernels/narrow_row_untilize_compute.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(SRC_DFB, "src"), experimental::ProducerOf(PAD_DFB, "pad")},
        .compile_time_args =
            {{"ct_dim", ct_dim}, {"out_rows", OUT_ROWS}, {"loop_factor", COMPUTE_LOOP_FACTOR}, {"test_id", TEST_ID}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::KernelSpec compact_spec{
        .unique_id = COMPACT,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_narrow_row/kernels/narrow_row_compact_dm.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(PAD_DFB, "pad")},
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"dst_addr",
                     "list_addr",
                     "pad_row_bytes",
                     "out_row_bytes",
                     "num_rows",
                     "engine_mode",
                     "dest_coords",
                     "max_packet_bytes",
                     "num_channels",
                     "num_iterations",
                     "test_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::ProgramSpec spec{
        .name = "pack_untilize_narrow_row",
        .kernels = {reader_spec, compute_spec, compact_spec},
        .dataflow_buffers = {src_dfb_spec, pad_dfb_spec},
        .work_units = {{.name = "main", .kernels = {READER, COMPUTE, COMPACT}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_addr}, {"src_bank_id", 0u}, {"num_tiles", ct_dim}, {"dram_page_stride", TILE_BYTES}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = COMPACT,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", out_addr},
                 {"list_addr", list_addr},
                 {"pad_row_bytes", pad_row_bytes},
                 {"out_row_bytes", out_row_bytes},
                 {"num_rows", OUT_ROWS},
                 {"engine_mode", cfg.engine_mode},
                 {"dest_coords", packed_coords},
                 {"max_packet_bytes", max_packet_bytes},
                 {"num_channels", cfg.num_channels},
                 {"num_iterations", COMPACT_ITERATIONS},
                 {"test_id", TEST_ID}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // ---- check --------------------------------------------------------------------------
    // out_bytes = 32 rows * matrix_w * 2 B is always a multiple of 64, so the read and the
    // guard band start word-aligned whatever matrix_w is. Individual ROW offsets need not be:
    // an odd matrix_w puts row r at a 2 mod 4 byte offset, which is exactly the byte-granular
    // placement SubFaceWidths is there to probe.
    const std::uint32_t read_bytes = out_bytes + GUARD_BYTES;
    std::vector<std::uint32_t> out_words;
    tt_metal::detail::ReadFromDeviceL1(device, CORE, out_addr, read_bytes, out_words);
    const auto* out_datums = reinterpret_cast<const std::uint16_t*>(out_words.data());

    std::uint32_t bad = 0;
    std::uint32_t first_bad_r = 0, first_bad_c = 0;
    std::uint16_t first_bad_got = 0, first_bad_want = 0;
    for (std::uint32_t r = 0; r < OUT_ROWS; r++) {
        for (std::uint32_t c = 0; c < matrix_w; c++) {
            const std::uint16_t want = datum_at(r, c, pad_w);
            const std::uint16_t got = out_datums[r * matrix_w + c];
            if (got != want) {
                if (bad == 0) {
                    first_bad_r = r;
                    first_bad_c = c;
                    first_bad_got = got;
                    first_bad_want = want;
                }
                bad++;
            }
        }
    }

    // Guard band: everything after the dense matrix must be untouched. Catches a gather that
    // writes more per row than it should, or one that overruns the last row.
    std::uint32_t guard_bad = 0;
    for (std::uint32_t w = out_bytes / sizeof(std::uint32_t); w < out_words.size(); w++) {
        if (out_words[w] != GUARD_FILL) {
            guard_bad++;
        }
    }

    if (bad != 0 || guard_bad != 0) {
        log_error(
            tt::LogTest,
            "ct_dim={} last_tile_w={} matrix_w={} engine={} ch={}: {}/{} datums wrong "
            "(first at row {} col {}: expected 0x{:04x}, got 0x{:04x}), {} guard words clobbered",
            ct_dim,
            cfg.last_tile_w,
            matrix_w,
            engine_name(cfg.engine_mode),
            cfg.num_channels,
            bad,
            OUT_ROWS * matrix_w,
            first_bad_r,
            first_bad_c,
            first_bad_want,
            first_bad_got,
            guard_bad);
        return false;
    }
    log_info(
        tt::LogTest,
        "ct_dim={} last_tile_w={} matrix_w={} ({} B/row, padded {} B/row) engine={} ch={}: OK",
        ct_dim,
        cfg.last_tile_w,
        matrix_w,
        out_row_bytes,
        pad_row_bytes,
        engine_name(cfg.engine_mode),
        cfg.num_channels);
    return true;
}

}  // namespace unit_tests::dm::quasar_narrow_row

// =============================================================================
// Test Suite: Quasar narrow-row pack-untilize via iDMA (HW untilize + iDMA compaction)
// =============================================================================

class QuasarNarrowRowUntilize : public QuasarMeshDeviceSingleCardFixture {};

// START HERE. One tile, last_tile_w = 16 -- the smallest shape that is actually narrow, and
// the one the RV_PACR path reports 1470.4 cyc/tile for. If this fails, nothing below is
// meaningful; run the iDMA functional examples (*QuasarIdmaOps*) to tell a broken kernel apart
// from a broken emulator.
TEST_F(QuasarNarrowRowUntilize, SingleTileHalfWidth) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/1);
    EXPECT_TRUE(run_narrow_row(devices_[0], buffers, {.ct_dim = 1, .last_tile_w = 16}));
}

// The widths RV_PACR supports, so the two paths can be compared on equal ground: 8, 16, 24 are
// narrow; 32 is the degenerate whole-tile case and must also come out right (matrix_w == pad_w,
// so the gather becomes a straight copy -- a good null check on the address math).
TEST_F(QuasarNarrowRowUntilize, WidthSweep) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/1);
    for (std::uint32_t last_tile_w : {8u, 16u, 24u, 32u}) {
        EXPECT_TRUE(run_narrow_row(devices_[0], buffers, {.ct_dim = 1, .last_tile_w = last_tile_w}))
            << "last_tile_w=" << last_tile_w;
    }
}

// Multi-tile rows: full tiles 0..ct_dim-2 at 32 datums each plus a narrow last tile. This is
// the shape a real matrix has, and it is where the gather stops being descriptor-bound -- rows
// grow to (ct_dim-1)*64 + 32 bytes, so the per-row payload starts to dominate the per-entry
// cost. ct_dim 8 is the half-sync 16-bit DEST limit for pack_untilize.
TEST_F(QuasarNarrowRowUntilize, TileRowSweep) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/8);
    for (std::uint32_t ct_dim : {1u, 2u, 4u, 8u}) {
        for (std::uint32_t last_tile_w : {8u, 16u, 32u}) {
            EXPECT_TRUE(run_narrow_row(devices_[0], buffers, {.ct_dim = ct_dim, .last_tile_w = last_tile_w}))
                << "ct_dim=" << ct_dim << " last_tile_w=" << last_tile_w;
        }
    }
}

// Widths BELOW the RV_PACR floor. RV_PACR's output address is 16-byte granular, so for a
// 16-bit format it cannot place rows closer than 8 datums apart (and for an 8-bit format, not
// closer than 16) -- narrow_row = 8 is its hard minimum, and widths that are not a multiple of
// 8 make it write a full 16-datum face-row that spills into the next output row, which it then
// has to overwrite in a second pass. iDMA addresses L1 by the byte, so none of that applies:
// these widths are just a smaller out_row_bytes. 1 and 3 also make out_row_bytes odd, which
// probes byte- rather than word-granular placement.
//
// A failure here is a FINDING, not a broken test: it says where iDMA's placement granularity
// actually stops, which is the number the narrow-row API can promise. The summary line at the
// end reports that boundary directly rather than leaving it to be read out of the assertions.
TEST_F(QuasarNarrowRowUntilize, SubFaceWidths) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/2);
    std::vector<std::uint32_t> ok_widths, bad_widths;
    for (std::uint32_t last_tile_w : {1u, 2u, 3u, 4u, 12u, 20u}) {
        const bool ok = run_narrow_row(devices_[0], buffers, {.ct_dim = 2, .last_tile_w = last_tile_w});
        (ok ? ok_widths : bad_widths).push_back(last_tile_w);
        EXPECT_TRUE(ok) << "last_tile_w=" << last_tile_w;
    }
    auto join = [](const std::vector<std::uint32_t>& v) {
        std::string s;
        for (auto x : v) {
            s += (s.empty() ? "" : ",") + std::to_string(x);
        }
        return s.empty() ? std::string("none") : s;
    };
    log_info(
        tt::LogTest,
        "sub-face widths (RV_PACR cannot do any of these): placed correctly = {}; failed = {}",
        join(ok_widths),
        join(bad_widths));
}

// THE DECISIVE TEST: is the iDMA compaction actually better than the workaround it replaces?
//
// Three shapes chosen around the measured knee. The first run of this test found the
// compaction is DESCRIPTOR-bound and flat at ~8.3 cyc/row up to ~80 B/row, then costs roughly
// one cycle per 16-20 B (one iDMA VC). So the engines have to be compared on both sides of
// that, because below it they are all paying for issue and above it they are paying for bytes:
//
//   A  ct_dim 1, w 16   ->   32 B/row   well below the knee, pure descriptor cost
//   B  ct_dim 4, w 16   ->  224 B/row   above the knee
//   C  ct_dim 8, w 28   ->  504 B/row   == the 32x252 matrix, ~6x past the knee
//
// All three engines at one channel, plus the two iDMA engines at eight channels on the shapes
// where bytes dominate -- comparing a knowingly bandwidth-starved iDMA against the NOC would
// understate it. The NOC path has no equivalent knob, so its single row is the whole story.
//
// This body only proves every engine produces the same correct output; the cycles come from
// report_narrow_row_from_csv.py, whose "vs workaround" table groups these by shape.
TEST_F(QuasarNarrowRowUntilize, EngineComparison) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/8);
    struct Shape {
        std::uint32_t ct_dim;
        std::uint32_t last_tile_w;
        bool fan_out;  // only worth it where the row is long enough to be data-bound
    };
    for (const Shape& s : {Shape{1, 16, false}, Shape{4, 16, true}, Shape{8, LAST_W_252, true}}) {
        for (std::uint32_t engine : {ENGINE_IDMA_SCATTER, ENGINE_IDMA_PER_ROW, ENGINE_NOC_PER_ROW}) {
            EXPECT_TRUE(run_narrow_row(
                devices_[0], buffers, {.ct_dim = s.ct_dim, .last_tile_w = s.last_tile_w, .engine_mode = engine}))
                << "engine=" << engine_name(engine) << " ct_dim=" << s.ct_dim;
            // Fan-out is an iDMA-only knob, and only pays past the knee.
            if (s.fan_out && engine != ENGINE_NOC_PER_ROW) {
                EXPECT_TRUE(run_narrow_row(
                    devices_[0],
                    buffers,
                    {.ct_dim = s.ct_dim,
                     .last_tile_w = s.last_tile_w,
                     .engine_mode = engine,
                     .num_channels = CHANNELS_ALL}))
                    << "engine=" << engine_name(engine) << " ct_dim=" << s.ct_dim << " ch=" << CHANNELS_ALL;
            }
        }
    }
}

// Does sub-splitting a row into packets help fan-out, or just cost issue?
//
// Fan-out round-robins PACKETS, so it needs at least num_channels of them to distribute. The
// first version of this test assumed that meant splitting each row into `channels` packets --
// which is right when the whole transfer is ONE packet, and wrong here. Both iDMA engines
// already emit one packet per ROW (the scatter list because each entry is its own transfer,
// the per-row loop because each row is its own transaction), so with 32 rows and at most 8
// channels there are always enough packets and the default `max_packet_bytes = out_row_bytes`
// leaves them at their natural size.
//
// Sub-splitting on top of that multiplies the packet count by `channels` while the bytes stay
// the same, and every packet still costs ~4-6 cycles to issue. Prediction: natural >= split
// everywhere, and materially better on the long shape. This test measures that instead of
// assuming it -- the old policy would have turned 504 B rows into 64 B packets, 256 of them.
TEST_F(QuasarNarrowRowUntilize, ChannelSweep) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/8);
    // Below the knee and well past it, so the answer is not read off a single size.
    for (auto [ct_dim, last_tile_w] :
         {std::pair<std::uint32_t, std::uint32_t>{1, 16}, std::pair<std::uint32_t, std::uint32_t>{8, LAST_W_252}}) {
        const std::uint32_t row_bytes = ((ct_dim - 1) * TILE_W + last_tile_w) * DATUM_BYTES;

        // Reference: one channel, natural packets.
        EXPECT_TRUE(run_narrow_row(devices_[0], buffers, {.ct_dim = ct_dim, .last_tile_w = last_tile_w}))
            << "ct_dim=" << ct_dim << " ch=1";

        // Fan-out over the natural one-packet-per-row granularity. Expected best.
        EXPECT_TRUE(run_narrow_row(
            devices_[0], buffers, {.ct_dim = ct_dim, .last_tile_w = last_tile_w, .num_channels = CHANNELS_ALL}))
            << "ct_dim=" << ct_dim << " ch=" << CHANNELS_ALL << " natural packets";

        // Fan-out with each row sub-split -- exactly the old policy, reproduced so the two can
        // be compared rather than argued about. A "split" that is not smaller than the row is
        // no split at all, which is what rules out the short shape (32 B row, floor 64 B).
        const std::uint32_t split = std::max(MIN_SPLIT_PACKET_BYTES, row_bytes / CHANNELS_ALL);
        if (split < row_bytes) {
            EXPECT_TRUE(run_narrow_row(
                devices_[0],
                buffers,
                {.ct_dim = ct_dim,
                 .last_tile_w = last_tile_w,
                 .num_channels = CHANNELS_ALL,
                 .max_packet_bytes = split}))
                << "ct_dim=" << ct_dim << " ch=" << CHANNELS_ALL << " split packets of " << split << " B";
        }
    }
}

}  // namespace tt::tt_metal
