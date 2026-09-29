// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar narrow-row pack-untilize, end to end (test id 918).
//
// THE PROBLEM. `_llk_pack_untilize_` cannot produce an untilized row narrower than a whole
// number of tiles on Quasar. Its output stride is face-granular and the compute API exposes
// it only in whole tiles (api/compute/pack_untilize.h static_asserts narrow_row == false for
// ARCH_QUASAR); Wormhole/Blackhole had a 16-byte output stride register, Quasar does not. So
// untilizing a matrix whose width is not a multiple of 32 leaves junk at the end of every
// row, and the workaround is for the consumer's NOC to issue a separate read per row to step
// over it -- one transaction per row, where one would do.
//
// WHAT THIS TEST DOES. Two stages in one program:
//
//   stage 1  stock whole-tile HW pack_untilize  ->  32 rows x (ct_dim*32) datums, padded
//   stage 2  an iDMA gather                     ->  32 rows x matrix_w datums, dense
//
// with matrix_w = (ct_dim - 1) * 32 + last_tile_w. The output is the dense narrow-row matrix
// the RV_PACR per-face-row path produces, and the host checks it datum for datum against a
// golden plus a guard band, for every shape and both engines.
//
// `EngineParity` runs the same shape through the iDMA gather and through the NOC-read-per-row
// workaround and requires both to be correct, which is what makes the performance comparison
// in README.md a like-for-like one.

#include <cstdint>
#include <string>
#include <vector>
#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "kernels/narrow_row_engine_mode.hpp"

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_narrow_row {

constexpr std::uint32_t TILE_H = 32;
constexpr std::uint32_t TILE_W = 32;
constexpr std::uint32_t FACE_DIM = 16;
constexpr std::uint32_t DATUM_BYTES = 2;  // Float16_b
constexpr std::uint32_t TILE_BYTES = TILE_H * TILE_W * DATUM_BYTES;
constexpr std::uint32_t OUT_ROWS = TILE_H;

// The engine_mode wire values live in the kernel-side header so the two ends of the
// runtime arg cannot drift apart.
using narrow_row::EngineMode;

// Fan out unconditionally: at 8 channels the gather is never more than 0.4% behind one
// channel on short rows, and 3.6x ahead at 512 B/row, where a single channel is data-bound at
// one VC's 16 B/cycle and loses to the workaround it replaces. CHANNELS_ALL is a sentinel the
// kernel resolves against the real VC count, so the 8 is not repeated here.
using narrow_row::CHANNELS_ALL;
using narrow_row::CHANNELS_MAX;

// The reference workload: a 32 x 252 Float16_b matrix. 252 datums needs 8 tiles to cover
// (7 x 32 = 224, + 28), so ct_dim 8 -- the half-sync 16-bit DEST limit for pack_untilize --
// keeping 28 datums of the last tile. Padded row 256 datums, dense row 252.
constexpr std::uint32_t LAST_W_252 = 28;

constexpr CoreCoord CORE = {0, 0};

// Guard band after the dense output: a gather that writes past its row must not go unnoticed.
constexpr std::uint32_t GUARD_BYTES = 256;
constexpr std::uint32_t GUARD_FILL = 0xA5A5A5A5;
// Fills the DRAM input past the tiles a run uses. The reader only streams ct_dim tiles, so
// any of this reaching the output is a bug worth seeing rather than a plausible-looking zero.
constexpr std::uint32_t SRC_PAD_FILL = 0xDEADBEEF;

// Switched rather than ternary so that adding an engine is a -Wswitch warning here
// instead of a run silently mislabelled as the NOC one in a failure message.
// CHANNELS_ALL is a sentinel resolved on the device, so report it as a word: printing the
// raw 0 would read as "zero channels" in a failure message.
std::string channel_label(std::uint32_t n) { return n == CHANNELS_ALL ? "all" : std::to_string(n); }

const char* engine_name(EngineMode e) {
    switch (e) {
        case EngineMode::IdmaPerRow: return "iDMA gather";
        case EngineMode::NocPerRow: return "NOC per-row";
    }
    return "unknown";
}

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
// would come back as zero and read as a data-movement bug. idx < 32 * 256 = 8192 fits the 13
// bits exactly at the widest shape tested (ct_dim 8).
inline std::uint16_t datum_at(std::uint32_t r, std::uint32_t c, std::uint32_t pad_w) {
    return static_cast<std::uint16_t>(0x4000u | ((r * pad_w + c) & 0x1FFFu));
}

// Tilized stimulus for a ct_dim-wide tile-row, in the layout the unpacker expects: per tile,
// four 16x16 faces in order top-left, top-right, bottom-left, bottom-right, each row-major,
// two datums per uint32 with the even datum in the low half. Matches gold_standard_tilize;
// hand-rolled so this test does not depend on the llk test target's translation units.
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
    std::uint32_t max_ct_dim = 0;                       // what the two above were sized for
};

Buffers make_buffers(const std::shared_ptr<distributed::MeshDevice>& mesh_device, std::uint32_t max_ct_dim) {
    auto mk = [&](std::uint32_t bytes, BufferType type) {
        // page_size == the whole buffer keeps it in a single bank, which is what the direct
        // reader (bank_id 0) and the L1 read-back both assume.
        distributed::DeviceLocalBufferConfig local{.page_size = bytes, .buffer_type = type};
        distributed::ReplicatedBufferConfig cfg{.size = bytes};
        return distributed::MeshBuffer::create(cfg, local, mesh_device.get());
    };
    const std::uint32_t max_out_bytes = OUT_ROWS * max_ct_dim * TILE_W * DATUM_BYTES + GUARD_BYTES;
    return {
        .src_dram = mk(max_ct_dim * TILE_BYTES, BufferType::DRAM),
        .out_l1 = mk(max_out_bytes, BufferType::L1),
        .max_ct_dim = max_ct_dim,
    };
}

struct RunConfig {
    std::uint32_t ct_dim = 1;
    std::uint32_t last_tile_w = 32;  // datums kept from the LAST tile; matrix_w derives from it
    EngineMode engine_mode = EngineMode::IdmaPerRow;
    std::uint32_t num_channels = CHANNELS_ALL;
};

// Builds and runs the two-stage program, then checks the dense output datum for datum.
// Returns true iff the narrow-row matrix is exactly right.
bool run_narrow_row(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Buffers& buffers, const RunConfig& cfg) {
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    // Outgrowing the buffers is silent otherwise: the stimulus resize() below would TRUNCATE
    // while the reader still streams cfg.ct_dim tiles, and the gather would write past out_l1
    // into neighbouring L1. Fail loudly instead.
    TT_FATAL(
        cfg.ct_dim <= buffers.max_ct_dim,
        "ct_dim {} exceeds the {} these buffers were sized for",
        cfg.ct_dim,
        buffers.max_ct_dim);
    // The other half of the shape. Above TILE_W, matrix_w exceeds pad_w and the gather reads
    // past the pad DFB and writes past out_l1; at 0, out_row_bytes is 0 and the kernel issues
    // a zero-length transfer that never acks, so it spins forever.
    TT_FATAL(
        cfg.last_tile_w >= 1 && cfg.last_tile_w <= TILE_W,
        "last_tile_w {} must be in [1, {}]",
        cfg.last_tile_w,
        TILE_W);

    const std::uint32_t ct_dim = cfg.ct_dim;
    const std::uint32_t pad_w = ct_dim * TILE_W;                             // padded row, datums
    const std::uint32_t matrix_w = (ct_dim - 1) * TILE_W + cfg.last_tile_w;  // dense row, datums
    const std::uint32_t pad_row_bytes = pad_w * DATUM_BYTES;
    const std::uint32_t out_row_bytes = matrix_w * DATUM_BYTES;
    const std::uint32_t out_bytes = OUT_ROWS * out_row_bytes;

    const std::uint32_t out_addr = buffers.out_l1->address();
    // Narrowed deliberately: the runtime-arg table is uint32_t, and the conversion inside
    // std::pair's constructor would otherwise be silent.
    const std::uint32_t src_addr = static_cast<std::uint32_t>(buffers.src_dram->address());

    // The NOC engine reads this core's own L1 (loopback), so it needs PHYSICAL noc coords --
    // logical {0,0} is physical (0,1) on the 1x3 emu.
    const CoreCoord physical_core =
        slow_dispatch::physical_device_from_unit_mesh(*mesh_device)->worker_core_from_logical_core(CORE);
    const std::uint32_t packed_coords = ((std::uint32_t)physical_core.x << 16) | (std::uint32_t)physical_core.y;

    // ---- stimulus -----------------------------------------------------------------------
    std::vector<std::uint32_t> src_vec = make_tilized_input(ct_dim);
    auto src_dram = buffers.src_dram;
    // EnqueueWriteMeshBuffer asserts the source FILLS the buffer, and the buffer is sized for
    // the widest shape the calling test body uses -- so a body that mixes ct_dim throws on its
    // first narrower run unless the tail is padded.
    src_vec.resize(src_dram->size() / sizeof(std::uint32_t), SRC_PAD_FILL);
    distributed::EnqueueWriteMeshBuffer(cq, src_dram, src_vec, /*blocking=*/true);

    // Prefill output + guard band, so "the gather never ran" and "the gather overran its row"
    // are both visible rather than passing on stale data.
    std::vector<std::uint32_t> out_init((out_bytes + GUARD_BYTES) / sizeof(std::uint32_t), GUARD_FILL);
    if (!slow_dispatch::WriteToL1(*mesh_device, CORE, out_addr, out_init)) {
        log_error(tt::LogTest, "ct_dim={}: failed to prefill the output buffer", ct_dim);
        return false;
    }

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
    // of them. Same convention the shipping pack_untilize consumers use.
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
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_narrow_row/kernels/narrow_row_untilize_compute.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(SRC_DFB, "src"), experimental::ProducerOf(PAD_DFB, "pad")},
        .compile_time_args = {{"ct_dim", ct_dim}, {"out_rows", OUT_ROWS}},
        .hw_config = experimental::ComputeHardwareConfig{},
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
                     "pad_row_bytes",
                     "out_row_bytes",
                     "num_rows",
                     "engine_mode",
                     "dest_coords",
                     "num_channels"},
            },
        .hw_config = experimental::DataMovementHardwareConfig{},
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
                 {"pad_row_bytes", pad_row_bytes},
                 {"out_row_bytes", out_row_bytes},
                 {"num_rows", OUT_ROWS},
                 {"engine_mode", static_cast<std::uint32_t>(cfg.engine_mode)},
                 {"dest_coords", packed_coords},
                 {"num_channels", cfg.num_channels}}),
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
    // placement SubFaceWidths probes.
    const std::uint32_t read_bytes = out_bytes + GUARD_BYTES;
    std::vector<std::uint32_t> out_words;
    const bool read_ok = slow_dispatch::ReadFromL1(*mesh_device, CORE, out_addr, read_bytes, out_words);
    // Guard the dereference below: a failed read leaves out_words empty, and data() would be
    // null. Checking the size also catches a short read, which would otherwise score the
    // untouched tail as a clean guard band.
    if (!read_ok || out_words.size() * sizeof(std::uint32_t) < read_bytes) {
        log_error(
            tt::LogTest,
            "ct_dim={} last_tile_w={}: L1 read-back returned {} B, expected {} B",
            ct_dim,
            cfg.last_tile_w,
            out_words.size() * sizeof(std::uint32_t),
            read_bytes);
        return false;
    }
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

    // Everything after the dense matrix must be untouched.
    std::uint32_t guard_bad = 0;
    for (std::uint32_t w = out_bytes / sizeof(std::uint32_t); w < out_words.size(); w++) {
        if (out_words[w] != GUARD_FILL) {
            guard_bad++;
        }
    }

    if (bad != 0 || guard_bad != 0) {
        // Only name a first wrong datum when there IS one: a guard-only failure means every
        // in-range datum was placed correctly and the gather overran, and printing
        // "row 0 col 0" from the never-assigned initialisers points at the wrong end of the
        // buffer.
        const std::string detail = bad != 0 ? fmt::format(
                                                  " (first at row {} col {}: expected 0x{:04x}, got 0x{:04x})",
                                                  first_bad_r,
                                                  first_bad_c,
                                                  first_bad_want,
                                                  first_bad_got)
                                            : std::string();
        log_error(
            tt::LogTest,
            "ct_dim={} last_tile_w={} matrix_w={} engine={} ch={}: {}/{} datums wrong{}, "
            "{} guard words clobbered",
            ct_dim,
            cfg.last_tile_w,
            matrix_w,
            engine_name(cfg.engine_mode),
            channel_label(cfg.num_channels),
            bad,
            OUT_ROWS * matrix_w,
            detail,
            guard_bad);
        return false;
    }
    return true;
}

}  // namespace unit_tests::dm::quasar_narrow_row

// =============================================================================
// Test Suite: Quasar narrow-row pack-untilize (HW untilize + iDMA compaction)
// =============================================================================

class QuasarNarrowRowUntilize : public QuasarMeshDeviceSingleCardFixture {};

// The layout contract across the shapes a real matrix has: full tiles at 32 datums each plus
// a narrow last tile. last_tile_w 32 is the degenerate whole-tile case (matrix_w == pad_w, so
// the gather becomes a straight copy), which is a good null check on the address math.
// ct_dim 8 is the half-sync 16-bit DEST limit for pack_untilize.
TEST_F(QuasarNarrowRowUntilize, WidthAndTileRowSweep) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/8);
    for (std::uint32_t ct_dim : {1u, 2u, 4u, 8u}) {
        for (std::uint32_t last_tile_w : {8u, 16u, 24u, 32u}) {
            EXPECT_TRUE(run_narrow_row(devices_[0], buffers, {.ct_dim = ct_dim, .last_tile_w = last_tile_w}))
                << "ct_dim=" << ct_dim << " last_tile_w=" << last_tile_w;
        }
    }
}

// Widths BELOW the RV_PACR floor. RV_PACR's output address is 16-byte granular, so for a
// 16-bit format it cannot place rows closer than 8 datums apart, and widths that are not a
// multiple of 8 make it write a full 16-datum face-row that spills into the next output row
// and has to be overwritten in a second pass. iDMA addresses L1 by the byte, so these are
// just a smaller out_row_bytes. 1 and 3 make matrix_w odd, so every other destination row
// starts at a 2 mod 4 byte offset, which probes byte- rather than word-granular placement.
TEST_F(QuasarNarrowRowUntilize, SubFaceWidths) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/2);
    // ct_dim 1 is the case that actually tests the claim: it produces 2 B to 40 B rows, below
    // RV_PACR's 16 B floor. At ct_dim 2 the leading full tile keeps every row at 66 B or
    // wider, which would let a reintroduced minimum width pass unnoticed.
    for (std::uint32_t ct_dim : {1u, 2u}) {
        for (std::uint32_t last_tile_w : {1u, 2u, 3u, 4u, 12u, 20u}) {
            EXPECT_TRUE(run_narrow_row(devices_[0], buffers, {.ct_dim = ct_dim, .last_tile_w = last_tile_w}))
                << "ct_dim=" << ct_dim << " last_tile_w=" << last_tile_w;
        }
    }
}

// The iDMA gather and the NOC-read-per-row workaround must produce identical, correct output
// on the same shape. That is what makes a timing comparison between them like-for-like: the
// baseline is the same operation, verified, not a straw man.
//
// The shapes are chosen so the comparison is not only over easy widths: 32 B and 504 B rows
// are 8-byte multiples, and 70 B (ct_dim 2, last_tile_w 3) puts every odd destination row at
// a 2 mod 4 offset. Quasar's NOC_L1_READ_ALIGNMENT_BYTES is 1, so the NOC arm can cover that
// and acts as a real cross-check on the iDMA placement rather than a redundant one.
//
// One channel is covered too. It is the configuration that must NOT be shipped -- data-bound
// past roughly 100 B/row, and slower than the workaround on long rows -- but it still has to
// be correct.
TEST_F(QuasarNarrowRowUntilize, EngineParity) {
    using namespace unit_tests::dm::quasar_narrow_row;
    if (should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    auto buffers = make_buffers(devices_[0], /*max_ct_dim=*/8);
    struct Shape {
        std::uint32_t ct_dim;
        std::uint32_t last_tile_w;
    };
    for (const Shape& sh : {Shape{1, 16}, Shape{2, 3}, Shape{8, LAST_W_252}}) {
        const RunConfig base{.ct_dim = sh.ct_dim, .last_tile_w = sh.last_tile_w};
        const std::uint32_t row_bytes = ((sh.ct_dim - 1) * TILE_W + sh.last_tile_w) * DATUM_BYTES;

        EXPECT_TRUE(run_narrow_row(devices_[0], buffers, base)) << "iDMA all channels, " << row_bytes << " B/row";

        RunConfig one_channel = base;
        one_channel.num_channels = 1;
        EXPECT_TRUE(run_narrow_row(devices_[0], buffers, one_channel)) << "iDMA 1ch, " << row_bytes << " B/row";

        RunConfig workaround = base;
        workaround.engine_mode = EngineMode::NocPerRow;
        EXPECT_TRUE(run_narrow_row(devices_[0], buffers, workaround)) << "NOC, " << row_bytes << " B/row";
    }

    // The clamp's over-range branch, which nothing else in the suite reaches: every other
    // iDMA run asks for CHANNELS_ALL or 1, both already inside the valid range. CHANNELS_MAX
    // + 1 is deliberately the exact boundary rather than some large value -- an off-by-one
    // clamp would pass it through and set req_end_vc one past the last VC, where a merely
    // huge request would still be caught. (The CHANNELS_ALL sentinel needs no case of its
    // own: every default iDMA run above sends it, and an unresolved 0 would underflow
    // req_end_vc to 0xFFFFFFFF.)
    const RunConfig over_range{.ct_dim = 8, .last_tile_w = LAST_W_252, .num_channels = CHANNELS_MAX + 1};
    EXPECT_TRUE(run_narrow_row(devices_[0], buffers, over_range)) << "iDMA over-range channel request";
}

}  // namespace tt::tt_metal
