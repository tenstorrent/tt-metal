// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal standalone repro for a Wormhole-B0-only LLK bug uncovered in the
// fused-Welford fp32 layernorm path.
//
// The bug: when fp32_dest_acc_en=true and DstSync::SyncHalf is used (block
// size = 4 tiles in dst), a binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCB>
// reading a CB whose primary buffer index is in Default (Tf32) unpack mode
// silently produces wrong output IF some prior copy_tile took the
// UnpackToDestFp32 path through a SECOND buffer index that aliases the SAME
// L1 allocation as the primary index. On Blackhole the same code is correct.
//
// This test runs the same compute kernel twice with the same data and the
// same expected math, differing only in the CB layout:
//
//   Case A (control): cb_primary and cb_alias have SEPARATE L1 allocations.
//                     Expected to pass on both WH and BH.
//
//   Case B (repro):   cb_primary and cb_alias share ONE L1 allocation via two
//                     CBFormatDescriptors on a single CBDescriptor. Expected
//                     to fail on WH (striped pattern, see below) and pass on
//                     BH.
//
// Expected failure shape on WH for Case B (from the originating layernorm
// regression): output of even-indexed SyncHalf blocks scales by ~(1+b)/b
// instead of just b (i.e. (1+rsqrt)*(x-bcast) instead of rsqrt*(x-bcast));
// output of odd-indexed blocks is mostly zero, sometimes with the first tile
// of the block carrying data.

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_HW = TILE_H * TILE_W;
constexpr uint32_t FP32_TILE_BYTES = TILE_HW * 4;

// Test parameters: block_size = 4 (forced by fp32_dest_acc on WH); n_blocks=4
// gives us two even and two odd SyncHalf blocks so the alternating-block
// pattern is unambiguous.
constexpr uint32_t kBlockSize = 4;
constexpr uint32_t kNBlocks = 4;
constexpr uint32_t kNTiles = kBlockSize * kNBlocks;

// Build a tile (row-major within the 32x32 logical tile) filled with the
// per-element function fn(row, col). Output is in physical tile layout
// (4 16x16 faces row-major within tile, row-major within face).
std::vector<float> make_tile(const std::function<float(uint32_t, uint32_t)>& fn) {
    std::vector<float> tile(TILE_HW, 0.0f);
    // Physical tile layout: face[fy][fx] occupies a 16x16 block. Row-major
    // face order: (0,0), (0,1), (1,0), (1,1). Within face: row-major.
    constexpr uint32_t FACE = 16;
    for (uint32_t fy = 0; fy < 2; ++fy) {
        for (uint32_t fx = 0; fx < 2; ++fx) {
            uint32_t face_base = (fy * 2 + fx) * FACE * FACE;
            for (uint32_t r = 0; r < FACE; ++r) {
                for (uint32_t c = 0; c < FACE; ++c) {
                    uint32_t logical_row = fy * FACE + r;
                    uint32_t logical_col = fx * FACE + c;
                    tile[face_base + r * FACE + c] = fn(logical_row, logical_col);
                }
            }
        }
    }
    return tile;
}

// Concatenate n tile buffers into one flat fp32 buffer.
std::vector<float> concat_tiles(const std::vector<std::vector<float>>& tiles) {
    std::vector<float> out;
    out.reserve(tiles.size() * TILE_HW);
    for (const auto& t : tiles) {
        out.insert(out.end(), t.begin(), t.end());
    }
    return out;
}

// Run one case. Returns true if the output matches expected within tol.
// `case_name` is "A" or "B". `alias_shares_allocation` controls whether the
// alias buffer index sits on the primary's CBDescriptor (Case B, shared) or
// on its own CBDescriptor (Case A, separate).
bool run_case(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const std::string& case_name,
    bool alias_shares_allocation) {
    fmt::print(
        "\n==== Case {} ({}) ====\n",
        case_name,
        alias_shares_allocation ? "multi-buffer-index alias on shared L1" : "no aliasing, separate L1 allocations");

    // ------------------------------------------------------------------
    // Inputs
    // ------------------------------------------------------------------
    // x[t](r, c) = (t + 1) + 0.01 * r + 0.1 * c  — distinct per tile,
    // row, and column so any cross-tile / cross-block leakage is visible.
    std::vector<std::vector<float>> x_tiles;
    x_tiles.reserve(kNTiles);
    for (uint32_t t = 0; t < kNTiles; ++t) {
        x_tiles.push_back(make_tile([t](uint32_t r, uint32_t c) {
            return static_cast<float>(t + 1) + 0.01f * static_cast<float>(r) + 0.1f * static_cast<float>(c);
        }));
    }
    auto x_data = concat_tiles(x_tiles);

    // bcast_a column-broadcast: column 0 holds the value; other columns
    // ignored by sub_tiles_bcast_cols.
    auto bcast_a_tile = make_tile([](uint32_t r, uint32_t /*c*/) { return 2.0f + 0.001f * static_cast<float>(r); });

    // bcast_b column-broadcast: pick a value clearly distinguishable from 1 so
    // the (1+b)/b vs. 1 ratio between buggy and correct output stands out.
    // b = 0.7 gives ratio (1+0.7)/0.7 = 2.4286 in the buggy case.
    auto bcast_b_tile = make_tile([](uint32_t r, uint32_t /*c*/) { return 0.7f + 0.001f * static_cast<float>(r); });

    // ------------------------------------------------------------------
    // Buffers
    // ------------------------------------------------------------------
    distributed::DeviceLocalBufferConfig dram_cfg{
        .page_size = FP32_TILE_BYTES,
        .buffer_type = BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig x_buf_cfg{.size = FP32_TILE_BYTES * kNTiles};
    distributed::ReplicatedBufferConfig tile_buf_cfg{.size = FP32_TILE_BYTES};

    auto x_buffer = distributed::MeshBuffer::create(x_buf_cfg, dram_cfg, mesh_device.get());
    auto a_buffer = distributed::MeshBuffer::create(tile_buf_cfg, dram_cfg, mesh_device.get());
    auto b_buffer = distributed::MeshBuffer::create(tile_buf_cfg, dram_cfg, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(x_buf_cfg, dram_cfg, mesh_device.get());

    distributed::EnqueueWriteMeshBuffer(cq, x_buffer, x_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, a_buffer, bcast_a_tile, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buffer, bcast_b_tile, false);

    // ------------------------------------------------------------------
    // ProgramDescriptor
    // ------------------------------------------------------------------
    constexpr CoreCoord core = {0, 0};
    CoreRangeSet core_set{CoreRange{core}};

    ProgramDescriptor pd;

    // CB 0: cb_primary (Float32, Default unpack mode). Holds block_size tiles
    // double-buffered (2 * block_size total) so reader/compute can overlap.
    {
        CBDescriptor cb;
        cb.total_size = 2 * kBlockSize * FP32_TILE_BYTES;
        cb.core_ranges = core_set;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = tt::DataFormat::Float32,
            .page_size = FP32_TILE_BYTES,
        });
        if (alias_shares_allocation) {
            // Case B: the alias index sits on the SAME CBDescriptor as the
            // primary. The two buffer indices share one L1 allocation but
            // have distinct producer/consumer counters.
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_29,
                .data_format = tt::DataFormat::Float32,
                .page_size = FP32_TILE_BYTES,
            });
        }
        pd.cbs.push_back(std::move(cb));
    }

    // CB 29: cb_alias as a SEPARATE allocation. Only for Case A.
    if (!alias_shares_allocation) {
        CBDescriptor cb;
        cb.total_size = 2 * kBlockSize * FP32_TILE_BYTES;
        cb.core_ranges = core_set;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_29,
            .data_format = tt::DataFormat::Float32,
            .page_size = FP32_TILE_BYTES,
        });
        pd.cbs.push_back(std::move(cb));
    }

    // CB 2: bcast_a
    {
        CBDescriptor cb;
        cb.total_size = FP32_TILE_BYTES;
        cb.core_ranges = core_set;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = tt::DataFormat::Float32,
            .page_size = FP32_TILE_BYTES,
        });
        pd.cbs.push_back(std::move(cb));
    }

    // CB 3: bcast_b
    {
        CBDescriptor cb;
        cb.total_size = FP32_TILE_BYTES;
        cb.core_ranges = core_set;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = tt::DataFormat::Float32,
            .page_size = FP32_TILE_BYTES,
        });
        pd.cbs.push_back(std::move(cb));
    }

    // CB 16: cb_out
    {
        CBDescriptor cb;
        cb.total_size = 2 * kBlockSize * FP32_TILE_BYTES;
        cb.core_ranges = core_set;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16,
            .data_format = tt::DataFormat::Float32,
            .page_size = FP32_TILE_BYTES,
        });
        pd.cbs.push_back(std::move(cb));
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------
    // Reader compile-time args: three TensorAccessorArgs (x, a, b).
    std::vector<uint32_t> reader_cta;
    TensorAccessorArgs(*x_buffer).append_to(reader_cta);
    TensorAccessorArgs(*a_buffer).append_to(reader_cta);
    TensorAccessorArgs(*b_buffer).append_to(reader_cta);

    std::vector<uint32_t> writer_cta;
    TensorAccessorArgs(*out_buffer).append_to(writer_cta);

    KernelDescriptor reader_kd;
    reader_kd.kernel_source = OVERRIDE_KERNEL_PREFIX "wh_alias_dest_reuse_repro/kernels/dataflow/reader.cpp";
    reader_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kd.core_ranges = core_set;
    reader_kd.compile_time_args = reader_cta;
    reader_kd.config = ReaderConfigDescriptor{};
    reader_kd.runtime_args.emplace_back(
        core,
        std::vector<uint32_t>{
            static_cast<uint32_t>(x_buffer->address()),
            static_cast<uint32_t>(a_buffer->address()),
            static_cast<uint32_t>(b_buffer->address()),
            kNBlocks,
            kBlockSize,
            alias_shares_allocation ? 0u : 1u,
        });
    pd.kernels.push_back(std::move(reader_kd));

    KernelDescriptor writer_kd;
    writer_kd.kernel_source = OVERRIDE_KERNEL_PREFIX "wh_alias_dest_reuse_repro/kernels/dataflow/writer.cpp";
    writer_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kd.core_ranges = core_set;
    writer_kd.compile_time_args = writer_cta;
    writer_kd.config = WriterConfigDescriptor{};
    writer_kd.runtime_args.emplace_back(
        core,
        std::vector<uint32_t>{
            static_cast<uint32_t>(out_buffer->address()),
            kNTiles,
        });
    pd.kernels.push_back(std::move(writer_kd));

    // Compute kernel with the targeted UnpackToDestMode configuration.
    // unpack_to_dest_mode[c_0] = Default, unpack_to_dest_mode[c_29] = UnpackToDestFp32.
    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_29)] = UnpackToDestMode::UnpackToDestFp32;

    KernelDescriptor compute_kd;
    compute_kd.kernel_source = OVERRIDE_KERNEL_PREFIX "wh_alias_dest_reuse_repro/kernels/compute/alias_dest_reuse.cpp";
    compute_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kd.core_ranges = core_set;
    compute_kd.compile_time_args = std::vector<uint32_t>{kNBlocks, kBlockSize};
    compute_kd.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,  // SyncHalf
        .unpack_to_dest_mode = std::move(unpack_modes),
        .math_approx_mode = false,
    };
    compute_kd.runtime_args.emplace_back(core, std::vector<uint32_t>{});
    pd.kernels.push_back(std::move(compute_kd));

    // ------------------------------------------------------------------
    // Build, enqueue, read back
    // ------------------------------------------------------------------
    Program program{pd};

    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<float> result;
    distributed::EnqueueReadMeshBuffer(cq, result, out_buffer, true);

    // ------------------------------------------------------------------
    // Verify
    // ------------------------------------------------------------------
    // Expected: out[t](r, c) = (x[t](r, c) - bcast_a(r, 0)) * bcast_b(r, 0)
    // (column 0 of the bcast tiles is what sub_/mul_tiles_bcast_cols uses).
    auto val_at = [](const std::vector<float>& flat, uint32_t tile_idx, uint32_t logical_row, uint32_t logical_col) {
        constexpr uint32_t FACE = 16;
        uint32_t fy = logical_row / FACE;
        uint32_t fx = logical_col / FACE;
        uint32_t r = logical_row % FACE;
        uint32_t c = logical_col % FACE;
        uint32_t face_base = (fy * 2 + fx) * FACE * FACE;
        return flat[tile_idx * TILE_HW + face_base + r * FACE + c];
    };

    // Per-element diffs are reported below for diagnostic visibility, but they are
    // NOT used to decide PASS/FAIL. On a correct arch the dest-reuse ELWMUL reads
    // cb_bcast_b via SrcA (TF32-quantized), and the preceding sub_tiles_bcast_cols
    // does the same for cb_in / cb_bcast_a. Both stages accumulate ~0.1% relative
    // rounding noise that, when (x - a) gets close to zero, can produce diffs of a
    // few milli-units on otherwise correct output. Those are not bug signatures.
    // The bug we're after produces structural failures: tiles that go to zero, or
    // tiles that come out 2x+ over-scaled (the accumulation signature). Both are
    // counted below and drive the verdict.
    constexpr float mismatch_print_tol = 1e-3f;
    uint32_t printed_mismatches = 0;

    // Per-tile ratio summary: pick the element at (row=0, col=0) of each tile
    // (face 0) and report actual/expected. For Case B on WH we expect:
    //   - even-indexed blocks (block 0, 2): ratio ~= (1+b)/b = ~2.4286
    //   - odd-indexed blocks (block 1, 3): actual ~= 0 (most tiles)
    fmt::print("Per-tile sample (row=0,col=0): expected vs actual (ratio = actual/expected)\n");
    fmt::print("  tile | block | expected | actual    | ratio\n");
    fmt::print("  -----+-------+----------+-----------+-------\n");

    uint32_t num_zero_tiles = 0;
    uint32_t num_overscaled_tiles = 0;
    for (uint32_t t = 0; t < kNTiles; ++t) {
        uint32_t block = t / kBlockSize;
        float x00 = static_cast<float>(t + 1);  // x[t](0, 0)
        float a00 = 2.0f;                       // bcast_a column 0 at row 0
        float b00 = 0.7f;                       // bcast_b column 0 at row 0
        float expected = (x00 - a00) * b00;
        float actual = val_at(result, t, 0, 0);
        float ratio = (std::abs(expected) > 1e-12f) ? (actual / expected) : 0.0f;
        fmt::print("  {:>4} | {:>5} | {:>+8.4f} | {:>+9.4f} | {:>+6.4f}\n", t, block, expected, actual, ratio);

        if (std::abs(actual) < 1e-4f && std::abs(expected) > 0.1f) {
            ++num_zero_tiles;
        }
        if (std::abs(expected) > 0.1f && ratio > 2.0f) {
            ++num_overscaled_tiles;
        }
    }

    // Full-element check.
    for (uint32_t t = 0; t < kNTiles; ++t) {
        for (uint32_t r = 0; r < TILE_H; ++r) {
            for (uint32_t c = 0; c < TILE_W; ++c) {
                float x_v = static_cast<float>(t + 1) + 0.01f * static_cast<float>(r) + 0.1f * static_cast<float>(c);
                float a_v = 2.0f + 0.001f * static_cast<float>(r);
                float b_v = 0.7f + 0.001f * static_cast<float>(r);
                float expected = (x_v - a_v) * b_v;
                float actual = val_at(result, t, r, c);
                if (std::abs(expected - actual) > mismatch_print_tol && printed_mismatches < 8) {
                    fmt::print(
                        "  mismatch (informational, not part of verdict): tile={} row={} col={} expected={:.6f} "
                        "actual={:.6f}\n",
                        t,
                        r,
                        c,
                        expected,
                        actual);
                    ++printed_mismatches;
                }
            }
        }
    }

    fmt::print(
        "Summary: zero-output tiles={}, over-scaled (>2x) tiles={}, total={}\n",
        num_zero_tiles,
        num_overscaled_tiles,
        kNTiles);

    // Verdict is driven purely by the structural metrics. Per-element diffs above
    // are reported for diagnostic visibility but excluded from the verdict because
    // a correct arch still produces ~0.1-0.3% TF32 rounding (the dest-reuse mul
    // reads cb_bcast_b via SrcA, masking fp32 to TF32). The bug we're after
    // produces zero-output tiles and ~2.4x over-scaled tiles -- both structural.
    bool structural_pass = (num_zero_tiles == 0) && (num_overscaled_tiles == 0);
    fmt::print("Case {} result: {}\n", case_name, structural_pass ? "PASS" : "FAIL");

    // On failure, dump comprehensive diagnostics so we don't need a re-run to see
    // what happened.
    if (!structural_pass) {
        fmt::print("\n--- FAILURE DIAGNOSTICS (Case {}) ---\n", case_name);

        // Per-tile classification of the (0,0) element. Helps spot the alternating
        // block pattern characteristic of the WH bug (correct/zero per even block,
        // all over-scaled per odd block, or vice versa).
        fmt::print("Per-tile classification at (row=0, col=0):\n");
        fmt::print("  tile | block | expected | actual    | ratio    | class\n");
        fmt::print("  -----+-------+----------+-----------+----------+------------\n");
        for (uint32_t t = 0; t < kNTiles; ++t) {
            uint32_t block = t / kBlockSize;
            float x00 = static_cast<float>(t + 1);
            float a00 = 2.0f;
            float b00 = 0.7f;
            float expected = (x00 - a00) * b00;
            float actual = val_at(result, t, 0, 0);
            float ratio = (std::abs(expected) > 1e-12f) ? (actual / expected) : 0.0f;
            const char* cls = "ok";
            if (std::abs(actual) < 1e-4f && std::abs(expected) > 0.1f) {
                cls = "ZERO";
            } else if (std::abs(expected) > 0.1f && ratio > 2.0f) {
                cls = "OVER-SCALED";
            } else if (std::abs(expected) > 0.1f && ratio < 0.5f && ratio > -0.5f) {
                cls = "UNDER-SCALED";
            } else if (std::abs(expected) > 0.1f && (ratio < 0.95f || ratio > 1.05f)) {
                cls = "off";
            }
            fmt::print(
                "  {:>4} | {:>5} | {:>+8.4f} | {:>+9.4f} | {:>+8.4f} | {}\n", t, block, expected, actual, ratio, cls);
        }

        // Full-block statistics: how many ZERO and OVER-SCALED across all elements,
        // grouped by SyncHalf block (block_size=4 with fp32_dest_acc on WH).
        fmt::print("\nPer-block element-level statistics:\n");
        fmt::print("  block | zero-elem | over-scaled-elem | total-bad / total-elem\n");
        fmt::print("  ------+-----------+------------------+-----------------------\n");
        uint32_t total_zero_elem = 0;
        uint32_t total_over_elem = 0;
        for (uint32_t blk = 0; blk < kNTiles / kBlockSize; ++blk) {
            uint32_t zero_elem = 0;
            uint32_t over_elem = 0;
            uint32_t total_elem = 0;
            for (uint32_t t = blk * kBlockSize; t < (blk + 1) * kBlockSize; ++t) {
                for (uint32_t r = 0; r < TILE_H; ++r) {
                    for (uint32_t c = 0; c < TILE_W; ++c) {
                        float x_v =
                            static_cast<float>(t + 1) + 0.01f * static_cast<float>(r) + 0.1f * static_cast<float>(c);
                        float a_v = 2.0f + 0.001f * static_cast<float>(r);
                        float b_v = 0.7f + 0.001f * static_cast<float>(r);
                        float expected = (x_v - a_v) * b_v;
                        float actual = val_at(result, t, r, c);
                        if (std::abs(expected) < 0.1f) {
                            continue;
                        }
                        ++total_elem;
                        float ratio = actual / expected;
                        if (std::abs(actual) < 1e-4f) {
                            ++zero_elem;
                        } else if (ratio > 2.0f) {
                            ++over_elem;
                        }
                    }
                }
            }
            total_zero_elem += zero_elem;
            total_over_elem += over_elem;
            fmt::print(
                "  {:>5} | {:>9} | {:>16} | {:>8} / {:>8}\n",
                blk,
                zero_elem,
                over_elem,
                zero_elem + over_elem,
                total_elem);
        }
        fmt::print(
            "  total |   {:>7} |     {:>12} |   {:>6} / {:>8}\n",
            total_zero_elem,
            total_over_elem,
            total_zero_elem + total_over_elem,
            kNTiles * TILE_HW);

        // Ratio histogram across all non-trivial expected elements. Helps see the
        // dominant ratio modes ("clusters near 2.4x and near 0" = striped bug;
        // "single cluster near 1" = correct).
        fmt::print("\nRatio histogram (only elements with |expected| > 0.1):\n");
        constexpr int kNBins = 9;
        const std::array<float, kNBins + 1> bin_edges = {
            -10.0f, -0.1f, 0.001f, 0.5f, 0.95f, 1.05f, 1.5f, 2.0f, 3.0f, 100.0f};
        const std::array<const char*, kNBins> bin_labels = {
            "<-0.1   ",
            "-0.1..0 ",
            "0..0.5  ",
            "0.5..0.95",
            "0.95..1.05",
            "1.05..1.5",
            "1.5..2.0",
            "2.0..3.0",
            ">3.0    "};
        std::array<uint32_t, kNBins> hist{};
        for (uint32_t t = 0; t < kNTiles; ++t) {
            for (uint32_t r = 0; r < TILE_H; ++r) {
                for (uint32_t c = 0; c < TILE_W; ++c) {
                    float x_v =
                        static_cast<float>(t + 1) + 0.01f * static_cast<float>(r) + 0.1f * static_cast<float>(c);
                    float a_v = 2.0f + 0.001f * static_cast<float>(r);
                    float b_v = 0.7f + 0.001f * static_cast<float>(r);
                    float expected = (x_v - a_v) * b_v;
                    if (std::abs(expected) < 0.1f) {
                        continue;
                    }
                    float actual = val_at(result, t, r, c);
                    float ratio = actual / expected;
                    for (int b = 0; b < kNBins; ++b) {
                        if (ratio >= bin_edges[b] && ratio < bin_edges[b + 1]) {
                            ++hist[b];
                            break;
                        }
                    }
                }
            }
        }
        for (int b = 0; b < kNBins; ++b) {
            fmt::print("  ratio in [{}]: {} elements\n", bin_labels[b], hist[b]);
        }
        fmt::print("--- end FAILURE DIAGNOSTICS (Case {}) ---\n\n", case_name);
    }

    return structural_pass;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    auto& cq = mesh_device->mesh_command_queue();

    bool case_a_pass = false;
    bool case_b_pass = false;
    try {
        case_a_pass = run_case(mesh_device, cq, "A", /*alias_shares_allocation=*/false);
        case_b_pass = run_case(mesh_device, cq, "B", /*alias_shares_allocation=*/true);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception: {}\n", e.what());
        mesh_device->close();
        return 2;
    }

    mesh_device->close();

    fmt::print("\n===== Overall =====\n");
    fmt::print("  Case A (no aliasing, separate L1 allocs): {}\n", case_a_pass ? "PASS" : "FAIL");
    fmt::print("  Case B (multi-buffer-index alias, shared L1): {}\n", case_b_pass ? "PASS" : "FAIL");
    if (!case_a_pass && !case_b_pass) {
        fmt::print(
            "\nBoth cases fail with the same striped pattern -- aliasing is NOT a required\n"
            "trigger condition. The bug is purely about the UnpackToDest-fp32-read +\n"
            "binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCB> + fp32_dest_acc_en=true sequence.\n"
            "Expected on Wormhole-B0. On Blackhole both cases should pass.\n");
    } else if (case_a_pass && case_b_pass) {
        fmt::print(
            "\nBoth cases pass. This is the expected outcome on Blackhole and on any arch\n"
            "where the bug has been fixed. On Wormhole-B0 a PASS here would mean the\n"
            "repro no longer triggers -- check that fp32_dest_acc_en=true is plumbed\n"
            "correctly into the compute kernel config.\n");
    } else if (case_a_pass && !case_b_pass) {
        fmt::print(
            "\nOnly Case B fails. This would mean multi-buffer-index aliasing IS a required\n"
            "trigger condition on this arch -- different from what we observed on WH.\n");
    } else {
        fmt::print(
            "\nOnly Case A fails. Unusual -- aliasing somehow masks the bug. See per-tile\n"
            "output above.\n");
    }
    return (case_a_pass && case_b_pass) ? 0 : 1;
}
