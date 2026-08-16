// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// risc_scan_bench — Gate-2 deciding microbenchmark host driver.
//
// Measures the scalar-RISC (BRISC/NCRISC) scan/emit rate constant "X" that
// prices every RISC-side materialization candidate of the top-k selector
// campaign (see RADIX_BUCKET_GPU.md gate 2 and the storm/research reports).
//
// Single Tensix core (0,0). Input: one row of N = 65536 bf16 values staged
// into L1. Seven kernel variants x three input patterns; each variant is
// verified bit-exactly against a CPU reference and reports elapsed wall-clock
// cycles for its inner loop only. FAILs are loud and the process exits 1.
//
// Structure follows the sibling examples (loopback, add_2_integers_in_riscv):
// unit mesh, DRAM staging buffers, TensorAccessor compile-time args, one
// program per (variant, pattern) cell.

#include <fmt/core.h>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// ---------------------------------------------------------------------------
// Geometry (must match the kernel's compile-time args exactly)
// ---------------------------------------------------------------------------
constexpr uint32_t N_ELEMS = 65536;
constexpr uint32_t N_WORDS = N_ELEMS / 2;  // u32 words of packed bf16 pairs
constexpr uint32_t CHUNK_ELEMS = 32768;    // v5 stream chunk (u16 index span)
constexpr uint32_t STAGE_PAGE = 8192;
constexpr uint32_t IN_BYTES = N_ELEMS * 2;            // 131072
constexpr uint32_t IN_PAGES = IN_BYTES / STAGE_PAGE;  // 16
constexpr uint32_t SPARSE_BYTES = 294912;             // worst case: 2 x 1024 groups x 144 B
constexpr uint32_t SPARSE_WORDS = SPARSE_BYTES / 4;   // 73728
constexpr uint32_t EMIT_CAP = 4096;                   // emitted (value,index) pairs
constexpr uint32_t EMIT_BYTES = EMIT_CAP * 8;         // 32768
constexpr uint32_t RESULT_BYTES = 256;                // 64 u32 (2 RISC sections of 16)
constexpr uint32_t HIST_BYTES = 2048;                 // 2 x 256 u32 bins

constexpr uint32_t IN_OFF = 0;
constexpr uint32_t SPARSE_OFF = IN_OFF + IN_BYTES;        // 131072
constexpr uint32_t EMIT_OFF = SPARSE_OFF + SPARSE_BYTES;  // 425984
constexpr uint32_t RESULT_OFF = EMIT_OFF + EMIT_BYTES;    // 458752
constexpr uint32_t HIST_OFF = RESULT_OFF + RESULT_BYTES;  // 459008
constexpr uint32_t L1_BYTES = HIST_OFF + HIST_BYTES;      // 461056

enum Pattern : int { PAT_UNIFORM = 0, PAT_CLUSTERED = 1, PAT_ALL_EQUAL = 2 };
static const char* pattern_name(int p) {
    switch (p) {
        case PAT_UNIFORM: return "uniform";
        case PAT_CLUSTERED: return "clustered";
        default: return "all-equal";
    }
}
static const char* variant_name(int v) {
    switch (v) {
        case 1: return "v1 hist local-RAM";
        case 2: return "v2 hist L1";
        case 3: return "v3 hist 4-sub";
        case 4: return "v4 dense emit";
        case 5: return "v5 sparse consume";
        case 6: return "v6 dual-RISC hist";
        default: return "v7 load floor";
    }
}

// Sign-magnitude XOR map — identical formula in the kernel (v4).
static uint32_t map_key(uint16_t b) { return (uint32_t)b ^ (0x8000u + ((uint32_t)(b >> 15)) * 0x7FFFu); }

struct CaseData {
    std::vector<uint16_t> elems;         // N_ELEMS bf16 bit patterns
    std::vector<uint32_t> in_words;      // packed pairs, N_WORDS
    std::vector<uint32_t> sparse_words;  // full SPARSE_WORDS, zero padded
    uint32_t t_mapped = 0;
    uint32_t seg_groups[2] = {0, 0};
    uint32_t sparse_pages = 0;
    // references
    std::vector<uint32_t> hist_full;                       // 256 bins over [0, N)
    std::vector<uint32_t> hist_first;                      // over [0, N/2)
    std::vector<uint32_t> hist_second;                     // over [N/2, N)
    std::vector<std::pair<uint32_t, uint32_t>> survivors;  // (bits, global idx) in index order
    uint32_t surv_csum = 0;
    uint32_t byte_sum = 0;
    uint32_t hist_csum_full = 0, hist_csum_first = 0, hist_csum_second = 0;
};

static CaseData build_case(int pattern) {
    CaseData c;
    c.elems.resize(N_ELEMS);
    std::mt19937 rng(0x5EEDu + (uint32_t)pattern);
    for (uint32_t i = 0; i < N_ELEMS; i++) {
        switch (pattern) {
            case PAT_UNIFORM: c.elems[i] = (uint16_t)(rng() & 0xFFFFu); break;
            case PAT_CLUSTERED: c.elems[i] = (uint16_t)(0x3F00u | (rng() & 0xFFu)); break;  // one bin
            default: c.elems[i] = 0x3F80u; break;                                           // all equal (1.0f)
        }
    }
    c.in_words.resize(N_WORDS);
    for (uint32_t w = 0; w < N_WORDS; w++) {
        c.in_words[w] = (uint32_t)c.elems[2 * w] | ((uint32_t)c.elems[2 * w + 1] << 16);
    }

    // Threshold: ~512 survivors for uniform/clustered; ALL survive for
    // all-equal (exercises the emit-cap overflow path — count must stay exact).
    std::vector<uint32_t> keys(N_ELEMS);
    for (uint32_t i = 0; i < N_ELEMS; i++) {
        keys[i] = map_key(c.elems[i]);
    }
    if (pattern == PAT_ALL_EQUAL) {
        c.t_mapped = map_key(c.elems[0]) - 1;
    } else {
        std::vector<uint32_t> sorted = keys;
        std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
        c.t_mapped = sorted[511];
    }

    // Survivors (strictly greater), in index order.
    for (uint32_t i = 0; i < N_ELEMS; i++) {
        if (keys[i] > c.t_mapped) {
            c.survivors.emplace_back((uint32_t)c.elems[i], i);
            c.surv_csum += (uint32_t)c.elems[i] + i;
        }
    }

    // Histograms of the raw high byte.
    c.hist_full.assign(256, 0);
    c.hist_first.assign(256, 0);
    c.hist_second.assign(256, 0);
    for (uint32_t i = 0; i < N_ELEMS; i++) {
        uint32_t hb = c.elems[i] >> 8;
        c.hist_full[hb]++;
        (i < N_ELEMS / 2 ? c.hist_first : c.hist_second)[hb]++;
        c.byte_sum += hb;
    }
    for (uint32_t b = 0; b < 256; b++) {
        c.hist_csum_full += c.hist_full[b] * (b + 1);
        c.hist_csum_first += c.hist_first[b] * (b + 1);
        c.hist_csum_second += c.hist_second[b] * (b + 1);
    }

    // v5 compressed stream: per 32768-element chunk, fused keys
    // (bits<<16)|(local+1) with 16:1 zero elision (a forced placeholder zero
    // word every 16 elided zeros), padded to 32-datum groups, each group
    // followed by 4 zero "counter" words the consumer skips by address math.
    c.sparse_words.assign(SPARSE_WORDS, 0);
    uint32_t wpos = 0;
    for (uint32_t seg = 0; seg < 2; seg++) {
        std::vector<uint32_t> datums;
        uint32_t zrun = 0;
        for (uint32_t local = 0; local < CHUNK_ELEMS; local++) {
            uint32_t g = seg * CHUNK_ELEMS + local;
            if (keys[g] > c.t_mapped) {
                datums.push_back(((uint32_t)c.elems[g] << 16) | (local + 1));
                zrun = 0;
            } else if (++zrun == 16) {
                datums.push_back(0);
                zrun = 0;
            }
        }
        while (datums.size() % 32 != 0) {
            datums.push_back(0);
        }
        c.seg_groups[seg] = (uint32_t)(datums.size() / 32);
        for (uint32_t grp = 0; grp < c.seg_groups[seg]; grp++) {
            for (uint32_t j = 0; j < 32; j++) {
                c.sparse_words[wpos++] = datums[grp * 32 + j];
            }
            wpos += 4;  // counter block (left zero)
        }
    }
    uint32_t stream_bytes = (c.seg_groups[0] + c.seg_groups[1]) * 36 * 4;
    c.sparse_pages = (stream_bytes + STAGE_PAGE - 1) / STAGE_PAGE;
    return c;
}

struct CellResult {
    bool pass = false;
    std::string reason;
    uint32_t cycles = 0;         // BRISC (or max of both for v6)
    uint32_t cycles_ncrisc = 0;  // v6 only
    uint32_t elems = 0;          // normalization denominator reported by kernel
    double cyc_per_elem = 0.0;
    double cyc_per_orig_elem = 0.0;  // v5: cycles / 65536
};

int main() {
    int exit_code = 0;
    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        constexpr CoreCoord core = {0, 0};

        // ------------------------- buffers (created once) -------------------
        auto mk_dram = [&](uint32_t size, uint32_t page) {
            distributed::DeviceLocalBufferConfig cfg{.page_size = page, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig rcfg{.size = size};
            return distributed::MeshBuffer::create(rcfg, cfg, mesh_device.get());
        };
        auto dram_in = mk_dram(IN_BYTES, STAGE_PAGE);
        auto dram_sparse = mk_dram(SPARSE_BYTES, STAGE_PAGE);
        auto dram_result = mk_dram(RESULT_BYTES, RESULT_BYTES);
        auto dram_hist = mk_dram(HIST_BYTES, HIST_BYTES);
        auto dram_emit = mk_dram(EMIT_BYTES, EMIT_BYTES);

        distributed::DeviceLocalBufferConfig l1_cfg{.page_size = L1_BYTES, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig l1_rcfg{.size = L1_BYTES};
        auto l1_scratch = distributed::MeshBuffer::create(l1_rcfg, l1_cfg, mesh_device.get());

        fmt::print(
            "risc_scan_bench: N={} bf16 elems, core (0,0), L1 scratch @ 0x{:x} ({} B)\n",
            N_ELEMS,
            l1_scratch->address(),
            L1_BYTES);
        fmt::print(
            "{:<20} {:<10} {:>12} {:>10} {:>12}  {}\n",
            "variant",
            "pattern",
            "cycles",
            "cyc/elem",
            "cyc/origelem",
            "check");
        fmt::print("{:-<84}\n", "");

        // X constants harvested for the decision rule.
        double x_v1_uniform = -1, x_v3_clustered = -1, x_v5_uniform_word = -1, x_v7_uniform = -1, x_v6_uniform_agg = -1,
               x_v4_uniform = -1;

        for (int variant = 1; variant <= 7; variant++) {
            for (int pattern = 0; pattern < 3; pattern++) {
                CaseData cd = build_case(pattern);

                distributed::EnqueueWriteMeshBuffer(cq, dram_in, cd.in_words, /*blocking=*/false);
                distributed::EnqueueWriteMeshBuffer(cq, dram_sparse, cd.sparse_words, /*blocking=*/false);

                Program program = CreateProgram();
                distributed::MeshWorkload workload;
                distributed::MeshCoordinateRange device_range(mesh_device->shape());

                uint32_t sem_id = CreateSemaphore(program, CoreRange(core), 0);

                auto make_ct_args = [&](uint32_t is_brisc, uint32_t start, uint32_t num) {
                    std::vector<uint32_t> ct = {
                        (uint32_t)variant,
                        is_brisc,
                        start,
                        num,
                        IN_OFF,
                        SPARSE_OFF,
                        EMIT_OFF,
                        RESULT_OFF,
                        HIST_OFF,
                        EMIT_CAP,
                        sem_id,
                        STAGE_PAGE};
                    TensorAccessorArgs(*dram_in->get_backing_buffer()).append_to(ct);
                    TensorAccessorArgs(*dram_sparse->get_backing_buffer()).append_to(ct);
                    TensorAccessorArgs(*dram_result->get_backing_buffer()).append_to(ct);
                    TensorAccessorArgs(*dram_hist->get_backing_buffer()).append_to(ct);
                    TensorAccessorArgs(*dram_emit->get_backing_buffer()).append_to(ct);
                    return ct;
                };
                const std::vector<uint32_t> rt_args = {
                    (uint32_t)l1_scratch->address(),
                    (uint32_t)dram_in->address(),
                    (uint32_t)dram_sparse->address(),
                    (uint32_t)dram_result->address(),
                    (uint32_t)dram_hist->address(),
                    (uint32_t)dram_emit->address(),
                    cd.t_mapped,
                    cd.seg_groups[0],
                    cd.seg_groups[1],
                    IN_PAGES,
                    cd.sparse_pages};

                const bool dual = (variant == 6);
                const uint32_t brisc_elems = dual ? N_ELEMS / 2 : N_ELEMS;
                KernelHandle brisc_k = CreateKernel(
                    program,
                    OVERRIDE_KERNEL_PREFIX "risc_scan_bench/kernels/scan_bench.cpp",
                    core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = make_ct_args(1, 0, brisc_elems)});
                SetRuntimeArgs(program, brisc_k, core, rt_args);
                if (dual) {
                    KernelHandle ncrisc_k = CreateKernel(
                        program,
                        OVERRIDE_KERNEL_PREFIX "risc_scan_bench/kernels/scan_bench.cpp",
                        core,
                        DataMovementConfig{
                            .processor = DataMovementProcessor::RISCV_1,
                            .noc = NOC::RISCV_1_default,
                            .compile_args = make_ct_args(0, N_ELEMS / 2, N_ELEMS / 2)});
                    SetRuntimeArgs(program, ncrisc_k, core, rt_args);
                }

                workload.add_program(device_range, std::move(program));
                distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
                distributed::Finish(cq);

                std::vector<uint32_t> res, hist, emit;
                distributed::EnqueueReadMeshBuffer(cq, res, dram_result, /*blocking=*/true);
                distributed::EnqueueReadMeshBuffer(cq, hist, dram_hist, /*blocking=*/true);
                distributed::EnqueueReadMeshBuffer(cq, emit, dram_emit, /*blocking=*/true);

                // ----------------------- verification --------------------------
                CellResult r;
                r.pass = true;
                auto fail = [&](const std::string& why) {
                    if (r.pass) {
                        r.pass = false;
                        r.reason = why;
                    }
                };
                const uint32_t b_cycles = res[0], b_csum = res[1], b_count = res[2], b_ovfl = res[3], b_magic = res[4],
                               b_elems = res[5];
                const uint32_t expect_magic_b = 0xBE9C0000u | ((uint32_t)variant << 8) | 1u;
                if (b_magic != expect_magic_b) {
                    fail(fmt::format("BRISC magic 0x{:08x} != 0x{:08x}", b_magic, expect_magic_b));
                }
                r.cycles = b_cycles;
                r.elems = b_elems;

                auto check_hist = [&](const std::vector<uint32_t>& ref, uint32_t bin_off, const char* who) {
                    for (uint32_t b = 0; b < 256; b++) {
                        if (hist[bin_off + b] != ref[b]) {
                            fail(fmt::format("{} hist bin {}: got {} want {}", who, b, hist[bin_off + b], ref[b]));
                            return;
                        }
                    }
                };
                auto check_emit = [&](uint32_t count, uint32_t csum, uint32_t ovfl) {
                    uint32_t ref_count = (uint32_t)cd.survivors.size();
                    if (count != ref_count) {
                        fail(fmt::format("survivor count {} != ref {}", count, ref_count));
                    }
                    if (csum != cd.surv_csum) {
                        fail(fmt::format("survivor csum 0x{:08x} != ref 0x{:08x}", csum, cd.surv_csum));
                    }
                    uint32_t want_ovfl = ref_count > EMIT_CAP ? 1 : 0;
                    if (ovfl != want_ovfl) {
                        fail(fmt::format("overflow flag {} != {}", ovfl, want_ovfl));
                    }
                    uint32_t n_check = std::min(ref_count, EMIT_CAP);
                    for (uint32_t i = 0; i < n_check; i++) {
                        if (emit[2 * i] != cd.survivors[i].first || emit[2 * i + 1] != cd.survivors[i].second) {
                            fail(fmt::format(
                                "emit[{}]=(0x{:x},{}) != ref (0x{:x},{})",
                                i,
                                emit[2 * i],
                                emit[2 * i + 1],
                                cd.survivors[i].first,
                                cd.survivors[i].second));
                            return;
                        }
                    }
                };

                switch (variant) {
                    case 1:
                    case 2:
                    case 3:
                        check_hist(cd.hist_full, 0, "BRISC");
                        if (b_csum != cd.hist_csum_full) {
                            fail(fmt::format("hist csum 0x{:08x} != 0x{:08x}", b_csum, cd.hist_csum_full));
                        }
                        if (b_count != N_ELEMS) {
                            fail("count != N");
                        }
                        r.cyc_per_elem = (double)b_cycles / N_ELEMS;
                        break;
                    case 4:
                        check_emit(b_count, b_csum, b_ovfl);
                        r.cyc_per_elem = (double)b_cycles / N_ELEMS;
                        break;
                    case 5: {
                        uint32_t datum_words = (cd.seg_groups[0] + cd.seg_groups[1]) * 32;
                        if (b_elems != datum_words) {
                            fail(fmt::format("stream words {} != ref {}", b_elems, datum_words));
                        }
                        check_emit(b_count, b_csum, b_ovfl);
                        r.cyc_per_elem = (double)b_cycles / datum_words;   // cyc/WORD (the X for (e))
                        r.cyc_per_orig_elem = (double)b_cycles / N_ELEMS;  // amortized per original element
                        break;
                    }
                    case 6: {
                        const uint32_t n_cycles = res[16], n_csum = res[17], n_count = res[18], n_magic = res[20];
                        const uint32_t expect_magic_n = 0xBE9C0000u | ((uint32_t)variant << 8);
                        if (n_magic != expect_magic_n) {
                            fail(fmt::format("NCRISC magic 0x{:08x} != 0x{:08x}", n_magic, expect_magic_n));
                        }
                        check_hist(cd.hist_first, 0, "BRISC");
                        check_hist(cd.hist_second, 256, "NCRISC");
                        if (b_csum != cd.hist_csum_first || n_csum != cd.hist_csum_second) {
                            fail("v6 hist csum mismatch");
                        }
                        if (b_count != N_ELEMS / 2 || n_count != N_ELEMS / 2) {
                            fail("v6 per-RISC count mismatch");
                        }
                        r.cycles_ncrisc = n_cycles;
                        r.cycles = std::max(b_cycles, n_cycles);
                        r.cyc_per_elem = (double)r.cycles / (N_ELEMS / 2);  // per-RISC rate
                        r.cyc_per_orig_elem = (double)r.cycles / N_ELEMS;   // aggregate (dual) rate
                        break;
                    }
                    case 7:
                        if (b_csum != cd.byte_sum) {
                            fail(fmt::format("byte sum 0x{:08x} != ref 0x{:08x}", b_csum, cd.byte_sum));
                        }
                        if (b_count != N_ELEMS) {
                            fail("count != N");
                        }
                        r.cyc_per_elem = (double)b_cycles / N_ELEMS;
                        break;
                }

                // ----------------------- report row ----------------------------
                fmt::print(
                    "{:<20} {:<10} {:>12} {:>10.3f} {:>12.3f}  {}\n",
                    variant_name(variant),
                    pattern_name(pattern),
                    r.cycles,
                    r.cyc_per_elem,
                    (variant == 5 || variant == 6) ? r.cyc_per_orig_elem : r.cyc_per_elem,
                    r.pass ? "PASS" : "FAIL");
                if (variant == 6) {
                    fmt::print("{:<20} {:<10} (BRISC {} cyc, NCRISC {} cyc)\n", "", "", b_cycles, r.cycles_ncrisc);
                }
                if (!r.pass) {
                    fmt::print(
                        "******** FAIL [{} / {}]: {} ********\n",
                        variant_name(variant),
                        pattern_name(pattern),
                        r.reason);
                    exit_code = 1;
                }

                if (r.pass) {
                    if (variant == 1 && pattern == PAT_UNIFORM) {
                        x_v1_uniform = r.cyc_per_elem;
                    }
                    if (variant == 3 && pattern == PAT_CLUSTERED) {
                        x_v3_clustered = r.cyc_per_elem;
                    }
                    if (variant == 4 && pattern == PAT_UNIFORM) {
                        x_v4_uniform = r.cyc_per_elem;
                    }
                    if (variant == 5 && pattern == PAT_UNIFORM) {
                        x_v5_uniform_word = r.cyc_per_elem;
                    }
                    if (variant == 6 && pattern == PAT_UNIFORM) {
                        x_v6_uniform_agg = r.cyc_per_orig_elem;
                    }
                    if (variant == 7 && pattern == PAT_UNIFORM) {
                        x_v7_uniform = r.cyc_per_elem;
                    }
                }
            }
        }

        fmt::print("{:-<84}\n", "");
        fmt::print("Gate-2 decision constants (see RUNBOOK.md for the full rule):\n");
        fmt::print("  X_load_floor   (v7 uniform, cyc/elem) = {:.3f}\n", x_v7_uniform);
        fmt::print("  X_hist         (v1 uniform, cyc/elem) = {:.3f}\n", x_v1_uniform);
        fmt::print("  X_hist_clust   (v3 clustered, cyc/elem) = {:.3f}\n", x_v3_clustered);
        fmt::print("  X_dense_emit   (v4 uniform, cyc/elem) = {:.3f}\n", x_v4_uniform);
        fmt::print("  X_consumer     (v5 uniform, cyc/WORD) = {:.3f}\n", x_v5_uniform_word);
        fmt::print("  X_dual_agg     (v6 uniform, cyc/elem aggregate) = {:.3f}\n", x_v6_uniform_agg);
        if (x_v5_uniform_word >= 0) {
            if (x_v5_uniform_word <= 6.0) {
                fmt::print("  VERDICT: X <= ~5-6 cyc/word — Gate-2 candidates (a)/(e) stay ALIVE.\n");
            } else if (x_v5_uniform_word >= 10.0) {
                fmt::print("  VERDICT: X >= 10 cyc/word — the RISC arm is KILLED (demote to oracle/emit-only).\n");
            } else {
                fmt::print("  VERDICT: X in the 6-10 gray band — one 16:1 cascade stage decides; see reports.\n");
            }
        }

        if (!mesh_device->close()) {
            exit_code = 1;
        }
    } catch (const std::exception& e) {
        fmt::print(stderr, "******** FAIL: exception: {} ********\n", e.what());
        return 1;
    }

    if (exit_code == 0) {
        fmt::print("ALL CELLS PASSED\n");
    } else {
        fmt::print("******** ONE OR MORE CELLS FAILED ********\n");
    }
    return exit_code;
}
