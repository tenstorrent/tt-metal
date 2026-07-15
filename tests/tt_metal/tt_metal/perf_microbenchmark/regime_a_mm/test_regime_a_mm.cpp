// SPDX-License-Identifier: Apache-2.0
//
// Regime-A DRAM-BW-optimal matmul prototype (INC1: correctness, all interleaved, reader==consumer).
// out[M,N] = in0[M,K] @ in1[K,N], output-stationary: each core owns an N-band, reads full in0 + its
// in1 band, feeds minimal_matmul's real compute.cpp, writes its output band. Correctness via constant
// inputs (in0=in1=1.0 -> every out element == K).
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_regime_a_mm --m 32 --k 2048 --n 2048 --gx 4 --gy 4 --kb 8

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>
#include <map>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_common.hpp"

using namespace tt;

static float bf16_to_float(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

int main(int argc, char** argv) {
    bool pass = true;
    uint32_t M = 32, K = 2048, N = 2048, gx = 4, gy = 4, kb = 8, num_tests = 5, preaders = 1;
    bool sharded = false, nosplit = false;
    std::vector<std::string> a(argv, argv + argc);
    std::tie(sharded, a) = test_args::has_command_option_and_remaining_args(a, "--sharded");
    std::tie(nosplit, a) =
        test_args::has_command_option_and_remaining_args(a, "--nosplit");  // force all-NOC0 (Phase-0 A/B)
    bool skipin0 = false, bcast = false;
    std::tie(skipin0, a) = test_args::has_command_option_and_remaining_args(a, "--skipin0");  // ablation: in0 read free
    bool skipin1 = false;
    std::tie(skipin1, a) =
        test_args::has_command_option_and_remaining_args(a, "--skipin1");  // ablation: in1 read free (isolate compute)
    std::tie(bcast, a) =
        test_args::has_command_option_and_remaining_args(a, "--bcast");  // Phase 2: dedicated-loader in0 broadcast
    bool bstream = false;  // in0 STREAMING broadcast (small ring cb0; works for large Mt where --bcast OOMs)
    std::tie(bstream, a) = test_args::has_command_option_and_remaining_args(a, "--bstream");
    uint32_t bdepth = 4;  // bstream cb0 ring depth (blocks in flight)
    std::tie(bdepth, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--bdepth", bdepth);
    bool bcontig = false;  // bstream diagnostic: loader reads each block as one contiguous burst (constant-input only)
    std::tie(bcontig, a) = test_args::has_command_option_and_remaining_args(a, "--bcontig");
    uint32_t nloaders = 1;
    std::tie(nloaders, a) =
        test_args::get_command_option_uint32_and_remaining_args(a, "--nloaders", nloaders);  // 1 or 2
    uint32_t ksplit = 0;
    std::tie(ksplit, a) =
        test_args::get_command_option_uint32_and_remaining_args(a, "--ksplit", ksplit);  // Pk K-slices/bank
    uint32_t nsring = 0;  // (b) N-SLICE + in0 ring all-gather: P N-sub-bands/bank, FULL K per core, NO reduction.
    std::tie(nsring, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nsring", nsring);
    uint32_t msplit =
        1;  // 2D: M-split factor Sm (each core owns an M-block); with --ksplit Pk --ring => 8*Pk*Sm cores.
    std::tie(msplit, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--msplit", msplit);
    uint32_t nslice = 1;  // 2D: N-slice factor Ns (each core owns N_band/Ns cols, FULL M) + K-split => 8*Pk*Ns cores.
    std::tie(nslice, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nslice", nslice);
    bool in1mcast = false;  // M-split: forward in1 to the Sm-1 slaves via ONE mcast (else Sm-1 unicasts). A/B.
    std::tie(in1mcast, a) = test_args::has_command_option_and_remaining_args(a, "--in1mcast");
    bool in0direct = false;  // in0: each core reads its full [M-block,k-slice] directly (no ring/forward). A/B vs ring.
    std::tie(in0direct, a) = test_args::has_command_option_and_remaining_args(a, "--in0direct");
    bool mshard =
        false;  // in0: M-shard ring all-gather -> read-once + CONTIGUOUS [M,Kt_local] cb0 (deep-K). kb=Kt_local.
    std::tie(mshard, a) = test_args::has_command_option_and_remaining_args(a, "--mshard");
    bool moverlap =
        false;  // OVERLAPPED deep-K: streaming M-shard ring (push per shard) + M_blocks_per_core=G + IN1_RESIDENT. Pk1.
    std::tie(moverlap, a) = test_args::has_command_option_and_remaining_args(a, "--moverlap");
    uint32_t modepth = 3;  // moverlap cb0 ring depth (shards in flight)
    std::tie(modepth, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--modepth", modepth);
    bool skipfwd = false;  // ablation (ring): skip the in0 ring FORWARD (+recv) to isolate forwarding cost
    std::tie(skipfwd, a) = test_args::has_command_option_and_remaining_args(a, "--skipfwd");
    bool noreduce =
        false;  // ablation (K-split): every core is bottom+top => copies+writes its partial, NO reduction chain
    std::tie(noreduce, a) = test_args::has_command_option_and_remaining_args(a, "--noreduce");
    bool lofi = false;  // ablation: MathFidelity LoFi (vs HiFi2) — isolate whether compute is fidelity-bound
    std::tie(lofi, a) = test_args::has_command_option_and_remaining_args(a, "--lofi");
    uint32_t sbh_ovr = 0;  // override subblock height (0 = auto largest_div(M,2)); bigger sbh => fewer in1 re-unpacks
    std::tie(sbh_ovr, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--sbh", sbh_ovr);
    uint32_t nsdepth = 2;  // nsring streaming ring depth (shards in flight in cb0); >=2 for double-buffer
    std::tie(nsdepth, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nsdepth", nsdepth);
    uint32_t gring = 0;  // (b) GLOBAL read-once ring: ONE ring over all 8P cores (each reads 1/(8P) of in0 ONCE)
    std::tie(gring, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--gring", gring);
    bool rectplace = false;
    std::tie(rectplace, a) =
        test_args::has_command_option_and_remaining_args(a, "--rect");  // Pk x 8 rect (cols=banks, rows=slices)
    bool fwd = false;  // in0 store-and-forward chain (K-split): read once per k-slice, unicast around the group
    std::tie(fwd, a) = test_args::has_command_option_and_remaining_args(a, "--fwd");
    bool ring = false;  // in0 ring all-gather (K-split): every core reads a shard, rotates around the ring
    std::tie(ring, a) = test_args::has_command_option_and_remaining_args(a, "--ring");
    std::string chain = "bank";  // in0 chain/ring order: bank (0..7) | nn (nearest-neighbor by NoC distance)
    std::tie(chain, a) = test_args::get_command_option_and_remaining_args(a, "--chain", chain);
    std::string in0risc = "other";  // ring in0 shard read: other (2nd RISC) | same (in1 RISC/NoC)
    std::tie(in0risc, a) = test_args::get_command_option_and_remaining_args(a, "--in0risc", in0risc);
    std::string in0order = "before";  // same-RISC read order vs in1: before | after | interleave
    std::tie(in0order, a) = test_args::get_command_option_and_remaining_args(a, "--in0order", in0order);
    uint32_t nsb = 0;  // large-Mt: N-sub-block width (tiles). 0 = full N_band. ring only.
    std::tie(nsb, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nsb", nsb);
    bool unified = false;  // UNIFIED (Ns,Pk,Sm) factorizer: 8*Pk*Ns*Sm ring grid, first-class padding (any factors).
    std::tie(unified, a) = test_args::has_command_option_and_remaining_args(a, "--unified");
    bool in0share =
        false;  // N-slice shared in0: leader (nn==0) rings + forwards in0 to nn>0 siblings (no redundant read/ring)
    std::tie(in0share, a) = test_args::has_command_option_and_remaining_args(a, "--in0share");
    bool in0scatter = false;  // in0 all-gather via direct scatter (1 round of G-1 writes) instead of G-1 ring rotations
    std::tie(in0scatter, a) = test_args::has_command_option_and_remaining_args(a, "--in0scatter");
    bool nsbcontig =
        false;  // diagnostic: read each N-sub-band contiguously (layout-optimal ceiling; constant-input only)
    std::tie(nsbcontig, a) = test_args::has_command_option_and_remaining_args(a, "--nsbcontig");
    std::tie(preaders, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--preaders", preaders);
    // Unified (Ns,Pk,Sm) factorizer: grid = 8 banks × Pk × Ns × Sm. mfac = Ns*Sm (both, not XOR — backward-compat
    // since Ns*Sm == the single non-1 factor in the legacy XOR cases). g = k*(Ns*Sm) + n*Sm + m (m innermost so
    // M-slaves are adjacent; reduction over k has stride Ns*Sm).
    uint32_t mfac = (msplit > 1 ? msplit : 1u) * (nslice > 1 ? nslice : 1u);
    if (unified) {
        ring = true;
        if (ksplit == 0) {
            ksplit = 1;
        }
    }  // unified => ring all-gather in0, K-split plumbing (Pk≥1)
    if (ksplit > 0) {
        sharded = true;
        preaders = ksplit * mfac;
    }
    // (b) N-slice + in0 ring: reuse the ring path (bank-adjacent placement + ring all-gather) but the P "slices"
    // are N-sub-bands (full K per core, no reduction). Activates the ring path via ksplit=1-style plumbing.
    if (nsring > 0) {
        sharded = true;
        ring = true;
        preaders = nsring;
    }
    // (b) global read-once ring reuses the nsring N-slice plumbing but ONE ring spans all 8P cores.
    if (gring > 0) {
        sharded = true;
        ring = true;
        preaders = gring;
        nsring = gring;
    }
    bool global_ring = (gring > 0);
    if (moverlap) {
        ring = true;  // overlapped deep-K uses the ring plumbing (M-shard ring, Pk1)
    }
    std::tie(M, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--m", M);
    std::tie(K, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--k", K);
    std::tie(N, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--n", N);
    std::tie(gx, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--gx", gx);
    std::tie(gy, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--gy", gy);
    std::tie(kb, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kb", kb);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        auto bf16 = tt::DataFormat::Float16_b;
        auto fp32 = tt::DataFormat::Float32;
        uint32_t tb = tt::tile_size(bf16), tf = tt::tile_size(fp32);

        auto cdiv = [](uint32_t x, uint32_t y) { return (x + y - 1) / y; };
        auto rup = [&](uint32_t x, uint32_t y) { return cdiv(x, y) * y; };
        uint32_t Mt = M / 32, Kt = K / 32, Nt = N / 32;
        uint32_t num_cores = sharded ? 8 * preaders : gx * gy;
        bool KP = (ksplit || nsring);                    // uses the K-split/ring kernel plumbing
        uint32_t ring_G = global_ring ? num_cores : 8u;  // global read-once ring spans all 8P cores; else per-bank 8
        // --- Padded/logical dims. For --unified, first-class padding makes ANY (Ns,Pk,Sm,kb,nsb) runnable: pad
        // Kt so each k-slice is a multiple of kb*8 (ring shards + kb), pad Mt to Sm-multiple, Nt to 8*Ns*nsb.
        // Padding is IDENTITY when divisible, so legacy paths are unchanged. Correctness: in0 pad tiles = 0 zero
        // the pad-K products, so out[real M,N] == K regardless of layout; padded output region is not checked.
        uint32_t Pk = ksplit ? ksplit : 1u, Ns = nslice ? nslice : 1u, Sm = msplit ? msplit : 1u;
        uint32_t Kt_local, N_band, N_own, N_sub, N_bpc, M_block, N_block;
        uint32_t Kt_s, Nt_s, N_band_s, N_own_s, Mt_s;  // padded strides for buffers/kernels (== logical when !unified)
        bool deepk_u = unified && (in0direct || mshard || moverlap);  // deep-K: 1 contiguous block/core, NO 8-ring
        if (unified) {
            if (deepk_u) {
                // Deep-K delivery (mshard/in0direct): the whole k-slice is ONE compute block (K_num_blocks_eff==1),
                // so kb IS the k-slice depth and there is no 8-shard ring -> DON'T round to kb*8. kb must cover the
                // (padded) k-slice: kb >= cdiv(Kt,Pk). This decouples compute-kb from the ring (deep kb => compute
                // eff).
                TT_FATAL(
                    kb >= cdiv(Kt, Pk),
                    "deep-K unified: kb {} must be >= ceil(Kt/Pk) {} (whole k-slice per block)",
                    kb,
                    cdiv(Kt, Pk));
                Kt_local = kb;  // one block == whole k-slice
            } else {
                Kt_local = rup(cdiv(Kt, Pk), kb * 8u);  // ring: k-slice padded to kb*8 (=> Keff=Kt_local/kb mult of 8)
            }
            Kt_s = Pk * Kt_local;  // padded K (buffer row stride)
            M_block = cdiv(Mt, Sm);
            Mt_s = Sm * M_block;       // M-block padded
            N_band = cdiv(Nt, 8u);     // per-bank width (pad Nt to 8)
            N_own = cdiv(N_band, Ns);  // per-core N (before nsb)
            N_sub = nsb ? nsb : N_own;
            N_bpc = cdiv(N_own, N_sub);
            N_own_s = N_bpc * N_sub;
            N_band_s = Ns * N_own_s;
            Nt_s = 8u * N_band_s;
            N_block = N_sub;
        } else {
            N_band = Nt / 8;
            Kt_local = ksplit ? (Kt / ksplit) : Kt;
            TT_FATAL(!ksplit || (Kt % ksplit == 0 && Nt % 8 == 0), "ksplit must divide Kt and 8|Nt");
            TT_FATAL(!nsring || (Nt % 8 == 0 && N_band % nsring == 0), "nsring: 8|Nt and nsring|N_band");
            TT_FATAL(KP || Nt % num_cores == 0, "Nt {} must divide num_cores {}", Nt, num_cores);
            TT_FATAL(Kt_local % kb == 0, "Kt_local {} must divide kb {}", Kt_local, kb);
            N_block = ksplit ? N_band : (Nt / num_cores);
            TT_FATAL(nslice == 1 || (N_band % nslice == 0), "nslice {} must divide N_band {}", nslice, N_band);
            N_own = (nslice > 1) ? (N_band / nslice) : N_band;
            N_bpc = 1;
            N_sub = (nslice > 1) ? N_own : N_block;
            if (ring && nsb > 0) {
                uint32_t span = (nslice > 1) ? N_own : N_band;
                TT_FATAL(span % nsb == 0, "nsb {} must divide N span {}", nsb, span);
                N_sub = nsb;
                N_bpc = span / nsb;
            }
            TT_FATAL(Mt % msplit == 0, "msplit {} must divide Mt {}", msplit, Mt);
            M_block = Mt / msplit;
            Kt_s = Kt;
            Nt_s = Nt;
            N_band_s = N_band;
            N_own_s = N_own;
            Mt_s = Mt;
        }
        uint32_t K_num_blocks_eff = Kt_local / kb;
        uint32_t K_num_blocks = (KP || unified) ? K_num_blocks_eff : (Kt / kb);
        // subblock: largest divisor of the block dim within the DST budget (<=4 tiles for fp32).
        auto largest_div = [](uint32_t v, uint32_t cap) {
            for (uint32_t d = cap; d >= 1; --d) {
                if (d <= v && v % d == 0) {
                    return d;
                }
            }
            return 1u;
        };
        uint32_t sbh = sbh_ovr ? largest_div(M_block, sbh_ovr) : largest_div(M_block, 2u);
        uint32_t sbw = largest_div(N_block, 4u / sbh);
        log_info(
            LogTest,
            "M{} K{} N{} | Mt{} Kt{} Nt{} | {} cores, N_block {}, kb {}, sb {}x{}",
            M,
            K,
            N,
            Mt,
            Kt,
            Nt,
            num_cores,
            N_block,
            kb,
            sbh,
            sbw);

        std::vector<CoreCoord> cores, loaders;
        std::vector<uint32_t> core_bank, core_suboff, core_noc;  // sharded: per-core bank id + sub-band + NoC
        std::set<CoreRange> cs;
        auto grid_all = device->compute_with_storage_grid_size();
        if (sharded) {
            // SPLIT-NOC multi-reader (Exp5): P readers/bank, alternating NOC0/NOC1, each near its per-NoC
            // optimal core. Sustains ~500 GB/s vs all-NOC0's ~390 for P>1. --nosplit forces all-NOC0.
            auto opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_0);
            auto opt1 = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_1);
            TT_FATAL(opt0.size() == 8 && opt1.size() == 8, "expected 8 bank-adjacent cores");
            auto grid2 = device->compute_with_storage_grid_size();
            std::set<CoreCoord> used;
            auto find_near = [&](CoreCoord t) -> CoreCoord {
                for (int d = 0; d < (int)(grid2.x + grid2.y); ++d) {
                    for (int dx = -d; dx <= d; ++dx) {
                        int dy = d - std::abs(dx);
                        for (int sy : {dy, -dy}) {
                            int x = (int)t.x + dx, y = (int)t.y + sy;
                            if (x >= 0 && x < (int)grid2.x && y >= 0 && y < (int)grid2.y) {
                                CoreCoord c(x, y);
                                if (!used.count(c)) {
                                    used.insert(c);
                                    return c;
                                }
                            }
                        }
                    }
                }
                TT_FATAL(false, "no free core near ({},{})", t.x, t.y);
                return t;
            };
            // Reserve the in0-broadcast loader core(s) FIRST (corner), so compute cores avoid them.
            if (bcast || bstream) {
                for (uint32_t l = 0; l < nloaders; ++l) {
                    loaders.push_back(find_near(CoreCoord(grid2.x - 1, grid2.y - 1 - l)));
                }
            }
            // M-split: place each (b,k) group's Sm cores as a CONTIGUOUS vertical strip (reader=m0 on top, slaves
            // below) so the in1 forward can be a single mcast to a clean rectangle. find_strip finds a free run.
            auto find_strip = [&](CoreCoord t, uint32_t len) -> CoreCoord {
                for (int d = 0; d < (int)(grid2.x + grid2.y); ++d) {
                    for (int dx = -d; dx <= d; ++dx) {
                        int dy = d - std::abs(dx);
                        for (int sy : {dy, -dy}) {
                            int x = (int)t.x + dx, y0 = (int)t.y + sy;
                            if (x < 0 || x >= (int)grid2.x || y0 < 0 || y0 + (int)len > (int)grid2.y) {
                                continue;
                            }
                            bool ok = true;
                            for (uint32_t l = 0; l < len; ++l) {
                                if (used.count(CoreCoord(x, y0 + l))) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok) {
                                for (uint32_t l = 0; l < len; ++l) {
                                    used.insert(CoreCoord(x, y0 + l));
                                }
                                return CoreCoord(x, y0);
                            }
                        }
                    }
                }
                TT_FATAL(false, "no free {}-strip near ({},{})", len, t.x, t.y);
                return t;
            };
            for (uint32_t b = 0; b < 8; ++b) {
                for (uint32_t p = 0; p < preaders; ++p) {
                    uint32_t noc =
                        nosplit ? 0 : (msplit > 1 ? ((p / msplit) & 1u) : (p & 1u));  // M-split: alternate by k-slice
                    CoreCoord c;
                    if (rectplace) {
                        c = CoreCoord(b, p);
                    } else if (in1mcast && msplit > 1) {  // strip placement only for the mcast path (contiguous rect)
                        uint32_t mm = p % msplit;
                        if (mm == 0) {
                            CoreCoord base = find_strip(noc ? opt1[b] : opt0[b], msplit);
                            c = base;
                        } else {
                            c = cores.back();
                            c.y += 1;
                        }  // next core down the strip (already reserved)
                    } else {
                        c = find_near(noc ? opt1[b] : opt0[b]);
                    }
                    cores.push_back(c);
                    core_bank.push_back(b);
                    core_suboff.push_back(p * N_block);
                    core_noc.push_back(noc);
                    cs.insert(CoreRange(c, c));
                }
            }
        } else {
            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord c(i % gx, i / gx);
                cores.push_back(c);
                cs.insert(CoreRange(c, c));
            }
        }
        CoreRangeSet all_cores(cs);

        // COREMAP (diagnostic, Part-6 role mapping; env-gated, no timing impact): emit one line per compute
        // core so the profiler driver can join physical (x,y) -> logical role. Sharded/unified factorizer
        // only. Core index i = bank*preaders + p; within a bank p is the g-index g=k*(Ns*Sm)+n*Sm+m, so
        // kslice=p/(Ns*Sm), nslice=(p%(Ns*Sm))/Sm, mslice=p%Sm. Reduction chains over k -> redpos=kslice.
        if (sharded && std::getenv("RA_COREMAP")) {
            uint32_t NsSm = (mfac ? mfac : 1u);  // == Ns*Sm
            for (uint32_t i = 0; i < cores.size(); ++i) {
                uint32_t p = (preaders ? i % preaders : 0u);
                uint32_t kk = p / NsSm, rem = p % NsSm;
                uint32_t nn = (msplit ? rem / msplit : 0u), mm = (msplit ? rem % msplit : 0u);
                log_info(
                    LogTest,
                    "COREMAP x={} y={} bank={} kslice={} nslice={} mslice={} redpos={} noc={}",
                    cores[i].x,
                    cores[i].y,
                    core_bank[i],
                    kk,
                    nn,
                    mm,
                    kk,
                    (i < core_noc.size() ? core_noc[i] : 0u));
            }
        }

        // DRAM interleaved buffers
        auto mk_buf = [&](uint64_t ntiles) {
            tt_metal::distributed::DeviceLocalBufferConfig l{
                .page_size = tb, .buffer_type = tt_metal::BufferType::DRAM};
            tt_metal::distributed::ReplicatedBufferConfig g{.size = ntiles * tb};
            return tt_metal::distributed::MeshBuffer::create(g, l, device.get());
        };
        auto in0_buf = mk_buf((uint64_t)Mt_s * Kt_s);
        auto in1_buf = mk_buf((uint64_t)Kt_s * Nt_s);
        auto out_buf = mk_buf((uint64_t)Mt_s * Nt_s);

        // fill in1=1.0 bf16 (0x3F80). in0: real tiles [m<Mt,k<Kt]=1, pad tiles=0 so pad-K products vanish
        // => out[real M,N] == K exactly, independent of layout/pad. Tiles are row-major: tile(m,k)=m*Kt_s+k.
        uint32_t words_per_tile = tb / (uint32_t)sizeof(uint32_t);
        auto fill_ones = [&](std::shared_ptr<tt_metal::distributed::MeshBuffer>& b, uint64_t ntiles) {
            std::vector<uint32_t> v(ntiles * words_per_tile, 0x3F803F80u);
            tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), b, v, false);
        };
        auto fill_in0 = [&]() {
            std::vector<uint32_t> v((uint64_t)Mt_s * Kt_s * words_per_tile, 0u);
            for (uint32_t m = 0; m < Mt; ++m) {
                for (uint32_t k = 0; k < Kt; ++k) {
                    uint64_t off = ((uint64_t)m * Kt_s + k) * words_per_tile;
                    for (uint32_t w = 0; w < words_per_tile; ++w) {
                        v[off + w] = 0x3F803F80u;
                    }
                }
            }
            tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), in0_buf, v, false);
        };
        fill_in0();
        fill_ones(in1_buf, (uint64_t)Kt_s * Nt_s);
        tt_metal::distributed::Finish(device->mesh_command_queue());

        tt_metal::Program program = tt_metal::Program();
        // Block sizes use N_sub (== N_block except on the ring N-sub-block path, where the compute/cb block
        // is the small N_sub while the reader still reads the full N_band in N_bpc strided sub-bands).
        uint32_t in0_blk = M_block * kb, in1_blk = kb * N_sub, out_blk = M_block * N_sub;
        // bcast: cb0 holds the FULL in0 and lives on ALL grid cores (uniform L1 base for the loader mcast).
        std::set<CoreRange> full_cs, loader_cs, other_cs;
        for (uint32_t y = 0; y < grid_all.y; ++y) {
            for (uint32_t x = 0; x < grid_all.x; ++x) {
                full_cs.insert(CoreRange(CoreCoord(x, y)));
            }
        }
        for (auto& c : loaders) {
            loader_cs.insert(CoreRange(c, c));
        }
        {
            std::set<CoreCoord> keep;
            for (auto& c : cores) {
                keep.insert(c);
            }
            for (auto& c : loaders) {
                keep.insert(c);
            }
            for (uint32_t y = 0; y < grid_all.y; ++y) {
                for (uint32_t x = 0; x < grid_all.x; ++x) {
                    if (!keep.count(CoreCoord(x, y))) {
                        other_cs.insert(CoreRange(CoreCoord(x, y)));
                    }
                }
            }
        }
        CoreRangeSet full_cores(full_cs), loader_cores(loader_cs), other_cores(other_cs);
        auto mkcb = [&](const CoreRangeSet& crs, uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t tsz) {
            tt_metal::CircularBufferConfig c(ntiles * tsz, {{idx, df}});
            c.set_page_size(idx, tsz);
            tt_metal::CreateCircularBuffer(program, crs, c);
        };
        uint32_t in0_full = M_block * Kt;  // full in0 (bcast cb0 size)
        // moverlap (overlapped deep-K): compute sees Mw-row M-blocks (Mw=M_block/8); shard=in0_blk/8, out=out_blk/8.
        uint32_t mo_shard = in0_blk / 8, mo_out = out_blk / 8;
        if (moverlap) {
            mkcb(all_cores, 0, modepth * mo_shard, bf16, tb);  // cb0 = D-slot recycling M-shard ring
            mkcb(all_cores, 1, N_bpc * in1_blk, bf16, tb);     // cb1 = in1 RESIDENT (all N_bpc deep blocks)
            mkcb(all_cores, 2, 2 * mo_out, bf16, tb);
            mkcb(all_cores, 3, mo_out, fp32, tf);
            mkcb(all_cores, 7, 2u, bf16, tb);  // cb_reduce unused (Pk1); stub
        } else if (bstream) {
            mkcb(
                full_cores,
                0,
                bdepth * in0_blk,
                bf16,
                tb);  // cb0 = small D-deep ring, all grid (uniform base for mcast)
        } else if (bcast) {
            mkcb(full_cores, 0, in0_full, bf16, tb);  // cb0 full in0, all grid (loader reads into + mcasts from it)
        } else if (nsring) {
            // streaming ring: cb0 = nsdepth shards (recycling), NOT the full k-slice (which would OOM at full K)
            mkcb(all_cores, 0, nsdepth * (K_num_blocks_eff / ring_G) * in0_blk, bf16, tb);
        } else if (KP && (fwd || ring)) {
            mkcb(
                all_cores,
                0,
                K_num_blocks_eff * in0_blk,
                bf16,
                tb);  // cb0 holds the full k-slice (store-and-forward / ring)
        } else {
            mkcb(all_cores, 0, 2 * in0_blk, bf16, tb);
        }
        if (!moverlap) {
            // deep-K (mshard/in0direct): in1 block is ONE deep [kb=Kt_local, N_sub] per nb -> 2 buffers suffice
            // (4-deep would blow L1 for large kb, forcing tiny nsb and killing in1 amortization).
            uint32_t in1_depth = (in0direct || mshard) ? 2u : (sharded ? 4u : 2u);
            mkcb(all_cores, 1, in1_depth * in1_blk, bf16, tb);  // sharded ring: 3 blocks in flight + compute
            mkcb(all_cores, 2, (nsring ? 2u : 2u) * out_blk, bf16, tb);
            mkcb(all_cores, 3, out_blk, fp32, tf);  // intermediate accumulator: single-buffered (matches factory)
        }
        uint32_t recv_sem = 0, fwd_sem = 0;
        if (KP) {
            if (nsring || moverlap) {
                if (!moverlap) {
                    mkcb(all_cores, 7, 2u, bf16, tb);
                }
            }  // nsring/moverlap: cb_reduce unused stub (moverlap alloc'd above)
            else {
                mkcb(all_cores, 7, 2 * out_blk, bf16, tb);  // cb_reduce (split-K running sum)
            }
            recv_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);  // reduction recv
            if (fwd || ring) {
                fwd_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);  // in0 chain/ring recv
            }
        }
        uint32_t redfree_sem = 0;  // ring reduction reverse credit (cb_reduce slot reuse across N-sub-blocks)
        if (ring) {
            redfree_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);
        }
        // N-slice shared-in0: leader (nn==0) rings + forwards full in0 to nn>0 siblings. Prototype: Sm==1 only.
        bool in0share_act = in0share && ring && (nslice > 1) && (msplit <= 1);
        uint32_t share_valid_sem = 0, share_ready_sem = 0;
        if (in0share_act) {
            share_valid_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);
            share_ready_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);
        }
        // Direct-scatter all-gather (ring only, no in0share): reuses fwd_sem for the recv count.
        bool in0scatter_act = in0scatter && ring && !in0share_act && !nsring && !moverlap && !in0direct && !mshard;
        uint32_t in1valid_sem = 0, in1ready_sem = 0;  // M-split (milestone 2): in1 forward reader<->slaves
        if (ring && msplit > 1) {
            in1valid_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);
            in1ready_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);
        }
        uint32_t in0ready_sem = 0;
        bool in0_same = (ring && in0risc == "same");
        if (in0_same) {
            in0ready_sem = tt_metal::CreateSemaphore(program, all_cores, 0u);  // reader->writer slot-0 signal
        }
        uint32_t valid0 = (bcast || bstream) ? tt_metal::CreateSemaphore(program, full_cores, 0u) : 0u;
        uint32_t valid1 = bcast ? tt_metal::CreateSemaphore(program, full_cores, 0u) : 0u;
        std::vector<uint32_t> bs_valid, bs_ready;  // per-loader valid (loader->recv) + ready (recv->loader) sems
        if (bstream) {
            for (uint32_t l = 0; l < nloaders; ++l) {
                bs_valid.push_back(tt_metal::CreateSemaphore(program, full_cores, 0u));
                bs_ready.push_back(tt_metal::CreateSemaphore(program, full_cores, 0u));
            }
        }
        uint32_t split_h = (bcast && nloaders == 2) ? (K_num_blocks / 2) : K_num_blocks;

        tt_metal::KernelHandle reader = 0, ncrisc = 0;
        tt_metal::KernelHandle readerA = 0, readerB = 0, writerA = 0, writerB = 0;  // split-NOC handles
        const char* KDIR = "tests/tt_metal/tt_metal/perf_microbenchmark/regime_a_mm/kernels/";
        auto krn = [&](const std::string& f) { return std::string(KDIR) + f; };
        if (!sharded) {
            // INC1: interleaved reader (in0+in1) on BRISC, writer on NCRISC.
            std::vector<uint32_t> rct = {M_block, kb, N_block, K_num_blocks, tb, Nt, Kt};
            tt_metal::TensorAccessorArgs(*in0_buf->get_reference_buffer()).append_to(rct);
            tt_metal::TensorAccessorArgs(*in1_buf->get_reference_buffer()).append_to(rct);
            reader = tt_metal::CreateKernel(
                program,
                krn("reader.cpp"),
                all_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = rct});
            std::vector<uint32_t> wct = {M_block, N_block, tb, Nt};
            tt_metal::TensorAccessorArgs(*out_buf->get_reference_buffer()).append_to(wct);
            ncrisc = tt_metal::CreateKernel(
                program,
                krn("writer.cpp"),
                all_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = wct});
        } else if (KP) {
            // K-split split-NOC: each core reads a contiguous K-slice of its bank (reader_sharded + offset);
            // in0-read + reduction-chain on the OTHER NoC (in0_reduce_writer). Compute uses REDUCE_K.
            // nsring (b): same plumbing but full K + N-sub-band per core + NO reduction (is_bottom=is_top=1).
            std::set<CoreRange> s0, s1;
            for (uint32_t i = 0; i < cores.size(); ++i) {
                (core_noc[i] ? s1 : s0).insert(CoreRange(cores[i], cores[i]));
            }
            CoreRangeSet g0(s0), g1(s1);
            // 16KB-burst constraint only applies to the contiguous readers (reader_sharded / full-band ring);
            // the N-sub-block ring uses strided sub-band reads (seg = N_sub tiles), so skip it there.
            if (!(ring && (nsb > 0 || nsring || unified))) {
                TT_FATAL((in1_blk * tb) % 16384 == 0, "ksplit: in1 block (kb*N_band) must be a multiple of 16KB");
            }
            bool deepk = (in0direct || mshard || moverlap);  // contiguous [M,Kt_local] cb0, kb=Kt_local, K_num_blocks=1
            if (moverlap) {
                TT_FATAL(ksplit == 1, "moverlap is Pk1 (no reduction) only");
                TT_FATAL(M_block % 8 == 0, "moverlap needs M_block%8==0");
            }
            uint32_t Wsh =
                (ring && !deepk) ? (K_num_blocks_eff / ring_G) : 1u;  // ring: blocks per shard; deep-K: unused
            if (ring && !deepk) {
                TT_FATAL(
                    K_num_blocks_eff % ring_G == 0,
                    "ring: K_num_blocks_eff {} must divide ring_G {}",
                    K_num_blocks_eff,
                    ring_G);
            }
            if (deepk) {
                TT_FATAL(
                    K_num_blocks_eff == 1,
                    "deep-K needs kb == Kt_local (K_num_blocks_eff==1), got {}",
                    K_num_blocks_eff);
            }
            if (mshard) {
                TT_FATAL(M_block % 8 == 0, "mshard needs M_block {} divisible by G=8", M_block);
            }
            // reader compile args: ring uses reader_ring (rotated shards), else reader_sharded (contiguous slice)
            std::vector<uint32_t> rct;
            if (ring) {
                uint32_t in0ord = (in0order == "after") ? 1u : (in0order == "interleave") ? 2u : 0u;
                rct = {
                    kb,
                    N_sub,
                    Wsh,
                    ring_G,
                    tb,
                    in0_same ? 1u : 0u,
                    M_block,
                    Kt_s,
                    in0ready_sem,
                    in0ord,
                    N_bpc,
                    N_band_s,
                    nsbcontig ? 1u : 0u,
                    (nsring || nslice > 1 || deepk || unified) ? 1u : 0u,
                    skipin1 ? 1u : 0u,
                    in1valid_sem,
                    in1ready_sem,
                    in1mcast ? 1u : 0u,
                    deepk ? 1u : 0u};
                tt_metal::TensorAccessorArgs(*in0_buf->get_reference_buffer())
                    .append_to(rct);  // for the in0 shard read
            } else {
                rct = {kb, N_block, K_num_blocks_eff, tb};
            }
            // writer compile args: nsring (streaming N-slice, no reduction) / ring / fwd / per-core-read
            std::vector<uint32_t> nct;
            if (moverlap) {
                nct = {
                    M_block,
                    kb,
                    N_sub,
                    K_num_blocks_eff,
                    tb,
                    Kt,
                    Nt,
                    Wsh,
                    8u,
                    fwd_sem,
                    redfree_sem,
                    modepth,
                    M_block / 8,
                    N_bpc};
            } else if (nsring) {
                nct = {M_block, kb, N_sub, K_num_blocks_eff, tb, Kt, Nt, Wsh, ring_G, fwd_sem, redfree_sem, nsdepth};
            } else if (ring) {
                nct = {
                    M_block,
                    kb,
                    N_sub,
                    K_num_blocks_eff,
                    tb,
                    Kt_s,
                    Nt_s,
                    Wsh,
                    8u,
                    fwd_sem,
                    recv_sem,
                    in0_same ? 1u : 0u,
                    in0ready_sem,
                    N_bpc,
                    redfree_sem,
                    skipin0 ? 1u : 0u,
                    in0direct ? 1u : 0u,
                    skipfwd ? 1u : 0u,
                    noreduce ? 1u : 0u,
                    mshard ? 1u : 0u,
                    in0share_act ? 1u : 0u,
                    share_valid_sem,
                    share_ready_sem,
                    in0scatter_act ? 1u : 0u};
            } else if (fwd) {
                nct = {M_block, kb, N_block, K_num_blocks_eff, tb, Kt, Nt, fwd_sem, recv_sem};
            } else {
                nct = {M_block, kb, N_block, K_num_blocks_eff, tb, Kt, Nt, recv_sem, skipin0 ? 1u : 0u};
            }
            tt_metal::TensorAccessorArgs(*in0_buf->get_reference_buffer()).append_to(nct);
            tt_metal::TensorAccessorArgs(*out_buf->get_reference_buffer()).append_to(nct);
            auto mk = [&](const std::string& src,
                          const CoreRangeSet& g,
                          tt_metal::DataMovementProcessor proc,
                          tt_metal::NOC noc,
                          const std::vector<uint32_t>& ct) -> tt_metal::KernelHandle {
                if (g.num_cores() == 0) {
                    return 0;
                }
                return tt_metal::CreateKernel(
                    program,
                    krn(src),
                    g,
                    tt_metal::DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct});
            };
            using P = tt_metal::DataMovementProcessor;
            const char* rsrc = ring ? "reader_ring.cpp" : "reader_sharded.cpp";
            const char* wsrc = moverlap ? "in0_mshard_overlap_writer.cpp"
                               : nsring ? "in0_nsring_writer.cpp"
                               : ring   ? "in0_ring_writer.cpp"
                                        : (fwd ? "in0_fwd_reduce_writer.cpp" : "in0_reduce_writer.cpp");
            readerA = mk(rsrc, g0, P::RISCV_0, tt_metal::NOC::RISCV_0_default, rct);
            readerB = mk(rsrc, g1, P::RISCV_1, tt_metal::NOC::RISCV_1_default, rct);
            writerA = mk(wsrc, g0, P::RISCV_1, tt_metal::NOC::RISCV_1_default, nct);
            writerB = mk(wsrc, g1, P::RISCV_0, tt_metal::NOC::RISCV_0_default, nct);
        } else {
            // Sharded split-NOC: in1 read on the reader's NoC; in0-read+output-write on the OTHER NoC.
            //   NOC0-reader core: reader RISCV_0/NOC0, in0_writer RISCV_1/NOC1.
            //   NOC1-reader core: reader RISCV_1/NOC1, in0_writer RISCV_0/NOC0.
            // Build the two NoC-group core sets.
            std::set<CoreRange> s0, s1;
            for (uint32_t i = 0; i < cores.size(); ++i) {
                (core_noc[i] ? s1 : s0).insert(CoreRange(cores[i], cores[i]));
            }
            CoreRangeSet g0(s0), g1(s1);
            // reader compile args + source (preaders==1: contiguous whole-bank; else: N-sub-division)
            std::string rsrc = (preaders == 1) ? "reader_sharded.cpp" : "reader_subband.cpp";
            std::vector<uint32_t> rct =
                (preaders == 1) ? std::vector<uint32_t>{kb, N_block, K_num_blocks, tb, skipin1 ? 1u : 0u}
                                : std::vector<uint32_t>{kb, N_block, N_band, K_num_blocks, tb, skipin1 ? 1u : 0u};
            if (preaders == 1) {
                TT_FATAL((in1_blk * tb) % 16384 == 0, "P=1: in1 block bytes must be a multiple of 16KB");
            }
            std::vector<uint32_t> nct = {
                M_block,
                kb,
                N_block,
                K_num_blocks,
                tb,
                Kt,
                Nt,
                skipin0 ? 1u : 0u,
                bcast ? 1u : 0u,
                valid0,
                valid1,
                split_h,
                bstream ? 1u : 0u,
                0u /*unused*/};
            tt_metal::TensorAccessorArgs(*in0_buf->get_reference_buffer()).append_to(nct);
            tt_metal::TensorAccessorArgs(*out_buf->get_reference_buffer()).append_to(nct);
            auto mk = [&](const std::string& src,
                          const CoreRangeSet& g,
                          tt_metal::DataMovementProcessor proc,
                          tt_metal::NOC noc,
                          const std::vector<uint32_t>& ct) -> tt_metal::KernelHandle {
                if (g.num_cores() == 0) {
                    return 0;
                }
                return tt_metal::CreateKernel(
                    program,
                    krn(src),
                    g,
                    tt_metal::DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct});
            };
            using P = tt_metal::DataMovementProcessor;
            readerA = mk(rsrc, g0, P::RISCV_0, tt_metal::NOC::RISCV_0_default, rct);  // NOC0-reader in1
            readerB = mk(rsrc, g1, P::RISCV_1, tt_metal::NOC::RISCV_1_default, rct);  // NOC1-reader in1
            writerA = mk(
                "in0_writer.cpp", g0, P::RISCV_1, tt_metal::NOC::RISCV_1_default, nct);  // NOC0-core: in0+out on NOC1
            writerB = mk(
                "in0_writer.cpp", g1, P::RISCV_0, tt_metal::NOC::RISCV_0_default, nct);  // NOC1-core: in0+out on NOC0
        }
        // compute (reuse minimal_matmul compute.cpp); K-split uses REDUCE_K + local K depth.
        // Ring N-sub-block: N_block_tiles=N_sub, N_blocks_per_core=N_bpc, in0 held resident across sub-blocks.
        bool ring_nsb =
            (ring && (nsb > 0 || nslice > 1 || unified));  // compute sees N_sub blocks (in0 resident across N_bpc)
        uint32_t sbw_c = ring_nsb ? largest_div(N_sub, 4u / sbh) : sbw;
        uint32_t nblk_c = ring_nsb ? N_sub : N_block;
        uint32_t nbpc_c = ring_nsb ? N_bpc : 1u;
        std::vector<uint32_t> cct = {
            ksplit ? K_num_blocks_eff : K_num_blocks, M_block, kb, nblk_c, 1u, nbpc_c, sbh, sbw_c};
        std::map<std::string, std::string> cdefs;
        if (moverlap) {
            // overlapped deep-K: compute Mw-row M-blocks (M_blocks_per_core=G), N_sub blocks (N_bpc), in1 RESIDENT.
            uint32_t Mw = M_block / 8, sbh_m = largest_div(Mw, 2u), sbw_m = largest_div(N_sub, 4u / sbh_m);
            cct = {K_num_blocks_eff, Mw, kb, N_sub, 8u, N_bpc, sbh_m, sbw_m};  // K_num_blocks_eff==1 (deep-K)
            cdefs["IN1_RESIDENT"] = "1";
        } else {
            if (ksplit) {
                cdefs["REDUCE_K"] = "1";
            }
            if (ring_nsb) {
                cdefs["IN0_KSLICE_RESIDENT"] = "1";
            }
        }
        auto compute = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
            all_cores,
            tt_metal::ComputeConfig{
                .math_fidelity = lofi ? MathFidelity::LoFi : MathFidelity::HiFi2,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .math_approx_mode = false,
                .compile_args = cct,
                .defines = cdefs});
        // in0 store-and-forward chains (per k-slice group = the 8 same-slice cores, one per bank).
        // Chain order is a permutation of banks; the head is the injector (reads in0 from DRAM). "bank"
        // order = 0..7; "nn" = greedy nearest-neighbor by physical NoC distance (short hops, less congestion).
        std::vector<uint32_t> fwd_inj(num_cores, 0), fwd_last(num_cores, 0), fwd_nx(num_cores, 0), fwd_ny(num_cores, 0);
        std::vector<uint32_t> ring_pos(num_cores, 0), ring_nx(num_cores, 0), ring_ny(num_cores, 0);
        std::vector<uint32_t> ring_px(num_cores, 0), ring_py(num_cores, 0);  // cyclic prev (nsring slotfree credit)
        if (global_ring) {
            // ONE ring over ALL num_cores cores (greedy nearest-neighbor Hamiltonian for forwarding locality).
            // Each core reads 1/num_cores of in0 (ring_pos'th shard) -> in0 read ONCE total, no P-redundancy.
            auto phys = [&](CoreCoord c) { return device->worker_core_from_logical_core(c); };
            std::vector<uint32_t> order, rem;
            for (uint32_t i = 0; i < num_cores; ++i) {
                rem.push_back(i);
            }
            uint32_t cur = 0;
            order.push_back(cur);
            rem.erase(rem.begin());
            while (!rem.empty()) {
                auto pc = phys(cores[cur]);
                uint32_t best = 0, bd = 1u << 30;
                for (uint32_t t = 0; t < rem.size(); ++t) {
                    auto tc = phys(cores[rem[t]]);
                    uint32_t d = std::abs((int)tc.x - (int)pc.x) + std::abs((int)tc.y - (int)pc.y);
                    if (d < bd) {
                        bd = d;
                        best = t;
                    }
                }
                cur = rem[best];
                order.push_back(cur);
                rem.erase(rem.begin() + best);
            }
            for (uint32_t pos = 0; pos < num_cores; ++pos) {
                uint32_t ci = order[pos];
                ring_pos[ci] = pos;
                auto nx = phys(cores[order[(pos + 1) % num_cores]]);
                ring_nx[ci] = nx.x;
                ring_ny[ci] = nx.y;
                auto pv = phys(cores[order[(pos + num_cores - 1) % num_cores]]);
                ring_px[ci] = pv.x;
                ring_py[ci] = pv.y;
            }
        } else if (KP && (fwd || ring)) {
            auto phys = [&](CoreCoord c) { return device->worker_core_from_logical_core(c); };
            for (uint32_t j = 0; j < preaders; ++j) {
                std::vector<uint32_t> order;
                if (chain == "nn") {
                    std::vector<uint32_t> rem = {0, 1, 2, 3, 4, 5, 6, 7};
                    uint32_t cur = 0;
                    order.push_back(cur);
                    rem.erase(rem.begin());
                    while (!rem.empty()) {
                        auto pc = phys(cores[cur * preaders + j]);
                        uint32_t best = 0, bd = 1u << 30;
                        for (uint32_t t = 0; t < rem.size(); ++t) {
                            auto tc = phys(cores[rem[t] * preaders + j]);
                            uint32_t d = std::abs((int)tc.x - (int)pc.x) + std::abs((int)tc.y - (int)pc.y);
                            if (d < bd) {
                                bd = d;
                                best = t;
                            }
                        }
                        cur = rem[best];
                        order.push_back(cur);
                        rem.erase(rem.begin() + best);
                    }
                } else {
                    for (uint32_t b = 0; b < 8; ++b) {
                        order.push_back(b);
                    }
                }
                for (uint32_t pos = 0; pos < 8; ++pos) {
                    uint32_t ci = order[pos] * preaders + j;
                    if (ring) {
                        ring_pos[ci] = pos;
                        auto p = phys(cores[order[(pos + 1) % 8] * preaders + j]);  // cyclic next
                        ring_nx[ci] = p.x;
                        ring_ny[ci] = p.y;
                        auto pv = phys(cores[order[(pos + 7) % 8] * preaders + j]);  // cyclic prev
                        ring_px[ci] = pv.x;
                        ring_py[ci] = pv.y;
                    } else {  // store-and-forward linear chain
                        fwd_inj[ci] = (pos == 0) ? 1u : 0u;
                        fwd_last[ci] = (pos == 7) ? 1u : 0u;
                        if (pos < 7) {
                            auto p = phys(cores[order[pos + 1] * preaders + j]);
                            fwd_nx[ci] = p.x;
                            fwd_ny[ci] = p.y;
                        }
                    }
                }
            }
        }
        for (uint32_t i = 0; i < num_cores; ++i) {
            uint32_t n0 = i * N_block;
            if (KP) {
                uint32_t slice = i % preaders, bank = core_bank[i];
                // 2D layout: group g = slice = k*fac + sub  (fac = Sm m-blocks OR Ns n-sub-bands). Reduction is
                // over k (fixed b,sub) => stride `fac` in the core index. nsring: `slice` is the N-sub-band index.
                uint32_t kk = slice / mfac, sub = slice % mfac;
                // g-ordering: sub = nn*Sm + mm  (m innermost => M-slaves adjacent; reduction over kk stride mfac).
                uint32_t mm = sub % Sm;                          // M-split: M-block index
                uint32_t nn = sub / Sm;                          // N-slice: N-sub-band index
                uint32_t k_start = nsring ? 0u : kk * Kt_local;  // tiles
                uint32_t m0v = nsring ? 0u : mm * M_block;       // M-block row offset (tiles)
                uint32_t nn_off =
                    nsring ? (slice * N_sub) : (nn * N_own_s);  // N-sub-band col offset within bank (tiles)
                uint32_t bank_n0 = bank * N_band_s + nn_off;    // this core's N offset
                // noreduce ablation: force is_bottom=1 (compute copies its partial, no cb_reduce wait) but keep the
                // REAL top (only kk==Pk-1 writes DRAM => 1 write/output, no redundant-write confound); writer skips
                // the reduction forward for non-top and just discards. Isolates reduction COMMUNICATION cost.
                uint32_t is_bottom = (nsring || noreduce) ? 1u : (kk == 0) ? 1u : 0u;
                uint32_t is_top = nsring ? 1u : (kk == ksplit - 1) ? 1u : 0u;
                // reduction next (k+1, same b,m) is Sm cores ahead; for top it's unused
                uint32_t nx = 0, ny = 0;
                if (!is_top) {
                    auto p = device->worker_core_from_logical_core(cores[i + mfac]);
                    nx = p.x;
                    ny = p.y;
                }
                auto rh = core_noc[i] ? readerB : readerA;
                auto wh = core_noc[i] ? writerB : writerA;
                uint32_t base_off = k_start * N_band_s * tb;  // K-slice byte offset into the bank
                if (ring) {
                    // red_prev = the core BELOW (k-1, same b,m) = Sm cores back; for the bottom it is unused.
                    uint32_t px, py;
                    {
                        auto p = device->worker_core_from_logical_core(cores[is_bottom ? i : i - mfac]);
                        px = p.x;
                        py = p.y;
                    }
                    // reader_ring: in1_addr, bank, vc, slice_off, ring_pos, in0_addr, m0, k_start, n_base ;
                    // in0_ring_writer: in0,out,m0,n0, k_start, ring_pos, ring_next(x,y), red_next(x,y), is_bottom,
                    // is_top, red_prev(x,y)
                    std::vector<uint32_t> ra = {
                        (uint32_t)in1_buf->address(),
                        bank,
                        bank & 0x3u,
                        base_off,
                        ring_pos[i],
                        (uint32_t)in0_buf->address(),
                        m0v,
                        k_start,
                        nn_off};
                    // M-split in1 forward: reader (mm==0) -> the Sm-1 slaves (i+1..i+Sm-1); slaves <- reader (i-mm).
                    if (msplit > 1) {
                        if (mm == 0) {
                            ra.push_back(1u);
                            ra.push_back(msplit - 1);  // role=reader, ndest = Sm-1 slaves
                            if (in1mcast) {
                                // slaves cores[i+1..i+Sm-1] are a contiguous vertical strip -> one mcast rect.
                                auto pa = device->worker_core_from_logical_core(cores[i + 1]);
                                auto pb = device->worker_core_from_logical_core(cores[i + msplit - 1]);
                                uint32_t px = pa.x, ylo = std::min(pa.y, pb.y), yhi = std::max(pa.y, pb.y);
                                // NOC1 (core_noc==1) wants start=max corner, end=min; NOC0 the reverse.
                                if (core_noc[i]) {
                                    ra.push_back(px);
                                    ra.push_back(yhi);
                                    ra.push_back(px);
                                    ra.push_back(ylo);
                                } else {
                                    ra.push_back(px);
                                    ra.push_back(ylo);
                                    ra.push_back(px);
                                    ra.push_back(yhi);
                                }
                            } else {
                                for (uint32_t s = 1; s < msplit; ++s) {
                                    auto p = device->worker_core_from_logical_core(cores[i + s]);
                                    ra.push_back(p.x);
                                    ra.push_back(p.y);
                                }
                            }
                        } else {
                            ra.push_back(0u);
                            ra.push_back(1u);  // role=slave, peer = the reader
                            auto p = device->worker_core_from_logical_core(cores[i - mm]);
                            ra.push_back(p.x);
                            ra.push_back(p.y);
                        }
                    } else {
                        ra.push_back(2u);
                        ra.push_back(0u);
                    }  // solo (no M-split)
                    tt_metal::SetRuntimeArgs(program, rh, cores[i], ra);
                    if (moverlap) {
                        // in0_mshard_overlap_writer: in0,out,m0,n0, ring_pos, next(x,y), prev(x,y)
                        tt_metal::SetRuntimeArgs(
                            program,
                            wh,
                            cores[i],
                            {(uint32_t)in0_buf->address(),
                             (uint32_t)out_buf->address(),
                             m0v,
                             bank_n0,
                             ring_pos[i],
                             ring_nx[i],
                             ring_ny[i],
                             ring_px[i],
                             ring_py[i]});
                    } else if (nsring) {
                        // in0_nsring_writer: in0,out,m0,n0, ring_pos, next(x,y), prev(x,y)
                        tt_metal::SetRuntimeArgs(
                            program,
                            wh,
                            cores[i],
                            {(uint32_t)in0_buf->address(),
                             (uint32_t)out_buf->address(),
                             0u,
                             bank_n0,
                             ring_pos[i],
                             ring_nx[i],
                             ring_ny[i],
                             ring_px[i],
                             ring_py[i]});
                    } else {
                        std::vector<uint32_t> wa = {
                            (uint32_t)in0_buf->address(),
                            (uint32_t)out_buf->address(),
                            m0v,
                            bank_n0,
                            k_start,
                            ring_pos[i],
                            ring_nx[i],
                            ring_ny[i],
                            nx,
                            ny,
                            is_bottom,
                            is_top,
                            px,
                            py};
                        if (in0share_act) {
                            // nn = N-slice index (Sm==1). Leader (nn==0) rings+forwards to nn>0 siblings
                            // (cores[i+1..i+Ns-1]); receiver (nn>0) skips ring, recvs from leader (cores[i-nn]).
                            if (nn == 0) {
                                wa.push_back(1u);
                                wa.push_back(nslice - 1);
                                for (uint32_t s = 1; s < nslice; ++s) {
                                    auto p = device->worker_core_from_logical_core(cores[i + s]);
                                    wa.push_back(p.x);
                                    wa.push_back(p.y);
                                }
                            } else {
                                wa.push_back(2u);
                                auto p = device->worker_core_from_logical_core(cores[i - nn]);
                                wa.push_back(p.x);
                                wa.push_back(p.y);
                            }
                        }
                        if (in0scatter_act) {
                            // scatter peers = the other 7 cores of this k-slice ring group (same slice, banks != mine).
                            uint32_t jj = i % preaders;
                            for (uint32_t b = 0; b < 8; ++b) {
                                uint32_t gi = b * preaders + jj;
                                if (gi == (uint32_t)i) {
                                    continue;
                                }
                                auto p = device->worker_core_from_logical_core(cores[gi]);
                                wa.push_back(p.x);
                                wa.push_back(p.y);
                            }
                        }
                        tt_metal::SetRuntimeArgs(program, wh, cores[i], wa);
                    }
                    // N_end must span ALL N_bpc sub-blocks (compute clamps n_tile_end to it); N_block alone would
                    // clamp sub-blocks 1..N_bpc-1 to empty -> compute/reader/writer block-count mismatch -> deadlock.
                    tt_metal::SetRuntimeArgs(program, compute, cores[i], {0u, M_block, 0u, N_bpc * N_sub, is_bottom});
                    continue;
                }
                tt_metal::SetRuntimeArgs(
                    program, rh, cores[i], {(uint32_t)in1_buf->address(), bank, bank & 0x3u, base_off});
                if (fwd) {
                    // in0_fwd_reduce_writer args: in0,out,m0,n0,k_start, fwd_next(x,y), red_next(x,y),
                    //   is_injector, is_fwd_last, is_bottom, is_top
                    tt_metal::SetRuntimeArgs(
                        program,
                        wh,
                        cores[i],
                        {(uint32_t)in0_buf->address(),
                         (uint32_t)out_buf->address(),
                         0u,
                         bank_n0,
                         k_start,
                         fwd_nx[i],
                         fwd_ny[i],
                         nx,
                         ny,
                         fwd_inj[i],
                         fwd_last[i],
                         is_bottom,
                         is_top});
                } else {
                    tt_metal::SetRuntimeArgs(
                        program,
                        wh,
                        cores[i],
                        {(uint32_t)in0_buf->address(),
                         (uint32_t)out_buf->address(),
                         0u,
                         bank_n0,
                         k_start,
                         nx,
                         ny,
                         is_bottom,
                         is_top});
                }
                tt_metal::SetRuntimeArgs(program, compute, cores[i], {0u, M_block, 0u, N_block, is_bottom});
            } else if (!sharded) {
                tt_metal::SetRuntimeArgs(
                    program, reader, cores[i], {(uint32_t)in0_buf->address(), (uint32_t)in1_buf->address(), 0u, n0});
                tt_metal::SetRuntimeArgs(program, ncrisc, cores[i], {(uint32_t)out_buf->address(), 0u, n0});
                tt_metal::SetRuntimeArgs(program, compute, cores[i], {0u, M_block, 0u, N_block, 1u});
            } else {
                auto rh = core_noc[i] ? readerB : readerA;
                auto wh = core_noc[i] ? writerB : writerA;
                uint32_t suboff = (preaders == 1) ? 0u : core_suboff[i];
                std::vector<uint32_t> ra =
                    (preaders == 1)
                        ? std::vector<uint32_t>{(uint32_t)in1_buf->address(), core_bank[i], core_bank[i] & 0x3u}
                        : std::vector<uint32_t>{
                              (uint32_t)in1_buf->address(), core_bank[i], suboff, core_bank[i] & 0x3u};
                tt_metal::SetRuntimeArgs(program, rh, cores[i], ra);
                std::vector<uint32_t> wr = {(uint32_t)in0_buf->address(), (uint32_t)out_buf->address(), 0u, n0};
                if (bstream) {
                    wr.push_back(nloaders);  // then L * {valid_id, ready_id, loader_x, loader_y}
                    for (uint32_t l = 0; l < nloaders; ++l) {
                        auto p = device->worker_core_from_logical_core(loaders[l]);
                        wr.push_back(bs_valid[l]);
                        wr.push_back(bs_ready[l]);
                        wr.push_back(p.x);
                        wr.push_back(p.y);
                    }
                }
                tt_metal::SetRuntimeArgs(program, wh, cores[i], wr);
                tt_metal::SetRuntimeArgs(program, compute, cores[i], {0u, M_block, 0u, N_block, 1u});
            }
        }

        // Phase 2: in0 broadcast loader (reads full in0 once, mcasts to all compute cores' cb0 + ready sem)
        // over two contiguous worker bands (cols 0..colA and colA+1..gx-1, avoiding the BH x=8-9 gap).
        if (bcast || bstream) {
            // Bounding box of the compute cores (logical), split at the x=8-9 gap into <=2 contiguous bands.
            uint32_t minx = grid_all.x, maxx = 0, miny = grid_all.y, maxy = 0;
            for (auto& c : cores) {
                minx = std::min<uint32_t>(minx, c.x);
                maxx = std::max<uint32_t>(maxx, c.x);
                miny = std::min<uint32_t>(miny, c.y);
                maxy = std::max<uint32_t>(maxy, c.y);
            }
            auto phys = [&](uint32_t x, uint32_t y) { return device->worker_core_from_logical_core(CoreCoord(x, y)); };
            std::vector<uint32_t> bands;
            uint32_t nbands = 0;
            auto add_band = [&](uint32_t cx0, uint32_t cx1) {
                if (cx1 < cx0 || cx1 < minx || cx0 > maxx) {
                    return;
                }
                cx0 = std::max(cx0, minx);
                cx1 = std::min(cx1, maxx);
                auto p0 = phys(cx0, miny), p1 = phys(cx1, maxy);
                uint32_t nd = (cx1 - cx0 + 1) * (maxy - miny + 1);
                bands.push_back(nd);
                bands.push_back(std::min(p0.x, p1.x));
                bands.push_back(std::min(p0.y, p1.y));  // NOC0 normal order
                bands.push_back(std::max(p0.x, p1.x));
                bands.push_back(std::max(p0.y, p1.y));
                ++nbands;
            };
            add_band(minx, std::min<uint32_t>(6, maxx));  // band A: cols <=6 (phys x1-7)
            if (maxx >= 7) {
                add_band(std::max<uint32_t>(7, minx), maxx);  // band B: cols >=7 (phys x10-13)
            }
            if (bstream) {
                // Streaming loaders: nloaders cores, K-blocks interleaved (k%L) so all inject mcasts in parallel.
                // cb0 = D-deep ring (global K-order); num_recv = compute cores. Each loader has its own valid/ready.
                for (uint32_t l = 0; l < nloaders; ++l) {
                    std::vector<uint32_t> lct = {
                        M_block,
                        kb,
                        K_num_blocks,
                        Kt,
                        tb,
                        0u /*cb0*/,
                        (uint32_t)cores.size(),
                        bdepth,
                        8u /*banks*/,
                        nbands,
                        bs_valid[l],
                        bs_ready[l],
                        bcontig ? 1u : 0u,
                        nloaders,
                        l};
                    auto loader = tt_metal::CreateKernel(
                        program,
                        krn("in0_bstream_loader.cpp"),
                        CoreRangeSet(std::set<CoreRange>{CoreRange(loaders[l], loaders[l])}),
                        tt_metal::DataMovementConfig{
                            .processor = tt_metal::DataMovementProcessor::RISCV_0,
                            .noc = tt_metal::NOC::RISCV_0_default,
                            .compile_args = lct});
                    std::vector<uint32_t> lrt = {(uint32_t)in0_buf->address()};
                    lrt.insert(lrt.end(), bands.begin(), bands.end());
                    tt_metal::SetRuntimeArgs(program, loader, loaders[l], lrt);
                }
            } else {
                uint32_t bchunk = std::max<uint32_t>(1, std::min<uint32_t>(K_num_blocks, 8));
                std::vector<uint32_t> lct = {
                    M_block, kb, K_num_blocks, Kt, tb, 0u /*cb0*/, 0u /*unused*/, 8u /*banks*/, nbands, bchunk};
                auto loader = tt_metal::CreateKernel(
                    program,
                    krn("in0_bcast_loader.cpp"),
                    loader_cores,
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt_metal::NOC::RISCV_0_default,
                        .compile_args = lct});
                // Split the K-range across loaders: loader l owns [kstart, kstart+kcount) with its own valid sem.
                for (uint32_t l = 0; l < loaders.size(); ++l) {
                    uint32_t kstart = (l == 0) ? 0u : split_h;
                    uint32_t kcount = (l == 0) ? split_h : (K_num_blocks - split_h);
                    uint32_t vid = (l == 0) ? valid0 : valid1;
                    std::vector<uint32_t> lrt = {(uint32_t)in0_buf->address(), vid, kstart, kcount};
                    lrt.insert(lrt.end(), bands.begin(), bands.end());
                    tt_metal::SetRuntimeArgs(program, loader, loaders[l], lrt);
                }
            }
            if (other_cores.num_cores() > 0) {
                tt_metal::CreateKernel(
                    program,
                    krn("noop.cpp"),
                    other_cores,
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});
            }
        }

        auto wl = tt_metal::distributed::MeshWorkload();
        wl.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        for (uint32_t i = 0; i < num_tests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);

        // verify: every out element of the REAL [Mt,Nt] tile subregion == K. Padded tiles (m>=Mt || n>=Nt) are
        // not written/checked; in0 pad tiles are 0 so real outputs equal K regardless of padding.
        std::vector<uint32_t> outv((uint64_t)Mt_s * Nt_s * tb / sizeof(uint32_t));
        tt_metal::distributed::EnqueueReadMeshBuffer(device->mesh_command_queue(), outv, out_buf, true);
        double exp = (double)K, maxerr = 0;
        uint64_t bad = 0, checked = 0;
        for (uint32_t m = 0; m < Mt; ++m) {
            for (uint32_t n = 0; n < Nt; ++n) {
                uint64_t off = ((uint64_t)m * Nt_s + n) * words_per_tile;
                for (uint32_t wi = 0; wi < words_per_tile; ++wi) {
                    uint32_t w = outv[off + wi];
                    for (int h = 0; h < 2; ++h) {
                        float f = bf16_to_float((w >> (16 * h)) & 0xFFFF);
                        double e = std::abs(f - exp) / exp;
                        if (e > maxerr) {
                            maxerr = e;
                        }
                        if (e > 0.01) {
                            ++bad;
                        }
                        ++checked;
                    }
                }
            }
        }
        log_info(LogTest, "expected {} per elem; max_rel_err {:.4f}; bad {} / {}", exp, maxerr, bad, checked);
        pass = (bad == 0);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
