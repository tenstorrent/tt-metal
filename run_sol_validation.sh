#!/bin/bash
# One-shot validation of the Sol-Attn changes of 2026-10-20 (see sol_attn_q_handoff.md §10).
# Build first with your usual flow. Each step is timeout-wrapped and logs to its own file, so a hang in one step
# cannot block the rest; the hang reproducer under the watcher runs last, and only if the Q2 step timed out.
#
#   ./run_sol_validation.sh            # correctness steps (base, q2, prod5s) + hang reproducer if needed
#   ./run_sol_validation.sh bench      # timed op benchmark: dense vs compact vs pack at 5 s / 10 s / 15 s
#   ./run_sol_validation.sh floor      # step 2 of sol_attn_kv_once_plan.md: the data-movement floor
#   ./run_sol_validation.sh gather     # gather mode at tau 1/2/3 (plan §7)
#   ./run_sol_validation.sh pages      # 2- vs 4-deep K/V buffers at 256-key pieces (plan §8c)
#   ./run_sol_validation.sh profile    # per-visit zones of the inner step, 10 s, dense + sparse (plan §8c)
#   ./run_sol_validation.sh profile_sparse   # the sparse half only (the decisive one; ~3x smaller CSV)
#   ./run_sol_validation.sh noprefix   # sparse op with the dense-prefix rule off: the straggler test (plan §10)
#   ./run_sol_validation.sh resident   # M1 resident mode: all mode-parametrized correctness tests, then the 5 s bench
#   ./run_sol_validation.sh qchunk_diag    # base-kernel determinism/sanity at q 64/128/256 (the q64 path is broken)
#   ./run_sol_validation.sh resident_diag   # resident error maps + q64 base determinism (~4 min)
#   ./run_sol_validation.sh resident_watch  # resident's failing geometries under the watcher (ASSERTs, CB/NoC faults)
#   ./run_sol_validation.sh resident2  # the 5 s timing alternation: dense / compact / resident with more reps + a profile
#   ./run_sol_validation.sh resident3  # 2026-10-21: nightly dense accuracy (all-gather DRAM-bank fix), resident
#                                      # per-call parity check with fresh inputs, profiled 5 s resident bench
#   ./run_sol_validation.sh prefix     # the separate prefix-row recompute: correctness at 5 s, then the timed
#                                      # bench with it (plan §10c).
#   ./run_sol_validation.sh m2         # M2 block-level resident mode: correctness vs compact, 5 s parity + timing,
#                                      # 5 s bench at tau 1/2/3 with the block density per call.
#   ./run_sol_validation.sh m2b        # head-boundary-core fix check (1184 + 5 s), profiled 5 s tau 1 rep, tau 1 bench.
#   ./run_sol_validation.sh m3         # M3 (in-place slot state, block-major steps): correctness, 5 s parity, tau 1/2/3
#                                      # bench, profiled sparse + dense reps (done 2026-10-22: correct, no speedup).
#   ./run_sol_validation.sh m4         # M4 (a-priori static row max): done 2026-10-22, tau 1 5.85 ms but PCC 0.992.
#   ./run_sol_validation.sh m5         # M5 (frozen first-step max): correctness, 5 s parity, tau 1/2/3 bench, exp A/B.
#   ./run_sol_validation.sh m6         # M6 (fused per-piece step, row sum as a matmul, running max always):
#                                      # correctness, 5 s parity, tau 1/2/3 bench, fine-zone profile. Rebuild first.
#                                      # (rebuild: the factory's score-scratch size changed).
#   ./run_sol_validation.sh q2         # any single step: base | q2 | prod5s | hang | bench | floor | gather
set -u
cd "$(dirname "$0")"
T=models/tt_dit/tests/models/minimax_h3/test_sol_compact_minimax_h3.py
B=models/tt_dit/tests/models/minimax_h3/test_sol_bench_minimax_h3.py
ONLY=${1:-all}

run() {
    local name=$1 secs=$2; shift 2
    echo "=== $name  (log_val_$name.txt)"
    timeout -k 30 "$secs" "$@" > "log_val_$name.txt" 2>&1
    local rc=$?
    echo "    exit $rc  (124 = timeout)"
    return $rc
}
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

# 1. Baseline modes, the resolved tail-block tests, Q1 granularity. Expected: all pass.
want base && run base 1800 pytest "$T" -s -k "4x8 and not q2 and not production and not timing"

# 2. Q2 (sol_mode=pack). Includes the compact q_chunk=64 run that hung before. Expected: all pass.
q2_rc=0
want q2 && { run q2 1500 pytest "$T" -s -k "4x8 and q2"; q2_rc=$?; }

# 3. Production 5 s shape: compact/chunk/gather at q_chunk=320, plus Q1+Q2 (pack at q_chunk=64, k_chunk=384).
want prod5s && run prod5s 1500 env SOL_COMPACT_FULL=1 pytest "$T" -s -k "4x8 and production_5s"

# 4. Only if step 2 wedged: the reproducer under the watcher, with the chain/multicast debug log on.
if [ "$ONLY" = hang ] || { [ "$ONLY" = all ] && [ "$q2_rc" = 124 ]; }; then
    run hang 900 env TT_METAL_WATCHER=5 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_LOGGER_LEVEL=Debug \
        pytest "$T" -s -k "4x8 and q2_packing_matches_unpacked and nl2"
    echo "    watcher log: generated/watcher/watcher.log  (look at waypoints on the CCL column and ETH cores too)"
fi

# NOTE: the bench's test function is named test_sol_bench_attention, so `-k sol` matches every variant; the sparse
# steps select by excluding dense and agather instead (also keeps the profiler CSVs ~3x smaller).
# 5. Timed op benchmark (one attention layer, op-only time between host syncs, 3 reps): the 2x gate. Needs the
#    MiniMax-H3 diffusers patch. dense and compact at the production chunk sizes; pack at q_chunk 64 with the
#    production k_chunk (384 at 5 s, 512 at 10 s / 15 s). SOL_TAU brackets the density; the log prints the achieved
#    density next to every time. Filters use the full ids (768p_5s etc.): a bare "5s" also matches "15s".
if want bench; then
    TAUS=${SOL_TAU:-1.0,2.0,3.0}
    run bench_dense   1200 pytest "$B" -s -k "4x8 and nl2 and dense"
    run bench_compact 1800 env SOL_BENCH_MODE=compact SOL_TAU="$TAUS" pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
    run bench_pack5s  1200 env SOL_BENCH_MODE=pack SOL_BENCH_CHUNKS=64x384 SOL_TAU="$TAUS" \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    run bench_pack    1800 env SOL_BENCH_MODE=pack SOL_BENCH_CHUNKS=64x512 SOL_TAU="$TAUS" \
        pytest "$B" -s -k "4x8 and nl2 and not 768p_5s and not dense and not agather"
fi
# 5b. Same measurement without the diffusers dependency (test_sol_op_timing in the standalone file), on randn
#     data (unstructured mask, density ~0.16 / ~0.10) and the ramp. One shape per invocation.
if want bench_standalone; then
    for shape in 5s 10s 15s; do
        run "bench_$shape" 1500 env SOL_TIMING=1 pytest "$T" -s -k "4x8 and nl2 and timing and $shape"
    done
fi

# 6. Step 2 of sol_attn_kv_once_plan.md: is the op time pure data movement? Compute drains every piece without
#    computing it (TT_SOL_DRAIN_ALL) and the averaged-block step is dropped (TT_SOL_POOLED_OFF); numerics are wrong
#    by design. floor_nopooled = reader/chain floor alone; floor_pooled = the same plus the averaged-block step.
#    Compare with the sol/compact lines in log_val_bench_compact.txt (tau 3.0 rows).
if want floor; then
    run floor_nopooled 1200 env SOL_BENCH_MODE=compact SOL_TAU=3.0 TT_SOL_DRAIN_ALL=1 TT_SOL_POOLED_OFF=1 \
        pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
    run floor_pooled 1200 env SOL_BENCH_MODE=compact SOL_TAU=3.0 TT_SOL_DRAIN_ALL=1 \
        pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
    # Second experiment (plan §5): fewer visits per core via larger query chunks, baseline sparse, tau 3.
    # Predicted at 10 s: 320 rows ~22 ms, 384 rows ~16 ms (or an L1 allocation error), against 27 ms at 256.
    run floor_q320 1200 env SOL_BENCH_MODE=compact SOL_TAU=3.0 SOL_BENCH_CHUNKS=320x512 \
        pytest "$B" -s -k "4x8 and nl2 and not 768p_15s and not dense and not agather"
    run floor_q384 1200 env SOL_BENCH_MODE=compact SOL_TAU=3.0 SOL_BENCH_CHUNKS=384x512 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_10s and not dense and not agather"
fi

# 7. Gather mode: no chain, each core reads only its kept blocks from DRAM and packs them into full pages.
#    Rejected earlier at density 0.3 to 0.6 (DRAM traffic); at 0.05 to 0.15 its traffic is 1 to 3 GB per chip per
#    layer. Predicted (plan §7): 10 s ~8-10 ms at tau 2, ~6-7 at tau 3; 5 s ~3.5-4.5; 15 s ~13-15 at tau 2.
if want gather; then
    run gather 1800 env SOL_BENCH_MODE=gather SOL_TAU=1.0,2.0,3.0 pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
fi

# 8. Deeper K/V buffering (plan §8): does a 3- or 4-chunk-deep K/V buffer let the reader and compute overlap
#    across the ~10-core chain? Baseline sparse at tau 3 plus dense, all shapes. If the sparse time at 10 s drops
#    from ~27 toward ~17 ms, the per-visit cost is reader PLUS compute today and the extra pages decouple them.
#    An L1 allocation error at 4 pages (or at 15 s) is expected and informative; 3 pages should fit.
#    2026-10-21: 3 pages at production chunks overflow L1 (1.636 MB vs 1.573; the base set is already 1.38 MB).
#    So the depth comparison runs at 256-key pieces instead: 2 pages (today's depth, half the bytes) against 4
#    pages (same buffer bytes as production). Same visits in both, so any gap is the buffering depth alone.
if want pages; then
    for p in 2 4; do
        run "pages${p}_sparse" 1500 env TT_SOL_KV_PAGES=$p SOL_BENCH_MODE=compact SOL_TAU=3.0 SOL_BENCH_CHUNKS=256x256 \
            pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
        run "pages${p}_dense" 900 env TT_SOL_KV_PAGES=$p SOL_BENCH_CHUNKS=256x256 \
            pytest "$B" -s -k "4x8 and nl2 and dense"
    done
fi

# 9. Per-phase profile of the inner step (plan §8c). The step's zones (Q@KT MM+Pack, Reduce max, SUB/EXP,
#    QKT@V MM+Pack, S_CORR_FUSED, ROW_NORM) light up under the device profiler when TT_SOL_STEP_PROFILE=1.
#    TT_SOL_STEP_PROFILE=1 = one zone per visit (RING_STEP / RING_DRAIN); =2 adds the phase zones; =3 the fine ones.
#    10 s only, dense and sparse at tau 3, 1 rep. Output: generated/profiler/.logs/profile_log_device.csv;
#    your parse_kernel_profile.py (see summarize_profile.sh) with --filter on the zone names gives per-zone
#    totals per core. The question: which zones of a 1-to-2-block step cost what they cost in a full step.
if want profile; then
    run profile_dense 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        pytest "$B" -s -k "4x8 and nl2 and dense and 768p_10s"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_dense_logs" 2>/dev/null
fi
if want profile || want profile_sparse; then
    # SOL_TAU, SOL_BENCH_PREFIX (fused|separate|off) and TT_SOL_STEP_PROFILE (1 = per-visit, 2 = + phases, 3 = + fine)
    # pass through: e.g. SOL_BENCH_PREFIX=off SOL_TAU=2.0 ./run_sol_validation.sh profile_sparse
    run profile_sparse 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE="${TT_SOL_STEP_PROFILE:-1}" SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=compact SOL_TAU="${SOL_TAU:-3.0}" SOL_BENCH_PREFIX="${SOL_BENCH_PREFIX:-fused}" \
        pytest "$B" -s -k "4x8 and nl2 and 768p_10s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_sparse_logs" 2>/dev/null
    echo "    profiler logs copied to profile_dense_logs/ and/or profile_sparse_logs/"
fi

# 10. The straggler test (plan §10). Profiling showed 28 of 32 chips finish the sparse op in ~9 ms at 10 s while
#     the 4 shard-0 chips take 26.6 ms: the dense-prefix rule gives ~15 cores there fully dense query chunks and
#     the chain lockstep drags the whole chip to dense pace. This runs the sparse op with that rule off, all three
#     shapes, tau 2 and 3. Prediction: 10 s ~9 to 11 ms (dense 29), 15 s ~20 to 25 ms (dense 66), 5 s ~3 to 4 ms.
# 10c. The separate prefix-row recompute (MiniMaxH3Attention._sol_prefix_dense_rows, the bench default now):
#   prefix_check: one layer at 5 s, dense vs off/fused/separate; prefix rows must match dense (PCC > 0.995),
#                 the other rows must be identical across the three sparse variants.
#   prefix_bench: the timed bench, tau 2 and 3; the log line shows the recompute's ms inside the op time.
#   (SOL_PREFIX_SEQ=9184 ./run_sol_validation.sh prefix checks the 10 s shape instead.)
if want prefix; then
    run prefix_check 1200 env SOL_TAU=2.0 pytest models/tt_dit/tests/models/minimax_h3/test_sol_prefix_minimax_h3.py -s -k "4x8 and nl2"
    run prefix_bench 1800 env SOL_BENCH_MODE=compact SOL_BENCH_PREFIX=separate SOL_TAU=2.0,3.0 \
        pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
fi

# 11. M1 resident mode (sol_attn_status_2026-10-21.md 4a): one pass of each K/V piece per core per ring iteration,
#     applied to every resident 64-row query chunk from L1. resident_check: output must match chunk mode at
#     q_chunk=64 (same mask, same math). resident_bench: the timed op at 5 s, tau 1/2/3, prefix recompute off.
if want resident; then
    run resident_check 1500 pytest "$T" -s -k "4x8 and resident"
    run resident_bench 1800 env SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0,2.0,3.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
fi

# 12. The 2026-10-21 resident bench alternated 12 / 4.3 / 11.5 ms over its three reps at 5 s. Same machine, three
#     modes, more reps: if dense alternates too it is the machine; if only resident does, it is the kernel.
#     resident_profile: per-run kernel spans (level-1 zones) of four resident reps, to see which RISC is long.
# 13. Resident under the watcher: the 2026-10-21 run produced garbage (PCC 0) at rows 1184 / k_chunk 256 (partial
#     last K and Q chunks) and 0.99935 instead of >0.9995 against dense at full density. The kernels' ASSERTs
#     (first-piece state creation, chunk counts) only fire under the watcher; a CB or NoC fault shows there too.
if want resident_watch; then
    # Assert-only watcher: the full instrumentation pushed the resident kernels past the 70,656 B kernel config buffer
    # (71,600 B on 2026-10-21); the sanitizers, pause, ring buffer, stack usage and waypoints are switched off.
    run resident_watch 1500 env TT_METAL_WATCHER=5 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
        TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 \
        TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 TT_METAL_WATCHER_DISABLE_WAYPOINT=1 \
        pytest "$T" -s -k "4x8 and resident and (matches_compact or small_k_chunk or ragged or determinism or dense_equivalence)"
fi

# 14. Resident diagnostics only (~4 min): the 8-geometry resident-vs-chunk maps (now also chunk-vs-dense and
#     resident-vs-dense, plus per-shard NaN / std per tensor) and the q_chunk=64 determinism of the base kernels.
#     Then three A/Bs on the clean cases (every piece kept; rows 512 = one resident chunk per core, rows 1024 =
#     two, rows 1184 = partial chunks): chains off (every core reads its own K/V; a chain-protocol fault vanishes),
#     pooled step off (a bias/pooled-entry fault vanishes), and the reader's per-iteration trace on a few cores of
#     three chips (push counts per iteration; a phase fault shows as neighbours disagreeing). Then 6 timed reps.
#     Each is its own process because these knobs are not in the program hash.
# 15. Which query chunk is the base kernel clean at? Dense + chunk mode twice each at q 64 / 128 / 256, rows 1024 and
#     1184 (~5 min). 2026-10-21: q64 is nondeterministic (dense differs run to run by 0.36) and returns ~1e33 at
#     rows 1184 -- every 64-row measurement sat on that. Resident follows SOL_RESIDENT_Q_CHUNK (default 64).
if want qchunk_diag; then
    run qchunk_diag 900 pytest "$T" -s -k "4x8 and base_determinism"
fi

if want resident_diag; then
    run resident_diag 900 pytest "$T" -s -k "4x8 and (resident_matches_chunk or q64_base_determinism or determinism and resident)"
    AB="4x8 and resident_matches_chunk and keepall and (r512k512 or r1024k512 or r1184k512)"
    run resident_diag_nochain 600 env TT_SOL_NO_CHAIN=1 pytest "$T" -s -k "$AB"
    run resident_diag_nopooled 600 env TT_SOL_POOLED_OFF=1 pytest "$T" -s -k "$AB"
    run resident_diag_trace 600 env TT_SOL_DEBUG=1 TT_METAL_DPRINT_CORES="(0,0)-(7,0)" TT_METAL_DPRINT_CHIPS="0,2,3" \
        TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1 pytest "$T" -s -k "4x8 and resident_matches_chunk and keepall and r1024k512"
    run resident_diag_timing 600 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=2.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
fi

if want resident2; then
    run resident2_dense 900 env SOL_BENCH_REPS=6 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    run resident2_compact 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=compact SOL_BENCH_PREFIX=off SOL_TAU=2.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    run resident2_resident 900 env SOL_BENCH_REPS=8 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=2.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    run resident_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=4 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=2.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    rm -rf profile_resident_logs
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_resident_logs" 2>/dev/null
    echo "    resident profiler logs copied to profile_resident_logs/"
fi

if want resident3; then
    # a) the dense ring op's own nightly accuracy test: the all-gather DRAM-bank fix for chips with different
    #    harvested DRAM banks (PCC 0.60 before; expect >= 0.994)
    run resident3_nightly 600 pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py -s \
        -k "logical_tensor_accuracy and noncausal"
    # b) resident vs chunk per call with fresh inputs every call: is the fast 4.3 ms parity a real answer?
    run resident3_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity"
    # c) the 5 s resident bench under the profiler with the new reader/writer/all-gather wait zones
    run resident3_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=4 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=2.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    rm -rf profile_resident3_logs
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_resident3_logs" 2>/dev/null
    echo "    resident3 profiler logs copied to profile_resident3_logs/"
fi

if want resident4; then
    # a) the all-gather alone, page by page, plus every chip's DRAM harvesting mask and bank table (~1 min)
    run resident4_ag 600 pytest models/tt_dit/tests/models/minimax_h3/test_ag_dram_banks.py -s -k "4x8"
    # b) resident vs chunk per call with fresh inputs at the real 5 s geometry (4736 rows, 384-key pieces)
    run resident4_parity 1200 pytest "$T" -s -k "4x8 and resident_call_parity"
fi

if want resident5; then
    # Resident op time against mask density at 5 s, density logged per call (2026-10-21: the 11.9 / 4.3 ms
    # alternation is two different masks; the profiler buffer had hidden it). tau sweep, 6 reps each.
    for tau in 1.0 2.0 3.0 4.0; do
        run "resident5_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
fi

if want noprefix; then
    run noprefix 1500 env SOL_BENCH_MODE=compact SOL_BENCH_PREFIX=off SOL_TAU=2.0,3.0 \
        pytest "$B" -s -k "4x8 and nl2 and not dense and not agather"
fi

# M2 (block-level exact sets in resident mode): resident takes the block-form mask and each 64-row chunk multiplies
# only its kept 64-key blocks of a piece, in place. a) correctness against compact mode (same mask, same exact set)
# over the small geometries, then dense equivalence at tau -1e4; b) per-call parity + timing at the real 5 s shape;
# c) the 5 s bench, tau sweep, 6 reps, block density logged per call (equals "density" now: the mask is block-form).
if want m2; then
    # The dense op's accuracy first (~30 s): the fused all-gather's backward writers mis-sized their op-signaler
    # args, so the neighbour bank table was misread and the backward half of the ring never landed (PCC 0.71,
    # probe of 2026-10-21). Fixed in the all-gather factory; this must now pass at 0.994.
    run m2_nightly 600 pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py -s -k "logical_tensor_accuracy and noncausal"
    # The floors on THIS machine: dense op and the all-gather alone at 5 s (the 8.5 / 3.2 ms numbers are from
    # the first, 130 W machine; the 180 W part speeds the compute-bound dense op more than the gather-bound sparse op).
    run m2_floor 900 env SOL_BENCH_REPS=6 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and (dense or agather)"
    run m2_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m2_parity 1200 pytest "$T" -s -k "4x8 and resident_call_parity"
    for tau in 1.0 2.0 3.0; do
        run "m2_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
fi

# M2b (2026-10-21, after the chain-off A/B): head-boundary cores are out of the chains in resident mode (their K/V
# CB pointer drifted from the chain's and the forward landed on the entry compute was reading). a) the two failing
# geometries again (~4 min); b) the per-event costs at 5 s, tau 1: one profiled rep with the level-1 zones
# (compute RING_STEP / RES_ROT / RES_DRAIN / RES_NORM, reader RD_CHAIN_TX / RD_CHAIN_RX / RD_RES_PIECES / RD_SYNC).
# The profiler keeps ~125 zones per RISC per core, so read it as "the first ~100 events of the op", which is enough
# for per-event costs; logs -> profile_m2b_logs/. c) the 5 s bench at tau 1 (the target: 3.85 ms = 2x of 7.7).
if want m2b; then
    run m2b_check 900 pytest "$T" -s -k "4x8 and resident_matches_compact and r1184 and keepall"
    run m2b_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    run m2b_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m2b_logs" 2>/dev/null
    run m2b_tau1.0 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
fi

# M3 (slot-indexed in-place state + block-major steps): a) correctness over the small geometries and dense
# equivalence; b) 5 s parity (finite, vs compact); c) the 5 s bench at tau 1/2/3; d) one profiled tau 1 rep (per
# piece cost; the RING_STEP zone is now one per block-step) and e) one profiled DENSE rep at 5 s, so the host-minus-
# device overhead of both ops on this machine is known (the 2x target is a host-time ratio).
if want m3; then
    run m3_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m3_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m3_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m3_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m3_logs" 2>/dev/null
    run m3_profile_dense 900 env TT_METAL_DEVICE_PROFILER=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m3_dense_logs" 2>/dev/null
fi

# M4 (static per-row max: the pre-pass supplies ||q|| x max||k|| + ln(64)/scale per query row as sol_row_max, the
# reader loads it into the max slots, the step drops the row-max reduce, the alpha exp and the in-place rescale).
# a) the bound itself on host (rows 512: every row's bound above its true max logit); b) the M3 correctness set;
# c) 5 s parity; d) the 5 s bench at tau 1/2/3; e) profiled sparse + dense reps. Dynamic-max A/B: the M3 path is
# what runs when sol_row_max is absent (the op rejects that in resident mode now, so the A/B is the m3 logs).
if want m4; then
    run m4_bound 600 pytest "$T" -s -k "4x8 and sol_row_max_bound"
    run m4_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m4_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m4_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m4_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m4_logs" 2>/dev/null
    run m4_profile_dense 900 env TT_METAL_DEVICE_PROFILER=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m4_dense_logs" 2>/dev/null
fi

# M5 (frozen max): resident mode now takes each chunk's row max on its first step and freezes it (the M4 a-priori
# bound was exact but too loose for bf16: PCC 0.9918 at 1184 rows). a) correctness vs compact, b) 5 s parity,
# c) the 5 s bench at tau 1/2/3, d) the same tau 1 bench with the approximate exp (SOL_EXP_APPROX=1: the step is
# now bound by the exp on the pack thread; the PCC of that variant comes from e) the 1024-row check with it on).
if want m5; then
    run m5_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m5_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m5_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m5_tau1.0_expapprox 900 env SOL_EXP_APPROX=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off \
        SOL_TAU=1.0 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    run m5_check_expapprox 900 env SOL_EXP_APPROX=1 pytest "$T" -s -k "4x8 and resident_matches_compact and r1024k512"
fi

# M6 (fused per-piece step + row sum as a matmul): per piece one step per sub-group of up to 4 chunks, each with its
# own kept blocks (the output accumulator is packed once per chunk per piece instead of per block) and the row sum
# comes from P x e0 (the column-identity tile, now zeroed by the writer before column 0 is written) instead of one
# L1-accumulate pack per probability tile.
# M5 result (2026-10-22): the frozen max was WRONG (PCC 0.69-0.82 everywhere): the pack-thread approximate exp
# saturates at input +0.72 (ckernel_sfpu_exp.h), so every key scoring above the frozen max got weight ~2. The
# frozen path and its TT_SOL_DYNAMIC_MAX switch are removed: resident compute always runs the running max (row
# max merged with the stored max once per chunk per piece, inside the fused step). The first m6 run (2026-10-22,
# before the removal) measured the frozen variant: tau 1 5.62 ms, correctness failing like m5; its logs are
# superseded by this run. The SOL_EXP_APPROX A/B of m5 is dropped: the resident step's exp is the pack-thread
# approximate exp regardless of that flag (the flag only reaches the alpha exp and the dense/compact paths).
# Rebuild first (compute kernel template and factory named args changed).
if want m6; then
    run m6_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m6_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m6_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    # Fine-zone profile at tau 1: the fused step cut only 0.16 ms off the frozen floor (5.78 -> 5.62 ms), so the
    # next lever has to come from where the ~5 us per (chunk, piece) actually goes. Please copy profile_m6_logs/.
    run m6_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m6_logs" 2>/dev/null
fi

# M7 (Tier 1, 2026-10-22): lazy rescale (the step rescales only when a row's max grew by more than 0.5/scale;
# decided on device from a column-max reduce of new - stored max), P@V and row-sum matmuls batched across the
# group's chunks (one setup per group), pooled entries stepped in sub-groups of 4 chunks. Same set as m6 plus the
# 47 % parity mask timing. Rebuild first.
if want m7; then
    run m7_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m7_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m7_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m7_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m7_logs" 2>/dev/null
fi

# M7 experiment (2026-10-22, after the M6 profile): how much of the op is the chain's lockstep? The profile shows
# cores spending 13-40 % of every ring iteration outside the compute steps, and chains ending between 3.4 and
# 4.9 ms on the same chip. TT_SOL_NO_CHAIN=1 makes every core fetch its own pieces (more NoC/DRAM traffic, no
# lockstep): the tau 1 time with it bounds what decoupling the chain could buy. Plus one 1024-row check that the
# no-chain path is still exact. No rebuild needed (factory-time switch).
if want m7x; then
    run m7x_nochain_tau1.0 900 env TT_SOL_NO_CHAIN=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off \
        SOL_TAU=1.0 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    run m7x_nochain_tau2.0 900 env TT_SOL_NO_CHAIN=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off \
        SOL_TAU=2.0 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    run m7x_nochain_check 900 env TT_SOL_NO_CHAIN=1 pytest "$T" -s -k "4x8 and resident_matches_compact and r1024k512"
    # Deeper K/V CB (TT_SOL_KV_PAGES, factory knob, +192 KB L1 per page): lets chain members drift apart by more
    # than one piece. An L1 allocation failure here just means the depth does not fit at 5 s.
    for pages in 3 4; do
        run "m7x_kv${pages}_tau1.0" 900 env TT_SOL_KV_PAGES=$pages SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=1.0 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
fi

# M7 experiment 2 (2026-10-22, after the M6 reader zones): the chain's per-piece delivery (25-35 us per piece on
# chain members vs ~13 us on the standalone fetchers, which end 1.4 ms earlier) paces 84 % of the cores. The chain
# is unicast store-and-forward with a one-piece handshake per hop, so its cycle grows with the chain length.
# TT_SOL_CHAIN_LEN=L splits every head chain into segments of at most L cores, each with its own DRAM injector
# (L x more K/V DRAM reads; multicast off). L = 4 and 2 at tau 1 measure the length effect; L = 1 is the no-chain
# case already measured (9.85 ms). No rebuild needed.
if want m7x2; then
    for len in 4 2; do
        run "m7x2_chainlen${len}_tau1.0" 900 env TT_SOL_CHAIN_LEN=$len SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=1.0 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m7x2_chainlen4_check 900 env TT_SOL_CHAIN_LEN=4 pytest "$T" -s -k "4x8 and resident_matches_compact and r1184k512"
fi

# M8 (chain overlap, 2026-10-22 night): a chain member now signals "ready" for piece p+1 right after pushing p and
# before forwarding p (chain_link.hpp signal_ready/wait_data; reader stream_piece with lookahead), so the upstream
# write of p+1 overlaps this core's forward of p instead of waiting for it. m7x2 showed shorter chains are slower
# (DRAM reads scale with the injector count: L=4 6.52 ms, L=2 8.05, L=1 9.85), so the chain stays long and the
# per-hop handshake is what changes. Same set as m7. Rebuild first (reader kernel changed).
if want m8; then
    run m8_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m8_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m8_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m8_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m8_logs" 2>/dev/null
    # The dense nightly must still pass: the chain protocol change touches a shared header (dense keeps receive()).
    run m8_dense_check 1200 pytest "$T" -s -k "4x8 and test_dense_equivalence"
fi

# M8 follow-ups (2026-10-22, 22:00). M8 measured 6.12 ms at tau 1 (M7 6.16): the early ready signal is only sent
# when the next page is already free, and with a 2-deep K/V CB that page (holding piece p-1) is still being
# computed when p arrives, so the arm rarely fires. Depth 3 alone gave nothing (m7x: the handshake serialized);
# the new handshake alone gives nothing (the buffer serializes). Together they should: m8x runs the M8 kernel
# with TT_SOL_KV_PAGES=3 (fits: ~73 KB of L1 spare), tau 1/2 and a 1024-row check. No rebuild needed.
if want m8x; then
    for tau in 1.0 2.0; do
        run "m8x_kv3_tau${tau}" 900 env TT_SOL_KV_PAGES=3 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off \
            SOL_TAU=$tau pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m8x_kv3_check 900 env TT_SOL_KV_PAGES=3 pytest "$T" -s -k "4x8 and resident_matches_compact and r1024k512"
    run m8x_kv3_profile 900 env TT_SOL_KV_PAGES=3 TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 \
        SOL_BENCH_REPS=1 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m8x_logs" 2>/dev/null
fi
# The dense equivalence check on its own (the m8 run of it was interrupted).
if want m8d; then
    run m8d_dense_check 1200 pytest "$T" -s -k "4x8 and test_dense_equivalence"
fi

# M9 (Tier 2, 2026-10-22 night): the compact per-piece step. Score scratch in a compact layout (each chunk's rows
# are exactly its kept width wide), so one step covers every chunk of the head that keeps a block of the piece;
# adjacent kept blocks merged into one run (wider matmul calls, blocked packs when >= 4 tiles); P@V at the widest
# DEST sub-block; members sorted by width so the width-dependent MOP setups are re-issued only on change. Same
# correctness set, parity, tau 1/2/3, profile. Rebuild first (compute kernel changed).
if want m9; then
    run m9_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m9_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m9_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m9_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m9_logs" 2>/dev/null
fi
# M9 ablation sweep (2026-10-23): the compact step did not move tau 1 (6.12 ms = M8), so the per-tile cost is
# measured directly: each run skips one stage's instructions (TT_SOL_ABLATE, results wrong, timing valid) and
# the drop against the baseline is that stage's share. qk = Q@K^T matmuls, exp = the SFPU exp, pack5 = the exp
# stage's packs, pv = P@V and row-sum matmuls, stats = row max / test / alpha / rescale, pack1 = the score packs.
# Each is a factory define (one kernel compile each, ~20 s per run). Rebuild first (compute kernel changed).
if want m9a; then
    run m9a_base_tau1.0 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    for knob in qk exp pack5 pv stats pack1; do
        run "m9a_${knob}_tau1.0" 900 env TT_SOL_ABLATE=$knob SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=1.0 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
fi

# Traced timings (2026-10-23): SOL_BENCH_TRACE=1 opens the mesh with a trace region and times the op as a captured
# trace replayed 6 times, i.e. device time + one synchronize and no per-call dispatch (the production model runs
# under trace). Dense, the gather alone, and resident at tau 1/2/3: the comparison that matters end to end, and
# the split of the untraced 6.1 ms into device time and dispatch. No rebuild needed.
if want trace; then
    run trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    run trace_agather 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and agather"
    for tau in 1.0 2.0 3.0; do
        run "trace_tau${tau}" 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off \
            SOL_TAU=$tau pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
fi

# Production acceptance (2026-10-23) for the clean branch sosborne/sol-attn-q-opt: the resident correctness set
# (vs compact, vs dense), the 5 s per-call parity, then the traced timings (dense, gather alone, resident at
# tau 1/2/3). No profile, no untraced bench. Rebuild first (compute kernel + factory changed). Expected: 11/11
# checks at PCC >= 0.999, trace_dense ~7.4 ms, trace_tau1.0 ~5.0 ms (1.48x), tau 2 ~4.0, tau 3 ~3.4.
if want prod; then
    run prod_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run prod_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    run prod_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    run prod_trace_agather 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and agather"
    for tau in 1.0 2.0 3.0; do
        run "prod_trace_tau${tau}" 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=$tau pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
fi

# M10 (perf branch sosborne/sol-attn-perf, 2026-10-23): the compact step's exp stage and P@V interleaved per member
# (P@V of the members finished in S5 batch k under the exp/packs of batch k+1; one PACK_DONE token per S5 batch,
# PACK_DONE capacity 15 in resident mode). Correctness set + 5 s parity, then the traced timings (dense for the
# same-session baseline, resident tau 1/2/3), then a profile. Rebuild first (compute kernel changed). Expected:
# 11/11 at PCC >= 0.999; trace_tau1.0 ~4.6-4.75 ms (from 4.97; dense ~7.4).
if want m10; then
    run m10_check 1500 pytest "$T" -s -k "4x8 and (resident_matches_compact or resident_dense_equivalence)"
    run m10_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    run m10_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    for tau in 1.0 2.0 3.0; do
        run "m10_trace_tau${tau}" 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=$tau pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m10_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m10_logs" 2>/dev/null
fi

# M11 (perf branch, 2026-10-23): the chain's counting (credit) protocol over a 3-deep K/V CB: a member grants a
# credit per K/V page it can reserve ahead (ChainLink::grant), the upstream forwards on credits
# (forward_credited) and bumps an arrival counter; two pieces can be in flight per link while the core computes a
# third. The 1024-row check (two pieces per iteration, the shape that caught M8's hang) and the 512-row check run
# first, then the 5 s parity, the traced timings and a profile. Rebuild first (reader kernel + factory changed).
# TT_SOL_CHAIN_CREDITS=0 is the A/B back to the M8 handshake (2 pages). A hang shows as exit 124 on a step.
if want m11; then
    run m11_check 900 pytest "$T" -s -k "4x8sp1tp0nl2 and resident_matches_compact and (r512 or r1024)"
    run m11_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    run m11_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    for tau in 1.0 2.0 3.0; do
        run "m11_trace_tau${tau}" 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=$tau pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m11_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m11_logs" 2>/dev/null
fi

# M11 A/B (2026-10-23, after the first m11 run): the 512/1024-row checks passed (PCC 0.9999) but the 5 s parity
# failed at PCC 0.973-0.979 on all four calls (finite outputs, so a wrong page rather than a hang or NaN). The
# small geometries never fill the lookahead (one or two pieces per iteration, two chunks per core), so the fault
# needs the real shape. Each run is one ~2 min parity at 5 s; read them together:
#   m8        credits off (M8 handshake, 2 pages)            -> must pass: the environment and build are sane
#   m8_kv3    credits off, 3 pages                            -> fails: the 3-deep CB itself is the fault
#   cred_kv2  credits on, 2 pages (one piece in flight)       -> fails: the credit protocol; passes: needs 2 in flight
#   r1184     credits on, 3 pages, the 1184-row check         -> a third geometry (partial chunks, 19 chunks/head)
#   watch     credits on, 3 pages, 5 s parity under the assert-only watcher: the stream's ASSERTs (chain
#             membership per piece, grant-cursor phase) and ChainLink's unicast-only assert only fire here;
#             generated/watcher/watcher.log is copied to log_val_m11ab_watcher.log.
if want m11ab; then
    P="4x8 and resident_call_parity and r4736"
    run m11ab_m8 600 env TT_SOL_CHAIN_CREDITS=0 pytest "$T" -s -k "$P"
    run m11ab_m8_kv3 600 env TT_SOL_CHAIN_CREDITS=0 TT_SOL_KV_PAGES=3 pytest "$T" -s -k "$P"
    run m11ab_cred_kv2 600 env TT_SOL_KV_PAGES=2 pytest "$T" -s -k "$P"
    run m11ab_r1184 900 pytest "$T" -s -k "4x8sp1tp0nl2 and resident_matches_compact and r1184k512"
    run m11ab_watch 900 env TT_METAL_WATCHER=5 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
        TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 \
        TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 TT_METAL_WATCHER_DISABLE_WAYPOINT=1 \
        pytest "$T" -s -k "$P"
    [ -f generated/watcher/watcher.log ] && cp generated/watcher/watcher.log "log_val_m11ab_watcher.log" 2>/dev/null
fi

# Design gate D0 (2026-10-23, sparse_attn_design_2026-10-23.md §2): traced dense and the gather alone in the same job,
# at the power the 2x target is judged at (180 W). If 2 x (gather + 0.25 ms) > dense, 2x at 5 s is out of reach for
# any in-chip design while the gather stays as it is. No rebuild needed.
if want design0; then
    run design0_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    run design0_trace_agather 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and agather"
fi

# M11 race fix (2026-10-23). m11ab: credits off passed at 2 and 3 pages, credits on failed at 3 pages (0.973) and
# worse at 2 (0.938). Cause: a member granted the page of a chunk it had pushed but not yet forwarded (compute pops
# an empty piece at once), so the upstream overwrote the chunk before the forward read it; the reserve now also
# counts the pages pending a forward. Run at 180 W; every timing compares only against this job's own runs (the
# m11 / m11ab runs were on a 130 W machine). Correctness first, and the job stops there on any failure:
#   check / r1184     512 / 1024 / 1184-row resident vs compact (credits, 3 pages)
#   par_kv3 / par_kv2 5 s parity, credits on at 3 and 2 pages: >= 0.9996 on every call means the race was the bug
# Then traced tau 1 (6 replays): dense (the 2x denominator), the M8 handshake at 2 pages (the baseline), credits at
# 2 and 3 pages; tau 2 / 3 for credits at 2 pages; one profile, credits at 2 pages (chain idle per core against the
# M9 profile's ~1.0 ms).
if want m11fix; then
    P="4x8 and resident_call_parity and r4736"
    ok=1
    run m11fix_check 900 pytest "$T" -s -k "4x8sp1tp0nl2 and resident_matches_compact and (r512 or r1024)" || ok=0
    run m11fix_r1184 900 pytest "$T" -s -k "4x8sp1tp0nl2 and resident_matches_compact and r1184k512" || ok=0
    run m11fix_par_kv3 600 env TT_SOL_KV_PAGES=3 pytest "$T" -s -k "$P" || ok=0
    run m11fix_par_kv2 600 env TT_SOL_KV_PAGES=2 pytest "$T" -s -k "$P" || ok=0
    if [ "$ok" != 1 ]; then
        echo "    m11fix: a correctness run failed; skipping the timings"
    elif ! python -c "import diffusers.models.transformers.transformer_minimax_h3" > /dev/null 2>&1; then
        echo "    m11fix: diffusers lacks transformer_minimax_h3 (the MiniMax-H3 patch); re-apply it, then run m11fix_t"
    else
        ONLY=m11fix_t
    fi
fi
if want m11fix_t; then
    S="4x8 and nl2 and 768p_5s and not dense and not agather"
    TR="SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off"
    run m11fix_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    run m11fix_trace_m8_tau1.0 900 env $TR SOL_TAU=1.0 TT_SOL_CHAIN_CREDITS=0 pytest "$B" -s -k "$S"
    run m11fix_trace_kv2_tau1.0 900 env $TR SOL_TAU=1.0 TT_SOL_KV_PAGES=2 pytest "$B" -s -k "$S"
    run m11fix_trace_kv3_tau1.0 900 env $TR SOL_TAU=1.0 TT_SOL_KV_PAGES=3 pytest "$B" -s -k "$S"
    for tau in 2.0 3.0; do
        run "m11fix_trace_kv2_tau${tau}" 900 env $TR SOL_TAU=$tau TT_SOL_KV_PAGES=2 pytest "$B" -s -k "$S"
    done
    run m11fix_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 TT_SOL_KV_PAGES=2 pytest "$B" -s -k "$S"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m11fix_logs" 2>/dev/null
fi

# D1 (2026-10-26, sparse_attn_design_2026-10-23.md §4.2): the resident piece stream delivered by multicast groups.
# Each head chain's injector reads every piece from DRAM and multicasts K, V and a running count to the other
# members (row rectangles); members credit the producer per page they can reserve, up to the 3-page K/V CB. No
# forwarding, so no slowest-member lockstep along a chain. Rebuild first (reader kernel + factory changed).
# Correctness gates the timings (a hang shows as exit 124):
#   check / r1184   512 / 1024 / 1184-row resident vs compact
#   parity          5 s per-call parity (the shape that fills the lookahead; M11 passed the small checks and failed here)
#   parity_kv2      the same with a 2-page CB (one piece ahead)
# Then traced (6 replays): dense, the M8 chain baseline (TT_SOL_RES_MCAST=0 TT_SOL_CHAIN_CREDITS=0), D1 at tau 1/2/3,
# and one D1 profile (per-core idle: RD_GROUP_RX / RD_GROUP_TX zones against the chain's RD_CHAIN_* ones).
# Pass: parity >= 0.9996 on every call; tau 1 below the M8 baseline by most of the ~1.0 ms chain idle.
if want d1; then
    P="4x8 and resident_call_parity and r4736"
    ok=1
    run d1_check 900 pytest "$T" -s -k "4x8sp1tp0nl2 and resident_matches_compact and (r512 or r1024)" || ok=0
    run d1_r1184 900 pytest "$T" -s -k "4x8sp1tp0nl2 and resident_matches_compact and r1184k512" || ok=0
    run d1_parity 600 pytest "$T" -s -k "$P" || ok=0
    run d1_parity_kv2 600 env TT_SOL_KV_PAGES=2 pytest "$T" -s -k "$P" || ok=0
    if [ "$ok" != 1 ]; then
        echo "    d1: a correctness run failed; skipping the timings"
    else
        ONLY=d1_t
    fi
fi
if want d1_t; then
    S="4x8 and nl2 and 768p_5s and not dense and not agather"
    TR="SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off"
    run d1_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    run d1_trace_m8_tau1.0 900 env $TR SOL_TAU=1.0 TT_SOL_RES_MCAST=0 TT_SOL_CHAIN_CREDITS=0 pytest "$B" -s -k "$S"
    for tau in 1.0 2.0 3.0; do
        run "d1_trace_tau${tau}" 900 env $TR SOL_TAU=$tau pytest "$B" -s -k "$S"
    done
    run d1_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 pytest "$B" -s -k "$S"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_d1_logs" 2>/dev/null
fi

# M10 timing only (2026-10-23): the first M10 build passed the correctness set but the 5 s runs overflowed the kernel
# config buffer (73424 > 70656); outlining the S1/S5 flushes and the P@V issue left it 336 B over (70992), so the
# resident build now compiles the compute kernel at O2 (factory; ~30 % smaller TRISC binaries, the level every
# profiled run already used). This step skips the 21-minute correctness set: 5 s parity (also the program-size
# check), traced dense + tau 1-3, profile.
if want m10t; then
    run m10t_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    run m10t_trace_dense 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and dense"
    for tau in 1.0 2.0 3.0; do
        run "m10t_trace_tau${tau}" 900 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident \
            SOL_BENCH_PREFIX=off SOL_TAU=$tau pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m10t_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m10_logs" 2>/dev/null
fi

# M9 timing only (the m9 correctness set passed 11/11 on the first build; the 5 s runs hit the kernel config
# buffer limit and were re-run after outlining the step's pack and grouping helpers).
if want m9t; then
    run m9t_parity 900 pytest "$T" -s -k "4x8 and resident_call_parity and r4736"
    for tau in 1.0 2.0 3.0; do
        run "m9t_tau${tau}" 900 env SOL_BENCH_REPS=6 SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=$tau \
            pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    done
    run m9t_profile 900 env TT_METAL_DEVICE_PROFILER=1 TT_SOL_STEP_PROFILE=1 SOL_BENCH_PROFILE=1 SOL_BENCH_REPS=1 \
        SOL_BENCH_MODE=resident SOL_BENCH_PREFIX=off SOL_TAU=1.0 \
        pytest "$B" -s -k "4x8 and nl2 and 768p_5s and not dense and not agather"
    [ -d generated/profiler/.logs ] && cp -r generated/profiler/.logs "profile_m9_logs" 2>/dev/null
fi

# M3 hang localization (2026-10-21: the first M3 build hung at 1024 rows / k 512 / tau 2 with 13 cores stuck;
# keep-all and the single-piece geometries passed). Reset the board first (the hung cores block device open).
# a) the failing geometry under the assert-only watcher: generated/watcher/watcher.log shows each core's last
#    waypoint, i.e. which CB or semaphore wait it sits in; b) the same geometry with one step per chunk
#    (TT_SOL_RES_CHUNK_MAJOR=1): a pass there puts the fault in the block grouping, a hang in the in-place state.
if want m3_hang; then
    run m3_hang_watch 600 env TT_METAL_WATCHER=5 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
        TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 \
        TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
        pytest "$T" -s -k "4x8 and resident_matches_compact and r1024k512 and tau2"
    [ -f generated/watcher/watcher.log ] && cp generated/watcher/watcher.log "log_val_m3_hang_watcher.log" 2>/dev/null
    run m3_hang_chunkmajor 600 env TT_SOL_RES_CHUNK_MAJOR=1 \
        pytest "$T" -s -k "4x8 and resident_matches_compact and r1024k512 and tau2"
fi

echo
echo "summary:"
for f in log_val_*.txt; do
    [ -f "$f" ] || continue
    printf '  %-24s %s\n' "$f" "$(grep -E '^=+ .*(passed|failed|error|skipped).* =+$' "$f" | tail -1)"
    grep -h "\[tail\]\|\[q2\]\|\[5s pack\]\|\[q1\]\|\[768p_\|\[parity\]\|\[ag \|\[banks\]\|\[resident\] rows=\|Main output - PCC" "$f" | sed 's/^.*INFO *| //' | sed 's/^/      /' | tail -40
done
