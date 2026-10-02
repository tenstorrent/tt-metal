# WIP handoff (temporary — delete once the plan below is done or moved)

Picked up 2026-09-29 on bh-lb-120-a07u24: every P150b opens at 12×10 (checked chips 0 / 3 / 7), the Release build is
current, bs1 baseline matches the old box (cold 15.6 / sustained 15.8 ms).

## Plan (agreed order)

From the e2e-vs-roofline analysis on the device-profile artifact (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn):

1. ~~bs1 fused SwiGLU with K_block 40.~~ Done, negative (NEGATIVE_RESULTS §61): at M=512 the SwiGLU is only 3% exposed
   and the fused kernel is data-movement-bound. bs1 stays unfused.
2. ~~Cheaper SwiGLU in FF1+FF3.~~ Landed (§62): Schraudolph exp + bare SFPARECIP, 71% less SFPU work, STS-B within
   noise; sustained −1.8 / −0.3 / −0.5% at bs8 / 16 / 32 against a no-SFPU ceiling of −3.3 / −3.8% (bs16 / 32).
   The tail exposure was minimal_matmul's output writer (§63, landed: cold bs16 / 32 −1.8 / −2.1%, sustained
   −0.7 / −0.6%, bs8 sustained +0.4%). Left: the partial-sum add (66-69 µs, 4% at bs16; only K_block 80 avoids it and
   that fits only small blocks), and the power cap eating most cold gains (sustained gets ~⅓).
3. SDPA DRAM traffic. Landed at bs8 / 16 (§64): only the K / V reads matter; K / V in L1 (`QWEN_HEADS_KV_L1`), SDPA
   −15 / −13%, cold −1.7 / −0.8%, sustained 0.0 / −0.6%, bit-identical. bs1 already has Q/K/V in L1. bs32 landed with
   4 quarter-batch QKV chunks (§66: replay −1.1 ms, cold −0.7 / −1.5 ms). The head-major QKV write (#57722) is untried.
4. SDPA compute (§65). Compute-only / DM-only floors measured (bs32: 519 / 483 µs against 597); the pack thread paced
   the Q·Kᵀ / exp loop. Landed: row sums on the math thread (SDPA −6 / −7 / −6%). Done with SDPA compute: the exp is
   already the SFPLOADMACRO Schraudolph path at its rated speed. SDPA's remaining gap to its floors is each core's
   cold start on its first head's K / V (a next-head prefetch was negative).

5. ~~Trid pipelining in the custom ops (§67).~~ Done (2026-09-30, commits 7d40ada / 9844602 / b9c2258). Heads op:
   compute-bound at bs8 / 16 / 32 (full within 3-5 µs of compute-only; Q's DRAM write fully hidden), cos / sin
   double-buffered (−2.7% per call, bit-identical, default `QWEN_HEADS_ROT_DB=1`). Add+norm: data-movement-bound
   everywhere (90-92% of its DM floor: DRAM traffic at bs32, per-wave read / write latency at bs8 / 16); exchange on its
   own trid landed but neutral (`QWEN_ADD_NORM_PART_TRID=1`); two-wave CBs negative (bs8 +1.8%, bs16 / 32 do not fit).
   E2e after all of it: bs8 76.7 / 99, bs16 143.3 / 194, bs32 291 / 378 ms cold / sustained (chips 1 / 2 / 0).

## Where we stopped (2026-10-02, project paused; HEAD a8ce76c on `amorrison/high-batch-optim`, not pushed)

The last two sessions built a roofline (an implementation-independent, asymptotic upper bound on performance) and split
every op group's gap to it by cause. No model code changed since cdb9143 (bs8 / 16 / 32) and 3164ba1 (bs1), so the
profiles and the e2e numbers below are current.

**Numbers** (e2e from `sustained_run.sh`, cold / sustained ms; device = one profiled replay at 1.35 GHz; roofline =
cold, SFPU hidden – SFPU serialized):

| batch | cold | sustained | × H200 sustained | device replay | roofline | cold ÷ roofline |
|---|---|---|---|---|---|---|
| 1 (3164ba1) | 14.1 | 14.3 | 2.63× | 13.96 | 6.92 – 7.50 | 1.88 – 2.04× |
| 8 (cdb9143) | 76.7 | 99.1 | 3.00× | 76.07 | 55.4 – 60.0 | 1.28 – 1.38× |
| 16 (cdb9143) | 143.3 | 193.1 | 2.87× | 142.1 | 110.8 – 120.0 | 1.19 – 1.29× |
| 32 (cdb9143) | 291.7 | 377.6 | 2.71× | 279.1 | 221.6 – 240.0 | 1.22 – 1.32× |

Sustained is cold plus the power cap (AICLK settles at ~1.0 GHz at bs16 / 32); the roofline is a cold, 1.35 GHz bound.

**Build.** The installed `_ttnncpp.so` / `_ttnn.so` include tt-metal 61fb987 (host C++ in the matmul 1D factory and the
device-op factory selector), built with `ninja -C build ttnn/_ttnncpp.so ttnn/_ttnn.so` and copied over `build/lib/`
and `ttnn/ttnn/_ttnn.so`. A fresh checkout or another box needs that rebuild, or bs1's FF1 / FF3 fail. After any
host-side C++ change: rebuild (PERF_GUIDE §4); a stale `.so` deadlocks in the first warmup prefill instead of erroring.

### The roofline (`perf_tools/profile_page.py`, `achievable` fields; the page calls it "roofline")

Cold, 1.35 GHz, 120 cores, every activation on chip, weights read once. Range = [max(FPU, SFPU, DRAM), FPU + SFPU].
- **FPU**, in sequence: matmul FLOPs at 89% of the LoFi peak (590.6 TFLOP/s: the LLK's 18.0 cycles per tile product at
  long K, matches GEMM_FLOPS' best Blackhole GEMM and this model's compute-only matmuls); SDPA's matmuls at tt-llk's rate
  for their shape (Q.K^T, K = 4 tiles: 23.9; P.[V | 1], K = 16, a ones column giving the softmax row sums: 19.2);
  eltwise passes at tt-llk perf-suite rates (add / mul 31.1, column-broadcast 28.6, row reduce 51.1 cycles per bfp8
  tile, L1 to L1) and the softmax row max at SDPA's demonstrated ~7 cycles a tile.
- **Formulation** (the user's choice, 2026-10-02): RMSNorm gamma folded into the next matmul's weights, QK-norm gamma
  into the RoPE cos / sin tables, rotate-half as a whole-tile swap, softmax max kept, norms' row sums as reduces.
- **SFPU**: exp per score (64 cycles a tile, SDPA's SFPLOADMACRO path), silu(gate)·up (383 per output tile, the fused
  SwiGLU pass), rsqrt per norm row on column 0 only (607 / 2).
- **DRAM**: bfp4 weights + embedding rows at 450 GB/s (~4.6 ms; never the bound).

Rates come from the tt-llk perf suite run on chip 1 (see "tt-llk perf harness" below; the CI warehouse copy needs a
service keypair we do not have). Earlier versions of the bound that the user rejected: hiding all vector work under the
FPU (gave the custom ops 0 ms), ones-vector row sums at 18 cycles (an N = 1 matmul is 55.7), SDPA at 89% of peak.

### Gap to the roofline by category (`profile_page.py` `GAP_LADDER` + `formulation()`)

Measured device-kernel-time ablation ladders of each op group's main call, standalone at the model's placement (every
full call within 1% of in-model except bs16 QKV, 529 vs 522 µs), plus the custom ops' and SDPA's passes beyond the
roofline formulation costed at the same tt-llk rates. ms over the replay:

| category | bs16 | bs32 | largest cells |
|---|---|---|---|
| inits, handshakes, blocking (compute only above roofline, less formulation) | 12.6 | 19.3 | heads 3.5 / 8.2, add+RMSNorm 3.6 / 5.8, QKV 2.1 / 3.0, FF1+FF3 1.9 / 2.1 |
| data movement not hidden (full - compute only) | 8.7 | 20.2 | add+RMSNorm 2.1 / 8.4, FF1+FF3 3.8 / 2.9, FF2 0.4 / 3.5, QKV 1.3 / 2.5 |
| formulation | 5.6 | 11.2 | FF1+FF3 partial-sum add 2.4 / 5.0, SDPA col_identity sums 1.2 / 2.4, heads 1.1 / 2.3, add+RMSNorm 0.9 / 1.6 |
| SFPU not hidden | 2.7 | 5.8 | SwiGLU 1.2 / 2.8, exp 0.8 / 1.5, heads rsqrt 0.7 / 1.4 |
| cross-core exchange (add+RMSNorm) | 0.6 | 0.6 | |
| in-model vs standalone, small unfused ops | 1.1 | 0.4 | bs16 post-MLP add+RMSNorm (a in L1) +22 µs a call in-model |
| host, dispatch, gaps between ops (cold - device) | 1.2 | **12.6** | |

Findings behind it: every batched matmul is compute-kernel-bound (compute only 78-90% of peak); the plain matmuls'
end-of-block intermediate -> out copy costs ~0 (QKV's excess is loop structure, not the copy); a matmul's math-free
skeleton alone takes 30-40% of compute only and is almost all hidden; SDPA sits at roofline + formulation (structure ~0;
its analytic formulation slightly overstates, structure -0.17 ms at bs32); bs32 FF2's compute only beats 89% (90%).

### Next, in the order I would take them

1. **bs32 host / dispatch: 12.6 ms** between cold and device replay (bs16: 1.2). Unexplained and the largest single bs32
   item: look at the device timeline's gaps between ops (tracy report) and at what bs32 does differently (4 QKV / heads
   chunks per layer, 503 ops per replay against 293).
2. **Inits / handshakes / blocking in the custom ops** (heads, add+RMSNorm): fusing their phases into the neighbouring
   matmuls (norm as a QKV / FF1 prologue, QK-norm + RoPE as a QKV epilogue) removes most of it and their data movement.
   Before that, per-phase device-profiler zones would split phase overhead from primitive slowness.
3. **Formulation fixes** with direct payback: gamma folded into weights / RoPE tables, rotate-half as a tile swap instead
   of the rotation matmul, V written by QKV instead of copied through, SDPA row sums as a ones column in P.V.
4. **Data movement not hidden**: FF1+FF3's output round trip through DRAM (an FF13 -> FF2 fusion in token chunks would
   remove it), add+RMSNorm's a / sum in DRAM at bs32 (8.4 ms).
5. **The power cap** (sustained = cold + 26-35%). Faster kernels have lowered the settled clock before (§59): measure
   J / pass, not only ms; tt-smi sampling is too sparse for J / inference today.
6. Housekeeping: upstream 61fb987 as its own tt-metal PR (432 matmul unit tests passed); `lut2` sigmoid for bs1.

### How to resume

- **Profile page** (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn, version 32, shared with the org). The template
  (sections, JS) lives only in the published page: read the artifact (Artifact tool, action read) to get the HTML, then
  `python3 perf_tools/profile_page.py <saved page.html> runs.json <out.html>` and republish to the same URL. Visible text
  says "roofline"; the JSON / JS fields and the generator still say `achievable`. runs.json for the current profiles:

```json
{"commit": "3164ba1 (bs1) · cdb9143 (bs8 / bs16 / bs32)", "date": "2026-10-01", "batches": [
 {"bs": 1, "report": "2026_10_01_19_06_29", "e2e": {"cold": 14.1, "sus": 14.3, "aiclk": "1350", "power": "n/a"}},
 {"bs": 8, "report": "2026_10_01_15_38_45", "e2e": {"cold": 76.7, "sus": 99.1, "aiclk": "1056 (1025-1168)", "power": 86}},
 {"bs": 16, "report": "2026_10_01_15_43_17", "e2e": {"cold": 143.3, "sus": 193.1, "aiclk": "1000 (993-1012)", "power": 104}},
 {"bs": 32, "report": "2026_10_01_15_48_29", "e2e": {"cold": 291.7, "sus": 377.6, "aiclk": "993 (981-1037)", "power": 139}}]}
```

- **After a kernel change**, re-run the ladders and update `GAP_LADDER` (device µs per call), or the breakdown goes
  stale. Each in its own process with `TT_VISIBLE_DEVICES=<chip>`:
  - matmuls: `MM_BLOCKS=4,40,8,1,8 bench_mm_gap_ladder.py ff13fused 16|32`; `bench_mm_gap_ladder.py qkv 16`,
    `MM_BLOCKS=8,8,8,1,8 bench_mm_gap_ladder.py qkv 8` (bs32 chunk), `ff2 16|32`, `wo 16|32`. It patches
    `compute_metal2.cpp` / `matmul_dataflow_common_metal2.hpp` in the repo and restores them in a finally block: run one
    at a time, and `git status` after any interrupted run (an interrupt can leave the patch in place).
  - SDPA: build trees with `sdpa_kernel_variants.py` (base, conly, and conly + noexp composed via its `VARIANTS` /
    `patch_dm`), run `bench_sdpa_floors.py <label> <bs>` from a directory with no `ttnn/` tree under the profiler, parse
    with `bench_sdpa_floors.py --parse <dir>` (K/V L1 arm; bs32's all-L1 arm fails to allocate, expected).
  - heads / add+RMSNorm: `bench_heads_placement_ablate.py 16 16` / `8 32`, `bench_add_norm_ablate.py 16 32` under
    `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 TT_METAL_PROFILER_DIR=<dir>`, read with
    `device_kernel_us.py <dir> <labels...>`.
- **tt-llk perf harness** (primitive cycles per tile): venv `tt_metal/tt-llk/tests/.venv` (`uv pip install -r
  requirements.txt`) and `tt_metal/tt-llk/tests/sfpi` symlinked to `runtime/sfpi` (same SFPI 7.80.0; untracked, keep it
  out of commits). Run tests by node id (`-k` cannot parse `->`) with `TT_VISIBLE_DEVICES=<chip> TT_LLK_DISABLE_ASSERTS=1`;
  results in `tt_metal/tt-llk/perf_data/runs/local-*/*.parquet`, per tile = TILE_LOOP mean / (loop_factor × tile_cnt).
  `perf_matmul` skips Half dest sync with a bfp output (#56073): use the SyncFull variants; its K sweep is {1, 4, 32}
  (K = 16 came from a scratch copy with `KT_DIMS = [16]`).
- **Tools** (`perf_tools/`): `profile_page.py` (page data, roofline, gap split), `bench_mm_gap_ladder.py`,
  `device_kernel_us.py`, `bench_mm_ablate.py` / `bench_ff13_fused_ablate.py` (wall-clock ablations),
  `capture_qkv_call.py` (`CAP_N=6144,19456,9728,2560` prints the model's matmul calls), `sdpa_kernel_variants.py`,
  `bench_sdpa_floors.py`, `bench_heads_placement_ablate.py`, `bench_add_norm_ablate.py`, `sustained_run.sh`.
- **Earlier leads, still open**: add+RMSNorm per-wave read / write latency (software-pipelined compute with CB 8 as a
  2-3 wave ring; larger read batches per barrier); the heads op's L1 read path once its compute is faster (~2 GB/s per
  core, one barrier per 26 KB unit).
