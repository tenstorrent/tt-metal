# Hand-off: sparse MLA full-layer KV push + two-level top-k (GLM-5.3, Kimi K3.1)

Status as of 2026-11-22 (see section 0 for the overnight update). This is an **experiment branch**, not for merge. It holds measurement-only switches and CI
tweaks. Read the whole document before running anything on a Galaxy.

## 0. Update 2026-11-22 (overnight, bh-glx-120-b07u08, 8x4 Galaxy)

Sections 1-8 below are the 2026-11-21 state. This section supersedes them where they disagree.

### 0.1 Commits on top of `4b689a36015`

| Commit | Change | Measured (traced, 8x4) |
|---|---|---|
| `c4de2ee0abf` | Plan exits early in dense mode; plan `dense_below_2` (third output + header word `kWordDense2`) and push `second_dense`, so a GLM full layer uses ONE plan (its shared successors') instead of two | Dense plan 166 -> 41 us. Bit-identical to the two-plan path (GLM layer, metadata path, 10k/102k/410k) |
| `230da3eb00d` | Push v2: plan `owner_major` table (one slot per owner: header + that owner's block bitmaps); push `inbox`: each chip sends every owner its slot (fused write+inc), arrivals = start fence, no plan-table all-gather; `reuse_inbox` for REST after PREFIX. Model: `SPARSE_KV_PUSH_INBOX` (default 1) | gather+push -> push: L14 1M 507 -> 336 us, L62 1M 1559 -> 1454 us, 46k -42 us; REST 108-120 -> 94-98 us. Bit-exact vs all-gathered tables (real GLM top-k L14/L62; GLM and K3.1 layers on the metadata path) |
| `67b99fb5e16` | Plan batches the L1 loads of its scatter/rewrite loops | Plan 222 -> 210 us (1M), 162 -> 149 us (46k) |
| `1f9284d5cb2` | Full-layer cutoff per model: GLM-5.3 204,800, K3.1 300,000 (`SPARSE_MLA_FULL_PUSH_DENSE_BELOW` still overrides) | see 0.2 |

Tests on this host: plan unit tests + push-vs-gather (real top-k) + dense-mode-vs-gather 22/22; sparse-MLA gate 6/6
(incl. traced-vs-untraced); determinism 6/6. Gate, same host, v2 vs v1 push (traced ms): K3.1 warm 3.330 vs 3.364,
GLM warm 3.007 vs 3.039, K3.1 long 6.350 vs 6.407, GLM long 5.051 vs 5.126.

### 0.2 Real-weights runner A/B (local, depth 8, full 1M request, host chunk times)

Depth 8 = K3.1 layers 0-7 (one working DSA layer, 3; the runner's LAST layer is KV-only, so the CI L24 leg has 5
working DSA layers, not 6) and GLM layers 0-7 (full 0, 1, 2, 6; shared 3-5). Run-to-run noise per 50k band ~+-0.3 ms.

- **GLM-5.3** (whole request): baseline (push off) 9,264-9,312 ms -> 8,212 ms (-11.6%). Push-always crosses the
  baseline at ~205k (-0.3 ms/chunk at 205k, -1.6 at 256k, -13.4 at 1M); hence the 204,800 cutoff.
- **K3.1** (whole request): 12,836-12,854 -> 12,686 ms (-1.2%, one DSA layer). Push-always crosses at ~300k
  (-0.13 ms at 307k, -0.73 at 512k, -2.2 at 1M per DSA layer). The 300k cutoff stays.
- K3.1 DSA layer at 1M in push mode (chip 0 RT timeline): indexer 6.08 ms, stems 0.43, top-k 0.77 (beside the idle
  gather), plan 0.28, push 2.23, sparse SDPA 0.82. GLM full layer at 1M: indexer 3.87, stems 0.42, top-k 0.59,
  plan 0.21, push 0.65, SDPA 0.88; the shared layer after it: REST push 49 us + SDPA.

### 0.2b Full-depth A/B (local, real weights, full 1M request; host times)

| Run | GLM-5.3 78 layers, SC1 8x4 | K3.1 L24 (5 working DSA layers) |
|---|---|---|
| Baseline (full-layer push off; shared layers already on push v2) | TTFT 92,216 ms, 9,384 tok/s @1020k | 46,043 ms |
| Full push, one cutoff (GLM 204,800 / K3.1 300,000) | **87,295 ms (-5.3%), 10,730 tok/s (+14.3%)** | 46,035 ms (loses ~3 ms/chunk at 300-600k, -4.2 at 1M) |
| Full push, per-layer cutoffs | 87,253 ms (no gain over one cutoff; not adopted) | **45,679 ms (-0.8%; never loses, -1.3..-5.1 ms/chunk from 300k)** |
| Full push everywhere (cutoff 1 slab) | - | 46,416 ms (+0.8%) |

GLM per-chunk at 900k-1M: 528.5 -> 467.5 ms (-11.5%); 300-400k -1.9%; below 300k within noise. KV PCC is identical
between baseline and push runs (same per-layer nope/pe values), PCC gates pass everywhere. The GLM perf gate on this
host reports the baseline slower than CI's configured one (9,384 vs 9,906 tok/s) and the push run faster (10,730).
K3.1 per-layer table (committed in run_multirank_pcc.sh's kimi_k3_1 case): `3:307200,7:870400,11:4194304,15:409600,
19:972800` (layer 11 never pushes). GLM per-layer crossovers (from RT, linear gather model) range 205k-717k
(layers 38-62 ~300-512k, 42 at 717k), but applying them changed nothing measurable.

### 0.3 Answers to the open questions

- **Why the push is slow at long context / low eth utilization.** It is fabric-bound, not kernel-bound. With
  dimension-ordered routing the 8-long Y ring carries ~2x the hop-bytes of the 4-long X ring (avg 2 vs 1 hops); at
  L62 1M the busiest Y links forward ~13 GB/s (about half of 25). Per sender, a loaded channel moves ~660 packets while
  others idle (direction imbalance follows from the routing). Packet payload 6-14 KB and deeper sender pipelining
  (mux buffers 2->4, kBatch 4->8) change nothing (1.46-1.50 ms). Real GLM top-k: a needed row is needed by ~6 chips
  (L62: 46.5k distinct rows/chip, 239k union). Simulated on that top-k, a per-row multicast tree (Y trunk, X
  branches) cuts the busiest link 62.6k -> 32.8k rows (~1.9x); the fabric's rectangle multicast does NOT help (63%
  of deliveries wasted, X links flood). A stepping stone: a fabric line multicast of each row along its owner's column
  (the Y trunk, into a relay slot at the same address on every chip of the column), then X unicast from the relay in
  each target row: busiest link 39.9k rows at L62 (1.57x; X becomes the limit), 6.6k at L14 (1.96x); an X line
  multicast from the relay reaches the full tree. A real tree needs MoE-style store-and-forward (`sparse_dispatch_2d`, on main
  since 2026-11-19, not on this branch), which also runs every link at 1-hop efficiency. Biggest remaining lever for
  GLM (shared layers' PREFIX push 1.45 ms vs 0.82 ms attention window at L62 -> ~0.6 ms exposed x 57 layers).
- **Why the planner is slow.** Not the NoC exchanges (each 6-18 us) but per-index L1 work on two RISCs: bucket+scatter
  ~66 us and rank rewrite ~70 us of ~210 us at 1M (DPRINT step timestamps). Load batching gives 6%. The earlier
  measured TRISC variant (208 -> 117 us) is the remaining big lever if done on supported APIs.
- **Cutoffs.** GLM ~200k, K3.1 ~300k (0.2). They move down as fixed costs drop.
- **Top-k faster / smaller arrays.** Below the cutoff top-k hides behind the gather, so small-T support (8-key blocks
  pay below ~128k) would not show. Above it top-k is serial (0.6-0.8 ms at 1M). Proposal, exact: process stage-3
  candidate blocks in descending block-max order (stage 2 already sorts them) and stop merging once the running
  2048th key >= the next chunk's best block max. A block whose max is >= the true 2048th key necessarily holds a top
  key, so this processes exactly the blocks that hold top keys. Real GLM 1M top-k (staged): those are 1,259 / 592 /
  340 / 956 of the 2,048 selected blocks at L0 / L14 / L30 / L62, i.e. stage 3 shrinks 1.6-6x (about -30% of top-k at
  1M). Needs the compute kernel to decide (running min vs stage-2 sorted maxima) and the writer to emit stage 2's
  value order instead of the ascending block table; the reader can keep feeding and compute pop unneeded chunks.
- **Fuse top-k into the indexer.** No: in the ring layout no core holds a full logit row, and the partial fusion that
  pays (indexer block maxima) is in.
- **K3.1: plan + push as one op / better overlap.** The table exchange is now inside the push (v2). Fusing the plan
  too would only save a launch (plan needs all 120 cores, push 24). Hiding plan + PREFIX behind the Q/KV stems is
  worth at most the stems (~0.43 ms) and would take their grid. Not worth it.

### 0.3b Per-layer behavior (important for cutoffs)

- **The push cost differs per layer, by up to 2.5x.** K3.1 L24 (DSA layers 3, 7, 11, 15, 19), push at 300k / 900k:
  1.13/2.13, 2.27/3.82, 3.14/4.76, 1.50/1.92, 2.88/3.94 ms. The dense gather costs every layer the same (~4.2 ms
  exposed at 1M). So layers 3 and 15 win from ~300-400k, while 7, 11 and 19 barely break even at 1M. With one cutoff
  (300k) K3.1 L24 LOSES ~3 ms/chunk at 300-600k and only wins from ~750k (-4.2 ms at 1M); whole request a wash
  (46,043 vs 46,035 ms), both PCC gates pass. The depth-8 tuning saw only layer 3 and was misleading.
  `SPARSE_MLA_FULL_PUSH_DENSE_BELOW_LAYERS="layer:tokens,..."` sets per-layer cutoffs (a value past the buffer = never).
- **GLM union sizes (real 1M top-k, staged layers):** union of all chips' rows = 3.4-29% of the context, per chip
  0.8-4.5%, reuse 3-8x, owners balanced (max/mean <= 1.2). A *sparse union all-gather* (each owner broadcasts only the
  rows anyone needs, over the efficient 1-hop ring) costs ~union% of a dense gather: for GLM about the same as the push
  (L62 ~1.2 vs 1.45 ms). For K3.1's deep layers (push ~13% of context per chip by its cost) it plausibly beats both the
  push and the dense gather. With push v2 every owner already holds all requesters' bitmaps of its blocks (inbox), so
  the union is local; missing: per-owner variable-length gather and the requester-side union-rank rewrite.

- **K3.1's top-k is far more spread than GLM's, but on a synthetic prompt.** Dumped from an eager L24 run (wrapper
  `dump_topk.py` in the overnight scratch): union / per-chip share of the context at 312k and 1M: layer 3 51/9.5% and
  55/6.5%; layer 7 73/19% and 62/12%; layer 11 94/28% and 83/15%; layer 15 52/13% and 33/5.7%; layer 19 92/25% and
  84/12% (GLM: union 3-29%, per chip 0.8-4.5%). The simulated busiest push link matches the measured push times
  (layer 11: 199k rows ~ 4.8 ms). A union all-gather would not help these deep layers. BUT the K3.1 leg's 1M request
  is the 8k golden prompt repeated (`_load_token_pool` tiles the trace), so the indexer picks the same tokens in many
  copies; real long documents likely favor the push much more. Get a real 1M K3.1 trace before tuning K3.1 further.

### 0.4 Harness (`tools/`, also see tools/README.md)

- `run_local.patch`: local copy of `run_multirank_pcc.sh` with `LOCAL_NUM_LAYERS/LOCAL_REQUEST_LEN/LOCAL_CHECK_PCC/
  LOCAL_WARMUP_CHUNKS` and the venv python for producer + runner (login shells reset PATH). Needs `TTRUN_DIR` with a
  `hostfile` of `localhost`. `LOCAL_CHECK_PCC=1` is required: it drives the producer's completion drain that logs
  `CHUNK_RETIRE` (per-chunk host spans).
- `drive2.sh <model> <tag> <layers> "<env>"`: one run (about 3 min K3.1, 3-13 min GLM incl. first compile).
  Extra per-rank env goes through `SCRATCH_RUNNER_ENV` (uncommitted hook in run_multirank_pcc.sh; tt-run does not
  forward env). `analyze.py A B` compares host chunk times per band; `rtsum.py`/`timeline.py` read the RT dump
  (uncommitted `SCRATCH_RT_DUMP/SCRATCH_RT_KEEP` hook in prefill_runner.py). RT mis-times sub-device-overlapped
  programs (top-k beside the gather shows negative durations).
- `tests/nightly/blackhole/sdpa/test_zz_push_bench_scratch.py` (`BENCH_INBOX=1`, `BENCH_LAYERS`, `BENCH_STARTS`,
  `BENCH_PAYLOAD`): per-step traced push timings on the staged real GLM 1M top-k, plus a bit-exact v1/v2 check, ~1 min.
  Compare FP8 buffers as bytes (random FP8 holds NaNs).

### 0.5 Next steps

1. Multicast-tree push (store-and-forward, `sparse_dispatch_2d`-style) for long contexts: ~1.9x less busiest-link load.
2. Top-k stage-3 early exit (0.3); then re-measure the cutoffs.
3. Plan on 5 RISCs (supported APIs) or a different rank scheme.
4. CI A/B of the four commits with `topk_push_long` (both runner legs) before productionizing.

## 1. TL;DR

- **Goal.** Make the sparse-MLA layers of GLM-5.3 and Kimi K3.1 cheaper at long context (target: 1M tokens) on a
  Blackhole Galaxy, by changing what a *full* (indexer-owning) layer does after its indexer.
- **Today (main).** A full layer runs its indexer, then the top-k (`topk_large_indices`, single-level) on a
  restricted core grid **in parallel with** an all-gather of the whole KV prefix ("overlap"). Then sparse SDPA runs.
  GLM's shared layers (no indexer) reuse the full layer's top-k. Instead of gathering, they *push* only the rows they
  reference (`sparse_push_plan` + `sparse_kv_push`).
- **Experiment, on 1M real-weights CI with the cutoff (§6.3).**
  - Full layers get a cutoff. Below it, nothing changes: top-k runs beside the prefix gather. From it upward, the
    layer pushes only the rows its top-k selected, as shared layers do.
  - The indexer also writes 32-key block maxima. Top-k takes them as `block_max_tensor`: two-level selection with no
    maxima pass of its own.
  - **GLM-5.3:** +10.5% throughput at 1020k, −3.0% TTFT for the 1M request. KV accuracy is unchanged. Contexts up to
    ~255k are about 1.3% slower.
  - **K3.1:** roughly break-even (−0.5…−1.1% up to 510k, +0.3% at 1M). KV accuracy is unchanged.
- **Main open questions.**
  1. Why K3.1 gains so little: about 0.12 ms per DSA layer, against about 2.35 ms per GLM full layer. The leading
     hypothesis is the push plan: GLM already pays for it in the baseline, K3.1 doesn't. **Needs a per-op Galaxy
     profile.**
  2. Remove the remaining ~1% short-context overhead (§8).
  3. One unexplained determinism failure (§7).

## 2. Branches

| Branch | Content |
|---|---|
| `pjosipovic/two-level-topk-impl` (head `1b8632fbd5b`) | The two-level `topk_large_indices` work. Reviewed and pushed; the base of both experiment branches. |
| `pjosipovic/topk-push-experiment` | Experiment: switches **on** by default, plus the cutoff, the indexer gate, CI tweaks and this document. |
| `pjosipovic/topk-push-baseline` | The same code with both switches **off** by default, plus the same CI tweaks and the same merged PRs. It's the A/B baseline. It lacks the two newest experiment commits (cutoff and indexer gate); both are inert with the switches off. |

Merged into both experiment branches (both open PRs from Iva Potkonjak, both merged cleanly):

- **#64501** (`pr-64501` = `ac3800278e2`): `sparse_kv_push` runs on every 2D fabric, not only `FABRIC_2D_TORUS_XY`.
  This is what makes the push run on LoudBox (2x4, `FABRIC_2D`) and on the 4x4 stage.
- **#63972** (`pr-63972` = `0dc08a77610`): dense mode. Chunks starting below a threshold gather the prefix densely
  instead of pushing, with the mode decided on device so one trace covers both. Shared layers use 6 slabs (30,720
  tokens on GLM).

Experiment commits, first-parent from `1b8632fbd5b`:

```
dadb9e9be15 WIP: indexer block maxima output; temporary model switches
946192ffef3 Merge pr-64501
5189f972209 EXPERIMENT: full layers push their KV rows; indexer block maxima feed top-k (defaults on)
3a158397eb8 EXPERIMENT CI: topk_push_long selects GLM-5.3 1M SC1 + K3.1 SC1 runner legs; K3.1 requests 1M
104ca5c0a52 EXPERIMENT CI: topk_push_det leg (determinism repro on 8x4)
b39003e609e EXPERIMENT CI: K3.1 runner L1_SMALL 3072
1f8c03eb847 Merge pr-63972
46d8c0b424a EXPERIMENT: full layers push only from a cutoff (~300k); below it top-k runs beside a threshold-gated gather
f8ca421af83 EXPERIMENT: indexer writes block maxima only from top-k's two-level cutover
```

## 3. Background: two-level top-k (`topk_large_indices`)

- **API.**
  - `ttnn.experimental.topk_large_indices(..., block_select=32, two_level_cutover=None, block_max_tensor=None)`.
  - Two-level applies only to k = 2048 and a last dimension in [65536, 2097152] that is a multiple of 32. Other
    inputs fall back to single-level. `two_level_cutover` or `block_max_tensor` on such an input raises.
  - The program decides at **run time** from the search length whether to run two-level. The default cutover is
    98,304.
  - At most 4 rows per core, unless `block_max_tensor` is given.
  - Without `block_max_tensor`, the program allocates its own `[rows, ceil(n/32)]` bf16 DRAM maxima tensor and keeps
    it while cached.
- **Standalone timing, 160 rows, Blackhole, 108 / 99 cores.**

  | Search length | Single-level | Two-level, computes maxima | Two-level, maxima supplied |
  |---|---|---|---|
  | 1M | ~4.57 ms (in model, LoudBox) | 1285 / 1348 µs | 814 / 846 µs |
  | 512k | | 856 / 887 µs | 634 / 690 µs |

- **In the K3.1 layer at 1M on LoudBox** (device profile): top-k is 4567 µs single-level, 1354 µs two-level, and
  610 µs with indexer maxima. The indexer goes from 5808 to 5944 µs (+136 µs) when it writes the maxima.
- **Tie choice.** Two-level returns the same value multiset as single-level, but may pick different keys among ties.
  On random-weight data with bf16 logits that moves layer output PCC to about 0.996 against single-level. That's
  expected, not a bug.

## 4. What the experiment changes (code map)

### 4.1 Indexer block-maxima output (`ttnn/cpp/ttnn/operations/experimental/indexer_score/`)

- **API.** `ring_indexer_score_dsa(..., block_max_output=None, block_max_from_length=0)`.
  - `block_max_output` is a `[B, 1, Sq, T/32]` bf16 row-major interleaved DRAM tensor that receives each score row's
    32-key block maxima.
  - It's fused-ring only, needs `num_groups == 1` and `block_size == 0`, and `kv_len % 32 == 0` on the scalar path.
  - Entries past `ceil(kv_len/32)` are not written.
- **Kernels.** All new code is behind the `INDEXER_BLOCK_MAX` define, which only the ring factory sets, so other
  programs are unchanged.
  - **Reader:** builds the 1.0 reduce scaler.
  - **Compute** (`unit_block_maxima`): before untilizing each masked unit, row-max reduces each k-tile
    (`reduce_block_max_row<1>`) into a new CB `block_max_cb`. The maxima land in column 0.
  - **Writer** (`write_block_maxima`): reads column 0 of each pooled tile. It writes runs through the same
    physical→logical tile fragments as the logits, staging each row at its destination's offset mod 16 (the BH NoC
    write alignment).
- **Gate.** `block_max_from_length` is passed as the compile-time arg `block_max_from_tiles`. Below it, compute and
  writer skip all maxima work, decided from the runtime `kv_len_tiles`, so it's trace-safe. The model passes 98,304,
  top-k's default cutover.
- **Verified.** Maxima equal the block max of the indexer's own logits exactly, on all 8 LoudBox chips, on both the
  scalar and metadata paths. Top-k with them passes the exact-top-k oracle.

### 4.2 Model wiring (`models/demos/deepseek_v3_d_p/tt/mla/`)

- **`indexer.py`.**
  - Env `SPARSE_MLA_INDEXER_BLOCK_MAX` (module default `_BLOCK_MAX_DEFAULT`: `"1"` on experiment, `"0"` on baseline).
    It's eligible when `index_topk_capacity == 2048`, `65536 <= width <= 2097152`, `width % 32 == 0`, and
    (traced path or `valid_pos % 32 == 0`).
  - It allocates the maxima tensor per call and stores it in `IndexerSelectionState.block_max`. `select_local` then
    passes `block_select=32, block_max_tensor=...`.
- **`mla.py`.**
  - Env `SPARSE_MLA_FULL_PUSH` (module default `_FULL_PUSH_DEFAULT`: `"1"` / `"0"`). A full layer with a push planner
    (`self._push`) takes the push path.
  - **Traced path** (metadata, overlap resources present, cutoff > 0): `_sparse_full_push_overlapped`.
    1. In the overlap sub-device manager, top-k runs on the top-k grid beside
       `high_bw_all_gather(gathered_active_below=cutoff)` into the layer's compact buffer.
    2. `finalize_distribution`.
    3. `sparse_push_plan(dense_below=cutoff)`.
    4. `sparse_kv_push(part=ALL)`, a no-op in dense mode.
    5. When the next layer is a shared layer, a second plan with the shared layers' threshold, used only for the next
       layer's prefix push (`_next_layer_prefetch`).
    6. Sparse SDPA over the compact buffer with `plan.compact_indices`.
  - **Eager path** (no metadata): serial full push, always pushing, top-k on the full grid. Dense mode needs device
    metadata, as in #63972.
  - **Cutoff:** `SPARSE_MLA_FULL_PUSH_DENSE_BELOW`, default 300,000. It's rounded up to whole slabs and capped so
    that cutoff + 1 slab fits the layer's buffer (`self._full_push_dense_below`).

### 4.3 CI tweaks (experiment and baseline branches only)

- **`test_sparse_mla_perf.py`:** `PERF_TARGETS_MS = {}` makes the sparse-MLA gate report-only, because it's two-sided
  and the jobs stop at the first failure. The traced-vs-untraced correctness check is still active.
- **`.github/workflows/blaze-models-prefill-tests.yaml`:** checkbox `llama31` → `topk_push_long`, and `mistral4` →
  `topk_push_det` (experiment only).
- **`tests/pipeline_reorg/blaze_models_prefill_tests.yaml`.**
  - Tag `topk_push_long` on "(GLM-5.3-744B) disaggregated prefill KV accuracy 1M tail@5k + perf 1M@5k, SC1" and on
    "(Kimi-K3.1) disaggregated prefill L24 ... SC1". The K3.1 leg's timeout is 90 min.
  - New leg "EXPERIMENT sparse MLA full-layer push determinism repro, 8x4" (`topk_push_det`, experiment only).
- **`models/demos/common/prefill/runners/ci/run_multirank_pcc.sh`, K3.1 case:** `REQUEST_LEN=1049600` (was
  2 × 5120). It also exports `PREFILL_L1_SMALL_SIZE=3072`; at 2048 the full-layer push OOMed in the compact-buffer
  sparse SDPA.

## 5. How to run on Galaxy

### 5.1 Through CI (what I used)

```
gh workflow run blaze-models-prefill-tests.yaml --ref pjosipovic/topk-push-experiment -f topk_push_long=true
gh workflow run blaze-models-prefill-tests.yaml --ref pjosipovic/topk-push-baseline  -f topk_push_long=true
gh workflow run blaze-models-prefill-tests.yaml --ref pjosipovic/topk-push-experiment -f topk_push_det=true
# the two perf legs (sparse MLA gate + real-weights GLM 55k chunked) used earlier:
gh workflow run blaze-models-prefill-tests.yaml --ref <branch> -f k3_mla_perf=true -f glm_chunked_perf_suite=true
```

- **Selection.** Ticked checkbox ids are matched against each leg's `test_type`, `test_group` and `test_tags`, and the
  workflow file is read from `--ref`. Check a selection with
  `python3 .github/scripts/utils/validate_test_type_selection.py tests/pipeline_reorg/blaze_models_prefill_tests.yaml <token>`.
- **Logs.** `gh run view --log` refuses while any job is still running. Use the API:
  ```
  gh api --allow-escape-sequences repos/tenstorrent/tt-metal/actions/jobs/<job_id>/logs | sed 's/\x1b\[[0-9;]*m//g'
  ```
- **What to grep.**
  - Runner legs: `throughput +@`, `TPS GATE`, `TTFT GATE`, `CHUNK TIME`, `KV PCC: nope=`.
  - Sparse-MLA gate: `sparse MLA Galaxy .* (un)?traced:`.
  - Chunked GLM test: `Chunk timing` tables.
- **Runner legs fail when faster than the baseline** (two-sided TPS/TTFT gate). A "FAIL (faster)" is a pass for us.
- **Runtimes.** A 1M runner leg takes about 30–40 min end to end; the K3.1 L24 bring-up is about 6 min after the build.

### 5.2 Directly on a Galaxy

- **Sparse-MLA gate, 8x4:**
  `python3 -m pytest models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_sparse_mla_perf -k kimi_k3_1`
  (or `-k glm_5_3`). For the 4x4 stage:
  `models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- <same pytest args>`.
  Use `TT_METAL_SHM_TRACKING_DISABLED=1 LOGURU_LEVEL=ERROR`.
- **Determinism:** `DS_PREFILL_REUSE_MODEL=1 python3 -m pytest models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_determinism.py -k "kimi_k3_1"`.
- **Real-weights runner:** `bash models/demos/common/prefill/runners/ci/run_multirank_pcc.sh glm53 sc1` (or
  `kimi_k3_1 sc1`). See the CI legs in the YAML for the environment.
- **A/B in one checkout:** use the environment switches, `SPARSE_MLA_FULL_PUSH=0|1`,
  `SPARSE_MLA_INDEXER_BLOCK_MAX=0|1` and `SPARSE_MLA_FULL_PUSH_DENSE_BELOW=<tokens>`. Note that `tt-run` does
  **not** forward `PREFILL_*` or other variables to ranks; the runner script passes what it needs through
  `RUNNER_ENV`.

### 5.3 Scratch tests in this folder (all run by path)

- **`exp_long_full.py`:** times one sparse-MLA layer (both models) untraced and traced, at any context.
  - `LONGFULL_CACHE`: buffer/cache tokens. The push needs **(cache + chunk) % 512 == 0**, e.g. on LoudBox 510720,
    1022720, 14080 or 49920; on Galaxy (5120 chunk) 51200 and 512000 work.
  - `LONGFULL_START`: chunk start (≤ cache).
  - `LONGFULL_FILL=1`: fills the prefix with real chunk forwards of random inputs (realistic-ish indices; zero caches
    massively flatter the push).
  - `LONGFULL_META=1`: drives the forward with device metadata, as the runner does. **This is required for the
    cutoff and dense paths.** The perf/determinism harness traces with host scalars, so it never reaches them.
  - `LONGFULL_SAVE=<prefix>`: saves outputs, for bit-exact A/B checks.
Committed with `--no-verify`: `exp_block_maxima.py` imports the top-k oracle from `tests/ttnn/...`, which the repo's
"models must not import tests" hook forbids. Move it under `tests/` if it's kept.

- **`exp_block_maxima.py`:** checks the indexer maxima against the logits' block max, and top-k with supplied maxima
  against the exact oracle, on every chip. Supports `BMAX_CACHE`, `BMAX_START` and `BMAX_META=1`.
- **`exp_determinism.py`:** the determinism test's on-device check at any cache size (`DET_CACHE`), run count
  (`DET_RUNS`) and prefixes (`DET_PREFIXES=zero,filled`).
- **`op_summary.py`:** per-op device-time medians from a Tracy `ops_perf_results*.csv`
  (`scripts/run_safe_pytest.sh --profile "<node id>"`; `-k` with spaces breaks under the tracy wrapper). It also
  matches each top-k with the following `HighBwAllGather`, i.e. the overlapped gather.

## 6. Results

### 6.1 LoudBox (2x4, Fabric2D)

- **Indexer maxima + two-level top-k only, overlap kept, K3.1 at 1M** (device profile):

  | Op | Single-level | With indexer maxima |
  |---|---|---|
  | Top-k | 4567 µs | 610 µs |
  | Indexer | 5808 µs | 5944 µs |
  | Overlapped KV gather | 6660 µs | 6465 µs |

  End to end is unchanged, because on LoudBox the gather (about 6.5 ms) is the long branch.
- **Push path, zero caches:** −30…−41% at 512k–1M. **Misleading:** with zero caches every key ties, so almost
  nothing is pushed.
- **Push path, filled prefix (random weights), 512k:** slower than the overlap. Each random query picks different
  keys, so about 240k rows per chip are pushed. Random weights are not representative; only real weights decide.
- **Push correctness:** given the same top-k, the push path is bit-identical to the gather path. The cutoff path
  (dense below, push above) is bit-identical to the baseline on the metadata path, eager and traced, both models.
- **Determinism stress, push on:** 45 checks (3 rounds × caches 14080 / 49920 / 510720 × K3.1, GLM, GLM no-indexer ×
  zero/filled), each with 50 eager runs, 50 replays and traced-vs-eager: all identical.

### 6.2 Galaxy, CI sparse-MLA gate (zero caches; push without cutoff; traced ms)

| Case | Baseline | Experiment |
|---|---|---|
| GLM long, 8x4 | 6.047 | 4.921 (−19%) |
| GLM long, 4x4 | 6.646 | 5.106 (−23%) |
| K3.1 long, 8x4 | 7.767 | 6.307 (−19%) |
| GLM warm, 8x4 | 2.537 | 2.986 (+18%) |
| GLM warm, 4x4 | 2.791 | 3.104 (+11%) |
| K3.1 warm, 8x4 | 3.023 | 3.244 (+7%) |

GLM no-indexer layers: unchanged.

### 6.3 Galaxy, real weights

- **GLM-5.3, 78 layers, 55k@5k chunked** (push without cutoff; top-k single-level, since the width is < 65536):
  traced +2.4% slower, untraced +1.6% (+6 → +11 ms per chunk as the context grows).
- **GLM-5.3, 1M request, SC1 runner** (throughput in tok/s; the latest baseline includes #63972):

  | Context | Baseline | Push, no cutoff | Push with cutoff (300k) + indexer maxima |
  |---|---|---|---|
  | 5k | 15,294 | −1.7% | 15,094 (−1.3%) |
  | 50k | 14,855 | −3.1% | 14,673 (−1.2%) |
  | 100k | 14,011 | −4.2% | 13,783 (−1.6%) |
  | 255k | 13,283 | −0.9% | 13,110 (−1.3%) |
  | 510k | 11,510 | +2.3% | 11,654 (+1.3%) |
  | 1020k | 9,877 | +11.8% | 10,917 (+10.5%) |
  | TTFT, 1M | 88.30 s | −3.7% | 85.61 s (−3.0%) |

  KV PCC minimum (nope) is about 0.843–0.846 in all runs (threshold 0.82; known run-to-run variance).
- **K3.1, 24 layers (6 DSA), 1M request, SC1 runner, L1_SMALL 3072:**

  | Context | Baseline | Push with cutoff + maxima |
  |---|---|---|
  | 5k | 26,805 | −0.8% |
  | 100k | 26,015 | −0.7% |
  | 255k | 24,904 | −0.5% |
  | 510k | 22,960 | −1.1% |
  | 1M | 20,597 | +0.3% |

  KV PCC minimum is 0.99667 in both. The no-cutoff run was −1.6…−3.4% up to 510k and +0.5% at 1M.

  The cutoff run above predates commit `f8ca421af83` (the indexer only writes maxima from 98k), so it still paid for
  maxima nothing read below 98k. **That commit hasn't been measured on Galaxy yet.**

### 6.4 Per-layer view at 1M

| | GLM-5.3 | K3.1 |
|---|---|---|
| Chunk time, baseline → experiment | 518.4 → 469.0 ms | 248.6 → 247.9 ms |
| Changed layers | 21 full | 6 DSA |
| Saving per changed layer | ≈ 2.35 ms | ≈ 0.12 ms |

The difference per layer is the open question (§8.1).

## 7. Known issues and gotchas

- **Unexplained determinism failure.** One CI run had K3.1 warm on 8x4, push on, fail the traced-vs-untraced check:
  PCC 0.999893, with eager vs eager 0.999966 and replay vs replay 0.999906.
  - Not reproduced since: 3× the same perf gate, 3× determinism with push on, 1× push off, 1× GLM, all on a Galaxy
    (`topk_push_det` leg), plus about 4,500 LoudBox forwards.
  - Code review found no race. The push has a start fence and an in-order done handshake. The plan-table all-gather
    has a downstream-started fence. Owners push only their own rows, after their KV write.
  - Treat it as open; a longer Galaxy soak is a cheap next step.
- **The push needs** a buffer row count that's a multiple of 512 (`plan_fits`); stripe chunk % 32 == 0; and
  `is_sparse_kv_push_supported` (which, with #64501, means a 2D fabric with routable links).
- **Dense / cutoff mode needs device metadata.** Host-scalar traces (the perf/determinism harness) always push. Use
  `LONGFULL_META=1` or the real runner.
- **K3.1 L1_SMALL.** At the default 2048 B per bank it OOMs on the first push-path sparse SDPA. 3072 works. The cause
  is an extra global semaphore in the compact-buffer path.
- **Two-sided perf gates** everywhere: the sparse-MLA gate (±5% untraced / ±3% traced), the runner TPS/TTFT gates
  (±1.5%), and the GLM chunked per-chunk gate (±3%). Faster fails too.
- **`high_bw_all_gather` without external semaphores** allocates its own pair in L1_SMALL on every program cache miss.
  #63972's dense gather therefore uses the model's shared gather semaphores; the full-layer overlap gather uses the
  overlap resources' semaphores.
- **Tie choice:** two-level vs single-level top-k changes which tied keys are chosen. Expect PCC ~0.996 between them
  on random data; it is not a bug.

## 8. Next steps (suggested order)

1. **Profile K3.1 DSA vs GLM full layer in the real runner on Galaxy.**
   - The workflow builds with Tracy only when `gemma4_d_p_layer_perf` is ticked (`tracy:` input of `build-artifact`).
     On this branch, make that also true for `topk_push_long`, and run the runner legs with the device profiler.
   - Compare per-op times for `sparse_push_plan`, the plan-table `high_bw_all_gather`, `sparse_kv_push`, the overlap
     `high_bw_all_gather`, `topk_large_indices`, `ring_indexer_score_dsa` and sparse SDPA, baseline vs experiment.
   - Count distinct pushed rows per chip, from the plan table or by instrumenting.
   - This tells whether K3.1 loses to the plan (fix: cheaper or skipped plan) or to the push volume.
2. **Skip the full layer's own plan below its cutoff.**
   - It's dead work there; only the dense gather matters.
   - Options: a device-side guard in `sparse_push_plan` (like `gathered_active_below`), or a single plan that serves
     both thresholds.
   - On GLM, avoid building two plans per full layer (its own + the shared layers').
3. **Measure `f8ca421af83`** (indexer maxima gated at 98k) with the `topk_push_long` A/B.
4. **Tune `SPARSE_MLA_FULL_PUSH_DENSE_BELOW` per model** on the GLM and K3.1 1M legs (try 400k and 500k). The GLM
   crossover appears to sit between 255k and 510k.
5. **Hide the full layer's push.**
   - Top-k is available before the Q/KV stems, so the plan and the PREFIX part of the push could run on the push
     sub-device beside the Q and KV stems, as `_next_layer_prefetch` already does for shared layers.
   - That leaves only the REST (current chunk) push serial.
   - This should lower the crossover, and it's the most promising structural win.
6. **Productionize what pays off.**
   - Replace the environment switches with model-config flags.
   - Add indexer unit tests for `block_max_output` / `block_max_from_length` (scalar and metadata paths, block-cyclic
     ring), and model tests for the cutoff path.
   - Revert the CI hacks: report-only gate, repurposed checkboxes, K3.1 request length.
   - Re-center the perf baselines.
   - Get reviews.
7. **Longer-term target:** fusing the indexer, top-k and the KV fetch into one op.
