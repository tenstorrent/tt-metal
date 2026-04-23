# Matmul helpers `opt_2` review and migration plan

Branch: `wransom/matmul_helpers_opt_2`
Status as of: 2026-04-23, 22 commits since divergence from `wransom/matmul_helpers_cleanup`.

---

## Part 1 — Foundation

### What the Claude matmul helpers are

`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}` and the companion
`bias_add_helpers.{hpp,inl}` are reusable compute-kernel helpers that the
matmul factories call into instead of writing their own MATH/pack/unpack
sequences. The design goal was a single optimization point — a SINGLE
implementation of `matmul_block` that handles all the variants (sharded
inputs, fused bias, packer L1 accumulation, K-block spilling to interm CB)
so that perf improvements made there apply to every caller automatically.

Pre-`opt_2`, those helpers were used by the non-mcast factory and SDPA. The
`opt_2` line of work extended their reach to the mcast factories.

### Two layout modes the helper supports

The compute kernel's pack phase has two modes, gated by the `ROW_MAJOR_OUTPUT`
compile define:

- **Subblock-major** (legacy, default off): the helper calls
  `pack_tile_block(subblock_tile_id)` — packs each subblock's tiles into the
  output CB contiguously. The writer kernel reads tiles in subblock-major
  order. Constraint: `out_subblock_w == per_core_N || out_subblock_h == 1`
  (single N-subblock per row-group OR single-row subblock). FATAL gate
  enforces this.

- **Row-major** (`ROW_MAJOR_OUTPUT=1`): the helper calls
  `pack_tile<true>(dst_idx, cb, absolute_offset)` — packs each tile at its
  row-major absolute position in the output CB. Writer reads tiles per
  M-row-group. The above constraint is gone — any `(out_subblock_h,
  out_subblock_w)` works.

The `row_major_output` field on the program-config struct is what tells the
factory to emit `ROW_MAJOR_OUTPUT=1`. It defaults to `False` for backward
compatibility — every existing model config keeps the legacy behavior unless
it explicitly opts in.

### Why row-major unlocks perf

Two compounding effects:
1. **Multi-row subblocks with `out_subblock_w < per_core_N` become legal**.
   Pre-row-major, you had to either pack a single-row subblock (h=1) or fill
   the entire N axis (w=per_core_N). Now you can pick any (h, w) pair that
   fits DST. Phase-3 measurement on 2048³ DRAM matmul: legacy
   auto-config picked subblock (1, 2) volume 6/8; row-major-enabled
   auto-config picks (1, 8) volume 8/8 → **-62%** on a synthetic baseline
   (more typical: -10 to -25% on real shapes).
2. **The helper's pack fast path activates when h=1**. Row-major layout at
   h=1 equals subblock-major layout, so the helper short-circuits to
   `pack_tile_block` with zero per-tile LLK overhead. No-op gain here, just
   free.

### Bug 3 — the corruption that gated row_major_output rollout

When `row_major_output=True`, the helper packs each subblock at absolute
offsets into out_cb per M-row-group (smaller granularity than the legacy
"reserve full out_block up front"). Pre-Bug-3, the factory shared `out_cb`
and `interm0_cb` L1 regions. The per-row-group out_cb writes overlapped with
unconsumed interm0 partials from later N-subblocks in the same row-group →
silent corruption (PCC 0.85-0.91).

Commit `b3c0b84c0ed` fixed this by forcing separate L1 regions for `out_cb`
and `interm0_cb` whenever `row_major_output` is on (mcast paths) or whenever
the non-mcast factory is in play (it now always emits `ROW_MAJOR_OUTPUT=1`).
The cost is roughly doubled output-space L1 footprint (out + interm vs
max(out, interm)) but only on the row_major_output codepath. Configs that
keep `row_major_output=False` see no change.

This is the single most important correctness fix on the branch — before it,
multi-row subblocks with `in1_num_subblocks > 2` produced bad data. After
it, they're correct across all mcast and non-mcast factories.

---

## Part 2 — What the branch contains

22 commits since divergence from `wransom/matmul_helpers_cleanup`, in seven
conceptual groups.

### Group A: Helper foundation (4 commits)
Predecessor work that landed here, unifying matmul_block for bmm + SDPA,
extracting transpose/reblock-untilize helpers, and adding isolation tests.
```
a063dd33c63  Unify matmul_block helper for bmm + SDPA, fix sharded bmm corruption
f7437f3a721  Extract transpose/reblock-untilize helpers, absorb matmul_reduce init
2e75c56de3a  Add isolated helper tests for kernel_lib matmul helpers
abc8c46e25e  Fix matmul_reduce_inplace test infra
```

### Group B: Helper bug fixes surfaced during opt_2 (2 commits)
```
6d1d61f0d55  Fix matmul_block helper: plumb actual in1 shard width
e33c9a2d54e  Fix Wormhole regressions in matmul_block helper + optimized factory
```

### Group C: Non-mcast FATAL relaxation (2 commits)
The non-mcast factory used to FATAL on multi-row subblocks. With the helper
fast path active there (always-on `ROW_MAJOR_OUTPUT=1` after Group D), the
constraint was obsolete.
```
382970df6e2  Allow multi-row subblocks for sharded-output non-multicast matmul
b3a9105f27e  Drop obsolete subblock FATAL on MatmulMultiCoreReuseProgramConfig
```

### Group D: Bug 3 fix (1 commit)
The corruption gate. Touches all four factory families (2D mcast, 1D mcast,
DRAM-sharded mcast, non-mcast).
```
b3c0b84c0ed  Fix row_major_output corruption from shared out_cb/interm0 L1 region
```

### Group E: row_major_output extension to mcast (2 commits)
The big functional unlock. Adds the `row_major_output` field to the two
mcast program-config types, plumbs it through the factories and writer
kernels (dual-mode pack/read), gates FATALs in matmul_device_operation on
the flag.
```
4a2c288dbb1  Enable row_major_output on mcast matmul factories (Phase 1 + 2 Path A)
539fa5ad753  Add Tracy perf scripts for mcast row_major_output demonstration
```
Hand-tuned demonstration scripts in `tests/scripts/matmul_perf/` show
**-41% / -62% / -12%** on 2D mcast fuse_bias / fast_path / multi_row vs
deliberately naive baselines.

### Group F: Auto-tuner infrastructure (5 commits)
A new module `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_tuner.{hpp,cpp}`
with two pure functions: `determine_largest_subblock` and
`determine_largest_in0_block_w`. Used in the `get_program_config` path that
synthesizes a config when the user calls `ttnn.matmul(a, b)` without one.
21 unit tests cover the math.
```
8e9f28d664b  Add matmul auto-tuner module (subblock + K-iteration selection)
3c13eb3       Route matmul auto-config through auto_tuner, enable row_major_output
f9eb6a344a0  Gate auto-config row_major_output on L1 CB fit check
467300309b5  Wire K-iteration tuner into get_mcast_1d_config
243e67167ab  Make L1 safety margin adaptive to output tensor footprint
```
The L1 safety margin is derived from the current op's output tensor
footprint (DRAM: 4 KB floor; sharded L1: per_core_M × per_core_N ×
tile_size; interleaved L1: div_up(Mt × Nt, num_banks) × tile_size). No
hardcoded magic numbers in the final state.

### Group G: Auto-tuner perf scripts (2 commits)
```
c12697ce84a  Add auto-tuner perf scripts: autotune vs pre-refactor vs hand-tuned
3cefddc73c6  Pivot auto-tuner perf scripts to DRAM-out: -12.2% on 2048^3
```

### Group H: Sentence-BERT proof of concept (2 commits)
```
0a421bb6a56  sentence-BERT: upgrade ff2 + qkv matmul configs to exploit helper fast path
7a89d6a8f55  sentence-BERT: raise device-perf threshold to lock in -21.4% matmul win
```

---

## Part 3 — The sentence-BERT result

This is the working blueprint for the migration effort.

### What we changed
File: `models/demos/wormhole/sentence_bert/ttnn/common.py`. Four mcast
program configs in this file. Edits per `0a421bb6a56`:

| Config | Before | After | Type of change |
|---|---|---|---|
| `ff1_matmul_program_config` | (1, 8), no rmo | (1, 8), `row_major_output=True` | Flag-only; subblock already at max DST volume |
| `ff2_program_config` | (1, 6), no rmo | **(4, 2)**, `row_major_output=True` | Flag + reshape; `(4, 2)` previously rejected by FATAL gate |
| `query_key_value_matmul_program_config` | (1, 6), no rmo | **(4, 2)**, `row_major_output=True` | Same as ff2 |
| `self_out_program_config` | (2, 4), no rmo | (2, 4), `row_major_output=True` | Flag-only; subblock already at max DST volume |

Twelve lines edited total. No model code changes. No test code changes (yet
— the perf-test threshold edit comes next).

### Measured results (WH n150)

| Metric | Baseline | After upgrade | Δ |
|---|---:|---:|---:|
| Total matmul device-kernel time (Tracy, sum across all matmul ops) | 13,385,817 ns | 10,515,489 ns | **-21.4%** |
| Device-kernel samples/sec (model perf test) | 460 sps | 546.5 sps | **+18.8%** |
| **End-to-end samples/sec (e2e perf test, 50-iter loop wall clock)** | **425 sps** | **506 sps** | **+19.1%** |
| End-to-end inference time per batch (8 samples) | 18.81 ms | 15.83 ms | **-15.8%** |
| PCC (numerical correctness) | passes | passes | no change |

The matmul-time win compounds almost 1:1 into device-kernel-total (matmul
is ~78% of device kernel time in this model) and propagates directly into
e2e wall-clock. **This is the real story to tell about each migrated
model**: an end-to-end FPS number that user-visible workloads care about,
backed by the device-kernel-time number that tells you the kernel actually
got faster (and isn't a measurement artifact).

### Regression guards locked in
1. **`test_perf_device_bare_metal_sentence_bert`** — `expected_perf` raised
   from 460 to 546.5 sps with ±3% margin. Floor is now 530.1 sps. Any
   future change that drops device-kernel sps below 530.1 fails this test.
2. **`test_ttnn_sentence_bert_model`** (PCC) — was already there. Continues
   to pass post-upgrade. Catches numerical regressions.
3. **`test_e2e_performant_sentencebert`** — runs but does not assert against
   a threshold today (just logs the FPS). **Recommendation for future
   work**: add a threshold here too, set to the new 506 sps with margin.

---

## Part 4 — Why this matters for the broader migration

There are 76 files under `models/` that construct
`MatmulMultiCoreReuseMultiCast{,1D}ProgramConfig` explicitly. Sentence-BERT
is one of them. The sentence-BERT pattern of edits — flag-add (mechanical),
optionally reshape subblock (per-config judgment) — applies to every one of
those files.

Expected per-model result range:
- **Models whose hand-tuned subblocks were already at max DST volume**
  (ff1, self_out pattern): **0% kernel win**, no edit beyond flag-add.
  Skip these — adding the flag without a measurable benefit is just
  CI-noise and review-noise. **Below the >5% bar.**
- **Models with one or more configs stuck at sub-max-DST volume because of
  the legacy writer constraint** (ff2, qkv pattern): **5% to 25% kernel win
  per such config**, scaled by that config's share of total inference time.
  These are worth keeping.
- **Models that also have `in0_block_w` set conservatively** (e.g.
  in0_block_w=1 or 2 when L1 has room for 4 or 8): can layer K-iteration
  upgrades on top. Phase-3 hand-tuned scripts showed K-iteration was
  responsible for ~30-40% of the synthetic-baseline wins.

Sentence-BERT was 2 of 4 configs reshapeable, 21.4% matmul-kernel win,
19.1% e2e win. A reasonable working assumption: **half of the 76 model
files have at least one upgradable config; per-model e2e wins will
typically land in the 5-20% range**, with outliers at both ends. Some
models won't move at all (skip) and a few may move dramatically (those are
the wins that justify the effort).

---

## Part 5 — Migration plan

### Bar for keeping a per-model change
Per your guidance: **keep only changes with >5% AND consistent (clearly not
noise) e2e improvement**. Concretely:
- Run the model's e2e perf test 3 times pre-edit, 3 times post-edit.
- Compare medians. Improvement must be ≥5%.
- Spread across the 3 runs in each direction must be small (e.g.
  std-dev/mean < 2%) — otherwise call it noise and skip.
- PCC must continue to pass.

If improvement is between 0 and 5%, or noise is large, **revert and move
on**. Adding rows to the diff for sub-5% wins clutters review and makes
regression hunting harder when the next person looks at this.

### The three tuning knobs (use in this order)

For each matmul config, three independent levers can deliver perf:

1. **`row_major_output=True` flag** — *required* for any of the unlocks
   below to take effect. Adds the `row_major_output=True,` kwarg to the
   `ProgramConfig(...)` constructor. By itself does nothing if the
   existing subblock is already at max DST volume; it just enables the
   helper's row-major pack path so further reshapes become legal. **This
   is the cheapest knob and the gate for the other two.**

2. **`out_subblock_h` / `out_subblock_w` reshape (subblock tuning)** —
   pick the (h, w) pair that maximizes `h * w` subject to `h * w ≤
   DST_capacity` (typically 8 for half-sync bf16, 4 for fp32_dest_acc_en),
   `h | per_core_M`, `w | per_core_N`. Multi-row pairs like `(4, 2)` and
   `(2, 4)` were rejected by FATAL pre-row-major; they're legal now with
   knob #1 on. Sentence-BERT's win came almost entirely from this knob:
   `(1, 6)` volume 6/8 → `(4, 2)` volume 8/8. Order of preference: `(1, N)`
   or `(N, 1)` if they fit (helper pack fast path), then `(h, w)` with
   both > 1 to fill DST.

3. **`in0_block_w` bump (K-iteration tuning)** — if the current value is
   small relative to `Kt` and L1 has headroom, larger `in0_block_w` cuts
   outer K-loop iterations and amortizes MATH/unpack setup. Conservative
   sentence-BERT did not need this knob; the Phase-3 hand-tuned scripts
   showed it was responsible for ~30-40% of the synthetic-baseline wins on
   shapes where in0_block_w was very small (1 or 2). When in doubt leave
   it alone — wrong values trigger L1-overflow at compile time, easy to
   notice.

**Auto-tuning** (the matmul auto-tuner module) is *not* the right
mechanism for these manual upgrades. The auto-tuner only fires on
`ttnn.matmul(a, b)` calls without an explicit program_config. Every model
in the migration list passes a program_config, so the tuner is bypassed.
Future work: a config-augmentation path that runs the tuner on
user-passed configs (sketched in Part 7); not in this branch. **For now,
manual tuning per the recipe below.**

### Per-model recipe (~30-90 min per model)

1. **Pick the model.** Initial priority list (justified below): Falcon7b,
   GPT-OSS, DeepSeek-V3, Whisper, BGE-large, BGE-m3, Stable-Diffusion-XL,
   Llama3, Vit, Sentence-BERT (already done as the reference).

2. **Find the configs.** `grep -rn "MatmulMultiCoreReuseMultiCast" models/demos/<model>`.

3. **Find the perf test and record its current `expected_perf` value.**
   Look for `models/demos/<model>/tests/perf/test_*_perf.py` and
   `test_*_e2e*.py`. Note the exact value used in the
   `expected_perf` parametrize tuple (e.g. `[8, 460.0, "sentence_bert"]`).
   You will need this value for both the baseline calibration AND the
   threshold update at step 8.

4. **Establish baseline.** Run the e2e perf test 3 times. Record median
   samples/sec and inference time. (If only a device-kernel perf test
   exists, run that too and record `AVG DEVICE KERNEL SAMPLES/S` from the
   output — that's the value the test threshold tracks.)

5. **Edit configs in `tt/` or `ttnn/` directory.** Apply the three knobs
   in order from the section above. Be conservative on knob #3.

6. **Run PCC tests.** Most models have a `test_ttnn_<model>_model.py`
   style test that asserts PCC. Must pass. If a model has multiple PCC
   tests (per-layer, per-encoder, full-model), run them all. **If any PCC
   test fails, revert and stop — perf gains do not justify wrong math.**

7. **Run e2e perf test 3x post-edit.** Compute median.

8. **Decision.**
   - If e2e median improved >5% AND not noise (std-dev/mean < 2% across
     runs) → **keep** the config edits, AND **raise the perf-test
     threshold** to the new measured value. Both edits go in the same
     commit. **The threshold update is mandatory** — see warning below.
   - Otherwise → `git checkout -- models/demos/<model>/...` and skip the
     model. Do not commit a config edit without a measurable, locked-in
     win.

> **⚠️ Why the perf-test threshold update is mandatory, not optional.**
> Model perf tests in `models/demos/*/tests/perf/` use a `±3% margin`
> band around `expected_perf`. The test fails if measured perf falls
> *outside* that band in either direction — too slow OR too fast. If you
> ship a config edit that improves perf by 19% (like sentence-BERT did)
> without raising the threshold, the test goes from "passing at 460" to
> "FAILING at 546 because 546 > 460 × 1.03 = 473.8". CI will flag the
> change as a regression and someone will revert it, undoing the win.
> Always update both numbers in the same commit, modeled after
> `7a89d6a8f55`. This pattern caught us during the sentence-BERT
> migration — verify the test passes locally with the new threshold
> before pushing.

9. **Commit.** One commit per model. Title: `<model>: upgrade matmul
   configs to exploit helper fast path`. Body: list edited configs,
   measured baseline + post numbers (device-kernel + e2e), PCC status,
   confirmation that the perf-test threshold was updated. Mirror the
   structure of `0a421bb6a56` (config edit) and `7a89d6a8f55` (threshold
   update) — these can be one combined commit per model rather than two,
   since they're a single logical change.

### Suggested initial 5-model batch
Ordered by likely impact:

1. **Falcon7b** (`models/demos/falcon7b_common/tt/model_config.py`) — LM
   head matmuls are large, perf-sensitive.
2. **DeepSeek-V3** (`models/demos/deepseek_v3*/`) — multiple config files,
   active development, perf matters.
3. **GPT-OSS** (`models/demos/gpt_oss/tt/{attention,experts,experts_throughput}/config.py`)
   — four config files, transformer-typical shapes.
4. **Stable-Diffusion-XL** (`models/demos/vision/generative/stable_diffusion/wormhole/`)
   — heavy matmul workload, multiple config sites.
5. **BGE-large** (`models/demos/wormhole/bge_large_en/ttnn/common.py`) —
   sentence-BERT analogue, similar shape pattern.

Run these one at a time, batching the per-model commit and threshold-raise
into a single commit per model. After the first 5, reassess: if the wins
are coming through consistently, the next batch can be parallelized
(different models don't conflict).

### Reportable outcome
At the end, the migration report should be a table:

| Model | Baseline e2e (sps) | After (sps) | Δ | Configs touched | Notes |
|---|---:|---:|---:|---:|---|
| sentence-BERT | 425 | 506 | +19.1% | 2/4 reshape + 4/4 flag | reference impl |
| Falcon7b | ? | ? | ? | ? | |
| DeepSeek-V3 | ? | ? | ? | ? | |
| ... | | | | | |
| Models skipped (no win >5%) | — | — | — | — | distilbert, ... |

This is the deliverable that justifies the effort. "Sentence-BERT 425→506
FPS" is exactly the right shape of statement.

---

## Part 6 — Correctness throughout

The single non-negotiable: **PCC must pass** on every changed model. The
helper itself is correct (Bug 3 fixed it for the only known corruption
case; 32/32 row-major-output unit tests + 21/21 auto-tuner gtests + 589/589
matmul unit tests cover the infrastructure). But model-specific PCC tests
catch numerical-regression interactions — particularly bf16 accumulation
order changing when `in0_block_w` grows, which is why the auto-tuner caps
K at `kMaxAutoTunedInBlockW = 4` (canary: silu + sharded output PCC drops
below 0.9999 above 4).

For each migrated model:
1. Run the PCC test. If it fails, revert.
2. If a model has multiple PCC tests (per-layer, per-encoder, full-model),
   run them all.
3. If PCC degrades but stays above the model's threshold, that's fine. If
   it drops below, the change is wrong even if perf improved.

The ordering — PCC first, then perf — should be enforced in the per-model
recipe. Don't celebrate a perf win that broke correctness.

---

## Part 7 — What still needs work after the migration

These are **out of scope for the migration but worth tracking**:

1. **Auto-tuner integration tests**. The 21 unit tests cover the pure
   functions; nothing explicitly validates the integration emits the right
   config on representative shapes. test_matmul.py's 589 tests provide
   incidental coverage but don't assert on tuner choices. Recommendation:
   add `test_matmul_auto_tuner_integration.py` that calls
   `ttnn.matmul(a, b)` on canonical shapes and asserts the synthesized
   config matches expectations.
2. **Auto-upgrade path for manual configs**. Today the auto-tuner only
   fires when no program_config is passed. The "auto-tuned solutions
   preferred" goal would be served by a path that augments user-passed
   configs (e.g. flip `row_major_output` to True if the existing subblock
   is compatible and L1 fits). Not in this branch. Designed-but-unbuilt;
   probably 1-2 days of work.
3. **`get_per_core_factor` undersubscription**. Pre-existing quirk where
   L1-output shapes can leave most cores idle. Not introduced by this
   branch. Documented as a follow-on; would amplify auto-tuner wins on the
   L1-output codepath.
4. **DRAM-sharded factories**. Don't have `row_major_output` field on
   their config types yet — factory hardcodes ROW_MAJOR_OUTPUT=1. Adding
   the field + plumbing for parity with mcast is a follow-on PR.

---

## TL;DR

- Branch infrastructure is clean and tested. Sentence-BERT is the working
  proof: **+19.1% e2e** from a 12-line edit to one file's matmul configs.
- Migration to the other 75 model files follows the sentence-BERT
  blueprint: per-config flag-add + optional subblock reshape, measure
  PCC + e2e perf, keep only changes with >5% e2e win, raise the perf-test
  threshold to lock the win.
- The reportable outcome is a table of "Model X went from N FPS to M FPS"
  rows, focused on e2e (the user-visible number) backed by device-kernel
  (proves the kernel actually got faster).
- Correctness gating per-model PCC test is non-negotiable. Perf gating at
  >5% e2e improvement keeps the diff and CI signal clean.
