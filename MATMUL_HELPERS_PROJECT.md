# Matmul Helpers Project — Single-source summary

Consolidated reference for the matmul helpers work. This file supersedes
the previous split between `MATMUL_HELPERS_OPT2_REVIEW.md`,
`MATMUL_HELPERS_OPT3_RESULTS.md`, and the per-phase memory notes.

- Current branch: `wransom/matmul_helpers_opt_3`
- Latest divergence date: 2026-04-24 (before the reduce-helper merge into main)
- Prior branches (all squashed into this narrative): `wransom/matmul_helpers_cleanup`,
  `wransom/matmul_helpers_opt`, `wransom/matmul_helpers_opt_2`
- Hardware used for verification: Wormhole B0 n150 + Blackhole p100a

---

## 1. Executive summary

**Project goal.** Replace the copy-pasted per-factory MATH/pack/unpack sequences in
TTNN's matmul factories with a single reusable compute-kernel helper (`matmul_block_helpers`),
then exploit the unified helper's row-major pack path to unlock subblock shapes that the legacy
subblock-major writer forbade. Net result: models whose subblock choices were pinned to
sub-max DST volume by the legacy FATAL gate can now use `(h, w)` pairs that fill DST, turning
every legal subblock into wall-clock perf.

**Current state of the branch.**
- Unified helper + non-mcast `row_major_output=1` landed (phase-1/2 on `matmul_helpers_cleanup`).
- 2D/1D mcast factories extended to honor `row_major_output=True` (opt_2 Groups D+E). Bug 3
  fix (separate `out_cb`/`interm0_cb` L1 regions) is what made this safe.
- Matmul auto-tuner module (subblock + K-iteration) wired into the default `ttnn.matmul(a, b)`
  path (opt_2 Group F). All existing unit tests pass.
- Per-model manual migration underway (opt_3). Two wins landed: **sentence-BERT −21.4%
  matmul / +19.1% e2e** (upstream), **Falcon7b prefill seq=1024 +19.9% device-kernel sps**
  (verified +27.3% on same-hardware A/B). Auto-tuner divisibility bug found on BH and fixed.
- Several models surveyed; most remaining wins are blocked on weights downloads or multi-chip
  hardware access. See section 4 for the full ledger.

**What's pending.** A handful of cleanly-tractable follow-ons (section 5), plus the impending
rebase onto main after the reduce-helper library merge (section 6).

---

## 2. Key concepts

### 2.1 The helper

`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}` + companion
`bias_add_helpers.{hpp,inl}` are reusable compute-kernel helpers. Matmul factories
call into them instead of writing their own MATH/pack/unpack sequences. Single
optimization point — a single implementation of `matmul_block` that handles all
the variants (sharded inputs, fused bias, packer L1 accumulation, K-block spilling
to interm CB) so that perf improvements there apply to every caller automatically.

Pre-opt_2, the helpers were used by the non-mcast factory and SDPA. Opt_2 extended
their reach to the 2D mcast, 1D mcast, and DRAM-sharded mcast factories.

### 2.2 Two pack layouts

The compute kernel's pack phase has two modes, gated by the `ROW_MAJOR_OUTPUT`
compile define:

- **Subblock-major** (legacy, default): helper calls `pack_tile_block(subblock_tile_id)`.
  Writer reads tiles in subblock-major order. FATAL gate: `out_subblock_w == per_core_N
  || out_subblock_h == 1` (single N-subblock per row-group OR single-row subblock).
- **Row-major** (`ROW_MAJOR_OUTPUT=1`): helper calls `pack_tile<true>(dst_idx, cb,
  absolute_offset)`. Writer reads per M-row-group. Constraint gone — any `(h, w)` works.

The `row_major_output` field on the program-config struct is what tells the factory
to emit `ROW_MAJOR_OUTPUT=1`. Defaults to `False` for back-compat — every existing
config keeps legacy behavior unless it opts in.

### 2.3 Why row-major unlocks perf

1. **Multi-row subblocks with `out_subblock_w < per_core_N` become legal**. Pre-row-major,
   you were stuck at either `h=1` or `w=per_core_N`. Now you can pick any `(h, w)` that fits
   DST. Sentence-BERT's ff2/qkv went `(1, 6)` volume 6/8 → `(4, 2)` volume 8/8 — same DST
   slot count, every slot used.
2. **Helper pack fast path activates when h=1**. Row-major at `h=1` equals subblock-major, so
   the helper short-circuits to `pack_tile_block` with zero per-tile LLK overhead. Free.

### 2.4 Bug 3 — the corruption that gated the rollout

With `row_major_output=True`, the helper packs per M-row-group at absolute offsets (smaller
granularity than legacy "reserve full out_block up front"). Pre-Bug-3, the factory **shared**
`out_cb` and `interm0_cb` L1 regions. Per-row-group out_cb writes overlapped with unconsumed
interm0 partials → silent corruption (PCC 0.85-0.91).

**Fix** (`b3c0b84c0ed`): factories force separate L1 regions for `out_cb`/`interm0_cb`
whenever `row_major_output` is on, or whenever the non-mcast factory is in play (which now
always emits `ROW_MAJOR_OUTPUT=1`). Cost: ~doubled output-space L1 footprint, but only on the
row-major codepath. Configs with `row_major_output=False` see no change.

Single most important correctness fix on the branch.

### 2.5 DST capacity abstraction

DST register-file capacity is not a constant. Varies with:
- Sync mode (half vs full sync): `DST_SYNC_MODE`
- fp32 accumulation (`fp32_dest_acc_en=True` halves capacity)
- Tile shape (non-32×32 tiles)
- Potentially across architectures (future-risk; today all three supported arches agree)

**Host-side query API is already abstract**. Auto-tuner callers pass
`DeviceComputeKernelConfig` and the tuner routes through
`ttnn::get_dest_reg_count(config, tile_shape)`, which handles sync mode, fp32, and custom
tile shapes. Kernel-side is `compute_kernel_lib::get_dest_limit()` in
`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`. 21 auto-tuner gtests cover the DST math across
all 4 capacity combos.

**Known-but-not-fixed** (section 5):
- Host-side `DATUMS_PER_ROW=16` and `DEST_REGISTER_FULL_SIZE=64*16` are re-`#define`d in
  `ttnn/.../compute_kernel_config.cpp` as copies of arch tensix_types.h constants.
- `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:151` has
  `max_subblock_w = fp32_dest_acc_en ? 4 : 8` — legacy hardcode bypassing `get_dest_reg_count`.

### 2.6 Auto-tuner module

`ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_tuner.{hpp,cpp}` exposes:
- `determine_largest_subblock(inputs)` — picks max `(h, w)` given DST capacity + per_core_M/N
  divisibility + optional constraints (`subblock_w_eq_per_core_n_required`, etc). Two tables
  for fast-path-first vs legacy ordering.
- `determine_largest_in0_block_w(inputs)` — picks max K-block size fitting in L1 budget.

Called from `get_program_config` in `matmul_program_config.cpp` when the user runs
`ttnn.matmul(a, b)` without an explicit `program_config`. **Does not fire on configs
passed explicitly** — hence the manual migration effort in opt_3. Unit tests in
`tests/ttnn/unit_tests/gtests/test_matmul_auto_tuner.cpp` (21 tests, all pass).

---

## 3. Project arc — what each phase delivered

### Phase 1 (`wransom/matmul_helpers_cleanup`, commits pre-opt_2)

Foundation. Unified `matmul_block` helper for bmm + SDPA. Extracted
transpose/reblock-untilize helpers. Absorbed matmul_reduce init. Added isolation tests.
Two bug fixes surfaced (the `in1_per_core_w` plumbing issue and the bias helper row-major
pack path).

- `a063dd33c63` Unify matmul_block helper for bmm + SDPA, fix sharded bmm corruption
- `f7437f3a721` Extract transpose/reblock-untilize helpers, absorb matmul_reduce init
- `2e75c56de3a` Add isolated helper tests for kernel_lib matmul helpers
- `abc8c46e25e` Fix matmul_reduce_inplace test infra
- `6d1d61f0d55` Fix matmul_block helper: plumb actual in1 shard width
- `e33c9a2d54e` Fix Wormhole regressions in matmul_block helper + optimized factory

### Phase 2 — WH verification (same branch)

BH phase-2 said "perf parity or better across canonical shapes". When the branch was
cross-checked on WH n150, surfaced two WH-specific regressions not visible on BH:

1. **4 L1-overflow transpose tests** — `per_core_M=32,per_core_N=16` transpose combos
   passed on WH main in 7 s but failed on branch with L1 > 1.5 MB. Root cause: commit
   `6b5f3dcbe73` (pre-Bug-3) **always** allocated separate out/interm CBs. With matching
   bf16 formats the branch now needed 2× output-region L1. WH's 1.5 MB L1 is 73 KB
   tighter than BH's 1.57 MB — fine on BH, overflow on WH. Fixed in `e33c9a2d54e`.
2. **+4.6% perf regression** on one shape, traced to a helper pack/wait simplification
   that hurt in a specific code path. Fixed same commit.

Final WH state after fixes: 816 correctness tests pass, 0 perf regressions, 2 wins,
parity with BH.

### `wransom/matmul_helpers_opt` — non-mcast FATAL relaxation + hand-tuned demonstrations

- `382970df6e2` Allow multi-row subblocks for sharded-output non-multicast matmul
- `b3a9105f27e` Drop obsolete subblock FATAL on `MatmulMultiCoreReuseProgramConfig`

Plus hand-tuned perf demonstrations (dropped from branch during rebase, preserved in memory
only): 512×512×512 + bias at 1 core (-75%), 2048³ DRAM matmul (-72% with optimal subblock +
ibw), and two smaller wins at -13.5% and -1.4%. Story intent: show what the helper could do
when fed the best-case config — motivated the auto-tuner work.

### `wransom/matmul_helpers_opt_2` — mcast extension + auto-tuner + three bug fixes

22 commits. Seven conceptual groups (A-H per the opt_2 review):
- **Group D (1 commit)**: Bug 3 fix (`b3c0b84c0ed`). Separate out_cb/interm0_cb L1 regions.
  Gates everything below.
- **Group E (2 commits)**: Row-major pack extended to 2D mcast + 1D mcast factories. Adds
  `row_major_output` field to mcast program-config types. Plumbs through writer kernels
  (dual-mode pack/read). Gates FATALs in `matmul_device_operation` on the flag.
  - `4a2c288dbb1` Enable row_major_output on mcast matmul factories (Phase 1 + 2 Path A)
  - `539fa5ad753` Add Tracy perf scripts for mcast row_major_output demonstration
- **Group F (5 commits)**: Auto-tuner module + integration. Pure gtest-able functions,
  adaptive L1 safety margin, K-iteration wiring for 1D mcast, row_major_output gated on L1
  fit check.
  - `8e9f28d664b` Add matmul auto-tuner module (subblock + K-iteration selection)
  - `3c13eb38b85` Route matmul auto-config through auto_tuner, enable row_major_output
  - `f9eb6a344a0` Gate auto-config row_major_output on L1 CB fit check
  - `467300309b5` Wire K-iteration tuner into get_mcast_1d_config
  - `243e67167ab` Make L1 safety margin adaptive to output tensor footprint
- **Group G (2 commits)**: auto-tuner Tracy perf scripts (`c12697ce84a`, `3cefddc73c6`).
- **Group H (2 commits)**: Sentence-BERT proof of concept — the first real-model win.

Three additional bug fixes during opt_2 not listed above:
- **Bug 1** — DRAM-sharded `in1_per_core_w` plumbing was overloaded (in1 CB read width vs
  output pack stride); diverged in DRAM-sharded only. Split into `in1_per_core_w` +
  `out_row_width` parameters in `matmul_block` helper. Fixed the N=4096 hang.
- **Bug 2** — Bias helper row-major pack path: bias values not in the right DST slots on the
  row-major path. Threaded through the bias helper alongside the main pack path.
- **Bug 3** — Described in 2.4 above. The big one.

**Sentence-BERT result** (reference migration for everything that follows):
- File: `models/demos/wormhole/sentence_bert/ttnn/common.py`. 4 mcast configs.
- Edits: ff2 + qkv reshape `(1, 6) → (4, 2)` with `row_major_output=True`. ff1 + self_out
  flag-only (already at max DST volume).
- Measured A/B (same session, WH n150, Tracy):
  - Baseline (no rmo): 13,385,817 ns total matmul time
  - rmo flag only: 13,392,078 ns (+0.05%, confirms flag alone is a no-op)
  - rmo + (4,2) reshape on ff2/qkv: 10,515,489 ns (**−21.4%**)
  - Model device-perf test: 460 → 546.5 sps (+18.8%)
  - E2E perf test: 425 → 506 sps (**+19.1%**, 18.81 → 15.83 ms per batch of 8)
- Commits: `0a421bb6a56` (config edit) + `7a89d6a8f55` (threshold bump to 546.5 sps with ±3%).

### `wransom/matmul_helpers_opt_3` — per-model migration

Current branch. Applies the sentence-BERT recipe to other models. See section 4 for the
results ledger. Also landed one auto-tuner correctness fix (`63d9bc9e4da`, see section 5.2).

---

## 4. Per-model results ledger

Recipe per model (ordered levers):
1. Try the sentence-BERT pattern — reshape `(out_subblock_h, out_subblock_w)` to max
   `h*w ≤ DST_capacity` with `h | per_core_M, w | per_core_N`, using `row_major_output=True`
   if the chosen `(h, w)` needs it (`h>1 AND w<per_core_N`).
2. L1 fit check: rmo forces separate out/interm CBs (~`per_core_M * per_core_N * tile_size`
   bytes added). If compile asserts on `Statically allocated circular buffers clash with L1
   buffers`, fall back to legacy-compatible max (`(1, w)` with `w = largest divisor of
   per_core_N that fits DST`, no rmo flag).
3. Skip lever 3 (K-iteration / `in0_block_w`) — sentence-BERT didn't use it; keeps recipe
   uniform.
4. Skip flag-only adds where existing subblock is already at max DST volume — rmo alone is a
   measured no-op (+0.05% on sentence-BERT).

Keep/revert gate: PCC must pass (hard). Device-kernel or e2e median must improve **≥5%** over
baseline. Same commit updates config + perf-test threshold.

### 4.1 Landed wins

| Model | Arch | Scope | Baseline | After | Δ | Configs touched | Commit |
|---|---|---|---:|---:|---:|---|---|
| sentence-BERT | WH | total matmul device-kernel time | 13.39M ns | 10.52M ns | **−21.4%** | ff2+qkv reshape (4,2)+rmo; ff1+self_out flag-only | `0a421bb6a56` + threshold `7a89d6a8f55` |
| sentence-BERT | WH | e2e model perf (50-iter wall clock) | 425 sps | 506 sps | **+19.1%** | (same as above) | (same) |
| Falcon7b | WH | prefill seq=1024 device-kernel sps (CI threshold) | 3120 | 3742 | **+19.9%** | mm_h_to_4h `(1,8)`, mm_4h_to_h `(1,6)` — both legacy-compatible h=1 | `0fbd77527c2` |
| Falcon7b | WH | prefill seq=1024 device-kernel sps (same-hw A/B) | 2938.7 (median of 3) | 3745.0 (median-aligned) | **+27.3%** | same edit; published +19.9% was conservative | — |

Calibration: CI hardware where Falcon7b's 3120 target was measured runs ~6% faster than this
n150 on identical code. Published deltas against CI thresholds under-represent on-this-hardware
deltas by roughly that gap.

### 4.2 Attempted and reverted

| Model | Arch | Attempt | Result | Reason |
|---|---|---|---:|---|
| sentence-BERT | BH | ff2+qkv (4,2)+rmo pattern | +1.56% | Below 5% bar; reverted |
| SDXL UNet 1024×1024 | BH | 11 legacy-compatible subblock upgrades | −0.26% (slower) | Below 5% bar; reverted |
| Falcon7b mm_4h_to_h | WH | (4,2)+rmo with `in0_block_w` dropped 8→4 | +0.2% (3745→3752) | K-iter doubling cancels DST gain; below 5% bar |
| DistilBERT wall-clock threshold tighten | WH | `expected_inference_time` 0.0338 → 0.0335 | Variance spike | One of 7 runs at 0.0380; reverted — not enough samples |

The **BH cross-arch finding**: two independent BH manual migrations (sentence-BERT, SDXL)
got ~0% delta from subblock reshape. The WH equivalents of these patterns landed −21.4% and
+19.9%. Plausible explanation: BH's matmul kernel time on these shapes is not dominated by
pack-phase overhead in the way WH's is (more cores per die, different NoC, potentially
different LLK pipeline behavior amortizing pack). **Action item before more BH-side manual
migrations**: empirically Tracy-decompose one BH model's time.

### 4.3 Auto-config models (no explicit program_config → auto-tuner handles them)

Per guidance, these still need pre/post auto-tuner profile A/B and threshold bumps. Status:

| Model / path | Measured on opt_3 (n150) | CI threshold | Notes |
|---|---:|---:|---|
| Falcon7b prefill seq=128 | 2095.3 sps | 2115 | Within ±2% of CI threshold; likely +4-7% after cross-hw correction; not verified |
| Falcon7b decode seq=128 | 636.3 sps | 647 | Same |
| Falcon7b decode seq=1024 | 574.4 sps | 572 | Same |
| Falcon7b decode seq=2047 | Tracy failure (pre-existing) | 548 | n/a |
| sentence-BERT device-perf | 546.5 sps | 546.5 | On threshold; auto-tuner + manual-config wins from upstream are stable |
| DistilBERT wall-clock seq=384 | 0.0326 s median (7 runs) | 0.0338 s (= actual × 1.05, so pre-auto-tuner actual ~0.0322 s) | **Flat, +1.2% within noise** — NOT a win |

**The DistilBERT lesson**: comparing against a threshold set as `actual × 1.05` (the
`dccb887fc5a` pattern) versus comparing against an actual measurement are different things. I
initially claimed DistilBERT was a −3.6% auto-tuner win; the real delta is +1.2% within noise.
Always pin baselines to actual measurements, not to thresholds.

### 4.4 Skipped / blocked

| Model | Category | Reason |
|---|---|---|
| owl-ViT | No upgrade available | All 4 configs have `per_core_M=1` → h=1 forced → already at max DST volume given shape |
| `demos/bert` | Maintainer-declared unsupported on WH | `@pytest.mark.skipif(is_wormhole_b0() or is_blackhole())`. 2 upgrade candidates would apply (qkv/query_by_key) but no validation path from n150 |
| BGE-large | Weights blocked | 1.3 GB HF download. 1 candidate (`self_out (1,4)→(2,4)`); estimated ~3% upside alone, below 5% bar |
| bert_tiny | Test skip + weights | `@pytest.mark.skip #26288` + `mrm8488/bert-tiny-finetuned-squadv2` not cached |
| DistilBERT device-perf (seq=768) | Test infra | Setup hangs in `GetNumPCIeDevices()` inside tracy subprocess. Pre-existing infra issue, `@pytest.mark.skip #26285` |
| Mamba | Weights blocked | 5 GB `state-spaces/mamba-2.8b` not cached |
| BGE-m3, Qwen3-embedding-8b | No device-perf test | Not in tree |
| ResNet50 | No upgrade room | Single mcast config at `per_core_M=per_core_N=1`; `in0_block_w=2` already at max (K shard = 2). No lever available |
| DeepSeek-V3, GPT-OSS, Llama3-70b-galaxy, T3000/TG demos | Multi-chip | Blocked on hardware access. Config counts: DeepSeek-V3 6+ files; GPT-OSS 22 configs across 3 files; Llama3-70b-galaxy 3+ files |
| ViT-WH, SDXL-WH, Segformer, MobileNetV2, SigLip, Qwen-VL, metal_BERT_large_11, gemma4 | Surveyed but deferred | See opt_3 doc for per-model notes |

### 4.5 Auto-config PCC sweep on BH (no regressions)

25+ PCC tests across SDXL attention / UNet / transformer block / feedforward / GEGLU /
crossattn / resnetblock / VAE: all pass. No additional auto-tuner bugs beyond the one fixed
in `63d9bc9e4da`. Breakdown:
- SDXL base attention: 4/4
- SDXL base UNet 1024×1024: 1/1
- SDXL base transformerblock / feedforward / GEGLU: 8/8
- SDXL base crossattn (up/down/mid) + resnetblock: 11/11
- SDXL VAE (encode/decode): 2/2 (512×512 deliberately skipped on BH)

---

## 5. Open items (all tracked, none blocking)

### 5.1 Auto-tuner / helpers follow-ons
1. **DRAM-sharded factory hardcode** at `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:151`:
   `max_subblock_w = fp32_dest_acc_en ? 4 : 8`. Thread full `DeviceComputeKernelConfig`
   through the function signature (~30 LOC across 2-3 callers) and route through
   `ttnn::get_dest_reg_count`. Not correctness-breaking today.
2. **Host-side `#define` duplication** in `compute_kernel_config.cpp`: `DATUMS_PER_ROW=16`
   and `DEST_REGISTER_FULL_SIZE=64*16` are copies of arch tensix_types.h constants. All
   current arches (WH-b0, BH, Quasar) agree, so numerically correct today. Direct `#include
   "tensix_types.h"` doesn't work from host targets. A HAL accessor would work (mirror
   `hal.get_arch_num_circular_buffers()`) but touches `tt_metal/llrt/hal.hpp` — hot surface,
   not changing pre-rebase. Holding pattern: keep the `#define`s with clearer inline comment
   + `static_assert` when a future arch needs divergence.
3. **Auto-tuner integration tests**. 21 unit tests cover the pure functions; nothing
   explicitly validates the integration emits the right config on canonical shapes. Add
   `test_matmul_auto_tuner_integration.py` that calls `ttnn.matmul(a, b)` on canonical shapes
   and asserts on synthesized configs.
4. **Auto-upgrade path for manual configs**. The auto-tuner only fires when no
   `program_config` is passed. Future path: augment user-passed configs (flip rmo, reshape
   subblock if L1 fits). Designed-but-unbuilt; ~1-2 days.
5. **DRAM-sharded `row_major_output` parity**. DRAM-sharded program-config types don't have
   `row_major_output` field yet — factory hardcodes `ROW_MAJOR_OUTPUT=1`. Adding the field +
   plumbing for parity with mcast is a follow-on.
6. **`get_per_core_factor` undersubscription**. Pre-existing quirk where L1-output shapes
   leave most cores idle. Not introduced by this branch. Would amplify auto-tuner wins on
   the L1-output codepath.
7. **L1 fit pre-check** — helper that estimates `in0_block_w × per_core_N × tile_size +
   per_core_M × per_core_N × tile_size × (2 if rmo else 1) + …` before touching the device,
   so we can bound which `(h, w)+rmo` combos are safe without compile-trial-and-error.

### 5.2 Auto-tuner bug found on BH and fixed (`63d9bc9e4da`)

For batched matmul with a height-sharded input A (multiple batch instances stacked along the
per-core height axis), the auto-tuner was being fed `per_core_M = shard_height_in_tiles` but
the downstream validator checks `out_subblock_h % M_per_batch_instance == 0`. When shard
height > per-batch M (e.g. 24 vs 12), the tuner picked subblocks like `(8, 1)` that passed
the per-core check but failed validation.

**Fix**: clamp tuner's `per_core_M` input at `min(per_core_M, M)` at both sharded-batched
call sites (`create_matmul_program_config` and `get_matmul_program_config` non-mcast else
branch). Found by BH sentence-BERT PCC test; fix makes it pass. Regression check: 557/557
matmul unit tests on BH p100a. **This was the real BH unblock** — the subsequent BH
sentence-BERT config edit added only +1.56% on top. The structural win came from the tuner
fix, not the manual-config recipe.

### 5.3 Build-system fix (`6bb8052df6e`)

`test_common_utils.cpp` uses `<gtest/gtest.h>` but the static lib target only linked
TTNN::CPP and relied on transitive PCH inheritance that held on WH but broke on BH on a
fresh tree. Cherry-picked from `opt_2`.

### 5.4 Per-model future opportunities (Falcon7b-specific)

1. `mm_4h_to_h (4,2)+rmo` with `in0_block_w=4`: empirically verified dead end (+0.2%, below
   5% bar). Staying with legacy `(1, 6) + in0_block_w=8` is correct for DENSE_4H_TO_H.
2. `FUSED_QKV_MM_OPTIMIZED_PROGCFG`: commented-out developer intent `(1, 7)` legacy-compatible
   volume 7/8. Only reachable from `seq_len == 2048` branch, currently skipped (`#40015`).
3. `QKT_OPTIMIZED_PROGCFG` / `QKTV_MM_OPTIMIZED_PROGCFG` lambdas: both have hardcoded
   `out_subblock_h=1, out_subblock_w=1` — caller subblock args ignored. For seq=1024
   (num_slices=4, tiles_per_shard=10): QKT `per_core_N=seq_len/32=32` allows `(1, 8)` volume
   8. Requires un-hardcoding the lambdas AND updating callers.
4. K-iteration tuning on mm_h_to_4h (`in0_block_w=3` over Kt=144 → 48 iters) and
   mm_4h_to_h (`in0_block_w=8` over Kt=576 → 72 iters). Would need L1 fit check.
5. LM head 2D mcast: subblock already (2, 4) volume 8/8 → flag-only no-op, skipped per recipe.
6. Decode-path configs: `MatmulMultiCoreReuseProgramConfig` lambda at `model_config.py:260`
   used in decode mode. Not surveyed.

### 5.5 Generic follow-ons
- Un-skip skipped perf tests (e.g. `test_perf_wh_bare_metal` prefill_seq2048 / `#40015`) to
  validate configs that only activate at those shapes.
- Auto-config models need pre/post auto-tuner profile A/B + threshold bumps. Methodology:
  checkout pre-`8e9f28d664b`, rebuild, run model's device-perf test, record; checkout
  `wransom/matmul_helpers_opt_2` (or later), rerun, compare. If faster, bump `expected_perf`.
  Bert_tiny / DistilBERT device-perf are `@pytest.mark.skip (#26288, #26285)` — those issues
  are the natural home for this workstream.

---

## 6. Rebase brief

### 6.1 This branch's code footprint (what conflicts are possible)

```
$ git diff --stat main...wransom/matmul_helpers_opt_3
```
approximates to:
- `tests/scripts/matmul_perf/` — new perf scripts (opt_2 Group G)
- `tests/ttnn/unit_tests/gtests/test_matmul_auto_tuner.cpp` — new file
- `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_tuner.{hpp,cpp}` — new files
- `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp` — auto-config
  routing changes
- `ttnn/cpp/ttnn/operations/matmul/device/factory/*` — Bug-3 fix (separate out/interm CBs)
  + `row_major_output` plumbing
- `ttnn/cpp/ttnn/operations/matmul/matmul.{hpp,cpp}` — `row_major_output` field on mcast
  program-config types
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
  and writer kernels — dual-mode pack/read
- `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}` + `bias_add_helpers.{hpp,inl}` —
  unified helper surface
- `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp` — no changes
  pending post-revert (HAL approach was tried and reverted)
- `models/demos/wormhole/sentence_bert/ttnn/common.py` — sentence-BERT config upgrade
- `models/demos/wormhole/sentence_bert/tests/perf/test_sentence_bert_perf.py` — threshold bump
- `models/demos/falcon7b_common/tt/model_config.py` — Falcon7b config upgrade
- `models/demos/falcon7b_common/tests/test_perf_falcon.py` +
  `models/demos/falcon7b_common/tests/test_falcon_device_perf.py` — threshold bumps
- `MATMUL_HELPERS_*.md` (docs) at branch root

### 6.2 Likely conflict surface with a reduce-helper merge into main

Most-likely conflicts (same files, overlapping regions):
- `ttnn/cpp/ttnn/kernel_lib/` — if reduce-helper also lives here and has adjacent changes
  in `matmul_block_helpers.{hpp,inl}` or adds new helper files
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/*.cpp` — if reduce-helper
  swapped in a reuseable reduce sequence used by matmul's fused bias path
- `ttnn/cpp/ttnn/operations/matmul/device/factory/*` — for any factory that also used
  the old reduce pattern

Unlikely to conflict but worth scanning:
- `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp` — only if
  reduce-helper added its own DST-capacity logic there
- `tt_metal/llrt/hal.hpp` + `tt_metal/hal.cpp` — only if reduce-helper extended HAL

Resolution guidance:
- Helper files: our changes are additive (new helpers like `bias_add_helpers`,
  transpose/reblock-untilize). Reduce-helper additions should merge side-by-side. Watch for
  `pack_tile` overload sets — row-major pack path overloads matter.
- Matmul compute kernels: our dual-mode pack/read pattern (`#if ROW_MAJOR_OUTPUT`) needs to
  survive any reduce-phase rewrite. If reduce is emitted before pack (typical: matmul-reduce
  fusion), the reduce stage must hand DST to the pack stage in whatever layout the existing
  pack path expects.
- `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`: our Bug-3 fix forces
  separate `out_cb`/`interm0_cb` regions when `row_major_output` is on. Reduce-helper likely
  doesn't touch this gate but if it adjusted CB allocation, reconcile carefully.

### 6.3 Post-rebase verification plan

In order, escalating cost:
1. **Build cleanly** on both WH and BH. `./build_metal.sh` to completion + `./build_metal.sh
   --build-ttnn-tests`. ccache should make this fast.
2. **Auto-tuner unit tests** — `unit_tests_ttnn --gtest_filter="MatmulAutoTuner*"`. 21 tests,
   sub-second runtime. Covers the pure-function math across all 4 DST capacity combos.
3. **Matmul unit tests** — `tests/ttnn/unit_tests/operations/matmul/test_matmul.py`. Historical
   count: 557 passing / 136 skipping on BH; similar on WH. Catches integration regressions.
4. **Sentence-BERT device-perf** — locks in the upstream reference win. Should measure
   ~546.5 sps on WH n150 (threshold set in `7a89d6a8f55`).
5. **Falcon7b device-perf** (prefill seq=1024) — locks in this branch's only landed per-model
   win. Should measure ~3745 sps on this n150 (threshold set to 3741 in `0fbd77527c2`).
6. **BH auto-config PCC sweep** (if BH access available) — 25+ tests, historically all pass.
   Confirms `63d9bc9e4da` divisibility fix still holds after the merge.

Expected-failing tests to ignore (pre-existing, unrelated to helpers):
- `models/demos/wormhole/distilbert/tests/test_perf_distilbert.py::test_distilbert_perf_device`
  — `@pytest.mark.skip (#26285)`, setup-hang in tracy subprocess `GetNumPCIeDevices()`
- `models/demos/wormhole/bert_tiny/tests/test_performance.py::test_perf_device_bare_metal`
  — `@pytest.mark.skip (#26288)`, weights missing for `mrm8488/bert-tiny-finetuned-squadv2`
- `models/demos/falcon7b_common/tests/test_perf_falcon.py prefill_seq2048` — intentional
  skip on `#40015`

### 6.4 Environment gotchas from past sessions

- **`ARCH_NAME` env var is a hint, not ground truth.** IRD containers often read
  `ARCH_NAME=blackhole` even on a WH n150 host. Always verify with
  `tt-smi -s | grep board_type` before doing arch-specific work. JIT cache
  `trisck.o.dephash` also reveals actual arch.
- **`/home/wransom` has a ~10 GB quota** that fills easily. HF cache was moved to
  `/localdev/wransom/huggingface_cache` with a symlink from `~/.cache/huggingface`.
- **DPRINT conflicts with Tracy.** Unset `TT_METAL_DPRINT_CORES` before running device-perf
  tests or tracy subprocess errors with exit code 4.
- **HF cache discovery for models this branch exercises:**
  - Falcon7b: `HUGGINGFACE_HUB_CACHE=/proj_sw/user_dev/macimovic/hf_data + HF_HUB_OFFLINE=1 +
    TRANSFORMERS_OFFLINE=1`.
  - Sentence-BERT: default cache at `/localdev/wransom/huggingface_cache/hub/` (has
    `models--emrecan--bert-base-turkish-cased-mean-nli-stsb-tr`); do NOT set
    `HUGGINGFACE_HUB_CACHE` override for sentence-BERT.
  - DistilBERT: `distilbert-base-uncased-distilled-squad` is in
    `/localdev/wransom/huggingface_cache/hub/`.
- **tt-smi -r between test crashes**, especially after tracy subprocess hangs.
- **Never run multiple device-using tests concurrently**. Only one pytest per device.

---

## 7. Resume protocol

When resuming this work in a new session:

1. **`git status`** to confirm branch is `wransom/matmul_helpers_opt_3` and working tree
   matches expectation. Commits on this branch may be local-only, not pushed — check
   `git log --oneline @{upstream}..HEAD`.
2. **Read this file first**. It supersedes `MATMUL_HELPERS_OPT2_REVIEW.md` and
   `MATMUL_HELPERS_OPT3_RESULTS.md` (those remain on disk for deep-dive reference but are
   cross-referenced from here; if they conflict with this file, this file wins).
3. **On new hardware (BH, T3000, etc.)**, survey `models/demos/blackhole/` or equivalent +
   any `*BH.py` / `*T3000*.py` variants for explicit mcast configs. Verify arch first
   (`tt-smi -s | grep board_type`, `ARCH_NAME` env var is not trustworthy).
4. **For per-model migration**, the recipe is in section 4 intro + the existing Falcon7b /
   sentence-BERT commits as templates. `0fbd77527c2` is the canonical single-model commit
   shape (config edit + threshold bump in one commit).
5. **For correctness changes** (auto-tuner bug fixes, helper fixes), the auto-tuner gtests
   + matmul unit tests + BH sentence-BERT PCC test are the fastest feedback loops in that
   order.
6. **If baselining against CI thresholds**, remember the DistilBERT lesson: many thresholds
   are set as `actual × 1.05` (lower-bound bumps per `dccb887fc5a`-style commits) or are on
   different hardware than yours. A same-hardware A/B against reverted-code is the clean way
   to measure an actual delta. CI-target comparisons are OK for "no regression" sanity but
   don't claim wins based on them.
7. **Safety**: never push to GitHub without explicit user confirmation. Never use sudo
   without confirmation. These commits are local-only unless you're told otherwise.

---

## 8. Pointers

- `tests/ttnn/unit_tests/gtests/test_matmul_auto_tuner.cpp` — 21 auto-tuner gtests. Read
  these if you want to understand the tuner's contract quickly.
- `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_tuner.{hpp,cpp}` — the tuner
  module itself; well-commented.
- `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}` + `bias_add_helpers.{hpp,inl}` —
  the unified helpers.
- `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` — kernel-side DST capacity query.
- Commits of interest (`git show <sha>` for detail):
  - `b3c0b84c0ed` — Bug 3 fix (separate out_cb/interm0_cb L1 regions; gates everything)
  - `8e9f28d664b` — Auto-tuner module add
  - `3c13eb38b85` — Auto-tuner routing + row_major_output on for auto-config
  - `4a2c288dbb1` — Enable row_major_output on mcast factories (Path A)
  - `0a421bb6a56` + `7a89d6a8f55` — Sentence-BERT reference migration (config + threshold)
  - `0fbd77527c2` — Falcon7b migration (single-commit template: config + thresholds)
  - `63d9bc9e4da` — Auto-tuner divisibility fix (the real BH unblock)
  - `6bb8052df6e` — Fresh-tree gtest build fix (BH-only)
- Pre-consolidation docs (now deleted, preserved in git history): `MATMUL_HELPERS_OPT2_REVIEW.md`
  was removed at this consolidation; view via
  `git show 36e5e979d03:MATMUL_HELPERS_OPT2_REVIEW.md` if a deeper migration recipe is needed.
  `MATMUL_HELPERS_OPT3_RESULTS.md` similarly — `git show
  36e5e979d03:MATMUL_HELPERS_OPT3_RESULTS.md`.
