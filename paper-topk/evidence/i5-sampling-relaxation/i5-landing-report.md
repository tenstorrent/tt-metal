# I5 Landing Report — Sampling call-site relaxation (routed ttnn.topk)

Date: 2026-08-17 (night shift follow-through) · Branch: `nkapre/sorting` @ working tree on `64dbfdeb1d4` · HW: Blackhole p150a, exclusive
Tree state: **UNCOMMITTED** — 3 modified files, ready for owner review:
`models/common/sampling/_utils.py`, `models/common/sampling/tt_sampling.py`, `models/common/modules/sampling/sampling_1d.py`

## Verdict

**Landed in the working tree and verified — with one prepared-patch defect found by the bit-exactness
study and corrected before any suite ran.** Values are bit-exact everywhere; every index difference is a
proven bf16 tie; the tp8 control is bit-identical; the greedy pick under the chain's tiebreak rule is
identical in 100% of rows; perf re-measured at 44.2×/41.5× with the pow2 control at 1.00×.

## 1. What was applied

1. `0001-utils-topk-route-predicate.patch` — applied verbatim (`topk_would_route_to_large_indices`
   mirror of `should_route_to_topk_large_indices`, topk.cpp:258-320).
2. `0002-tt-sampling-relax-routed-topk-callsites.patch` — applied, then **amended** (see §2):
   the multi-step half-1 "+W/2 offset restore" block and the `tt_topk_half_offset` tensor were
   **removed**; the dtype-normalization typecast was kept (now unconditional for both halves).
   Amended diff saved as `night/i5-sampling-relaxation/0002-amended-tt-sampling-relax-routed-topk-callsites.patch`.
3. `0003-sampling-1d-relax-routed-topk-callsites.patch` — applied verbatim.

`git apply --check` clean; all three files pass `py_compile`.

## 2. Patch defect found and corrected (the study earned its keep)

**The prepared 0002's +W/2 offset restore was wrong.** Run 1 of the bit-exactness study flagged
20,479 non-tie index diffs on the tt_sampling multi-step half-1 path — every "new" index exactly
`old + 64128`. Root cause, confirmed in source:

- **The stock single-core topk factory never reads the supplied `indices_tensor`.**
  `GENERATE_INDICES` is pinned to `"1"` (GH issue **#36329**) in
  `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp:185-195`
  ("The parameter, its binding and its run arg are provisioned for the fix rather than consumed").
  Every shape this relaxation fires on is a single-core-today shape, so **today's production calls
  already receive 0-based per-half positions**, not the iota values the prepared patch assumed.
- Globalization is owned **downstream**: `tt_indices_device_offsets` (tt_sampling.py:1013, +0 for
  half-0 slots, +`padded_per_device` for half-1 slots) / `index_offsets` in sampling_1d. The routed
  composite also returns 0-based positions, so the correct relaxed call needs **no offset at all**;
  the prepared +W/2 add would have **double-counted** with the downstream offsets add.
- Related README §2.1 correction: at padded 65536 the stock op **also** emits uint32
  (`compute_output_specs`, topk_device_operation.cpp:294-302 — output dtype comes from the op
  contract, never from the supplied indices tensor), so "stock uint16 vs route uint32" was wrong;
  the kept typecast is a harmless, value-exact normalization to the module's expected dtype, not a
  behavior restore.
- sampling_1d (0003) needed no change — its `cat([r, r])` local-indices design already matches
  device reality, which is why its study variant was perfect on the first run.
- The prepared patch's int32 tensor-add pattern itself (README test-plan item 6) **did execute
  correctly on device** in run 1 (results shifted by exactly the programmed +64128), so the op
  support question is settled even though the block is now deleted.

After the amendment, the re-run study is clean everywhere (§3).

## 3. Bit-exactness study (the headline deliverable)

Standalone script `night/i5-sampling-relaxation/i5_bitexact_study.py`; results JSON
`i5_bitexact_results.json`; log `study_run2.log`. N=20 trials/shape, rows=32, k=32, identical
seeded inputs per trial to both call forms; all comparisons in the int-bits domain
(bf16 values as uint16 bit patterns; indices as exact integers). Production-faithful inputs:
padding lanes carry `torch.finfo(bf16).min` (the mask value vocab_padding actually lands, finite),
valid widths 37984 (qwen36) / 64128×2 (split) / 19008 (tp8).

OLD = production call form (`sub_core_grids=None`, `indices_tensor=` production iota/dtype,
`stable=True` for tt_sampling, no stable for sampling_1d). NEW = exactly what the patched code
emits (bare `ttnn.topk(x, k, dim=-1)` + dtype normalization; tp8 = identical old call, gate False —
asserted `topk_would_route_to_large_indices` gate value per trial on the real device tensors).

| shape (rows×trials = rows) | value-multiset exact | value-seq exact | index-seq exact | index-set exact | index diffs (positions) | diffs proven ties* | non-tie diffs | pad-lane winners | post-tiebreak pick identical |
|---|---|---|---|---|---|---|---|---|---|
| A qwen36 W=65536 (uint32→uint16 cast path) | **100%** (640/640) | **100%** | 0.3% | 70.5% | 5,389 | **5,389 = 100%** | **0** | 0 | **100%** (640/640) |
| B1 split W=64128 ×2 halves, sampling_1d form | **100%** (1280/1280) | **100%** | 0.5% | 67.0% | 11,449 | **11,449 = 100%** | **0** | 0 | **100%** (640/640, gathered) |
| B2 split W=64128 ×2 halves, tt_sampling form (stable=True; downstream-offsets emulated) | **100%** (1280/1280) | **100%** | 0.2% | 69.0% | 10,918 | **10,918 = 100%** | **0** | 0 | **100%** (640/640, gathered) |
| C tp8 W=32768 control (gate must be inert) | **100%** | **100%** | **100%** | **100%** | 0 | — | 0 | 0 | **100%** — bit-identical in every field |

\* tie proof per differing position: `input[old_idx] == input[new_idx]` **bitwise** in the bf16 input row.

The honest tie story:
- **Top-k value multiset and value sequence are bit-exact always** — both engines return the exact
  sorted top-32 of the same canonicalized input. This is the field the sampled-token distribution
  depends on, and it never moved by a bit.
- **Index sequences differ in most rows** (index-seq exact only ~0.2–0.5%) — expected and honest:
  a randn bf16 row at 64–65K samples has many exact duplicates (bf16 has ~256 distinct values per
  binade), and the two engines order tied duplicates differently. Every single one of the
  27,756 differing positions across A/B1/B2 was individually proven a bitwise tie. Zero non-tie
  diffs. Zero winners in padding lanes (R4 sentinel concern never materialized).
- **Post-tiebreak greedy pick** (the chain's `_adjust_values_for_tiebreak` documented guarantee:
  boost the lowest-global-index holder of the row max): identical old-vs-new in **all 2,560
  gathered rows** across A/B1/B2. Both engines return *all* tied maxima when ≤32, so the tiebreak
  pass sees the same candidate set either way.
- **tp8 control**: gate returned False on every trial; call form identical; output bit-identical in
  values AND indices in all 640 rows — the relaxation is provably inert off its fire set.

### End-to-end (real chain, real device)

Real `Sampling1D` (vocab 128256, 1×1 mesh → split path, routed branch fires), old = gate
monkeypatched off (today's call form), new = patched module, same logits + same seed tensor
per step, 20 decode steps per mode:

- **greedy k=1**: 639/640 tokens identical; the 1 diff is a proven bf16 tied-max row
  (both tokens hold the bitwise row max). Note: sampling_1d has **no** `_adjust_values_for_tiebreak`
  pass (that lives in tt_sampling only), so its greedy pick among exact tied maxima is
  order-dependent **today too** — the relaxation changes which arbitrary tied token wins, not
  whether ties are stable (they already aren't).
- **random k=10 p=0.9, fixed seeds**: 565/640 tokens identical; all 75 diffs bit-equal-logits
  ties (identical value multiset ⇒ identical sampled distribution and sampled *position*; only the
  token id label at tied values differs). Matches README's declared residual exactly.
- **Call-surface proof** (recorded `ttnn.topk` kwargs): patched module at v128256 emits
  `['dim','k']` (relaxed) on both halves; gate-off and v32768 pow2 control emit
  `['dim','indices_tensor','k','sub_core_grids']` (today's form, bit-for-bit).
- tt_sampling's e2e greedy guarantee is covered by the study's post-tiebreak-pick emulation (100%)
  plus the dedicated `test_tiebreak_input_adjust.py` suite (22 passed, below). A full TTSampling
  multi-device forward needs >1 chip and could not run on this box.

## 4. Route-fires proof (predicted engine per shape)

- Helper gate asserted on real device tensors every trial: True at W=65536 and W=64128,
  False at W=32768. Zero mispredictions in 160 gate evaluations.
- Engine identification by measurement: qwen36 old form median 11.5 ms vs new 0.28 ms host-coarse
  (study), and canonically 9,590 µs (1 core) vs 216.9 µs (130 cores) in the scenario cells —
  routed composite engaged. tp8 control: 171.35 vs 171.19 µs on 65 cores both sides — same stock
  bitonic both forms.
- **Refusal control**: `Sampling1D` pinned to a `sub_core_grid_topk` at the same routing shape
  keeps `indices_tensor`+`sub_core_grids` in every topk call (`i5_subgrid_control.py` — PASSED).

## 5. Test suites

All via `scripts/run_safe_pytest.sh --run-all` (flock-serialized), logs in
`night/i5-sampling-relaxation/suites/`:

| suite | result |
|---|---|
| models/common/tests/test_sampling.py | 11 passed, 7 skipped, **1 failed — PREEXISTING** (see below) |
| models/common/tests/modules/sampling/test_sampling_1d.py | 4 passed, 136 skipped, 50 deselected — device tests **blanket-skipped on BH** by `models/common/tests/conftest.py:130` (`ttnn_mesh_device` fixture: "Blackhole device is not supported for this test yet") — pre-existing gate, not overridden; the standalone study/e2e/trace runs above are the BH device evidence for exactly these paths |
| models/common/tests/test_sampling_vocab_padding.py | 17 passed |
| tests/ttnn/unit_tests/operations/reduce/test_tiebreak_input_adjust.py | 22 passed (the §2.3 greedy-tiebreak guard) |
| tests/ttnn/unit_tests/operations/reduce/test_sampling.py | 26 passed |
| **charter guards** | |
| tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py | 62 passed, 1 skipped |
| tests/ttnn/nightly/.../test_topk_large_indices.py (`-m "not requires_host_iommu"`) | 154 passed, 2 deselected |
| tests/ttnn/unit_tests/operations/reduce/test_topk.py | 220 passed, 8 skipped, 80 xfailed |

**The one failure is preexisting and unrelated**: `test_log_probs_calculation[blackhole-fabric_linear-shape0]`
(LogProbsCalculator at [1,1,32,151936], `AttributeError: 'NoneType' object has no attribute 'dtype'`) —
a T3K-shaped logprobs test running on 1 device where `LogProbsCalculator._is_supported` requires
8/32 devices and returns None. **Reproduced identically with the three patches stashed**
(`suites/baseline_logprobs.log`, baseline exit=1). Not touched by this change (no topk call in it).

## 6. Trace smoke

`i5_trace_smoke.py` (mirrors the BH-skipped `test_sampling1d_trace_capture[1x1-topk]`): Sampling1D
v128256 on 1×1 with 32 MB trace region — warmup outside capture, `decode_forward` captured with the
**relaxed routed branch confirmed inside the capture** (recorded kwargs `['dim','k']` ×2), no
in-capture cache miss, `execute_trace` replayed twice, both replays token-identical to eager.
PASSED (`suites/trace_smoke.log`).

## 7. Perf confirmation (canonical scenario cells, this tree, this session)

`_canonical_topk_sweep.py --model-scenarios --scenarios sampling_qwen36_tp4,sampling_1chip_split,sampling_tp8_pow2
--out generated/canonical_sweep/i5_landing` — all cells MEASURED (device-kernel-duration
methodology, same harness as the ledger's scenarios1; deltas are order-of-magnitude so 5-iter
cells suffice per profiling guidance):

| scenario (rows=32, k=32) | today/stocknow | relaxed (routed) | headroom | cores |
|---|---:|---:|---:|---|
| sampling_qwen36_tp4 (W=65536) | 9,590.2 µs | **216.9 µs** | **44.2×** | 1 → 130 |
| sampling_1chip_split (W=64128, per half) | 8,923.9 µs | **214.9 µs** | **41.5×** (×2 calls/token) | 1 → 130 |
| sampling_tp8_pow2 (W=32768, control) | 171.35 µs | 171.19 µs | **1.00×** — gate inert, parity row | 65 → 65 |

Matches the README's motivating numbers (9,596.3/217.0, 8,923.5/215.2, 171.3/171.3) within noise.
This is per decode step, every step, on every affected BH text demo. Full table:
`generated/canonical_sweep/i5_landing/scenarios_table.{csv,md}`.

## 8. Residuals and notes for the owner

- **R3 (tie order among exactly-tied bf16 values) is the only behavioral delta**, quantified above:
  k>1 random users may draw a different *token id* among bit-equal candidates (75/640 steps-tokens
  at randn-worst-case duplicate density; identical value distribution); sampling_1d greedy among
  tied maxima is order-dependent (1/640) — but it already is today (no tiebreak pass there, and
  today's single-core order is itself arbitrary). tt_sampling greedy is fully guarded (100% pick
  identity + 22-test guard suite).
- **README corrections** (for the PR description): (1) stock also emits uint32 at padded 65536 —
  the dtype claim in §2.1 was inverted for the stock side; (2) the +W/2 offset-restore rationale
  in §2.1/R2 was based on the stock op honoring `indices_tensor` values — it does not (GH #36329);
  the landed tree carries no offset ops and no `tt_topk_half_offset` tensor (R9 moot).
- The `indices_tensor` arguments retained on the non-relaxed paths remain dead weight on
  single-core shapes (op ignores them, #36329) — out of scope here, but worth a follow-up note.
- tt_sampling's multi-step split also has a pre-existing latent width quirk (indices tensor is
  `padded_vocab//2` wide while its topk halves would need `x_width`-wide indices; harmless today
  solely because of #36329; sampling_1d fixed this — its conftest comment calls it the "TTTv1
  2x width mismatch"). Unchanged by this landing.
- Artifacts: study script + JSON + logs, trace smoke, sub-grid control, amended 0002 patch — all
  under `night/i5-sampling-relaxation/`; scenario cells under `generated/canonical_sweep/i5_landing/`.
