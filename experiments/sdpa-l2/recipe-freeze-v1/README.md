# SDPA recipe and evidence freeze — 2026-09-21

This is the immutable research handoff for production PR 1. The Git tag
`sdpa-recipes-20260921-v1` identifies the complete snapshot; `manifest.json`
records the selected files' SHA256 hashes and the original research/main bases.
Do not amend this snapshot after production extraction. Corrections require a
new version with a documented reason. Nothing is being promoted automatically.

## Selected recipes

| ID | QK / PV | DST and recurrence | External preparation |
| --- | --- | --- | --- |
| D | HiFi4 / HiFi4 | FP32 DST, numerator and denominator; full-FP32 subtraction, accurate exp path | None; ordinary BF16 QKV |
| C | HiFi4 / HiFi2 | FP32 DST, numerator and denominator; retained matched cubic-exp/subtraction path | None; ordinary BF16 QKV |
| B | HiFi2 / HiFi2 | BF16 DST; compensated high/low BF16 recurrent state | None; ordinary BF16 QKV |
| A | HiFi2 / HiFi2 | Original BF16 streaming baseline, uncompensated state | None; ordinary BF16 QKV |
| E_bf16 | LoFi / LoFi | BF16 DST; compensated state, retained E exp/normalization | Q RNE7, KV RNE5 stored BF16 |
| E_bfp8 | LoFi / LoFi | Same as E_bf16 | Q RNE7 BF16, KV RNE5 then native BFP8 packing |
| E_bfp4 | LoFi / LoFi | Same as E_bf16 | Q RNE7 BF16, KV final shared-grid RNE with saturation to BFP4 |

Bit counts include the leading significand bit. Previous E is E_bfp8;
previous G is E_bfp4. F is not selected. Plain/no-custom-rounding E runs are
diagnostic controls, not extra recommended recipes. No claim says rounding
improves every stress case. No recipe has a universal absolute error bound.

These descriptions are summaries, not substitutes for the pinned definitions:

- `flux2-frontier-v1/device_attention.py`: canonical base defines, preparation,
  formats, and generic-op construction; `prepare` is identity for A/B/C/D.
- `compute-sprint-v3/pareto/collect.py`: exact final D/C/B/A/E/G selection.
- D/C: `compute-sprint-v3/fp32/compute_early.cpp` and `early_guard.hpp`, with
  their recipe-specific flags. Conditional FP32 L1 state fusion is retained.
- B: `compute-sprint-v3/review/compute_valid.cpp` and its included headers.
- E: `compute-sprint-v3/compensated/group2_valid/compute_streaming.hpp`.
  B/E group two numerator contributions before compensation; denominator
  compensation and changed-max/final-flush handling remain as qualified.
- E storage variants: `compute-sprint-v3/kv-precision-v1/bench.py::build` and
  `prepare`. Its CB layout, formats and Q/K/V preparation are authoritative.
- A: the unchanged selection in the matched collector, not whatever future
  upstream main happens to implement. Existing model configurations are not
  automatically numerically identical to this frozen A.

All paths above are relative to `experiments/sdpa-l2` in the snapshot.

## Qualification boundaries

Latest optimized specializations: Blackhole, noncausal/unmasked, Q chunk256,
K chunk512, D128, BF16 output. They are not blanket implementations for ring,
causal, masks, tails, MLA, arbitrary shapes or other architectures. C/D and
B/E have different existing CB layouts/input depths; preserve each until
separately requalified. The fixed state-bank capacities are correctness
invariants for the FP32 fusion, not tuning suggestions.

The v3 reassociation gate was per-case candidate L2 <= 1.05 × baseline L2
+ 0.0001 percentage points, with separate zero-reference absolute checks.
This is a relative-regression gate, not permission to erase existing absolute
accuracy goals or known stress failures. A refactor must rerun numerical,
trace, cache-hit, input-immutability and lifetime tests. Bitwise equivalence
is required where promised, not assumed across accepted reassociation changes.

## Decisive evidence

- `compute-sprint-v3/pareto/matched-v1.json`: matched D/C/B/A/old-E/old-G suite.
- `compute-sprint-v3/kv-precision-v1/{accuracy,perf,smoke}-v1.json`: current E
  family. 63 accuracy records; old E/G outputs reproduced exactly.
- `compute-sprint-v3/no-rounding-v1/`: additional plain-input diagnostics,
  raw metrics, scripts and final PNG; no new recommended recipe.
- `compute-sprint-v3/fp32/integrity-early-*.json`, `review/B-valid-*.json`, and
  `compensated/{e,g}-valid-*.json`: optimization qualification and actual
  distinct-input timings. Their reports distinguish decisive and rejected runs.
- `compute-sprint-v3/unchanged-A-resident-v2.json`: unchanged A performance.
- Final plots and reports remain beside their raw records; the manifest pins
  every included source/result artifact instead of relying on report numbers.
- FLUX `replacement-suite-01`, Wan `suite-01`, and exact-shape tuning evidence
  are historical model integration evidence. They predate final v3 compute
  changes and do not qualify the freshly extracted production implementation.

Resident TFLOP/s/core excludes preprocessing and recurring input DM and favors
unchanged maxima. It is not chip TFLOPs or model throughput. Common-mode and
near-zero-reference caveats must remain visible in PR comparisons. Qualification
of one original-input quantization route does not qualify all producer routes.

## Snapshot contents and exclusions

The snapshot retains the entire committed research parent, all six existing
tracked modifications, and selected previously untracked experiment sources,
reports, JSON/CSV evidence and plots. Final FLUX images and Wan videos/previews
are included. The manifest documents which files were added; it does not claim
to preserve every file under the research directory.

Untracked compiler outputs, Python caches, AppleDouble files, logs, raw `.pt`
dumps and `.tracy` captures are excluded. They remain untouched in the original
workspace. Reports/JSON retain their diagnostics and hashes where present.
Earlier already-committed files remain available through the parent history.
No HF weights, tokens, virtual environments or JIT caches belong in the snapshot.

`freeze.py create` uses an isolated temporary Git index, then creates a new
research snapshot branch and annotated tag without changing the original
branch/index or staging its working changes. `freeze.py verify --ref TAG`
checks the selected Git blobs against the manifest, including untracked-at-
capture sources and media. Creation refuses to reuse existing refs/manifests.

## Production handoff

PR 1 starts on fresh main, not on this branch. Extract production primitives
and policies; do not cherry-pick research harnesses, experimental flags or model
instrumentation wholesale. Preserve old config/default semantics via an
additive resolver. Explicit recipes must reject unsupported shapes/conflicting
numeric settings instead of silently changing recipe or falling back.

PR 2 expands feature/platform coverage and migrates callers. PR 3 deletes
unreachable non-streaming loops while preserving shared decode/sparse/CCL
utilities. The usage audit is a dated research-tree inventory, not proof that
fresh main has the same call graph; refresh affected areas during extraction.
