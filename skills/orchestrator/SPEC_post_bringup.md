# SPEC: Post-Bringup Orchestrator Extension

**Status:** Design approved, not yet implemented.
**Date:** 2026-05-28
**Companion to:** `skills/orchestrator/SPEC.md` (the original bringup spec)
**Owner:** ssinghal@tenstorrent.com

## Purpose

Extend the bringup orchestrator beyond the original five phases (architecture → reference → ttnn → debug → optimization) so it also drives the integration work that turns "all blocks at-ceiling on device" into "all use cases work end-to-end at HF parity with optimized perf." That covers: loading real HF weights, running at production layer counts, building the autoregressive loop, wiring per-use-case demos, and pipeline-level perf optimization (paged_update_cache + reusable metal trace + targeted tracy work).

Stays model-agnostic. The new phases discover use cases by inspecting the HF model's class hierarchy at run time; nothing in the orchestrator or skills hardcodes a specific model's use case names.

## Goals

- `/bringup <hf_id>` runs all the way to "all use cases pass HF parity gates, perf characterized" without manual phase transitions.
- New work captured as reusable skill files so the patterns survive past the orchestrator.
- The state schema gains a second axis (`use_cases[]`) so per-use-case work has a place to live, while existing per-component phases are untouched.

## Non-goals

- Multi-batch / continuous batching. Single-batch v1.
- Multi-host orchestration. Single device target.
- Beam search. Greedy + top-k/top-p only.
- Auto-tuning over a search space of compute kernel configs. The optimization skill applies known-good presets and targeted improvements, not a search.
- Serving harness (FastAPI/gRPC). Demos remain CLI-only.

## Architecture

Two axes of state:

- `components[]` — existing per-block DAG. Gains one new phase: `real_weights`.
- `use_cases[]` — NEW per-use-case axis. Has two phases: `generation` and `perf`.

Pipeline order (same scheduler shape as today; just more phases):

```
architecture (per-model; populates BOTH components[] and use_cases[])
  → reference / ttnn / debug / optimization (per-component, existing)
  → real_weights (per-component, NEW)
  → generation (per-use-case, NEW)
  → perf (per-use-case, NEW)
  → done
```

Device-locked discipline preserved end-to-end: real_weights, generation, perf each dispatch one worker per tick.

### Why `use_cases` is a new axis

`generation` and `perf` operate on integrated pipelines, not single blocks. The AR loop infrastructure is built once but each use case has its own demo + perf characterization. Stuffing this into `components[]` would conflate "the model's blocks" with "the pipelines you can run." Keeping them separate makes the DAG cleaner and the resume / inspect semantics obvious.

## Model-agnostic primitives

The orchestrator and the new skills know about a fixed set of generic concepts. Models describe themselves to the orchestrator through these.

**I/O modalities:** `text`, `audio`, `image`, `video`, `none`. The generation skill has wired helpers per modality (HF tokenizer for text, log-mel + scipy.wavfile for audio, etc.).

**Validation metrics:** `bleu`, `wer`, `ecapa_cos`, `perplexity`, `accuracy`, `mse`, `pcc`. The generation skill has computation helpers under `demo/validate.py` per metric (`sacrebleu`, `jiwer`, etc.).

**Boolean flags:**
- `needs_ar` — true if the use case has `.generate()` (or inherits from `GenerationMixin`). If false, the generation phase becomes a one-shot forward, no KV cache.
- `needs_audio_out` — true if `output_modality == "audio"`. Drives whether the generation phase wires a vocoder.

**Validation threshold expression:** parity-relative (`"HF - 1.0"`, `"HF + 0.05"`) or absolute (`"≥ 0.95"`). Parsed and enforced by the generation worker.

If a future model needs a metric or modality outside this set, that's a small additive change in two places (`demo/validate.py` + the architecture skill's known-metric list).

## New skills

Four skills. Three new + one refresh.

### `skills/integration/SKILL.md` — NEW

**Purpose:** Take a HF safetensors checkpoint and load it into the TTNN modules built during bringup. Re-validate that each block still hits PCC > 0.99 against the HF PyTorch module at production layer counts.

**Patterns documented:**
- Safetensors index inspection: read `model.safetensors.index.json` to find top-level prefixes.
- Per-block weight-loader functions: `<sub>_<kind>_weights(hf_sd, layer_idx) -> nested PyTorch dict`. The nested shape matches what the TTNN module's `__init__` expects (re-read the module's `__init__` for each kind to derive the shape).
- Weight tying: shared embedding ↔ LM head ↔ output projection — one tensor referenced in multiple places.
- Missing-from-checkpoint buffers: sinusoidal positional embeddings, distance embeddings, etc. — rebuilt deterministically per HF's source.
- Realistic-input trick: random N(0,1) inputs saturate bf16 ~600×; use embed-derived inputs for attention/encoder/decoder leaves (run the previous embedding + LN + position-add chain on host to get realistic dynamic range).
- Function-scoped device fixture in pytest to avoid bank_manager OOM as weights accumulate across tests.
- Two-stage validation: reduced-config (2-layer) for quick iteration, full-config (production layer counts) for the final gate. PCC drift expected with depth — gate at PCC > 0.99 at full config, but expect 0.9999 leaves → ~0.99 at 24-layer depth.

**Worker contract (`real-weights-worker.md`):** for one component, write/extend `tt/weight_loader.py`, run the parameterized PCC test (`tests/test_real_hf_weights.py`), report PCC + status.

### `skills/generation/SKILL.md` — NEW

**Purpose:** Wire a working end-to-end pipeline for one use case: encoder forward → (optional AR decode with KV cache + sampling + EOS) → (optional audio post-processing) → output. Plus a demo CLI and an e2e validation test.

**Patterns documented:**
- KV cache shapes: self-attn `[B, num_heads, max_seq, head_dim]` per layer (updated in place each step); cross-attn `[B, num_heads, enc_seq, head_dim]` per layer (populated once after encoder).
- `decode_step` contract: persistent input/position/mask tensors as inputs, logits as output. Single-token Q, full-cache K/V via SDPA.
- Prefill order: encoder forward → populate cross-attn cache → run decoder warmup at the model-specific prefix tokens (HF convention: `[decoder_start_token_id, lang_id]` for NLLB-style decoders; varies per model — read HF's `.generate()` to find the prefix).
- AR loop: greedy (argmax) v1; top-k/top-p via `models/common/sampling/tt_sampling.py`; logits processors via `models/common/generation_utils.py`.
- EOS detection: stop when sampled token == `eos_token_id` from HF config.
- Demo CLI shape: typer-based, modality-appropriate args:
  - text in → `--src "<text>" --src-lang <code> --tgt-lang <code>`
  - audio in → `--wav <path>` (+ langs)
  - audio out → `--out <path>`
- Always runs HF reference alongside TTNN for side-by-side output (mandatory user-trust check).
- Validation by output modality:
  - text out → `bleu` (`sacrebleu`) or `wer` (`jiwer`)
  - audio out → `ecapa_cos` (primary) with re-ASR `char_similarity` as fallback if ECAPA scorer not available
  - encoder-only embeddings → `pcc` against HF on a representative input
- Hybrid host/device boundaries: parts that legitimately stay on HF host (tokenizer-bound char prep for TTS, etc.) are documented in the use case's `hybrid_notes` field. Not all blocks need to be on device; the orchestrator's "no shortcuts" guard for ttnn-phase ≠ "no host ops anywhere downstream of bringup."

**Worker contract (`generation-worker.md`):** for one use case, write `tt/<use_case>_model.py` + `demo/demo_<use_case>.py` + `tests/test_e2e_<use_case>.py`. Run the e2e test and assert the validation metric passes against HF parity. The first use case that needs AR pays the cost of building `kv_cache.py` + `text_generator.py` (or equivalent); subsequent use cases reuse those.

### `skills/perf/SKILL.md` — NEW

**Purpose:** Pipeline-level perf for one use case. Two sub-passes: structural (trace) then targeted (tracy + sharding/fusion within the hot path).

**Why this is distinct from `skills/optimization/`:** the existing `optimization` skill is per-block. It can tune one matmul's kernel config or shard one ttnn op. The `perf` skill operates on the integrated pipeline — touching `kv_cache.py` + `text_generator.py` + the cached-attention path in the attention block simultaneously. These are cross-block refactors with cross-block correctness invariants.

**Patterns documented:**
- The trace pitfall: a metal trace captured with `ttnn.update_cache(update_idx=int_pos)` bakes the int into the kernel sequence — single-trace replay across positions doesn't work.
- The fix: `ttnn.experimental.paged_update_cache(update_idxs_tensor=cur_pos_tt)`. Position lives in a device tensor; one trace replays for all positions and across `generate()` calls.
- Persistent buffers for all per-step inputs: input_ids, position_ids, self-mask. Hot loop reduces to `copy_host_to_device_tensor → execute_trace → host argmax`.
- KV cache reset between `generate()` calls: `copy_host_to_device_tensor` to zero in place (preserves buffer addresses for the trace; do NOT free and reallocate).
- Trace capture happens ONCE at end of first generate() warmup; reused thereafter.
- Tracy harness pattern (from existing `optimization` skill): 1 warmup + N timed/profiled invocations; output CSV bucketed by op-code and `memory_config.memory_layout`.
- Host-vs-device bound triage: if `total_ms < kernel_time_sum + small_host_margin` → host-dispatch limited → trace fixes it. If `total_ms ≈ kernel_time_sum` → device-compute limited → attack sharding / fusion / lower precision.
- Real diminishing-returns lesson from the SeamlessM4T-v2 bringup: trace+reusable trace delivered 1.21× because the floor is device kernel time, not host. The "50% from trace" hypothesis was wrong for that model. Further wins need compute-side attacks (sharding the big Q/K/V/O matmuls, fusing LN+Linear chains, lowering precision on non-PCC-sensitive ops).

**Worker contract (`perf-worker.md`):** ONE phase (`perf`) with TWO sub-passes INSIDE the worker. Both pre-passes share state and run as a single tick (one device dispatch from the orchestrator's perspective; the worker manages the sub-pass sequencing internally).

1. **Trace sub-pass**: migrate the use case's AR path to `paged_update_cache`. Capture reusable trace. Validate HF parity preserved + measure replay speedup. Skipped if `needs_ar=false`.
2. **Tracy sub-pass**: profile integrated pipeline. Apply ONE targeted optimization based on tracy findings (sharding the largest matmul, fusing a hot LN+Linear chain, lowering precision on a non-PCC-sensitive matmul). Validate PCC + measure delta.

Reports baseline + after numbers in `models/demos/<slug>/PERF_NOTES.md`.

### `skills/optimization/SKILL.md` — REFRESH

**Purpose:** Per-block tuning. Compute kernel config (HiFi4 + fp32_dest_acc), memory layout (DRAM TILE), weight dtype, sharding individual matmuls, fusing block-internal sequences.

**Refresh:**
- Add explicit scope statement at the top: "This skill is per-block. For pipeline-level perf (paged_update_cache, reusable trace across `generate()` calls, integrated tracy), see `skills/perf/`."
- Cross-link to the perf skill for the patterns that don't fit per-block work.
- Document the at-ceiling outcome explicitly: for leaf blocks already at HiFi4 + fp32_dest_acc + bf16 DRAM TILE, "no improvement found → status=ok" is a valid result. The real wins live in `skills/perf/`.

### `skills/architecture/SKILL.md` — REFRESH

**Refresh:**
- Add `## Use case inventory` section documenting the model-agnostic discovery procedure (inspect HF class hierarchy → derive use_cases[] entries).
- Document the seven-metric known set and the threshold-expression syntax.
- Document the schema additions to `architecture_inventory.json` (use_cases[] entries).
- Update `ARCHITECTURE.md` template to include a `## Use cases` markdown table.

## State schema additions

The existing state schema (`SPEC.md` §State schema) gains a sibling array.

```json
{
  "schema_version": 1,
  "model_id": "...",
  "model_slug": "...",
  "components": [ ... existing ... ],
  "use_cases": [
    {
      "name": "<short_token, e.g. t2tt>",
      "description": "<one-sentence>",
      "input_modality": "text|audio|image|video|none",
      "output_modality": "text|audio|image|video|none",
      "components_used": ["<comp_name>", ...],
      "needs_ar": true,
      "needs_audio_out": false,
      "hf_class": "<e.g. SeamlessM4Tv2ForTextToText>",
      "validation_metric": "bleu",
      "validation_threshold": "HF - 1.0",
      "hybrid_notes": "<optional: parts that should stay on HF host>",
      "generation":     {"status": "pending", "attempts": 0},
      "perf":           {"status": "pending", "attempts": 0}
    }
  ],
  "locks": { ... existing ... },
  "tick_log": [ ... existing ... ],
  "config": { ... existing ... }
}
```

Each component also gains a `real_weights` column (parallel to `reference`, `ttnn`, `debug`, `optimization`).

## Dispatch logic additions

`lib/dag.py::eligible_blocks` decision tree extension (in order, after the existing rules):

1. Architecture pending → architecture worker (unchanged in shape; now also emits use_cases).
2. Existing per-component phases (reference / ttnn / debug / optimization).
3. **NEW**: Any component with `real_weights.status ∈ {pending, failing}` AND `ttnn.status=done` AND `optimization.status=done` → real-weights worker. Per-component, single-block, device-locked.
4. **NEW**: Any use_case with `generation.status ∈ {pending, failing}` AND every `components_used` entry has both `ttnn.status=done` AND `real_weights.status=done` → generation worker. Per-use-case, single dispatch, device-locked.
5. **NEW**: Any use_case with `generation.status=done` AND `perf.status ∈ {pending, failing}` → perf worker. Per-use-case, device-locked.
6. Done check: all components have `optimization` + `real_weights` finished AND all use_cases have `generation` + `perf` finished.
7. Deadlock check (unchanged shape; extended to scan use_case phases too).

## Guard extension

`lib/guard.py` adds `verify_use_case()` mirror of `verify_block()`:

- **Static check**: `tt/<use_case>_model.py` must import the TTNN block modules in `use_case.components_used`. No copy-paste of block code (would defeat the bringup work).
- **Demo CLI check**: `demo/demo_<use_case>.py` must invoke an HF reference path for side-by-side output. Grep for the HF class name.
- **E2E test check**: `tests/test_e2e_<use_case>.py` must enforce the `validation_metric` gate against HF using the use case's `validation_threshold`. Parse the threshold expression and confirm the assertion matches.

`verify_use_case()` is called by the tick after generation-worker and perf-worker returns. Failure routes to debug-worker (extended to handle use_case-scope failures).

## Tick.md changes

Step 3 (decision tree) gains three new branches: real_weights, generation, perf.

Step 4 (mutation) extends status logic to use_case phases: same status enum, same `attempts++ → blocked-at-max` policy.

Step 5 (guard) runs:
- After real-weights success: `lint_block` (existing) + new "weights actually loaded" check (assert at least N parameters loaded from HF state_dict, with N derived from the HF module).
- After generation success: `verify_use_case` + parity-gate assertion against HF using the use case's validation_metric + threshold.
- After perf success: parity preserved (re-run the e2e test) + perf delta reported (no regression threshold — even 0× is acceptable, just report).

Failure in any of these → standard failing/blocked transition.

## Worker contracts (uniform with existing)

All three new workers return JSON last-line. The original `block` key now generalizes to `target` (a string that's either a component name or a use_case name, depending on phase). The original `pcc` field generalizes to `metric` (a single-entry dict keyed by metric name).

```json
{
  "target": "<component_name or use_case_name>",
  "target_type": "component|use_case",
  "phase": "real_weights|generation|perf",
  "status": "ok|fail|blocked",
  "metric": {"<metric_name>": <float>},
  "artifacts": ["<paths>"],
  "notes": "<str>",
  "last_error": "<str|null>",
  "hang_detected": false
}
```

For backward compatibility, existing workers (reference, ttnn, debug, optimization) can keep returning `block` + `pcc`; the tick parses both old and new shapes. New workers MUST use the new shape.

## Implementation order

Phase A: skills (5 files written/refreshed). Independent — can be done without orchestrator changes.

Phase B: orchestrator state + dispatch (4 files: state.py, dag.py, guard.py, + tests). Adds the use_cases axis and the three new dispatch branches.

Phase C: worker prompts (4 files: 3 new + 1 update to architecture-worker.md).

Phase D: tick.md + SKILL.md + SPEC.md updates.

Phase E: smoke validation against a hand-crafted state fixture (no real subagent dispatch — just lib-level eligible_blocks correctness).

Per the existing orchestrator's design, each implementation phase is a per-tick unit of work; the writing-plans skill will break it into per-step checklists with concrete file paths and test commands.

## Verification

After all phases:

1. `pytest skills/orchestrator/lib/tests/ -v` — all existing tests pass + new tests for state/dag/guard cover the use_cases axis and the three new phases.
2. The dag.py decision tree correctly walks the new pipeline against a hand-crafted fixture state at `skills/orchestrator/lib/tests/fixtures/post_bringup_fixture.json` containing: 2 components (both at `ttnn=done, optimization=done`), 2 use_cases (1 with `needs_ar=true`, 1 with `needs_ar=false`). Walk-through assertion:
   - At start: eligible = real_weights for first component.
   - After both components real_weights=done: eligible = generation for first use_case.
   - After first use_case generation=done: eligible = generation for second OR perf for first (priority order documented in dag.py).
   - After all use_cases generation=done: eligible = perf for first use_case.
   - After all use_cases perf=done: eligible = done.
3. Documentation cross-links between skills are accurate (skills/optimization ↔ skills/perf, skills/architecture ↔ skills/generation, etc.). Verified by a small lint script checking that referenced files exist.
4. Re-running `/bringup --resume models/demos/facebook_seamless_m4t_v2_large` under the extended orchestrator (after this work lands) correctly recognizes that the manual Phases 1-9c work was done out-of-band and either (a) verifies the existing state and auto-advances to done, or (b) re-runs the new phases over the existing artifacts without breaking them. The resume_normalize logic + idempotency of the worker prompts make this safe.

## Out of scope (deferred)

- Auto-generation of `validate.py` helper logic for novel metrics. New metrics need a manual addition.
- Sub-batching / multi-sequence packing for higher throughput.
- Beam search / nucleus sampling beyond top-k/top-p basics.
- Multi-host orchestration for very large models.
- Serving harness. CLI demos only.
- A "universal demo" CLI. Each use case keeps its own script.

## Risk callouts

- **Worker scope vs cross-block changes.** The perf worker is allowed to touch multiple TTNN block files in one tick (since paged_update_cache migration is cross-block). This is a one-tick exception to the orchestrator's "one block per tick" rule. The exception is gated by the `verify_use_case` correctness check after.
- **Generation worker is the first per-use-case dispatch.** It's also the heaviest (writes a model wrapper, a demo CLI, an e2e test, all in one tick). If it crashes mid-tick, the orchestrator's resume semantics need to demote partial state cleanly. The existing `resume_normalize` logic extended to use_case phases handles this.
- **Architecture worker downstream impact.** Adding use_cases inventory changes what every subsequent worker reads. If the architecture worker emits malformed use_cases (e.g. wrong components_used), the generation phase will deadlock. The orchestrator's guard validates the use_cases schema on architecture-worker return.
