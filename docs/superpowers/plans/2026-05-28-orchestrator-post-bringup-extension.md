# Orchestrator Post-Bringup Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the bringup orchestrator and skills/ so a single `/bringup <hf_id>` autonomously runs all the way from "model on disk" to "all use cases pass HF parity gates with optimized perf." Adds three new phases (`real_weights`, `generation`, `perf`) and three new skills (`integration`, `generation`, `perf`) — all model-agnostic.

**Architecture:** State gains a second axis `use_cases[]` (parallel to existing `components[]`). The architecture worker discovers use cases by inspecting the HF model's class hierarchy at run time. Three new orchestrator phases dispatch per-component (real_weights) or per-use-case (generation, perf). Skills capture reusable patterns from the SeamlessM4T-v2 bringup so future models reuse them.

**Tech Stack:** Python 3.10, TTNN (Wormhole/Blackhole), pytest, the existing orchestrator at `skills/orchestrator/`, sacrebleu/jiwer for validation metrics.

**Spec reference:** `skills/orchestrator/SPEC_post_bringup.md` (commit `b7c0addaadb`)

---

## File Structure

**Files this plan creates:**

```
skills/integration/SKILL.md                          (NEW — real HF weights loader + per-block re-val patterns)
skills/generation/SKILL.md                           (NEW — AR loop + demos + e2e validation patterns)
skills/perf/SKILL.md                                 (NEW — pipeline-level perf: paged_update_cache + reusable trace + integrated tracy)
skills/orchestrator/workers/real-weights-worker.md   (NEW — per-component dispatch)
skills/orchestrator/workers/generation-worker.md     (NEW — per-use-case dispatch)
skills/orchestrator/workers/perf-worker.md           (NEW — per-use-case dispatch, two sub-passes)
skills/orchestrator/lib/tests/fixtures/post_bringup_fixture.json   (NEW — hand-crafted state for smoke walk)
```

**Files this plan modifies:**

```
skills/architecture/SKILL.md                         (refresh — use_cases discovery procedure)
skills/optimization/SKILL.md                         (refresh — explicit per-block scope; cross-link to perf)
skills/orchestrator/SKILL.md                         (refresh — new phases mentioned in overview)
skills/orchestrator/SPEC.md                          (refresh — pointer to SPEC_post_bringup.md)
skills/orchestrator/tick.md                          (extend Step 3 decision tree + Step 4 mutation + Step 5 guard)
skills/orchestrator/workers/architecture-worker.md   (extend — emit use_cases inventory)
skills/orchestrator/lib/state.py                     (extend — USE_CASE_PHASES, validate use_cases[], extend bootstrap/resume/render)
skills/orchestrator/lib/dag.py                       (extend — new branches in eligible_blocks)
skills/orchestrator/lib/guard.py                     (extend — verify_use_case)
skills/orchestrator/lib/tests/test_state.py          (extend — use_cases fixtures + validation tests)
skills/orchestrator/lib/tests/test_dag.py            (extend — new branches coverage)
skills/orchestrator/lib/tests/test_guard.py          (extend — verify_use_case tests)
```

**Files NOT modified:** the 24 SeamlessM4T-v2 TTNN block files, the `skills/reference/` / `skills/ttnn/` / `skills/debug/` skills, the existing per-component worker prompts.

---

## Phase A — Skills (foundation)

### Task A1: Write `skills/integration/SKILL.md`

- [ ] **Step 1: Create `skills/integration/SKILL.md`**

Content sections:

```markdown
# SKILL: HF Weight Integration

## Purpose
Load real HuggingFace safetensors weights into TTNN module instances
and re-validate that each block hits PCC > 0.99 against the HF
PyTorch reference at production layer counts.

## When to use
- After the bringup phase produces TTNN modules with synthetic weights.
- Before any end-to-end demo or use-case validation.
- When a model's checkpoint is updated and per-block PCC needs to be
  re-verified.

## Process
### 1. Inspect the safetensors index
... (full procedure: read `model.safetensors.index.json` for top-level
prefixes; map prefixes to TTNN sub-models)

### 2. Write the weight loader (`tt/weight_loader.py`)
... (one fn per block kind, taking `(hf_state_dict, layer_idx)` and
returning a nested PyTorch state_dict matching the TTNN module's
`__init__` expectation)

### 3. Handle the awkward cases
- Weight tying (shared embedding ↔ LM head ↔ etc)
- Buffers not in the checkpoint (sinusoidal positional embeddings,
  distance embeddings — rebuilt deterministically per HF source)
- Per-layer indexing (HF flat keys `text_encoder.layers.0.q_proj.weight`)

### 4. Set up a parametric PCC test
... (one consolidated `tests/test_real_hf_weights.py` parametrized
over all blocks, with session-scoped HF state_dict fixture and
function-scoped device fixture to avoid bank_manager OOM)

### 5. The realistic-input trick
... (random N(0,1) inputs saturate bf16 ~600×; for attention/decoder
layers, derive realistic inputs from embed + LN + position-add chain)

### 6. Two-stage validation
... (reduced-config 2-layer for iteration; full-config 24-layer for
final gate; expect PCC drift with depth)

## Output artifacts
- `tt/weight_loader.py`
- `tests/test_real_hf_weights.py`
- One row per block in the BRINGUP_LOG.md `real_weights` column.

## Failure modes
- PCC drops at depth → investigate per-layer accumulation; check
  for stale dtype assumptions.
- Missing checkpoint key → reverse-map: which HF source line writes
  this attr to state_dict?
- HF module instantiation fails → wrong config keys; cross-reference
  with HF's source `__init__` signature.

## Reference implementation
- `models/demos/facebook_seamless_m4t_v2_large/tt/weight_loader.py` —
  full working example covering 24 components.
- `models/demos/facebook_seamless_m4t_v2_large/tests/test_real_hf_weights.py` —
  parametric harness reference.
```

- [ ] **Step 2: Manual smoke**

Read the file end-to-end. Confirm: it's a SKILL.md following the
conventions of existing skills (skills/architecture/SKILL.md, etc.).
No model-specific names except in the "Reference implementation"
section.

- [ ] **Step 3: Commit**

```bash
git add skills/integration/SKILL.md
git commit -m "skills/integration: new skill for HF weight loading + per-block re-validation"
```

### Task A2: Write `skills/generation/SKILL.md`

- [ ] **Step 1: Create `skills/generation/SKILL.md`**

Content sections:

```markdown
# SKILL: Generation (AR Loop + Demos + E2E Validation)

## Purpose
Wire a working end-to-end pipeline for one use case: encoder forward
→ (optional AR decode with KV cache + sampling + EOS) → (optional
audio post-processing) → output. Plus a demo CLI and an e2e
validation test gated against HF parity.

## When to use
- After all `components_used` for the use case have ttnn=done AND
  real_weights=done.
- Once per use case (subsequent use cases reuse the AR
  infrastructure built by the first).

## Process
### 1. Determine the pipeline shape
... (read use_case spec: needs_ar? needs_audio_out? components_used?)
... (decide pipeline: encoder-only vs encoder+AR-decoder vs +vocoder)

### 2. KV cache (skip if needs_ar=false)
... (Self-attn: per-layer [B, num_heads, max_seq, head_dim]; updated
in place each step.)
... (Cross-attn: per-layer [B, num_heads, enc_seq, head_dim];
populated once after encoder.)
... (Use `paged_update_cache` from day one — see skills/perf/SKILL.md
for why update_cache(int_pos) breaks trace reuse.)

### 3. decode_step contract
... (persistent input/position/mask tensors as inputs; logits as
output; single-token Q, full-cache K/V via SDPA)

### 4. Prefill order
... (encoder forward → populate cross-attn cache → run decoder
warmup at model-specific prefix tokens)
... (For NLLB-style: `[decoder_start_token_id, lang_id]`)
... (Read HF's `.generate()` to find the prefix for the model.)

### 5. AR loop + sampling + EOS
... (Greedy = argmax; top-k/top-p via models/common/sampling/tt_sampling.py;
logits processors via models/common/generation_utils.py)
... (Stop on EOS or max_new_tokens)

### 6. Demo CLI shape
... (typer-based)
... (Args by input modality: text → `--src/--src-lang/--tgt-lang`;
audio → `--wav --src-lang`)
... (`--out` for audio output)
... (ALWAYS run HF reference alongside TTNN for side-by-side output;
mandatory user-trust check)

### 7. Validation by output modality
... (text out (translation) → bleu via sacrebleu)
... (text out (transcription) → wer via jiwer)
... (audio out → ecapa_cos primary; re-ASR char_similarity fallback)
... (embeddings → pcc against HF)

### 8. E2E test
... (For each known sample input, compute metric vs HF reference;
gate per use_case.validation_threshold; parity-relative thresholds
like "HF - 1.0" or absolute like "≥ 0.95")

### 9. Hybrid host/device boundaries
... (Document any parts that legitimately stay on HF host in the
use_case's hybrid_notes field — e.g. tokenizer-bound char prep for
TTS pipelines. The orchestrator's "no shortcuts" guard for ttnn-phase
does NOT extend to "no host ops anywhere downstream of bringup.")

## Output artifacts
- `tt/<use_case>_model.py`
- `demo/demo_<use_case>.py`
- `demo/validate.py` (helpers per metric; expand as needed)
- `demo/inputs/<use_case>_samples.json` or `.wav`
- `tests/test_e2e_<use_case>.py`

## Failure modes
- TTNN tokens diverge from HF mid-generation → logits PCC issue;
  check bf16 precision on lm_head; switch to argmax-token-match as
  the gate metric (not logits PCC).
- BLEU/WER drops below gate → enable HF/TTNN side-by-side print;
  inspect which samples diverge.
- Audio output sounds wrong → suspect vocoder lang_id (use
  vocoder_lang_code_to_id from HF config, not text decoder's
  tgt_lang_id).

## Reference implementations
- T2TT: `models/demos/facebook_seamless_m4t_v2_large/tt/text_to_text_model.py`
- S2TT/ASR: `.../tt/speech_to_text_model.py`
- T2ST: `.../tt/text_to_speech_model.py`
- S2ST: `.../tt/speech_to_speech_model.py`
- The shared TextGenerator: `.../tt/text_generator.py`

## Cross-references
- For KV cache patterns: also see `skills/perf/SKILL.md` (paged_update_cache).
- For per-block tuning beyond what generation needs: `skills/optimization/SKILL.md`.
```

- [ ] **Step 2: Manual smoke**

Read end-to-end. Confirm model-agnostic throughout.

- [ ] **Step 3: Commit**

```bash
git add skills/generation/SKILL.md
git commit -m "skills/generation: new skill for AR loop + demos + e2e validation"
```

### Task A3: Write `skills/perf/SKILL.md`

- [ ] **Step 1: Create `skills/perf/SKILL.md`**

Content sections:

```markdown
# SKILL: Pipeline Perf (paged_update_cache + reusable trace + targeted tracy)

## Purpose
Pipeline-level perf optimization for one use case. Distinct from
skills/optimization/ (which is per-block); this skill covers
cross-block refactors and integrated-pipeline tuning.

## When to use
- After a use case's generation phase is done (e2e test passes).
- Before declaring the model "ready" for deployment.

## Why this skill exists (and not just skills/optimization/)
Per-block optimization can tune one matmul's kernel config or shard
one ttnn op. The wins documented here REQUIRE touching multiple
files at once (kv_cache.py + text_generator.py + the cached-attention
path in the attention block), and the correctness invariants span
the whole AR loop, not one block.

## Process
### Sub-pass 1: trace
... (Skipped if needs_ar=false.)

The trace pitfall:
... (ttnn.update_cache(update_idx=int_pos) bakes the int into the
trace; single-trace replay across positions breaks.)

The fix:
... (Migrate to ttnn.experimental.paged_update_cache(update_idxs_tensor=
cur_pos_tt). Position lives in a device tensor; one trace replays
for all positions AND across generate() calls.)

Persistent buffers:
... (input_ids, position_ids, self-mask. Hot loop reduces to
copy_host_to_device_tensor → execute_trace → host argmax.)

KV cache reset between generate() calls:
... (copy_host_to_device_tensor to zero in place; preserves buffer
addresses for the trace. Do NOT free and reallocate.)

Trace lifecycle:
... (Capture once at end of first generate() warmup; reused thereafter.)

### Sub-pass 2: tracy
... (1 warmup + N timed/profiled invocations; output CSV bucketed
by op-code and memory_config.memory_layout.)

Host-vs-device bound triage:
... (If total_ms < kernel_time_sum + small_host_margin → host-dispatch
limited → trace fixes it. If total_ms ≈ kernel_time_sum → device-compute
limited → attack sharding / fusion / lower precision.)

Apply ONE targeted optimization:
... (Based on tracy findings: shard the largest matmul, fuse a hot
LN+Linear chain, lower precision on a non-PCC-sensitive matmul.)

Validate correctness:
... (Re-run the e2e test. HF parity must be preserved. PCC tests on
touched blocks must still pass.)

## Reality check (from SeamlessM4T-v2)
Trace + reusable trace delivered 1.21× because the floor is device
kernel time, not host. The "50% from trace" hypothesis was wrong for
that model. Further wins require compute-side attacks.

## Output artifacts
- Modified tt/kv_cache.py + tt/text_generator.py (or equivalents)
- tt/profile_<use_case>.py — tracy harness
- PERF_NOTES.md — baseline + after numbers, top hot ops, applied
  optimizations, recommendations for further work

## Failure modes
- paged_update_cache shape mismatch — check whether the cache
  layout needs adjustment (sometimes blocked / paged layout needed
  in real paging contexts; for single-page-per-cache the existing
  contiguous layout works).
- Trace capture fails on a specific op — back out cleanly, document
  in PERF_NOTES.md.
- Tracy shows zero clear hot op — fold the work into a single
  "characterized, no optimization yielded > 5% win" note. That's a
  legitimate outcome.

## Reference implementation
- models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py
  (paged_update_cache + reusable trace)
- models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py
  (tracy harness)
- models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md
  (write-up format)

## Cross-references
- For per-block tuning (sharding individual matmul, kernel config):
  skills/optimization/SKILL.md
- For HF parity baseline before perf attack: skills/generation/SKILL.md
```

- [ ] **Step 2: Manual smoke**

Read end-to-end. Confirm model-agnostic + the scope boundary with
skills/optimization/ is crystal clear.

- [ ] **Step 3: Commit**

```bash
git add skills/perf/SKILL.md
git commit -m "skills/perf: new skill for pipeline-level perf (paged_update_cache + reusable trace + tracy)"
```

### Task A4: Refresh `skills/optimization/SKILL.md`

- [ ] **Step 1: Add scope statement at the top**

At the top of `skills/optimization/SKILL.md`, after the existing
"# SKILL: Performance Optimization" heading, insert:

```markdown
## Scope: per-block

This skill is **per-block**. It tunes one TTNN module at a time:
compute kernel config (HiFi4 + fp32_dest_acc), memory layout (DRAM
TILE), weight dtype, sharding individual matmuls, fusing
block-internal sequences.

For **pipeline-level perf** — cross-block refactors like
paged_update_cache migration, reusable metal trace across
generate() calls, integrated tracy on the full AR pipeline — see
`skills/perf/SKILL.md`. Those patterns require touching multiple
block files at once and don't fit per-block dispatch.

A leaf block already using HiFi4 + fp32_dest_acc + bf16 DRAM TILE
is at-ceiling for this skill. "No improvement found → status=ok"
is a valid outcome.
```

- [ ] **Step 2: Add cross-link in the existing tracy section**

In `skills/optimization/SKILL.md` where the tracy harness pattern
is documented, add a paragraph noting that the harness here is
block-scoped; pipeline-scoped tracy harness lives in
`skills/perf/SKILL.md`.

- [ ] **Step 3: Commit**

```bash
git add skills/optimization/SKILL.md
git commit -m "skills/optimization: clarify per-block scope; cross-link to skills/perf"
```

### Task A5: Refresh `skills/architecture/SKILL.md`

- [ ] **Step 1: Add use_cases discovery section**

After the existing component-inventory section, add:

```markdown
## Use case inventory

In addition to the component DAG, the architecture worker emits a
`use_cases[]` array describing every distinct inference path the
model exposes. This is what the orchestrator uses to dispatch
per-use-case work (generation, perf).

### How to discover use cases (model-agnostic)
1. Inspect the HF model's class hierarchy:
   - Multiple task-specific classes (e.g. `XxxForAToB`) → one use
     case per class.
   - Single class with a task/modality arg → one per task value.
   - Single class with no .generate() → one inference use case
     (encoder-only, classifier, etc.).
2. For each, derive the model-agnostic fields:
   - name: lowercase token from HF class name (opaque to orchestrator)
   - input_modality / output_modality from class signatures: text /
     audio / image / video / none
   - components_used: subset of components[] the use case touches
   - needs_ar: class has .generate() or inherits from GenerationMixin
   - needs_audio_out: output_modality == "audio"
   - validation_metric: pick from {bleu, wer, ecapa_cos, perplexity,
     accuracy, mse, pcc} based on output modality + use case type
   - validation_threshold: parity-relative like "HF - 1.0" or
     absolute like "≥ 0.95"
   - hybrid_notes: optional notes on parts that should stay on HF host

### Schema in architecture_inventory.json
```json
{
  "components": [ ... existing ... ],
  "use_cases": [
    {
      "name": "<short_token>",
      "description": "<one sentence>",
      "input_modality": "text|audio|image|video|none",
      "output_modality": "text|audio|image|video|none",
      "components_used": ["<comp>", ...],
      "needs_ar": true,
      "needs_audio_out": false,
      "hf_class": "<HF class name>",
      "validation_metric": "<name>",
      "validation_threshold": "<expression>",
      "hybrid_notes": null
    }
  ]
}
```

### ARCHITECTURE.md additions
Add a `## Use cases` markdown table to the model's ARCHITECTURE.md
listing each entry plus the "Components used" column.
```

- [ ] **Step 2: Commit**

```bash
git add skills/architecture/SKILL.md
git commit -m "skills/architecture: refresh — use_cases discovery procedure + schema"
```

---

## Phase B — Orchestrator state + dispatch

### Task B1: Extend `lib/state.py`

- [ ] **Step 1: Write the failing tests first (TDD)**

Add to `skills/orchestrator/lib/tests/test_state.py`:

```python
def test_bootstrap_includes_empty_use_cases():
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    assert state["use_cases"] == []

def test_real_weights_phase_default_pending_per_component():
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["components"].append(_minimal_component("Linear"))
    # After bootstrap, manually appended components don't auto-get
    # real_weights status; the validator should accept missing
    # real_weights (treated as pending by dag.py).
    save_state(tmp / "s.json", state)  # must not raise

def test_use_case_validation_required_keys():
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["use_cases"].append({"name": "x"})  # missing required keys
    with pytest.raises(SchemaError):
        save_state(tmp / "s.json", state)
```

Run: `pytest skills/orchestrator/lib/tests/test_state.py -k "use_cases or real_weights" -v`
Expect: FAIL (USE_CASE_PHASES not defined, use_cases not validated).

- [ ] **Step 2: Implement in `lib/state.py`**

Add module-level constants:
```python
USE_CASE_PHASES = ("generation", "perf")
REQUIRED_USE_CASE_KEYS = frozenset({
    "name", "description", "input_modality", "output_modality",
    "components_used", "needs_ar", "needs_audio_out", "hf_class",
    "validation_metric", "validation_threshold",
})
```

Add `real_weights` to `PHASE_NAMES`.

In `bootstrap()`: add `state["use_cases"] = []`.

In `_validate()`: add use_cases validation — each entry must have all REQUIRED_USE_CASE_KEYS, validation_metric ∈ known set, components_used ⊆ component names.

In `resume_normalize()`: extend the in_progress→pending demotion loop to use_case phases too.

In `render_log()`: add a "## Use cases" section after components, with status columns for generation + perf.

- [ ] **Step 3: Run tests, confirm green**

```bash
cd /local/ttuser/ssinghal/tt-metal && source python_env/bin/activate && PYTHONPATH=$(pwd) pytest skills/orchestrator/lib/tests/test_state.py -v
```

Expect: all tests pass (39 prior + 3 new).

- [ ] **Step 4: Commit**

```bash
git add skills/orchestrator/lib/state.py skills/orchestrator/lib/tests/test_state.py
git commit -m "skills/orchestrator: extend state.py with use_cases axis + real_weights phase"
```

### Task B2: Extend `lib/dag.py`

- [ ] **Step 1: Write the failing tests first**

Add to `skills/orchestrator/lib/tests/test_dag.py`:

```python
def test_real_weights_dispatched_after_optimization():
    state = _state([_component("A", reference="done", ttnn="done",
                               optimization="done")])
    # A has no real_weights yet — should dispatch
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "A", "worker": "real_weights"}

def test_generation_dispatched_after_all_components_real_weights_done():
    state = _state([
        _component("Linear", reference="done", ttnn="done",
                   optimization="done", real_weights="done"),
    ])
    state["use_cases"].append({
        "name": "x", "input_modality": "text", "output_modality": "text",
        "components_used": ["Linear"], "needs_ar": True,
        "needs_audio_out": False, "hf_class": "XxxForX",
        "validation_metric": "bleu", "validation_threshold": "HF - 1.0",
        "description": "", "hybrid_notes": None,
        "generation": {"status": "pending"}, "perf": {"status": "pending"},
    })
    result = eligible_blocks(state)
    assert result["phase"] == "device"
    assert result["worker"] == "generation"
    assert result["use_case"] == "x"

def test_generation_blocked_if_dep_not_real_weights_done():
    # Component used by use_case has ttnn=done but real_weights=pending
    # → use case NOT eligible for generation
    ...

def test_perf_dispatched_after_generation_done(): ...
def test_done_requires_all_use_cases_complete(): ...
```

Run: expect FAIL.

- [ ] **Step 2: Implement in `lib/dag.py`**

Add new branches in `eligible_blocks` decision tree (in order after the existing rules):

```python
# Branch 3-extended: real_weights candidates
for comp in components:
    rw_status = _phase_status(comp, "real_weights")
    if (rw_status in (None, "pending", "failing")
        and rw_status != "blocked"
        and _is_finished(comp, "ttnn")
        and _is_finished(comp, "optimization")):
        return {"phase": "device", "block": comp["name"], "worker": "real_weights"}

# Branch 4: generation candidates
for uc in use_cases:
    gen_status = _phase_status(uc, "generation")
    if gen_status not in (None, "pending", "failing"):
        continue
    # Check all components_used have real_weights=done
    deps_ok = all(
        _is_finished(_find_component(state, name), "real_weights")
        for name in uc["components_used"]
    )
    if deps_ok:
        return {"phase": "device", "use_case": uc["name"], "worker": "generation"}

# Branch 5: perf candidates
for uc in use_cases:
    if (_is_finished(uc, "generation")
        and _phase_status(uc, "perf") in (None, "pending", "failing")):
        return {"phase": "device", "use_case": uc["name"], "worker": "perf"}
```

Extend `done` check + `deadlock` check.

- [ ] **Step 3: Run tests, confirm green**

```bash
pytest skills/orchestrator/lib/tests/test_dag.py -v
```

Expect: prior + 5 new tests pass.

- [ ] **Step 4: Commit**

```bash
git add skills/orchestrator/lib/dag.py skills/orchestrator/lib/tests/test_dag.py
git commit -m "skills/orchestrator: extend dag.py with real_weights / generation / perf branches"
```

### Task B3: Extend `lib/guard.py`

- [ ] **Step 1: Write the failing test first**

Add to `skills/orchestrator/lib/tests/test_guard.py`:

```python
def test_verify_use_case_rejects_copy_paste(tmp_path):
    # tt/<use_case>_model.py must import the TTNN modules, not duplicate them
    model_file = tmp_path / "x_model.py"
    model_file.write_text("import ttnn\n\n# no import of components\n")
    use_case = {"name": "x", "components_used": ["Linear"], "hf_class": "XX"}
    verdict = verify_use_case(model_file, use_case)
    assert not verdict.ok
    assert any("import" in str(i).lower() for i in verdict.issues)

def test_verify_use_case_demo_must_call_hf_reference(tmp_path): ...
def test_verify_use_case_test_must_enforce_threshold(tmp_path): ...
```

- [ ] **Step 2: Implement `verify_use_case` in `lib/guard.py`**

```python
@dataclass(frozen=True)
class UseCaseVerdict:
    issues: list[str]
    @property
    def ok(self) -> bool:
        return not self.issues

def verify_use_case(model_path, use_case, demo_path=None, test_path=None) -> UseCaseVerdict:
    issues = []
    # Static check: model file imports each components_used module
    src = Path(model_path).read_text()
    for comp_name in use_case.get("components_used", []):
        module_path = f"tt.{comp_name}"  # ish
        if module_path.split(".")[-1] not in src:
            issues.append(f"{model_path}: doesn't import {comp_name}")
    # Demo file mentions hf_class
    if demo_path:
        demo_src = Path(demo_path).read_text()
        hf = use_case.get("hf_class", "")
        if hf and hf not in demo_src:
            issues.append(f"{demo_path}: doesn't run HF reference {hf}")
    # Test file enforces validation_metric + threshold
    if test_path:
        test_src = Path(test_path).read_text()
        metric = use_case.get("validation_metric", "")
        if metric not in test_src:
            issues.append(f"{test_path}: doesn't reference metric {metric}")
    return UseCaseVerdict(issues=issues)
```

- [ ] **Step 3: Run tests, confirm green**

```bash
pytest skills/orchestrator/lib/tests/test_guard.py -v
```

- [ ] **Step 4: Commit**

```bash
git add skills/orchestrator/lib/guard.py skills/orchestrator/lib/tests/test_guard.py
git commit -m "skills/orchestrator: add verify_use_case to guard.py"
```

### Task B4: Smoke fixture + walk-through test

- [ ] **Step 1: Create the fixture**

Write `skills/orchestrator/lib/tests/fixtures/post_bringup_fixture.json`:

```json
{
  "schema_version": 1,
  "model_id": "acme/fake",
  "model_slug": "acme_fake",
  ...
  "components": [
    {"name": "Linear", "kind": "linear", "depends_on": [],
     "reference_impl": "models/demos/bert/tt/ttnn_optimized_bert.py",
     "reference": {"status": "done", "pcc": 1.0, "attempts": 1},
     "ttnn": {"status": "done", "pcc": 0.999, "attempts": 1},
     "debug": {"status": "n/a"},
     "optimization": {"status": "done"}},
    ... (one more component) ...
  ],
  "use_cases": [
    {"name": "uc_ar", "needs_ar": true, ...,
     "generation": {"status": "pending"}, "perf": {"status": "pending"}},
    {"name": "uc_one_shot", "needs_ar": false, ...,
     "generation": {"status": "pending"}, "perf": {"status": "pending"}}
  ],
  ...
}
```

- [ ] **Step 2: Write the walk-through test**

```python
def test_smoke_post_bringup_walk():
    state = load_state("skills/orchestrator/lib/tests/fixtures/post_bringup_fixture.json")
    # Step 1: components need real_weights
    r = eligible_blocks(state)
    assert r["worker"] == "real_weights"
    # Mark both components real_weights=done
    for c in state["components"]: c["real_weights"] = {"status": "done"}
    # Step 2: generation eligible
    r = eligible_blocks(state)
    assert r["worker"] == "generation"
    # Mark both use_cases generation=done
    for uc in state["use_cases"]: uc["generation"] = {"status": "done"}
    # Step 3: perf eligible
    r = eligible_blocks(state)
    assert r["worker"] == "perf"
    # Mark both perf=done
    for uc in state["use_cases"]: uc["perf"] = {"status": "done"}
    # Step 4: done
    assert eligible_blocks(state) == {"phase": "done"}
```

Run: `pytest -k smoke_post_bringup_walk -v`.

- [ ] **Step 3: Commit**

```bash
git add skills/orchestrator/lib/tests/fixtures/ skills/orchestrator/lib/tests/test_e2e_smoke.py
git commit -m "skills/orchestrator: smoke fixture + walk-through test for post-bringup phases"
```

---

## Phase C — Worker prompts

### Task C1: Write `workers/real-weights-worker.md`

- [ ] **Step 1: Create the file**

Follow the existing worker prompt template (architecture-worker.md as a reference). Required sections:
- Skill binding: `Skill(integration)`.
- Input spec JSON shape (single component target).
- Output JSON shape (target/target_type/metric form per SPEC).
- Process steps invoking the integration skill.
- Anti-shortcut clauses (forward returns ttnn.Tensor; no banned strings).

Length target: 60-100 lines.

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/real-weights-worker.md
git commit -m "skills/orchestrator: real-weights-worker.md prompt"
```

### Task C2: Write `workers/generation-worker.md`

- [ ] **Step 1: Create the file**

Similar structure. Skill binding `Skill(generation)`. Input spec is a single use_case (with all its fields). Output JSON uses `target_type=use_case`. Process notes the AR-loop reuse pattern (first use case builds infrastructure, subsequent ones import it).

Length target: 80-120 lines.

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/generation-worker.md
git commit -m "skills/orchestrator: generation-worker.md prompt"
```

### Task C3: Write `workers/perf-worker.md`

- [ ] **Step 1: Create the file**

Skill binding `Skill(perf)`. Two sub-pass structure documented:
1. Trace sub-pass (skip if needs_ar=false).
2. Tracy sub-pass.

Process clarifies the cross-block scope (worker may touch multiple TTNN block files; this is an intentional exception to "one block per tick" gated by verify_use_case after).

Length target: 80-120 lines.

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/perf-worker.md
git commit -m "skills/orchestrator: perf-worker.md prompt"
```

### Task C4: Update `workers/architecture-worker.md`

- [ ] **Step 1: Extend Step 4 (inventory generation)**

Add the use_cases discovery procedure (cross-reference to skills/architecture/SKILL.md for the full algorithm).

- [ ] **Step 2: Update the output schema example**

Include the `use_cases[]` array in the architecture_inventory.json sample.

- [ ] **Step 3: Commit**

```bash
git add skills/orchestrator/workers/architecture-worker.md
git commit -m "skills/orchestrator: architecture-worker emits use_cases inventory"
```

---

## Phase D — Tick + SKILL.md + SPEC.md updates

### Task D1: Update `tick.md`

- [ ] **Step 1: Extend Step 3 (decision tree)**

Add three new branches matching the dag.py logic: real_weights (per-component), generation (per-use-case), perf (per-use-case). Each dispatches via `Agent` with `subagent_type=general-purpose` invoking the corresponding worker prompt.

- [ ] **Step 2: Extend Step 4 (mutation logic)**

Same status enum (ok/fail/blocked/etc.) for use_case phases. Same attempts++ → blocked-at-max policy.

- [ ] **Step 3: Extend Step 5 (guard checks)**

- After real_weights ok: lint_block + new "weights actually loaded" assertion (≥ N parameters loaded).
- After generation ok: verify_use_case + parity-gate assertion via e2e test rerun.
- After perf ok: parity preserved (e2e test rerun) + perf delta reported.

- [ ] **Step 4: Commit**

```bash
git add skills/orchestrator/tick.md
git commit -m "skills/orchestrator: tick.md handles real_weights / generation / perf branches"
```

### Task D2: Update `SKILL.md` (orchestrator entry)

- [ ] **Step 1: Add new-phase mentions**

In the overview section: extend the "phase list" to include real_weights / generation / perf.
In the glossary: define use_case + the new phases.

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/SKILL.md
git commit -m "skills/orchestrator: SKILL.md mentions extended phase set"
```

### Task D3: Update `SPEC.md`

- [ ] **Step 1: Add pointer at the top**

At the top of `skills/orchestrator/SPEC.md`, after the front matter, add:

```markdown
> **See also:** `SPEC_post_bringup.md` documents the post-bringup
> phase extensions (real_weights / generation / perf) and the
> `use_cases[]` state axis.
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/SPEC.md
git commit -m "skills/orchestrator: SPEC.md points to SPEC_post_bringup.md"
```

---

## Phase E — End-to-end smoke validation

### Task E1: Re-run the lib test suite + integration check

- [ ] **Step 1: Full lib test run**

```bash
cd /local/ttuser/ssinghal/tt-metal && source python_env/bin/activate && PYTHONPATH=$(pwd) pytest skills/orchestrator/lib/tests/ -v
```

Expected: all prior tests still pass + new tests for use_cases / new phases.

- [ ] **Step 2: Manual walk-through with eligible_blocks**

Run the smoke fixture walk-through (Task B4 test) as a final sanity check.

- [ ] **Step 3: Verify documentation cross-links**

Grep for `skills/integration`, `skills/generation`, `skills/perf` across the skill set. Confirm every cross-reference points at a real file.

```bash
for ref in integration generation perf; do
  echo "=== ${ref} ===";
  grep -rn "skills/${ref}" skills/ | grep -v ${ref}/SKILL.md;
done
```

- [ ] **Step 4: Commit (if anything needs touch-up)**

If any cross-reference is broken, fix and commit.

### Task E2: Documentation note in BRINGUP_LOG.md template

- [ ] **Step 1: Update the renderer to include use_cases section**

`state.render_log()` (already extended in Task B1) should render a "## Use cases" table after the existing "## Block Status" table. Confirm via a manual call against the smoke fixture.

```bash
python -c "
from skills.orchestrator.lib.state import load_state, render_log
state = load_state('skills/orchestrator/lib/tests/fixtures/post_bringup_fixture.json')
print(render_log(state))
"
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/lib/state.py
git commit -m "skills/orchestrator: render_log includes use_cases table"
```

---

## Verification across the whole plan

After all 18 tasks land:

1. `pytest skills/orchestrator/lib/tests/ -v` — every unit test green (prior 39+ plus ~10 new for use_cases / new phases).
2. Smoke fixture walk-through passes (Task B4): correctly dispatches real_weights → generation → perf → done.
3. Documentation cross-link check passes (Task E1 Step 3).
4. `git log --oneline skills/` shows the ~18 commits in order.
5. `cat skills/orchestrator/SPEC_post_bringup.md` — the design + the commits + the resulting code are coherent.

After this lands, a future `/bringup <hf_id>` autonomously runs the full pipeline including demos for all use cases and pipeline-level perf optimization, without manual phase transitions.

## Out of scope (will not be done in this plan)

- Implementing the integration / generation / perf workers' actual TTNN code for any model. The skills + workers + orchestrator are the framework; running them against a real model is a separate execution.
- Backporting the new phases to `models/demos/facebook_seamless_m4t_v2_large/` (since that work was done manually). It can be re-run later under `--resume` if desired.
- Performance regression tracking across versions. Each `/bringup` run captures perf numbers; comparison across runs is a follow-up.
- A "skip phase X" CLI nudge for use cases (e.g. some models might want `--skip perf` for fast iteration). The orchestrator's existing `--skip` flag works at the block level; extending to use_case level is deferred.
