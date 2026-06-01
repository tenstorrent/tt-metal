# Optimization-Transfer Model Bring-up — Design

**Date:** 2026-06-01
**Status:** Design approved; ready for implementation planning
**Author:** sdawle (with Claude Code)

## Problem

The existing relay-race bring-up skills (`reference → ttnn → integration → generation → perf`)
reliably produce a *correct* TTNN model (PCC > 0.99) for a new model, but they rebuild each
block **op-by-op naively**. The performance optimizations that already exist in sibling models
— fused ops and, critically, their **program / memory / compute-kernel configs** — do **not**
carry over. The result is a correct-but-slow model that then needs a separate, manual perf pass.

Concrete example (the first vertical slice, see below): `SeamlessMHA._project_and_split`
(`models/demos/facebook_seamless_m4t_v2_large/tt/seamless_mha.py:109-119`) splits heads with a
naive `linear → reshape → transpose`. The code's own docstring notes the `ReshapeViewDeviceOperation`
is **~40% of the block's device time** and only L1-pins it as a workaround — it never uses the
fused `nlp_create_qkv_heads`, which already exists with tuned configs in `tt_transformers` and
the Whisper decoder.

## Goal

Build an **LLM-driven, PCC-gated optimization-transfer system** that brings up a new model on
the device by **transferring the optimizations already implemented across the repo**, at
**op-subsequence granularity** (finer than attention/MLP/norm). A single new model is assembled
from **multiple donors** — its attention head-split from one model, its MLP fusion from another —
whichever already has the optimized version of that op-subsequence.

The differentiator vs. the skills: the system carries over **the fused op *and its config***
(the config is the optimization), and verifies the result both **structurally** (the fusion did
not change the model) and **numerically** (PCC), including **over long decode horizons** (drift
accumulation for reasoning/"thinking" models that emit thousands of tokens).

## Non-goals

- Not a replacement for the skills' *correctness* path; it consumes the PyTorch reference the
  `reference` skill produces.
- Not a deterministic compiler. The matcher is an LLM; determinism comes from the **gates**, not
  the matcher.
- The first spec covers the architecture + one vertical slice. Exhaustive rule mining, the full
  repair loop, and the long-decode/perf machinery are scaled on the proven skeleton (see Scope).

## Architecture

Three layers plus an external operator.

| Layer | Purpose | Backbone | Claude Code role |
|-------|---------|----------|------------------|
| **A — Knowledge base builder** | Mine fused-op knowledge from the repo into a structured corpus | Standalone offline pipeline (Python + LLM) | builds/maintains it; **not** a runtime dependency |
| **B — Matcher + config transfer + fx structural gate** | New-model op-graph → proposed fused ops + transferred configs, structurally validated | Anthropic API matcher + deterministic tool functions | — |
| **C — Bring-up + repair + verification loop** | Codegen, on-device PCC, long-decode drift, perf; repair on failure | **Standalone LangGraph framework, Anthropic API** | external **debugger / editor** only, on demand |

### Layer A — Knowledge base

A retrieval corpus of fused-op knowledge. **Sources:**

- `models/tt_transformers`, `models/tt_dit`, `models/demos` — fused ops **in use** + their **real
  configs**, paired with the torch op-subsequence they replace, parameterized by model dims.
- `tests/ttnn/unit_tests/operations` — the **full inventory of supported fused ops** (including
  ones no model uses yet) + valid example configs/shapes. Cross-referencing against the in-use set
  flags **`supported_unused`** ops (untapped optimizations).
- `tech_reports` — rationale / constraints / gotchas (the *why* and *when*).

**Entry schema** (one structured record per fusable pattern):

```
id, fused_op            # e.g. ttnn.experimental.nlp_create_qkv_heads
category                # attention.qkv | attention.sdpa | norm | rope | mlp.matmul_act | ccl | kv_cache ...
torch_pattern           # the canonical op-subsequence it replaces, as an fx-matchable op list
signature               # in/out dtypes, layout (TILE/RM), shape constraints (e.g. head_dim % 32 == 0)
config_template         # program/memory/compute-kernel config + which fields scale with which dims
                        #   (n_heads, n_kv, hidden, seq, core grid) — THE optimization payload
source                  # provenance: file:line in tt_transformers/tt_dit/demos, or unit-test path
usage_examples          # real call sites w/ concrete dims -> few-shot for the matcher
applicability_notes     # from tech_reports: when valid + gotchas (round KV to 512, HiFi4+fp32_dest_acc, ...)
status                  # in_use | supported_unused
accumulation_sensitive  # bool: does this op feed the AR feedback path (KV dtype, SDPA acc, norm, rope)?
```

**Build = three mining passes (hybrid):**
1. **Curated extraction** over `tt_transformers`/`tt_dit`/`demos`: locate fused-op call sites,
   capture the surrounding config construction + the torch-reference equivalent; LLM-assisted
   summarization into entries with config templates + provenance.
2. **Unit-test inventory** over `tests/ttnn/unit_tests/operations`: enumerate every op tested + the
   configs/shapes exercised → the supported set; flag `supported_unused`.
3. **tech_reports** pass: extract rationale/constraints → `applicability_notes`.

**Storage:** structured files under `kb/` (one record per pattern) + an index. The old
`repo_rag_tool` becomes the retriever (filter by category/op-shape; usage examples = few-shot).

### Layer B — Matcher, config transfer, structural gate

Per block of the new model:
1. **Reference + trace.** Build the PyTorch reference (the `reference` skill), `torch.export` /
   `make_fx` trace → canonical fine-grained op graph. **fx is not the matcher** — it is the
   ground-truth op graph the matcher matches *against*.
2. **Retrieve.** `repo_rag` pulls candidate KB entries by op-category + shape signature.
3. **LLM match + config transfer (primary).** The matcher (Anthropic API) proposes, for each
   subsequence: `fused_op` + a **concrete config parameterized from the donor's `config_template`
   to the new model's dims** + rationale + source provenance + the unfused fallback it replaces.
   The LLM is chosen as primary matcher specifically because **carrying the correct config across
   requires reading the donor's real code** — an fx subgraph-rewrite would insert the fused op but
   lose the config (the actual optimization). Uses **prompt caching** on the KB/few-shot context.
4. **fx structural gate (pre-device, cheap).** Verify each proposal maps to an **actual contiguous
   subsequence** in the traced graph, and the rewritten graph is **dataflow-equivalent** to the
   original (same input→output topology; nothing dropped/added/reordered semantically). Failures
   bounce back to the matcher with the structural error — they never reach the device.

### Layer C — Codegen + verification + repair loop (LangGraph)

5. **Codegen.** Emit the TTNN package: fused ops + transferred configs where matched, naive
   fallback where not. **Every fusion keeps its fallback** (needed for bisection).
6. **Per-block PCC.** Real ttnn execution on device; per-op trace tensors (from the instrumenter)
   give per-block PCC vs. golden.
7. **Repair loop (agentic).** On PCC < 0.99, localize the **culprit fusion** — by toggling
   individual fusions to their fallback (A/B) and/or per-op trace diff — then feed **structured
   diagnosis** (which op, measured rel_l2/PCC, the config tried) back to the matcher, which
   re-proposes using KB `applicability_notes` (e.g. needs `fp32_dest_acc` + HiFi4). Bounded by
   `max_iterations`; exhausted → handoff (see Execution model).
8. **Full PCC + AR coherence + long-decode drift gate.** Free-run to a horizon representative of
   the model's realistic output length (for thinking models, several K tokens), measuring the
   **drift trajectory**: token-match-rate decay, rolling/cumulative KL, `first_divergence_step`,
   **and the per-token degradation slope** — not just endpoint PCC. **Accumulation-aware
   attribution:** teacher-forced isolates per-step numerics; free-run exposes compounding. *TF
   passes but free-run diverges early ⇒ accumulation failure*, and repair targets only the
   `accumulation_sensitive` ops (KV dtype, SDPA acc precision, norm, rope), not single-forward
   fusions. Failures are attributed along **two axes**: per-block PCC (wrong *now*) and long-decode
   drift (wrong *over time*) — different culprits, different fixes.
9. **Perf gate.** Profile the traced path vs. a **naive baseline build (all fusions off)**.
   Confirms each transfer actually helped; a fusion that passes PCC but does not move perf is
   **logged, not silently kept**.
10. **Report.** Per-fusion provenance: which optimization came from which donor, PCC, perf delta.

The throughline: **the LLM proposes, fx proves structure, PCC proves numerics, the long-decode
gate proves stability over time, the profiler proves it was worth it** — failures route back to
the matcher with enough diagnosis to re-propose intelligently rather than blind-sweep configs.

## Execution model — autonomous, but Claude-Code-supervisable

Layer C is a **standalone autonomous framework with zero runtime dependency on Claude Code**. It
runs and scales unattended (batch over many models, CI, overnight) with the Anthropic API matcher.
Claude Code is an **external operator** used **on demand** when issues arise. This works because C
is transparent and resumable, not a black box:

1. **Persisted checkpoints.** LangGraph checkpointer writes state to disk (SQLite) after every
   node → inspect any step, **resume from a checkpoint** instead of re-running.
2. **Structured artifacts per node** to a run dir: proposed fusions + transferred configs (JSON),
   per-block PCC, trace tensors, long-decode drift curves, failure diagnosis → debug without
   re-running on device.
3. **Failure/exhaustion = clean handoff, not crash.** The `human` node surfaces a complete
   diagnosis bundle (per CLAUDE.md autonomous-mode: stop after N attempts on the same failure);
   Claude Code picks it up with the `debug` skill, fixes code/config/prompt/KB rule, resumes from
   the checkpoint.
4. **Optional breakpoints.** LangGraph `interrupt()` at risky points (before applying a fusion set,
   before a device run) for semi-attended runs.

The framework is a **normal living codebase Claude Code edits** — nodes, tools, KB rules, prompts
are plain Python/files; "changes in the flow" are code edits with full repo context. The
deterministic tools (fx, codegen, PCC harness, profiler) are **plain functions with unit tests**,
not graph nodes — so debugging is debugging a tested function, not a node.

## First vertical slice (the named first target)

Prove the whole loop end-to-end on one motivated case before investing in breadth/depth:

- **Model:** SeamlessM4T-v2, `SeamlessMHA` (`tt/seamless_mha.py`).
- **Fusion:** `nlp_create_qkv_heads` (+ `nlp_concat_heads` on the output) — replace the naive
  `linear → reshape → transpose` head-split (run ×3 for Q/K/V) and the mirror `transpose → reshape`
  on output.
- **Path proven:** KB entry → retrieve → LLM match + config transfer → fx structural gate →
  codegen → on-device PCC.
- **Why this one:**
  - Documented, measured perf motivation already in the code (~40% reshape cost; currently only
    L1-pinned, not fused) → clear before/after for the perf gate.
  - Donors exist (`tt_transformers` `nlp_create_qkv_heads(_decode)`/`nlp_concat_heads(_decode)`;
    Whisper decoder cited in the code) → KB will have config templates to transfer.
  - Ready harness: golden tensors (`reference/golden/seamless_mha.pt`) + unit test
    (`tests/test_tt_seamless_mha.py`) for the structural + numerical gates.
  - **Non-trivial nuance:** SeamlessMHA is BART-style with **separate** q/k/v projections
    (4 projections, `bias=True`), not a fused QKV matmul — so the matcher must select the
    **separate-Q/K/V head-split variant** and decide whether to also fuse the three projections.
    Exactly the config/variant judgment the LLM matcher + KB usage-examples is meant to resolve,
    with PCC as the backstop. A meaningful proof, not a trivial one.

## Scope decomposition

Three sub-projects; **build a thin vertical slice through all three first** (the named slice
above), then scale **breadth** (more rules) and **depth** (full repair loop, long-decode gate,
perf gate):

1. **KB builder** (Layer A) — mine the four sources. Validated by "contains the known fusions with
   correct config templates + the `supported_unused` set."
2. **Matcher + config-transfer + fx structural gate** (Layer B) — validated *offline* (proposes the
   right fusion + config for a known model, no device needed).
3. **Codegen + on-device verification + repair loop** (Layer C) — needs device + the LangGraph
   nodes wired.

The alternative (KB-first / waterfall) is rejected: it risks building a KB whose shape doesn't fit
what the matcher actually needs.

## Relationship to the existing scaffold (`agent.sh` / `tt_bringup_agent`)

This design *is* "completing the stubs" of the generated LangGraph agent, with the stubs mapped to
real components:

| Stub | Becomes |
|------|---------|
| `repo_rag_tool` | KB retriever (Layer A corpus) |
| `op_mapper_tool` | LLM matcher + config transfer (Layer B) |
| `instrument_model` / `tt_run_and_capture` | fx op-graph extractor + real ttnn execution + per-op trace goldens |
| `codegen` + `templates` | emit TTNN package from matched fused ops + configs |
| `build_runner` / `test_runner` / `profiler` | real subprocess build + pytest + tracy on the traced path |
| `accuracy` / `drift` / `autoregressive` / `trajectory_bisect` / `fix` nodes | PCC + long-decode drift verification + repair loop |
| `kv_cache_analyzer` / `ccl_*` | real analysis |
| **(new)** | **KB builder** (Layer A) — absent from the scaffold |

## Open questions / risks

- **fx tracing robustness** on models with data-dependent control flow — mitigated by tracing
  per-block and falling back to module-signature matching where a block won't trace.
- **`nlp_create_qkv_heads` variant coverage** for separate-projection (BART-style) attention —
  confirm the separate-Q/K/V head-split variant exists and its config knobs during the slice.
- **Culprit localization cost** when many fusions are applied — A/B fallback toggling is O(n)
  device runs worst case; per-op trace diff should localize most cases without full A/B.
- **Long-decode gate runtime** — multi-K-token free-runs are expensive; gate length should be
  configurable per model and the perf/drift runs should reuse the metal trace.
