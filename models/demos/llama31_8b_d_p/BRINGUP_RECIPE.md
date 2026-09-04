<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bring-up recipe — `models/demos/llama31_8b_d_p`

**Target:** a clean, functional, PCC-verified TTNN **prefill** implementation of Llama-3.x 8B
in `models/demos/llama31_8b_d_p`, module-by-module verified against a torch/HF reference,
with CCL living inside the modules, ending in integration with the model-agnostic
disaggregated-prefill engine (`models/demos/common/prefill/`).

**Who this is for:** an agent (Claude) executing the bring-up end to end. Read this file top to
bottom once, then execute phases **in order**. Do not skip a phase. Do not start a phase whose
predecessor's gates have not been recorded as `PASS`, `PASS-WITH-DEVIATION` + a `DEC`, or `BLOCKED`
with a `07_RISKS.md` entry **naming the later phase that owns it**. A `FAIL` stops everything.

**Non-goals for this iteration:** decode, performance optimisation, trace/2CQ, multi-galaxy
pipeline parallel, quantised weights. Functional correctness + cleanliness + tests only.

This document has one authority level. Everything in it is an instruction, except the closing
**Provenance** section, which is a changelog and is labelled as such.

---

## Start here (first 5 minutes)

```bash
cd /home/mstojkovic/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct
mkdir -p models/demos/llama31_8b_d_p/bringup_log/raw

# the four documents to read before writing any code (in this order)
sed -n '1,120p' models/demos/minimax_m3/README.md
sed -n '1,95p'  models/demos/gpt_oss_d_p/README.md
cat models/demos/minimax_m3/tt/dense_mlp.py                      # the Llama MLP template
cat models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py # the unit-test template
```

Then execute the phases below in the order of this map. Note that **P10 runs before P9** — the
numbering is the gate index's, the order is the execution order, and §Phase P9 says why.

| Phase | Produces | Gate(s) | Device |
|---|---|---|---|
| **P0** Model card | `bringup_log/00_MODEL_CARD.md`, package skeleton | `G-CARD` | none |
| **P1** Reference | `01_REFERENCE.md`, `tests/test_factory.py`, `conftest.py`, bundled `config.json` | `G-REF` | host |
| **P2** Survey | `02_SURVEY.md` (reuse-vs-write table) | `G-SURVEY` | none |
| **P3** Outline | `03_OUTLINE.md` (file tree + shapes) | `G-OUTLINE` | none |
| **P4** CCL plan | `04_CCL_PLAN.md` (collective placement + residual scheme) | `G-CCL-PLAN` | none |
| **P5** Modules | `tt/{config,ccl,rms_norm,rope,mlp}.py`, `tt/attention/*`, `utils/*` + unit tests | `G-MESH` `G-RMS` `G-ROPE` `G-MLP` `G-ATTN` `G-KV` | 1 card |
| **P6** Assembly | `tt/{layer,embedding,model_config,model}.py` + tests | `G-LAYER` `G-WEIGHTS` `G-MODEL` | 1 card |
| **P7** Chunked + golden | `tt/tt_prefill_runtime.py`, `scripts/*` | `G-CHUNK` `G-GOLDEN` `G-RUNTIME` | 1 card |
| **P8** Multi-device | collectives enabled, `tt/attention/dense_sp.py`, `tests/galaxy_prefill_kv_pcc.py` | `G-FABRIC-MATRIX` `G-KV-TP8` `G-SP-RING` `G-CHUNK-ATTN` `G-TP-PARITY` `G-RACE` `G-SEMAPHORE` `G-MESH-KV` `G-WEIGHTS`(ext) | mesh |
| **P10** Disagg prefill | `tt/runners/{adapters,manifests,kv_chunk_table}` + one line in `models/demos/common/prefill/adapter.py` | `G-ADAPTER` `G-REQUEST` `G-MOCK-MIG` `G-KV-TABLE` `G-LOOPBACK` | mesh |
| **P9** Cleanliness | `README.md`, lint clean | `G-CLEAN` | none |

Keep this checklist in `bringup_log/06_GATES.md` and tick it as you go. If you must stop early, stop
**on a gate boundary** and leave `06_GATES.md` stating exactly which phase is next.

---

## The machine this recipe was written for (verified — do not re-derive)

- Hardware: **Blackhole Galaxy, 32 devices**. `ttnn.get_num_devices() == 32`,
  `ttnn.get_arch_name() == 'blackhole'`. `build/` + `python_env/` work; ttnn opens and closes the
  cluster cleanly.
- The compute grid is **(12, 10)**, *not* 8×8. Two different grids depend on that and they pull in
  opposite directions; read P4 (the CCL core grid, derived) and P5.5 (the SDPA program grid, pinned)
  before copying any grid size out of a template.
- `transformers==5.12.1`, `huggingface_hub==1.16.1`. Version-specific traps: P1.
- **Real weights are staged** at `/home/mstojkovic/models/Llama-3.1-8B-Instruct` (15 GB, 4
  safetensors shards + tokenizer + config). `export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct`.
  So `G-WEIGHTS` (real half), `G-MODEL` top-1, `G-GOLDEN`, `G-CHUNK`/`G-MESH-KV`-vs-golden,
  `G-REQUEST` and `G-MOCK-MIG` are all **runnable**; none of them may be recorded `BLOCKED` for
  lack of weights. Keep a `requires_hf_reference` skip marker anyway so the suite still runs on a
  weightless machine.
- Network + HF access are available; `meta-llama/Llama-3.1-8B-Instruct` is reachable (gated repo,
  org token in the environment). The live `config.json` byte-matches the bundled
  `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json` dims.
- BH Galaxy mesh descriptors live in `tt_metal/fabric/mesh_graph_descriptors/` — e.g.
  `bh_galaxy_sp4_torus_xy_graph_descriptor.textproto`,
  `32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`. The Ring topology P8 needs the torus
  descriptor; a Ring topology on a plain `FABRIC_1D` fabric **hangs** rather than erroring.
- **Never write the HF token into any file, log, or agent prompt.** It is read from the environment
  only.

If you are porting this recipe to a LoudBox / T3K / N300, the machine-specific parts are: the
`(4,8)` target, the submesh rule in P8 step 1, the 8×8 SDPA grid vs the (12,10) compute grid, and
`num_links`. Everything else transfers.

---

## 0. The agent contract (read first)

Five rules govern the whole run. They are not advice.

1. **Sequential.** Phases in the order of the map above (P0 → P8, then P10, then P9). Each phase
   ends in a *gate*. A gate is a command with a numeric threshold and a recorded verdict. No
   forward progress on a `FAIL`. A `BLOCKED` gate may be carried forward only if a `07_RISKS.md`
   entry names the phase that will run it (P7 has one such gate by construction; see P7).
2. **Every judgement call is logged.** Anything you *decide* rather than *read* becomes a `DEC-NNN`
   entry in `bringup_log/05_DECISIONS.md` (§1.3), and the code that embodies it carries a
   `# DEC-NNN` comment. If a reviewer cannot reconstruct *why* a number or a pattern was chosen from
   the logs alone, the logging failed.
3. **Provenance over memory.** Never write a model dimension, a threshold, or an op name from
   recall. Read it from `config.json`, from a repo file (cite `path:line`), or derive it and show
   the formula. Anything you could not verify is marked `UNVERIFIED` and listed in
   `bringup_log/07_RISKS.md`.
4. **Reuse before writing.** Before implementing anything, find whether it already exists in
   `models/demos/minimax_m3`, `models/demos/gpt_oss_d_p`, `models/demos/deepseek_v3_d_p`,
   `models/tt_transformers`, or `models/demos/common/`. Reuse means *import*, not copy-paste. A
   copy-paste is a `DEC` with a justification.
5. **Clean means clean.** No dead code, no commented-out experiments, no un-documented env vars, no
   `try/except: pass`, no torch fallback on a device path without an accompanying GitHub-issue note
   in `07_RISKS.md`. Every module gets a docstring that names its HF anchor.

### 0.1 Ground rules for the environment

```bash
cd /home/mstojkovic/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
```

- `HF_MODEL` points at the checkpoint dir (safetensors + `config.json` + tokenizer).
- Single-card work (P0–P7) needs **one** Wormhole/Blackhole card. Multi-device work (P8+) needs the
  real mesh.
- Never `git commit` unless explicitly asked. Never push. Work on the current branch.

### 0.2 Orchestration hygiene — do not mutate the worktree while a phase session is live

If the bring-up is driven by an orchestrator with per-phase sessions, three failure modes come from
one cause: committing or renaming while a phase session is still working in the same worktree.
Committing the moment commits were authorised put a P7 file into a commit whose message says
"P0-P6"; renaming the package mid-P7 broke that session's own path references; renaming the
golden-trace directory did the same to `$PREFILL_TRACE_DIR`. All were recoverable. All were
avoidable. The subtle one is the fourth: after the rename the orchestrator verified imports,
citations and a 23-test smoke run — and declared it clean **without re-running the finished phase's
gates**, so P7's whole evidence base briefly existed only against paths no longer in the tree.

**Rules:**

1. **Commit only at a phase boundary, with no session live.** "Commits are authorised" is not
   "commit right now". Check for a running session first.
2. **Never rename, move, or restructure while a session is live.** A rename is the worst case: it
   invalidates in-flight path references silently and rewrites the provenance of raw logs.
3. **After any path-affecting change, re-run the previous phase's *gates*** — not a smoke test.
   Import checks and a citation pass prove the tree is wired; they prove nothing about the gates.
4. **Raw logs from before a rename keep the old path.** That is correct, not stale: they record what
   actually ran. Rewriting them would make the evidence less trustworthy. Note the boundary in the
   ledger instead.
5. If a mid-phase mutation is genuinely unavoidable, **message the live session** with the exact
   change before making it, rather than letting it discover the breakage.

---

## 1. Logging protocol — the deliverable that makes this iteration evaluable

Create the log root **before Phase 0 does anything else**:

```bash
mkdir -p models/demos/llama31_8b_d_p/bringup_log/raw
```

### 1.1 Files

| File | Purpose | Written in |
|---|---|---|
| `bringup_log/00_MODEL_CARD.md` | Every architectural fact + its provenance | P0 |
| `bringup_log/01_REFERENCE.md` | What the reference is, how it is invoked, how it was validated | P1 |
| `bringup_log/02_SURVEY.md` | Repo survey: reuse-vs-write-fresh table with `path:line` citations | P2 |
| `bringup_log/03_OUTLINE.md` | The planned file tree + per-file contract (interfaces, shapes) | P3 |
| `bringup_log/04_CCL_PLAN.md` | Parallelism map + every collective in the model, by module | P4 |
| `bringup_log/05_DECISIONS.md` | **Append-only** `DEC-NNN` decision log | all phases |
| `bringup_log/06_GATES.md` | **Append-only** gate ledger: command, threshold, measured, verdict | all phases |
| `bringup_log/07_RISKS.md` | Open questions, `UNVERIFIED` facts, known gaps, filed issues | all phases |
| `bringup_log/08_PREFILL_INTEGRATION.md` | Adapter/runtime contract mapping + gate transcripts | P10 |
| `bringup_log/raw/<GATE-ID>_<UTC-timestamp>.log` | Verbatim stdout of every gate command | all phases |

Rules for all of them: plain Markdown, no ANSI escapes, tables aligned, ISO-8601 UTC dates,
numbers with the precision they were measured at (PCC to 5 decimals). Append-only files are never
rewritten — a superseded entry gets a new entry that says `Supersedes DEC-012`.

**Write the logs as you go, never at the end.** Create all nine files with their headings in P0 (a
one-line `_(pending)_` body is fine), then fill each one *during* its phase and append each `DEC` at
the moment the decision is made. A log reconstructed afterwards from memory records what you
remember choosing, not what you chose — which defeats the only purpose it has.

### 1.2 Raw-output rule

Every gate command runs under `tee`:

```bash
G=G-MLP; TS=$(date -u +%Y%m%dT%H%M%SZ)
pytest models/demos/llama31_8b_d_p/tests/unit/test_mlp_vs_ref.py -x -q 2>&1 \
  | tee models/demos/llama31_8b_d_p/bringup_log/raw/${G}_${TS}.log
```

The ledger entry then cites the raw filename. A gate with no raw log did not happen.

### 1.3 Decision entry template (`05_DECISIONS.md`)

Copy verbatim, one block per decision, numbered monotonically from `DEC-001`.

```markdown
### DEC-014 — Fuse Q/K/V projections into one matmul?
- **Phase / module:** P5 / attention
- **Date (UTC):** 2026-09-03
- **Trigger:** implementing `tt/attention/weights.py`; the source patterns disagree.
- **Question:** load `q_proj`/`k_proj`/`v_proj` as three column-parallel weights, or pre-fuse into
  one `[hidden, (nq+2*nkv)*head_dim]` weight and split heads on device?
- **Options considered:**
  1. Three separate matmuls — simplest, matches `models/demos/gpt_oss_d_p/tt/attention/weights.py:NN`; 3 matmul
     launches per layer.
  2. Fused QKV — one matmul, needs `ttnn.experimental.nlp_create_qkv_heads`; matches
     `models/tt_transformers/tt/attention.py:NN` and `load_checkpoints.fuse_qkv_meta:494`.
- **Choice:** option 1.
- **Why:** this iteration is functional-first; the fused path adds a head-split op whose GQA
  (32 Q / 8 KV) layout is a second thing to debug before any PCC exists. Option 2 is the perf
  follow-up.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/attention/weights.py:1-209` (three-weight pattern,
  PCC-verified at 0.99 by `tests/unit/test_attention_vs_ref.py`).
- **Confidence:** high.
- **Falsifier:** if `G-ATTN` passes but per-layer wall-clock is dominated by matmul launch overhead,
  the choice was wrong for the next iteration (not for this one).
- **Revisit if:** perf work starts, or a fused kernel is required by the SP ring-SDPA path.
- **Blast radius:** `tt/attention/weights.py`, `tt/attention/prefill.py`, `G-ATTN`.
```

**A `DEC` is mandatory when you:**

- pick a number that is not read verbatim from `config.json` (tile padding, chunk size, `num_links`,
  dtype, program-config block sizes, PCC threshold);
- choose between two or more existing repo patterns;
- deviate from this recipe in any way;
- lower a PCC threshold or mark a sub-case `xfail`/`skip`;
- introduce an environment variable (also document it in the package `README.md`);
- leave something unimplemented, stubbed, or torch-fallback;
- discover the reference and the repo disagree.

### 1.4 Gate ledger entry template (`06_GATES.md`)

A summary table at the top (append a row per gate), then one detail block per gate below it.

```markdown
| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| G-RMS | P5 | RMSNorm vs torch, 1x1 mesh | PCC ≥ 0.9999 | 0.9999955 (floor 0.9999986) | PASS | 2026-09-03 | `raw/G-RMS_20260903T101500Z.log` |
```

```markdown
### G-RMS — RMSNorm vs torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_rms_norm_vs_ref.py -x -q`
- **Mesh / device:** (1,1), Blackhole
- **Inputs:** seq_len ∈ {32, 512, 4096}, hidden 4096, **standard-normal** inputs, real layer-0 norm
  weights, seed 0
- **Reference dtype policy:** weights and inputs rounded to the device dtype, all remaining math in
  fp32 (§2.2). The reference weight is **fp32**, not bf16.
- **Threshold:** PCC ≥ 0.9999, with the gap to the computed noise floor recorded (source: §2,
  Appendix A)
- **Noise floor (computed):** 0.9999986
- **Measured:** 0.9999955 / 0.9999955 / 0.9999955 — ~3x the floor
- **Verdict:** PASS
- **Negative control:** zero-gain probe, `max|out| = 0.0`
- **Deviations:** none
- **Notes:** `eps` read from `config.json:rms_norm_eps` = 1e-05. No `+1` weight fold (Llama is a
  plain RMSNorm, unlike Gemma) — see `DEC-004`.
```

Verdicts are exactly one of: `PASS`, `FAIL`, `PASS-WITH-DEVIATION` (requires a `DEC`), `BLOCKED`
(requires a `07_RISKS.md` entry naming the blocker), `NOT-RUN` (requires the reason).

**For a gate that produces a number, four fields are not optional and a block missing any of them
is incomplete:** the **input distribution**, the **reference's dtype policy**, the **computed noise
floor**, and a **negative control** — a deliberately wrong variant that the same assertion must
reject. A test that only ever sees the correct input cannot tell you it is measuring anything. An
op or a configuration that must *refuse* counts as a control (`G-MESH`'s sub-axis TP, `G-SP-RING`'s
`fp32_dest_acc_en=True`, `G-RUNTIME`'s nine refusals); a bit-identity comparison across repeated
runs counts as its own (`G-RACE`). §2.5 explains why a *PCC-based* control is sometimes not strong
enough. The four doc-review gates (`G-CARD`, `G-SURVEY`, `G-OUTLINE`, `G-CCL-PLAN`) have no numbers
and are exempt; every other gate in Appendix A carries all four.

### 1.5 Progress checkpoint

After each phase, append to `06_GATES.md` a two-line status:

```
STATUS after P5: gates PASS=6 FAIL=0 DEVIATION=1 BLOCKED=0 | next: P6 (layer assembly)
Open DECs needing review: DEC-009 (bf8_b KV dtype), DEC-011 (o_proj reduce-scatter vs all-reduce)
```

### 1.6 Verify your own citations mechanically

Write `models/demos/llama31_8b_d_p/scripts/verify_citations.py` in P0 and **extend it in every
phase**: a list of `(path, line, substring_that_must_be_on_that_line)` triples plus a pass that
re-resolves every backtick-quoted `path:line` in the logs, the recipe, the README and the package's
own docstrings. Run it as part of every doc gate.

It is not busywork. On its first run it caught five wrong `path:line` refs in this document and
five in P0's own output, and it went on finding more in every later phase. **An unverified `path:line` is
worse than no citation, because it reads as authoritative.** Two things it must handle: abbreviated
refs (a bare basename continuing an earlier full citation), and *citation shadowing* — once the
package has its own `config.py`, `model.py` and `layer.py`, a bare `model.py:211` is genuinely
ambiguous, and the resolver must either resolve package-local first or require the line to be in
range for every candidate.

---

## 2. How to set a threshold — read this before writing a single gate

Every numeric threshold in this recipe was set by the method in this section. If you change a
threshold, change it by this method and log the measurement.

### 2.1 Do **not** gate against another implementation's PCC

The obvious plan — "measure what `models/tt_transformers`' Llama gets on this box, then set your
threshold from it" — is wrong, and it is wrong in the direction that ratifies degraded modules. It
was tried and killed by two device A/B runs.

**(a) Cross-implementation PCCs are not comparable, because the *reference* precision differs.**
`models/tt_transformers/tests/test_rms_norm.py:77` builds its reference via `reference_rms_norm()`,
whose HF weights load at `torch_dtype: bfloat16` (from `config.json`), so HF's
`self.weight * hidden_states` multiplies by a **bf16-rounded** weight. That reference *shares the
device's own rounding*, which inflates the PCC. A reference built with an fp32 weight is strictly
harder. Measured on identical inputs and real layer-0 norm weights: the fp32-weight reference gives
**0.99995** where the oracle's bf16-weight reference reports **0.9999867** — the same device output,
two incomparable numbers.

**(b) Input distribution is a red herring** — it is the tempting explanation for the gap above, and
it is wrong. The torch bf16 noise floor is **0.9999986 under `rand[0,1)` and 0.9999986 under
`randn`** — identical. Distribution shifts the measured value in the 5th decimal; it does not move
the floor, and it must never be used to pick whichever distribution passes. State the distribution
in the gate block and prefer standard-normal, which is the harder of the two for a norm.

The measurement *is* still worth doing once, for two reasons that are not thresholds: it proves the
HF reference accessors work on this transformers version, and it shows how loose a guessed threshold
can be. Four `models/tt_transformers` oracles, real Llama-3.1-8B-Instruct weights, single Blackhole
card:

| Oracle | Measured PCC | The guess this replaced | Gate now (Appendix A) |
|---|---|---|---|
| `tests/test_rms_norm.py` | 0.9999867 / 0.9999886 | `G-RMS` ≥ 0.999 | **≥ 0.9999** |
| `tests/test_mlp.py` (seq 512, `bfloat8_b`) | 0.9995823 | `G-MLP` ≥ 0.99 @bf8_b | **≥ 0.999 @bf8_b** |
| `tests/test_attention_prefill.py` | 0.9996099 / 0.9996010 | `G-ATTN` ≥ 0.99 | **≥ 0.999** |
| `tests/test_decoder_prefill.py` | 0.9999985 / 0.9999981 | `G-LAYER` ≥ 0.99 | **≥ 0.999** |

Every original threshold was one to two orders of magnitude too loose: a fresh module landing at
0.99 would have been recorded `PASS` while sitting ~40x further from the reference than an existing
implementation on the same ops, dtype and silicon. Treat the band between a gate and the number a
correct kernel can actually reach as *investigate*, never *pass*.

### 2.2 The method that works: gate on the gap to the **noise floor**

For each gate, compute the floor in torch: **round inputs and weights to the device dtype, do all
remaining math in fp32, and PCC that against the fp32 reference.** That number is
implementation-independent, distribution-stable, and is the best any correct kernel can do.

Then record three things and gate on the **gap**:

| record | meaning |
|---|---|
| `floor` | torch noise floor for this dtype and shape — computed, never guessed. bf8_b floors are far lower than bf16 |
| `measured` | your module on device |
| `gap` | `error_ratio = (1 - measured) / (1 - floor)` |

At the floor is a pass. **20x+ off the floor is a finding even when the absolute PCC looks
pretty** — that is exactly the band a threshold copied from a README waves through. Keep the
absolute thresholds in Appendix A as floors you must clear, but clearing one while sitting far off
the noise floor gets investigated, not recorded as a clean `PASS`.

The per-stage budgets this recipe uses (Appendix A): stages you implement yourself ≤ **3x** their
floor; a whole block ≤ **8x**; the depth-accumulation step between consecutive layers ≤ **4x**.
Those are ceilings set before the measurements existed, and none of them was refitted afterwards.
**Set a threshold before you look at the number it will be compared to** — a threshold chosen after
the measurement cannot fail.

**Keep one definition of the floor helpers.** `quantize_like_device()` and `err_ratio()` belong in
`tests/test_factory.py`, imported everywhere; two copies drift and then two gates disagree about
what a floor is.

### 2.3 The limit of the floor model: it does not describe a **fused** kernel's interior

The floor model above is only valid for ops whose interior arithmetic you can mirror. It breaks on
fused kernels, and you must expect that rather than debug it:

`ttnn.transformer.scaled_dot_product_attention` alone (bf16 Q/K/V, GQA 32/8, `head_dim` 128)
measures **0.9999204** against a modelled floor of **0.9999989** — a **71x** gap. Sweeping
`q_chunk`/`k_chunk` over {32,128,256} moves it by under 4%; `exp_approx_mode` not at all. And that
one term is the *entire* block-level gap of `G-ATTN`: every stage implemented in this package
measures **1.00–1.47x** of its own floor. (The SP ring op is better but not free: **7.98x**.)

So do not read a large block-level gap as "our code is wrong" before isolating the fused kernel.
The sanctioned handling: **separate error budgets per stage**, measured rather than assumed, plus a
**permanent standalone probe** of the fused kernel so its slack is *named and tracked*. A budget
that lumps the kernel's 71x in with our stages' 1.5x can absorb a real regression without anyone
noticing.

### 2.4 Always pass an explicit `compute_kernel_config` — and `fp32_dest_acc_en` polarity is per-op

Build the config once via `ttnn.init_device_compute_kernel_config(...)` and pass it to **every** op
that accepts one: each `ttnn.linear`, the SDPA call, the norms. Passing nothing is a silent
precision regression, and every module in the templates that omits one inherits it.

**The op on its own** — `ttnn.rms_norm`, this box, fp32-weight reference, hidden 4096, real
weights. (This is a different experiment from `G-RMS`, which scores the whole module; the op-level
A/B is the one that isolates the flag.)

| `compute_kernel_config` | PCC (rand 32/512) | PCC (randn 32/512) |
|---|---|---|
| **none** | 0.9999440 / 0.9999531 | 0.9999652 / 0.9999648 |
| HiFi2, `fp32_dest_acc_en=False` | 0.9999369 / 0.9999407 | 0.9999607 / 0.9999590 |
| **HiFi4, `fp32_dest_acc_en=True`** | **0.9999969 / 0.9999968** | **0.9999971 / 0.9999971** |
| torch bf16 floor | 0.9999986 / 0.9999987 | 0.9999986 |

`MathFidelity` alone is a **no-op** here — HiFi2 is marginally *worse* than the default.
`fp32_dest_acc_en=True` removes ~25x of the error and lands within ~2x of the floor. At the
*module* level the same change moves `G-RMS` from 0.9999697 to 0.9999955 — ~7x — because the module
is more than that one op (P5.2).

But the *default* for that flag differs by op, so "copy the template's config" is wrong in both
directions:

| op | no config | `=True` | `=False` |
|---|---|---|---|
| `ttnn.rms_norm` | 0.9999652 | **0.9999971** | 0.9999607 |
| `ttnn.linear`, bf8_b weights | 0.9999143 | **0.9999143** (bit-identical) | **0.9925392** (96x worse) |
| `ttnn.linear`, bf16 weights | 0.9999852 | **0.9999852** (bit-identical) | **0.9917529** (1168x worse) |
| attention block, bf8_b | — | **0.9997449** | **0.9963324** (38.7x worse) |
| attention block, bf16 | — | **0.9998033** | **0.9959098** (107.6x worse) |

The matmul's own default **already enables** fp32 accumulation — the opposite of the norm. So the
danger is not omitting the flag on a matmul; it is **carrying
`models/demos/gpt_oss_d_p/tt/attention/config.py:71`'s explicit `fp32_dest_acc_en=False` forward**,
which costs two to three orders of magnitude.

**Note precisely what caught this,** because it is the whole argument for §2.2: the degraded
attention block scores **0.9963** at bf8_b. That **clears a 0.99 gate** — the plausible number, and
it would have been logged a clean `PASS` — and **fails the 0.999 gate §2.1 arrives at**. The
tightened threshold caught it; the floor comparison then explained *why* in one measurement instead
of a debugging session.

Make `fp32_dest_acc_en=True` the package default, pass it explicitly everywhere, and A/B it in-suite
so a regression shows up as a number rather than a mystery. Where it is refused or costs accuracy,
log a `DEC` with both measurements — never drop it silently. **There is exactly one op in this model
where `False` is mandatory rather than a preference:** the SP ring SDPA, see P8.

Two related facts, so you do not go looking:

- **`ttnn.BlackholeComputeKernelConfig` does not exist.** `hasattr(ttnn, "BlackholeComputeKernelConfig")`
  is `False` (`ttnn/ttnn/__init__.py:305` exports only the Wormhole name) and where it is defined it
  is the *same object* (`ttnn/ttnn/types.py:61`). An "arch branch to pick the kernel-config class" is
  a no-op. Use `ttnn.WormholeComputeKernelConfig` on Blackhole — the name is misleading, not wrong
  (`models/demos/gpt_oss_d_p/tt/attention/config.py:103`, rename tracked as issue #51998) — or better,
  the `ttnn.init_device_compute_kernel_config` factory. Fields: `math_fidelity`, `math_approx_mode`,
  `fp32_dest_acc_en`, `packer_l1_acc`.

### 2.5 Two numerical traps that make a probe or a control lie

- **`bfloat16` is exact only up to 256.** A positional read-back probe using integer position ids at
  `max_seq_len=384` failed with *greatest relative difference 1/257*: **257 is not representable in
  bf16 and rounds to 256**, so 64 of 384 rows "mismatched" while the cache was perfectly correct.
  Any probe that encodes indices, positions or ids as tensor *values* must keep every value ≤ 256,
  or split the id across lanes. (The fix here: 4 chunks of 64 with the head id in its own lane
  block, which also covers more `kv_actual` offsets than a single 3×128 sweep would.) **A failing probe is
  not evidence of a failing module until the probe's own numerics are checked.**
- **A PCC-based negative control is too weak for a layout bug.** The `G-KV-TP8` control reads mesh
  column `(c+1)%8` as KV head `c` — a completely wrong head→column map — and still scores
  **PCC 0.99890**, high *by construction* because half the probe's lanes carry the head-independent
  position. Correlation does not notice a permutation that preserves most of the distribution.
  **Gate every layout, mapping or address claim on bit-equality** (`torch.equal`, `rtol=atol=0`),
  not on PCC. The same control at model level, scored against the golden, does collapse
  (**-0.03809**) — but only because 32 layers of arithmetic amplified it, which is the wrong
  instrument to rely on for a one-hop mapping.

---

## Phase P0 — Model card: pin down *exactly* what is being built

**Goal:** a single table where every architectural fact has a value and a source. This is what
prevents an entire bring-up from being built on a mis-remembered head count.

### Steps

1. Create the log root (§1), **all nine log files with their top-level headings**, and
   `models/demos/llama31_8b_d_p/{tt,tests/unit,scripts,docs}` with `__init__.py` files
   (**no `reference/`** — see P1: Llama does not need a vendored torch reference, so do not create an
   empty package for it)
   carrying the SPDX header (copy the three-line header style from
   `models/demos/gpt_oss_d_p/tt/__init__.py`). Write `scripts/verify_citations.py` (§1.6) here too.
2. **Resolve the model identity.** `llama31_8b` is a directory name, not a HuggingFace id. Determine
   the exact checkpoint:
   - `echo $HF_MODEL` and `cat $HF_MODEL/config.json` — on this machine the checkpoint is staged
     (see §The machine), and its `config.json` byte-matches the bundled
     `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json` dims, so the card's
     dims are confirmed against the real checkpoint rather than against a stand-in.
   - Bundle the resolved `config.json` verbatim in the package and assert byte-identity in a test,
     so dimension-only tests need neither network nor checkpoint.
   - Write a `DEC` naming the resolved repo id / local path and how you resolved it. **If the
     intended checkpoint is ever ambiguous (e.g. no public "Llama-3.2-8B" exists), say so explicitly
     in the card and in `07_RISKS.md`, proceed on the Llama-3.1-8B-Instruct dims, and flag it as the
     single assumption the user must confirm.** Do not stall on it.
3. Fill the card. Every row needs `Source`. Expected values for Llama-3.1-8B-Instruct — **confirm
   each against the checkpoint you resolved and log any mismatch as a `DEC`:**

| Fact | Expected | Source key |
|---|---|---|
| architecture | `LlamaForCausalLM` | `architectures[0]` |
| layers | 32 | `num_hidden_layers` |
| hidden | 4096 | `hidden_size` |
| FFN intermediate | 14336 | `intermediate_size` |
| activation | `silu` (SwiGLU: `down(silu(gate(x)) * up(x))`) | `hidden_act` |
| Q heads | 32 | `num_attention_heads` |
| KV heads | 8 → **GQA, group = 4** | `num_key_value_heads` |
| head_dim | 128 (derived: `hidden/num_attention_heads`) | derived |
| norm | RMSNorm, eps 1e-05, **plain** (no `+1` fold) | `rms_norm_eps` |
| RoPE | θ = 500000.0, **full rotary** (rotary_dim = head_dim) | `rope_theta` **in the JSON file** — but a live `transformers` config object has no such attribute and `getattr(cfg, "rope_theta", …)` returns your default instead; see P1 trap 1 |
| RoPE scaling | `llama3`: factor 8.0, low 1.0, high 4.0, orig_max_pos 8192 | `rope_scaling` |
| max positions | 131072 | `max_position_embeddings` |
| vocab | 128256 | `vocab_size` |
| attention bias | false | `attention_bias` |
| MLP bias | false | `mlp_bias` |
| tied embeddings | false | `tie_word_embeddings` |
| QK-norm | none | absent from config |
| attention sinks | none | absent from config |
| sliding window | none (all layers full-causal) | absent from config |
| MoE | none — **dense FFN on every layer** | absent from config |

4. Add a **"what this model does NOT have"** section to the card. It is as load-bearing as the
   positives, because the closest in-repo templates (`gpt_oss_d_p`, `minimax_m3`) *do* have those
   features and copying them in is the most likely source of wasted work:
   *no MoE / router / expert-parallelism, no attention sinks, no sliding-window alternation, no
   QK-norm, no partial RoPE, no MLA, no sparse attention, no biases anywhere.*
   Llama 8B is the **simplest** shape in this family — dense MLP + GQA + full RoPE.
5. Record the deployment target: mesh shape, TP, SP. Derive, do not guess:
   - **`TP` must equal `num_key_value_heads` = 8.** This is a hard equality, not a bound. The
     packed KV cache allocates **exactly one KV head per chip**
     (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:95-99` hard-codes the `1`, commented *"Per-chip
     cache is one head"*). At any smaller TP the model produces more local KV heads than the cache
     slot holds and `update_padded_kv_cache` dies with
     `TT_FATAL: cache and input num-heads dim must match`
     (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`) — **including for chunk 0**, so on a single
     card no model-level KV write is possible at all. At TP > 8 you would have to replicate KV heads,
     which needs its own `DEC`. SDPA itself is satisfied either way:
     `TT_FATAL(nqh >= nkv && nqh % nkv == 0)` (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:97-101`) gives
     `4 >= 1 && 4 % 1 == 0` at TP=8.
   - TP must divide `hidden` (4096) and `intermediate` (14336) tile-aligned: `14336/8 = 1792` and
     `4096/8 = 512`, both multiples of 32. Show the arithmetic.
   - SP = the other mesh axis. `CHUNK_SIZE % (SP*32) == 0` and `MAX_SEQ_LEN % CHUNK_SIZE == 0`
     (source: `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` "Shared setup").
   - Write the chosen `(mesh_shape, tp, sp)` and the arithmetic into the card **and** `04_CCL_PLAN.md`.
     For this machine that is `(4, 8)`, TP=8 on the columns, SP=4 on the rows.

**The consequence of the TP=8 equality is a gate-design rule, not a footnote.** A single-card
`(1,1)` run has `nkv = tp = 1`, a head count *the model never produces on the deployment mesh*. So
`G-KV` at `(1,1)` is a valid test of the cache **primitive** and of the `head_dim=128` geometry, and
is **not** a test of the model → cache path. P8 owns that (`G-KV-TP8`). Generalise it: **a gate that
passes on a mesh the deployment never uses can be testing a configuration the model cannot
produce** — say so in the gate block rather than letting the `PASS` imply more than it covers.

### Gate `G-CARD`

- **PASS when:** every row of the card has a non-empty `Source`; zero rows say "from memory"; the
  "does NOT have" section exists; the `(mesh, TP, SP)` arithmetic is shown, including the
  `TP == num_key_value_heads` derivation; every `UNVERIFIED` row also appears in `07_RISKS.md`;
  `verify_citations.py` is clean.
- **Command:** document review + `python models/demos/llama31_8b_d_p/scripts/verify_citations.py`.
  Record the verdict and the list of `UNVERIFIED` rows.

---

## Phase P1 — Reference implementation

**Goal:** a torch oracle you can call per-module and per-model, deterministic, cheap, and *known
correct*.

### Decide the reference strategy (in this order of preference)

1. **HF `transformers` `LlamaForCausalLM` directly.** Llama is first-class in `transformers` (no
   `trust_remote_code`). This is the strongly preferred option — nothing to vendor, nothing to keep
   in sync. `models/tt_transformers/tt/model_config.py` already exposes per-module accessors built
   on it: `reference_transformer` (`:4037`), `reference_decoder` (`:4393`),
   `reference_attention` (`:4410`), `reference_mlp` (`:4365`), `reference_rms_norm` (`:4167`),
   `reference_embedding` (`:4379`), `reference_lm_head` (`:4027`). These are the canonical llama
   oracles in this repo and `models/tt_transformers/tests/test_mlp.py` shows the exact usage.
2. **A self-contained `reference/model.py`** in the package — NOT used for Llama (no `reference/`
   package is created at all; see the P0 skeleton) — the `minimax_m3`/`gpt_oss_d_p` pattern
   (`models/demos/minimax_m3/reference/model.py`, `models/demos/gpt_oss_d_p/reference/model.py`).
   Only needed when HF cannot load the checkpoint (M3's case: a VL package shipping no modeling
   code). **Llama does not need this** — if you write one anyway, that is a `DEC` with a reason.
3. Per-test hand-written math (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` writes its own
   attention in ~40 lines and drives both sides from identical random weights). Use this **in
   addition** for module tests: it removes the HF-load cost from the inner loop and makes the test
   runnable on a bare card with no checkpoint.

**Take this combination:** *hand-written torch math inside each unit test, driven by identical
random weights* (fast, no checkpoint, runs anywhere) **plus** *HF for the layer/model-level tests
with real weights*. Log it as a `DEC`. Note that `ModelArgs.reference_*` **raises without
`HF_MODEL`** (`models/tt_transformers/tt/model_config.py:702`); weights are staged here so option 1
works, but the in-test torch math is still the better oracle for P5/P6 — no checkpoint, faster, and
gate-validated against HF at `G-REF`.

### `transformers` 5.12.1 — five traps, one of which is silent and fatal

1. **`rope_theta` is not an attribute, and `getattr` with a default is a trap.** Measured on this
   box:

   | expression | actual result on transformers 5.12.1 |
   |---|---|
   | `cfg.rope_theta` | **raises `AttributeError`** |
   | `getattr(cfg, "rope_theta", DEFAULT)` | **returns `DEFAULT`** — silently substitutes a wrong theta |
   | `cfg.rope_scaling` | a full dict, and it **contains** `rope_theta: 500000.0` |
   | `cfg.to_dict()` | has neither key — only `rope_parameters` |

   So the `getattr(cfg, "rope_theta", DEFAULT)` pattern at
   `models/demos/gpt_oss_d_p/tt/model_config.py:76` and
   `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:185` does not fail loudly here — it
   **succeeds with a hard-coded theta** (10000.0 against Llama's 500000.0), giving a RoPE that is
   wrong at every position, with no exception anywhere. **This is the highest-severity
   silent-wrongness trap in the whole bring-up.** Route theta and scaling through
   `models/tt_transformers/tt/common.py:165` `get_rope_theta` and `:183` `get_rope_scaling`. Both
   take a **dict**, and it must be the **raw `config.json`** you loaded and bundled in P0 — *not*
   `cfg.to_dict()`, which is the tempting one and is exactly the dict the table above says has
   neither key. Read them in exactly **one** place and assert non-`None`.
2. **Decide dict-vs-object for `hf_config` explicitly and hold it.** The templates pass an object
   (`models/demos/minimax_m3/tt/dense_mlp.py:47` does `hf_config.hidden_size`); a
   `config.json`-loading helper returns a dict; `get_rope_theta` wants a dict. A silent mix is how
   `None` dims get in. Normalise once, in one constructor.
3. **Torch references must set `cfg._attn_implementation = "eager"` *and* pass an explicit causal
   mask.** `eager_attention_forward` applies only the mask handed to it, so `attention_mask=None`
   yields **non-causal** attention, silently. Assert causality directly: perturb the last token and
   check rows `[:-1]` are unchanged at `max|Δ| = 0`.
4. **`get_rot_transformation_mat(dhead=32)` ignores its argument** —
   `models/tt_transformers/tt/common.py:564` hard-codes 32. Call it with no args.
5. **The HF reference wrappers branch on a forward signature.** `HfAttentionWrapper` /
   `HfDecoderWrapper` in `models/tt_transformers/tt/model_config.py` check whether
   `position_embeddings` is in the layer's forward signature (`reference_attention:4410`,
   `reference_decoder:4393`). Confirm that branch resolves correctly for
   `LlamaAttention`/`LlamaDecoderLayer` before trusting a low PCC — a wrapper feeding RoPE twice, or
   not at all, looks exactly like a model bug.

### Steps

1. Write `bringup_log/01_REFERENCE.md`: which option, how to invoke it, its dtype policy
   (compute the reference in **fp32**, cast only at the comparison boundary — see
   `scripts/generate_golden_kv_cache.py` header in `minimax_m3`), and its determinism check.
   The dtype policy is not a detail: §2.1 shows that a bf16-weight reference shares the device's
   rounding and inflates every number downstream.
2. Write `models/demos/llama31_8b_d_p/tests/test_factory.py`, modelled on
   `models/demos/minimax_m3/tests/test_factory.py`:
   - `llama_config_dims()` → loads the bundled/dereferenced `config.json` (no HF, no network);
   - `requires_hf_reference` → `pytest.mark.skipif` on `HF_MODEL` not being a directory;
   - `TestFactory.setup_test(mesh_device, ...)` → builds `MeshConfig` + `CCLManager` once;
   - the **noise-floor helpers** (§2.2), as the single definition in the package.
   Bundle the resolved `config.json` at `models/demos/llama31_8b_d_p/configs/<ModelName>/config.json`
   so dimension-only tests need neither network nor checkpoint (the `minimax_m3` convention).
3. Write `models/demos/llama31_8b_d_p/conftest.py` with a session-scoped `state_dict` fixture and a
   `--skip-model-load` option — copy the shape from `models/demos/minimax_m3/conftest.py`.
   The `mesh_device` and `reset_seeds` fixtures come from the repo root `conftest.py`
   (`conftest.py:554` and `conftest.py:34`) — do **not** redefine them.
4. Prove the reference is deterministic: run it twice on the same input, assert bit-identical
   output. Record both hashes.

### Gate `G-REF`

- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q`
  (pattern: `models/demos/minimax_m3/tests/unit/test_reference_model.py`).
- **PASS when:** (a) the reference produces a fixed-seed hidden-state tensor twice, bit-identical;
  (b) the hand-written and HF references agree to PCC ≥ 0.9999 on one layer — expect **bit-exact**
  (PCC 1.0, `max|Δ| = 0.0`) if the hand-written math is a faithful transcription;
  (c) `01_REFERENCE.md` documents the invocation and the dtype policy.
- **Log:** the two hashes and the cross-reference PCC.
- **Read the bit-exactness honestly.** Two oracles agreeing bit-exactly proves the transcription is
  faithful; it does **not** prove either is right about the architecture. Both could share a
  misreading. That is what the P0 card's per-row provenance is for.

---

## Phase P2 — Repo survey: reuse vs write fresh

**Goal:** `bringup_log/02_SURVEY.md` — a table that decides, for every piece of the model, whether it
is imported or written, with a citation. This is the phase that keeps the package small.

### Where to look (all verified present in this tree)

| Location | What it is | Why it matters here |
|---|---|---|
| `models/demos/minimax_m3/` | **The template of record** for a prefill package. Read its `README.md` "Layout" section first. | Directory shape, `MeshConfig`/`CCLManager` split, `tests/unit/test_*_vs_ref.py` convention, golden-KV scripts, `residual.py` sharded-residual contract |
| `models/demos/gpt_oss_d_p/` | The closest **shape** to Llama: GQA attention, plain RMSNorm, `tt/attention/` split into `config/weights/prefill/operations/kv_cache/dense_sp` | Direct template for `tt/attention/*`, `tt/rms_norm.py`, `tt/layer.py`, `tt/model.py`, `tt/ccl.py`, `tt/tt_prefill_runtime.py`, `tt/runners/adapters/` |
| `models/demos/minimax_m3/tt/dense_mlp.py` | A **dense SwiGLU FFN with the TP collective inside it** | The single best template for Llama's MLP. Read it before writing `tt/mlp.py` |
| `models/demos/deepseek_v3_d_p/` | Shared substrate: `tt/tt_ccl.py`, MoE modules, the chunked-KV and indexed-RoPE ops | Source of `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` and `rotary_embedding_indexed`; ignore everything MoE/MLA |
| `models/tt_transformers/` | The repo's llama home | `tt/load_checkpoints.py` (`convert_hf_qkv_to_meta_format:451`, `map_hf_to_meta_keys:800`, `reverse_permute:891`), `tt/common.py` (`precompute_freqs:489`, `apply_scaling:437` **llama3 rope scaling**, `get_prefill_rot_mat:534`, `get_rot_transformation_mat:562`, `get_rope_theta:165`, `get_rope_scaling:183`), `tt/model_config.py` `reference_*` accessors, `model_params/Llama-3.1-8B-Instruct/config.json`, `tests/test_{mlp,attention,decoder,model,rms_norm,rope}*.py` |
| `models/common/modules/`, `models/common/models/llama3_8b/` | A shared module library (`MLP1D/2D`, `RMSNorm1D/2D`, `Attention1D`, `RotarySetup1D`, `Embedding1D`, `LMHead1D`, cached `TT_CCL`) **and a complete Llama-3.1-8B** | Evaluate them explicitly — and see below for why neither can be the base |
| `models/demos/llama3_70b_galaxy/` | Galaxy llama decode: `llama_ccl.py`, `llama_attention.py`, `distributed_norm.py` | Second opinion on llama-specific CCL placement |
| `models/demos/common/prefill/` | The engine + its two docs (`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`, `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`) | P10 |
| `models/common/utility_functions.py` | `comp_pcc`, `comp_allclose`, `is_blackhole` | Every test |

### `models/common/` (TTTv2) — evaluated, and why it is **not** the base

This is the first question a reviewer asks, so answer it in the survey with evidence, not taste:

- **`MLP2D`'s "2D" is 2D *tensor* parallelism, not TP × SP.** Its prefill path reduce-scatters on
  `cluster_axis=1` and closes with `all_reduce(cluster_axis=0)`
  (`models/common/modules/mlp/mlp_2d.py:461`). With SP on the row axis, that all-reduce would sum
  activations belonging to **different tokens** — silently wrong, and it would still produce
  plausible-looking PCC on a one-row mesh. The tempting shortcut *"an MLP is token-pointwise, so SP
  looks like DP to it"* holds for the math but **not** for this module's collectives. This is
  exactly the class of bug a single-row gate cannot see.
- **There is no `Attention2D`**, and `models/common/models/llama3_8b/model.py:890` raises
  `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")` on a 32-device cluster. No
  chunked-prefill runtime, no `models/demos/common/prefill` adapter.

So `models/demos/minimax_m3/tt/dense_mlp.py` is the MLP template — it collectives on the **TP axis
only**, which is what makes it SP-safe — and `models/demos/gpt_oss_d_p/tt/attention/` is the
attention template. **P9 requirement:** the package `README.md` must carry this answer.

### Steps

1. Read, in this order: `models/demos/minimax_m3/README.md`, `models/demos/gpt_oss_d_p/README.md`,
   `models/demos/minimax_m3/tt/dense_mlp.py`, `models/demos/gpt_oss_d_p/tt/{ccl,config,rms_norm,layer}.py`,
   `models/demos/gpt_oss_d_p/tt/attention/__init__.py`, `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`,
   `models/tt_transformers/tests/test_mlp.py`.
2. Fill the survey table. One row per model component; columns:
   `Component | Decision (import / adapt / write) | Source path:line | Why | DEC`.
   Components to cover at minimum: RMSNorm, RoPE (incl. llama3 scaling), RoPE transformation matrix,
   Q/K/V + O projections, GQA head split, SDPA (prefill, causal), KV cache alloc + chunk write,
   dense SwiGLU MLP, embedding, LM head, decoder layer, model, CCL manager, mesh config, weight
   loading / HF→Meta key mapping, weight tilizing cache, chunked-prefill runtime, prefill adapter,
   golden-KV generation, per-module tests.
3. **Explicitly list what you will NOT bring over** from the templates and why (MoE, router,
   dispatch/combine, sinks, sliding window, QK-norm, MSA/sparse, MLA, MXFP4 loaders). One line each.
   This section is the anti-bloat control.
4. For anything genuinely missing from the repo, open a `07_RISKS.md` entry rather than inventing a
   kernel.

### Gate `G-SURVEY`

- **PASS when:** every component row has a decision + citation; the "not bringing over" list exists;
  the `models/common/` verdict is present with its two citations; no row's decision is "write" where
  an importable equivalent exists (justify with a `DEC` if it is); `verify_citations.py` re-verifies
  every `path:line` in the survey.

---

## Phase P3 — Package outline

**Goal:** `bringup_log/03_OUTLINE.md` — the target file tree, with each file's responsibility,
public interface, and the tensor shapes crossing it. Write this **before** writing code.

### The outline to produce (derived from `minimax_m3` + `gpt_oss_d_p`; adapt, and log deviations)

```
models/demos/llama31_8b_d_p/
├── README.md                     # arch table, deployment path, status, run commands, env vars
├── BRINGUP_RECIPE.md             # this file
├── __init__.py
├── conftest.py                   # session `state_dict` fixture + --skip-model-load
├── configs/<ModelName>/config.json   # bundled dims (no network needed for dim-only tests)
├── bringup_log/                  # §1 — the evaluation artefact
├── tt/
│   ├── __init__.py
│   ├── config.py                 # MeshConfig: mappers + collective wrappers  (template: minimax_m3/config.py:21)
│   ├── ccl.py                    # CCLManager: subdevice, semaphores, scratch (template: gpt_oss_d_p/tt/ccl.py:17)
│   ├── model_config.py           # ModelArgs + the ONE normalised hf_config constructor (P1 trap 2)
│   ├── rms_norm.py               # RMSNorm (plain) + distributed variant
│   ├── rope.py                   # llama3-scaled cos/sin tables + transformation matrix
│   ├── mlp.py                    # dense SwiGLU FFN, column/row parallel, TP collective inside
│   ├── attention/
│   │   ├── __init__.py           # class Attention: builds config+weights, dispatches forward
│   │   ├── config.py             # @dataclass AttentionConfig, ProgramConfig (incl. the pinned SDPA grid)
│   │   ├── weights.py            # load/shard/tilize q,k,v,o; HF→Meta swizzle
│   │   ├── operations.py         # small reusable tensor ops (head split, rope apply, RS/AG helpers)
│   │   ├── prefill.py            # attention_forward(): the one-shot + cache-backed path
│   │   ├── kv_cache.py           # LlamaKVCache, allocate_kv_cache(), write_kv_chunk()
│   │   └── dense_sp.py           # SP ring-SDPA path (P8; stub with NotImplementedError until then)
│   ├── embedding.py              # token embedding (TP-sharded vocab or replicated — DEC)
│   ├── lm_head.py                # `V/TP` shard; prefill's product is the KV cache, but G-MODEL needs top-1
│   ├── layer.py                  # DecoderLayer: norm→attn→residual→norm→mlp→residual
│   ├── model.py                  # Model: embedding → layers → norm → (lm_head); prefill_forward
│   ├── tt_prefill_runtime.py     # chunked-prefill runtime satisfying the engine's §2 contract
│   └── runners/
│       ├── __init__.py
│       ├── kv_chunk_table.py     # block-cyclic KV address table builder (migration)
│       ├── adapters/llama.py     # PrefillModelAdapter subclass
│       └── manifests/llama31_8b_d_p.json
├── utils/
│   ├── __init__.py
│   ├── general_utils.py          # get_cache_file_name, get_default_num_links (copy from gpt_oss_d_p/utils)
│   └── substate.py               # substate() state-dict prefix splitter
├── scripts/
│   ├── verify_citations.py           # §1.6 — written in P0, extended every phase
│   ├── generate_golden_kv_cache.py   # torch reference → per-layer golden KV (template: minimax_m3/scripts)
│   └── verify_golden_kv.py
└── tests/
    ├── __init__.py
    ├── test_factory.py           # fixtures + the single definition of the noise-floor helpers
    ├── unit/                     # per-module PCC vs reference
    │   ├── test_reference_model.py            # G-REF
    │   ├── test_mesh_config.py                # G-MESH
    │   ├── test_ccl_semaphores.py             # G-SEMAPHORE
    │   ├── test_rms_norm_vs_ref.py            # G-RMS
    │   ├── test_rope_vs_ref.py                # G-ROPE
    │   ├── test_mlp_vs_ref.py                 # G-MLP
    │   ├── test_attention_vs_ref.py           # G-ATTN
    │   ├── test_kv_cache_vs_ref.py            # G-KV
    │   ├── test_kv_cache_tp8.py               # G-KV-TP8   (P8; the model → cache path)
    │   ├── test_attention_chunked_vs_ref.py   # G-CHUNK    (deltas 1-2)
    │   ├── test_sp_attention_chunked.py       # G-SP-RING + G-CHUNK-ATTN (P8; delta 3)
    │   ├── test_decoder_layer_vs_ref.py       # G-LAYER
    │   ├── test_embedding_vs_ref.py
    │   ├── test_lm_head_vs_ref.py
    │   ├── test_weight_loading.py             # G-WEIGHTS
    │   ├── test_tp_parity.py                  # G-TP-PARITY
    │   ├── test_model_vs_ref.py               # G-MODEL
    │   ├── test_prefill_runtime_chunked.py    # G-RUNTIME  (P7; the engine's §2 contract, statically)
    │   ├── test_prefill_adapter.py            # G-ADAPTER  (P10)
    │   └── test_kv_chunk_table.py             # G-KV-TABLE (P10)
    ├── fabric_topology_matrix.py # G-FABRIC-MATRIX: subprocess-isolated (mesh, topology, links) sweep
    └── galaxy_prefill_kv_pcc.py  # G-MESH-KV, G-RACE: per-layer KV PCC vs golden on the target mesh
```

**Every gate in Appendix A owns something in this tree, and the two are added in the same edit.**
For a pytest gate that is a test file — check the index against the tree, one row at a time, and
note that `G-MESH`, `G-SEMAPHORE`, `G-WEIGHTS` and `G-TP-PARITY` are the four easiest to leave
unowned because no `test_<module>_vs_ref.py` naturally covers them. For the engine gates
(`G-REQUEST`, `G-MOCK-MIG`, `G-LOOPBACK`) it is a verbatim two-terminal transcript in
`bringup_log/raw/`, and for the doc gates it is a section of the log they audit. **An unowned gate
silently becomes a `NOT-RUN`.**

There is deliberately **no `reference/` package** (P1).

### Conventions to honour (all observed in the templates — deviating is a `DEC`)

- **Module signature.** Every TT module takes, in this order:
  `(mesh_device, hf_config, state_dict, ...)` then keyword-only
  `mesh_config=`, `ccl_manager=`, `tensor_cache_path=`, `weight_dtype=`.
  Forward is `__call__` (or `forward` for `nn.Module` subclasses like `RMSNorm`).
- **State-dict splitting** is the caller's job, via `substate(state_dict, "mlp")` —
  `models/demos/gpt_oss_d_p/utils/substate.py`. Modules receive an already-stripped sub-dict.
- **Weight loading** goes through `ttnn.as_tensor(..., cache_file_name=get_cache_file_name(path, name))`
  so the tilized weight is persisted and reloaded. **Every module must build correctly from an
  empty `state_dict` when a cache path exists** ("cache-only mode") — the runner relies on it. See
  `models/demos/minimax_m3/tt/dense_mlp.py::_load` for the exact shape of this branch. Put the
  **mesh shape and the dtype in the cache path** (as `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75`
  does): the tilized tensor is already sharded, so a cache written at one mesh shape is wrong at
  another, and the symptom is "one layer runs on garbage" three phases later.
- **HF `[out, in]` → ttnn `[in, out]`:** transpose at load time
  (`weight.transpose(-1,-2).unsqueeze(0).unsqueeze(0)`), never at runtime.
- **Pass an explicit `compute_kernel_config` to every op that accepts one** (§2.4). This is not a
  style rule; omitting it costs measurable precision on the norms, and inheriting a template's
  explicit `fp32_dest_acc_en=False` costs two to three orders of magnitude on the matmuls.
- **Deallocate eagerly.** `t.deallocate(True)` after last use; free the big input before allocating
  the big output (see the comment in `models/demos/minimax_m3/config.py::allreduce`).
- **Docstring anchors.** Each module's docstring names the HF anchor
  (`transformers.models.llama.modeling_llama.LlamaMLP`) and the source template it mirrors.
- **No env-var magic** beyond what `README.md` documents in a table.

### Gate `G-OUTLINE`

- **PASS when:** `03_OUTLINE.md` lists every file with (i) one-sentence responsibility, (ii) public
  interface signature, (iii) input/output tensor shapes with dtype and layout, (iv) the template it
  mirrors (`path:line`); **every Appendix A gate maps to a named owner in the tree** — a test file,
  a harness, or (for the two-terminal engine gates) a raw transcript; and the per-layer
  tensor-shape table (below) is filled in with real numbers.

Fill this table for the chosen `(mesh, TP, SP)` — the `models/demos/gpt_oss_d_p/README.md` "shapes & correctness
notes" table is the model to follow. `S_loc = S/SP`, `TP = 8`:

| tensor | shape (per chip) | dtype | layout |
|---|---|---|---|
| hidden in | `[1, 1, S_loc, 4096]` | bf16 | TILE |
| Q | `[1, 32/TP, S_loc, 128]` | bf16 | TILE |
| K, V | `[1, 8/TP, S_loc, 128]` = `[1, 1, S_loc, 128]` at TP=8 | bf16 | TILE |
| KV cache | `[users*layers, 1, max_seq_len/SP, 128]`, block-cyclic, shard row `[1,1,32,128]` | bf8_b (`DEC`) | TILE |
| attn out (pre-o_proj) | `[1, 32/TP, S_loc, 128]` | bf16 | TILE |
| MLP gate/up | `[1, 1, S_loc, 14336/TP]` | bf16 | TILE |
| residual | `[1, 1, S_loc, 4096]` or `[1,1,S_loc,4096/TP]` if sharded (`DEC`) | bf16 | TILE |

---

## Phase P4 — Parallelism & CCL plan

**Goal:** `bringup_log/04_CCL_PLAN.md`. The user requirement is explicit: **CCLs are part of the
modules** — attention, MLP, RMSNorm each own their own collectives. The layer and the model never
call a collective directly.

### The pattern to implement (this is the repo's converged answer — do not invent another)

Two objects, both created **once per model**:

1. **`CCLManager`** (`tt/ccl.py`) — owns *persistent CCL resources*: the CCL sub-device, the
   reduce-scatter / all-gather **ping-pong semaphore** sets, barrier semaphores, ring-attention
   semaphore pair, and reusable ring-gather scratch buffers. Template:
   `models/demos/gpt_oss_d_p/tt/ccl.py:17` (139 lines, fully commented) — itself mirroring
   `models/demos/minimax_m3/tt/ccl.py:9`. Key properties to preserve:
   - the **CCL core range derives from `mesh_device.compute_with_storage_grid_size()`** (this
     Blackhole is (12,10), wider than 8×8; hard-coding 8×8 breaks the ring-SDPA grid-offset assert).
     This is the **opposite** of the SDPA *program* grid, which stays a pinned 8×8 — see P5.5. The
     two grids look alike and must not be unified.
   - semaphores are allocated **once**, never per layer or per chunk;
   - handing out semaphores cycles a ping-pong index (`get_rs_ping_pong_semaphore()`,
     `get_ag_ping_pong_semaphore()`, `get_barrier_semaphore()`), so back-to-back collectives never
     reuse a semaphore that may still be in flight. **This is the single most common source of
     nondeterministic multi-device PCC failures.**
   - be aware that the barrier ping-pong is only **2 deep** — RS takes `barrier[0]`, AG `barrier[1]`,
     the next RS `barrier[0]` again, a one-op gap — and that `reset_global_semaphores` deliberately
     skips the barrier and ring-attention semaphores (`models/demos/gpt_oss_d_p/tt/ccl.py:132`, an
     open upstream TODO) while chunked prefill **does** reuse one `CCLManager` across chunks. Write
     a `DEC` either way. If `G-RACE` fails, deepening the barrier ring from 2 to 4 is the first
     move, before suspecting the model.
2. **`MeshConfig`** (`tt/config.py`) — owns *the parallelism decision and the collective wrappers*:
   `shard_mapper`, `column_parallel`, `row_parallel`, `sequence_parallel`, `shard_size`, and the
   three collectives `allreduce(t, ccl, axis=...)`, `allgather(t, ccl, axis=, dim=)`,
   `reduce_scatter(t, ccl, dim=, axis=)`. Template: `models/demos/minimax_m3/config.py:21`
   (`allreduce:77`, `allgather:135`, `reduce_scatter:155`).
   TP is the only knob; SP = the other axis, derived. `_validate()` rejects sub-axis TP.
   **Neither in-repo copy is a superset — build the union.** `models/demos/minimax_m3/config.py:21`
   has `reduce_scatter`; `models/demos/gpt_oss_d_p/tt/config.py:19` does **not**, and its
   `_VALIDATED_*` at `:15-16` already pins `(4,8)`/TP=8 — this target.

**Modules then call `self.mesh_config.<collective>(t, self.ccl_manager, ...)` themselves.**
Never raw `ttnn.experimental.*` inside a module — that is how semaphore reuse bugs get in. (The one
allowed exception in the templates is `ttnn.all_gather` for the tiny RMSNorm stats tensor; if you
use it, log a `DEC`.)

**Canonical collective ops** (usage counts measured across `minimax_m3`, `gpt_oss_d_p`,
`deepseek_v3_d_p`, `tt_transformers` in this tree): `ttnn.experimental.all_gather_async` (29 uses),
`ttnn.experimental.reduce_scatter_minimal_async` (18), `ttnn.experimental.all_reduce_async` (2).
An all-reduce is implemented as **reduce-scatter + all-gather** (see `MeshConfig.allreduce`), not as
`all_reduce_async`. `num_links` comes from `utils/general_utils.get_default_num_links(mesh_device)`, whose single-row
behaviour has a gate consequence — P8 step 3.

### The collective placement for Llama — write this table with justification

| Module | Where the collective sits | Which collective | Why |
|---|---|---|---|
| `RMSNorm` | inside `forward` | none when the input is full-emb replicated; `all_gather` of the `[1,1,32,32]` stats tensor when the residual is emb/TP-sharded (`rms_norm_pre_all_gather` → AG → `rms_norm_post_all_gather`) | template: `models/demos/gpt_oss_d_p/tt/rms_norm.py` (both branches, gated on `is_distributed`) |
| `MLP` (dense SwiGLU) | end of `__call__`, after `down_proj` | `reduce_scatter` (sharded residual) **or** `allreduce` (replicated residual) on the TP axis | `down_proj` is row-parallel → each TP device holds a *partial sum*, so a TP collective is mandatory. Template: `models/demos/minimax_m3/tt/dense_mlp.py::__call__` |
| `Attention` | end of the forward, after `o_proj` | same choice as MLP (`reduce_scatter` / `allreduce`) on the TP axis | `o_proj` is row-parallel over the head dim |
| `Attention` (SP path) | inside SDPA | ring-attention halo exchange, via the ring-SDPA op using `ccl_manager.ring_attention_ccl_semaphore_handles` + `ring_attention_ccl_core_grid_offset` | P8 only; template `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py` |
| `Embedding` | after lookup | `all_gather` if the vocab is TP-sharded | `DEC` — or replicate the embedding table and skip it |
| `LM head` | after the matmul | `all_gather` on the vocab shard | only if logits are needed |
| `DecoderLayer` / `Model` | **never** | — | residual adds are elementwise-local by construction |

**Collectives go on the TP axis only.** That is what makes every module SP-safe, and it is the
concrete difference between this plan and `models/common/modules/mlp/mlp_2d.py:461` (P2).

### Residual-layout decision (do this consciously, it touches every module)

Two consistent schemes; pick one, log it, and hold it everywhere:

- **A — replicated residual (full emb):** every module returns `[1,1,S_loc,4096]`; attention and MLP
  close with a full **all-reduce**.
- **B — sharded residual (emb/TP):** the residual stream is `[1,1,S_loc,4096/TP]`; attention and MLP
  close with a **reduce-scatter only**, and the norm either all-gathers first or runs the 3-op
  distributed RMSNorm. Requires `4096/TP % 32 == 0` (true at TP=8: 4096/8 = 512).

**Take A for this iteration**, and note carefully *why*, because the obvious reason is wrong.

- The **wrong** reason: "B is unproven because `models/demos/gpt_oss_d_p/tt/rms_norm.py:33` pins
  `is_distributed = False` with the condition commented out." That branch is indeed dormant, but B
  does not require it: `models/demos/minimax_m3/tt/residual.py:26` ships **scheme B by default**,
  with `DEFAULT_NORM_MODE = "gather_first"` (`:32`), which all-gathers the residual shard and runs
  one ordinary single-pass norm. Only **B-with-distributed-norm** is unproven.
- The **right** reason is cost equivalence on a *dense* model: A and B issue the **identical**
  collectives per layer — 2 reduce-scatters + 2 all-gathers, same sizes, same axis. Minimax's B win
  comes from sharing one gathered norm output across several MoE consumers
  (`models/demos/minimax_m3/tt/residual.py:9-11`), and Llama has no such consumers. A additionally
  keeps `G-TP-PARITY` a direct device-vs-device comparison, and a replicated embedding already
  yields a full-width residual.

Wire the `scatter_output` parameter from `models/demos/minimax_m3/tt/dense_mlp.py` from day one so
switching to B later is a flag, not a rewrite — but make any module that cannot honour it **refuse**
`scatter_output=True` loudly rather than half-wiring the scheme.

### Gate `G-CCL-PLAN`

- **PASS when:** `04_CCL_PLAN.md` contains: the `(mesh, TP, SP)` arithmetic; the collective-placement
  table above with every row justified; the residual-scheme `DEC` with the cost-equivalence argument;
  the semaphore-lifetime statement ("allocated once in `CCLManager.__init__`, cycled per call, never
  per layer") **and its depth**; and a list of every collective call site with its `cluster_axis`,
  `dim`, and `topology`.

---

## Phase P5 — Naive module implementations, bottom-up

**Goal:** each module written and PCC-verified on **one card** (`(1,1)` mesh, TP=1, SP=1, no CCL)
before any multi-device work. This is the order the templates were built in and it is the order that
isolates failures.

For **each** module below, the loop is identical:

1. Write the module in `tt/`, following the P3 conventions.
2. Write `tests/unit/test_<module>_vs_ref.py`, following
   `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` (read it once; it is the model
   answer): module docstring explaining the block and the run command, a torch reference in the test
   file, **identical random weights driving both sides**, `@pytest.mark.parametrize("mesh_device",
   [(1,1)], indirect=True)` for every test that puts a *module* on a card, `reset_seeds`, `comp_pcc`
   from `models/common/utility_functions`, `logger.info` the PCC, `assert passing`.
3. **Compute the noise floor in the same test.** Record the error ratio always; *assert* it where
   Appendix A states a bound for that gate (§2.2). Add a **negative control** that the same
   assertion must reject.
4. Run it, `tee` the raw log, record the gate with its input distribution, reference dtype policy,
   floor, measured value and control.
5. Log every judgement call as a `DEC`.

Multi-device parametrisations are added in P8, not here. Tests needing fabric take
`@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
indirect=True)` (pattern: `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py:126`).

### P5.1 `tt/config.py` + `tt/ccl.py` + `utils/`

Port `MeshConfig` (the **union** of `models/demos/minimax_m3/config.py` and
`models/demos/gpt_oss_d_p/tt/config.py` — P4) and `CCLManager` (from
`models/demos/gpt_oss_d_p/tt/ccl.py`), **deleting** what Llama does not need (the ring-gather scratch
buffers can stay — the SP path in P8 uses them — but MoE-specific pieces, and any `ep_axis`, go).
Copy `utils/general_utils.py` and `utils/substate.py` from `models/demos/gpt_oss_d_p/utils/`. Set
`_VALIDATED_MESH_SHAPE` / `_VALIDATED_TP` to your P0 target.

**Gate `G-MESH`:** two tests in one file. (a) **Device-free** — `MeshConfig((1,8), tp=8)` yields
`sp=1, tp=8, shard_size(4096)=512, shard_size(14336)=1792`, and sub-axis TP (e.g.
`MeshConfig((1,8), tp=4)`) **raises**. `MeshConfig` accepts any shape whose TP divides the column
axis; `_VALIDATED_MESH_SHAPE` / `_VALIDATED_TP` name the *deployment* target and warn when you are
off it — only sub-axis TP is a refusal, because only that one produces a wrong tensor. (b) **On a
card** — `CCLManager` constructs without error, allocates its semaphores exactly once (assert the
list lengths, and again after dozens of getter cycles), and reports the real compute grid and CCL
offset. Only (b) needs a device, so only (b) takes the `mesh_device` fixture.

### P5.2 `tt/rms_norm.py`

Plain RMSNorm: `out = rms_norm(x) * weight`. **No Gemma `+1` fold** (P0 card). Keep the
`is_distributed` branch from `models/demos/gpt_oss_d_p/tt/rms_norm.py` but make it a constructor
argument defaulting to `False` until P8. Weight is reshaped to `(1,1,-1,ttnn.TILE_SIZE)` and stored
`ROW_MAJOR`. Pass an explicit `compute_kernel_config` with `fp32_dest_acc_en=True` — this is the op
whose default does *not* already enable it (§2.4).

**Gate `G-RMS`:** `test_rms_norm_vs_ref.py`, seq_len ∈ {32, 512, 4096}, **PCC ≥ 0.9999**, with the
gap to the computed floor (0.9999986) recorded. Drive it with **standard-normal** inputs and an
**fp32** reference weight (§2.1). Negative control: a zero-gain probe must produce `max|out| = 0.0`
— a Gemma `(1 + weight)` fold would return the normalised input instead.

This gate is also the cheapest demonstration of §2.4: the same module measures **0.9999697** with
no `compute_kernel_config` and **0.9999955** with `fp32_dest_acc_en=True` — ~22x off the floor
versus ~3x off it, at an absolute PCC that clears 0.9999 either way. That ~3x is also why `G-RMS`
*records* its ratio instead of asserting §2.2's 3x stage bound: a correct module sits right on it.

### P5.3 `tt/rope.py`

Llama-3 scaled RoPE. **Reuse, do not rewrite:** `models/tt_transformers/tt/common.py`
`precompute_freqs:489` + `apply_scaling:437` (`rope_type="llama3"` uses
`compute_llama3_parameters:405`, which takes **three** args `(freqs, scale_factor, orig_context_len)` —
`low_freq_factor = 1` and `high_freq_factor = 4` are **local constants** at `common.py:407-408`, NOT read
from `config.json`; benign for Llama-3.x, silently wrong for any model that changes them — assert
them against the config rather than trusting them),
`get_prefill_rot_mat:534`, `get_rot_transformation_mat:562` (call it with **no args** — it ignores
`dhead`, P1).

**θ and the scaling parameters are read here and nowhere else,** via
`models/tt_transformers/tt/common.py:165` `get_rope_theta` / `:183` `get_rope_scaling`, and asserted
non-`None`. P1 trap 1 is why: a `getattr(cfg, "rope_theta", 10000.0)` anywhere in the package
produces a RoPE that is wrong at every position and raises nothing.

**Two conventions exist, there is a ttnn op for each, and mixing them is *the* classic RoPE bug:**

| convention | layout | ttnn op | weight handling |
|---|---|---|---|
| **Meta / llama** | interleaved pairs | `ttnn.experimental.rotary_embedding_llama` (+ a transformation matrix from `get_rot_transformation_mat:562`) | Q/K **projection weights must be `reverse_permute`d** at load: `load_checkpoints.convert_hf_qkv_to_meta_format:451`, `reverse_permute:891` |
| **HF** | halves concatenated, `rotate_half` over the full head | `ttnn.experimental.rotary_embedding_hf` | no permute; weights stay in HF layout |

`models/tt_transformers/tt/attention.py:641-723` implements **both** (dispatch at `:159-173` is the
better anchor), selected by
`ModelArgs.use_hf_rope`, whose default is `False` (`models/tt_transformers/tt/model_config.py:623`) — i.e. llama runs the Meta
path today, and the comment there notes HF is intended to become the only one (issue #37605).

**Take the Meta path (`rotary_embedding_llama`)**: it is what both prefill templates use
(`models/demos/gpt_oss_d_p/tt/attention/operations.py:87`, `models/demos/minimax_m3/tt/attention/operations.py:93`), so the
whole surrounding prefill scaffolding already assumes it. Do the Q/K `reverse_permute` **inside the
attention weight loader**, so a weight can never reach the device un-swizzled by a path that forgot,
state the choice in the module docstring, and log it as a `DEC` that names `rotary_embedding_hf` as
the alternative that removes the permute (the likely direction of travel).

`test_attention_vs_ref.py` in `gpt_oss_d_p` builds **both** cos/sin tables from one set of
frequencies (`_build_cos_sin`) and feeds the Meta pair to the device while the torch reference uses
the HF pair — copy that structure exactly, so the test cannot silently compare two different RoPEs
and call it a pass. Also expose a chunk-offset table builder (an *indexed* RoPE) separately from the
contiguous prefill one, and have the contiguous builder assert `start_pos <= seq_len`: chunked
prefill in P7 must not reach for it.

**Gate `G-ROPE`:** `test_rope_vs_ref.py` — apply RoPE to a random `[1, n_heads, S, 128]` tensor on
device and compare against the HF-convention torch `rotate_half` path applied to the
correspondingly-permuted input. **PCC ≥ 0.999** (expect ~0.99999). Negative control: feeding an
HF-layout tensor to the Meta op must collapse (measured **0.01296**) — without it, 0.99999 could
mean "both sides are wrong the same way". Also assert the llama3 scaling actually took
effect: the scaled `inv_freq` must differ from the unscaled one for positions beyond
`original_max_position_embeddings` (a test that passes with scaling silently disabled is worthless).

### P5.4 `tt/mlp.py` — dense SwiGLU

`down(silu(gate(x)) * up(x))`, `intermediate_size` 14336, no biases.
`gate_proj`/`up_proj` **column-parallel** (shard the intermediate dim), `down_proj`
**row-parallel** (shard the input/intermediate dim) + the TP collective from P4.
Template: `models/demos/minimax_m3/tt/dense_mlp.py` — take its structure, replace the clamped
`swigluoai` activation with plain SwiGLU (`ttnn.silu(gate) * up`, or `ttnn.mul(..., input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`
if available — check, and log which).

**Gate `G-MLP`:** `test_mlp_vs_ref.py`, seq_len ∈ {32, 512, 4096}, **PCC ≥ 0.999 @bf8_b** and
**≥ 0.9995 @bf16**, and **≤ 3x the computed floor** at each dtype. Run both; record both. Negative
control: applying SiLU to `up` instead of `gate` must collapse (measured **0.6462**) — this is what
proves the fused unary is on the argument you think it is. If bf8_b misses its threshold, do not
lower it — log a `DEC` and keep bf16 for this iteration.

### P5.5 `tt/attention/` — GQA, full RoPE, causal SDPA

Split the directory exactly as `models/demos/gpt_oss_d_p/tt/attention/` does; it separates the
concerns that otherwise tangle:

- `config.py` — `@dataclass AttentionConfig` (hidden_size, num_heads, num_kv_heads, head_dim,
  max_seq_len, rms_norm_eps, `scaling = head_dim**-0.5`, `sequence_parallel`) and `ProgramConfig`.
  **Drop** `sliding_window`, `sinks`, `layer_types` — Llama has none (P0 card). **Do not copy
  `models/demos/gpt_oss_d_p/tt/attention/config.py:71`'s `fp32_dest_acc_en: bool = False`** (§2.4).
- `weights.py` — load, transpose, Meta-swizzle, shard, tilize `q/k/v/o_proj`. Q/O are
  column/row-parallel over heads; K/V are column-parallel over KV heads. At TP = 8 each chip holds
  exactly **one** KV head, which is the deployment configuration (P0).
- `operations.py` — head split/merge, RoPE application, the reduce-scatter/all-gather tail helper.
- `prefill.py` — `attention_forward(...)`: qkv proj → head split → RoPE → causal SDPA → merge heads
  → `o_proj` → TP collective.
- `kv_cache.py` — `LlamaKVCache` (packed K/V), `allocate_kv_cache()`, `write_kv_chunk()`.
- `dense_sp.py` — SP ring path; create the file with a `NotImplementedError` and a docstring
  pointing at `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py`. It is filled in P8.
- `__init__.py` — `class Attention` assembling config + weights and dispatching to `prefill.py`
  (`models/demos/gpt_oss_d_p/tt/attention/__init__.py:28` is the template; delete the `is_sliding` logic).

**GQA is native to the SDPA op — there is no on-chip KV repeat.**
`ttnn.transformer.scaled_dot_product_attention` handles the group itself; evidence:
`models/demos/gpt_oss_d_p/tt/attention/prefill.py:34-49` passes `q` with 8 local Q heads and `k`/`v`
with 1 local KV head. The call shape:

```python
ttnn.transformer.scaled_dot_product_attention(
    tt_q, tt_k, tt_v,
    is_causal=True,
    scale=config.scaling,                 # == 1/sqrt(head_dim); pass it explicitly
    program_config=...,                   # ttnn.SDPAProgramConfig
    compute_kernel_config=...,            # fp32_dest_acc_en=True here (§2.4)
)
```

For Llama **drop** `sliding_window_size=` and `attention_sink=` — they are gpt-oss-only arguments.

**The SDPA program grid stays a pinned 8×8. Do not derive it from the device grid.** This looks like
a portability improvement and it is a P8-only landmine:

- this machine's compute grid is **(12, 10)** (`compute_with_storage_grid_size()`), not 8×8;
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`
  asserts **`ccl_core_grid_offset.x >= sdpa_grid.x`**, and the CCL offset is pinned at
  `grid.x - 1 = 11`;
- with the SDPA grid at 8: `11 >= 8`, fine. Derived from the device grid at 12: `11 >= 12`, **fails**.

The failure mode is the dangerous kind: a derived grid **passes every P5 single-card gate** — none
of which runs the ring path — and only fails at SP > 1 in P8, long after the choice looks settled.
So: keep the SDPA program grid an **explicit named field defaulting to 8×8**
(`ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False,
q_chunk_size=..., k_chunk_size=...)`, as `models/demos/gpt_oss_d_p/tt/attention/config.py:95-100`
does), and **assert `sdpa_grid.x <= grid.x - 1` at construction** so the constraint fails at build
time instead of two phases later. Give the SP ring path its own program config rather than mutating
this one.

**Gate `G-ATTN`:** `test_attention_vs_ref.py` — full block (QKV → GQA split → RoPE → causal SDPA →
o_proj) vs an in-test **fp32** torch reference, identical random weights, seq_len ∈ {128, 512, 2048},
`(1,1)` mesh. **PCC ≥ 0.999**; each stage you implement yourself **≤ 3x** its floor; the whole block
**≤ 8x**. The torch reference must build the causal mask explicitly
(`torch.triu(full((S,S), -inf), diagonal=1)`) and `repeat_interleave` the KV heads by the GQA group
— copy the reference from `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py::_torch_attention`,
removing the sink column and the sliding term. Negative control: Q/K weights loaded *without* the
Meta `reverse_permute` (measured **0.9475** — note how *high* a badly broken variant scores, which
is the whole argument of §2.1). Assert as an invariant that only Q and K are rotated.

**Expect the block to sit further off its floor than its parts, and attribute it before debugging
it** — §2.3 measures where the gap lives. Keep the standalone SDPA probe permanently, so the
kernel's slack stays a named, tracked term rather than budget the block silently grants itself.

### P5.6 KV cache

The KV cache is **the output of prefill** — its correctness is the whole point. Layout: SP-sharded,
**block-cyclic**, per-user slots. Writes go through
`ttnn.experimental.deepseek_prefill.update_padded_kv_cache` (source:
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/`; the constraint
`kv_actual_global % 32 == 0` is documented in
`models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` "Multi-turn conversations").
Templates: `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py` (177 lines) and
`models/demos/minimax_m3/tests/unit/test_kv_cache_{write,gqa_sp}_vs_ref.py`.

**`bfloat8_b` is the cache dtype** (P3, and every threshold in Appendix A assumes it); bf16 is a
measurement mode, run to produce the delta that justifies bf8_b. Log that delta as a `DEC` — the
PCC cost measured, not assumed — and log block size, per-user slot sizing (`MAX_SEQ_LEN`), and the
fact that K is stored **post-RoPE** (it is, in every template — say so). Changing the dtype is a
`DEC` whose blast radius is every KV threshold.

**Keep gpt-oss's block geometry unless you have a reason not to.**
`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27` defines
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` and shards with
`shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]` (`:87`). Matching it is what lets
P10 reuse the producer's existing packed-K/V read-back instead of writing a fourth reader
(see P10 step 5). Diverging from it is a `DEC` whose blast radius includes `G-MOCK-MIG`.
`head_dim` is parameterised by that shard spec and 128 is tile-aligned — measured clean.

`write_kv_chunk` writes **one user per call** and must assert it
(`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:149`): the op ignores the leading batch dim, so a
`batch > 1` tensor would silently write only `slot_idx`. Multi-user prefill loops `slot_idx + b` at
the call site. That assert is about the *batch* dim; the **head** count is policed by the op's own
`TT_FATAL` (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`), which is why the `nkv = TP` mapping
must be proved by a TP=8 run rather than assumed (P8).

**Gate `G-KV`:** `test_kv_cache_vs_ref.py` — write a chunk at the real `head_dim = 128`, read it
back, compare against the torch reference's post-RoPE K and raw V. **PCC ≥ 0.99** at the chosen cache
dtype and **≤ 3x its floor** (record the bf16 number too; expect both essentially *at* the floor).
Additionally assert, **bit-exactly** (`rtol=atol=0`): a positional read-back that puts every row at
its own global position; the written region only — pad-tail positions untouched, another slot exactly
zero, and an earlier chunk unchanged after a later chunk's write.

Two constraints on that positional probe, both learned the hard way: encode positions as values
**≤ 256** or bf16 rounds 257 to 256 and the probe fails on a correct cache (§2.5); and cover several
`kv_actual` offsets, not one.

**What `G-KV` does not prove** — the model → cache path, for the reason P0 gives. Say so in the
gate block rather than letting the `PASS` imply it; `G-KV-TP8` in P8 is what closes it.

### Gate summary for P5

All of `G-MESH`, `G-RMS`, `G-ROPE`, `G-MLP`, `G-ATTN`, `G-KV` must be `PASS` before P6.

---

## Phase P6 — Layer and model assembly

### P6.1 `tt/layer.py`

```
residual = x
h = input_layernorm(x)          -> Attention(...)  -> h
x = residual + h
residual = x
h = post_attention_layernorm(x) -> MLP(h)          -> h
x = residual + h
```

Template: `models/demos/gpt_oss_d_p/tt/layer.py` — take it and delete the MoE branch and the
`layer_types` plumbing. **Do keep a bring-up probe**: a per-layer L2 / mean-abs / signed-mean dump of
each residual delta behind one env var. It is the fastest tool for finding *which* sublayer drifts in
a 32-layer stack, and its output belongs in `bringup_log/raw/`. Document the env var in `README.md`.

Keep the `ttnn.move(hidden_states)` re-allocation guard for long sequences (`seqlen > 32*1024`) and
the eager `deallocate(True)` calls — both are load-bearing for long-context DRAM pressure.

**Gate `G-LAYER`:** `test_decoder_layer_vs_ref.py` vs `ModelArgs.reference_decoder()`
(`models/tt_transformers/tt/model_config.py:4393`) or an in-test fp32 torch layer, seq_len ∈ {128,
512, 2048}, `(1,1)`. **PCC ≥ 0.999** and **≤ 8x the computed floor**. Negative control: swap the two
norm gains (measured **0.9471** — again, note how high a broken layer scores).

**`G-LAYER` and `G-MODEL` are integration checks and may never substitute for a missing or weak
sublayer gate.** Two measured reasons, and one plausible-sounding reason that is **false**:

- *True:* **a layer or model PCC cannot localise.** One bad sublayer in a 32-layer stack moves an
  aggregate that a dozen other causes also move. This is the entire content of the rule, and it is
  why the delta probe and the per-layer PCC curve exist.
- *True:* **the layer's floor is looser than its sublayers'.** Every additional bf8_b weight lowers
  the achievable PCC, so a layer threshold that a *sublayer* would fail is arithmetically normal —
  which is exactly why the sublayer thresholds must be kept and met on their own.
- *False:* "the residual stream dominates the correlation, so a layer PCC launders a degraded
  sublayer." Measured against **one** consistent fp32 reference, the layer scores **below** its own
  attention block at both dtypes (bf8_b 0.9995864 vs 0.9997554; bf16 0.9997674 vs 0.9998129) — a
  layer PCC is a *harder* test here, not an easier one. The masking mechanism is real but small: for
  `y = r + s`, a perturbation of `s` is attenuated in `y` by exactly `||y||/||s||`, which measures
  **1.12x** on the gate's random weights and **1.73x / 1.23x** on real layer-0 weights. That cannot
  turn 0.9996 into 0.9999985. The number that suggested otherwise came from comparing two different
  `tt_transformers` test files with different reference constructions — the cross-test comparison
  §2.1 forbids.

### P6.2 `tt/embedding.py`, `tt/model_config.py`

- `model_config.py::ModelArgs` — state-dict loading (`load_state_dict` via safetensors, then
  `map_hf_to_meta_keys` / `convert_hf_qkv_to_meta_format` from
  `models/tt_transformers/tt/load_checkpoints.py`), `weight_cache_path(dtype)` **including the mesh
  shape** (P3), `get_state_dict_prefix(module, layer_idx)`. Template:
  `models/demos/minimax_m3/tt/model_config.py:22`.
- `embedding.py` — a replicated table is fine for a first pass (`DEC`); `vocab_size` 128256 is
  tile-friendly (128256/32 = 4008).

**Gate `G-WEIGHTS`:** `test_weight_loading.py` loads the real checkpoint and asserts (a) every
expected key is consumed — **no silently-unused weights and no missing weights**; print both sets;
(b) a cache-only rebuild (empty `state_dict` + populated `tensor_cache_path`) produces bit-identical
device tensors; (c) a sample of device weights is **bit-exact** against the checkpoint
(`rtol=atol=0`) *through* the loader's transpose, Q/K Meta swizzle and dtype ladder — PCC would not
catch a swizzle applied twice. Negative control: bypass `map_hf_to_meta_keys` and every key must go
missing. This gate catches the failure mode where a renamed key means a layer quietly runs on random
weights. Note it runs on **one card**: cache-only at TP > 1 is a P8 extension (below).

### P6.3 `tt/model.py`

`Model`: embedding → `[DecoderLayer] * n_layers` → final norm → (optional lm_head).
Public surface, matching the templates (`models/demos/gpt_oss_d_p/tt/model.py:41`,
`models/demos/minimax_m3/tt/model.py:87`): `prepare_inputs_prefill`, `prefill_forward`,
`process_output_prefill`.

**Gate `G-MODEL`:** `test_model_vs_ref.py` on a **reduced layer count** (`n_layers=2`, then 4) vs the
HF reference with the same weights, seq_len ∈ {128, 512}. **hidden-state PCC ≥ 0.999**, **≤ 8x the
floor**, and **top-1 token agreement = 100%** on the last position. Build the model
`with_lm_head=True` by default so this half of the gate is never conditional. Then the
full 32-layer run: record the per-layer hidden-state PCC curve into `bringup_log/raw/` and gate the
**step** between consecutive layers at **≤ 4x** from layer 3 onward. A monotone decay is normal; a
*step* at one layer is a bug and must be chased before P7. Negative control: rotate the per-layer
weights (measured **0.1612**).

Using HuggingFace itself as the oracle here is admissible only with self-checks attached: confirm
causality directly (`max|Δ|` on rows `[:-1]` = 0 after perturbing the last token), confirm the
in-test fp32 reference matches HF at PCC 1.0, and state the dtype policy. See P1 trap 3.

---

## Phase P7 — Chunked prefill + golden KV

**Goal:** the multi-chunk path (the deployment path) and a torch golden KV cache to check it against.

1. `scripts/generate_golden_kv_cache.py` — run the torch reference in **fp32**, save per-layer
   post-RoPE K and raw V. Output layout (copy it exactly; the engine's producer read-back expects
   it — `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py` header):
   ```
   {trace_dir}/metadata.json                       # prompt, token_ids, model info
   {trace_dir}/kv_cache/layer_<i>.safetensors      # key_cache_layer_<i>, value_cache_layer_<i>
                                                   # [1, num_kv_heads, seq_len, head_dim], HF layout
   ```
   Stream weights per layer via mmap and write per layer — do not hold 32 layers of KV in RAM.
   Store the golden at **fp32**, not the template's bf16: it is the reference, and §2.1 is about
   exactly this. Drive it one `LlamaDecoderLayer` at a time and prove the streamed driver equals
   `LlamaModel`'s own loop at `rtol=atol=0`, so the streaming is not itself the thing under test.
   The trace directory comes from `$PREFILL_TRACE_DIR` — the engine already owns that variable, so
   do not invent a package one.
2. `scripts/verify_golden_kv.py` — compare a device KV read-back against the golden, per layer,
   reporting min/mean PCC per layer for K and V.
3. `tt/tt_prefill_runtime.py` — the chunked runtime. Build it to the engine's contract *now* so P10
   is wiring, not rework: `compile(kv_cache)`, `make_chunk_input(token_ids)`,
   `prefill_chunk(input, kv_cache, *, slot_id, actual_start, actual_end, request_id=0,
   d2h_service=..., metadata_msg=...)`, plus `mesh_device` and a `config` exposing
   `chunk_size/max_seq_len/first_layer_idx/is_first_rank/is_last_rank`. Full contract:
   `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` §2 — but see the P10 warning about
   what that section omits, and write the signature with the extra two parameters from the start.
   Template: `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:96`.
   **The runtime must not own the KV cache** — the engine allocates it and passes it in.
4. `tests/unit/test_attention_chunked_vs_ref.py` — the chunked-vs-one-shot equivalence test
   (template: `models/demos/minimax_m3/tests/unit/test_attention_chunked_vs_ref.py`).

**A chunked prefill differs from a one-shot in exactly three places.** Name them, because P7 can
measure two of them and not the third:

| delta | what changes | owned by |
|---|---|---|
| **1** | the RoPE table and its per-chunk position offset (an *indexed* RoPE, not the contiguous one) | P7 |
| **2** | the cache-write offset — `kv_actual_global` advancing per chunk | P7 |
| **3** | the attention core: chunk *k*'s queries must attend the prefix **read back out of the cache** | P8 |

**Gate `G-CHUNK`** covers deltas 1 and 2, and the decomposition is exact rather than approximate:
feed **the same hidden states** — from one one-shot forward of the real 32-layer model — to both
KV producers, and deltas 1+2 are then the *entire* difference between them. Drive the cache through
`write_kv_chunk` one head at a time (head `h` → slot `h`), which is the same op, the same DRAM
`NdShard` geometry and the same `head_dim = 128` a chip performs at TP=8 — this is why the gate can
run on one card even though a *model-level* KV write cannot (P0). So:

- **chunked vs one-shot: PCC ≥ 0.999, asserted per layer.** Given identical inputs these two
  producers should be *exact*; measured **1.00000**. This is a per-op claim and it is not an
  accumulated-depth statistic — see P8's `G-CHUNK-ATTN` for what happens when depth gets folded into
  a mutual-PCC number.
- both paths vs the fp32 golden: **≥ 0.99** (K) / **≥ 0.98** (V, consistently the weaker of the
  two), with the layer-0 error ratio **≤ 3x** the bf8_b storage floor and the per-layer error
  **step ≤ 4x** from layer 3.
- negative control: rope every chunk at `kv_actual_global = 0` and the mutual PCC must collapse
  (measured 0.706 / 0.655).

**Gate `G-GOLDEN`:** `verify_golden_kv.py` runs clean over all 32 layers and prints a per-layer
table; the table goes into `bringup_log/raw/`. It imports no ttnn — the device-vs-golden scoring
lives in `G-CHUNK`. Negative controls: a zeroed layer and a deleted layer must both make it exit
non-zero.

**Gate `G-RUNTIME`:** the runtime satisfies the engine's §2 contract *statically* — every `config`
name present, every engine-called method present with the documented parameters, and **every refusal
loud and matched on its message**. Audit the engine's own call site (an AST walk over
`prefill_runner.py`), not the doc, and give the audit its own negative control.

**Delta 3 cannot run here.** It needs the ring path and TP=8, both of which P7 does not own, so
record it as its own gate row (`G-CHUNK-ATTN`) with verdict `BLOCKED` and a `07_RISKS.md` entry
naming P8 as the owner. Do **not** weaken `G-CHUNK` to cover it, do not move P7 to a multi-device
mesh to make it run, and make the runtime **refuse** the unsupported single-card configuration
loudly instead of silently running a different core.

---

## Phase P8 — Multi-device: TP, SP, and the CCL gates

Only now does the mesh come in. Everything here is about proving the collectives are correct **and**
race-free — and about the fact that the mesh has failure modes the single-card phases cannot show
you.

### Step 1 — open **submeshes**, never a top-level partial mesh

The obvious form — parametrise the `mesh_device` fixture over `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)` as
the template does (`models/demos/minimax_m3/tests/test_factory.py:89`
`parametrize_mesh_with_fabric`) — **cannot work on this galaxy.** Opening `(1,8)` or `(2,8)` as a
*top-level* mesh dies in fabric bring-up:

```
Fabric Router Sync: Timeout after 10000 ms on Device 1. Expected status 0xa2b2c2d2
  (LOCAL_HANDSHAKE_COMPLETE) … furthest-behind stage: STARTED
```

(`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:200`) — the routers on the opened
devices wait for an ethernet handshake with partners *outside* the mesh, which have no kernel
running. Reproduced with and without `TT_MESH_GRAPH_DESC_PATH`, under `STRICT_INIT` and
`RELAXED_INIT`.

**So: open the full `(4,8)` once and `mesh_device.create_submesh(...)` per case**
(`tt_metal/api/tt-metalium/mesh_device.hpp:307`). This is machine-specific, not universal: on a
LoudBox / T3K, `(1,8)` *is* the whole machine and the template's form is correct. A port must switch
back.

### Step 2 — overlapping submeshes need `quiesce_devices()`, and forgetting it **hangs the box**

`tt_metal/api/tt-metalium/mesh_device.hpp:296-305` requires a barrier "between phases that use
overlapping submeshes on the same physical devices" and names `quiesce_devices()` (`:305`). Nothing
enforces it, and `G-TP-PARITY` — which compares `(1,1)` against `(1,TP)` — is exactly such a pair.

Measured, one variable at a time: a `(1,2)` collective then a `(1,8)` collective with both submeshes
live and **no barrier** → **hang**; the same two phases with `parent.quiesce_devices()` between →
fine; `(1,8)` alone in its own process → fine.

Two consequences, and neither is a code comment:

1. **A hang is not contained.** After one, *every* later collective on the box hangs too — including
   a `(4,8)` all-reduce that had passed forty seconds earlier — until `tt-smi -r`. A pytest session
   that hits one turns every remaining gate into a false `FAIL`. **Any harness that can hang must
   run its cases in subprocesses with a timeout**, so a hang becomes a recorded measurement rather
   than a lost session. That harness is `tests/fabric_topology_matrix.py` and its gate is
   **`G-FABRIC-MATRIX`**: sweep (mesh, topology, links, axis), assert each case matches its stated
   expectation, and run it *first*, before any numerical multi-device gate.
2. **The first diagnosis of that hang was wrong, and plausible.** The hanging run was configured as
   the CCL plan prescribes (`Topology.Linear`, `num_links=1`) on a `(1,8)` submesh, and there was a
   tidy story: the system mesh is `MeshShape([8, 4])`, so a logical `(4,8)` row of 8 is linear index
   `r*8 + c` → physical `(idx // 4, idx % 4)` = two physical rows, and a non-cyclic route along that
   axis plausibly does not exist. `(1,8)` + Ring then passed at 1 and 2 links, which *appeared* to
   confirm it. Running `(1,8)` + Linear **alone** falsified the whole story: the variable was the
   overlap, which nobody was varying deliberately. Keep the wrong argument in the log next to the
   measurement that killed it.

### Step 3 — the rest of the phase

3. Add the submesh parametrisations to the P5/P6 unit tests: `(1,2)`, `(1,4)`, `(1,8)`, **`(2,8)`**
   and the target `(4,8)`. `(2,8)` is not optional: `get_default_num_links` returns **1** for any
   single-row mesh (`models/demos/gpt_oss_d_p/utils/general_utils.py:33`), so `(1,N)` parity runs
   `num_links=1` + `Topology.Linear` and **never touches the deployment fabric**. `(2,8)` is the
   cheapest shape that exercises 2-link Ring. Add `device_params` with the right `fabric_config`
   (`FABRIC_1D` vs `FABRIC_1D_RING` — `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:122`
   selects between them; log which and why, and remember a Ring topology on a plain `FABRIC_1D`
   fabric hangs rather than erroring).
4. **Run the TP=8 KV gate before anything sequence-parallel.** `G-KV-TP8` at `(1,8)`: `sp = 1` makes
   the block-cyclic sequence layout the identity, so the only thing under test is the head/feature
   distribution — a failure can only be the mapper. This closes the hole P0 and `G-KV` name.
5. Turn on the collectives in the modules (they were written with the branch in place in P5).
6. Enable the distributed RMSNorm branch **only if** the residual scheme is B.
7. Implement `tt/attention/dense_sp.py` (ring SDPA over the block-cyclic SP cache), using
   `ccl_manager.ring_attention_ccl_semaphore_handles` and
   `ccl_manager.ring_attention_ccl_core_grid_offset`. Note the constraint from
   `models/demos/gpt_oss_d_p/tt/ccl.py`: **the ring-attention CCL workers and the SDPA compute cores
   must not overlap**; the CCL workers take the last compute column and the offset must derive from
   the real `compute_with_storage_grid_size()` — while the SDPA program grid stays 8×8 (P5.5).

**The ring op requires `fp32_dest_acc_en=False`, structurally.** This is not a preference and not a
regression of §2.4: `use_streaming_compute = !fp32_dest_acc_en`
(`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1304`) and
`kv_actual_isl` requires the streaming path (`:1306`), so for chunked prefill the two flags are
mutually exclusive by construction and `True` is refused with a `TT_FATAL`. Measured cost: the ring
op alone sits **7.98x** off its noise floor (against the single-card SDPA's 71x, §2.3), and end to
end the chunked path carries **1.45x** the error of the one-shot path (min K 0.99695 vs 0.99789).
Expect it, attribute it, and set any future KV threshold against the **chunked** number.

### Gates

- **`G-FABRIC-MATRIX`** — the (mesh, topology, links, axis) sweep of steps 1–2, subprocess-isolated
  with a timeout. Every case matches its stated expectation, *including* the ones expected to fail
  or hang. Run it before every other P8 gate.
- **`G-KV-TP8`** — the model → cache path at TP=8 on `(1,8)`. **Head `c` → mesh column `c` asserted
  bit-exactly** (`rtol=atol=0`), plus 32 layers of model-produced K/V vs the fp32 golden at
  `G-CHUNK`'s carried-over thresholds (**K ≥ 0.99, V ≥ 0.98**, layer-0 error ratio ≤ 3x) and
  written-region-only checks. Carrying P7's thresholds rather than picking fresh ones is deliberate:
  a threshold chosen here would be fitted to this measurement and could not fail, whereas carrying
  them makes the TP split's cost readable directly. Negative control: read column `(c+1)%8` as head
  `c`. **Gate the mapping on bit-equality, not PCC** — that control still scores **0.99890** (§2.5).
- **`G-SP-RING`** — `dense_sp_attention` **alone** vs an fp32 torch reference on the same values,
  **PCC ≥ 0.99**, with the `fp32_dest_acc_en` A/B recorded (including the `TT_FATAL` text when
  `True` is refused) and the error ratio to its own floor reported.
- **`G-CHUNK-ATTN`** — the P7 blocker, now runnable: chunk *k*'s queries attending the prefix read
  back out of the cache. **The threshold must name a depth.** Assert **≥ 0.999 at layer 1** (one
  attention layer) — that is the per-op claim. Gate the deep layers with the per-layer error **step**
  (≤ 4x) and both paths' PCC against the fp32 golden at `G-CHUNK`'s thresholds; record the
  accumulated min over 32 layers without gating it. The numbers say why:

  | layer | ring vs one-shot, K |
  |---|---|
  | 0 (no attention has run) | **1.00000** |
  | 1 (**one** attention layer) | **0.99996** |
  | 8 | 0.99952 |
  | 22 (the min) | **0.99628** |

  Layer 22's K is one attention output pushed through 22 residual streams, each amplifying the
  difference; holding it to a per-op threshold measures **depth**, not the op. The general rule,
  which also binds `G-CHUNK` and `G-MODEL`: **a mutual-PCC gate must name the depth at which it
  applies.** "Path A == path B" is a per-op claim; the min over a 32-layer stack is a different
  quantity and needs a different instrument.
- **`G-TP-PARITY`** — for each module, the multi-device output must match the single-device output to
  **PCC ≥ 0.999**. Collectives are mathematically exact up to reduction order; a large drop here is a
  sharding bug, not precision. Test by running the same module with the same weights on `(1,1)` and
  on the multi-device shape and comparing **device outputs to each other** (not just each to torch)
  — sharper than PCC-vs-torch because it removes the reference's own error. Run all five shapes:
  `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)`. At SP > 1 the multi-device output is a token slice,
  so compare it against the corresponding slice of the `(1,1)` output — the TP claim is unchanged
  and the extra rows are what put the 2-link Ring transport under test. Negative control: rotate the
  reference by one TP shard (≤ 0.95). This gate holds two overlapping submeshes at once:
  `quiesce_devices()` between phases is mandatory.
- **`G-RACE`** — run the full-model KV PCC harness **three times in one process on one
  `CCLManager`** and assert the results are **bit-identical**. Non-determinism here means a
  semaphore is being reused while in flight — check the ping-pong cycling and that `CCLManager` is
  constructed once, not per layer. Log all three hashes. Note the scope of a pass: hundreds of
  all-reduces is not hundreds of thousands, and it says nothing about multi-user slots.
- **`G-SEMAPHORE`** — assert `CCLManager` allocates its CCL state once: instantiate the model and
  check the semaphore list lengths equal the constants (not `n_layers ×` them), at construction,
  after dozens of getter cycles, and after a real multi-layer harness run.
- **`G-MESH-KV`** — `tests/galaxy_prefill_kv_pcc.py` on the target mesh: per-layer K/V PCC vs golden,
  one-shot and chunked, at more than one chunk size. Record the **min across layers** for K and V,
  per run configuration, in a status table in the package `README.md` (the `minimax_m3` README's
  "Status" table is the format).
- **`G-WEIGHTS` (P8 extension)** — re-run the cache-only assertion **at TP=8**, where the cache is
  actually sharded. `ttnn.as_tensor` caches the already-sharded tensor, so a stale or wrong-shape
  cache presents as "one layer runs on garbage" and is first visible here, not at `G-WEIGHTS`.

---

## Phase P10 — Disaggregated-prefill integration

**This phase runs before P9.** See the note at the top of Phase P9.

**Read these two documents in full before writing anything in this phase:**

- `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` — the adapter + runtime contract
  (its own §1 adapter, §2 runtime, §3 registration, §4 validation, plus the closing checklist —
  section numbers there, not this recipe's).
- `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` — the three config files, the two
  gates, and what a `PASS` there does and does not prove.

**Two things the contract doc does not say, both of which cost a run to find:**

1. **§2's `prefill_chunk` signature is incomplete.** The engine also always passes `d2h_service`
   **and `metadata_msg`** (`models/demos/common/prefill/runners/prefill_runner.py:364`). A runtime
   written to the doc dies with a `TypeError` on its first served chunk — after the mesh is open and
   the weights are loaded. Audit the engine's real call site, not the prose (that is `G-RUNTIME`).
2. **The engine mutates the config `load_hf_config` returns** — it assigns `max_seq_len` on the next
   line (`models/demos/common/prefill/runners/prefill_runner.py:477`). A frozen dataclass cannot be
   returned as-is; return a mutable subclass, and do not discover this at serving time.

Working template to mirror end to end: `models/demos/gpt_oss_d_p/tt/runners/` —
`adapters/gpt_oss.py:41` (`GptOssPrefillAdapter`), `kv_chunk_table.py`,
`manifests/gpt_oss_d_p.json`.

### Steps

1. **`tt/runners/adapters/llama.py`** — subclass `PrefillModelAdapter`
   (`models/demos/common/prefill/adapter.py:104`). Set `name = "llama31_8b_d_p"`, `model_config`,
   `hf_model_default`, `ttnn_cache_default`, `prefill_trace_default`, `l1_small_size`,
   `supports_dflash = False`. (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:45-49` shows
   five of them, and what an empty default means: `ttnn_cache_default = ""` ⇒ no cache,
   `prefill_trace_default = ""` ⇒ trace must come from `PREFILL_TRACE_DIR`.) Implement
   `load_hf_config`, `weight_cache_path(mesh_shape)`,
   `allocate_kv_cache(*, mesh_device, hf_config, params)` returning a `KvCaches` subclass, and
   `build_runtime(*, mesh_device, hf_config, params)`. Read knobs from `params`
   (`PrefillRunParams`, `adapter.py:46`), **never** from `os.environ`.
   **Keep the module import-light** — no torch, no ttnn, no reference model at module scope,
   including via a convenience helper. The H2D producers import adapters, and P9 gates this.
2. **`tt/runners/manifests/llama31_8b_d_p.json`** — `{ "env": { "PREFILL_MODEL": "llama31_8b_d_p" } }`.
   Pin what the deployment needs (the mesh-graph descriptor for the torus, the fabric config) and
   nothing that belongs to the caller (prompt, chunk count).
3. **Register** in `ADAPTER_PATHS` in `models/demos/common/prefill/adapter.py`:
   `"llama31_8b_d_p": "models.demos.llama31_8b_d_p.tt.runners.adapters.llama:LlamaPrefillAdapter"`
   (the dict starts at `adapter.py:277`). One line; the import stays lazy.
4. **`tt/runners/kv_chunk_table.py`** + the runtime's optional migration hooks
   (`build_kv_chunk_table`, `kv_migration_base_address`, `set_layer_ack_channel`) using
   `serialize_kv_chunk_table` from `models/demos/common/prefill/runners/migration.py`. Anything you
   do not implement — the multi-rank merge, for one — must **raise**, naming its risk id, rather than
   silently discarding an argument.
5. **Wire the producer's KV read-back for your cache layout.** The device-less reader that powers
   `PREFILL_PRODUCER_CHECK_PCC` is **not** adapter-dispatched through the adapter object — it
   branches on `ADAPTER.name` in
   `models/demos/common/prefill/runners/prefill_producer.py`'s `_read_slot_kv_and_check_pcc`
   (`:511`):
   ```python
   _PACKED_GQA_MODELS = ("gpt_oss_d_p", "llama31_8b_d_p")   # prefill_producer.py:508
   ...
   if ADAPTER.name == "minimax_m3":        return _read_slot_kv_and_check_pcc_m3(...)
   if ADAPTER.name in _PACKED_GQA_MODELS:  return _read_slot_kv_and_check_pcc_gpt_oss(...)
   return _read_slot_kv_and_check_pcc_mla(...)              # DeepSeek / Kimi merged MLA
   ```
   The MLA fallback (`:696`) is **wrong for Llama**, so without a branch this gate silently checks
   the wrong bytes. `_read_slot_kv_and_check_pcc_gpt_oss` (`:544`) is the **plain packed-K/V,
   block-cyclic GQA reader** — exactly Llama's layout — so read it first and, if your `kv_cache.py`
   keeps gpt-oss's `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` and packing, add `llama31_8b_d_p` to that
   branch rather than writing a fourth reader. (The older two-layout wording in
   `ADDING_A_PREFILL_MODEL.md` §4 predates the gpt-oss reader; the code is the current truth.) This
   touches shared code: record a `DEC` plus an entry in `08_PREFILL_INTEGRATION.md`, generalise the
   name check rather than duplicating the function, and make the reader's log line name
   `ADAPTER.name` instead of a hard-coded model.
6. Populate the weight cache and stage a golden trace (P7's script output).

### Gates

- **`G-ADAPTER`** — the checklist at the end of `ADDING_A_PREFILL_MODEL.md`, item by item, each with
  evidence. Plus: zero abstract methods left; `PREFILL_MODEL=llama31_8b_d_p` resolves through the
  registry; the registry-fed pytest `variant` fixture
  (`models/demos/deepseek_v3_d_p/tests/conftest.py:365`) picks it up; every `model_dims` constant
  equals `config.json`; and **adapter import is measured** — time it and assert no heavy module
  landed in `sys.modules`.
- **`G-REQUEST`** — the two-terminal request-mode run from `ADDING_A_PREFILL_MODEL.md` §4:
  ```bash
  # terminal A — runner
  PREFILL_MODEL=llama31_8b_d_p PREFILL_SP=<sp> PREFILL_TP=<tp> PREFILL_H2D_SERVICE_ID=llama_prefill \
    python -m models.demos.common.prefill.runners.prefill_runner
  # terminal B — producer
  PREFILL_MODEL=llama31_8b_d_p PREFILL_SP=<sp> PREFILL_TP=<tp> PREFILL_H2D_SERVICE_ID=llama_prefill \
  PREFILL_PRODUCER_CHUNKS=11 \
    python -m models.demos.common.prefill.runners.prefill_producer
  ```
  Every shared variable must match on both sides or the byte layout disagrees;
  `PREFILL_MAX_SEQ_LEN ≥ chunks * PREFILL_CHUNK_SIZE`. Choose `max_seq_len` **strictly greater** than
  `chunk_size`: at equality the per-chip cache shard leaves the ring op no room and attention falls
  back to the one-shot bootstrap — a correct but *different* core, so anything you measure is
  measuring the wrong path. `PASS` = every chunk accepted and served, the shutdown sentinel
  received, clean exit.
- **`G-MOCK-MIG`** (= the doc's **Gate 1**) — `PREFILL_MOCK_MIGRATION=1` on the runner (single-rank
  only) + `PREFILL_PRODUCER_CHECK_PCC=1` on the producer. Expect
  `[producer] KV cache PCC PASSED` (default threshold `PREFILL_STANDALONE_CHUNKED_PCC` = 0.93 —
  record the *measured* per-layer minimum, not just the pass). This gate proves both that prefill
  writes correct KV and that `build_kv_chunk_table` is correct — **and it is the strongest evidence
  in the whole bring-up**, because it is a second, device-less reader in a different process
  agreeing with the on-device `G-MESH-KV` number at the same shape. Compare the two explicitly.
- **`G-KV-TABLE`** — the address table on its own, which `G-MOCK-MIG` cannot isolate: one PCC over
  one slot cannot separate a wrong address table from a numerical problem. Assert the protobuf round
  trip, head → config → chip, position → address and K/V separation by reading DRAM back over UMD
  and comparing **bit-exactly** (`torch.equal`), with a negative control that reads one head through
  another's config. Same reasoning as `G-KV-TP8`: a mapping claim needs bit-equality, not
  correlation (§2.5).
- **`G-LOOPBACK`** (= the doc's **Gate 2**) — real DRAM→transport→DRAM copy via `migration_driver`
  with `--verify-migration dst-bytes`. **Requires the tt-llm-engine binaries.** It verifies the
  *engine's* model-agnostic byte copy, not this model, so it is legitimate to declare it
  out-of-scope with a `DEC` — but then the residual gap must be enumerated as a named risk, and the
  unimplemented multi-rank path must raise (step 4). If you neither run it nor scope it out, it is
  `BLOCKED` with the reason in `07_RISKS.md`. Do not fake it.

Record in `08_PREFILL_INTEGRATION.md`: the contract mapping (each abstract method → your
implementation, `path:line`), the full env matrix used, verbatim gate transcripts, and the
limitations you inherited (loopback-only verification; cross-talk invisible with one prompt unless
`PREFILL_PRODUCER_SLOT_TRACES` is used; a layer subset makes a `PASS` a sample).

---

## Phase P9 — Cleanliness gate

**Run this last, after P10.** P9's gate is a whole-package sweep: no TODOs without a filed issue,
every env var in the README's table, every `tt/` module owning a test, `README.md` complete, and
import hygiene on the adapter. **P10 adds `tt/runners/adapters/`, a manifest, a KV-chunk-table module and new env vars** —
so a P9 that precedes it audits a package that is about to change, and has to be redone. Worse, one
of P9's own items (item 8 below) is **unrunnable before P10**, because no adapter exists yet. The
gate index keeps P9's numbering; only its position in the run changes.

Cleanliness is a deliverable of this iteration, not a courtesy. Run it as a real gate, not a vibe
check.

1. `pre-commit run --files $(git diff --name-only main...HEAD)` — clean.
2. Every new file has the SPDX header pair.
3. `grep -rn "TODO\|FIXME\|XXX\|HACK" models/demos/llama31_8b_d_p/` — every hit is either resolved or
   has a `07_RISKS.md` entry with a filed-issue reference.
4. `grep -rn "except.*:\s*pass\|except Exception" models/demos/llama31_8b_d_p/` — none on a
   correctness path (a bring-up probe that must never break a run is the only allowed case, and it
   logs). List the survivors and say why each is off the correctness path.
5. No unused imports, no dead branches, no commented-out code, no leftover `print` (use `loguru`).
6. Every `os.environ` / `os.getenv` read in the package appears in the `README.md` env-var table.
   Generate the list with a grep, not by hand — a hand-written list misses the ones that matter.
7. `README.md` is complete: architecture table, deployment path, status table with measured PCC, run
   commands, env-var table, layout section, the "why not `models/common/`" answer (P2), and a "what
   is not implemented" section. Model it on `models/demos/minimax_m3/README.md`.
8. Import hygiene: `python -c "import models.demos.llama31_8b_d_p.tt.runners.adapters.llama"` must
   be **cheap** — no reference-model, device, or runtime imports at module load
   (`ADDING_A_PREFILL_MODEL.md` requires this; the H2D producers import adapters). Measure it
   against the template as a ratio.
9. Test inventory: every `tt/` module has a corresponding test. List the mapping in `06_GATES.md`
   and **close** gaps rather than flagging them.
10. `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` — 0 mismatched, 0 unresolved,
    over the logs, the recipe, the README **and** the package's own docstrings (§1.6).
11. Re-run the whole test suite after P9's own edits, and re-run at least two numerical gates in
    isolation to confirm they still reproduce their recorded digits.

**Gate `G-CLEAN`:** all eleven items, each with its command and output recorded. Expect this sweep to
find real defects in the *logs* as well as the code — stale checklist rows, duplicated raw logs, a
risk register out of date, and un-verified citations in prose that no earlier phase scanned. Fix
them; they are part of the deliverable.

---

## Appendix A — Gate index and thresholds

Every threshold below was set by the method in §2; §2.1 shows what the plausible guesses would have
been and how far off they are. **Every numeric gate additionally records its
input distribution, its reference dtype policy, its computed noise floor, the error ratio to that
floor, and a negative control** (§1.4). Absolute thresholds are floors you must clear; clearing one
while sitting far off the noise floor is a finding, not a `PASS`.

| Gate | Phase | Proves | Threshold | Device |
|---|---|---|---|---|
| `G-CARD` | P0 | every fact has provenance | doc review + citations clean | — |
| `G-REF` | P1 | reference is deterministic and self-consistent | bit-identical ×2; cross-ref PCC ≥ 0.9999 (expect bit-exact) | host |
| `G-SURVEY` | P2 | reuse decided with citations | doc review + citations clean | — |
| `G-OUTLINE` | P3 | file tree + shapes pinned; every gate owns a file | doc review | — |
| `G-CCL-PLAN` | P4 | every collective placed and justified | doc review | — |
| `G-MESH` | P5.1 | MeshConfig arithmetic + refusals; semaphores allocated once | exact asserts | 1 card |
| `G-RMS` | P5.2 | RMSNorm | **PCC ≥ 0.9999**; gap to floor recorded | (1,1) |
| `G-ROPE` | P5.3 | RoPE + llama3 scaling active | PCC ≥ 0.999; control must collapse | (1,1) |
| `G-MLP` | P5.4 | dense SwiGLU | **≥ 0.999 @bf8_b, ≥ 0.9995 @bf16**, ≤ 3x floor | (1,1) |
| `G-ATTN` | P5.5 | GQA + RoPE + causal SDPA + o_proj | **PCC ≥ 0.999**; own stages ≤ 3x floor, block ≤ 8x | (1,1) |
| `G-KV` | P5.6 | cache **primitive**: write correctness, no collateral writes | PCC ≥ 0.99 @bf8_b, ≤ 3x floor; positional read-back **bit-exact** | (1,1) |
| `G-LAYER` | P6.1 | decoder layer (integration check) | **PCC ≥ 0.999**, ≤ 8x floor | (1,1) |
| `G-WEIGHTS` | P6.2 | no missing/unused keys; cache-only rebuild identical; loader bit-exact | exact | 1 card |
| `G-MODEL` | P6.3 | full stack hidden states; top-1 (integration check) | **≥ 0.999**, ≤ 8x floor, per-layer step ≤ 4x from L3; 100% top-1 | (1,1) |
| `G-CHUNK` | P7 | chunked ≡ one-shot for **deltas 1–2** (indexed RoPE, chunk write), and both vs the fp32 golden | ≥ 0.999 mutual **per layer** (expect exact); ≥ 0.99 K / ≥ 0.98 V vs golden; L0 ratio ≤ 3x; step ≤ 4x | (1,1) |
| `G-GOLDEN` | P7 | golden trace structure is sound over all layers | clean table; generator + verifier exit 0; streamed driver == HF's own loop bit-exactly | host (imports no ttnn) |
| `G-RUNTIME` | P7 | the runtime satisfies the engine's §2 contract, statically | every name and parameter present; every refusal matched on its message | none |
| `G-FABRIC-MATRIX` | P8 | which (mesh, topology, links, axis) combinations can run a collective | every case matches its stated expectation, subprocess-isolated | target mesh |
| `G-KV-TP8` | P8 | the **model → cache** path at TP=8 | head→column **bit-exact**; K ≥ 0.99 / V ≥ 0.98 vs golden; L0 ratio ≤ 3x; rotated-column control | (1,8) |
| `G-SP-RING` | P8 | the ring SDPA alone, and the `fp32_dest_acc_en` A/B | PCC ≥ 0.99 vs fp32 torch; ratio to its own floor recorded | (4,8) |
| `G-CHUNK-ATTN` | P8 | chunk *k* attending the prefix read out of the cache | **≥ 0.999 at layer 1**; deep layers gated by step ≤ 4x; both vs golden | (4,8) |
| `G-TP-PARITY` | P8 | collectives are exact | PCC ≥ 0.999 vs single-device on 5 shapes incl. (2,8); control ≤ 0.95 | submeshes |
| `G-RACE` | P8 | no semaphore races | 3 runs bit-identical, one process, one `CCLManager` | target mesh |
| `G-SEMAPHORE` | P8 | CCL state allocated once | exact list lengths | target mesh |
| `G-MESH-KV` | P8 | full-model KV vs golden on target mesh | per-layer min recorded; K ≥ 0.99 / V ≥ 0.98 | target mesh |
| `G-WEIGHTS` (P8 ext) | P8 | cache-only rebuild **at TP=8**, where the cache is sharded | every device tensor SHA-256-identical | target mesh |
| `G-ADAPTER` | P10 | engine contract satisfied; adapter import stays cheap | checklist; 0 abstract methods; no heavy module at import | — |
| `G-REQUEST` | P10 | request-mode serving works | every chunk served; clean shutdown | target mesh |
| `G-MOCK-MIG` | P10 | KV + chunk table correct (doc Gate 1) | producer PCC ≥ 0.93 (`PREFILL_STANDALONE_CHUNKED_PCC`); measured min recorded | single rank |
| `G-KV-TABLE` | P10 | the address table itself, isolated from numerics | **bit-exact** over UMD read-back; control must fail | target mesh |
| `G-LOOPBACK` | P10 | real migration copy (doc Gate 2) | `dst-bytes` identical | + engine binaries |
| `G-CLEAN` | P9 | cleanliness (11 items) | all pass | — |

Add a **per-phase regression gate** as well: the whole package suite, 0 failed, after each phase's
additions. It is cheap and it is the only thing that catches a new phase breaking an old gate's test
file rather than its numbers.

### A.2 Where these thresholds come from

Not from another implementation's README, and not from another implementation's PCC — §2.1 explains
why that is unsound, and §2.2 gives the method that replaced it. In short:

- **Absolute PCC thresholds** are set one to two orders of magnitude tighter than the values this
  recipe originally guessed (`0.999`/`0.99` copied from `models/demos/gpt_oss_d_p/README.md`'s
  tier-1 line and `models/tt_transformers/tests/test_mlp.py`), because measurement showed the
  guesses were loose enough to pass an attention block carrying a 38.7x precision regression —
  the same misconfiguration costs 96x–1168x on a bare matmul (§2.4).
- **Error-ratio budgets** (3x stage / 8x block / 4x depth step) come from measurement of stages we
  implement ourselves, and are stated *before* the measurement they gate.
- **`0.93`** is the disaggregated producer's own default (`PREFILL_STANDALONE_CHUNKED_PCC`,
  documented in `PREFILL_MIGRATION_TESTING.md` Gate 1) and is the engine's number, not ours.
- **Layout, mapping and address claims are gated on bit-equality**, never PCC (§2.5).

Whole-model KV numbers degrade with depth and cache dtype. Set your whole-model thresholds from your
own measured bf16 baseline and log the bf16→bf8_b delta as the justification. **Never lower a
threshold to make a test green** without a `DEC` that states the measured value, the suspected cause,
and the follow-up — and never *raise* one after seeing the measurement either, which is the same
error with a friendlier face.

---

## Appendix B — Failure playbook

Consult before opening a debugging session; each of these has a first move that is not "stare at the
matmul".

| Symptom | Most likely cause | First move |
|---|---|---|
| Attention PCC ~0.5–0.9, norms fine | RoPE convention mismatch (HF halves vs Meta interleaved), or Q/K weights not `reverse_permute`d | Build both cos/sin tables from one frequency set, as `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py::_build_cos_sin` does; test RoPE alone (`G-ROPE`) |
| RoPE wrong at **every** position, nothing raised | `getattr(cfg, "rope_theta", DEFAULT)` returned the default (P1 trap 1) | Route θ through `get_rope_theta` on a **dict** and assert non-`None` |
| PCC good at short seq, bad past ~8192 | llama3 RoPE scaling not applied | Assert scaled ≠ unscaled `inv_freq` beyond `original_max_position_embeddings` |
| A module clears its gate but sits 20x+ off its floor | a silently wrong `compute_kernel_config` — most often an inherited `fp32_dest_acc_en=False` | A/B the flag in-suite (§2.4); isolate the fused kernel before suspecting your own stages (§2.3) |
| Multi-device PCC varies run to run | semaphore reused while in flight; `CCLManager` built per layer | `G-RACE` + `G-SEMAPHORE`; check the ping-pong cycling; deepen the barrier ring from 2 to 4 |
| Ring-SDPA assert `ccl_core_grid_offset.x >= sdpa_grid.x` | SDPA program grid **derived** from the device grid instead of pinned to 8×8 | Pin it and assert `sdpa_grid.x <= grid.x - 1` at construction (P5.5) |
| The whole box hangs, and stays hung | two overlapping submeshes with no `quiesce_devices()` between them | `tt-smi -r`; add the barrier; run hang-capable cases in subprocesses with a timeout (P8 step 2) |
| Fabric router sync timeout on a `(1,8)`/`(2,8)` open | a top-level partial mesh on this galaxy | Open `(4,8)` and `create_submesh` (P8 step 1) |
| `update_padded_kv_cache`: "cache and input num-heads dim must match" | TP ≠ `num_key_value_heads` | There is one legal TP here, and it is 8 (P0) |
| A layout/mapping test passes and the layout is still wrong | it was gated on PCC, which a permutation barely moves (0.99890) | Re-gate on `torch.equal`, `rtol=atol=0` (§2.5) |
| A probe fails with "greatest relative difference 1/257" | integer-valued bf16 tensor above 256 | Keep probe values ≤ 256 or split the id across lanes (§2.5) |
| A mutual-PCC gate fails only at depth | the metric is measuring depth, not the op | State the depth the threshold applies at; gate deep layers on the error step (P8, `G-CHUNK-ATTN`) |
| One layer runs on garbage; others fine | state-dict key renamed / not consumed, **or** a weight cache written at a different mesh shape | `G-WEIGHTS` (no missing, no unused keys) and its TP=8 extension |
| A *step* in the per-layer PCC curve | a specific sublayer's logic error | The per-layer delta probe: a growing signed-mean localises it |
| Cache-only build silently wrong | an un-cached weight (biases, sidecars) has no source when `state_dict` is empty | Fail loud, do not default — `models/demos/minimax_m3/tt/mlp.py` raises rather than running bias-free |
| Runner logs `device map ... not found; skipping KV read` | the device-map sidecar was not published; every PCC silently vanished | `PREFILL_MIGRATION_TESTING.md` Gate 1 — check `serialize_device_map`, and clear stale `/tmp` maps |
| `TypeError` on the first served chunk | `prefill_chunk` written to the doc's signature, which omits `d2h_service` / `metadata_msg` | P10, and `G-RUNTIME`'s AST audit of the real call site |
| Producer PCC computed over plausible-but-wrong bytes | `ADAPTER.name` not added to the packed-GQA branch; the MLA reader ran | P10 step 5 |
| `PREFILL_*` set in the shell has no effect under `tt-run` | `tt-run` forwards only `TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes | Set it in the binding's `global_env` or the model manifest |
| Producer ack drain hangs | `PREFILL_NUM_LAYERS` differs between runner and producer (ack count = layers × chunks) | Pin the real depth on both |
| Chunk write asserts | `kv_actual_global % 32 != 0`, or `CHUNK_SIZE % (SP*32) != 0`, or `MAX_SEQ_LEN % CHUNK_SIZE != 0` | Re-check the P0 shape arithmetic |
| Everything passes but the numbers look too good | you measured the SP bootstrap because `max_seq_len == chunk_size` | Make `max_seq_len > chunk_size`; log which core actually ran |

---

## Appendix C — Definition of done for this iteration

1. `models/demos/llama31_8b_d_p/` contains the P3 tree, no dead files.
2. Every gate in Appendix A is `PASS` (or `PASS-WITH-DEVIATION` with a `DEC`, or explicitly scoped
   out with one), recorded in `bringup_log/06_GATES.md` with raw logs. **A gate with no raw log did
   not happen.**
3. `G-LOOPBACK` is `PASS`, `BLOCKED` with a stated reason, or out-of-scope by a `DEC` whose residual
   gap is a named risk.
4. `bringup_log/` reads as a coherent narrative: a reviewer can reconstruct every judgement call,
   its alternatives, and its evidence, without reading the code.
5. `README.md` carries a status table with measured PCC numbers, the "why not `models/common/`"
   answer, and a "not implemented" section.
6. `07_RISKS.md` lists every `UNVERIFIED` fact, every gap, and every follow-up, each with an owner
   slot and (where applicable) a filed issue — and its summary table agrees with its own body.
7. `verify_citations.py` reports 0 mismatched and 0 unresolved across the logs, the recipe, the
   README and the package's docstrings.

---

## Provenance — what the executed run changed about this recipe

*Not instructions. A changelog, kept so the corrections above can be audited against the run that
produced them.* The full record is `bringup_log/` (`05_DECISIONS.md` for the reasoning,
`06_GATES.md` for the numbers, `07_RISKS.md` for what is still open); the resulting package is
described in [`README.md`](README.md). This recipe was executed once, end to end, on the machine
described above: 11 phases, every gate in Appendix A, 169 tests passing.

The corrections the run forced, and where they now live:

1. **Thresholds.** Every guessed PCC threshold was one to two orders of magnitude too loose; the
   noise-floor method (§2) replaced "copy a README's number", and an `fp32_dest_acc_en` regression
   costing 38x–1168x depending on the op — which the original `G-ATTN` threshold would have passed
   as clean — is what proved it.
2. **Phase order.** P10 now runs before P9, because P9 audits a package P10 changes and one P9 item
   is unrunnable before an adapter exists.
3. **`TP == num_key_value_heads`** was found to be an equality, not a bound — which retired a
   `(8,4)`/TP=4 fallback and exposed that `G-KV` at `(1,1)` tests a configuration the model never
   produces (§P0, `G-KV-TP8`).
4. **Silent-wrongness traps** that no reading of the templates would have caught: the
   `getattr(cfg, "rope_theta", …)` substitution on transformers 5.12.1 (P1), a derived SDPA grid
   that passes every single-card gate and fails only at SP > 1 (P5.5), `quiesce_devices()` (P8),
   and a PCC-based layout control that passes on a rotated mapping (§2.5).
5. **Machine-specific facts** — the (12,10) compute grid, submeshes instead of top-level partial
   meshes, `num_links` on single-row meshes — were promoted from discoveries to §The machine and P8.
6. **Gates added by the run:** `G-RUNTIME`, `G-FABRIC-MATRIX`, `G-KV-TP8`, `G-SP-RING`,
   `G-CHUNK-ATTN`, `G-KV-TABLE`, plus four test files for gates that previously had nowhere to live.
7. **Orchestration hygiene** (§0.2) was written from three self-inflicted incidents in this run.
8. **Two claims were retired as wrong, not merely refined:** that a layer PCC launders a degraded
   sublayer (P6.1 — the rule survives on different grounds), and that scheme B is unproven (P4 — the
   real argument is cost equivalence on a dense model).
9. **Still open:** the multi-rank KV-chunk-table merge is untested and the code raises rather than
   guessing — `07_RISKS.md` R-040.

**Where the earlier appendix labels went**, for the ~150 references to them in `bringup_log/` and in
the package's docstrings:

| was | now |
|---|---|
| Appendix D (this machine) | §The machine this recipe was written for |
| Appendix D (SDPA GQA, program config) | P5.5 |
| Appendix D (transformers 5.x wrapper caveat) | P1, trap 5 |
| Appendix E (oracle baselines, caveat) | §2.1; the caveat's *justification* is corrected in P6.1 |
| Appendix E.1, E.2 | §2.1, §2.2 |
| Appendix E.3, E.4 | §2.4 |
| Appendix E.5 | §2.3 |
| Appendix E.6 | §2.5 |
| Appendix F.1 | §The machine |
| Appendix F.2 | P1, "transformers 5.12.1 — five traps" |
| Appendix F.3 | P2, "`models/common/` — evaluated" |
| Appendix F.4 | P4, `MeshConfig` |
| Appendix F.5 | P4, residual-layout decision |
| Appendix F.6 | P0 step 5, P5.6, P8 `G-KV-TP8` |
| Appendix F.7 | §1.6 |
| Appendix F.8 | P5.5, §2.4 |
| Appendix F.9 | P3 |
| Appendix F.10 | P4, P8 step 3, `G-WEIGHTS` (P8 ext) |
| Appendix F.11 (hygiene) | §0.2 |
| Appendix F.11 (submeshes) | P8 step 1 |
| Appendix F.12 (`quiesce_devices`) | P8 step 2 |
| Appendix F.12 (phase order) | the phase map, and Phase P9's opening |
| Appendix F.13 (`G-CHUNK-ATTN` depth, ring flag) | P8 |
| Appendix F.13 (P10 executed) | P10 |
