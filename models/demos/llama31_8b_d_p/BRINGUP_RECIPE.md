<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bring-up recipe — `models/demos/llama31_8b_d_p`

**Target:** a clean, functional, PCC-verified TTNN **prefill** implementation of Llama-3.x 8B
in `models/demos/llama31_8b_d_p`, module-by-module verified against a torch/HF reference,
with CCL living inside the modules, ending in integration with the model-agnostic
disaggregated-prefill engine (`models/demos/common/prefill/`).

**Who this is for:** an agent (Claude) executing the bring-up end to end. Read this file top to
bottom once, then execute phases **in order**. Do not skip a phase. Do not start a phase whose
predecessor's gate has not been recorded as `PASS` (or `PASS-WITH-DEVIATION` + a `DEC` entry).

**Non-goals for this iteration:** decode, performance optimisation, trace/2CQ, multi-galaxy
pipeline parallel, quantised weights. Functional correctness + cleanliness + tests only.

---

## Start here (first 5 minutes)

```bash
cd /home/mstojkovic/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
mkdir -p models/demos/llama31_8b_d_p/bringup_log/raw

# the four documents to read before writing any code (in this order)
sed -n '1,120p' models/demos/minimax_m3/README.md
sed -n '1,95p'  models/demos/gpt_oss_d_p/README.md
cat models/demos/minimax_m3/tt/dense_mlp.py                      # the Llama MLP template
cat models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py # the unit-test template
```

Then execute P0 → P10 below, in order. The phase map:

| Phase | Produces | Gate(s) | Device |
|---|---|---|---|
| **P0** Model card | `bringup_log/00_MODEL_CARD.md`, package skeleton | `G-CARD` | none |
| **P1** Reference | `01_REFERENCE.md`, `tests/test_factory.py`, `conftest.py`, bundled `config.json` | `G-REF` | host |
| **P2** Survey | `02_SURVEY.md` (reuse-vs-write table) | `G-SURVEY` | none |
| **P3** Outline | `03_OUTLINE.md` (file tree + shapes) | `G-OUTLINE` | none |
| **P4** CCL plan | `04_CCL_PLAN.md` (collective placement + residual scheme) | `G-CCL-PLAN` | none |
| **P5** Modules | `tt/{config,ccl,rms_norm,rope,mlp}.py`, `tt/attention/*`, `utils/*` + unit tests | `G-MESH` `G-RMS` `G-ROPE` `G-MLP` `G-ATTN` `G-KV` | 1 card |
| **P6** Assembly | `tt/{layer,embedding,model_config,model}.py` + tests | `G-LAYER` `G-WEIGHTS` `G-MODEL` | 1 card |
| **P7** Chunked + golden | `tt/tt_prefill_runtime.py`, `scripts/*` | `G-CHUNK` `G-GOLDEN` | 1 card |
| **P8** Multi-device | collectives enabled, `tt/attention/dense_sp.py`, `tests/galaxy_prefill_kv_pcc.py` | `G-TP-PARITY` `G-RACE` `G-SEMAPHORE` `G-MESH-KV` | mesh |
| **P9** Cleanliness | `README.md`, lint clean | `G-CLEAN` | none |
| **P10** Disagg prefill | `tt/runners/{adapters,manifests,kv_chunk_table}` + one line in `models/demos/common/prefill/adapter.py` | `G-ADAPTER` `G-REQUEST` `G-MOCK-MIG` `G-LOOPBACK` | mesh |

Keep this checklist in `bringup_log/06_GATES.md` and tick it as you go. If you must stop early, stop
**on a gate boundary** and leave `06_GATES.md` stating exactly which phase is next.

---

## 0. The agent contract (read first)

Five rules govern the whole run. They are not advice.

1. **Sequential.** Phases P0 → P10. Each phase ends in a *gate*. A gate is a command with a
   numeric threshold and a recorded verdict. No forward progress on a `FAIL`.
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
- Single-card work (P0–P6) needs **one** Wormhole/Blackhole card. Multi-device work (P7+) needs the
  real mesh.
- Never `git commit` unless explicitly asked. Never push. Work on the current branch.

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
| G-RMS | P5 | RMSNorm vs torch, 1x1 mesh | PCC ≥ 0.999 | 0.99998 | PASS | 2026-09-03 | `raw/G-RMS_20260903T101500Z.log` |
```

```markdown
### G-RMS — RMSNorm vs torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_rms_norm_vs_ref.py -x -q`
- **Mesh / device:** (1,1), Wormhole N150
- **Inputs:** seq_len ∈ {32, 512, 4096}, hidden 4096, random weights, seed 0
- **Threshold:** PCC ≥ 0.999 (source: §A.2)
- **Measured:** 0.99998 / 0.99997 / 0.99997
- **Verdict:** PASS
- **Deviations:** none
- **Notes:** `eps` read from `config.json:rms_norm_eps` = 1e-05. No `+1` weight fold (Llama is a
  plain RMSNorm, unlike Gemma) — see `DEC-004`.
```

Verdicts are exactly one of: `PASS`, `FAIL`, `PASS-WITH-DEVIATION` (requires a `DEC`), `BLOCKED`
(requires a `07_RISKS.md` entry naming the blocker), `NOT-RUN` (requires the reason).

### 1.5 Progress checkpoint

After each phase, append to `06_GATES.md` a two-line status:

```
STATUS after P5: gates PASS=6 FAIL=0 DEVIATION=1 BLOCKED=0 | next: P6 (layer assembly)
Open DECs needing review: DEC-009 (bf8_b KV dtype), DEC-011 (o_proj reduce-scatter vs all-reduce)
```

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
   `models/demos/gpt_oss_d_p/tt/__init__.py`).
2. **Resolve the model identity.** `llama31_8b` is a directory name, not a HuggingFace id. Determine
   the exact checkpoint:
   - `echo $HF_MODEL` and `cat $HF_MODEL/config.json` if a checkpoint is staged;
   - otherwise fall back to the bundled dims at
     `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json` (verified present) and
     record in `00_MODEL_CARD.md` that dims come from the bundled config, not the live checkpoint.
   - Write a `DEC` naming the resolved repo id / local path and how you resolved it. **If the
     intended checkpoint is ambiguous (e.g. no public "Llama-3.2-8B" exists), say so explicitly in
     the card and in `07_RISKS.md`, proceed on the Llama-3.1-8B-Instruct dims, and flag it as the
     single assumption the user must confirm.** Do not stall on it.
3. Fill the card. Every row needs `Source`. Expected values for Llama-3.1-8B-Instruct (from
   `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`) — **confirm each against
   the checkpoint you resolved and log any mismatch as a `DEC`:**

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
| RoPE | θ = 500000.0, **full rotary** (rotary_dim = head_dim) | `rope_theta` |
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
   - TP must divide both `num_attention_heads` (32) and `num_key_value_heads` (8) → **TP ∈ {1,2,4,8}**
     without KV-head replication; TP > 8 requires replicating KV heads → `DEC` required.
   - TP must divide `hidden` (4096) and `intermediate` (14336) tile-aligned:
     `14336/TP` must be a multiple of 32 → TP ∈ {1,2,4,8,...,448}. Combined with the KV constraint:
     **TP ≤ 8**.
   - SP = the other mesh axis. `CHUNK_SIZE % (SP*32) == 0` and `MAX_SEQ_LEN % CHUNK_SIZE == 0`
     (source: `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` "Shared setup").
   - Write the chosen `(mesh_shape, tp, sp)` and the arithmetic into the card **and** `04_CCL_PLAN.md`.

### Gate `G-CARD`

- **PASS when:** every row of the card has a non-empty `Source`; zero rows say "from memory"; the
  "does NOT have" section exists; the `(mesh, TP, SP)` arithmetic is shown; every `UNVERIFIED` row
  also appears in `07_RISKS.md`.
- **Command:** none (document review). Record the verdict and the list of `UNVERIFIED` rows.

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
   package is created at all; see the P0 skeleton and `DEC-003`) — the `minimax_m3`/`gpt_oss_d_p` pattern
   (`models/demos/minimax_m3/reference/model.py`, `models/demos/gpt_oss_d_p/reference/model.py`).
   Only needed when HF cannot load the checkpoint (M3's case: a VL package shipping no modeling
   code). **Llama does not need this** — if you write one anyway, that is a `DEC` with a reason.
3. Per-test hand-written math (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` writes its own
   attention in ~40 lines and drives both sides from identical random weights). Use this **in
   addition** for module tests: it removes the HF-load cost from the inner loop and makes the test
   runnable on a bare card with no checkpoint.

**Recommended combination, and the one to take unless evidence says otherwise:**
*hand-written torch math inside each unit test, driven by identical random weights* (fast, no
checkpoint, runs anywhere) **plus** *HF `reference_*` accessors for the layer/model-level tests with
real weights*. Log this as `DEC-002`.

### Steps

1. Write `bringup_log/01_REFERENCE.md`: which option, how to invoke it, its dtype policy
   (compute the reference in **fp32**, cast only at the comparison boundary — see
   `scripts/generate_golden_kv_cache.py` header in `minimax_m3`), and its determinism check.
2. Write `models/demos/llama31_8b_d_p/tests/test_factory.py`, modelled on
   `models/demos/minimax_m3/tests/test_factory.py`:
   - `llama_config_dims()` → loads the bundled/dereferenced `config.json` (no HF, no network);
   - `requires_hf_reference` → `pytest.mark.skipif` on `HF_MODEL` not being a directory;
   - `TestFactory.setup_test(mesh_device, ...)` → builds `MeshConfig` + `CCLManager` once.
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
  (b) if both a hand-written and an HF reference exist, they agree to PCC ≥ 0.9999 on one layer;
  (c) `01_REFERENCE.md` documents the invocation and the dtype policy.
- **Log:** the two hashes and the cross-reference PCC.

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
| `models/tt_transformers/` | The repo's llama home | `tt/load_checkpoints.py` (`convert_hf_qkv_to_meta_format:451`, `map_hf_to_meta_keys:800`, `reverse_permute:891`), `tt/common.py` (`precompute_freqs:489`, `apply_scaling:437` **llama3 rope scaling**, `get_prefill_rot_mat:534`, `get_rot_transformation_mat:562`), `tt/model_config.py` `reference_*` accessors, `model_params/Llama-3.1-8B-Instruct/config.json`, `tests/test_{mlp,attention,decoder,model,rms_norm,rope}*.py` |
| `models/demos/llama3_70b_galaxy/` | Galaxy llama decode: `llama_ccl.py`, `llama_attention.py`, `distributed_norm.py` | Second opinion on llama-specific CCL placement |
| `models/demos/common/prefill/` | The engine + its two docs (`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`, `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`) | P10 |
| `models/common/utility_functions.py` | `comp_pcc`, `comp_allclose`, `is_blackhole` | Every test |

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
  no row's decision is "write" where an importable equivalent exists (justify with a `DEC` if it is).

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
├── reference/
│   └── __init__.py               # (only if a self-contained torch reference is justified — DEC)
├── tt/
│   ├── __init__.py
│   ├── config.py                 # MeshConfig: mappers + collective wrappers  (template: minimax_m3/config.py:21)
│   ├── ccl.py                    # CCLManager: subdevice, semaphores, scratch (template: gpt_oss_d_p/tt/ccl.py:17)
│   ├── model_config.py           # ModelArgs: state-dict load, weight-cache path, prefixes
│   ├── rms_norm.py               # RMSNorm (plain) + distributed variant
│   ├── rope.py                   # llama3-scaled cos/sin tables + transformation matrix
│   ├── mlp.py                    # dense SwiGLU FFN, column/row parallel, TP collective inside
│   ├── attention/
│   │   ├── __init__.py           # class Attention: builds config+weights, dispatches forward
│   │   ├── config.py             # @dataclass AttentionConfig, ProgramConfig
│   │   ├── weights.py            # load/shard/tilize q,k,v,o; HF→Meta swizzle
│   │   ├── operations.py         # small reusable tensor ops (head split, rope apply, RS/AG helpers)
│   │   ├── prefill.py            # attention_forward(): the one-shot + cache-backed path
│   │   ├── kv_cache.py           # LlamaKVCache, allocate_kv_cache(), write_kv_chunk()
│   │   └── dense_sp.py           # SP ring-SDPA path (P8; stub with NotImplementedError until then)
│   ├── embedding.py              # token embedding (TP-sharded vocab or replicated — DEC)
│   ├── lm_head.py                # only if logits are needed; prefill's output is the KV cache
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
│   ├── generate_golden_kv_cache.py   # torch reference → per-layer golden KV (template: minimax_m3/scripts)
│   └── verify_golden_kv.py
└── tests/
    ├── __init__.py
    ├── test_factory.py
    ├── unit/                     # single-card, per-module PCC vs reference
    │   ├── test_reference_model.py
    │   ├── test_rms_norm_vs_ref.py
    │   ├── test_rope_vs_ref.py
    │   ├── test_mlp_vs_ref.py
    │   ├── test_attention_vs_ref.py
    │   ├── test_kv_cache_vs_ref.py
    │   ├── test_attention_chunked_vs_ref.py
    │   ├── test_decoder_layer_vs_ref.py
    │   └── test_model_vs_ref.py
    └── galaxy_prefill_kv_pcc.py  # multi-device harness: per-layer KV PCC vs golden
```

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
  `models/demos/minimax_m3/tt/dense_mlp.py::_load` for the exact shape of this branch.
- **HF `[out, in]` → ttnn `[in, out]`:** transpose at load time
  (`weight.transpose(-1,-2).unsqueeze(0).unsqueeze(0)`), never at runtime.
- **Deallocate eagerly.** `t.deallocate(True)` after last use; free the big input before allocating
  the big output (see the comment in `models/demos/minimax_m3/config.py::allreduce`).
- **Docstring anchors.** Each module's docstring names the HF anchor
  (`transformers.models.llama.modeling_llama.LlamaMLP`) and the source template it mirrors.
- **No env-var magic** beyond what `README.md` documents in a table.

### Gate `G-OUTLINE`

- **PASS when:** `03_OUTLINE.md` lists every file with (i) one-sentence responsibility, (ii) public
  interface signature, (iii) input/output tensor shapes with dtype and layout, (iv) the template it
  mirrors (`path:line`); and the per-layer tensor-shape table (below) is filled in with real numbers.

Fill this table for the chosen `(mesh, TP, SP)` — the `models/demos/gpt_oss_d_p/README.md` "shapes & correctness
notes" table is the model to follow. `S_loc = S/SP`, `TP = tp`:

| tensor | shape (per chip) | dtype | layout |
|---|---|---|---|
| hidden in | `[1, 1, S_loc, 4096]` | bf16 | TILE |
| Q | `[1, 32/TP, S_loc, 128]` | bf16 | TILE |
| K, V | `[1, 8/TP, S_loc, 128]` | bf16 | TILE |
| KV cache | (block-cyclic; fill in) | bf8_b (`DEC`) | TILE |
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
   - the CCL core range derives from `mesh_device.compute_with_storage_grid_size()` (Blackhole is
     wider than 8×8; hard-coding 8×8 breaks the ring-SDPA grid-offset assert);
   - semaphores are allocated **once**, never per layer or per chunk;
   - handing out semaphores cycles a ping-pong index (`get_rs_ping_pong_semaphore()`,
     `get_ag_ping_pong_semaphore()`, `get_barrier_semaphore()`), so back-to-back collectives never
     reuse a semaphore that may still be in flight. **This is the single most common source of
     nondeterministic multi-device PCC failures.**
2. **`MeshConfig`** (`tt/config.py`) — owns *the parallelism decision and the collective wrappers*:
   `shard_mapper`, `column_parallel`, `row_parallel`, `sequence_parallel`, `shard_size`, and the
   three collectives `allreduce(t, ccl, axis=...)`, `allgather(t, ccl, axis=, dim=)`,
   `reduce_scatter(t, ccl, dim=, axis=)`. Template: `models/demos/minimax_m3/config.py:21`
   (`allreduce:77`, `allgather:135`, `reduce_scatter:155`).
   TP is the only knob; SP = the other axis, derived. `_validate()` rejects sub-axis TP.

**Modules then call `self.mesh_config.<collective>(t, self.ccl_manager, ...)` themselves.**
Never raw `ttnn.experimental.*` inside a module — that is how semaphore reuse bugs get in. (The one
allowed exception in the templates is `ttnn.all_gather` for the tiny RMSNorm stats tensor; if you
use it, log a `DEC`.)

**Canonical collective ops** (usage counts measured across `minimax_m3`, `gpt_oss_d_p`,
`deepseek_v3_d_p`, `tt_transformers` in this tree): `ttnn.experimental.all_gather_async` (29 uses),
`ttnn.experimental.reduce_scatter_minimal_async` (18), `ttnn.experimental.all_reduce_async` (2).
An all-reduce is implemented as **reduce-scatter + all-gather** (see `MeshConfig.allreduce`), not as
`all_reduce_async`. `num_links` comes from `utils/general_utils.get_default_num_links(mesh_device)`
(2 on Blackhole, 4 on Wormhole, 1 for a single-row mesh).

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

### Residual-layout decision (do this consciously, it touches every module)

Two consistent schemes; pick one, log it, and hold it everywhere:

- **A — replicated residual (full emb):** every module returns `[1,1,S_loc,4096]`; attention and MLP
  close with a full **all-reduce**. Simplest; more DRAM and one extra all-gather per sublayer.
- **B — sharded residual (emb/TP):** the residual stream is `[1,1,S_loc,4096/TP]`; attention and MLP
  close with a **reduce-scatter only**, and the norm either all-gathers first or runs the 3-op
  distributed RMSNorm. Faster; requires `4096/TP % 32 == 0` (true for TP ≤ 8, since 4096/8 = 512).
  Template + the knobs that A/B them: `models/demos/minimax_m3/tt/residual.py` and its `README.md`
  env-var table.

**Recommendation for this iteration: A (replicated residual), for the first functional pass.** It
removes a whole class of layout bugs from the P5–P7 debugging surface, and B is a contained
follow-up because it only changes the *tail* of MLP/attention plus the norm input. Log as `DEC`, and
structure `mlp.py` / `attention/prefill.py` with the `scatter_output` parameter from
`models/demos/minimax_m3/tt/dense_mlp.py` so switching to B later is a flag, not a rewrite.

### Gate `G-CCL-PLAN`

- **PASS when:** `04_CCL_PLAN.md` contains: the `(mesh, TP, SP)` arithmetic; the collective-placement
  table above with every row justified; the residual-scheme `DEC`; the semaphore-lifetime statement
  ("allocated once in `CCLManager.__init__`, cycled per call, never per layer"); and a list of every
  collective call site with its `cluster_axis`, `dim`, and `topology`.

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
   [(1,1)], indirect=True)`, `reset_seeds`, `comp_pcc` from `models/common/utility_functions`,
   `logger.info` the PCC, `assert passing`.
3. Run it, `tee` the raw log, record the gate.
4. Log every judgement call as a `DEC`.

Multi-device parametrisations are added in P8, not here. Tests needing fabric take
`@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
indirect=True)` (pattern: `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py:126`).

### P5.1 `tt/config.py` + `tt/ccl.py` + `utils/`

Port `MeshConfig` (from `models/demos/minimax_m3/config.py`) and `CCLManager` (from `models/demos/gpt_oss_d_p/tt/ccl.py`),
**deleting** what Llama does not need (the ring-gather scratch buffers can stay — the SP path in P8
uses them — but MoE-specific pieces go). Copy `utils/general_utils.py` and `utils/substate.py` from
`models/demos/gpt_oss_d_p/utils/`. Set `_VALIDATED_MESH_SHAPE` / `_VALIDATED_TP` to your P0 target.

**Gate `G-MESH`:** a device-free unit test asserting `MeshConfig((1,8), tp=8)` yields
`sp=1, tp=8, shard_size(4096)=512, shard_size(14336)=1792`, and that `MeshConfig((1,8), tp=4)`
raises. Plus: `CCLManager` constructs on the real mesh without error and allocates its semaphores
exactly once (assert the list lengths).

### P5.2 `tt/rms_norm.py`

Plain RMSNorm: `out = rms_norm(x) * weight`. **No Gemma `+1` fold** (P0 card). Keep the
`is_distributed` branch from `models/demos/gpt_oss_d_p/tt/rms_norm.py` but leave it `False` until P8.
Weight is reshaped to `(1,1,-1,ttnn.TILE_SIZE)` and stored `ROW_MAJOR`.

**Gate `G-RMS`:** `test_rms_norm_vs_ref.py`, seq_len ∈ {32, 512, 4096}, **PCC ≥ 0.999**.

### P5.3 `tt/rope.py`

Llama-3 scaled RoPE. **Reuse, do not rewrite:** `models/tt_transformers/tt/common.py`
`precompute_freqs:489` + `apply_scaling:437` (`rope_type="llama3"` uses
`compute_llama3_parameters:405`, which takes **three** args `(freqs, scale_factor, orig_context_len)` —
`low_freq_factor = 1` and `high_freq_factor = 4` are **local constants** at `common.py:407-408`, NOT read
from `config.json`; benign for Llama-3.x, silently wrong for any model that changes them),
`get_prefill_rot_mat:534`, `get_rot_transformation_mat:562`.

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
whole surrounding prefill scaffolding already assumes it. Convert Q/K weights at load time, state
the choice in the module docstring, and log it as a `DEC` that names `rotary_embedding_hf` as the
alternative that removes the permute (the likely direction of travel).

`test_attention_vs_ref.py` in `gpt_oss_d_p` builds **both** cos/sin tables from one set of
frequencies (`_build_cos_sin`) and feeds the Meta pair to the device while the torch reference uses
the HF pair — copy that structure exactly, so the test cannot silently compare two different RoPEs
and call it a pass.

**Gate `G-ROPE`:** `test_rope_vs_ref.py` — apply RoPE to a random `[1, n_heads, S, 128]` tensor on
device and compare against the HF-convention torch `rotate_half` path applied to the
correspondingly-permuted input. **PCC ≥ 0.999**. Also assert the llama3 scaling actually took
effect: the scaled `inv_freq` must differ from the unscaled one for positions beyond
`original_max_position_embeddings` (a test that passes with scaling silently disabled is worthless).

### P5.4 `tt/mlp.py` — dense SwiGLU

`down(silu(gate(x)) * up(x))`, `intermediate_size` 14336, no biases.
`gate_proj`/`up_proj` **column-parallel** (shard the intermediate dim), `down_proj`
**row-parallel** (shard the input/intermediate dim) + the TP collective from P4.
Template: `models/demos/minimax_m3/tt/dense_mlp.py` — take its structure, replace the clamped
`swigluoai` activation with plain SwiGLU (`ttnn.silu(gate) * up`, or `ttnn.mul(..., input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`
if available — check, and log which).

**Gate `G-MLP`:** `test_mlp_vs_ref.py`, seq_len ∈ {32, 512, 4096}, **PCC ≥ 0.99** at
`weight_dtype=bfloat8_b`, and **≥ 0.999** at `bfloat16`. Run both; record both. If bf8_b misses
0.99, do not lower the threshold — log a `DEC` and keep bf16 for this iteration.

### P5.5 `tt/attention/` — GQA, full RoPE, causal SDPA

Split the directory exactly as `models/demos/gpt_oss_d_p/tt/attention/` does; it separates the four concerns that
otherwise tangle:

- `config.py` — `@dataclass AttentionConfig` (hidden_size, num_heads, num_kv_heads, head_dim,
  max_seq_len, rms_norm_eps, `scaling = head_dim**-0.5`, `sequence_parallel`) and `ProgramConfig`.
  **Drop** `sliding_window`, `sinks`, `layer_types` — Llama has none (P0 card).
- `weights.py` — load, transpose, Meta-swizzle, shard, tilize `q/k/v/o_proj`. Q/O are
  column/row-parallel over heads; K/V are column-parallel over KV heads. `num_kv_heads/TP` must be
  ≥ 1: with 8 KV heads, TP ≤ 8 needs no replication (P0).
- `operations.py` — head split/merge, RoPE application, the reduce-scatter/all-gather tail helper.
- `prefill.py` — `attention_forward(...)`: qkv proj → head split (GQA: `32/TP` Q heads share
  `8/TP` KV heads, **no on-chip KV repeat** — `ttnn.transformer.scaled_dot_product_attention`
  handles the GQA group; verify this against the op's signature and log it) → RoPE → causal SDPA →
  merge heads → `o_proj` → TP collective.
- `kv_cache.py` — `LlamaKVCache` (packed K/V), `allocate_kv_cache()`, `write_kv_chunk()`.
- `dense_sp.py` — SP ring path; create the file with a `NotImplementedError` and a docstring
  pointing at `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py`. It is filled in P8 only if SP > 1.
- `__init__.py` — `class Attention` assembling config + weights and dispatching to `prefill.py`
  (`models/demos/gpt_oss_d_p/tt/attention/__init__.py:28` is the template; delete the `is_sliding` logic).

**Gate `G-ATTN`:** `test_attention_vs_ref.py` — full block (QKV → GQA split → RoPE → causal SDPA →
o_proj) vs an in-test torch reference, identical random weights, seq_len ∈ {128, 512, 2048},
`(1,1)` mesh. **PCC ≥ 0.99**. The torch reference must build the causal mask explicitly
(`torch.triu(full((S,S), -inf), diagonal=1)`) and `repeat_interleave` the KV heads by the GQA group
— copy the reference from `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py::_torch_attention`,
removing the sink column and the sliding term.

### P5.6 KV cache

The KV cache is **the output of prefill** — its correctness is the whole point. Layout: SP-sharded,
**block-cyclic**, per-user slots. Writes go through
`ttnn.experimental.deepseek_prefill.update_padded_kv_cache` (source:
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/`; the constraint
`kv_actual_global % 32 == 0` is documented in
`models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` "Multi-turn conversations").
Templates: `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py` (177 lines) and
`models/demos/minimax_m3/tests/unit/test_kv_cache_{write,gqa_sp}_vs_ref.py`.

Decide and log: cache dtype (bf8_b vs bf16 — a `DEC`, with the PCC cost measured, not assumed),
block size, per-user slot sizing (`MAX_SEQ_LEN`), and whether K is stored post-RoPE (it is, in every
template — say so).

**Keep gpt-oss's block geometry unless you have a reason not to.**
`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27` defines
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` and shards with
`shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]` (`:87`). Matching it is what lets
P10 reuse the producer's existing packed-K/V read-back instead of writing a fourth reader
(see P10 step 5). Diverging from it is a `DEC` whose blast radius includes `G-MOCK-MIG`.

**Gate `G-KV`:** `test_kv_cache_vs_ref.py` — write a chunk, read it back, compare against the torch
reference's post-RoPE K and raw V. **PCC ≥ 0.99** at the chosen cache dtype (record the bf16 number
too). Additionally assert the **written region only**: pad-tail positions must be untouched, and
positions from an earlier chunk must be unchanged after a later chunk's write.

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

Template: `models/demos/gpt_oss_d_p/tt/layer.py` — take it and delete the MoE branch, the
`layer_types` plumbing, and (unless you want it) the `_DELTA_PROBE`. **Do keep a bring-up probe**:
a per-layer L2 / mean-abs / signed-mean dump of each residual delta behind one env var. It is the
fastest tool for finding *which* sublayer drifts in a 32-layer stack, and its output belongs in
`bringup_log/raw/`. Document the env var in `README.md`.

Keep the `ttnn.move(hidden_states)` re-allocation guard for long sequences (`seqlen > 32*1024`) and
the eager `deallocate(True)` calls — both are load-bearing for long-context DRAM pressure.

**Gate `G-LAYER`:** `test_decoder_layer_vs_ref.py` vs `ModelArgs.reference_decoder()`
(`models/tt_transformers/tt/model_config.py:4393`) or an in-test torch layer, real or random
weights, seq_len ∈ {128, 512, 2048}, `(1,1)`. **PCC ≥ 0.99**.

### P6.2 `tt/embedding.py`, `tt/model_config.py`

- `model_config.py::ModelArgs` — state-dict loading (`load_state_dict` via safetensors, then
  `map_hf_to_meta_keys` / `convert_hf_qkv_to_meta_format` from
  `models/tt_transformers/tt/load_checkpoints.py`), `weight_cache_path(dtype)`,
  `get_state_dict_prefix(module, layer_idx)`. Template: `models/demos/minimax_m3/tt/model_config.py:22`.
- `embedding.py` — replicated table is fine for a first pass (`DEC`); `vocab_size` 128256 is
  tile-friendly (128256/32 = 4008).

**Gate `G-WEIGHTS`:** a test that loads the real checkpoint and asserts (a) every expected key is
consumed — **no silently-unused weights and no missing weights**; print both sets; (b) a
cache-only rebuild (empty `state_dict` + populated `tensor_cache_path`) produces bit-identical
device tensors. This gate catches the failure mode where a renamed key means a layer quietly runs on
random weights.

### P6.3 `tt/model.py`

`Model`: embedding → `[DecoderLayer] * n_layers` → final norm → (optional lm_head).
Public surface, matching the templates (`models/demos/gpt_oss_d_p/tt/model.py:41`,
`models/demos/minimax_m3/tt/model.py:87`): `prepare_inputs_prefill`, `prefill_forward`,
`process_output_prefill`.

**Gate `G-MODEL`:** `test_model_vs_ref.py` on a **reduced layer count** (`n_layers=2`, then 4) vs the
HF reference with the same weights, seq_len ∈ {128, 512}. **hidden-state PCC ≥ 0.99**; if a
lm_head exists, also **top-1 token agreement = 100%** on the last position. Then the full 32-layer
run: record the per-layer hidden-state PCC curve into `bringup_log/raw/` — a monotone decay is
normal, a *step* at one layer is a bug and must be chased before P7.

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
2. `scripts/verify_golden_kv.py` — compare a device KV read-back against the golden, per layer,
   reporting min/mean PCC per layer for K and V.
3. `tt/tt_prefill_runtime.py` — the chunked runtime. Build it to the engine's contract *now* so P10
   is wiring, not rework: `compile(kv_cache)`, `make_chunk_input(token_ids)`,
   `prefill_chunk(input, kv_cache, *, slot_id, actual_start, actual_end, request_id=0)`, plus
   `mesh_device` and a `config` exposing `chunk_size/max_seq_len/first_layer_idx/is_first_rank/
   is_last_rank`. Full contract: `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` §2.
   Template: `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:96`.
   **The runtime must not own the KV cache** — the engine allocates it and passes it in.
4. `tests/unit/test_attention_chunked_vs_ref.py` — the chunked-vs-one-shot equivalence test
   (template: `models/demos/minimax_m3/tests/unit/test_attention_chunked_vs_ref.py`).

**Gate `G-CHUNK`:** for the same token sequence, one-shot prefill and N-chunk prefill produce KV
caches agreeing to **PCC ≥ 0.999** *and* both agree with the golden to **PCC ≥ 0.99** (K) /
**≥ 0.98** (V, which is consistently the weaker of the two — see the `minimax_m3` README status
table: K 0.963 / V 0.879 at bf8_b cache dtype on a 60-layer model; set your threshold from your own
measured bf16-vs-bf8_b delta and log the choice).

**Gate `G-GOLDEN`:** `verify_golden_kv.py` runs clean over all 32 layers and prints a per-layer
table; the table goes into `bringup_log/raw/`.

---

## Phase P8 — Multi-device: TP, SP, and the CCL gates

Only now does the mesh come in. Everything here is about proving the collectives are correct **and
race-free**.

### Steps

1. Add multi-device parametrisations to the P5/P6 unit tests: `(1,2)`, `(1,4)`, `(1,8)`, and the
   target shape. Add `device_params` with `fabric_config=ttnn.FabricConfig.FABRIC_1D` (or
   `FABRIC_1D_RING` if the topology is a ring — `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:122`
   selects between them; log which and why).
2. Turn on the collectives in the modules (they were written with the branch in place in P5).
3. Enable the distributed RMSNorm branch **only if** the residual scheme is B.
4. If SP > 1, implement `tt/attention/dense_sp.py` (ring SDPA over the block-cyclic SP cache),
   using `ccl_manager.ring_attention_ccl_semaphore_handles` and
   `ccl_manager.ring_attention_ccl_core_grid_offset`. Note the constraint from
   `models/demos/gpt_oss_d_p/tt/ccl.py`: **the ring-attention CCL workers and the SDPA compute cores must not
   overlap**; the CCL workers take the last compute column and the offset must derive from the real
   `compute_with_storage_grid_size()`.

### Gates

- **`G-TP-PARITY`** — for each module, the multi-device output must match the single-device output to
  **PCC ≥ 0.999**. Collectives are mathematically exact up to reduction order; a large drop here is a
  sharding bug, not precision. Test by running the same module with the same weights on `(1,1)` and
  on `(1,TP)` and comparing device outputs to each other (not just each to torch) — this is a
  sharper test than PCC-vs-torch because it removes the reference's own error.
- **`G-RACE`** — run the full-model KV PCC harness **three times** and assert the results are
  **bit-identical** (`models/demos/minimax_m3/README.md` reports exactly this: "race-free (3 runs bit-identical)").
  Non-determinism here means a semaphore is being reused while in flight — check the ping-pong
  cycling and that `CCLManager` is constructed once, not per layer. Log all three hashes.
- **`G-SEMAPHORE`** — assert `CCLManager` allocates its semaphores once: instantiate the model and
  check the manager's semaphore list lengths equal the constants (not `n_layers ×` them).
- **`G-MESH-KV`** — `tests/galaxy_prefill_kv_pcc.py` on the target mesh: per-layer K/V PCC vs golden,
  one-shot and chunked. Record the **min across layers** for K and V, per run configuration, in a
  status table in the package `README.md` (the `minimax_m3` README's "Status" table is the format).

---

## Phase P9 — Cleanliness gate

The user asked for this explicitly. Run it as a real gate, not a vibe check.

1. `pre-commit run --files $(git diff --name-only main...HEAD)` — clean.
2. Every new file has the SPDX header pair.
3. `grep -rn "TODO\|FIXME\|XXX\|HACK" models/demos/llama31_8b_d_p/` — every hit is either resolved or
   has a `07_RISKS.md` entry with a filed-issue reference.
4. `grep -rn "except.*:\s*pass\|except Exception" models/demos/llama31_8b_d_p/` — none on a
   correctness path (a bring-up probe that must never break a run is the only allowed case, and it
   logs).
5. No unused imports, no dead branches, no commented-out code, no leftover `print` (use `loguru`).
6. Every `os.environ` / `os.getenv` read in the package appears in the `README.md` env-var table.
7. `README.md` is complete: architecture table, deployment path, status table with measured PCC, run
   commands, env-var table, layout section, and a "what is not implemented" section. Model it on
   `models/demos/minimax_m3/README.md`.
8. Import hygiene: `python -c "import models.demos.llama31_8b_d_p.tt.runners.adapters.llama"` must
   be **cheap** — no reference-model, device, or runtime imports at module load
   (`ADDING_A_PREFILL_MODEL.md` requires this; the H2D producers import adapters).
9. Test inventory: every `tt/` module has a corresponding `tests/unit/test_*_vs_ref.py`. List the
   mapping in `06_GATES.md` and flag gaps.

**Gate `G-CLEAN`:** all nine items, each with its command and output recorded.

---

## Phase P10 — Disaggregated-prefill integration

**Read these two documents in full before writing anything in this phase:**

- `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` — the adapter + runtime contract
  (§1 adapter, §2 runtime, §3 registration, §4 validation, checklist).
- `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` — the three config files, the two
  gates, and what a `PASS` there does and does not prove.

Working template to mirror end to end: `models/demos/gpt_oss_d_p/tt/runners/` —
`adapters/gpt_oss.py:41` (`GptOssPrefillAdapter`), `kv_chunk_table.py`,
`manifests/gpt_oss_d_p.json`.

### Steps

1. **`tt/runners/adapters/llama.py`** — subclass `PrefillModelAdapter`
   (`models/demos/common/prefill/adapter.py:104`). Set `name = "llama31_8b_d_p"`, `model_config`,
   `hf_model_default`, `ttnn_cache_default`, `prefill_trace_default`, `l1_small_size`,
   `supports_dflash = False`. (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:45-49` shows
   the exact five and what an empty default means: `ttnn_cache_default = ""` ⇒ no cache,
   `prefill_trace_default = ""` ⇒ trace must come from `PREFILL_TRACE_DIR`.) Implement `load_hf_config`, `weight_cache_path(mesh_shape)`,
   `allocate_kv_cache(*, mesh_device, hf_config, params)` returning a `KvCaches` subclass, and
   `build_runtime(*, mesh_device, hf_config, params)`. Read knobs from `params`
   (`PrefillRunParams`, `adapter.py:46`), **never** from `os.environ`.
   Keep the module import-light.
2. **`tt/runners/manifests/llama31_8b_d_p.json`** — `{ "env": { "PREFILL_MODEL": "llama31_8b_d_p" } }`.
3. **Register** in `ADAPTER_PATHS` in `models/demos/common/prefill/adapter.py`:
   `"llama31_8b_d_p": "models.demos.llama31_8b_d_p.tt.runners.adapters.llama:LlamaPrefillAdapter"`
   (the dict starts at `adapter.py:277`). One line; the import stays lazy.
4. **`tt/runners/kv_chunk_table.py`** + the runtime's optional migration hooks
   (`build_kv_chunk_table`, `kv_migration_base_address`, `set_layer_ack_channel`) using
   `serialize_kv_chunk_table` from `models/demos/common/prefill/runners/migration.py`.
5. **Wire the producer's KV read-back for your cache layout.** The device-less reader that powers
   `PREFILL_PRODUCER_CHECK_PCC` is **not** adapter-dispatched through the adapter object — it
   branches on `ADAPTER.name` in
   `models/demos/common/prefill/runners/prefill_producer.py`'s `_read_slot_kv_and_check_pcc`, which
   at the time this recipe was written (before P10's edit, and at `:503`) read:
   ```python
   if ADAPTER.name == "minimax_m3":   return _read_slot_kv_and_check_pcc_m3(...)
   if ADAPTER.name == "gpt_oss_d_p":  return _read_slot_kv_and_check_pcc_gpt_oss(...)
   return _read_slot_kv_and_check_pcc_mla(...)          # DeepSeek / Kimi merged MLA
   ```
   The MLA fallback is **wrong for Llama**, so without a branch this gate silently checks the wrong
   bytes. `_read_slot_kv_and_check_pcc_gpt_oss` is the **plain packed-K/V, block-cyclic GQA reader**
   — exactly Llama's layout — so read it first and, if your `kv_cache.py` keeps gpt-oss's
   `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` and packing, add `llama31_8b_d_p` to that branch rather than
   writing a fourth reader. (Note the older two-layout wording in `ADDING_A_PREFILL_MODEL.md` §4
   predates the gpt-oss reader; the code above is the current truth.) This touches shared code:
   record a `DEC` plus an entry in `08_PREFILL_INTEGRATION.md`, and prefer generalising the name
   check over duplicating the function.
6. Populate the weight cache and stage a golden trace (P7's script output).

### Gates

- **`G-ADAPTER`** — the checklist at the end of `ADDING_A_PREFILL_MODEL.md`, item by item, each with
  evidence. Plus: `PREFILL_MODEL=llama31_8b_d_p` resolves through the registry, and the pytest
  `variant` fixture (`models/demos/deepseek_v3_d_p/tests/conftest.py:365` — the registry-fed fixture
  the doc refers to) picks it up.
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
  `PREFILL_MAX_SEQ_LEN ≥ chunks * PREFILL_CHUNK_SIZE`.
- **`G-MOCK-MIG`** (= the doc's **Gate 1**) — `PREFILL_MOCK_MIGRATION=1` on the runner (single-rank
  only) + `PREFILL_PRODUCER_CHECK_PCC=1` on the producer. Expect
  `[producer] KV cache PCC PASSED` (default threshold `PREFILL_STANDALONE_CHUNKED_PCC` = 0.93 —
  record the *measured* value, not just the pass). This gate proves both that prefill writes correct
  KV and that `build_kv_chunk_table` is correct.
- **`G-LOOPBACK`** (= the doc's **Gate 2**) — real DRAM→transport→DRAM copy via `migration_driver`
  with `--verify-migration dst-bytes`. **Requires the tt-llm-engine binaries**; if they are not
  available in this environment, record `BLOCKED` with the reason in `07_RISKS.md` and stop there —
  do not fake it.

Record in `08_PREFILL_INTEGRATION.md`: the contract mapping (each abstract method → your
implementation, `path:line`), the full env matrix used, verbatim gate transcripts, and the
limitations you inherited (loopback-only verification; cross-talk invisible with one prompt unless
`PREFILL_PRODUCER_SLOT_TRACES` is used; a layer subset makes a `PASS` a sample).

---

## Appendix A — Gate index and thresholds

| Gate | Phase | Proves | Threshold | Device |
|---|---|---|---|---|
| `G-CARD` | P0 | every fact has provenance | doc review | — |
| `G-REF` | P1 | reference is deterministic and self-consistent | bit-identical; cross-ref PCC ≥ 0.9999 | host |
| `G-SURVEY` | P2 | reuse decided with citations | doc review | — |
| `G-OUTLINE` | P3 | file tree + shapes pinned | doc review | — |
| `G-CCL-PLAN` | P4 | every collective placed and justified | doc review | — |
| `G-MESH` | P5.1 | MeshConfig arithmetic; semaphores allocated once | exact asserts | 1 card |
| `G-RMS` | P5.2 | RMSNorm | PCC ≥ 0.999 | (1,1) |
| `G-ROPE` | P5.3 | RoPE + llama3 scaling active | PCC ≥ 0.999 | (1,1) |
| `G-MLP` | P5.4 | dense SwiGLU | ≥ 0.99 @bf8_b, ≥ 0.999 @bf16 | (1,1) |
| `G-ATTN` | P5.5 | GQA + RoPE + causal SDPA + o_proj | PCC ≥ 0.99 | (1,1) |
| `G-KV` | P5.6 | cache write correctness + no collateral writes | PCC ≥ 0.99 | (1,1) |
| `G-LAYER` | P6.1 | decoder layer | PCC ≥ 0.99 | (1,1) |
| `G-WEIGHTS` | P6.2 | no missing/unused keys; cache-only rebuild identical | exact | 1 card |
| `G-MODEL` | P6.3 | full stack hidden states; top-1 agreement | ≥ 0.99; 100% top-1 | (1,1) |
| `G-CHUNK` | P7 | chunked ≡ one-shot | ≥ 0.999 mutual; ≥ 0.99 K / ≥ 0.98 V vs golden | **(1,8)+** for the KV half (TP must equal 8 — Appendix F.6); **(4,8)** for the attention half |
| `G-GOLDEN` | P7 | golden trace structure is sound over all layers | clean table | host (imports no ttnn); the device-vs-golden comparison is scored inside `G-CHUNK` |
| `G-TP-PARITY` | P8 | collectives are exact | PCC ≥ 0.999 vs single-device | (1,TP) |
| `G-RACE` | P8 | no semaphore races | 3 runs bit-identical | target mesh |
| `G-SEMAPHORE` | P8 | CCL state allocated once | exact | target mesh |
| `G-MESH-KV` | P8 | full-model KV vs golden on target mesh | per-layer min recorded | target mesh |
| `G-CLEAN` | P9 | cleanliness (9 items) | all pass | — |
| `G-ADAPTER` | P10 | engine contract satisfied | checklist | — |
| `G-REQUEST` | P10 | request-mode serving works | run completes | target mesh |
| `G-MOCK-MIG` | P10 | KV + chunk table correct (doc Gate 1) | producer PCC ≥ 0.93 | single rank |
| `G-LOOPBACK` | P10 | real migration copy (doc Gate 2) | `dst-bytes` identical | + engine binaries |

### A.2 Where these thresholds come from

`0.999` for norm/RoPE and `0.99` for attention/router-class blocks are the thresholds
`models/demos/gpt_oss_d_p/README.md` states for its tier-1 single-card tests ("norm/rope ≥0.999, attn/router
≥0.99, expert bf4 ≥0.98"), and `models/tt_transformers/tests/test_mlp.py` uses `0.99`.
`0.93` is the disaggregated producer's own default (`PREFILL_STANDALONE_CHUNKED_PCC`, documented in
`PREFILL_MIGRATION_TESTING.md` Gate 1). Whole-model KV numbers degrade with depth and cache dtype —
`minimax_m3`'s shipped status is K 0.963 / V 0.879 min-across-60-layers at bf8_b. **Set your
whole-model thresholds from your own measured bf16 baseline, and log the bf16→bf8_b delta as the
justification.** Never lower a threshold to make a test green without a `DEC` that states the
measured value, the suspected cause, and the follow-up.

---

## Appendix B — Failure playbook

Consult before opening a debugging session; these are the failure modes the templates' comments
document, and each has a first move that is not "stare at the matmul".

| Symptom | Most likely cause | First move |
|---|---|---|
| Attention PCC ~0.5–0.9, norms fine | RoPE convention mismatch (HF halves vs Meta interleaved), or Q/K weights not `reverse_permute`d | Build both cos/sin tables from one frequency set, as `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py::_build_cos_sin` does; test RoPE alone (`G-ROPE`) |
| PCC good at short seq, bad past ~8192 | llama3 RoPE scaling not applied | Assert scaled ≠ unscaled `inv_freq` beyond `original_max_position_embeddings` |
| Multi-device PCC varies run to run | semaphore reused while in flight; `CCLManager` built per layer | `G-RACE` + `G-SEMAPHORE`; check the ping-pong cycling |
| Ring-SDPA assert `ccl_core_grid_offset.x >= sdpa_grid.x` | CCL cores derived from a hard-coded 8×8 grid | Derive from `mesh_device.compute_with_storage_grid_size()` (`models/demos/gpt_oss_d_p/tt/ccl.py::_init_subdevice`) |
| One layer runs on garbage; others fine | state-dict key renamed / not consumed | `G-WEIGHTS` (assert no missing and no unused keys) |
| A *step* in the per-layer PCC curve | a specific sublayer's logic error | The per-layer delta probe (`_delta_stats` in `models/demos/gpt_oss_d_p/tt/layer.py`): a growing signed-mean localises it |
| Cache-only build silently wrong | an un-cached weight (biases, sidecars) has no source when `state_dict` is empty | Fail loud, do not default — `models/demos/minimax_m3/tt/mlp.py` raises rather than running bias-free |
| Runner logs `device map ... not found; skipping KV read` | the device-map sidecar was not published; every PCC silently vanished | `PREFILL_MIGRATION_TESTING.md` Gate 1 — check `serialize_device_map`, and clear stale `/tmp` maps |
| `PREFILL_*` set in the shell has no effect under `tt-run` | `tt-run` forwards only `TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes | Set it in the binding's `global_env` or the model manifest |
| Producer ack drain hangs | `PREFILL_NUM_LAYERS` differs between runner and producer (ack count = layers × chunks) | Pin the real depth on both |
| Chunk write asserts | `kv_actual_global % 32 != 0`, or `CHUNK_SIZE % (SP*32) != 0`, or `MAX_SEQ_LEN % CHUNK_SIZE != 0` | Re-check the P0 shape arithmetic |

---

## Appendix C — Definition of done for this iteration

1. `models/demos/llama31_8b_d_p/` contains the P3 tree, no dead files.
2. Gates `G-CARD` … `G-CLEAN` are all `PASS` (or `PASS-WITH-DEVIATION` with a `DEC`), recorded in
   `bringup_log/06_GATES.md` with raw logs.
3. P10 gates `G-ADAPTER`, `G-REQUEST`, `G-MOCK-MIG` are `PASS`; `G-LOOPBACK` is `PASS` or `BLOCKED`
   with a stated reason.
4. `bringup_log/` reads as a coherent narrative: a reviewer can reconstruct every judgement call,
   its alternatives, and its evidence, without reading the code.
5. `README.md` carries a status table with measured PCC numbers, and a "not implemented" section.
6. `07_RISKS.md` lists every `UNVERIFIED` fact, every gap, and every follow-up, each with an owner
   slot and (where applicable) a filed issue.
# Addendum to fold into BRINGUP_RECIPE.md (verified 2026-09-03, apply after Session A returns)

## Appendix D — This machine (verified, do not re-derive)
- Hardware: **Blackhole Galaxy, 32 devices**. `ttnn.get_num_devices()==32`, `ttnn.get_arch_name()=='blackhole'`.
- `build/` + `python_env/` working; ttnn imports and opens/closes the cluster cleanly.
- `transformers==5.12.1`, `huggingface_hub==1.16.1`.
- Network + HF access available; `meta-llama/Llama-3.1-8B-Instruct` is reachable (gated repo, org token in env).
  Live `config.json` byte-matches the bundled
  `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json` dims -> the P0 card's dims are
  confirmed against the real checkpoint.
- Checkpoint staged at `/home/mstojkovic/models/Llama-3.1-8B-Instruct` (safetensors, `original/` excluded).
  Export `HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct`.
- BH Galaxy mesh descriptors: `tt_metal/fabric/mesh_graph_descriptors/` — e.g.
  `bh_galaxy_sp4_torus_xy_graph_descriptor.textproto`, `32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`.
- NEVER write the HF token into any file, log, or agent prompt. It is read from the environment only.

## P5.5 — SDPA: GQA is native (RESOLVED, was "verify and log")
`ttnn.transformer.scaled_dot_product_attention` handles the GQA group itself — **no on-chip KV repeat**.
Evidence: `models/demos/gpt_oss_d_p/tt/attention/prefill.py:34-49` passes q with 8 local Q heads and k/v
with 1 local KV head. Exact call shape:
```python
ttnn.transformer.scaled_dot_product_attention(
    tt_q, tt_k, tt_v,
    is_causal=True,
    scale=config.scaling,                 # == 1/sqrt(head_dim); pass explicitly
    program_config=...,                   # ttnn.SDPAProgramConfig
    compute_kernel_config=...,
)
```
For Llama **drop** `sliding_window_size=` and `attention_sink=` (gpt-oss-only args).

## P5.5 — Program / compute-kernel config on Blackhole (RESOLVED)
- Use **`ttnn.WormholeComputeKernelConfig`** even on Blackhole. The name is misleading, not wrong; it is
  what `models/demos/gpt_oss_d_p/tt/attention/config.py:103` uses on this exact BH Galaxy. A rename is
  tracked as issue #51998. Do not hunt for a `BlackholeComputeKernelConfig`.
  Fields: `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`.
- `ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False,
  q_chunk_size=..., k_chunk_size=...)` — gpt-oss pins **8x8** for the SDPA program grid on BH
  (`config.py:95-100`). This is deliberately DIFFERENT from the CCL core grid, which must derive from
  `mesh_device.compute_with_storage_grid_size()` (BH is wider than 8x8). Do not unify them.

## P1 — transformers 5.x caveat
The reference wrappers in `models/tt_transformers/tt/model_config.py` (`HfAttentionWrapper`,
`HfDecoderWrapper`) branch on whether `position_embeddings` is in the layer's forward signature
(`reference_attention:4410`, `reference_decoder:4393`). On transformers 5.12.1 confirm that branch resolves
correctly for `LlamaAttention`/`LlamaDecoderLayer` before trusting a low PCC — a wrapper feeding RoPE twice
(or not at all) looks exactly like a model bug.

## Appendix E — Measured oracle baselines on THIS machine (evidence-based thresholds)

The existing `models/tt_transformers` Llama implementation runs on this box against the real
Llama-3.1-8B-Instruct checkpoint. It is both a **compat proof** (HF reference accessors work on
transformers 5.12.1) and a **PCC target**: a fresh module that lands materially below the number an
existing implementation already achieves on the same op set and dtype is a bug, not a precision limit.

Reproduce with:
```bash
export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct MESH_DEVICE=N150
pytest models/tt_transformers/tests/test_mlp.py -x -q -k 512
```

All four ran green on 2026-09-03, single Blackhole card (`MESH_DEVICE=N150`), real
Llama-3.1-8B-Instruct weights:

| Oracle | Measured PCC | Recipe's guessed gate | **Revised gate (evidence-based)** |
|---|---|---|---|
| `tests/test_rms_norm.py` | 0.9999867 / 0.9999886 | `G-RMS` >= 0.999 | **>= 0.9999** |
| `tests/test_mlp.py` (seq 512, `bfloat8_b`) | 0.9995823 | `G-MLP` >= 0.99 @bf8_b | **>= 0.999 @bf8_b** |
| `tests/test_attention_prefill.py` | 0.9996099 / 0.9996010 | `G-ATTN` >= 0.99 | **>= 0.999** |
| `tests/test_decoder_prefill.py` | 0.9999985 / 0.9999981 | `G-LAYER` >= 0.99 | **>= 0.999** (see caveat) |

Every original threshold was 1-2 orders of magnitude too loose. A fresh module landing at 0.99 would
have been recorded as PASS while sitting ~40x further from the reference than the existing
implementation on the same ops, dtype and silicon. **Apply the revised column.** Treat the band
between the revised gate and the guessed one as *investigate*, never *pass*.

### Caveat that matters more than the numbers
`test_decoder_prefill` scores **0.9999985** — HIGHER than either of its own sublayers (attention
0.99961, MLP 0.99958). The residual stream dominates the correlation, so a full-layer PCC partially
launders a degraded sublayer. Consequences for the gate design:
- The sublayer gates (`G-RMS`, `G-ROPE`, `G-MLP`, `G-ATTN`, `G-KV`) are the real evidence. `G-LAYER`
  and `G-MODEL` are *integration* checks and must never be accepted as a substitute for a missing or
  weak sublayer gate.
- A layer PCC that looks great while a sublayer gate is at 0.99 is the signature of this masking, not
  proof the sublayer is fine.
- This is also why the recipe's per-layer delta probe matters: magnitude ratios localise what a
  residual-dominated PCC hides.

### E.1 CORRECTION — do NOT gate against another implementation's PCC
The instruction that stood here ("measure what the existing implementation gets, then set your
threshold from it") is **wrong**, and it was wrong in a way that ratifies degraded modules. Two device
A/B runs on this box killed it:

**(a) Cross-implementation PCCs are not comparable, because the reference precision differs.**
`models/tt_transformers/tests/test_rms_norm.py:77` builds its reference via `reference_rms_norm()`,
whose HF weights load at `torch_dtype: bfloat16` (from `config.json`), so HF's
`self.weight * hidden_states` multiplies by a **bf16-rounded** weight. That reference *shares the
device's own rounding*, which inflates the PCC. A reference built with an fp32 weight is strictly
harder. Measured on identical inputs and real layer-0 norm weights: fp32-weight reference gives
**0.99995** where the oracle's bf16-weight reference reports **0.9999867** — the same device output,
two incomparable numbers.

**(b) Input distribution is a red herring.** An earlier draft blamed distribution. The torch bf16
noise floor is **0.9999986 under `rand[0,1)` and 0.9999986 under `randn`** — identical. Distribution
shifts the measured value in the 5th decimal; it does not move the floor, and it must never be used to
choose the distribution that passes.

### E.2 The method that actually works: gate on the gap to the NOISE FLOOR
For each gate, compute the floor in torch: **round inputs and weights to the device dtype, do all
remaining math in fp32, and PCC that against the fp32 reference.** That number is
implementation-independent, distribution-stable, and is the best any correct kernel can do.

Then record three things and gate on the **gap**:

| record | meaning |
|---|---|
| `floor` | torch noise floor for this dtype/shape (computed, never guessed — bf8_b floors are far lower than bf16) |
| `measured` | your module on device |
| `gap` | `floor - measured` |

At the floor = pass. **20x+ off the floor is a finding even when the absolute PCC looks pretty** — that
is exactly the band a threshold copied from a README waves through. Keep the absolute thresholds in
Appendix A as floors you must clear, but clearing one while sitting far off the noise floor gets
investigated, not recorded as a clean PASS. Every gate detail block states the **input distribution**
and the **reference's dtype policy**.

### E.3 Consequence found this way: always pass an explicit `compute_kernel_config`
`ttnn.rms_norm` on this box, fp32-weight reference, hidden 4096, real weights:

| `compute_kernel_config` | PCC (rand 32/512) | PCC (randn 32/512) |
|---|---|---|
| **none** | 0.9999440 / 0.9999531 | 0.9999652 / 0.9999648 |
| HiFi2, `fp32_dest_acc_en=False` | 0.9999369 / 0.9999407 | 0.9999607 / 0.9999590 |
| **HiFi4, `fp32_dest_acc_en=True`** | **0.9999969 / 0.9999968** | **0.9999971 / 0.9999971** |
| torch bf16 floor | 0.9999986 / 0.9999987 | 0.9999986 |

**`MathFidelity` alone is a no-op here** (HiFi2 is marginally *worse* than the default);
**`fp32_dest_acc_en=True` removes ~25x of the error and reaches the floor.** Passing no config is a
silent precision regression, and every module in the templates that omits one inherits it. Build the
config once via `ttnn.init_device_compute_kernel_config` and pass it to **every** op that accepts one
(each `ttnn.linear`, the SDPA call, the norms). Where `fp32_dest_acc_en=True` is refused or costs
accuracy, log a DEC with both measurements — never drop it silently.

## Appendix F — Corrections from executed phases (these SUPERSEDE the inline text)

Found by actually running P0-P2. Where this appendix and an earlier section disagree, this wins.

### F.1 Real weights ARE available — the "no checkpoint" plan is void
`/home/mstojkovic/models/Llama-3.1-8B-Instruct` (15 GB, 4 safetensors shards + tokenizer + config).
`export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct`. So `G-WEIGHTS` (real half),
`G-MODEL` top-1, `G-GOLDEN`, `G-CHUNK`/`G-MESH-KV`-vs-golden, `G-REQUEST` and `G-MOCK-MIG` are
**runnable, not BLOCKED**. Any earlier instruction to record them BLOCKED for lack of weights is void.
Keep the `requires_hf_reference` skip marker anyway so the suite still runs on a weightless machine.

### F.2 `transformers` 5.12.1 gotchas
- **`hf_config.rope_theta` does not exist as an attribute** (CORRECTED — an earlier draft of this
  appendix said it exists and is `None`; measured on this box, it raises `AttributeError`). The truth is
  worse than a `None`:

  | expression | actual result on transformers 5.12.1 |
  |---|---|
  | `cfg.rope_theta` | **raises `AttributeError`** |
  | `getattr(cfg, "rope_theta", DEFAULT)` | **returns `DEFAULT`** — silently substitutes a wrong theta |
  | `cfg.rope_scaling` | a full dict, and it **contains** `rope_theta: 500000.0` |
  | `cfg.to_dict()` | has neither key — only `rope_parameters` |

  So the `getattr(cfg, "rope_theta", DEFAULT)` pattern at `gpt_oss_d_p/tt/model_config.py:76` and
  `tt_prefill_runtime.py:185` does not fail loudly here — it **succeeds with a hard-coded theta**
  (10000.0 against Llama's 500000.0), giving a RoPE that is wrong at every position with no exception
  anywhere. This is the highest-severity silent-wrongness trap in the whole bring-up. Route theta and
  scaling through `models/tt_transformers/tt/common.py:165 get_rope_theta` / `:183 get_rope_scaling`
  (both take a **dict**), read them in exactly ONE place, and assert non-`None`.
- **Decide dict-vs-object for `hf_config` explicitly and hold it.** Templates pass an object
  (`minimax_m3/tt/dense_mlp.py:47` does `hf_config.hidden_size`); `llama_config_dims()` returns a dict;
  `get_rope_theta` wants a dict. A silent mix is how `None` dims get in.
- `ModelArgs.reference_*` accessors **raise without `HF_MODEL`**
  (`models/tt_transformers/tt/model_config.py:702`), so the
  recipe's "preferred option 1" is unreachable on a weightless box. With F.1 they now work, but the
  in-test torch math remains the better oracle for P5/P6: no checkpoint, faster, and gate-validated.
- `get_rot_transformation_mat(dhead=32)` **ignores its argument** (`common.py:564` hard-codes 32).
  Call it with no args.
- Torch references must set `cfg._attn_implementation = "eager"` **and** pass an explicit causal mask:
  `eager_attention_forward` applies only the mask handed to it, so `attention_mask=None` yields
  **non-causal** attention silently.

### F.3 `models/common/modules/` (TTTv2) — considered, and why it is NOT the base
A shared, unit-tested module library exists (`MLP1D/2D`, `RMSNorm1D/2D`, `Attention1D`,
`RotarySetup1D`, `Embedding1D`, `LMHead1D`, cached `TT_CCL` via `get_tt_ccl`), and
`models/common/models/llama3_8b/` is a complete Llama-3.1-8B implementation. Both were evaluated:
- **`MLP2D`'s "2D" is 2D *tensor* parallelism, not TP x SP.** Its prefill path reduce-scatters on
  `cluster_axis=1` and closes with `all_reduce(cluster_axis=0)` (`mlp/mlp_2d.py:461`). With SP on the
  row axis that all-reduce would sum activations belonging to **different tokens** — silently wrong,
  and it would still produce plausible-looking PCC on a 1-row mesh. Do not reuse it for SP prefill.
  (The tempting shortcut "an MLP is token-pointwise, so SP looks like DP to it" is exactly what this
  refutes: it holds for the math, not for this module's collectives.)
- **There is no `Attention2D`**, and `models/common/models/llama3_8b/model.py:890` raises
  `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")`. No chunked-prefill runtime,
  no `common/prefill` adapter.
So: `minimax_m3/tt/dense_mlp.py` remains the MLP template (it collectives on the TP axis **only**,
which is what makes it SP-safe) and `gpt_oss_d_p/tt/attention/` remains the attention template.
**P9 requirement:** the README must carry a "why not `models/common/modules` / `models/common/models/llama3_8b`"
line, because it is the first question a reviewer asks.

### F.4 Neither in-repo `MeshConfig` is a superset
`minimax_m3/config.py:21` has `reduce_scatter`; `gpt_oss_d_p/tt/config.py:19` does **not**, and its
`_VALIDATED_*` at `:15-16` already pins `(4,8)`/TP=8 — our target. Build the union; do not assume
either copy is complete.

### F.5 The distributed-RMSNorm branch is dormant — but that is NOT an argument against scheme B
`gpt_oss_d_p/tt/rms_norm.py:33` pins `self.is_distributed = False` with the condition commented out, so
that 3-op distributed-norm path has never been exercised. **An earlier draft over-read this as
"scheme B is unproven".** It is not: `models/demos/minimax_m3/tt/residual.py:26` ships **scheme B by
default**, with `DEFAULT_NORM_MODE = "gather_first"` (`:32`) — which all-gathers the residual shard and
runs one ordinary single-pass norm, never entering the dormant branch. Only **B-with-distributed-norm**
is unproven.

The real argument for **A on a dense model** is cost equivalence: for a dense layer, A and B issue the
**identical** collectives (2 reduce-scatters + 2 all-gathers per layer, same sizes, same axis).
Minimax's B win comes from sharing one gathered norm output across several MoE consumers
(`residual.py:9-11`) — a saving Llama has no consumers for. A additionally keeps `G-TP-PARITY` a direct
device-vs-device comparison, and a replicated embedding already yields a full-width residual. Wire
`scatter_output` from day one anyway.

### F.6 `n_kv=1` at TP=8 — RETIRED; the real residual risk is `head_dim`
The concern was that TP=8 leaves **1 KV head per chip** and that only SDPA was proven there. Checked,
and `n_kv=1` is in fact the **production-exercised** configuration on this exact hardware:
- `gpt_oss_d_p/tt/attention/kv_cache.py:98` allocates
  `torch.zeros(num_users * num_layers, 1, seq_local, head_dim)` — one KV head per chip, hardcoded,
  commented *"Per-chip cache is one head"* — and `update_padded_kv_cache` writes into it unchanged.
- `gpt_oss_d_p/README.md` marks **P5 (TP=8 / SP=4 on the 4x8 Blackhole Galaxy, full 36L, real weights,
  per-layer KV-cache PCC vs golden)** and **P6 (ring SDPA on every SP chunk)** as complete.
- SDPA's own guard is satisfied: `TT_FATAL(nqh >= nkv && nqh % nkv == 0)`
  (`sdpa_device_operation.cpp:97-101`); at TP=8 Llama gives `4 >= 1 && 4 % 1 == 0`.

So `update_padded_kv_cache` and the ring SDPA are both exercised at `n_kv=1` at this exact
(mesh, TP, SP). **No topology change is needed and the `(8,4)`/TP=4 fallback is not on the critical
path.**

**CORRECTED BY P7 — this section had the risk backwards.** It concluded "the residual delta is
`head_dim`, not head count". The head *count* is in fact the binding constraint, and it constrains the
**mesh**, not the cache:

- The packed cache allocates **exactly one KV head per chip** (`tt/attention/kv_cache.py:130` hard-codes
  the `1`). Llama has `num_key_value_heads = 8`. Therefore **`TP` must equal 8** — it is the only mesh
  width whose per-chip KV-head count the cache can hold.
- At any smaller TP the model produces more local KV heads than the cache slot, and
  `update_padded_kv_cache` dies with `TT_FATAL: cache and input num-heads dim must match`
  (`update_padded_kv_cache_device_operation.cpp:230`) — **including for chunk 0**. So on a single card
  **no model-level KV write is possible at all.**
- Consequence for a gate already recorded PASS: **`G-KV` ran at `(1,1)` with `nkv = tp = 1`, a head
  count the model never produces on that mesh.** It is a valid test of the cache *primitive* and of the
  head_dim=128 geometry, but it does **not** cover the model -> cache path. `R-027` tracks this.
- `head_dim` 64 -> 128 was real but minor: the shard spec
  `shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK(=32), head_dim]` (`kv_cache.py:87`)
  parameterises it and 128 is tile-aligned; P7 measured it clean.
- Also note `write_kv_chunk` writes **one user per call** and asserts it
  (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:148`, mirrored at
  `models/demos/llama31_8b_d_p/tt/attention/kv_cache.py:181`): the op ignores the leading batch dim,
  so a `batch > 1` tensor would silently write only `slot_idx`. Multi-user prefill must loop
  `slot_idx + b` at the call site. The **head** count is not this assert's business — it is the op's
  own `TT_FATAL` cited above, which is why the `nkv = tp` mapping has to be proved by a TP=8 run
  rather than assumed. *(P9 correction: this bullet previously read "takes one KV head per call",
  citing the bare basename `kv_cache.py` at line 181. Both halves were wrong — the basename resolves
  to gpt-oss's copy, which has only 177 lines, and the assert is about the batch dim, not heads.
  `DEC-120`.)*

**So the first thing P8 must run is a `(1,8)`/TP=8 parametrisation** — no SP, cheap, and it proves the
model -> cache mesh-mapper step (KV head `c` -> mesh column `c`) *before* any sequence-parallel bug can
arrive tangled up with it. The general lesson: a gate that passes on a mesh the deployment never uses
can be testing a configuration the model cannot produce.

### F.7 Verify your own citations mechanically
`models/demos/llama31_8b_d_p/scripts/verify_citations.py` (built in P0) caught 5 wrong `path:line`
refs in this recipe *and* 5 in P0's own first draft. Extend it every phase and run it as part of each
doc gate. An unverified `path:line` is worse than no citation: it reads as authoritative.

### F.8 The SDPA program grid must stay 8x8 — deriving it from the device grid breaks P8 silently
Appendix D says gpt-oss "pins 8x8 deliberately". Here is the assert that makes it mandatory, because
a survey pass in this very bring-up recommended the opposite and it would have shipped:

- This machine's compute grid is **(12, 10)** (`compute_with_storage_grid_size()`), not 8x8.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421` asserts
  **`ccl_core_grid_offset.x >= sdpa_grid.x`**, and the CCL offset is pinned at `grid.x - 1 = 11`.
- With the SDPA grid at 8: `11 >= 8` OK. Derived from the device grid at 12: `11 >= 12` **FAILS**.

The failure mode is the dangerous kind: a derived grid **passes every P5 single-card gate** (which
never runs the ring path) and only fails at SP > 1 in P8, long after the choice looks settled. Keep the
SDPA program grid an explicit named field defaulting to 8x8, and assert `sdpa_grid.x <= grid.x - 1` at
construction so the constraint fails at build time instead of two phases later.

Related, and measured: **`ttnn.BlackholeComputeKernelConfig` does not exist** —
`hasattr(ttnn, "BlackholeComputeKernelConfig")` is `False` (`ttnn/ttnn/__init__.py:305` exports only the
Wormhole name), and where it is defined it is the *same object* (`ttnn/ttnn/types.py:61`). So an
"arch-branch to pick the kernel-config class" is a no-op. Prefer
`ttnn.init_device_compute_kernel_config(...)`.

### F.9 Four gates in the index had no test file in the planned tree
`G-MESH`, `G-SEMAPHORE`, `G-WEIGHTS` and `G-TP-PARITY` appear in the gate index but no file in the P3
tree owned them. P5 adds `tests/unit/test_mesh_config.py`, `tests/unit/test_ccl_semaphores.py`,
`tests/unit/test_weight_loading.py`, `tests/unit/test_tp_parity.py`. **When adding a gate, add its
file in the same edit** — an unowned gate silently becomes a NOT-RUN.

### F.10 Two coverage holes in the gate ladder (P7/P8 own them)
- **`G-TP-PARITY` never touches the deployment fabric.** `get_default_num_links` returns **1** for any
  single-row mesh (`gpt_oss_d_p/utils/general_utils.py:33`), so `(1,2)/(1,4)/(1,8)` parity runs
  `num_links=1` + `Topology.Linear`. The 2-link **Ring** path is first exercised by `G-MESH-KV`/`G-RACE`
  on `(4,8)`. **Add a `(2,8)` parametrisation** so parity covers the real transport.
- **Semaphore reuse across chunks is only 2 deep.** RS takes `barrier[0]`, AG `barrier[1]`, the next RS
  `barrier[0]` again — a one-op gap; and `reset_global_semaphores` deliberately skips the barrier and
  ring-attention semaphores (`gpt_oss_d_p/tt/ccl.py:132`, an open upstream TODO) while chunked prefill
  **does** reuse one `CCLManager` across chunks. P7 owes a DEC either way. If `G-RACE` fails, deepening
  the barrier ring from 2 to 4 is the first move, before suspecting the model.
- **The weight cache is mesh-shape dependent and cache-only is never proven at TP>1.** `ttnn.as_tensor`
  caches the already-sharded tensor and cache-only mode is load-bearing for the runner, but `G-WEIGHTS`
  runs on one card. A stale/wrong-shape cache presents as "one layer runs on garbage", first visible at
  `G-MESH-KV`. Put the mesh shape in `weight_cache_path` (as `adapters/gpt_oss.py:75` does) and re-run
  the cache-only assertion on `(4,8)` in P8.

### E.4 `fp32_dest_acc_en` polarity is PER-OP — and the template's explicit `False` is harmful
E.3 established that the norm needs `fp32_dest_acc_en=True`. Measured across ops, the *default* differs
by op, so "just copy the template's config" is wrong in both directions:

| op | no config | `=True` | `=False` |
|---|---|---|---|
| `ttnn.rms_norm` | 0.9999652 | **0.9999971** | 0.9999607 |
| `ttnn.linear`, bf8_b weights | 0.9999143 | **0.9999143** (bit-identical) | **0.9925392** (96x worse) |
| `ttnn.linear`, bf16 weights | 0.9999852 | **0.9999852** (bit-identical) | **0.9917529** (1168x worse) |
| attention block, bf8_b | — | **0.9997449** | **0.9963324** (38.7x worse) |
| attention block, bf16 | — | **0.9998033** | **0.9959098** (107.6x worse) |

The matmul's own default **already enables** fp32 accumulation — the opposite of the norm. So the
danger is not omitting the flag on a matmul; it is **carrying `models/demos/gpt_oss_d_p/tt/attention/config.py:71`'s
explicit `fp32_dest_acc_en=False` forward**, which costs two to three orders of magnitude.

**Note precisely what caught this**, because it is the whole argument for E.2: the degraded attention
block scores **0.9963** at bf8_b. That **clears the recipe's original guessed 0.99 gate** — it would
have been logged a clean PASS — and **fails the evidence-based 0.999 gate**. The tightened threshold
caught it; the noise-floor comparison then explained *why* in one measurement instead of a debugging
session. Make `fp32_dest_acc_en=True` the package default, pass it explicitly everywhere, and A/B it
in-suite so a regression shows up as a number rather than a mystery.

### E.5 A storage-dtype noise floor does NOT model a fused kernel's interior
E.2's floor model (round inputs/weights to the device dtype, rest in fp32) is only valid for ops whose
interior arithmetic you can mirror. It breaks on fused kernels, exactly as E.2's falsifier predicted:

`ttnn.transformer.scaled_dot_product_attention` alone (bf16 Q/K/V, GQA 32/8, head_dim 128) measures
**0.9999204** against a modelled floor of **0.9999989** — a **71x** gap. Sweeping `q_chunk`/`k_chunk`
over {32,128,256} moves it under 4%; `exp_approx_mode` not at all. And that single term is the *entire*
block-level gap: every stage we implement ourselves measures **1.00-1.47x** of its floor.

So do not read a large gap as "our code is wrong" without first isolating the fused kernel. The
sanctioned handling: **separate error budgets per stage** (measured, not assumed) plus a **permanent
standalone probe** of the fused kernel, so its slack is *named and tracked* rather than silently
granted to the whole block. A budget that lumps the kernel's 71x in with our stages' 1.5x can absorb a
real regression without anyone noticing.

### E.6 `bfloat16` is exact only up to 256 — beware integer-valued probe tensors
A positional read-back probe using integer position ids at `max_seq_len=384` failed with *greatest
relative difference 1/257*: **257 is not representable in bf16 and rounds to 256**, so 64 of 384 rows
"mismatched" while the cache was perfectly correct. Any probe that encodes indices, positions or ids as
bf16 tensor *values* must keep every value <= 256, or split the id across lanes (the fix here: 4 chunks
of 64 with the head id in its own lane block, which also covers more `kv_actual` offsets than the
original 3x128). A failing probe is not evidence of a failing module until the probe's own numerics are
checked.

### F.11 Orchestration hygiene — do not mutate the worktree while a phase session is live
Three separate problems in this run traced to one cause: the orchestrator committing and renaming
while a phase session was still working in the same worktree. All were recoverable; all were
avoidable.

| What happened | Consequence |
|---|---|
| Committed mid-P7 the moment commits were authorised | A P7 file (`scripts/generate_golden_kv_cache.py`) landed in the commit whose message says **"P0-P6"**. The history is now mildly inaccurate — the content is fine, the label is not. |
| Renamed the package while the P7 session was still alive | Its path references stopped resolving mid-run; it had to re-derive where its own files had gone. Nothing was lost, but only because a rename is a move. |
| Renamed the golden-trace directory too | Same, for `$PREFILL_TRACE_DIR`. |

**And the subtle one — the verification gap.** After the rename the orchestrator verified imports,
`verify_citations.py`, and a 23-test smoke (`G-MESH` + `G-RMS`) — then declared the rename clean. But
it did **not re-run the gates of the phase that had just finished**, so P7's entire evidence base
briefly existed only against paths no longer in the tree. The P7 session caught this and re-ran both
its test files at the new path (16 passed, every number identical:
`raw/G-P7-POSTRENAME_20260903T211141Z.log`). A gate's evidence is only as good as the paths it was
recorded against.

**Rules, for this run and any re-run of this recipe:**
1. **Commit only at a phase boundary, with no session live.** "Commits are authorised" is not
   "commit right now". Check for a running session first.
2. **Never rename, move, or restructure while a session is live.** A rename is the worst case: it
   invalidates in-flight path references silently and rewrites the provenance of raw logs.
3. **After any path-affecting change, re-run the previous phase's gates**, not a smoke test. Import
   checks and a citation pass prove the tree is wired; they prove nothing about the gates.
4. **Raw logs from before a rename keep the old path** — that is correct, not stale. They record what
   actually ran. Rewriting them would make the evidence less trustworthy; note the boundary in the
   ledger instead.
5. If a mid-phase mutation is genuinely unavoidable, **message the live session** with the exact
   change before making it, rather than letting it discover the breakage.

### F.11 — P8 step 1 is not runnable as written on this machine: the sub-shapes must be **submeshes**

`BRINGUP_RECIPE.md:831` tells P8 to "add multi-device parametrisations to the P5/P6 unit tests:
`(1,2)`, `(1,4)`, `(1,8)`, and the target shape", and step 1 then talks about `device_params` with a
`fabric_config` — i.e. it assumes the `mesh_device` fixture opens each shape directly, exactly as
`models/demos/minimax_m3/tests/test_factory.py:89` `parametrize_mesh_with_fabric` does.

**On this Blackhole galaxy that cannot work.** Opening `(1,8)` or `(2,8)` as a *top-level* mesh dies
in fabric bring-up:

```
Fabric Router Sync: Timeout after 10000 ms on Device 1. Expected status 0xa2b2c2d2
  (LOCAL_HANDSHAKE_COMPLETE) … furthest-behind stage: STARTED
```
(`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:200`) — the routers on the opened
devices wait for an ethernet handshake with partners *outside* the mesh, which have no kernel
running. Reproduced with and without `TT_MESH_GRAPH_DESC_PATH`, under `STRICT_INIT` and
`RELAXED_INIT`. The fix is `DEC-080`: open the full `(4,8)` once and
`mesh_device.create_submesh(...)` per case (`tt_metal/api/tt-metalium/mesh_device.hpp:307`).

This is machine-specific, not universal: on a LoudBox / T3K, `(1,8)` **is** the whole machine and the
minimax form is the right one. A port must switch back. `R-031`.

### F.12 — Submeshes that overlap need `quiesce_devices()`, and forgetting it **hangs the machine**

`tt_metal/api/tt-metalium/mesh_device.hpp:296` requires a barrier "between phases that use
overlapping submeshes on the same physical devices" and names `quiesce_devices()` (`:305`). Nothing
enforces it, and `G-TP-PARITY` — which compares `(1,1)` against `(1,TP)` — is exactly such a pair.

Measured, one variable at a time: `(1,2)` collective then `(1,8)` collective with both submeshes live
and **no barrier** → **hang**; the same two phases with `parent.quiesce_devices()` between → ok;
`(1,8)` alone in its own process → ok.

**Two things make this worth an appendix entry rather than a code comment.**

1. **A hang is not contained.** After one, *every* later collective on the box hangs too — including a
   `(4,8)` all-reduce that had passed forty seconds earlier — until `tt-smi -r`. A pytest session that
   hits it turns every remaining gate into a false FAIL. Any harness that can hang should run its
   cases in **subprocesses with a timeout** so a hang is a recorded measurement (`DEC-082`,
   `tests/fabric_topology_matrix.py`).
2. **The first diagnosis was wrong, and plausible.** The hanging run was configured as `DEC-020`
   prescribes (`Topology.Linear`, `num_links=1`) on a `(1,8)` submesh, and there was a tidy story: the
   system mesh is `MeshShape([8, 4])`, so a logical `(4,8)` row of 8 is linear index `r*8 + c` →
   physical `(idx // 4, idx % 4)` = two physical rows, and a non-cyclic route along that axis
   plausibly does not exist. `(1,8)` + Ring then passed at 1 and 2 links, which *appeared* to confirm
   it. Running `(1,8)` + Linear **alone** falsified the whole story. `DEC-081` keeps the wrong
   argument on the record next to the measurement that killed it, because the failure mode — a
   variable nobody was varying deliberately — is the transferable lesson.

`R-032`.

### F.13 — `G-CHUNK-ATTN`'s threshold needs a stated **depth**, or it measures depth instead of the op

`bringup_log/06_GATES.md:29` (P7's row; the gate is not in Appendix A) states `>= 0.999 chunked ==
one-shot on the attention OUTPUT`. P8's first implementation compared the **KV product at every
layer** and reported the min over 32 layers, which failed at 0.99628 — and that failure was an
artefact of the metric, not of the ring:

| layer | ring vs one-shot, K |
|---|---|
| 0 (no attention has run) | **1.00000** |
| 1 (**one** attention layer) | **0.99996** |
| 8 | 0.99952 |
| 22 (the min) | **0.99628** |

Layer 31's K is one attention output pushed through 31 residual streams, each amplifying the
difference; holding it to a per-op threshold measures depth. `DEC-085` resolves it: assert `0.999` at
**layer 1**, and gate the deep layers with the per-layer error **step** (`DEC-047`'s unchanged 4.0x
from layer 3 — measured 1.90x) plus both paths' PCC against the fp32 golden at `G-CHUNK`'s
thresholds. **Every threshold used was set before the measurement existed**; none was refitted.

The general rule, which also applies to `G-CHUNK` and `G-MODEL`: **a mutual-PCC gate must name the
depth at which it applies.** "Path A == path B" is a per-op claim; the min over a 32-layer stack is a
different quantity and needs a different instrument.

Related, and measured (`DEC-084`, `G-SP-RING`): the ring's `fp32_dest_acc_en=False` is **not** a
preference. `use_streaming_compute = !fp32_dest_acc_en`
(`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1304`) and
`kv_actual_isl` requires the streaming path (`:1306`), so for chunked prefill the two flags are
mutually exclusive by construction and `True` is refused with a `TT_FATAL`. Cost, measured: the ring
op alone sits **7.98x** off its noise floor (against the single-card SDPA's **71x**, E.5), and end to
end the chunked path carries **1.45x** the error of the one-shot path (min K 0.99695 vs 0.99789).
Expect it, attribute it, and set any future KV threshold against the **chunked** number. `R-033`.


### F.12 Phase order correction — run P10 (integration) BEFORE P9 (cleanliness)
The recipe orders the phases P9 (cleanliness) then P10 (disaggregated-prefill integration). That is
the wrong way round and this run swapped them.

P9's gate is a whole-package sweep: no TODOs without a filed issue, every env var in the README's
table, every `tt/` module owning a test, `README.md` complete, import hygiene on the adapter. **P10
then adds `tt/runners/adapters/`, a manifest, a KV-chunk-table module and new env vars** — so a P9 run
that precedes it is auditing a package that is about to change, and has to be redone. Worse, one of
P9's own items (*"importing an adapter stays cheap — no reference-model, device or runtime imports at
module load"*) is **unrunnable before P10**, because no adapter exists yet.

**Run P10, then P9 as the final sweep.** The gate index in Appendix A keeps its numbering; only the
execution order changes. Anyone re-running this recipe should do the same.


### F.13 P10 executed: what changed in the two shared files, and the citations that moved
P10 ran and made the two edits outside the package that step 5 and step 3 above authorise. The
snippet in step 5 is therefore **historical** — the current dispatch is:

```python
_PACKED_GQA_MODELS = ("gpt_oss_d_p", "llama31_8b_d_p")   # prefill_producer.py:508
...
if ADAPTER.name == "minimax_m3":        return _read_slot_kv_and_check_pcc_m3(...)
if ADAPTER.name in _PACKED_GQA_MODELS:  return _read_slot_kv_and_check_pcc_gpt_oss(...)
return _read_slot_kv_and_check_pcc_mla(...)
```

with the reader's log line now naming `ADAPTER.name` instead of a hard-coded "GPT-OSS" (`DEC-105`).
Line numbers in `prefill_producer.py` shifted by +8/+11 as a result — `_read_slot_kv_and_check_pcc`
is now at `:511`, `_read_slot_kv_and_check_pcc_gpt_oss` at `:544`, `_read_slot_kv_and_check_pcc_mla`
at `:696` — and `scripts/verify_citations.py` was updated accordingly (662/662, 745/745).

**Two things the recipe's P10 section does not warn about, both of which cost a run to find:**

1. **`ADDING_A_PREFILL_MODEL.md` §2's `prefill_chunk` signature is incomplete.** The engine also
   always passes `d2h_service` **and `metadata_msg`** (`prefill_runner.py:364`). A runtime written to
   the doc dies with a `TypeError` on its first served chunk, after the mesh is open and the weights
   are loaded. `DEC-106`.
2. **The engine mutates the config `load_hf_config` returns** (`prefill_runner.py:477`), so a frozen
   dataclass cannot be returned as-is. `DEC-100`.

**And one gate the recipe's Appendix A does not list but should:** `G-KV-TABLE`. `G-MOCK-MIG` scores
a PCC over one slot and cannot separate a wrong address table from a numerical problem; `DEC-111`
explains why that needed its own bit-exact gate, on the same reasoning `DEC-087` used for `G-KV-TP8`.
