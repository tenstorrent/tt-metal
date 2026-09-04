# 06 — Gate ledger (append-only)

Verdicts: `PASS` · `FAIL` · `PASS-WITH-DEVIATION` (needs a `DEC`) · `BLOCKED` (needs a `07_RISKS.md`
entry naming the blocker) · `NOT-RUN` (needs the reason). A gate with no raw log did not happen.

| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| G-CARD | P0 | every architectural fact has provenance | doc review + 0 bad citations | 53/53 citations verified, 115/115 doc refs resolved, 0 `UNVERIFIED` rows | PASS | 2026-09-04 | `raw/G-CARD_20260904T034338Z.log` (+ `raw/G-CARD_20260904T034303Z.log`) |
| G-REF | P1 | the reference is deterministic and self-consistent | bit-identical x2; hand-written vs HF PCC >= 0.9999 (expect bit-exact) | 10/10 tests pass; both oracles SHA-256 `0a163300…`; PCC **1.0**, `max|Δ| = 0.0` | PASS | 2026-09-04 | `raw/G-REF_20260904T035140Z.log` |
| G-SURVEY | P2 | reuse decided, with citations | doc review + 0 bad citations | 30 component rows, 30 with a decision, 27 with a full `path:line`; 123/123 citations verified, 195/195 doc refs resolved | PASS | 2026-09-04 | `raw/G-SURVEY_20260904T035957Z.log` |
| G-OUTLINE | P3 | file tree + shapes pinned; every gate owns a file | doc review + 0 bad citations | 41 files contracted (49/49 non-`__init__` tree files); **32/32** Appendix A gate rows owned; 18/18 shape rows filled; 225/225 citations, 338/338 doc refs | PASS | 2026-09-04 | `raw/G-OUTLINE_20260904T083915Z.log` |
| G-CCL-PLAN | P4 | every collective placed and justified | doc review + 0 bad citations | 8/8 module placement rows justified; **9** collective call sites with `dim`/`axis`/topology/`num_links`; 14 semaphores (6+4+2+2), barrier depth 2; 253/253 citations, 404/404 doc refs | PASS | 2026-09-04 | `raw/G-CCL-PLAN_20260904T084559Z.log` |
| G-MESH | P5.1 | `MeshConfig` arithmetic + refusals; `CCLManager` builds and allocates its semaphores once | exact asserts | 16/16 tests pass; `shard_size(4096)=512`, `shard_size(14336)=1792`; grid **(12,10)**, CCL offset **(11,0)**; semaphores **6/4/2/2 = 14**, unchanged after 128 barrier cycles; 4/4 sub-axis-TP shapes refused | PASS | 2026-09-04 | `raw/G-MESH_20260904T085727Z.log` |
| G-RMS | P5.2 | plain RMSNorm vs an fp32 torch reference, `(1,1)` | PCC >= 0.9999; gap to the floor recorded | random weights **0.9999957 / 0.9999958 / 0.9999957** (floor 0.9999973/0.9999973/0.9999972 -> **1.56 / 1.57 / 1.54x**); real layer-0 weights **0.9999971** x3 (floor 0.9999986 -> **2.11 / 2.13 / 2.12x**) | PASS | 2026-09-04 | `raw/G-RMS_20260904T090144Z.log` |
| G-ROPE | P5.3 | llama3-scaled RoPE + the Meta convention, `(1,1)` | PCC >= 0.999; control must collapse | **0.9999969 / 0.9999964 / 0.9999959** (floor 0.9999983/0.9999982/0.9999980 -> **1.76 / 1.95 / 2.08x**); control **0.01367**; tables bit-identical to the test's own Meta tables | PASS | 2026-09-04 | `raw/G-ROPE_20260904T091040Z.log` |

```
STATUS after P0: gates PASS=1 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P1 (reference)
STATUS after P1: gates PASS=2 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P2 (repo survey)
STATUS after P2: gates PASS=3 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P3 (package outline)
Open DECs needing review: DEC-002 (docs/ + scripts/__init__.py vs the P3 tree — P3/P9 must settle),
DEC-004 (chunk size deferred to P7), DEC-008 (TestFactory.setup_test deferred to P5.1),
DEC-012 (checkpoint loader parked in tests/test_factory.py until P6.2),
DEC-013 (import gpt_oss_d_p/utils rather than copying it, against the P3 tree's comment)
```

**STOPPED HERE, ON A GATE BOUNDARY.** P0, P1 and P2 are complete and gated; **P3 (package outline,
`03_OUTLINE.md`, gate `G-OUTLINE`) is next**, followed by P4. No device code exists yet: `tt/` holds
only its `__init__.py`. The two things P3 must settle before it writes the tree are `DEC-002`
(does the package keep `docs/` and `scripts/__init__.py`, and does it vendor a copy of the recipe?)
and `DEC-013` (is `utils/` created at all, given the helpers are imported from
`models/demos/gpt_oss_d_p/utils/`).

---

### G-CARD — every architectural fact has a source
- **Command:** document review of `00_MODEL_CARD.md` +
  `python models/demos/llama31_8b_d_p/scripts/verify_citations.py`
- **Mesh / device:** none for the review; `ttnn.get_num_devices()` / `get_arch_name()` opened and
  closed the cluster once to record the machine facts.
- **Threshold:** every card row has a non-empty `Source`; zero rows say "from memory"; the
  "does NOT have" section exists; the `(mesh, TP, SP)` arithmetic is shown including the
  `TP == num_key_value_heads` derivation; every `UNVERIFIED` row also appears in `07_RISKS.md`;
  `verify_citations.py` exits 0. Source: recipe P0 `G-CARD`.
- **Measured:**
  - card rows with a `Source`: **100%** (2 identity + 25 architecture + 5 derived-geometry + 4
    deployment rows); rows sourced "from memory": **0**; rows marked `UNVERIFIED`: **0** (so the
    `07_RISKS.md` cross-check is vacuously satisfied, and §5 says so explicitly).
  - `verify_citations.py`: `citations checked 53 / verified 53 / mismatched 0 / missing 0`;
    `doc refs scanned 115 / resolved 115 / unresolved 0`; exit 0.
  - identity: `md5 3cd5831d379b509d53afade0e24c36e9` for all three of
    `$HF_MODEL/config.json`, `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`,
    and the bundled `configs/Llama-3.1-8B-Instruct/config.json`.
  - machine: `get_num_devices() = 32`, `get_arch_name() = 'blackhole'`,
    `hasattr(ttnn, "BlackholeComputeKernelConfig") = False` — all three as the recipe states.
  - arithmetic, computed in the raw log rather than asserted in prose: `head_dim 4096/32 = 128`;
    `4096/8 = 512 (%32=0)`; `14336/8 = 1792 (%32=0)`; `128256/8 = 16032 (%32=0)`;
    `nqh 4 >= nkv 1 && 4%1 == 0`; `SP = 4` → `CHUNK_SIZE % 128 == 0`.
- **Verdict:** **PASS**
- **Negative control:** this gate produces no PCC, so §1.4 exempts it — but it has one anyway, and it
  fired. The first two runs of `verify_citations.py` **failed** (exit 1): six of this agent's own
  first-draft `CITES` line numbers were wrong (`sdpa_device_operation.cpp` 99→98,
  `PREFILL_MIGRATION_TESTING.md` 31→62, the vendored `config.json` 7→13,
  `models/demos/minimax_m3/conftest.py` 17→16, `models/demos/gpt_oss_d_p/utils/substate.py` 6→15), and a
  citation-shadowing false positive resolved a bare `tt/config.py` onto a gpt-oss file. Both raw logs
  are kept; the failing run is the control that shows the check discriminates.
- **Deviations:** none to the gate. Two recipe-internal conflicts were resolved by `DEC-002`
  (`docs/` + `scripts/__init__.py`, and not vendoring the recipe into the package) and one by
  `DEC-009` (which test file owns the bundled-config byte-identity assertion).
- **What this does NOT prove:** that the card's *interpretation* of the config is right — only that
  every claim is traceable. `hidden_act: silu` is a fact; "SwiGLU is `down(silu(gate)*up)`" is a
  reading of it, and `G-REF` is what checks the reading. It also does not prove the checkpoint's
  **weights** match the config (`G-WEIGHTS`, P6), nor that the live HF repo agrees (`07_RISKS.md`
  R-002).
- **Notes:** `raw/G-CARD_20260904T034303Z.log` is the full transcript (identity, config dump, device
  facts, arithmetic, first citation pass); `raw/G-CARD_20260904T034338Z.log` is the re-run of the
  identity + citation passes after `DEC-003`'s fix, and is the log the verdict cites.

---

---

### G-REF — the reference is deterministic and self-consistent
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q`
- **Mesh / device:** **none** — host only, no `ttnn` import, no `mesh_device` fixture (`DEC-010`).
  Runs on a box with no card and while the mesh is busy.
- **Input distribution:** standard normal (`torch.randn`), seed 0, for hidden states and for every
  random weight (`scale = 0.02` for projections, `1.0 + 0.02*randn` for norm gains). randn is the
  harder of the two distributions for a norm and the recipe requires the choice to be stated rather
  than shopped (§2.1(b)). Real layer-0 weights from the staged checkpoint are used by the two
  `requires_hf_reference` tests. seq_len 128 (random weights) / 64 (real weights).
- **Reference dtype policy:** **fp32 everywhere** (`DEC-006`). The HF layer is built bare from the
  bundled config and `.float()`ed with `_attn_implementation = "eager"` — never `from_pretrained`,
  which would load at the checkpoint's `torch_dtype: bfloat16` and give a reference that shares the
  device's own rounding. Checkpoint tensors are `.float()`ed at load. Nothing is rounded to bf16 in
  this gate.
- **Threshold:** (a) two runs bit-identical; (b) hand-written vs HF PCC ≥ 0.9999 on one decoder
  layer, expecting bit-exact; (c) `01_REFERENCE.md` documents the invocation and the dtype policy.
  Source: recipe P1 `G-REF` / Appendix A.
- **Noise floor (computed):** **not applicable, and that is the honest answer.** Both sides here are
  fp32 torch on identical inputs, so the floor for this comparison is exact equality — PCC 1.0, and
  an error ratio of 0/0. The gate is therefore stated as bit-exactness; the 0.9999 threshold is a
  ceiling on how far a *faithful transcription* may drift, not a target to sit near. Measured
  `max|Δ| = 0.0`, i.e. at the floor. The floor helpers themselves (`quantize_like_device`,
  `err_ratio`) ship in `tests/test_factory.py` in this phase but are first *used* by P5.
- **Measured:** **10 passed, 0 failed, 13.6 s.**

  | Check | Result |
  |---|---|
  | determinism, hand-written (SHA-256 ×2) | `0a16330068bce3807911a80462fad1b38c84cfd3f08ba9098df38156f67cb96b` / identical |
  | determinism, HF `LlamaDecoderLayer` (SHA-256 ×2) | `0a16330068bce3807911a80462fad1b38c84cfd3f08ba9098df38156f67cb96b` / identical |
  | hand-written vs HF, random weights, seq 128 | **PCC 1.0**, `max|Δ| = 0.000e+00` |
  | hand-written vs HF, **real layer-0** weights, seq 64 | **PCC 1.0**, `max|Δ| = 0.000e+00` |
  | llama3 `inv_freq` vs `models/tt_transformers/tt/common.py:489` | `max|Δ| = 0.000e+00` |
  | bundled `config.json` vs checkpoint's | byte-identical (`filecmp`, `shallow=False`) |
  | `rope_theta` on transformers 5.12.1 | `cfg.rope_theta` → `AttributeError`; `getattr(cfg,"rope_theta",10000.0)` → **10000.0**; `get_rope_theta(config.json)` → **500000.0** |
  | `position_embeddings` in `LlamaAttention.forward` / `LlamaDecoderLayer.forward` | present in both (trap 5 branch resolves to the modern path) |

  The two oracles produce the **same hash**, which is the bit-exactness result restated.
- **Negative control:** five, all of which must fail and all of which do.
  1. **Causality** (recipe P1 trap 3): perturbing the last token moves rows `[:-1]` by
     `max|Δ| = 0.000e+00` with an explicit mask and by **`3.515e+00`** with `attention_mask=None` —
     HF's own default. Measured identically for both oracles, so the transcription reproduces the
     trap rather than hiding it.
  2. **Wrong theta** (10000.0, the value the `getattr` trap substitutes): **PCC 0.87437**.
  3. **GQA `repeat` instead of `repeat_interleave`**: **PCC 0.55396**.
  4. **No RoPE at all** (`cos=1`, `sin=0`): **PCC 0.86480**.
  5. **llama3 scaling inactive**: scaled vs unscaled `inv_freq` differ by `max|Δ| = 8.894e-04`, and
     `cos` at position 8192 by `1.882e+00` — so the piecewise scaling is demonstrably applied, not
     assumed.

  Note what controls 2 and 4 say about thresholds: a RoPE that is **wrong at every position** still
  scores 0.86–0.87 at the layer level. Anything in the 0.9 band is not "nearly right".
- **Verdict:** **PASS**
- **Deviations:** none to the gate. Three recipe-ordering issues were resolved by `DEC-008`
  (`TestFactory.setup_test` deferred to P5.1 — it builds `MeshConfig`/`CCLManager`, which are P5
  deliverables), `DEC-012` (the checkpoint loader parked in `tests/test_factory.py` until `ModelArgs`
  exists in P6.2) and `DEC-009` (the bundled-config byte-identity assertion lives in this file).
- **What this does NOT prove:** that either oracle is **right about Llama's architecture**. Two
  transcriptions agreeing bit-exactly proves the transcription is faithful and nothing more; a shared
  misreading survives untouched (`07_RISKS.md` R-006). It proves nothing about device code — none
  exists yet — nothing about layers other than 0 with real weights, nothing about the weight
  *mapping* (`G-WEIGHTS`), and nothing at sequence lengths past 128, where the llama3 scaling's
  interpolated band starts to matter (`G-ROPE` and `G-MODEL` own that).
- **Raw log:** `raw/G-REF_20260904T035140Z.log`. The earlier `raw/G-REF_20260904T035011Z.log` is the
  same suite before the trap-5 signature test was added; both are kept.

---

### G-SURVEY — reuse decided, with citations
- **Command:** document review of `02_SURVEY.md` (structure checks scripted into the raw log) +
  `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` + the per-phase regression run
  `pytest models/demos/llama31_8b_d_p -q`
- **Mesh / device:** none.
- **Threshold:** every component row has a decision + citation; the "not bringing over" list exists;
  the `models/common/` verdict is present with its two citations; no row's decision is "write" where
  an importable equivalent exists (else a `DEC`); the verifier re-verifies every `path:line` in the
  survey. Source: recipe P2 `G-SURVEY`.
- **Measured:**
  - **30** component rows (the recipe's minimum list is 20; the extra 10 are `substate`,
    `num_links`/cache naming, the noise-floor helpers, the KV chunk table, the fabric descriptors,
    the PCC helpers, the compute-kernel-config convention, indexed RoPE, the ring SDPA, and the
    normalised `hf_config` constructor).
  - **30/30** rows carry a decision. Split: **adapt 16**, **import 9** (6 plain + `import + convention`
    + `import (config, not code)` + `import the op`), **import-the-op-adapt-the-caller 3**,
    **adapt + import the math 1**, **copy-with-a-`DEC` 1**, **write 0**.
  - **27/30** rows carry a full `path:line`. The three that do not (23, 27, 30) cite a whole file
    plus abbreviated `:NN` refs in the same row, or — row 30 — two `.textproto` config files where a
    line number would be meaningless. Both forms are resolved by the verifier's pass 2.
  - **15** "not bringing over" entries (MoE, router, EP dispatch/combine, shared expert,
    `swigluoai`, sinks, sliding window, QK-norm, partial RoPE, YaRN/mscale, MLA, sparse/MSA, MXFP4,
    biases, decode/paged/trace).
  - `models/common/` verdict present with both required citations
    (`models/common/modules/mlp/mlp_2d.py:461`, `models/common/models/llama3_8b/model.py:890`), plus
    two the recipe does not require and that this run verified independently:
    `models/common/modules/mlp/mlp_2d.py:256`/`:259` (the `cluster_axis = 1` reduce-scatter) and
    `models/common/modules/attention/attention_1d.py:319` (the only `Attention*` class under
    `models/common/modules/`).
  - `verify_citations.py`: `citations checked 123 / verified 123 / mismatched 0 / missing 0`;
    `doc refs scanned 195 / resolved 195 / unresolved 0`; exit 0.
  - per-phase regression: `10 passed` (the whole package suite, 13.6 s) — unchanged from P1, i.e.
    P2 broke nothing.
- **Verdict:** **PASS**
- **Negative control:** the citation pass, again, and it fired twice.
  1. Of the 70 `CITES` entries added for this survey, **one** was wrong on the first run —
     `models/demos/gpt_oss_d_p/README.md:24` (the P7-unification roadmap line is at `:26`) — cited
     inside `DEC-013` as the reason a cross-package import is acceptable. It would have read as
     authoritative.
  2. The gate's own **structure check** was wrong before the survey was: the first run reported
     "rows with a decision AND a path:line: 18/30", which was a broken regex (the `**bold**` markers
     around `**import**`), not a broken survey. Both raw logs are kept; the first
     (`raw/G-SURVEY_20260904T035900Z.log`) is the control showing the check discriminates, the
     second is the verdict's evidence. *A failing check is not evidence of a failing artefact until
     the check's own logic is verified* — the same lesson §2.5 states for probes.
- **Deviations:** none to the gate. One decision deviates from the recipe's *tree comment* rather
  than from the gate: `DEC-013` imports `models/demos/gpt_oss_d_p/utils/` instead of copying it, as
  the P3 tree's `# (copy from gpt_oss_d_p/utils)` suggests, because agent-contract rule 4 says reuse
  means import.
- **What this does NOT prove:** that the reuse decisions are *correct*. A row saying "import X" is
  not evidence that X works at Llama's shapes — nothing in P2 executes any of the cited code.
  `07_RISKS.md` R-007 (the chunked-KV / indexed-RoPE ops have no Llama-shaped upstream exercise) and
  R-009 (no dense, bias-free, full-RoPE attention template exists) name the two widest gaps, and
  P5's gates are what close them. It also does not prove the *absence* of a better template
  somewhere in the tree — the search was directed by the recipe's list of locations, not exhaustive.

---

### G-OUTLINE — file tree + shapes pinned; every gate owns a file
- **Command:** document review of `03_OUTLINE.md`, with the structure checks scripted into the raw
  log, + `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` + the per-phase regression
  run `pytest models/demos/llama31_8b_d_p -q`
- **Mesh / device:** none. P3 writes no code and opens no device.
- **Threshold:** `03_OUTLINE.md` lists every file with (i) a one-sentence responsibility, (ii) a
  public interface signature, (iii) input/output tensor shapes with dtype and layout, (iv) the
  template it mirrors (`path:line`); **every Appendix A gate maps to a named owner in the tree**; and
  the per-layer tensor-shape table is filled in with real numbers. Source: recipe P3 `G-OUTLINE`
  (`BRINGUP_RECIPE.md:857-863`).
- **Measured:**
  - **32/32** Appendix A gate rows have a named owner (§4), plus the per-phase regression gate. The
    Appendix A table has 32 gate rows, not 33: `G-WEIGHTS` appears twice (P6.2 and its P8 extension)
    and `G-KV-TP8` is easy to miscount because it is the one gate id containing a digit.
  - **41** files in the committed tree at the end of P10 (9 exist today); **49/49** non-`__init__.py`
    files are named in a per-file contract, including all 22 test/harness files (§2.15's per-test-file
    table) and `README.md` (§2.16).
  - **18/18** rows of the per-layer shape table carry four filled cells with no unresolved `/TP` or
    `/SP` — real numbers at `(4,8)`, TP=8, SP=4: `512`, `128`, `1792`, `16032`, `[1,4,S_loc,128]`,
    `[1,1,S_loc,128]`.
  - **6** deviations from the recipe's tree, each labelled `[DEV-n]` and each justified in §1.1.
  - `verify_citations.py`: `citations checked 225 / verified 225 / mismatched 0 / missing 0`;
    `doc refs scanned 338 / resolved 338 / unresolved 0`; exit 0. `CITES` grew by **102** entries
    this phase (123 → 225).
  - per-phase regression: `10 passed` in 13.76 s — unchanged from P1/P2, i.e. P3 broke nothing.
- **Verdict:** **PASS**
- **Negative control:** doc gates produce no PCC, so §1.4's four numeric fields are waived
  (`BRINGUP_RECIPE.md:280-281`) — but the two mechanical checks both fired before they passed, which
  is what shows they discriminate.
  1. **The citation verifier caught one wrong line** in this phase's own first draft of `CITES`:
     `models/demos/gpt_oss_d_p/tt/attention/prefill.py:104` was claimed to hold
     `activation_dtype = ttnn.bfloat16` (it holds `hidden_size = hidden_states.shape[-1]`; the
     activation-dtype ladder is at `:106-109`). That reference is cited inside `DEC-022` as the
     evidence for the dtype ladder, so it would have read as authoritative.
  2. **The structure check failed its first run**, correctly: 27/49 tree files had a contract entry,
     because the 22 test files were covered only collectively in §2.15 and in the gate-owner map. The
     fix was the per-test-file table (gate / mesh / reference / negative control per file), which is
     the more useful artefact — P5 now inherits its control for each gate rather than inventing one.
     A third failing run was the check's **own** bug, not the document's: its gate regex was
     `G-[A-Z-]+`, which silently dropped `G-KV-TP8` and reported 31 gates instead of 32. *A failing
     check is not evidence of a failing artefact until the check's own logic is verified* — the same
     lesson §2.5 states for probes, and the second time this run has hit it (`G-SURVEY`'s control 2).
- **Deviations:** none to the gate. Six deviations from the recipe's **tree** (`03_OUTLINE.md` §1.1),
  of which two are new judgement calls (`DEC-017`, `DEC-018`), one is a recipe omission that is not a
  judgement call at all (`[DEV-4]`: `tt/runners/adapters/__init__.py`, without which the registry
  cannot import the adapter), and three follow from earlier decisions. One recipe sentence had to be
  *resolved* rather than obeyed (`03_OUTLINE.md` §5.1: `BRINGUP_RECIPE.md:1022-1024` requires both
  that sub-axis TP raises and that TP need only *divide* the axis — `4` divides `8`, so the two
  halves contradict each other; the refusal is taken as binding).
- **What this does NOT prove:** that any of these interfaces or shapes are **right**. Nothing in P3
  runs. Every signature here is a prediction that P5–P10 may falsify, and the shape table is
  arithmetic on the P0 card rather than a measurement — a wrong `config.json` reading would propagate
  through it untouched (`G-WEIGHTS` and `G-MODEL` are what catch that). It also does not prove the
  gate→owner map is *sufficient*: a gate with a file is not a gate with a test, and four gates
  (`G-MESH`, `G-SEMAPHORE`, `G-WEIGHTS`, `G-TP-PARITY`) exist only because the recipe warns they are
  the ones that go unowned. Finally, the three deferred numbers (`CHUNK_SIZE`, `MAX_SEQ_LEN`,
  matmul block sizes) mean §3's KV-cache row is parameterised, not pinned.
- **Raw log:** `raw/G-OUTLINE_20260904T083915Z.log`

---

### G-CCL-PLAN — every collective placed and justified
- **Command:** document review of `04_CCL_PLAN.md`, with the structure checks scripted into the raw
  log, + `python models/demos/llama31_8b_d_p/scripts/verify_citations.py` + the per-phase regression
  run `pytest models/demos/llama31_8b_d_p -q`
- **Mesh / device:** none. P4 writes no code and opens no device.
- **Threshold:** `04_CCL_PLAN.md` contains the `(mesh, TP, SP)` arithmetic; the collective-placement
  table with **every row justified**; the residual-scheme `DEC` **with the cost-equivalence
  argument**; the semaphore-lifetime statement ("allocated once in `CCLManager.__init__`, cycled per
  call, never per layer") **and its depth**; and a list of **every** collective call site with its
  `cluster_axis`, `dim` and `topology`. Source: recipe P4 `G-CCL-PLAN`
  (`BRINGUP_RECIPE.md:975-981`).
- **Measured:**
  - `(4, 8)`, TP=8 on `tp_axis = 1`, SP = `32 / 8 = 4` on `sp_axis = 0`, `num_links` 2 at `(4,8)` /
    1 on any `(1,N)` submesh. `TP == num_key_value_heads == 8` stated as an equality with both bounds.
  - **8/8** placement rows (Embedding, RMSNorm, Attention, Attention-SP, MLP, LM head, DecoderLayer,
    Model) carry a non-empty justification, and each justification says *why the tensor is incomplete
    at that point* rather than which collective is conventional there.
  - **9** collective call sites, each with all four of `dim`, `axis`, topology and `num_links` filled
    — 4 of them the scheme-B/P8 seams that **refuse** until then, 2 of them SP-axis and P8-only.
  - **3/3** scatter-width rows arithmetically correct: `4096/8 = 512 = 16*32`,
    `128256/8 = 16032 = 501*32`, `256/8 = 32`. No `pad_size` is needed anywhere — computed, not
    asserted.
  - Semaphores: `3*2 = 6` RS + `2*2 = 4` AG + `2*1 = 2` barrier + 2 ring-attention = **14**, and the
    table's arithmetic re-checks in the raw log. Barrier depth **2**, with the one-op reuse gap
    written out and 4 barrier-consuming collectives per layer counted (128 per forward).
  - Collective op usage counts **re-measured** on this tree rather than quoted from the recipe:
    `all_gather_async` **29**, `reduce_scatter_minimal_async` **18**, `all_reduce_async` **2** —
    identical to `BRINGUP_RECIPE.md:925-927`.
  - All three BH-galaxy torus mesh-graph descriptors named in §7 exist (listed in the raw log).
  - `verify_citations.py`: `citations checked 253 / verified 253 / mismatched 0 / missing 0`;
    `doc refs scanned 404 / resolved 404 / unresolved 0`; exit 0. `CITES` grew by **28** this phase
    (225 → 253).
  - per-phase regression: `10 passed` in 13.77 s — unchanged.
- **Verdict:** **PASS**
- **Negative control:** doc gate, so §1.4's four numeric fields are waived
  (`BRINGUP_RECIPE.md:280-281`). The structure check is the control and it **failed its first run**,
  on two real defects in this phase's own document:
  1. §1.1 asserted "TP = 8 is an equality" without ever writing the words
     `num_key_value_heads` — the token the recipe's own derivation turns on, and the one a reviewer
     greps for. Fixed by stating `TP == num_key_value_heads == 8` explicitly.
  2. The semaphore-lifetime statement was line-wrapped mid-phrase, so the required sentence
     ("allocated once in `CCLManager.__init__`") did not exist as a contiguous string anywhere in the
     document. Cosmetic to a human reader and invisible to `grep` — which is the failure mode the
     check exists for.
  The check's other three sections re-derive arithmetic rather than matching text (scatter widths,
  semaphore counts, the 14 total), so a wrong number in the document fails them regardless of how it
  is worded.
- **Deviations:** none to the gate, and none to the recipe's plan: the two-object pattern
  (`CCLManager` + `MeshConfig`), the TP-axis-only rule, all-reduce as RS+AG rather than
  `all_reduce_async`, and scheme A are all taken as the recipe prescribes. Five decisions were logged
  (`DEC-024`–`DEC-028`), of which `DEC-026` (ship barrier depth 2, do not reset across chunks) and
  `DEC-028` (the one allowed raw `ttnn.all_gather`) are the two the recipe explicitly demands "either
  way".
- **What this does NOT prove:** **nothing about any collective actually running.** No device was
  opened; every row of §5 is a plan, and four of them describe code that will *refuse* to run in this
  iteration. Specifically unproven: that barrier depth 2 is sufficient (`G-RACE`), that the chosen
  topology/fabric pairing works on this box (`G-FABRIC-MATRIX`), that the `(4,8)` shape can be opened
  and submeshed at all (P8 step 1), that scheme A's cost-equivalence claim — an **op-count**
  argument, not a measurement — holds in device time, and that the SP ring path's semaphores and grid
  offset compose with the pinned 8x8 SDPA grid (`G-SP-RING`). The plan also cannot prove
  *completeness* of the call-site list: it enumerates the collectives this design will issue, and a
  module written in P5 that reaches for a collective not in this table is a deviation the table will
  not detect on its own — `G-CLEAN`'s "no raw `ttnn.experimental.*` in a module" grep is what closes
  that.
- **Raw log:** `raw/G-CCL-PLAN_20260904T084559Z.log`

---

### G-MESH — `MeshConfig` arithmetic and refusals; `CCLManager` allocates once
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_mesh_config.py
  models/demos/llama31_8b_d_p/tests/unit/test_ccl_semaphores.py -x -q`
- **Mesh / device:** (a) none — device-free arithmetic; (b) `(1,1)`, Blackhole. Only (b) takes the
  `mesh_device` fixture (`BRINGUP_RECIPE.md:1027`).
- **Input distribution:** n/a — this gate has no numeric input. Its inputs are mesh shapes:
  `(1,1)`, `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)`, `(8,4)`, and the four refused
  `(mesh, tp)` pairs below.
- **Reference dtype policy:** n/a — no reference tensor. The "reference" is arithmetic stated in
  `00_MODEL_CARD.md` §4 and re-derived from the bundled `config.json` in the test rather than
  restated as a literal.
- **Threshold:** exact asserts (`BRINGUP_RECIPE.md:1726`). No PCC, so §1.4's floor field does not
  apply.
- **Noise floor (computed):** n/a.
- **Measured:**
  - **16/16 tests pass** (9 device-free + 7 on the card).
  - shard arithmetic at `(1,8)`/TP=8: `sp=1`, `shard_size(4096)=512`, `shard_size(14336)=1792`,
    both `% 32 == 0`.
  - deployment `(4,8)`/TP=8: `sp=4`; `tp == num_key_value_heads == 8`; `local_q=4`, `local_kv=1`,
    so SDPA's `nqh >= nkv && nqh % nkv == 0` holds as `4 >= 1 && 4 % 1 == 0`.
  - real compute grid **(12, 10)** — not 8x8 — and `ring_attention_ccl_core_grid_offset = (11, 0)`,
    i.e. `grid.x - 1`. `num_links = 1` at `(1,1)`
    (`models/demos/gpt_oss_d_p/utils/general_utils.py:33`: a single-row mesh gets 1 link).
  - the build-time form of the ring op's assert holds: a pinned 8x8 SDPA grid needs
    `8 <= grid.x - 1 = 11`.
  - semaphore inventory **6 RS / 4 AG / 2 barrier / 2 ring-attention = 14**, identical after
    32 layers x 4 collectives = 128 barrier cycles; all three ping-pongs cycle with period 2.
- **Verdict:** **PASS**
- **Negative control:** four, and all four fired.
  1. **Sub-axis TP refuses.** `MeshConfig((1,8), tp=4)`, `((1,8), tp=2)`, `((4,8), tp=4)` and
     `((1,8), tp=16)` all raise `ValueError: ... sub-axis TP is unsupported`. §1.4 admits a
     configuration that must *refuse* as a control (`BRINGUP_RECIPE.md:277-278`).
  2. **Its complement.** `(1,2)`, `(1,4)`, `(2,8)`, `(8,4)` with matching TP must **warn and
     build** — without this, "raise on anything unusual" would satisfy control 1 while making
     every `(1,1)` P5 gate unrunnable.
  3. **A per-layer `CCLManager`.** Three managers stand in for three layers and produce
     `3 x 14 = 42` semaphores, the shape of the bug (`n_layers x` the constant).
  4. **`reset_global_semaphores` must not rewind the barrier index**, asserted, because `DEC-026`
     ships the template's deliberate skip and a later change must show up as a failure here.
- **Deviations:** none to the gate. Two `DEC`s came out of writing it: `DEC-029` (four dead
  members of the `CCLManager` template dropped, which shortens `03_OUTLINE.md` §2.2's attribute
  list) and `DEC-034` (the `prefer-expect-error` hook fires on the fixture's name in **prose**, so
  a docstring had to be reworded).
- **Raw log:** `raw/G-MESH_20260904T085727Z.log`
- **What this does NOT prove:**
  - **that any collective works.** At `(1,1)` no collective is issued: `MeshConfig`'s three
    wrappers are never called by this gate, only constructed around. `G-TP-PARITY` and
    `G-FABRIC-MATRIX` (P8) are what exercise them.
  - **that the semaphores are correct under concurrency.** This is a counting and cycling test on
    a single card. `G-RACE` (three runs, one process, one `CCLManager`, bit-identical) is the one
    that can see a race, and `G-SEMAPHORE`'s target-mesh half is P8's.
  - **that `num_links` is right for the deployment.** `(1,1)` yields 1 link by the helper's
    single-row branch, so the `num_links = 2` the `(4,8)` deployment uses is untested here (P8
    step 3).
  - **that the deployment mesh can be opened at all.** The `(4,8)` assertions in this gate are
    arithmetic on a shape tuple, not a device open (P8 step 1).
- **Notes:** `G-SEMAPHORE` is a **P8** gate and is *not* being recorded as PASS here; its
  one-card half runs in this file because `G-MESH` already requires the assertion
  (`BRINGUP_RECIPE.md:1025-1026`) and writing it twice would let the two copies disagree
  (`03_OUTLINE.md` §1.1 `[DEV-6]`). The 16 tests above include those 5.

---

### G-RMS — plain RMSNorm vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_rms_norm_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole
- **Input distribution:** **standard normal** activations, `[1, 1, S, 4096]`, `S ∈ {32, 512, 4096}`,
  `reset_seeds` (repo-root `conftest.py:34`, seed 213919). Two weight sources, both run
  (`DEC-035`): standard-normal random weights, and the **real** `model.layers.0.input_layernorm.weight`.
  Stated because it must never be chosen to pass — recipe §2.1(b) measures the bf16 floor as
  identical under `rand[0,1)` and `randn`, and randn is the harder of the two for a norm.
- **Reference dtype policy:** reference weight and activations **fp32**, all arithmetic fp32
  (`DEC-006`). Only what the device *stores* is quantised, and only to compute the floor: bf16
  activations and a bf16 norm weight in its stored `(1, 1, 128, 32)` shape (`DEC-022`). A
  bf16-weight reference would share the device's own rounding and flatter the number — recipe
  §2.1(a) measures 0.9999867 against 0.99995 for that mistake.
- **Threshold:** PCC >= 0.9999 (`BRINGUP_RECIPE.md:1728`). The error ratio is **recorded, not
  asserted**: a correct module sits right on §2.2's 3x stage bound, so asserting it would gate on
  the wrong side of the noise (`BRINGUP_RECIPE.md:1043-1046`).
- **Noise floor (computed):** **0.9999973 / 0.9999973 / 0.9999972** with random weights;
  **0.9999986** at all three lengths with the real layer-0 weight. The floor **moves with the
  weight distribution** — a trained norm gain is a narrow positive distribution, not standard
  normal — which is the whole reason `DEC-035` runs both rather than picking one.
- **Measured:**

  | weights | S=32 | S=512 | S=4096 |
  |---|---|---|---|
  | random, PCC | 0.9999957 | 0.9999958 | 0.9999957 |
  | random, floor | 0.9999973 | 0.9999973 | 0.9999972 |
  | random, ratio | **1.56x** | **1.57x** | **1.54x** |
  | real layer-0, PCC | 0.9999971 | 0.9999971 | 0.9999971 |
  | real layer-0, floor | 0.9999986 | 0.9999986 | 0.9999986 |
  | real layer-0, ratio | **2.11x** | **2.13x** | **2.12x** |

  The real-weight figure **reproduces the recipe's expected value against the same floor**: §2.4
  predicts 0.9999955 with `fp32_dest_acc_en=True` against a 0.9999986 floor, and this module
  measures **0.9999971** — slightly better, at 2.1x the floor rather than ~3x.
- **The §2.4 A/B, measured in-suite** (`DEC-014`'s falsifier, `DEC-030`):

  | S | `fp32_dest_acc_en=True` | `=False` | error reduction |
  |---|---|---|---|
  | 32 | 0.9999957 (1.56x floor) | 0.9999707 (10.79x) | **6.90x** |
  | 512 | 0.9999958 (1.57x floor) | 0.9999633 (13.58x) | **8.66x** |

  So the flag is worth ~7-9x of module error on this box, against the recipe's stated ~7x — and
  note that **both** settings clear the 0.9999 threshold at S=32, which is precisely §2.2's point:
  the absolute PCC does not distinguish them and the ratio to the floor does.
- **Negative control:** the **zero-gain probe** — `weight = 0` must give `max|out| = 0.0`.
  Measured **0.0**. A Gemma `(1 + weight)` fold would instead return the normalised input, whose
  per-channel magnitude is ~1, so this is the discriminator between plain and folded RMSNorm and
  therefore between Llama and the two nearest templates' feature set (`00_MODEL_CARD.md` §3).
  A second control: building with an empty `state_dict` and no `tensor_cache_path` must raise
  rather than run on a `None` gain — Appendix B's "cache-only build silently wrong" row.
- **Deviations:** none to the gate. `DEC-035` records the two-weight-source choice, forced by the
  recipe specifying the input three different ways.
- **Raw log:** `raw/G-RMS_20260904T090144Z.log`
- **What this does NOT prove:**
  - **the distributed (3-op) branch.** It is dormant (`is_distributed=False`, `DEC-025`) and no
    test executes it. `DEC-031` and `07_RISKS.md` R-011 record that the template's version of that
    branch would in fact raise `TypeError` — which is exactly what a never-executed branch is worth.
  - **cache-only loading.** The gate builds from a `state_dict` every time; the
    `tensor_cache_path` branch is only proven to *refuse* when absent. `G-WEIGHTS` (P6.2) owns the
    positive case.
  - **that the norm is wired into anything.** `G-LAYER` and `G-MODEL` own placement — this gate
    would pass equally if the two norms in a layer were swapped, which is `G-LAYER`'s own control.
  - **`eps` correctness beyond agreement.** Both sides read `1e-05` from the same bundled
    `config.json`, so a wrong value in that file would cancel. `G-CARD` is the provenance check.

---

### G-ROPE — llama3-scaled RoPE, Meta convention, vs the HF `rotate_half` reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_rope_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole (two of the eleven tests are host-only)
- **Input distribution:** **standard normal**, `[1, 4, S, 128]` — 4 heads, the local Q-head count
  per chip at the deployment TP=8 — with `S ∈ {32, 512, 4096}`, `reset_seeds`.
- **Reference dtype policy:** fp32 input, fp32 cos/sin, fp32 arithmetic. The **HF** convention
  (`x * cos + rotate_half(x) * sin`) is the reference; the device runs the **Meta** convention, and
  both tables come from **one** frequency set, as
  `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83` `_build_cos_sin` does, so the
  test cannot silently compare two different RoPEs. For the floor, the three tensors the device
  stores — input, cos, sin — are quantised to bf16 and the rest stays fp32.
- **Threshold:** PCC >= 0.999 (`BRINGUP_RECIPE.md:1729`), expecting ~0.99999.
- **Noise floor (computed):** **0.9999983 / 0.9999982 / 0.9999980**.
- **Measured:** **0.9999969 / 0.9999964 / 0.9999959** at S = 32 / 512 / 4096 — **1.76x / 1.95x /
  2.08x** the floor. 11/11 tests pass.
  - `rope_params` reads `(theta, factor, original_max_position_embeddings) =
    (500000.0, 8.0, 8192)` through `get_rope_theta` / `get_rope_scaling` on the raw dict, each
    asserted non-`None` — which is `07_RISKS.md` R-005 enforced in device code rather than by
    convention.
  - `build_prefill_rope`'s device tables are **bit-identical** (`torch.equal`, `max|Δ| = 0.0`) to
    the test's independently built Meta tables at all three lengths. A table layout is a mapping
    claim, so it is gated on bit-equality, not PCC (recipe §2.5).
  - `get_rot_transformation_mat()` and `get_rot_transformation_mat(dhead=128)` are
    `torch.equal` and both 32x32 — recipe P1 trap 4 confirmed on this version, not assumed.
- **Negative control:** two, both fired.
  1. **HF-layout tensor into the Meta op** — the classic convention mismatch — scores
     **0.01367** (the recipe measured 0.01296 for the same mistake). Without it, 0.99999 could
     equally mean both sides are wrong the same way.
  2. **The llama3 scaling must be provably active.** Asserted on the piecewise **band structure**
     of the frequencies `apply_scaling` consumes: of 64 frequencies, **29 low** (wavelength >
     8192) divided by exactly 8.0, **29 high** (wavelength < 2048) `torch.equal` to the base
     frequencies, and **6 mid** strictly interpolated between the two. This is a **deviation from
     the recipe's stated control** and `DEC-036` records why: the recipe asks that the scaled and
     unscaled tables differ "for positions beyond `original_max_position_embeddings`"
     (`BRINGUP_RECIPE.md:1093-1095`), and measured, `max|cos_scaled - cos_unscaled|` is
     **1.99933 inside** the window and **1.99398 beyond** it — both saturated at the theoretical
     maximum of 2, because llama3 scaling divides long-wavelength frequencies at *every* position
     and `cos` oscillates. That assertion therefore cannot fail for the reason it exists. Both
     numbers are still recorded, as the recipe asks; the band test is what gates.
  3. A third refusal, counted with the controls: the contiguous builder must reject
     `start_pos > seq_len` (the `gather_cos_sin` out-of-bounds landmine), and the indexed builder
     must reject `chunk_size % (32*sp) != 0` and `max_seq_len % chunk_size != 0`.
- **Deviations:** `DEC-036` (the scaling control, above) and `DEC-033` (three signature deviations
  from `03_OUTLINE.md` §2.5, all forced by the helpers being wrapped). Verdict is `PASS` rather
  than `PASS-WITH-DEVIATION` because the gate's own threshold and its required controls are met —
  `DEC-036` makes the control *stronger* than specified, not weaker.
- **Raw logs:** `raw/G-ROPE_20260904T091040Z.log` (the verdict) and
  `raw/G-ROPE_20260904T090652Z.log` (a **failed** earlier run, kept deliberately: the first
  attempt at the recipe's scaling assertion recovered each frequency from a cos table by `arccos`,
  and for the lowest frequency `cos(1 * f)` rounds to 1.0 in fp32, giving `0/0 = nan`. That is
  `LANDMINES.md`'s "a failing probe is not evidence of a failing module until the probe's own
  numerics are checked", hit live — the module was correct throughout.) Three intermediate passing
  runs made while adding tests were discarded as superseded; they were authoring runs, not gate runs.
- **What this does NOT prove:**
  - **that Q/K projection weights are `reverse_permute`d on the real load path.** The HF -> Meta
    permutation is applied *by the test*, not by `tt/attention/weights.py`, which does not exist
    until P5.5. `G-ATTN`'s "loaded without the Meta permute" control (recipe: 0.9475) is what
    closes it — and note how high that broken variant scores.
  - **the indexed RoPE numerically.** `build_indexed_rope` is exercised only structurally, at
    SP=1, where the block-cyclic reorder is the identity; the table is asserted `torch.equal` to
    the plain Meta table and the two shape constraints are asserted as refusals. Nothing about the
    SP>1 layout is testable on `(1,1)`. `G-CHUNK` (P7) and `G-CHUNK-ATTN` (P8) own it.
  - **long-context correctness.** The longest sequence gated is 4096, well inside
    `original_max_position_embeddings` = 8192, so the scaled band of the tables is never exercised
    *by the device*: the band structure is proved on the host. Appendix B's "PCC good at short seq,
    bad past ~8192" symptom would still be invisible here; `G-MODEL` at long context is where it
    would show.
  - **that the RoPE is applied to the right tensors.** Only Q and K may be rotated; this gate
    rotates a bare tensor. `G-ATTN` asserts the invariant.

---

```
STATUS after P3: gates PASS=4 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P4 (parallelism + CCL plan)
STATUS after P4: gates PASS=5 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P5 (naive module implementations, bottom-up)
STATUS after P5.1-P5.3: gates PASS=8 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P5.4 (dense SwiGLU MLP, G-MLP)
Per-phase regression gate: whole package suite **47 passed, 0 failed**
(`raw/P5-REGRESSION_20260904T092212Z.log`)
Citations after P5.3: **279/279 verified, 0 mismatched; 522/522 doc refs resolved**, exit 0
(`raw/G-CITE_20260904T092334Z.log`)
Open DECs needing review: DEC-004 (chunk size deferred to P7 — now a parameter of build_indexed_rope, DEC-033),
DEC-012 (checkpoint loader moves to ModelArgs in P6.2), DEC-013/DEC-018 (import gpt_oss_d_p/utils; no utils/ package),
DEC-019 (three Q/K/V projections + nlp_create_qkv_heads(q, cat(k,v))),
DEC-021 (bf8_b KV dtype; the bf16 delta is owed at G-KV), DEC-025 (residual scheme A; scatter_output seam must refuse),
DEC-026 (barrier depth 2 — G-RACE's first move if it fails), DEC-027 (descriptor not yet pinned; G-FABRIC-MATRIX picks it),
DEC-029 (four dead CCLManager members dropped — P8 re-checks if the ring path wants a sub-device),
DEC-030 (compute-kernel config home; P6.2 may want to own it), DEC-032 (derive_head_dim moves into ModelArgs at P6.2),
DEC-031/R-011 (upstream fix owed against gpt_oss_d_p's dormant distributed RMSNorm),
DEC-037/R-012 (the root .gitignore excluded every raw log; re-included in-package, kit fix owed)
Closed this phase: DEC-008 (TestFactory.setup_test written in P5.1), DEC-017 (docs/ and scripts/__init__.py deleted),
R-005 (rope_theta substitution now enforced in device code), R-010 (in-package assert added)
```

**STOPPED HERE, ON A GATE BOUNDARY.** Supersedes the end-of-P2 and end-of-P4 stop notes above.
**P0-P4 and P5.1, P5.2, P5.3 are complete and gated; P5.4 (the dense SwiGLU MLP, `G-MLP`) is
next**, then P5.5 (attention, `G-ATTN`) and P5.6 (the KV cache, `G-KV`). Phase P5 is split across
sessions deliberately; this session's scope was P5.1-P5.3 only.

Device code now exists: `tt/{config,ccl,rms_norm,rope}.py`, with `tests/unit/{test_mesh_config,
test_ccl_semaphores,test_rms_norm_vs_ref,test_rope_vs_ref}.py`. `tests/test_factory.py` gained
`TestFactory.setup_test`. `docs/` and `scripts/__init__.py` are gone (`DEC-017`, executed).

What P5.4 can rely on, and what it must not assume:

1. **`MeshConfig` and `CCLManager` are constructed and counted, never exercised.** No collective
   has run on this box in this package. `MLP.__call__`'s TP tail must therefore be written behind
   `if self.mesh_config.tp > 1`, and `G-MLP` at `(1,1)` will not execute it — the first real
   collective is P8's.
2. **`default_compute_kernel_config(mesh_device)` in `tt/config.py` is the only compute-kernel
   config** (`DEC-030`). Pass it to both `ttnn.linear` calls and to `down_proj`'s. Do **not**
   copy `models/demos/gpt_oss_d_p/tt/attention/config.py:71`'s explicit `fp32_dest_acc_en=False`;
   measured here, `False` costs 6.9-8.7x on the norm alone, and recipe §2.4 puts it at 96-1168x on
   a matmul.
3. **`derive_head_dim(hf)` in `tt/config.py`** is the package's one head-dim derivation
   (`DEC-032`); `tt/mlp.py` does not need it, but `tt/attention/` (P5.5) does, and it must call it
   rather than reach for `hf_config.head_dim`, which does not exist.
4. **`G-MLP` gates both dtypes** — `>= 0.999 @bf8_b` and `>= 0.9995 @bf16`, **and `<= 3x` the
   computed floor at each** (`BRINGUP_RECIPE.md:1730`). Unlike `G-RMS`, that ratio bound is
   *asserted*, so compute a separate floor per dtype: quantise the weights at the dtype under test
   and the activations at bf16 (`DEC-022`).
5. **The negative control is SiLU on `up` instead of `gate`** (recipe: 0.6462). It is what proves
   the fused unary is on the argument you think it is.
6. **`scatter_output` must be wired and must refuse** what it cannot honour (`DEC-025`), not
   half-implemented.
7. **Run `pre-commit run --files ...` before recording any `path:line`** — and note `DEC-034`: the
   `prefer-expect-error` hook is a `pygrep`, so it fires on the fixture's name in comments and
   docstrings too, not only on a call.
