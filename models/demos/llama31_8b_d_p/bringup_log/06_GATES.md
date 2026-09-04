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

```
STATUS after P3: gates PASS=4 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P4 (parallelism + CCL plan)
STATUS after P4: gates PASS=5 FAIL=0 DEVIATION=0 BLOCKED=0 | next: P5 (naive module implementations, bottom-up)
Open DECs needing review: DEC-004 (chunk size deferred to P7), DEC-008 (TestFactory.setup_test lands in P5.1),
DEC-012 (checkpoint loader moves to ModelArgs in P6.2), DEC-013/DEC-018 (import gpt_oss_d_p/utils; no utils/ package),
DEC-017 (P5.1 must DELETE docs/ and scripts/__init__.py), DEC-019 (three Q/K/V projections + nlp_create_qkv_heads(q, cat(k,v))),
DEC-021 (bf8_b KV dtype; the bf16 delta is owed at G-KV), DEC-025 (residual scheme A; scatter_output seam must refuse),
DEC-026 (barrier depth 2 — G-RACE's first move if it fails), DEC-027 (descriptor not yet pinned; G-FABRIC-MATRIX picks it)
```

**STOPPED HERE, ON A GATE BOUNDARY.** Supersedes the end-of-P2 stop note above. **P0, P1, P2, P3 and
P4 are complete and gated; P5 (naive module implementations, bottom-up — `G-MESH`, `G-RMS`, `G-ROPE`,
`G-MLP`, `G-ATTN`, `G-KV`) is next.** Still no device code: `tt/` holds only its `__init__.py`.

What P5.1 must do **first**, before writing `MeshConfig`:

1. **Delete `docs/` (and `docs/.gitkeep`) and `scripts/__init__.py`** — `DEC-017`. They are the last
   two artefacts of P0's literal reading and the committed tree (`03_OUTLINE.md` §1) does not contain
   them.
2. Add `TestFactory.setup_test(mesh_device, ...)` to `tests/test_factory.py` **in the same edit** that
   creates `tt/config.py` and `tt/ccl.py` — `DEC-008`.
3. Set `_VALIDATED_MESH_SHAPE = (4, 8)` / `_VALIDATED_TP = 8`, and make `_validate()` **raise** on
   sub-axis TP (`tp != mesh_shape[tp_axis]`), warn otherwise — `03_OUTLINE.md` §5.1 explains which
   half of `BRINGUP_RECIPE.md:1022-1024` is binding and why.
4. Use the repo-root `expect_error` fixture (`conftest.py:948`) for the refusal assertions, **not**
   `pytest.raises`, which the `prefer-expect-error` hook rejects in any `tests/` file
   (`.pre-commit-config.yaml:51`).
