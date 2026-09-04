# 06 — Gate ledger (append-only)

Verdicts: `PASS` · `FAIL` · `PASS-WITH-DEVIATION` (needs a `DEC`) · `BLOCKED` (needs a `07_RISKS.md`
entry naming the blocker) · `NOT-RUN` (needs the reason). A gate with no raw log did not happen.

| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| G-CARD | P0 | every architectural fact has provenance | doc review + 0 bad citations | 53/53 citations verified, 115/115 doc refs resolved, 0 `UNVERIFIED` rows | PASS | 2026-09-04 | `raw/G-CARD_20260904T034338Z.log` (+ `raw/G-CARD_20260904T034303Z.log`) |
| G-REF | P1 | the reference is deterministic and self-consistent | bit-identical x2; hand-written vs HF PCC >= 0.9999 (expect bit-exact) | 10/10 tests pass; both oracles SHA-256 `0a163300…`; PCC **1.0**, `max|Δ| = 0.0` | PASS | 2026-09-04 | `raw/G-REF_20260904T035140Z.log` |
| G-SURVEY | P2 | reuse decided, with citations | doc review + 0 bad citations | 30 component rows, 30 with a decision, 27 with a full `path:line`; 123/123 citations verified, 195/195 doc refs resolved | PASS | 2026-09-04 | `raw/G-SURVEY_20260904T035957Z.log` |

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
