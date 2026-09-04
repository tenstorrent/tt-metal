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
| G-MLP | P5.4 | dense SwiGLU (`down(silu(gate)*up)`), `(1,1)` | PCC >= 0.999 @bf8_b, >= 0.9995 @bf16, **<= 3x floor** at each | bf8_b **0.9999133 / 0.9999144 / 0.9999144** (floor 0.9999213/0.9999221/0.9999222 -> **1.10 / 1.10 / 1.10x**); bf16 **0.9999851 / 0.9999852 / 0.9999852** (floor 0.9999929 -> **2.11 / 2.10 / 2.09x**); control **0.64715 / 0.64722** | PASS | 2026-09-04 | `raw/G-MLP_20260904T093653Z.log` |
| G-ATTN | P5.5 | GQA + full RoPE + causal SDPA + `o_proj`, `(1,1)` | PCC >= 0.999; own stages <= 3x floor; block <= 8x | bf8_b block **0.9997364 / 0.9997080 / 0.9996723** (raw **2.21 / 2.17 / 2.22x**); bf16 block **0.9998463 / 0.9998275 / 0.9998029** (raw **12.32 / 11.82 / 12.17x**, SDPA-attributed residual **1.10 / 1.01 / 0.70x**); stages **1.00-2.50x**; fused SDPA **26.7-28.6x** in-pipeline, **52.8-55.0x** standalone; control **0.51174** | PASS-WITH-DEVIATION (`DEC-042`) | 2026-09-04 | `raw/G-ATTN_20260904T095359Z.log` |
| G-KV | P5.6 | KV cache **primitive**: write correctness, position map, no collateral writes | PCC >= 0.99 @bf8_b, <= 3x floor; positional read-back **bit-exact** | bf8_b worst-of-8-heads K **0.9999743** / V **0.9999752** (**1.00x** the floor at every head and both seq lens); bf16 **0.9999986** (**1.00x**); 128 positions x 4 `kv_actual` offsets **bit-identical**; pad tail + 3 other (user, layer) slots **exactly zero**; bf8_b costs **17.8x** on K / **17.7x** on V vs bf16 | PASS | 2026-09-04 | `raw/G-KV_20260904T100312Z.log` |
| G-LAYER | P6.1 | one decoder layer, norm->attn->residual->norm->MLP->residual (integration check) | PCC >= 0.999, <= 8x floor | bf8_b **0.9997665 / 0.9998273 / 0.9998736** (floors 0.9998709/0.9998953/0.9999138 -> **1.81 / 1.65 / 1.47x**); bf16 **0.9998774 / 0.9999172 / 0.9999480** (floors 0.9999826/0.9999859/0.9999885 -> **7.05 / 5.86 / 4.51x**); SDPA-attributed residual **1.13-1.15x** / **2.01-2.14x**; real weights + real input **0.9998649** (floor 0.9999647 -> **3.82x**); controls **0.99864** / **0.99993** / **0.66830** | PASS | 2026-09-04 | `raw/G-LAYER_20260904T113153Z.log` |
| G-WEIGHTS | P6.2 | no missing/unused keys; cache-only rebuild identical; loader bit-exact | exact | **291/291** keys, 0 missing, 0 unused (all 32 layers); **12/12** device tensors `max\|delta\| = 0.000e+00` through transpose + Q/K Meta swizzle + dtype ladder; **12/12** SHA-256 identical on a cache-only rebuild; 3/3 controls discriminate | PASS | 2026-09-04 | `raw/G-WEIGHTS_20260904T113320Z.log` |

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

---

### G-MLP — dense SwiGLU vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_mlp_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole. TP=1, so the module's TP all-reduce tail
  (`bringup_log/04_CCL_PLAN.md` §5 row 2) is **not executed** — see "what this does NOT prove".
- **Input distribution:** `x` **standard normal**, `[1, 1, S, 4096]` with `S ∈ {32, 512, 4096}`;
  the three projection weights `randn * 0.02`, the scale both templates use
  (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:169`,
  `models/demos/minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py:98`). At that scale `gate` and
  `up` land at std ~1.3, so SiLU is exercised across the curved part of its range rather than in a
  locally-linear tail — stated because the distribution must never be chosen to pass.
- **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic,
  `torch.nn.functional.silu`. Identical random weights drive both sides.
- **Noise floor (computed):** per dtype, and per recipe §2.2's literal definition — quantise the
  **inputs and weights** to the device dtype, all remaining math in fp32. bf8_b weights are
  quantised in the `[1, 1, in, out]` orientation the device stores them in, because `bfloat8_b`
  shares an exponent per tile and quantising HF's `[out, in]` would block along the wrong axis.
  The bf16 *intermediates* the device also stores are deliberately **not** quantised: that is the
  conservative reading, since quantising them would lower the floor and flatter every ratio below.
  - bf8_b: **0.9999213 / 0.9999221 / 0.9999222** at S = 32 / 512 / 4096
  - bf16: **0.9999929** at all three
- **Threshold:** PCC >= **0.999** @bf8_b and >= **0.9995** @bf16, **and <= 3x the floor at each
  dtype** (`BRINGUP_RECIPE.md:1769`). Unlike `G-RMS`, Appendix A states a ratio bound for this gate,
  so the ratio is **asserted**, not merely recorded.
- **Measured:** 14/14 tests pass.
  - bf8_b: **0.9999133 / 0.9999144 / 0.9999144** -> **1.10x / 1.10x / 1.10x** the floor
  - bf16: **0.9999851 / 0.9999852 / 0.9999852** -> **2.11x / 2.10x / 2.09x** the floor
  - Both dtypes run and both are recorded, as `BRINGUP_RECIPE.md:1769` requires. bf8_b clears its
    threshold comfortably, so `DEC-021`'s "keep bf16 if bf8_b misses" contingency is not needed.
  - PCC is flat in sequence length to 7 decimal places, which is what a token-pointwise block
    should do; the S=32 bf8_b value differs only because a 32-row activation is one tile tall.
- **Negative control:** SiLU applied to `up` instead of `gate` — the mistake that proves the fused
  unary is on the argument the code claims — scores **0.64715** (bf8_b) / **0.64722** (bf16). The
  recipe measured **0.6462** for the same mistake. Driven by swapping the `gate_proj` / `up_proj`
  entries of the state dict, so the control runs the **real module** and not a hand-copied device
  path that could drift from it.
- **Two A/Bs, recorded as measurements (direction asserted, not a fitted threshold):**
  1. **`fp32_dest_acc_en`, recipe §2.4 reproduced on this box, at module level:**
     bf8_b `True` **0.9999144** (1.10x) vs `False` **0.9925127** (**96.13x** the floor);
     bf16 `True` **0.9999852** (2.10x) vs `False` **0.9917324** (**1167.80x**). The recipe's table
     predicts 96x and 1168x for the bare `ttnn.linear` — matched to three significant figures
     through a three-matmul module. This is the single most valuable number in the gate: an
     inherited `fp32_dest_acc_en=False` from
     `models/demos/gpt_oss_d_p/tt/attention/config.py:71` would still have cleared a 0.99 gate at
     bf8_b (0.99251) and even the 0.999 bf8_b threshold is what rejects it.
  2. **The SwiGLU spelling** (`DEC-039`): fused `input_tensor_a_activations=[SILU]` vs a separate
     `ttnn.silu` are **numerically identical** at both dtypes (0.9999144 / 0.9999852 either way).
     The fused keyword is bound (`ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:1469`)
     but **absent from `ttnn.mul.__doc__`**, so availability had to be established by calling it.
- **Refusals (counted with the controls):** an empty `state_dict` with no `tensor_cache_path` raises
  `ValueError` rather than building three `None` projections (Appendix B's "cache-only build
  silently wrong"), and `scatter_output=True` raises `NotImplementedError` rather than half-wiring
  residual scheme B (`DEC-038`). Both via the repo-root `expect_error` fixture (`conftest.py:948`),
  because the `prefer-expect-error` hook forbids the alternative in `tests/`.
- **Verdict:** **PASS**
- **Deviations:** none to the gate. Two new decisions: `DEC-038` (the `scatter_output` refusal) and
  `DEC-039` (the fused-SiLU spelling, with its measurement).
- **What this does NOT prove:**
  - **The TP collective.** At `(1,1)` `tp == 1` and `MLP.__call__`'s all-reduce tail is skipped
    entirely, so `bringup_log/04_CCL_PLAN.md` §5 row 2 has still never executed in this package.
    `G-TP-PARITY` (P8) owns it. This is the recipe's own "a gate that passes on a mesh the
    deployment never uses" caveat (`BRINGUP_RECIPE.md:576-578`) applied to a collective rather than
    to a head count.
  - **Column/row-parallel sharding.** At TP=1 `column_parallel` and `row_parallel` produce the same
    (unsharded) tensor, so this gate cannot tell the two mappers apart. `G-TP-PARITY` and
    `G-WEIGHTS` (P8 ext) own that.
  - **The cache-only build.** `tensor_cache_path` is exercised only as the *absence* that makes the
    weightless build refuse; no tilized weight is written or reloaded here. `G-WEIGHTS` (P6.2) owns
    the round trip.
  - **Real weights.** All numbers above are on random weights. A trained `gate_proj` is not
    standard-normal, and `G-RMS` measured that the floor itself moves between random and real
    weights; for the MLP the real-weight comparison arrives with `G-LAYER` / `G-MODEL`.

---

### G-ATTN — the GQA attention block vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_attention_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole (compute grid **(12,10)**). TP=1, so the module's TP
  all-reduce tail is not executed — `bringup_log/04_CCL_PLAN.md` §5 row 1 is P8's.
- **Input distribution:** `x` **standard normal** `[1, 1, S, 4096]` with `S ∈ {128, 512, 2048}`; the
  four projection weights `randn * 0.02`, the scale both templates use
  (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:169`). The standalone SDPA probe
  uses iid standard-normal Q/K/V. `reset_seeds`.
- **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic. The causal mask is
  built **explicitly** (`torch.triu(full((S,S), -inf), diagonal=1)`) and the KV heads are
  `repeat_interleave`d by the GQA group — the device does neither, because
  `ttnn.transformer.scaled_dot_product_attention` is causal and group-aware internally
  (`nqh >= nkv && nqh % nkv == 0`,
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`). cos/sin come
  from **one** frequency set (`tt/rope.py::llama3_freqs`): the reference takes the HF pair, the
  device takes the Meta pair via `build_prefill_rope`, so the test cannot silently compare two
  different RoPEs.
- **Noise floor (computed):** per dtype, and — for the stage budgets — **locally per stage**. A
  chained floor (quantise once, propagate) carries upstream rounding and inflates `1 - floor`; it
  put `concat_heads`, a pure layout op, at **0.01x**, which is not a kernel beating arithmetic but a
  broken floor. Each stage's floor is now computed from that stage's own quantised inputs.
  - block floor: bf8_b **0.9998805 / 0.9998657 / 0.9998521**; bf16 **0.9999875 / 0.9999854 / 0.9999838**
- **Threshold:** PCC >= **0.999**; stages this package implements **<= 3x**; block **<= 8x**
  (`BRINGUP_RECIPE.md:1770`). See **Deviations** for how the block budget is applied.
- **Measured:** 17/17 tests pass.

  | dtype | S | block PCC | raw ratio | SDPA-attributed residual |
  |---|---|---|---|---|
  | bf8_b | 128 | 0.9997364 | 2.21x | 1.03x |
  | bf8_b | 512 | 0.9997080 | 2.17x | 1.00x |
  | bf8_b | 2048 | 0.9996723 | 2.22x | 0.96x |
  | bf16 | 128 | 0.9998463 | 12.32x | 1.10x |
  | bf16 | 512 | 0.9998275 | 11.82x | 1.01x |
  | bf16 | 2048 | 0.9998029 | 12.17x | 0.70x |

  Per-stage, **stage-isolated** (each stage fed the reference's own quantised input, so the number
  is the stage and not the accumulation), at S=512:

  | stage | bf8_b PCC / ratio | bf16 PCC / ratio |
  |---|---|---|
  | `qkv_proj_q` | 0.9999728 / **1.06x** | 0.9999958 / **1.52x** |
  | `qkv_proj_k` | 0.9999727 / **1.06x** | 0.9999958 / **1.51x** |
  | `qkv_proj_v` | 0.9999726 / **1.06x** | 0.9999958 / **1.52x** |
  | `rope_q` | 0.9999954 / **2.50x** | 0.9999954 / **2.50x** |
  | `rope_k` | 0.9999954 / **2.48x** | 0.9999954 / **2.48x** |
  | `concat_heads` | 0.9999986 / **1.00x** | 0.9999986 / **1.00x** |
  | `o_proj` | 0.9999727 / **1.06x** | 0.9999958 / **1.52x** |
  | `sdpa_fused` (**not** in the 3x budget) | 0.9998361 / **26.72x** | 0.9998361 / **26.72x** |

  Every hand-written stage is **1.00x-2.50x** of its floor; the recipe's own run measured
  1.00-1.47x for the same set. The RoPE stages are dtype-independent, as they must be — they touch
  no weight.
- **The fused kernel, isolated and tracked (recipe §2.3):**
  - **standalone probe**, iid bf16 Q/K/V, GQA 32/8, head_dim 128: PCC **0.9998309 / 0.9998127 /
    0.9998087** against modelled floors 0.9999969 / 0.9999966 / 0.9999964 — **54.86x / 54.95x /
    52.83x**. The recipe measured 0.9999204 at 71x; same order, same conclusion. Kept permanently
    so the slack stays a named term.
  - **in-pipeline**, on the block's own post-RoPE tensors: **26.7x-28.6x** (lower than the iid probe
    because real post-RoPE Q/K are correlated across the head dim).
  - **the attribution is quantitative, not rhetorical.** `1 - PCC` is variance-like, so independent
    error sources add to first order. floor error + the kernel's own excess predicts the block PCC
    to 5-6 decimals: bf8_b S=512 predicted **0.9997079** vs measured **0.9997080**; bf16 S=512
    predicted **0.9998277** vs measured **0.9998275**. Subtracting it leaves a residual of
    **0.70x-1.10x** — this package's code, sitting *at* its floor.
- **Negative control:** Q/K weights reaching the device **without** the Meta `reverse_permute`
  scores **0.51174** (bf8_b) / **0.51178** (bf16). Constructed by pre-applying
  `models/tt_transformers/tt/load_checkpoints.py:895` `permute`, the exact inverse of the loader's
  `reverse_permute` (`:891`), so the control runs the **real loader** rather than bypassing it.
  **This is a discrepancy with the recipe, and in the safe direction:** `BRINGUP_RECIPE.md:1214`
  expects ~**0.9475** for the same mistake, i.e. a control that barely fires; measured here it
  collapses to 0.51. At head_dim 128 with full rotary the unswizzled weight scrambles 128 channels
  per head, and both Q *and* K are unswizzled — which may be the difference from whatever variant
  produced 0.9475. Either way the control discriminates, and the recipe's warning about *how high*
  a broken variant can score is unaffected: 0.9475 would still have cleared a 0.99 gate.
- **Invariants and refusals (counted with the controls), all fired:**
  1. **Only Q and K are rotated.** Scored against a reference that also rotates V: **0.71785**,
     versus **0.9998275** for the correct reference. A V-rotating reference must and does fit worse.
  2. **A derived SDPA program grid is refused at construction.** On this (12,10) box the CCL offset
     is `grid.x - 1 = 11` and the pinned SDPA grid is 8, so `11 >= 8` holds; `ProgramConfig(sdpa_grid_x=12)`
     raises, naming
     `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`.
     This is the landmine that otherwise passes **every** single-card gate and fails only at SP > 1.
  3. **`cached_len > 0` on the dense path is refused** (`NotImplementedError` naming the
     chunk-position-aware SDPA), rather than running an `is_causal` mask that is off by `cached_len`
     and silently wrong.
  4. **A weightless build is refused** — no `state_dict` and no `tensor_cache_path` raises rather
     than building four `None` projections.
- **Verdict:** **PASS-WITH-DEVIATION** — `DEC-042`.
- **Deviations:**
  - **`DEC-042`** — the 8x block budget is asserted raw at **bf8_b** (where it holds at 2.17-2.22x)
    and, at both dtypes, on the **SDPA-attributed residual** (0.70-1.10x against the same 8x). The
    raw bf16 ratio, 11.8-12.3x, is recorded and not asserted. The arithmetic in `DEC-042` shows an
    8x raw budget at bf16 is unreachable for **any** correct implementation given this kernel: it
    would require the kernel at <= ~17x its own floor, and the recipe's own §2.3 measurement of it
    is 71x. Note the direction of the paradox — bf16 has the *higher* absolute PCC and the *worse*
    ratio, because a smaller floor error divides the same fixed slack.
  - **`DEC-040`** — `ProgramConfig` delegates the compute-kernel config to `tt/config.py` instead of
    holding the outline's four local fields.
  - **`DEC-041`** — attention's `apply_reduce_scatter` refuses (scheme B seam), and is deliberately
    untested because nothing in this iteration calls it.
  - **`DEC-043`** — the head-split op's Python keyword is `num_heads`, not `num_q_heads` as
    `DEC-019` and `03_OUTLINE.md` §2.7 both spell it.
- **What this does NOT prove:**
  - **The TP collective, or any sharding.** At `(1,1)` `tp == 1`: `apply_allreduce` returns its
    input untouched, and `column_parallel` / `row_parallel` produce the same unsharded tensor, so
    this gate cannot tell the two mappers apart. `G-TP-PARITY` and `G-WEIGHTS` (P8 ext) own that.
  - **The GQA head->column map.** At TP=1 all 32 Q and all 8 KV heads live on one chip, so the
    per-chip `nq=4 / nkv=1` configuration the deployment actually runs is never built here. The
    group *arithmetic* is exercised (SDPA sees 32/8, group 4); the *distribution* is not.
  - **The KV-cache write.** `kv_cache=None` throughout — `G-KV` owns the primitive and `G-KV-TP8`
    the model -> cache path.
  - **The indexed RoPE.** `apply_rope`'s `kv_actual_global` branch is wired and never taken here;
    `G-CHUNK` (P7) owns it.
  - **Real weights.** All numbers are on random weights; `G-LAYER` / `G-MODEL` bring the real ones.
  - **Long context.** The longest gated sequence is 2048, well inside
    `original_max_position_embeddings` = 8192, so the *scaled* band of the llama3 RoPE tables is
    never exercised on device (the band structure is proved on the host by `G-ROPE`).

---

### G-KV — the KV-cache primitive: write, read back, and write nothing else
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_cache_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole. `sp = 1`, so the **block-cyclic layout degenerates to the
  identity** (local row == global position) and no inverse reorder is needed for read-back — which
  is also precisely why this gate cannot test the reorder.
- **Input distribution:** two payloads, deliberately different.
  - *PCC half:* realistic **post-RoPE K** and **raw V** — standard-normal `x [S, 4096]`,
    `randn * 0.02` k/v projections, the package's own llama3 RoPE tables — at `S ∈ {128, 512}`,
    all 8 KV heads, one per layer slot. A cache round trip should be measured on the values it will
    actually hold.
  - *Bit-exact half:* an integer payload where **every row names its own global position**, plus a
    head-id lane block and a chunk-id lane block. 4 chunks x 32 tokens = 128 positions at
    `kv_actual ∈ {0, 32, 64, 96}`.
- **Reference dtype policy:** fp32 reference; only the tensor the device *stores* is quantised, for
  the floor.
- **Noise floor (computed):** `quantize_like_device(ref, cache_dtype)` against the fp32 reference,
  per head. The write is a copy, not arithmetic, so the device should land *on* it — it does.
  - bf8_b: **0.9999743**-**0.9999754** (K), **0.9999752**-**0.9999753** (V)
  - bf16: **0.9999986**
- **Threshold:** PCC >= **0.99** at the cache dtype and **<= 3x its floor**
  (`BRINGUP_RECIPE.md:1771`); the layout claims on **bit-equality** (`torch.equal`, `rtol=atol=0`),
  never PCC (§2.5).
- **Measured:** 15/15 tests pass.
  - **Round trip, worst of 8 heads:** bf8_b `S=128` K **0.9999743** / V **0.9999753**; `S=512`
    K **0.9999754** / V **0.9999752**. bf16 **0.9999986** throughout. **Every head at both seq
    lengths measures a ratio of 1.00x** — exactly at its floor, as a pure copy must.
  - **Positional read-back, bit-exact:** 128 rows across 4 chunks, `torch.equal` at both dtypes,
    and re-checked after **every** chunk write, so "an earlier chunk is unchanged after a later
    chunk's write" is asserted 10 times (chunks 0..k for each k) rather than once.
  - **No collateral writes, bit-exact:** writing `(user 0, layer 1)` of a 2-user x 2-layer cache
    leaves the **352-position pad tail** exactly zero and all **3** other `(user, layer)` slots
    exactly zero, at both dtypes. Target `(0, 1)` rather than `(0, 0)` on purpose, so a
    `slot = user * num_layers + layer` bug would show.
  - **Geometry, asserted exactly:** per-chip `(64, 1, 384, 128)` bf8_b TILE for 2 users x 32 layers,
    DRAM `NdShardSpec` `[1, 1, 32, 128]`, and `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 32` — the
    producer's block geometry, kept so P10 can reuse its packed-GQA read-back
    (`BRINGUP_RECIPE.md:1241-1243`).
  - **The dtype delta `DEC-021` owed:** bf8_b K **0.9999756** / V **0.9999757** versus bf16
    **0.9999986** / **0.9999986** — **17.8x** the error on K and **17.7x** on V, for half the bytes
    (128 vs 256 per token per head). Measured, not assumed; `DEC-021` stands.
- **Negative controls / refusals — five, all fired:**
  1. **The bit-exactness assertion itself is the control for the position map**, and it *caught a
     real failure*: the first version of the probe, built to `BRINGUP_RECIPE.md:1260-1262`'s stated
     "<= 256" ceiling, failed at bf8_b on chunk 2's **odd** rows with `max|delta| = 1.0`. The cache
     was correct; the probe was not. Measured, the first inexact integer is **129** at bf8_b and
     **257** at bf16 — §2.5's ceiling is the **bf16** ceiling, and the cache dtype is bf8_b.
     `DEC-044`. This is §2.5's own warning ("a failing probe is not evidence of a failing module
     until the probe's own numerics are checked") firing against §2.5's own rule.
  2. `write_kv_chunk` with `batch = 2` raises — the op ignores the leading dim and would silently
     write only `slot_idx`.
  3. `slot_idx` out of range raises (a silent OOB write into another user's cache).
  4. `layer_idx` out of range raises.
  5. A non-tile-aligned `kv_actual` raises (breaks the block-cyclic per-device write); and
     `max_seq_len = 48` at `sp = 1` raises because `seq_local` would not be tile-aligned.
- **Verdict:** **PASS**
- **Deviations:** `DEC-044` (probe ceiling 128, not 256 — the probe changed, no threshold did) and
  `DEC-045` (`expect_error`'s `message` is matched as a **regex**, not the substring its docstring
  describes, so `"multiple of TILE_SIZE*sp"` silently never matched; the matcher now uses a
  metachar-free substring).
- **What this does NOT prove** — stated here rather than left for the `PASS` to imply:
  - **The model -> cache path.** At `(1,1)` the model emits all 8 KV heads on one chip while the
    per-chip cache holds exactly **one**, and the write op refuses the mismatch outright
    (`TT_FATAL: cache and input num-heads dim must match`,
    `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
    This gate therefore drives the op **one head at a time**, into its own layer slot. `G-KV-TP8`
    (P8) owns the real path; `07_RISKS.md` R-001 is the standing gap. `bringup_log/00_MODEL_CARD.md`
    §4.3 states the general rule: a gate that passes on a mesh the deployment never uses can be
    testing a configuration the model cannot produce.
  - **The block-cyclic reorder.** At `sp = 1` it is the identity. `G-KV-TP8` and `G-MESH-KV` own it.
  - **The head -> mesh-column map.** There is one column here. That map is a layout claim and must
    be gated on bit-equality at TP=8 — recipe §2.5 measured a *rotated* map still scoring PCC
    0.99890, which is why `G-KV-TP8` has a rotated-column control.
  - **Cache reads.** Nothing here reads the cache back *for attention*; `attention/prefill.py`
    refuses `cached_len > 0` and `G-CHUNK-ATTN` (P8) owns the read path.
  - **Positions past 128.** The bit-exact probe stops at 127 (`DEC-044`). A deployment chunk is far
    longer, so `G-KV-TP8`'s probe must split the position id across lane blocks rather than raise
    the ceiling.

---

```
STATUS after P5.4-P5.6: gates PASS=10 FAIL=0 DEVIATION=1 BLOCKED=0 | next: P6 (layer and model assembly)
Per-phase regression gate: whole package suite **93 passed, 0 failed**
(`raw/P5-REGRESSION_20260904T100738Z.log`)
Citations after P5.6: **330/330 verified, 0 mismatched; 629/629 doc refs resolved**, exit 0
(`raw/G-CITE_20260904T101355Z.log`). CITES grew 279 -> 330: every load-bearing P5.4-P5.6 ref was
promoted into the content-checked list, because the doc-ref pass only range-checks (R-016).
Open DECs needing review: DEC-004 (chunk size deferred to P7), DEC-012 (checkpoint loader moves to
ModelArgs in P6.2), DEC-013/DEC-018 (import gpt_oss_d_p/utils; no utils/ package),
DEC-021 (bf8_b KV dtype — the owed delta is now MEASURED at G-KV: 17.8x on K, 17.7x on V, for half
the bytes; the decision stands), DEC-025 (residual scheme A), DEC-026 (barrier depth 2 — G-RACE's
first move if it fails), DEC-027 (descriptor not yet pinned; G-FABRIC-MATRIX picks it),
DEC-029 (four dead CCLManager members dropped), DEC-030/DEC-040 (compute-kernel config home; P6.2
may want to own it), DEC-032 (derive_head_dim moves into ModelArgs at P6.2),
DEC-038/DEC-041 (the scatter_output seams refuse; P8 owns scheme B),
DEC-042/R-015 (**the one that needs a decision, not just review**: G-ATTN's 8x block budget is
unreachable at bf16 for any correct implementation given the fused SDPA kernel; P6's G-LAYER (8x)
and G-MODEL (8x + 4x step) contain the same kernel and will meet the same wall),
DEC-044/R-013 (bf8_b's exact-integer ceiling is 128, not the recipe's 256 — P8/P10 probes must
split the id across lanes), DEC-045/R-014 (expect_error's message is a regex)
Closed this phase: R-009 (the dense/bias-free/full-RoPE adaptation is now measured, G-ATTN),
R-007 (write half — the KV write path is bit-exact at head_dim 128), R-012 (kit half — the recipe
now carries the raw-log .gitignore guidance)
```

**STOPPED HERE, ON A GATE BOUNDARY.** Supersedes the end-of-P5.3 stop note above.
**P0-P4 and all of P5 (P5.1-P5.6) are complete and gated; P6 (layer and model assembly) is next.**
This session's scope was P5.4, P5.5 and P5.6 only — `tt/layer.py` and `tt/model.py` are deliberately
not written.

All six P5 gates are recorded: `G-MESH`, `G-RMS`, `G-ROPE`, `G-MLP`, `G-KV` are `PASS` and `G-ATTN`
is `PASS-WITH-DEVIATION` (`DEC-042`), so `BRINGUP_RECIPE.md:1267-1269`'s "all of G-MESH, G-RMS,
G-ROPE, G-MLP, G-ATTN, G-KV must be PASS before P6" is satisfied under §1.4's definition of the
verdicts.

Device code now exists: `tt/{config,ccl,rms_norm,rope,mlp}.py` and
`tt/attention/{__init__,config,weights,operations,prefill,kv_cache,dense_sp}.py`, with
`tests/unit/{test_mesh_config,test_ccl_semaphores,test_rms_norm_vs_ref,test_rope_vs_ref,
test_mlp_vs_ref,test_attention_vs_ref,test_kv_cache_vs_ref}.py`.

What P6 can rely on, and what it must not assume:

1. **No collective has ever executed in this package.** Every P5 gate ran at `(1,1)`, where
   `tp == 1` and both module tails (`MLP.__call__`, `attention/operations.apply_allreduce`) return
   their input untouched. `bringup_log/04_CCL_PLAN.md` §5 rows 1-2 are still unexercised, and so are
   `column_parallel` / `row_parallel` as *distinct* mappers. `tt/layer.py` must not read a P5 `PASS`
   as evidence that the TP path works.
2. **`Attention` takes an `AttentionConfig`, not the raw `hf` dict** — the one deliberate exception
   to the module signature convention (`bringup_log/03_OUTLINE.md` §5). `tt/layer.py` builds that
   config **once** and shares it across all 32 layers: there is no per-layer `dataclasses.replace`,
   because Llama has no sliding-window alternation.
3. **`ProgramConfig.get_compute_kernel_config` now takes `mesh_device`** (`DEC-040`), and
   `ProgramConfig.validate_grid(mesh_device)` runs at `Attention.__init__`. Build the program config
   once per model, not per layer.
4. **`attention_forward` refuses `cached_len > 0`** and `MLP`/`operations.apply_reduce_scatter`
   refuse `scatter_output=True`. All three refusals are load-bearing, not stubs: P6 must not route
   around them.
5. **The KV cache is proved as a primitive only.** `write_kv_chunk` is bit-exact at `head_dim = 128`
   and writes nothing it should not, but the **model -> cache** path has never run — at TP=1 the op
   refuses the model's 8 local KV heads outright. If `tt/model.py` wires a cache write, its first
   real test is P8's `G-KV-TP8`.
6. **`G-LAYER` and `G-MODEL` will hit `R-015`.** Both blocks contain
   `ttnn.transformer.scaled_dot_product_attention`, whose slack accounts for the whole of `G-ATTN`'s
   block gap (26.7-28.6x in-pipeline, 52.8-55.0x standalone). `G-MODEL`'s **4x per-layer step** is
   the tighter constraint and needs the same attribution `DEC-042` sets out — decide it before
   measuring, not after.
7. **Stage floors must be computed locally.** A floor propagated through a quantised chain carries
   the upstream stages' rounding and produces ratios below 1.0 (measured: `concat_heads` at 0.01x),
   which is a broken floor, not a kernel beating arithmetic. `G-ATTN`'s stage helper is the pattern.
8. **Never read a `path:line` out of a multi-file `cat -n`** (R-016), and pick refusal-message
   substrings with no regex metacharacters (`DEC-045`).

---

### Re-run after the citation corrections (end of P5.6)

Every `path:line` in this session's modules and log entries was re-resolved and **21 were wrong**
(R-016): four refs into `models/demos/gpt_oss_d_p/tt/attention/operations.py` carried a **+209 line
offset** — they had been read out of a `cat -n weights.py operations.py` whose numbering ran across
both files — and seventeen refs into `BRINGUP_RECIPE.md` had been **interpolated from the section
headings rather than read**, e.g. Appendix A's `G-ATTN` row cited as `:1731` when it is at `:1770`.
The verifier's doc-ref pass reported all of the in-range ones as `resolved`, because that pass
checks the line *number*, not the line's *content*.

All are corrected, and `CITES` now content-checks **every** recipe reference this package makes
(`RCP` entries, 359 total citations, up from 279 at the end of P5.3) so a future recipe edit that
shifts a section produces a `MISMATCH` rather than a silent lie.

Because the modules changed after the first gate runs — docstrings and comments only, no executable
line — all three gates and the regression were **re-run** so the recorded evidence matches the tree:

| Gate | Result | Raw log |
|---|---|---|
| `G-MLP` | 14 passed, 0 failed | `raw/G-MLP_20260904T101937Z.log` |
| `G-ATTN` | 17 passed, 0 failed | `raw/G-ATTN_20260904T102107Z.log` |
| `G-KV` | 15 passed, 0 failed | `raw/G-KV_20260904T102206Z.log` |
| per-phase regression | **93 passed, 0 failed** | `raw/P5-REGRESSION_20260904T102256Z.log` |
| citations | 359/359 verified, 631/631 doc refs resolved, exit 0 | `raw/G-CITE_20260904T102606Z.log` |

The measured numbers are identical to the first runs in every case; the earlier raw logs
(`G-MLP_20260904T093653Z`, `G-ATTN_20260904T095359Z`, `G-KV_20260904T100312Z`) are kept because the
detail blocks above quote them, and because §0.2 rule 4 is right that a log records what actually
ran. **These re-run logs are the ones the verdicts rest on.**

### G-LAYER — one decoder layer vs an fp32 torch reference
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_decoder_layer_vs_ref.py -x -q`
- **Mesh / device:** `(1,1)`, Blackhole. TP=1, so **no collective executes** — the layer never calls
  one by construction (`04_CCL_PLAN.md` §4) and both sublayer tails are no-ops at `tp == 1`.
- **Inputs (distribution AND scale — `R-018`):** three arms.
  1. **Gate arm:** `x` **standard normal** `[1, 1, S, 4096]`, `S ∈ {128, 512, 2048}`; seven
     projections `randn * 0.02`; two norm gains `1 + randn * 0.02`. Both weight dtypes.
  2. **Real-weight arm:** real layer-0 weights (all nine tensors) with a `randn` input — recorded
     because it is what showed the control was weak, not because it gates anything.
  3. **Real-weight, real-input arm:** real layer-0 weights **and** real `embed_tokens` rows
     (RMS **0.0106**, i.e. ~100x smaller than `randn`), which is what layer 0 actually receives.
- **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic; explicit causal mask
  `triu(full((S,S), -inf), 1)`; KV heads `repeat_interleave`d by the GQA group. The staged reference
  in the test is **bit-exact** against both the `G-REF` oracle and HF `LlamaDecoderLayer`
  (PCC 1.000000000, `max|Δ| = 0.0`), so the gate is scored against math that was already gated.
- **Noise floor (computed):** the same fp32 layer with its **inputs and weights** rounded to the
  device dtypes and everything else in fp32. Internal intermediates are **not** quantised.
- **Threshold:** PCC ≥ 0.999 and ≤ 8x the floor (`BRINGUP_RECIPE.md:1372`, Appendix A `:1854`),
  applied per `DEC-051`: the raw ratio asserted at bf8_b, and the **SDPA-attributed residual**
  asserted at ≤ 8x at both dtypes.
- **Measured:**

  | dtype | S | PCC | floor | raw ratio | kernel excess (x floor err) | attributed residual |
  |---|---|---|---|---|---|---|
  | bf8_b | 128 | 0.9997665 | 0.9998709 | 1.81x | 0.68x | **1.13x** |
  | bf8_b | 512 | 0.9998273 | 0.9998953 | 1.65x | 0.51x | **1.14x** |
  | bf8_b | 2048 | 0.9998736 | 0.9999138 | 1.47x | 0.31x | **1.15x** |
  | bf16 | 128 | 0.9998774 | 0.9999826 | 7.05x | 5.04x | **2.01x** |
  | bf16 | 512 | 0.9999172 | 0.9999859 | 5.86x | 3.83x | **2.03x** |
  | bf16 | 2048 | 0.9999480 | 0.9999885 | 4.51x | 2.37x | **2.14x** |

  Real weights + real input, bf8_b, S=512: **0.9998649** against a floor of **0.9999647** — **3.82x**.
  This is the number comparable with recipe §2.1's table (also measured on real weights).
- **Verdict:** **PASS** — and note that unlike `G-ATTN` the **raw 8x holds at both dtypes**
  (worst 7.05x). `R-015`'s wall does not reach layer level: the layer adds two norms and three more
  quantised projections to the floor's error budget while the fused kernel's absolute slack is
  unchanged, so the kernel's share falls. `DEC-051` carries the arithmetic.
- **Negative controls (three, and only the third discriminates — `DEC-058`, `R-018`):**
  - random weights, norm gains swapped: **0.99864** (bf8_b) / **0.99873** (bf16) — rejected by the
    gate's assertion, but by 1.4e-3;
  - real layer-0 weights, `randn` input, gains swapped: **0.99993** — *further* from failing,
    because the residual attenuates the sublayers by ~65x at that input scale;
  - real layer-0 weights, **real** embedding input, gains swapped: **0.66830** (recipe quotes
    0.9471), attenuation 3.40x. This is the control the verdict rests on.
- **Additional assertions:** causality on the **device** layer — perturbing the last token leaves
  rows `[:-1]` at `max|Δ| = 0.000e+00` while the last row moves by 8.969; the `LLAMA_DELTA_PROBE`
  helper runs on a real tensor and warns rather than raising on garbage (the package's one
  `except Exception`).
- **Deviations:** none from the threshold. `DEC-051` (attribution method), `DEC-054` (signature),
  `DEC-058` (control input scale) are the phase's judgement calls.
- **What this gate does not prove.** It is an **integration** check
  (`BRINGUP_RECIPE.md:1375`): it cannot localise a sublayer fault, and it may not substitute for
  `G-RMS`/`G-ROPE`/`G-MLP`/`G-ATTN`, all of which are met on their own. It also runs at TP=1, so
  neither module's TP collective has ever executed (P8), and it writes no KV cache — at TP=1 the
  packed cache refuses the model's 8 local KV heads outright (`00_MODEL_CARD.md` §4.1, `R-001`).

### G-WEIGHTS — real-checkpoint weight loading, bit-exact
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py -x -q`
- **Mesh / device:** one card, `(1,1)`. Cache-only at TP > 1 is the P8 extension
  (`BRINGUP_RECIPE.md:1411`).
- **Inputs (distribution):** not a synthetic distribution — the inputs **are** the real
  Llama-3.1-8B-Instruct checkpoint tensors at their stored dtype, `torch.bfloat16` (measured and
  logged by `torch_dtype_of`, matching `00_MODEL_CARD.md` §2). Stated rather than omitted because
  §1.4 requires it.
- **Reference dtype policy:** the **same torch tensor object** drives both sides — the device path
  and `quantize_like_device` — with **no fp32 detour**, so a bf16→bf8_b vs fp32→bf8_b double-rounding
  difference cannot be mistaken for a loader fault. Every comparison is `torch.equal`
  (`rtol = atol = 0`), never PCC: recipe §2.5 measured a completely wrong mapping still scoring
  PCC 0.99890, and a transpose or swizzle applied twice is that class of bug
  (`BRINGUP_RECIPE.md:1408-1409`).
- **Noise floor:** not applicable, and that is the point — a bit-exactness gate has no floor because
  the tolerance is zero. The nearest equivalent, recorded instead: `max|Δ| = 0.000e+00` on all
  twelve tensors.
- **Threshold:** exact, in three parts (`BRINGUP_RECIPE.md:1404-1411`).
- **Measured:**
  - **(a) no missing, no unused.** Checkpoint keys **291**, expected **291**, missing **0**,
    unused **0** — over all 32 layers, read from `model.safetensors.index.json` with no tensor data
    loaded. Both difference sets printed (both empty).
  - **(c) every device weight bit-exact.** All **12/12** tensors of a one-layer model at
    `max|Δ| = 0.000e+00` *through* the loader's transpose, the Q/K Meta swizzle and the dtype ladder:
    `embed_tokens` `(128256, 4096)` bf16; both layer norms and `model.norm` `(1,1,128,32)` bf16;
    `q_proj` `(1,1,4096,4096)` and `k_proj`/`v_proj` `(1,1,4096,1024)` and `o_proj`
    `(1,1,4096,4096)` bf8_b; `gate_proj`/`up_proj` `(1,1,4096,14336)` and `down_proj`
    `(1,1,14336,4096)` bf8_b; `lm_head` `(1,1,4096,128256)` bf8_b. Per-tensor SHA-256 logged.
    `DEC-055` records why this is all twelve of one layer rather than a sample of 32.
  - **(b) cache-only rebuild bit-identical.** 12/12 SHA-256 identical between a checkpoint build and
    a rebuild from an **empty** `state_dict` against the same `tensor_cache_bfp8_1x1` directory;
    12 cache files written.
- **Verdict:** **PASS**
- **Negative controls (three, all discriminate):**
  1. **Meta-renamed keys** (`map_hf_to_meta_keys`): 291 keys in, **0** still consumable by this
     package (`layers.0.attention.wq.weight`, …), and `Model` **raises** rather than building on
     `None`s. This replaces the recipe's "bypass `map_hf_to_meta_keys`" wording, which is
     inapplicable because this package does not apply that map — `DEC-046`.
  2. **Double Meta swizzle:** with Q/K pre-`reverse_permute`d, `q_proj` and `k_proj` stop being
     bit-equal to the clean load while `v_proj` and `o_proj` stay equal — i.e. the check sees a
     transform applied twice, which is exactly what PCC would not (`DEC-047`).
  3. **Cross-dtype cache:** a bf16 build wrote **12** cache files into `tensor_cache_bf16_1x1` and
     **0** into `tensor_cache_bfp8_1x1`; ttnn additionally suffixes each file
     `_dtype_<DT>_layout_<L>.tensorbin`, so both defences hold (`DEC-048`).
- **Refusals asserted:** `weight_cache_path` with `TT_CACHE_PATH` unset (closes `R-003`);
  `load_state_dict(convert_to_meta_format=True)` (`DEC-047`); a `transformers` config **object**
  passed as `hf_config` (recipe P1 trap 1 — the test also re-asserts that this version's config has
  no `rope_theta` attribute).
- **Deviations:** `DEC-046` (no Meta key mapping, control inverted), `DEC-047`, `DEC-048`,
  `DEC-055` (one layer for the per-tensor and cache halves).
- **What this gate does not prove.** The **32-layer** cache-only rebuild, and any rebuild at TP > 1
  where the cached tensor is sharded — `G-WEIGHTS (P8 ext)` owns both. It also says nothing about
  whether the loaded weights are *numerically useful*; `G-MODEL`'s top-1 against HF is what closes
  that.

### G-MODEL — full stack hidden states + top-1 agreement (P6.3)
- **Command:** `pytest models/demos/llama31_8b_d_p/tests/unit/test_model_vs_ref.py -x -q`
- **Mesh / device:** (1,1), Blackhole. **Input:** real tokenized-prompt token ids, real checkpoint.
  **Reference dtype policy:** fp32 HF `LlamaForCausalLM`, eager; floor from inputs and weights only
  (no intermediates) — see the correction below.
- **Threshold (per the phase text, `DEC-053`):** at 2 and 4 layers PCC ≥ 0.999 and ≤ 8x floor; at full
  depth per-layer step ≤ 4x from L3; 100% top-1 at every depth.
- **Measured:**
  - 2 layers / s128: **0.9997314** (floor 0.9998795, **2.23x**), top-1 agrees
  - 4 layers / s128: **2.36x**, top-1 agrees
  - **32 layers / s512: 0.9984849**, top-1 **374 == 374**, worst per-layer step **1.27x** (budget 4x),
    curve smooth and monotone with **no step anywhere**
  - **Floor correction (`R-021`):** the gate's floor used HF's own class, which computes RoPE cos/sin
    internally in fp32, so it omitted the bf16 rounding the device pays on those tables in all 32
    layers. `1 - floor` = 5.4300e-04 (fp32 tables) vs **9.9160e-04** (bf16 tables) → the same
    measurement is **2.79x** against the incomplete floor and **1.53x** against the correct one. The
    omitted term is 45% of the correct floor error. A staged chain reproduced the gate's floor exactly
    (0.9994570) with fp32 tables, isolating the cause to that single term.
- **Negative control:** layer weights rotated by one → **0.16180**.
- **Verdict:** **PASS** (phase text). Recorded honestly: the full-depth absolute PCC 0.9984849 is
  **below** the 0.999 that Appendix A's compressed row appeared to require at all depths — that
  conflict is `R-020` and is fixed in the kit rather than worked around here.
- **What this does NOT prove:** §2.3.1's kernel attribution does not hold at this depth (`R-022`): the
  substituted chain scored 0.9981153, *worse* than the device, so the subtraction over-removes and the
  attributed residual came out at 0.63x. What it does establish is the fused kernel's **share**:
  58.9% of total model-scale error.
